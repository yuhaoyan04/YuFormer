"""Cross-pod single-writer lease for shared training/materialization outputs."""
from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import ConfigError


class RunLease:
    """Append-only generation lease using atomic shared-filesystem ``mkdir``.

    POSIX ``flock`` is not a sufficient ownership boundary across the project's
    NFS clients.  Every takeover creates a new generation and every heartbeat
    and publish checks that it is still the latest generation, fencing a paused
    writer that resumes after a stale takeover.
    """

    def __init__(
        self,
        out_dir: str | Path,
        run_name: str,
        *,
        heartbeat_seconds: float = 30.0,
        stale_seconds: float = 24 * 60 * 60,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.run_name = run_name
        self.lock_root = self.out_dir / ".run.lease"
        self.path: Path | None = None
        self.token = uuid.uuid4().hex
        self.owner_pid = os.getpid()
        self.heartbeat_seconds = heartbeat_seconds
        self.stale_seconds = stale_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _generations(self) -> list[tuple[int, Path]]:
        generations: list[tuple[int, Path]] = []
        for path in self.lock_root.glob("generation-*"):
            if not path.is_dir():
                continue
            try:
                generations.append((int(path.name.removeprefix("generation-")), path))
            except ValueError:
                continue
        return generations

    @staticmethod
    def _owner(path: Path) -> tuple[dict[str, Any], float, bool]:
        try:
            owner = json.loads((path / "owner.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            owner = {}
        token = str(owner.get("token") or "unknown")
        heartbeat = path / f"heartbeat-{token}"
        try:
            age = time.time() - (heartbeat if heartbeat.exists() else path).stat().st_mtime
        except OSError:
            age = 0.0
        released = (path / f"released-{token}").is_file()
        return owner, max(0.0, age), released

    @staticmethod
    def _same_host_owner_dead(owner: dict[str, Any]) -> bool:
        """Return true only when a local owner PID is conclusively gone.

        Cross-pod PIDs are not comparable, so those owners remain governed by
        heartbeat expiry.  This local fast path makes SIGKILL/OOM recovery
        immediate without weakening the shared-filesystem generation fence.
        """
        if owner.get("hostname") != socket.gethostname():
            return False
        try:
            pid = int(owner.get("pid", 0))
        except (TypeError, ValueError):
            return False
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
            if raw[raw.rfind(")") + 2:].split()[0] == "Z":
                return True
        except ProcessLookupError:
            return True
        except (PermissionError, OSError, IndexError):
            return False
        return False

    def acquire(self) -> "RunLease":
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.lock_root.mkdir(exist_ok=True)
        owner = {
            "schema_version": 2,
            "run_name": self.run_name,
            "token": self.token,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "started_at": datetime.now(timezone.utc).isoformat(),
            "started_unix": time.time(),
        }
        for _attempt in range(16):
            generations = self._generations()
            if generations:
                current_number, current_path = max(generations)
                current, age, released = self._owner(current_path)
                owner_dead = self._same_host_owner_dead(current)
                if not released and not owner_dead and age <= self.stale_seconds:
                    raise ConfigError(
                        f"输出目录已被另一个训练任务占用: {self.out_dir}; "
                        f"owner={json.dumps(current, ensure_ascii=False, sort_keys=True)}"
                    )
            else:
                current_number = 0
            path = self.lock_root / f"generation-{current_number + 1:020d}"
            try:
                path.mkdir()
            except FileExistsError:
                continue
            (path / "owner.json").write_text(
                json.dumps(owner, ensure_ascii=False, sort_keys=True), encoding="utf-8"
            )
            (path / f"heartbeat-{self.token}").touch()
            self.path = path
            # Compatibility/status surface for operators; ownership is the
            # generation directory, not this replaceable metadata file.
            temporary = self.out_dir / f".run.lock.tmp-{os.getpid()}"
            temporary.write_text(json.dumps(owner, sort_keys=True), encoding="utf-8")
            os.replace(temporary, self.out_dir / "run.lock")
            self._thread = threading.Thread(target=self._heartbeat, daemon=True)
            self._thread.start()
            return self
        raise ConfigError(f"无法竞争共享 lease: {self.lock_root}")

    def assert_owner(self) -> None:
        if self.path is None:
            raise ConfigError("run lease 尚未获取或已释放")
        generations = self._generations()
        if not generations or max(generations)[1] != self.path:
            raise ConfigError(f"run lease 已被更新 generation fence: {self.path}")
        owner, _age, released = self._owner(self.path)
        if owner.get("token") != self.token or released:
            raise ConfigError(f"run lease 所有权已丢失: {self.path}")

    def _heartbeat(self) -> None:
        failures = 0
        while not self._stop.wait(self.heartbeat_seconds):
            try:
                self.renew()
                failures = 0
            except ConfigError:
                os._exit(74)
            except OSError:
                failures += 1
                if failures >= 3:
                    os._exit(74)

    def renew(self) -> None:
        """Refresh the heartbeat after proving this generation still owns the run."""
        self.assert_owner()
        assert self.path is not None
        (self.path / f"heartbeat-{self.token}").touch(exist_ok=True)

    def release(self) -> None:
        if self.path is None or os.getpid() != self.owner_pid:
            return
        self._stop.set()
        try:
            (self.path / f"released-{self.token}").touch(exist_ok=True)
        finally:
            self.path = None

    def __enter__(self) -> "RunLease":
        return self.acquire()

    def __exit__(self, *exc: object) -> None:
        self.release()
