"""实验追踪（experiments/_common 契约，2026-09-26 实现）。

每臂一个 run 目录：

  <results_dir>/<arm_name>/
    metrics.jsonl   # 每步/每事件一行：{"step": int, "wallclock": float, ...}
    manifest.json   # 实验身份：config hash / tokenizer hash / 数据快照 hash / seed / 环境

用法（各实验目录只读引用，不改这里）：

  from tracker import RunTracker, file_sha256, canonical_sha256

  tr = RunTracker(results_dir, arm_name="e0_vocab64k", config=cfg_dict, seed=1234)
  tr.log_metrics(step=100, loss=4.21, lr=2e-4)
  tr.write_manifest(tokenizer_sha256=..., data_sha256=..., params=..., extra={...})

确定主义：JSON 全部 sort_keys 序列化；manifest 写入用 tmp+os.replace。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


def canonical_sha256(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def file_sha256(path: str | Path, chunk_size: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def edge_sha256(path: str | Path, edge_bytes: int = 1 << 20) -> str:
    """大文件快速身份：size + 首尾各 edge_bytes 的 sha256（启动期守卫用）。"""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode("ascii"))
    with open(p, "rb") as f:
        h.update(f.read(edge_bytes))
        if size > edge_bytes:
            f.seek(max(0, size - edge_bytes))
            h.update(f.read(edge_bytes))
    return h.hexdigest()


class RunTracker:
    def __init__(
        self,
        results_dir: str | Path,
        arm_name: str,
        config: dict[str, Any],
        seed: int,
    ) -> None:
        self.arm_name = arm_name
        self.config = config
        self.seed = seed
        self.run_dir = Path(results_dir) / arm_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self.run_dir / "metrics.jsonl"
        self._manifest_path = self.run_dir / "manifest.json"
        self._created = time.time()
        self._wall0 = time.monotonic()
        self._last_step = self._read_last_step()

    def _read_last_step(self) -> int | None:
        if not self._metrics_path.is_file():
            return None
        last: int | None = None
        with open(self._metrics_path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    step = int(json.loads(line)["step"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid metrics record at {self._metrics_path}:{line_number}"
                    ) from exc
                if last is not None and step < last:
                    raise ValueError(
                        f"non-monotonic metrics at {self._metrics_path}:{line_number}: "
                        f"step {step} < previous {last}"
                    )
                last = step
        return last

    @property
    def config_sha256(self) -> str:
        return canonical_sha256(self.config)

    def log_metrics(self, step: int, **metrics: Any) -> None:
        step = int(step)
        if self._last_step is not None and step < self._last_step:
            raise ValueError(
                f"refusing non-monotonic metric append: step {step} < {self._last_step} "
                f"for {self._metrics_path}"
            )
        record = {
            "step": step,
            "wallclock": round(time.monotonic() - self._wall0, 3),
            **{k: (float(v) if isinstance(v, float) else v) for k, v in metrics.items()},
        }
        with open(self._metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._last_step = step

    def read_metrics(self) -> list[dict[str, Any]]:
        if not self._metrics_path.is_file():
            return []
        out = []
        with open(self._metrics_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def rewind_to_step(self, step: int, *, reason: str) -> Path | None:
        """Roll metrics back to the last committed checkpoint without losing evidence.

        A process can log several steps after its most recent checkpoint and then
        be killed.  Resuming from that checkpoint must replace the uncommitted
        suffix; appending behind it would create a non-monotonic or misleading
        timeline.  The suffix is archived before the canonical file is replaced.
        """
        step = int(step)
        if step < 0:
            raise ValueError("rewind step must be non-negative")
        if not self._metrics_path.is_file():
            self._last_step = None
            return None
        kept: list[str] = []
        orphaned: list[str] = []
        with self._metrics_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record_step = int(json.loads(line)["step"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"invalid metrics record at {self._metrics_path}:{line_number}"
                    ) from exc
                (kept if record_step <= step else orphaned).append(line)
        if not orphaned:
            self._last_step = self._read_last_step()
            return None

        stamp = time.time_ns()
        archive = self.run_dir / f"metrics.orphaned-after-{step:08d}-{stamp}.jsonl"
        archive_tmp = archive.with_name(f".{archive.name}.tmp-{os.getpid()}")
        archive_tmp.write_text(
            json.dumps(
                {"checkpoint_step": step, "reason": reason},
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n"
            + "".join(orphaned),
            encoding="utf-8",
        )
        os.replace(archive_tmp, archive)

        metrics_tmp = self._metrics_path.with_name(
            f".{self._metrics_path.name}.tmp-{os.getpid()}"
        )
        metrics_tmp.write_text("".join(kept), encoding="utf-8")
        os.replace(metrics_tmp, self._metrics_path)
        self._last_step = self._read_last_step()
        return archive

    def write_manifest(self, **fields: Any) -> Path:
        manifest = {
            "arm_name": self.arm_name,
            "created": self._created,
            "seed": self.seed,
            "config_sha256": self.config_sha256,
            "config": self.config,
            **fields,
        }
        tmp = self._manifest_path.with_name(self._manifest_path.name + f".tmp-{os.getpid()}")
        tmp.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(tmp, self._manifest_path)
        return self._manifest_path
