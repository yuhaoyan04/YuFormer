"""Immutable experiment identity and per-attempt provenance.

The identity contains only inputs that define the optimization trajectory.  Runtime
facts such as host name and package versions are recorded in an attempt manifest so
that a fault-tolerance restart can move hosts without changing the experiment id.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


IDENTITY_FILE = "run_identity.json"
_SOURCE_SUFFIXES = {".py", ".sh", ".toml", ".yaml", ".yml"}
_IGNORED_PARTS = {".git", ".pytest_cache", "__pycache__", ".ipynb_checkpoints"}


def canonical_sha256(value: Any) -> str:
    blob = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_tree_sha256(paths: Iterable[str | Path], repo_root: str | Path) -> str:
    """Hash relevant source files, including untracked files, in stable path order."""
    root = Path(repo_root).resolve()
    files: set[Path] = set()
    for raw in paths:
        path = Path(raw).resolve()
        if path.is_file():
            files.add(path)
            continue
        for candidate in path.rglob("*"):
            if (
                candidate.is_file()
                and candidate.suffix in _SOURCE_SUFFIXES
                and not (_IGNORED_PARTS & set(candidate.parts))
            ):
                files.add(candidate)
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.as_posix()):
        try:
            name = path.relative_to(root).as_posix()
        except ValueError:
            name = path.as_posix()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(file_sha256(path)))
    return digest.hexdigest()


def git_provenance(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)

    def run(*args: str) -> str | None:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = run("status", "--porcelain=v1", "--untracked-files=normal")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status),
    }


def build_run_identity(
    *,
    run_name: str,
    config: dict[str, Any],
    model_config_path: str | Path,
    tokenizer_sha256: str,
    data: list[dict[str, Any]],
    dataset_release: dict[str, Any] | None,
    source_paths: Iterable[str | Path],
    repo_root: str | Path,
    backend: str,
) -> dict[str, Any]:
    model_path = Path(model_config_path)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_name": run_name,
        "backend": backend,
        "config_sha256": canonical_sha256(config),
        "model_config": {
            "path": model_path.as_posix(),
            "sha256": file_sha256(model_path),
        },
        "tokenizer_sha256": tokenizer_sha256,
        "data": data,
        "dataset_release": dataset_release,
        "source_tree_sha256": source_tree_sha256(source_paths, repo_root),
        "git": git_provenance(repo_root),
    }
    payload["run_id"] = canonical_sha256(payload)[:20]
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def ensure_run_identity(out_dir: str | Path, identity: dict[str, Any]) -> Path:
    """Create the identity once, or reject any attempt to mutate an existing run."""
    root = Path(out_dir)
    path = root / IDENTITY_FILE
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != identity:
            keys = sorted(
                key
                for key in set(existing) | set(identity)
                if existing.get(key) != identity.get(key)
            )
            raise RuntimeError(
                "run identity mismatch for existing out_dir "
                f"{root}: changed fields={keys}; use a new output directory"
            )
        return path

    # The writer lease is acquired before identity construction so concurrent
    # launchers cannot both adopt a fresh output directory.  These entries are
    # runtime control state, not legacy training artifacts.
    allowed = {"run.lock", ".run.lease"}
    unexpected = (
        sorted(item.name for item in root.iterdir() if item.name not in allowed)
        if root.exists()
        else []
    )
    if unexpected:
        raise RuntimeError(
            f"refusing to adopt non-empty legacy out_dir {root}: entries={unexpected[:8]}; "
            "use a new output directory or migrate it explicitly"
        )
    _atomic_json(path, identity)
    return path


def write_attempt_manifest(
    out_dir: str | Path,
    identity: dict[str, Any],
    *,
    world_size: int,
    argv: list[str] | None = None,
) -> Path:
    """Record environment facts for this launch without changing the run identity."""
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        }
    except Exception as exc:  # pragma: no cover - torch is a required runtime dependency
        torch_info = {"error": repr(exc)}
    timestamp_ns = time.time_ns()
    payload = {
        "schema_version": 1,
        "run_id": identity["run_id"],
        "created_at_unix_ns": timestamp_ns,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "torch": torch_info,
        "world_size": world_size,
        "argv": list(sys.argv if argv is None else argv),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    path = Path(out_dir) / "attempts" / f"attempt-{timestamp_ns}-{os.getpid()}.json"
    _atomic_json(path, payload)
    return path
