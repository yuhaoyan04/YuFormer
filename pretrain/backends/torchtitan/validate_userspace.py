#!/usr/bin/env python3
"""Validate the pinned TorchTitan userspace without initializing CUDA devices."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
LOCK = json.loads((ROOT / "backend.lock.json").read_text(encoding="utf-8"))


def version_tuple(value: str | None) -> tuple[int, ...]:
    match = re.match(r"\s*(\d+(?:\.\d+)*)", value or "")
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def source_probe(source: Path) -> tuple[str | None, bool | None]:
    if not (source / ".git").is_dir():
        return None, None
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        capture_output=True, check=False, text=True, timeout=15,
    )
    dirty = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True, check=False, text=True, timeout=15,
    )
    return (
        commit.stdout.strip() if commit.returncode == 0 else None,
        bool(dirty.stdout.strip()) if dirty.returncode == 0 else None,
    )


def broken_shebangs(environment: Path) -> list[str]:
    broken = []
    for path in sorted((environment / "bin").iterdir()):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            first = path.open("rb").readline(4096)
        except OSError:
            continue
        if first.startswith(b"#!"):
            interpreter = Path(os.fsdecode(first[2:].strip().split(b" ", 1)[0]))
            if interpreter.is_absolute() and not interpreter.exists():
                broken.append(path.name)
    return broken


def probe(environment: Path, source: Path, freeze: Path) -> dict[str, Any]:
    import torch

    try:
        from torch.distributed.fsdp import fully_shard  # noqa: F401
        fsdp2 = True
    except (ImportError, AttributeError):
        fsdp2 = False
    try:
        import torch.distributed.checkpoint  # noqa: F401
        dcp = True
    except ImportError:
        dcp = False
    try:
        import torchtitan
        torchtitan_import = True
        torchtitan_version = getattr(torchtitan, "__version__", None)
    except ImportError:
        torchtitan_import = False
        torchtitan_version = None
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True, check=False, text=True, timeout=300,
    )
    commit, dirty = source_probe(source)
    actual = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "executable": sys.executable,
        "prefix": sys.prefix,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torchvision": package_version("torchvision"),
        "torchao": package_version("torchao"),
        "torch_checkpointing": package_version("torch-checkpointing"),
        "triton": package_version("triton"),
        "torchtitan": torchtitan_version,
        "torchtitan_import": torchtitan_import,
        "fsdp2": fsdp2,
        "distributed_checkpoint": dcp,
        "pip_check_rc": pip_check.returncode,
        "pip_check_output": (pip_check.stdout + pip_check.stderr).strip(),
        "source_commit": commit,
        "source_dirty": dirty,
        "broken_console_shebangs": broken_shebangs(environment),
        "freeze_path": str(freeze),
        "freeze_sha256": sha256(freeze),
    }
    expected_cuda = version_tuple(LOCK["cuda"]["toolkit"])[:2]
    checks = {
        "python_3_12": version_tuple(actual["python"])[:2] == (3, 12),
        "environment_prefix": Path(sys.prefix).resolve() == environment.resolve(),
        "pytorch_2_14_0": version_tuple(actual["torch"])[:3] == (2, 14, 0),
        "cuda_build_matches_lock": version_tuple(actual["torch_cuda"])[:2] == expected_cuda,
        "torchvision_0_29_0": version_tuple(actual["torchvision"])[:3] == (0, 29, 0),
        "torchao_0_18_0": version_tuple(actual["torchao"])[:3] == (0, 18, 0),
        "torch_checkpointing_0_1_0": version_tuple(
            actual["torch_checkpointing"]
        )[:3] == (0, 1, 0),
        "triton_3_8_0": version_tuple(actual["triton"])[:3] == (3, 8, 0),
        "torchtitan_0_3": version_tuple(actual["torchtitan"])[:2] == (0, 3),
        "torchtitan_import": actual["torchtitan_import"] is True,
        "fsdp2": actual["fsdp2"] is True,
        "distributed_checkpoint": actual["distributed_checkpoint"] is True,
        "pip_check": actual["pip_check_rc"] == 0,
        "pinned_source_commit": actual["source_commit"] == LOCK["torchtitan"]["commit"],
        "clean_pinned_source": actual["source_dirty"] is False,
        "console_shebangs": not actual["broken_console_shebangs"],
        "dependency_freeze": actual["freeze_sha256"] is not None,
    }
    return {
        "schema_version": 1,
        "status": "ready" if all(checks.values()) else "blocked",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "actual": actual,
        "expected": LOCK,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.environment, args.source, args.freeze)
    atomic_json(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
