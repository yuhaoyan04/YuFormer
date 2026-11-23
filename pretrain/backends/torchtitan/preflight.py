#!/usr/bin/env python3
"""Fail-fast environment qualification for the YuFormer TorchTitan backend."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
LOCK = json.loads((ROOT / "backend.lock.json").read_text(encoding="utf-8"))


def atomic_write_text(path: Path, text: str) -> None:
    """Publish a complete report when several launchers preflight in parallel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"\s*(\d+(?:\.\d+)*)", value)
    return tuple(int(part) for part in match.group(1).split(".")) if match else ()


def at_least(actual: str | None, minimum: str) -> bool:
    if not actual:
        return False
    left = version_tuple(actual)
    right = version_tuple(minimum)
    width = max(len(left), len(right))
    return left + (0,) * (width - len(left)) >= right + (0,) * (width - len(right))


def probe_environment() -> dict[str, Any]:
    probe: dict[str, Any] = {
        "python": ".".join(str(x) for x in sys.version_info[:3]),
        "executable": sys.executable,
    }
    try:
        driver = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
        probe["nvidia_driver"] = (
            driver.stdout.splitlines()[0].strip()
            if driver.returncode == 0 and driver.stdout.strip()
            else None
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        probe["nvidia_driver"] = None

    try:
        import torch

        probe.update(
            {
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu_capability": (
                    list(torch.cuda.get_device_capability(0))
                    if torch.cuda.is_available()
                    else None
                ),
            }
        )
        try:
            from torch.distributed.fsdp import fully_shard  # noqa: F401

            probe["fsdp2"] = True
        except (ImportError, AttributeError):
            probe["fsdp2"] = False
        try:
            import torch.distributed.checkpoint  # noqa: F401

            probe["dcp"] = True
        except ImportError:
            probe["dcp"] = False
    except ImportError:
        probe.update(
            {
                "torch": None,
                "torch_cuda": None,
                "cuda_available": False,
                "gpu_capability": None,
                "fsdp2": False,
                "dcp": False,
            }
        )

    for package in ("torchvision", "torchao", "triton", "torch_checkpointing"):
        try:
            probe[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            probe[package] = None

    try:
        import torchtitan

        probe["torchtitan"] = getattr(torchtitan, "__version__", None)
    except ImportError:
        probe["torchtitan"] = None

    source_root = os.environ.get("YUFORMER_TORCHTITAN_ROOT")
    probe["torchtitan_source_root"] = source_root
    probe["torchtitan_commit"] = None
    probe["torchtitan_source_dirty"] = None
    if source_root:
        completed = subprocess.run(
            ["git", "-C", source_root, "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
        if completed.returncode == 0:
            probe["torchtitan_commit"] = completed.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", source_root, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
        if dirty.returncode == 0:
            probe["torchtitan_source_dirty"] = bool(dirty.stdout.strip())
    return probe


def evaluate_probe(probe: dict[str, Any]) -> dict[str, Any]:
    minimum_driver = LOCK["cuda"]["minimum_driver_linux"]
    expected_cuda = version_tuple(LOCK["cuda"]["toolkit"])[:2]
    expected_commit = LOCK["torchtitan"]["commit"]
    checks = {
        "python_3_12": version_tuple(probe.get("python") or "")[:2] == (3, 12),
        "pytorch_2_14_0": version_tuple(probe.get("torch") or "")[:3]
        == (2, 14, 0),
        "cuda_build_matches_lock": version_tuple(probe.get("torch_cuda") or "")[:2]
        == expected_cuda,
        "torchvision_0_29_0": version_tuple(probe.get("torchvision") or "")[:3]
        == (0, 29, 0),
        "torchao_0_18_0": version_tuple(probe.get("torchao") or "")[:3]
        == (0, 18, 0),
        "torch_checkpointing_0_1_0": version_tuple(
            probe.get("torch_checkpointing") or ""
        )[:3] == (0, 1, 0),
        "triton_3_8_0": version_tuple(probe.get("triton") or "")[:3]
        == (3, 8, 0),
        "nvidia_driver": at_least(probe.get("nvidia_driver"), minimum_driver),
        "hopper_or_newer": tuple(probe.get("gpu_capability") or ()) >= (9, 0),
        "cuda_available": probe.get("cuda_available") is True,
        "fsdp2": probe.get("fsdp2") is True,
        "distributed_checkpoint": probe.get("dcp") is True,
        "torchtitan_0_3": version_tuple(probe.get("torchtitan") or "")[:2] == (0, 3),
        "pinned_source_commit": probe.get("torchtitan_commit") == expected_commit,
        "clean_pinned_source": probe.get("torchtitan_source_dirty") is False,
    }
    return {
        "schema_version": 1,
        "status": "ready" if all(checks.values()) else "blocked",
        "checks": checks,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "expected": LOCK,
        "actual": probe,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = evaluate_probe(probe_environment())
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.out:
        atomic_write_text(args.out, rendered)
    print(rendered, end="")
    return 0 if result["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
