#!/usr/bin/env python3
"""Create the deterministic pretrain code-freeze candidate manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TREE_SCOPES = (
    "pretrain/backends",
    "pretrain/configs",
    "pretrain/docs",
    "pretrain/scripts",
    "pretrain/src",
    "pretrain/tests",
    'experiments/_common',
    "experiments/e3_data_mix/configs",
    "experiments/e3_data_mix/scripts",
    "experiments/e3_data_mix/tests",
    "experiments/e4_curriculum/configs",
    "experiments/e4_curriculum/scripts",
    "experiments/e4_curriculum/tests",
    "model/configs",
    "model/src",
    "model/tests",
)
SINGLE_FILES = (
    "pretrain/README.md",
    "pretrain/pyproject.toml",
    "experiments/e1_scaling/README.md",
    "experiments/e2_architecture/README.md",
    "experiments/e3_data_mix/README.md",
    "experiments/e4_curriculum/README.md",
)
STATIC_SUFFIXES = {".c", ".env", ".json", ".md", ".py", ".sh", ".toml", ".yaml", ".yml"}
EXCLUDED = {"pretrain/tests/verify/verify_report.json"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files(root: Path) -> list[str]:
    paths: set[str] = set(SINGLE_FILES)
    for scope in TREE_SCOPES:
        for path in (root / scope).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix in STATIC_SUFFIXES:
                paths.add(path.relative_to(root).as_posix())
    return sorted(path for path in paths if path not in EXCLUDED)


def build_manifest(root: Path) -> dict[str, object]:
    files = {
        relative: {
            "bytes": (root / relative).stat().st_size,
            "sha256": sha256(root / relative),
        }
        for relative in collect_files(root)
    }
    identity = "".join(
        f"{relative}\0{metadata['sha256']}\n"
        for relative, metadata in files.items()
    )
    return {
        "candidate_scope_note": (
            "Source, configuration, tests, and static documentation; "
            "runtime-generated verification reports are excluded."
        ),
        "candidate_sha256": hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "excluded_generated_files": sorted(EXCLUDED),
        "file_count": len(files),
        "files": files,
        "schema_version": 1,
        "status": "PASS",
    }


def atomic_json(path: Path, payload: dict[str, object]) -> None:
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
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data/health/pretrain-code-freeze-candidate.json",
    )
    args = parser.parse_args()
    manifest = build_manifest(ROOT)
    atomic_json(args.out, manifest)
    print(json.dumps({key: manifest[key] for key in ("status", "file_count", "candidate_sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
