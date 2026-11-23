#!/usr/bin/env python3
"""Bind an already materialized tokenizer corpus to a newer PASS release.

This is only valid when the release keeps the exact immutable data snapshot and
train-order artifact used by the corpus. It is intended for governance evidence
that is signed while a long corpus extraction is running.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_bytes: int = 64 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_artifact(snapshot: dict[str, Any], path: Path) -> dict[str, Any]:
    resolved = path.resolve().as_posix()
    matches = [
        row for row in snapshot.get("artifacts", [])
        if isinstance(row, dict)
        and isinstance(row.get("path"), str)
        and (resolved == row["path"] or resolved.endswith("/" + row["path"]))
    ]
    if len(matches) != 1:
        raise ValueError(f"release 中未唯一声明 train order: {path}")
    return matches[0]


def bind(manifest_path: Path, release_path: Path, order_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS" or release.get("status") != "PASS":
        raise ValueError("corpus manifest and release must both be PASS")
    if manifest.get("snapshot_id") != release.get("snapshot_id"):
        raise ValueError("immutable snapshot changed; corpus must be rebuilt")
    dataset_id = release.get("dataset_id")
    if not dataset_id:
        raise ValueError("release lacks dataset_id")
    artifact = snapshot_artifact(release, order_path)
    declared_order = manifest.get("order") or {}
    actual_bytes = order_path.stat().st_size
    actual_sha = sha256_file(order_path)
    if (
        actual_bytes != int(artifact.get("bytes", -1))
        or actual_sha != artifact.get("sha256")
        or actual_bytes != int(declared_order.get("bytes", -2))
        or actual_sha != declared_order.get("sha256")
    ):
        raise ValueError("train order no longer matches release and corpus manifest")
    previous = manifest.get("dataset_id")
    manifest["dataset_id"] = dataset_id
    manifest["release_status_at_build"] = release["status"]
    manifest["release_binding"] = {
        "release_path": str(release_path),
        "release_sha256": sha256_file(release_path),
        "previous_dataset_id": previous,
        "reason": "governance evidence changed while extraction ran; immutable snapshot unchanged",
    }
    temporary = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--order", type=Path, required=True)
    args = parser.parse_args()
    manifest = bind(args.manifest, args.release, args.order)
    print(json.dumps({
        "status": manifest["status"],
        "dataset_id": manifest["dataset_id"],
        "snapshot_id": manifest["snapshot_id"],
        "manifest": str(args.manifest),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
