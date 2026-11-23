#!/usr/bin/env python3
"""Publish a hash-bound derived release for frozen 1T order tables."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "data-process/src"))
from yuformer_data.scale.materialize import config_echo, load_mix_config  # noqa: E402

STAGES = ("warmup", "stable", "anneal")
DOMAINS = ("web", "code", "math", "chinese", "academic", "books",
           "encyclopedia", "qa", "multilingual")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(32 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(root: Path, path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    identity_before = (before.st_size, before.st_dev, before.st_ino, before.st_mtime_ns)
    identity_after = (after.st_size, after.st_dev, after.st_ino, after.st_mtime_ns)
    if identity_before != identity_after:
        raise ValueError(f"artifact changed while hashing: {path}")
    return {"path": str(path.relative_to(root)), "bytes": after.st_size,
            "device": after.st_dev, "inode": after.st_ino, "mtime_ns": after.st_mtime_ns,
            "sha256": digest}


def validation_parquet_domains(artifacts: list[dict[str, Any]]) -> set[str]:
    return {
        Path(row["path"]).stem
        for row in artifacts
        if "/val_sets/" in row.get("path", "")
        and row["path"].endswith(".parquet")
        and not row["path"].endswith("_summary.parquet")
    }


def upstream_snapshot_identity(manifest: dict[str, Any]) -> str:
    identity = manifest.get("upstream_snapshot_sha256")
    snapshot = manifest.get("upstream_snapshot") or {}
    if not isinstance(identity, str) or not identity or snapshot.get("snapshot_sha256") != identity:
        raise ValueError("S6 manifest has an unclosed upstream snapshot identity")
    return identity


def audit_order(path: Path, *, expected_docs: int, expected_tokens: int) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    required = {"position", "doc_id", "domain", "n_tokens"}
    if not required.issubset(parquet.schema_arrow.names):
        raise ValueError(f"order schema is incomplete: {path}")
    rows = 0
    tokens = 0
    previous_doc_id = None
    for batch in parquet.iter_batches(
            batch_size=1 << 20, columns=["position", "doc_id", "n_tokens"]):
        if batch.num_rows == 0:
            continue
        positions = batch.column(0).to_numpy(zero_copy_only=False)
        doc_ids = batch.column(1)
        lengths = batch.column(2).to_numpy(zero_copy_only=False)
        expected = np.arange(rows, rows + batch.num_rows, dtype=positions.dtype)
        if not np.array_equal(positions, expected):
            raise ValueError(f"order positions are not contiguous at row {rows}: {path}")
        if pa.types.is_large_string(doc_ids.type):
            offset_dtype, offset_width = np.int64, 8
        elif pa.types.is_string(doc_ids.type):
            offset_dtype, offset_width = np.int32, 4
        else:
            raise ValueError(f"unsupported doc_id type {doc_ids.type}: {path}")
        offsets = np.frombuffer(doc_ids.buffers()[1], dtype=offset_dtype,
                                count=len(doc_ids) + 1, offset=doc_ids.offset * offset_width)
        if doc_ids.null_count or not bool(np.all(np.diff(offsets) == 32)):
            raise ValueError(f"order contains null/non-32-byte doc_id: {path}")
        data = memoryview(doc_ids.buffers()[2])[int(offsets[0]):int(offsets[-1])]
        id_values = np.frombuffer(data, dtype="S32", count=len(doc_ids))
        if (len(id_values) > 1 and bool(np.any(id_values[1:] <= id_values[:-1]))) or (
                previous_doc_id is not None and id_values[0] <= previous_doc_id):
            raise ValueError(f"order doc_ids are not globally strict-sorted: {path}")
        previous_doc_id = id_values[-1]
        rows += batch.num_rows
        tokens += int(lengths.astype(np.uint64).sum(dtype=np.uint64))
    if rows != expected_docs or tokens != expected_tokens:
        raise ValueError(
            f"order accounting mismatch: {path}: docs={rows}/{expected_docs}, "
            f"tokens={tokens}/{expected_tokens}"
        )
    return {"docs": rows, "reference_tokens": tokens, "positions_contiguous": True,
            "doc_ids_strictly_sorted": True}


def build(root: Path) -> dict[str, Any]:
    freeze_path = root / "experiments/e4_curriculum/results/frozen_1t_recipe.json"
    state_path = root / "data/health/pretrain-1t-order-materialization.json"
    base_path = root / "data/processed/releases/mix_m3_e1_100b-20261019.release.json"
    freeze, state, base = (json.loads(path.read_text(encoding="utf-8"))
                           for path in (freeze_path, state_path, base_path))
    if freeze.get("status") != "PASS" or state.get("status") != "PASS" or base.get("status") != "PASS":
        raise ValueError("freeze, 1T order state, and base release must all PASS")
    freeze_tool = root / "experiments/e4_curriculum/scripts/freeze_1t_recipe.py"
    if freeze.get("freeze_tool_sha256") != sha256(freeze_tool):
        raise ValueError("frozen recipe uses a stale freeze tool")
    base_artifacts = {
        str(row.get("path")): row for row in base.get("artifacts", [])
        if isinstance(row, dict) and isinstance(row.get("path"), str)
    }
    base_manifest_path = root / "data/processed/mix_order/mix_m3_e1_100b.manifest.json"
    base_manifest_record = base_artifacts.get(str(base_manifest_path.relative_to(root)))
    if (not base_manifest_record or not base_manifest_path.is_file()
            or int(base_manifest_record.get("bytes", -1)) != base_manifest_path.stat().st_size
            or base_manifest_record.get("sha256") != sha256(base_manifest_path)):
        raise ValueError("base release no longer binds its S6 manifest")
    base_upstream_snapshot = upstream_snapshot_identity(
        json.loads(base_manifest_path.read_text(encoding="utf-8"))
    )
    for relative, expected in (freeze.get("outputs") or {}).items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"frozen recipe output drift: {relative}")
    jobs = state.get("jobs") or {}
    if set(jobs) != set(STAGES) or any(jobs[name].get("status") != "PASS" for name in STAGES):
        raise ValueError("1T order state does not contain exactly three PASS jobs")
    artifacts, stages = [], {}
    for stage in STAGES:
        config_path = root / freeze["stages"][stage]["config"]
        cfg = load_mix_config(config_path)
        mix_name = cfg["mix_name"]
        manifest_path = root / f"data/processed/mix_order/{mix_name}.manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_echo") != config_echo(cfg):
            raise ValueError(f"{stage}: manifest does not match the frozen mix config")
        if len(manifest.get("order_tables", [])) != 1:
            raise ValueError(f"{stage}: expected exactly one order table")
        if upstream_snapshot_identity(manifest) != base_upstream_snapshot:
            raise ValueError(f"{stage}: upstream snapshot differs from the base release")
        order_path = manifest_path.parent / manifest["order_tables"][0]
        members_path = root / "data/processed" / manifest["members"]
        expected_docs = int(manifest["stage_stats"]["stage2"]["docs"])
        expected_tokens = int(manifest["stage_stats"]["stage2"]["tokens"])
        order_audit = audit_order(
            order_path, expected_docs=expected_docs, expected_tokens=expected_tokens
        )
        published = {
            str(row.get("path")): row for row in manifest.get("published_artifacts", [])
            if isinstance(row, dict)
        }
        current_artifacts = [artifact(root, path)
                             for path in (config_path, manifest_path, order_path, members_path)]
        processed_root = root / "data/processed"
        for path, current in zip(
                (config_path, manifest_path, order_path, members_path), current_artifacts):
            if path in (order_path, members_path):
                relative = path.relative_to(processed_root).as_posix()
                recorded = published.get(relative)
                if not recorded or recorded.get("sha256") != current["sha256"]:
                    raise ValueError(f"{stage}: published artifact hash mismatch: {relative}")
            artifacts.append(current)
        stages[stage] = {
            "mix_name": mix_name, "config": str(config_path.relative_to(root)),
            "manifest": str(manifest_path.relative_to(root)),
            "order": str(order_path.relative_to(root)),
            "members": str(members_path.relative_to(root)),
            "selection_reference_tokens": expected_tokens,
            "selection_docs": expected_docs,
            "order_audit": order_audit,
            "target_final_tokens": int(cfg["target_final_tokens"]),
        }
    # Validation remains the immutable held-out split from the same snapshot.
    for row in base["artifacts"]:
        if "/val_sets/" in row.get("path", ""):
            artifacts.append(row)
    validation_parquets = validation_parquet_domains(artifacts)
    if validation_parquets != set(DOMAINS):
        raise ValueError(
            f"base release validation roster is not exact: {sorted(validation_parquets)}"
        )
    return {
        "schema_version": 1, "status": "PASS", "release_name": "pretrain-1t-frozen",
        "builder_tool_sha256": sha256(Path(__file__)),
        "dataset_id": base["dataset_id"], "snapshot_id": base["snapshot_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frozen_recipe": str(freeze_path.relative_to(root)),
        "frozen_recipe_sha256": sha256(freeze_path), "stages": stages,
        "artifacts": sorted(artifacts, key=lambda row: row["path"]),
    }


def publish_release(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
        old = {key: value for key, value in previous.items() if key != "created_at"}
        new = {key: value for key, value in result.items() if key != "created_at"}
        if old == new and isinstance(previous.get("created_at"), str):
            return previous
    except (OSError, ValueError, AttributeError):
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/processed/releases/pretrain-1t-frozen.release.json")
    args = parser.parse_args()
    result = build(args.root.resolve())
    result = publish_release(args.output, result)
    print(json.dumps({"status": result["status"], "stages": len(result["stages"]),
                      "artifacts": len(result["artifacts"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
