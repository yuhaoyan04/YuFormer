#!/usr/bin/env python3
"""Independently verify a snapshot-bound tokenizer training corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


REQUIRED_COLUMNS = {"doc_id", "domain", "n_tokens_reference", "text"}


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
        raise ValueError(f"snapshot 中未唯一声明 train order: {path}")
    return matches[0]


def load_validation_ids(val_dir: Path) -> set[str]:
    ids: set[str] = set()
    rows = 0
    for path in sorted(val_dir.glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=131_072, columns=["doc_id"]):
            values = batch.column(0).to_pylist()
            rows += len(values)
            ids.update(values)
    if len(ids) != rows:
        raise ValueError(f"validation doc_id 本身不唯一: rows={rows}, unique={len(ids)}")
    return ids


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--order", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    errors: list[str] = []
    manifest_path = args.corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    artifact = snapshot_artifact(snapshot, args.order)
    val_ids = load_validation_ids(args.val_dir)

    checks = {
        "manifest_pass": manifest.get("status") == "PASS",
        "release_pass": snapshot.get("status") == "PASS",
        "snapshot_id_exact": manifest.get("snapshot_id") == snapshot.get("snapshot_id"),
        "dataset_id_exact": manifest.get("dataset_id") == snapshot.get("dataset_id"),
        "order_bytes_snapshot_exact": args.order.stat().st_size == int(artifact.get("bytes", -1)),
        "order_sha_snapshot_exact": sha256_file(args.order) == artifact.get("sha256"),
        "order_bytes_manifest_exact": int((manifest.get("order") or {}).get("bytes", -1)) == int(artifact.get("bytes", -2)),
        "order_sha_manifest_exact": (manifest.get("order") or {}).get("sha256") == artifact.get("sha256"),
        "recipe_sha_exact": (manifest.get("recipe") or {}).get("sha256") == sha256_file(args.recipe),
    }

    declared_files = manifest.get("files") or []
    declared_by_name = {
        str(row.get("path")): row for row in declared_files if isinstance(row, dict)
    }
    actual_paths = sorted(args.corpus.glob("*.parquet"))
    checks["file_roster_exact"] = set(declared_by_name) == {p.name for p in actual_paths}
    domain_stats: dict[str, dict[str, int]] = {}
    val_intersection = 0
    malformed_ids = 0
    for path in actual_paths:
        domain = path.stem
        declared = declared_by_name.get(path.name) or {}
        file_checks = {
            "bytes_exact": path.stat().st_size == int(declared.get("bytes", -1)),
            "sha256_exact": sha256_file(path) == declared.get("sha256"),
        }
        parquet = pq.ParquetFile(path)
        file_checks["schema_exact"] = set(parquet.schema_arrow.names) == REQUIRED_COLUMNS
        stats = {"docs": 0, "reference_tokens": 0, "utf8_bytes": 0}
        if file_checks["schema_exact"]:
            for batch in parquet.iter_batches(
                batch_size=65_536,
                columns=["doc_id", "domain", "n_tokens_reference", "text"],
            ):
                doc_ids = batch.column(0).to_pylist()
                domains = batch.column(1).to_pylist()
                tokens = batch.column(2).to_pylist()
                texts = batch.column(3).to_pylist()
                stats["docs"] += len(doc_ids)
                stats["reference_tokens"] += sum(int(value) for value in tokens)
                stats["utf8_bytes"] += sum(len(text.encode("utf-8")) for text in texts)
                val_intersection += sum(doc_id in val_ids for doc_id in doc_ids)
                malformed_ids += sum(
                    not isinstance(doc_id, str)
                    or len(doc_id) != 32
                    or any(ch not in "0123456789abcdef" for ch in doc_id)
                    for doc_id in doc_ids
                )
                if any(value != domain for value in domains):
                    file_checks["domain_column_exact"] = False
        file_checks.setdefault("domain_column_exact", True)
        expected = (manifest.get("domains") or {}).get(domain)
        file_checks["stats_exact"] = stats == expected
        domain_stats[domain] = stats
        for name, passed in file_checks.items():
            if not passed:
                errors.append(f"{path.name}:{name}")

    totals = {
        key: sum(row[key] for row in domain_stats.values())
        for key in ("docs", "reference_tokens", "utf8_bytes")
    }
    checks["totals_exact"] = totals == manifest.get("totals")
    checks["validation_intersection_zero"] = val_intersection == 0
    checks["doc_ids_well_formed"] = malformed_ids == 0
    targets = (manifest.get("recipe") or {}).get("target_reference_tokens") or {}
    checks["all_domain_targets_met"] = bool(targets) and all(
        domain in domain_stats
        and domain_stats[domain]["reference_tokens"] >= int(target)
        for domain, target in targets.items()
    )
    errors.extend(sorted(name for name, passed in checks.items() if not passed))
    payload = {
        "schema_version": 1,
        "audit": "tokenizer_corpus_release",
        "status": "PASS" if not errors else "FAIL",
        "snapshot_id": snapshot.get("snapshot_id"),
        "dataset_id": snapshot.get("dataset_id"),
        "checks": checks,
        "errors": errors,
        "validation_ids": len(val_ids),
        "validation_intersection": val_intersection,
        "malformed_doc_ids": malformed_ids,
        "domains": domain_stats,
        "totals": totals,
    }
    atomic_json(args.out, payload)
    print(json.dumps({"status": payload["status"], "out": str(args.out), "errors": errors}, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
