#!/usr/bin/env python3
"""Exact D3 audit: validation IDs are unique, quota-complete, and absent from train."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def audit_train_val_ids(
    members_path: Path,
    val_root: Path,
    *,
    batch_size: int = 1_000_000,
) -> dict:
    global_summary = json.loads(
        (val_root / "_val_global_summary.json").read_text(encoding="utf-8")
    )
    declared = global_summary.get("per_domain") or {}
    val_arrays: list[pa.Array] = []
    per_domain: dict[str, dict] = {}
    flags: list[str] = []
    for domain, expected in sorted(declared.items()):
        path = val_root / f"{domain}.parquet"
        if not path.is_file():
            flags.append(f"{domain}: validation parquet missing")
            continue
        table = pq.read_table(path, columns=["doc_id", "domain", "n_tokens"])
        ids = table.column("doc_id").combine_chunks()
        domains = table.column("domain").combine_chunks()
        tokens = table.column("n_tokens").combine_chunks()
        unique = int(pc.count_distinct(ids).as_py() or 0)
        rows = len(ids)
        token_sum = int(pc.sum(tokens).as_py() or 0)
        domain_values = set(pc.unique(domains).to_pylist())
        expected_docs = int(expected.get("val_docs") or 0)
        expected_tokens = int(expected.get("val_tokens") or 0)
        quota = int(expected.get("quota_tokens") or 0)
        passed = (
            unique == rows == expected_docs
            and token_sum == expected_tokens
            and token_sum >= quota
            and domain_values == {domain}
        )
        if not passed:
            flags.append(f"{domain}: validation metadata/uniqueness mismatch")
        per_domain[domain] = {
            "rows": rows,
            "unique_ids": unique,
            "tokens": token_sum,
            "quota_tokens": quota,
            "domain_values": sorted(str(value) for value in domain_values),
            "passed": passed,
        }
        val_arrays.append(ids)

    if not val_arrays:
        flags.append("no validation IDs found")
        val_ids = pa.array([], type=pa.string())
    else:
        val_ids = pa.concat_arrays(val_arrays)
    global_val_unique = int(pc.count_distinct(val_ids).as_py() or 0)
    if global_val_unique != len(val_ids):
        flags.append(
            f"validation IDs overlap across domains: rows={len(val_ids)} unique={global_val_unique}"
        )

    train_rows = 0
    intersection = 0
    parquet = pq.ParquetFile(members_path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=["doc_id"]):
        ids = batch.column(0)
        train_rows += len(ids)
        intersection += int(pc.sum(pc.is_in(ids, value_set=val_ids)).as_py() or 0)
    if intersection:
        flags.append(f"train/validation intersection contains {intersection} rows")

    return {
        "schema_version": 1,
        "status": "PASS" if not flags else "FAIL",
        "gate": "D3",
        "members_path": str(members_path),
        "val_root": str(val_root),
        "train_rows_scanned": train_rows,
        "validation_rows": len(val_ids),
        "validation_unique_ids": global_val_unique,
        "train_val_intersection_rows": intersection,
        "per_domain": per_domain,
        "flags": flags,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", type=Path, required=True)
    parser.add_argument("--val-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit_train_val_ids(
        args.members.resolve(), args.val_root.resolve(), batch_size=args.batch_size
    )
    destination = args.out.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, destination)
    print(
        json.dumps(
            {
                "status": result["status"],
                "train_rows_scanned": result["train_rows_scanned"],
                "validation_rows": result["validation_rows"],
                "intersection": result["train_val_intersection_rows"],
                "flags": result["flags"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
