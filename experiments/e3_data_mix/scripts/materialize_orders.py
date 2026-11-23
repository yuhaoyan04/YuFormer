#!/usr/bin/env python3
"""Derive the eight E3 order tables from the audited 100B source release.

Broad arms use the source's content-hash order as common random numbers and take
the largest complete-document prefix that fits each domain quota.  The quality
arm takes the highest-composite prefix per domain, with source doc_id order as
the deterministic tie break, then restores content-hash order for training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[3]
SCHEMA = pa.schema([
    pa.field("position", pa.int64(), nullable=False),
    pa.field("doc_id", pa.string(), nullable=False),
    pa.field("domain", pa.string(), nullable=False),
    pa.field("n_tokens", pa.uint32(), nullable=False),
])


def sha256(path: Path, chunk_size: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def allocate(weights: dict[str, float], total: int) -> dict[str, int]:
    raw = {key: total * value for key, value in weights.items()}
    quotas = {key: int(value) for key, value in raw.items()}
    remaining = total - sum(quotas.values())
    order = sorted(weights, key=lambda key: (-(raw[key] - quotas[key]), key))
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def prefix_count(tokens: np.ndarray, quota: int) -> int:
    if quota <= 0 or not len(tokens):
        return 0
    count = int(np.searchsorted(np.cumsum(tokens, dtype=np.uint64), quota, side="right"))
    return count or 1


def _with_positions(table: pa.Table, start: int) -> pa.Table:
    table = table.select(["doc_id", "domain", "n_tokens"])
    return pa.Table.from_arrays(
        [pa.array(np.arange(start, start + len(table), dtype=np.int64)), *table.columns],
        schema=SCHEMA,
    )


def materialize_broad(source_order: Path, arms: dict[str, Any],
                      quotas: dict[str, dict[str, int]], temporary: dict[str, Path]) -> dict[str, Any]:
    writers = {name: pq.ParquetWriter(path, SCHEMA, compression="zstd") for name, path in temporary.items()}
    totals = {name: {domain: 0 for domain in arm["weights"]} for name, arm in arms.items()}
    counts = {name: 0 for name in arms}
    try:
        parquet = pq.ParquetFile(source_order)
        for batch in parquet.iter_batches(batch_size=1_000_000,
                                          columns=["doc_id", "domain", "n_tokens"]):
            table = pa.Table.from_batches([batch])
            domains = np.asarray(table["domain"].to_pylist(), dtype=object)
            ntokens = table["n_tokens"].combine_chunks().to_numpy(
                zero_copy_only=False).astype(np.uint64, copy=False)
            for name, arm in arms.items():
                chosen = np.zeros(len(table), dtype=bool)
                for domain in arm["weights"]:
                    positions = np.flatnonzero(domains == domain)
                    if not len(positions):
                        continue
                    remaining = quotas[name][domain] - totals[name][domain]
                    if remaining <= 0:
                        continue
                    cumulative = np.cumsum(ntokens[positions], dtype=np.uint64)
                    count = int(np.searchsorted(cumulative, remaining, side="right"))
                    if count == 0 and totals[name][domain] == 0:
                        count = 1
                    selected = positions[:count]
                    chosen[selected] = True
                    totals[name][domain] += int(ntokens[selected].sum(dtype=np.uint64))
                indices = np.flatnonzero(chosen)
                if not len(indices):
                    continue
                selected = table.take(pa.array(indices))
                positioned = _with_positions(selected, counts[name])
                writers[name].write_table(positioned, row_group_size=1_000_000)
                counts[name] += len(positioned)
    finally:
        for writer in writers.values():
            writer.close()
    return {name: {"docs": counts[name], "tokens_by_domain": totals[name]} for name in arms}


def materialize_quality(members: Path, quota: dict[str, int], output: Path) -> dict[str, Any]:
    source = pq.read_table(members, columns=["doc_id", "domain", "composite", "n_tokens"])
    selected_parts = []
    tokens_by_domain = {}
    for domain, domain_quota in quota.items():
        domain_table = source.filter(pc.equal(source["domain"], domain))
        scores = domain_table["composite"].combine_chunks().to_numpy(zero_copy_only=False)
        if not np.isfinite(scores).all():
            raise ValueError(f"{domain}: non-finite composite score")
        # Members are already doc_id-sorted; stable sorting preserves the exact
        # documented doc_id tie break for equal composite scores.
        order = np.argsort(-scores, kind="stable")
        tokens = domain_table["n_tokens"].combine_chunks().to_numpy(zero_copy_only=False)
        count = prefix_count(tokens[order], domain_quota)
        chosen = domain_table.take(pa.array(order[:count]))
        selected_parts.append(chosen)
        tokens_by_domain[domain] = int(chosen["n_tokens"].combine_chunks().to_numpy(
            zero_copy_only=False).sum(dtype=np.uint64))
    selected = pa.concat_tables(selected_parts).sort_by([("doc_id", "ascending")])
    positioned = _with_positions(selected, 0)
    pq.write_table(positioned, output, compression="zstd", row_group_size=1_000_000)
    return {"docs": len(positioned), "tokens_by_domain": tokens_by_domain}


def materialize(root: Path, matrix_path: Path, output_dir: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    source_order = root / cfg["source_manifest"]
    source_manifest = json.loads(source_order.read_text(encoding="utf-8"))
    order_names = source_manifest.get("order_tables", [])
    if len(order_names) != 1:
        raise ValueError("E3 source must expose exactly one random order table")
    source_order = source_order.parent / order_names[0]
    members = root / cfg["source_members"]
    release_path = root / cfg["source_release"]
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if release.get("status") != "PASS":
        raise ValueError("source release is not PASS")
    matrix_sha = sha256(matrix_path)
    inputs = {str(path.relative_to(root)): {"size": path.stat().st_size, "sha256": sha256(path)}
              for path in (matrix_path, members, source_order, release_path)}
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    previous = None
    try:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    if (previous and previous.get("status") == "PASS"
            and previous.get("materializer_version") == 2 and previous.get("inputs") == inputs):
        output_rows = previous.get("outputs", {})
        if output_rows and all((output_dir / row["file"]).is_file()
                               and sha256(output_dir / row["file"]) == row["sha256"]
                               for row in output_rows.values()):
            return {**previous, "idempotent_skip": True}

    total = int(cfg["budget_tokens_per_arm"])
    arms = cfg["arms"]
    quotas = {name: allocate({key: float(value) for key, value in arm["weights"].items()}, total)
              for name, arm in arms.items()}
    broad = {name: arm for name, arm in arms.items() if arm["selection"] == "deterministic_broad"}
    quality = {name: arm for name, arm in arms.items() if arm["selection"] == "quality_prefix"}
    if len(quality) != 1:
        raise ValueError("E3 requires exactly one quality_prefix arm")
    temporary = {name: output_dir / f".{name}.parquet.tmp-{os.getpid()}" for name in broad}
    for path in temporary.values():
        path.unlink(missing_ok=True)
    quality_name = next(iter(quality))
    quality_tmp = output_dir / f".{quality_name}.parquet.tmp-{os.getpid()}"
    quality_tmp.unlink(missing_ok=True)
    try:
        stats = materialize_broad(source_order, broad, quotas, temporary)
        stats[quality_name] = materialize_quality(members, quotas[quality_name], quality_tmp)
        temporary[quality_name] = quality_tmp
        outputs = {}
        for name in sorted(arms):
            final = output_dir / f"{name}.parquet"
            os.replace(temporary[name], final)
            actual = stats[name]["tokens_by_domain"]
            shortfalls = {domain: quotas[name][domain] - actual[domain] for domain in quotas[name]}
            if any(value < 0 for value in shortfalls.values()):
                raise ValueError(f"{name}: complete-document selection exceeded quota")
            outputs[name] = {
                "file": final.name, "sha256": sha256(final), "size": final.stat().st_size,
                "selection": arms[name]["selection"], "quotas": quotas[name], **stats[name],
                "shortfall_tokens_by_domain": shortfalls,
                "total_tokens": sum(actual.values()),
            }
        manifest = {
            "schema_version": 1, "status": "PASS",
            "materializer_version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "release_name": "e3-data-mix-orders-20261021",
            "dataset_id": release.get("dataset_id"),
            "snapshot_id": release.get("snapshot_id"),
            "matrix_sha256": matrix_sha, "source_snapshot_id": release.get("snapshot_id"),
            "inputs": inputs, "outputs": outputs,
            "artifacts": [
                {"path": str((output_dir / row["file"]).relative_to(root)),
                 "bytes": row["size"], "sha256": row["sha256"]}
                for row in outputs.values()
            ],
        }
        tmp_manifest = manifest_path.with_name(f".{manifest_path.name}.tmp-{os.getpid()}")
        tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp_manifest, manifest_path)
        return manifest
    except Exception:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        quality_tmp.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--matrix", type=Path,
                        default=ROOT / "experiments/e3_data_mix/configs/e3_matrix.yaml")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data/experiments/e3_data_mix/orders")
    args = parser.parse_args()
    result = materialize(args.root.resolve(), args.matrix.resolve(), args.output_dir.resolve())
    print(json.dumps({"status": result["status"], "arms": len(result["outputs"]),
                      "idempotent_skip": result.get("idempotent_skip", False)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
