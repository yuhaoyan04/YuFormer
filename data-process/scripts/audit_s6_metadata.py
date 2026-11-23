#!/usr/bin/env python3
"""Audit S6 quota, curriculum, validation, config, and lineage metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "data-process" / "src"))

from yuformer_data.s6_audit import (  # noqa: E402
    audit_dedup_ledger,
    audit_length_distribution,
    audit_quantile_coverage,
    audit_s6_metadata,
)
from yuformer_data.scale.materialize import config_echo, load_mix_config  # noqa: E402


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--mix-config", type=Path, required=True)
    parser.add_argument(
        "--consistency-audit",
        type=Path,
        default=None,
        help="current audit_consistency JSON (defaults to newest dated artifact)",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.processed_root.resolve()
    cfg = load_mix_config(args.mix_config.resolve())
    mix_name = cfg["mix_name"]
    summary = _load(root / "stage_s6_global_summary.json")
    manifest = _load(root / "mix_order" / f"{mix_name}.manifest.json")
    val_summary = _load(root / "val_sets" / "_val_global_summary.json")
    result = audit_s6_metadata(
        summary,
        manifest,
        val_summary,
        expected_config_echo=config_echo(cfg),
    )
    gold = summary.get("goldrule_research") or {}
    gold_rel = manifest.get("goldrule_research")
    gold_path = root / str(gold_rel) if gold_rel else None
    required_tracks = list(gold.get("required_finite_tracks") or [])
    coverage = gold.get("track_coverage") or {}
    gold_errors: list[str] = []
    actual_gold_sha = None
    actual_gold_rows = None
    if not gold_path or not gold_path.is_file():
        gold_errors.append("declared goldrule parquet is missing")
    else:
        actual_gold_sha = _sha256(gold_path)
        parquet = pq.ParquetFile(gold_path)
        actual_gold_rows = parquet.metadata.num_rows
        columns = set(parquet.schema_arrow.names)
        if any(name not in columns for name in required_tracks):
            gold_errors.append("required finite-track columns are missing")
    if int(gold.get("rows") or 0) < int(gold.get("min_total_rows") or 0):
        gold_errors.append("goldrule row count is below configured floor")
    if actual_gold_rows != int(gold.get("rows") or 0):
        gold_errors.append("parquet row count does not match summary")
    if not required_tracks or any(
        int((coverage.get(name) or {}).get("nonfinite") or 0) != 0
        or float((coverage.get(name) or {}).get("fraction") or 0.0) != 1.0
        for name in required_tracks
    ):
        gold_errors.append("required score tracks are not 100% finite")
    declared_gold_shas = {
        str(gold.get("sha256") or ""),
        str(manifest.get("goldrule_research_sha256") or ""),
        str(actual_gold_sha or ""),
    }
    if "" in declared_gold_shas or len(declared_gold_shas) != 1:
        gold_errors.append("goldrule SHA-256 identity does not close")
    result["checks"]["goldrule_research"] = {
        "passed": not gold_errors,
        "detail": "published C4 parquet meets row floor, schema, finite-track, and SHA-256 contracts",
        "errors": gold_errors,
        "rows": actual_gold_rows,
        "sha256": actual_gold_sha,
        "required_finite_tracks": required_tracks,
    }
    result["gates"]["C4"] = {"status": "PASS" if not gold_errors else "FAIL"}
    stage_s5 = _load(root / "stage_s5_global_summary.json")
    consistency_path = args.consistency_audit
    if consistency_path is None:
        candidates = sorted(root.glob("audit_consistency_*.json"))
        if not candidates:
            raise FileNotFoundError("no audit_consistency_*.json under processed root")
        consistency_path = candidates[-1]
    consistency = _load(consistency_path.resolve())
    result["consistency_audit"] = str(consistency_path.resolve())
    ndv2_global = _load(root / "stage_ndv2_global_summary.json")
    ndv2_decisions = {
        domain: _load(root / "s3_decisions" / f"{domain}.summary.json")
        for domain in (ndv2_global.get("per_domain") or {})
    }
    ndv2_lookup_rows = {
        domain: {
            "ids_rows": (root / "s3_lookup" / domain / "lookup.ids.bin").stat().st_size // 16,
            "codes_rows": (root / "s3_lookup" / domain / "lookup.codes.bin").stat().st_size,
        }
        for domain in (ndv2_global.get("per_domain") or {})
    }
    result["checks"]["dedup_ledger"] = audit_dedup_ledger(
        stage_s5, consistency, ndv2_global, ndv2_decisions, ndv2_lookup_rows
    )
    result["gates"]["B2"] = {
        "status": result["checks"]["dedup_ledger"]["status"]
    }
    score_suffix = {
        "composite_v2": "",
        "rule_composite": ".rule",
        "hybrid_composite": ".hybrid",
        "hybrid_v2": ".hybrid2",
    }[cfg["score_field"]]
    tables_root = root / "s4_quantiles" / "tables"
    quantile_meta = _load(tables_root / f"_build{score_suffix}.summary.json")
    result["checks"]["quantile_coverage"] = audit_quantile_coverage(
        stage_s5, quantile_meta, tables_root
    )
    result["gates"]["C3"] = {
        "status": result["checks"]["quantile_coverage"]["status"]
    }
    length_cfg = cfg.get("length_audit") or {}
    length_audit = audit_length_distribution(
        stage_s5,
        primary_context=int(length_cfg.get("primary_context", 4096)),
        long_context_min_tokens=int(length_cfg.get("long_context_min_tokens", 1_000_000_000)),
    )
    result["checks"]["length_distribution"] = length_audit
    result["checks"].pop("length_bucket_rows", None)
    result["gates"]["E3"] = {
        "status": length_audit["status"],
        "reason": length_audit["detail"],
    }
    # audit_s6_metadata computes the core metadata status before this CLI adds
    # B2/C3/E3. Fold the added machine gates back into the process exit status.
    if any(
        result["gates"][gate]["status"] != "PASS"
        for gate in ("B2", "C3", "C4", "E3")
    ):
        result["status"] = "FAIL"
    destination = args.out.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, destination)
    print(json.dumps({"status": result["status"], "gates": result["gates"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
