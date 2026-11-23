"""Stage-to-stage consistency audit for a declared release lineage.

Reads per-shard summary.json files (SFS-heavy; uses a thread pool) for s0/s1/s2/s3/decontam
plus decision parquet footers and emits a reconciliation report.  A release scope is
mandatory: failures from explicitly excluded sources remain visible as warnings, while
any error, missing stage, or conservation failure in an included source fails the audit.
No corpus parquet is read end-to-end.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

STAGES = ("s0", "s1", "s2", "s3", "decontam")


def iter_summaries(stage_dir: Path):
    for p in stage_dir.rglob("*.summary.json"):
        yield p


def load_summary(path: Path):
    try:
        with open(path, "rb") as f:
            d = json.load(f)
        return path, d, None
    except Exception as e:
        return path, {}, f"unreadable: {e}"


def aggregate_stage(
    processed_root: Path,
    stage: str,
    workers: int = 32,
    *,
    verify_output_footers: bool = True,
):
    stage_dir = processed_root / stage
    if not stage_dir.is_dir():
        return {
            "missing_dir": True,
            "per_source": {},
            "n_summaries": 0,
            "n_success_summaries": 0,
            "n_error_summaries": 0,
            "error_counts_by_source": {},
            "error_kind_counts": {},
            "error_summaries_sample": [],
            "artifact_error_counts_by_source": {},
            "artifact_error_sample": [],
            "n_artifact_errors": 0,
        }
    per_source: dict[str, dict] = {}
    errors: list[dict] = []
    error_counts_by_source: collections.Counter[str] = collections.Counter()
    error_kind_counts: collections.Counter[str] = collections.Counter()
    output_records: list[tuple[str, Path, int, int]] = []
    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for path, d, err in ex.map(load_summary, iter_summaries(stage_dir)):
            n += 1
            if err or not isinstance(d, dict) or d.get("errors") or d.get("error"):
                source = d.get("source") if isinstance(d, dict) else None
                if not source:
                    try:
                        source = path.relative_to(stage_dir).parts[0]
                    except (ValueError, IndexError):
                        source = "?"
                detail = err or d.get("error") or d.get("errors")
                error_counts_by_source[str(source)] += 1
                if isinstance(detail, dict):
                    for kind, count in detail.items():
                        error_kind_counts[str(kind)] += int(count)
                else:
                    error_kind_counts[str(detail)] += 1
                errors.append({"path": str(path.relative_to(processed_root)), "detail": detail})
                continue
            src = d.get("source") or "?"
            agg = per_source.setdefault(
                src,
                {
                    "n_shards": 0,
                    "in_rows": 0,
                    "out_rows": 0,
                    "output_bytes": 0,
                    "decisions": {},
                },
            )
            agg["n_shards"] += 1
            agg["in_rows"] += int(d.get("in_rows") or 0)
            agg["out_rows"] += int(d.get("out_rows") or 0)
            agg["output_bytes"] += int(d.get("output_bytes") or 0)
            for k, v in (d.get("decisions") or {}).items():
                agg["decisions"][k] = agg["decisions"].get(k, 0) + int(v)
            output_path = path.with_name(
                path.name[: -len(".summary.json")] + ".parquet"
            )
            output_records.append(
                (
                    str(src),
                    output_path,
                    int(d.get("out_rows") or 0),
                    int(d.get("output_bytes") or 0),
                )
            )
    artifact_errors: list[dict] = []
    artifact_error_counts: collections.Counter[str] = collections.Counter()
    if verify_output_footers:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            checks = ex.map(parquet_rows, (record[1] for record in output_records))
            for (rows, err), (source, path, expected_rows, expected_bytes) in zip(
                checks, output_records
            ):
                detail = None
                if err:
                    detail = err
                elif rows != expected_rows:
                    detail = f"row_count: expected={expected_rows}, actual={rows}"
                elif expected_bytes and path.stat().st_size != expected_bytes:
                    detail = (
                        f"file_size: expected={expected_bytes}, "
                        f"actual={path.stat().st_size}"
                    )
                if detail:
                    artifact_error_counts[source] += 1
                    if len(artifact_errors) < 50:
                        artifact_errors.append(
                            {
                                "path": str(path.relative_to(processed_root)),
                                "detail": detail,
                            }
                        )
    return {
        "missing_dir": False,
        "per_source": per_source,
        "n_summaries": n,
        "n_success_summaries": n - len(errors),
        "error_summaries_sample": errors[:50],
        "n_error_summaries": len(errors),
        "error_counts_by_source": dict(sorted(error_counts_by_source.items())),
        "error_kind_counts": dict(sorted(error_kind_counts.items())),
        "n_artifact_errors": sum(artifact_error_counts.values()),
        "artifact_error_counts_by_source": dict(sorted(artifact_error_counts.items())),
        "artifact_error_sample": artifact_errors,
    }


def parquet_rows(path: Path):
    import pyarrow.parquet as pq

    try:
        return pq.ParquetFile(path).metadata.num_rows, None
    except Exception as e:
        return 0, f"unreadable: {e}"


def _decision_summary(files: list[Path], workers: int, *, group_by_stem: bool = False) -> dict:
    by_parent: collections.Counter[str] = collections.Counter()
    unreadable: list[dict[str, str]] = []
    zero_row_files = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for (rows, err), path in zip(ex.map(parquet_rows, files), files):
            if err:
                unreadable.append({"path": str(path), "error": err})
                continue
            key = path.name.removesuffix(".parquet") if group_by_stem else path.parent.name
            by_parent[key] += rows
            zero_row_files += int(rows == 0)
    return {
        "files": len(files),
        "rows": sum(by_parent.values()),
        "zero_row_files": zero_row_files,
        "rows_by_parent": dict(sorted(by_parent.items())),
        "unreadable_files": unreadable[:50],
        "n_unreadable_files": len(unreadable),
    }


def decision_totals(processed_root: Path, workers: int = 32):
    out = {}
    s2d = processed_root / "s2_decisions"
    if s2d.is_dir():
        files = sorted(s2d.rglob("*.parquet"))
        out["s2_decisions"] = _decision_summary(files, workers)
    s3d = processed_root / "s3_decisions"
    if s3d.is_dir():
        current = [p for p in sorted(s3d.glob("*.parquet")) if not p.name.endswith(".e17_backup.parquet")]
        backups = sorted(s3d.glob("*.e17_backup.parquet"))
        out["s3_decisions"] = _decision_summary(current, workers, group_by_stem=True)
        out["s3_decisions_e17_backup"] = _decision_summary(backups, workers, group_by_stem=True)
    return out


def load_scope(path: Path) -> tuple[list[str], dict[str, str]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    scope = payload.get("audit_scope") if isinstance(payload, dict) else None
    if not isinstance(scope, dict):
        raise ValueError(f"{path}: missing audit_scope mapping")
    included = scope.get("included_sources")
    excluded = scope.get("excluded_sources", {})
    if not isinstance(included, list) or not included or not all(isinstance(x, str) for x in included):
        raise ValueError(f"{path}: audit_scope.included_sources must be a non-empty string list")
    if len(included) != len(set(included)):
        raise ValueError(f"{path}: duplicate included_sources")
    if not isinstance(excluded, dict) or not all(isinstance(k, str) and isinstance(v, str) and v for k, v in excluded.items()):
        raise ValueError(f"{path}: audit_scope.excluded_sources must map source to non-empty reason")
    overlap = sorted(set(included) & set(excluded))
    if overlap:
        raise ValueError(f"{path}: sources both included and excluded: {overlap}")
    return sorted(included), dict(sorted(excluded.items()))


def reconcile(agg: dict[str, dict], included: list[str], excluded: dict[str, str]):
    stages = [s for s in STAGES if not agg[s].get("missing_dir")]
    observed = {src for s in stages for src in agg[s]["per_source"]}
    observed.update(src for s in stages for src in agg[s].get("error_counts_by_source", {}))
    sources = sorted(set(included) | set(excluded) | observed)
    rows = []
    for src in sources:
        if src in included:
            disposition = "included"
        elif src in excluded:
            disposition = "excluded"
        else:
            disposition = "undeclared"
        row = {"source": src, "disposition": disposition}
        for s in stages:
            a = agg[s]["per_source"].get(src)
            row[f"{s}_errors"] = agg[s].get("error_counts_by_source", {}).get(src, 0)
            row[f"{s}_artifact_errors"] = agg[s].get("artifact_error_counts_by_source", {}).get(src, 0)
            if a is None:
                row[f"{s}_shards"] = None
                row[f"{s}_out_rows"] = None
            else:
                row[f"{s}_shards"] = a["n_shards"]
                row[f"{s}_out_rows"] = a["out_rows"]
                row[f"{s}_in_rows"] = a["in_rows"]
                row[f"{s}_drops"] = {k: v for k, v in a["decisions"].items() if k != "keep"}
        rows.append(row)
    flags: list[str] = []
    warnings: list[str] = []
    for stage in STAGES:
        if agg[stage].get("missing_dir"):
            flags.append(f"{stage}: stage directory is missing")
    for row in rows:
        src = row["source"]
        if row["disposition"] == "undeclared":
            flags.append(f"{src}: observed in snapshot but absent from audit scope")
            continue
        if row["disposition"] == "excluded":
            errors = sum(int(row.get(f"{s}_errors") or 0) for s in STAGES)
            successes = sum(int(row.get(f"{s}_shards") or 0) for s in STAGES)
            warnings.append(
                f"{src}: excluded ({excluded[src]}); observed successes={successes}, errors={errors}"
            )
            continue
        for stage in STAGES:
            if int(row.get(f"{stage}_errors") or 0):
                flags.append(f"{src}: {stage} has {row[f'{stage}_errors']} error summaries")
            if int(row.get(f"{stage}_artifact_errors") or 0):
                flags.append(
                    f"{src}: {stage} has {row[f'{stage}_artifact_errors']} output artifact errors"
                )
            if row.get(f"{stage}_shards") is None:
                flags.append(f"{src}: missing successful {stage} summaries")
        if row.get("s1_shards") is not None and row.get("s0_shards") is not None and row["s1_shards"] != row["s0_shards"]:
            flags.append(f"{src}: shard mismatch s0={row['s0_shards']} s1={row['s1_shards']}")
        if row.get("s2_shards") is not None and row.get("s1_shards") is not None and row["s2_shards"] != row["s1_shards"]:
            flags.append(f"{src}: shard mismatch s1={row['s1_shards']} s2={row['s2_shards']}")
        if row.get("s3_shards") is not None and row.get("s2_shards") is not None and row["s3_shards"] != row["s2_shards"]:
            flags.append(f"{src}: shard mismatch s2={row['s2_shards']} s3={row['s3_shards']}")
        if row.get("decontam_shards") is not None and row.get("s3_shards") is not None and row["decontam_shards"] != row["s3_shards"]:
            flags.append(f"{src}: shard mismatch s3={row['s3_shards']} decontam={row['decontam_shards']}")
        if row.get("s0_out_rows") is not None and row.get("s1_in_rows") is not None and row["s0_out_rows"] != row["s1_in_rows"]:
            flags.append(f"{src}: row conservation s0_out={row['s0_out_rows']} vs s1_in={row['s1_in_rows']}")
        if row.get("s1_out_rows") is not None and row.get("s2_in_rows") is not None and row["s1_out_rows"] != row["s2_in_rows"]:
            flags.append(f"{src}: row conservation s1_out={row['s1_out_rows']} vs s2_in={row['s2_in_rows']}")
        if row.get("s2_out_rows") is not None and row.get("s3_in_rows") is not None and row["s2_out_rows"] != row["s3_in_rows"]:
            flags.append(f"{src}: row conservation s2_out={row['s2_out_rows']} vs s3_in={row['s3_in_rows']}")
        if row.get("s3_out_rows") is not None and row.get("decontam_in_rows") is not None and row["s3_out_rows"] != row["decontam_in_rows"]:
            flags.append(f"{src}: row conservation s3_out={row['s3_out_rows']} vs decontam_in={row['decontam_in_rows']}")
    return rows, flags, warnings


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed-root", default="../data/processed")
    ap.add_argument("--scope", required=True, help="YAML containing audit_scope.included_sources/excluded_sources")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument(
        "--skip-output-footers",
        action="store_true",
        help="diagnostic-only fast mode; a release audit must verify parquet footers",
    )
    ap.add_argument("--out", default=None, help="optional output JSON path")
    args = ap.parse_args(argv)
    root = Path(args.processed_root).resolve()
    scope_path = Path(args.scope).resolve()
    included, excluded = load_scope(scope_path)

    agg = {
        s: aggregate_stage(
            root,
            s,
            args.workers,
            verify_output_footers=not args.skip_output_footers,
        )
        for s in STAGES
    }
    table, flags, warnings = reconcile(agg, included, excluded)
    decisions = decision_totals(root, args.workers)
    for name, summary in decisions.items():
        if summary["n_unreadable_files"]:
            flags.append(f"{name}: {summary['n_unreadable_files']} unreadable parquet files")

    totals = {}
    for s in STAGES:
        if agg[s].get("missing_dir"):
            continue
        totals[s] = {
            "n_summaries": agg[s]["n_summaries"],
            "n_success_summaries": agg[s].get("n_success_summaries", 0),
            "n_error_summaries": agg[s]["n_error_summaries"],
            "n_artifact_errors": agg[s]["n_artifact_errors"],
            "in_rows": sum(a["in_rows"] for a in agg[s]["per_source"].values()),
            "out_rows": sum(a["out_rows"] for a in agg[s]["per_source"].values()),
            "output_bytes": sum(a["output_bytes"] for a in agg[s]["per_source"].values()),
        }

    report = {
        "processed_root": str(root),
        "scope_path": str(scope_path),
        "audit_scope": {"included_sources": included, "excluded_sources": excluded},
        "totals": totals,
        "per_stage_error_summaries": {s: agg[s]["n_error_summaries"] for s in STAGES},
        "per_stage_error_counts_by_source": {s: agg[s].get("error_counts_by_source", {}) for s in STAGES},
        "per_stage_error_kind_counts": {s: agg[s].get("error_kind_counts", {}) for s in STAGES},
        "error_summary_samples": {s: agg[s].get("error_summaries_sample", []) for s in STAGES},
        "per_stage_artifact_error_counts_by_source": {
            s: agg[s].get("artifact_error_counts_by_source", {}) for s in STAGES
        },
        "artifact_error_samples": {
            s: agg[s].get("artifact_error_sample", []) for s in STAGES
        },
        "output_footers_verified": not args.skip_output_footers,
        "reconciliation_table": table,
        "flags": flags,
        "warnings": warnings,
        "decision_parquet_summary": decisions,
        "status": "PASS" if not flags else "FAIL",
    }

    print(json.dumps(totals, indent=2, sort_keys=True))
    print("\n=== FLAGS (must be reviewed) ===")
    for f in flags:
        print(" !", f)
    if not flags:
        print("  (none)")
    print("\n=== WARNINGS (explicitly excluded lineage) ===")
    for warning in warnings:
        print(" ~", warning)
    if not warnings:
        print("  (none)")
    print("\n=== per-source table ===")
    hdr = ["source", "disposition"] + [f"{s}_shards" for s in STAGES] + [f"{s}_errors" for s in STAGES] + [f"{s}_out_rows" for s in STAGES]
    print("\t".join(hdr))
    for row in table:
        print("\t".join(str(row.get(h)) for h in hdr))
    print("\n=== decision parquet summary ===")
    print(json.dumps(decisions, indent=2, sort_keys=True))

    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, destination)
        print(f"\nreport written: {args.out}", file=sys.stderr)
    return 0 if not flags else 1


if __name__ == "__main__":
    sys.exit(main())
