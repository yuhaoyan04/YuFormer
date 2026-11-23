#!/usr/bin/env python3
"""C2 audit: full-snapshot quality-score distributions by source and language.

The audit scans only the score and language columns. Quantiles use a deterministic
fixed-width histogram, so memory is independent of corpus size and the absolute
quantile error is bounded by ``1 / bins`` for scores in [0, 1]. Counts, nulls,
out-of-range values, and endpoint masses are exact.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds
import yaml


QUANTILES = (0.01, 0.25, 0.50, 0.75, 0.99)


def _new_group(bins: int) -> dict[str, Any]:
    return {
        "hist": np.zeros(bins + 1, dtype=np.int64),
        "rank_hist": np.zeros(bins + 1, dtype=np.int64),
        "rows": 0,
        "null_or_nonfinite": 0,
        "out_of_range": 0,
        "zeros": 0,
        "ones": 0,
        "rank_rows": 0,
        "rank_null_or_nonfinite": 0,
        "rank_out_of_range": 0,
    }


def _add(values: np.ndarray, group: dict[str, Any], bins: int, *, rank: bool = False) -> None:
    prefix = "rank_" if rank else ""
    group[f"{prefix}rows"] += int(len(values))
    try:
        array = np.asarray(values)
        if array.dtype.kind not in "fiu":
            array = array.astype(np.float64)
    except (TypeError, ValueError):
        group[f"{prefix}null_or_nonfinite"] += int(len(values))
        return
    finite = np.isfinite(array)
    group[f"{prefix}null_or_nonfinite"] += int((~finite).sum())
    valid = array[finite]
    in_range = (valid >= 0.0) & (valid <= 1.0)
    group[f"{prefix}out_of_range"] += int((~in_range).sum())
    valid = valid[in_range]
    if not rank:
        group["zeros"] += int((valid == 0.0).sum())
        group["ones"] += int((valid == 1.0).sum())
    if len(valid):
        indices = np.minimum((valid * bins).astype(np.int64), bins)
        group[f"{prefix}hist"] += np.bincount(indices, minlength=bins + 1)


def _quantiles(hist: np.ndarray, bins: int) -> list[float | None]:
    count = int(hist.sum())
    if not count:
        return [None for _ in QUANTILES]
    cumulative = np.cumsum(hist)
    values = []
    for q in QUANTILES:
        target = max(1, math.ceil(q * count))
        index = int(np.searchsorted(cumulative, target, side="left"))
        values.append(round(index / bins, 6))
    return values


def _finish(group: dict[str, Any], bins: int, *, min_group_rows: int) -> dict[str, Any]:
    valid_rows = int(group["hist"].sum())
    rank_valid_rows = int(group["rank_hist"].sum())
    quantiles = _quantiles(group["hist"], bins)
    rank_quantiles = _quantiles(group["rank_hist"], bins)
    evaluated = valid_rows >= min_group_rows
    endpoint_ratio = (group["zeros"] + group["ones"]) / valid_rows if valid_rows else None
    spread = (
        quantiles[-1] - quantiles[0]
        if quantiles[0] is not None and quantiles[-1] is not None
        else None
    )
    healthy = (
        evaluated
        and group["null_or_nonfinite"] == 0
        and group["out_of_range"] == 0
        and endpoint_ratio is not None
        and endpoint_ratio <= 0.05
        and spread is not None
        and spread >= 0.05
    )
    return {
        "rows": int(group["rows"]),
        "valid_score_rows": valid_rows,
        "null_or_nonfinite": int(group["null_or_nonfinite"]),
        "out_of_range": int(group["out_of_range"]),
        "zero_rows": int(group["zeros"]),
        "one_rows": int(group["ones"]),
        "endpoint_ratio": endpoint_ratio,
        "percentiles_p1_p25_p50_p75_p99": quantiles,
        "rank_valid_rows": rank_valid_rows,
        "rank_null_or_nonfinite": int(group["rank_null_or_nonfinite"]),
        "rank_out_of_range": int(group["rank_out_of_range"]),
        "domain_rank_percentiles_p1_p25_p50_p75_p99": rank_quantiles,
        "evaluated": evaluated,
        "healthy": healthy,
    }


def _language(value: Any) -> str:
    if value is None:
        return "und"
    normalized = str(value).strip().lower().replace("_", "-")
    return normalized or "und"


def _merge_group(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in (
        "rows",
        "null_or_nonfinite",
        "out_of_range",
        "zeros",
        "ones",
        "rank_rows",
        "rank_null_or_nonfinite",
        "rank_out_of_range",
    ):
        target[key] += source[key]
    target["hist"] += source["hist"]
    target["rank_hist"] += source["rank_hist"]


def audit_score_distributions(
    s4_root: Path,
    scope_path: Path,
    *,
    score_field: str = "hybrid_v2",
    rank_field: str = "domain_quantile",
    bins: int = 10_000,
    batch_size: int = 262_144,
    min_group_rows: int = 1_000,
    workers: int = 4,
    files_per_task: int = 64,
    progress_every: int = 250,
) -> dict[str, Any]:
    scope = yaml.safe_load(scope_path.read_text(encoding="utf-8")) or {}
    sources = list((scope.get("audit_scope") or {}).get("included_sources") or [])
    per_source = {source: _new_group(bins) for source in sources}
    per_language: dict[str, dict[str, Any]] = {}
    file_errors: list[dict[str, Any]] = []
    files_scanned = 0

    tasks: list[tuple[str, list[Path]]] = []
    expected_files = 0
    for source in sources:
        paths = sorted((s4_root / source).rglob("*.parquet"))
        if not paths:
            file_errors.append({"source": source, "path": "", "error": "no parquet files"})
            continue
        expected_files += len(paths)
        tasks.extend(
            (source, paths[start : start + files_per_task])
            for start in range(0, len(paths), files_per_task)
        )

    def scan_file(task: tuple[str, list[Path]]):
        source, paths = task
        source_group = _new_group(bins)
        language_groups: dict[str, dict[str, Any]] = {}
        try:
            scanner = ds.dataset(
                [str(path) for path in paths], format="parquet"
            ).scanner(
                columns=["lang", score_field, rank_field],
                batch_size=batch_size,
                use_threads=True,
                batch_readahead=16,
                fragment_readahead=16,
            )
            # Dataset scanners expose the original tiny row-group batches.
            # Coalesce projected columns once per bounded file block so NumPy
            # histograms run on large contiguous arrays instead of tens of
            # thousands of ~2K-row fragments.
            table = scanner.to_table().combine_chunks()
            for batch in table.to_batches(max_chunksize=batch_size):
                languages = batch.column(0)
                scores = np.asarray(batch.column(1).to_numpy(zero_copy_only=False))
                ranks = np.asarray(batch.column(2).to_numpy(zero_copy_only=False))
                # Language cardinality per shard is tiny. Group with Arrow and
                # normalize only the unique labels; converting every row to a
                # Python string makes a billion-row audit GIL-bound.
                for raw_language in pc.unique(languages).to_pylist():
                    arrow_mask = (
                        pc.is_null(languages)
                        if raw_language is None
                        else pc.equal(languages, raw_language)
                    )
                    mask = np.asarray(
                        pc.fill_null(arrow_mask, False).to_numpy(zero_copy_only=False),
                        dtype=bool,
                    )
                    language = _language(raw_language)
                    group = language_groups.setdefault(language, _new_group(bins))
                    _add(scores[mask], group, bins)
                    _add(ranks[mask], group, bins, rank=True)
            # Language groups partition every row, so their sum is exactly the
            # source distribution. Deriving it avoids histogramming each score
            # twice on predominantly monolingual sources.
            for group in language_groups.values():
                _merge_group(source_group, group)
            return source, paths, source_group, language_groups, None
        except Exception as exc:
            return source, paths, None, None, f"{type(exc).__name__}: {exc}"

    if workers < 1:
        raise ValueError("workers must be >= 1")
    if files_per_task < 1:
        raise ValueError("files_per_task must be >= 1")
    started = time.monotonic()
    task_iter = iter(tasks)
    processed_files = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="score-audit") as pool:
        pending = set()
        for _ in range(min(len(tasks), workers * 2)):
            pending.add(pool.submit(scan_file, next(task_iter)))
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            completed_files = 0
            for future in completed:
                source, paths, source_group, language_groups, error = future.result()
                processed_files += len(paths)
                completed_files += len(paths)
                if error:
                    file_errors.append(
                        {
                            "source": source,
                            "path": str(paths[0]),
                            "files_in_failed_task": len(paths),
                            "error": error,
                        }
                    )
                else:
                    _merge_group(per_source[source], source_group)
                    for language, group in language_groups.items():
                        _merge_group(per_language.setdefault(language, _new_group(bins)), group)
                    files_scanned += len(paths)
                try:
                    pending.add(pool.submit(scan_file, next(task_iter)))
                except StopIteration:
                    pass
            if progress_every and (
                processed_files % progress_every < completed_files
                or processed_files == expected_files
            ):
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f"score audit: {processed_files}/{expected_files} files "
                    f"({processed_files / elapsed:.1f} files/s), errors={len(file_errors)}",
                    file=sys.stderr,
                    flush=True,
                )

    source_report = {
        source: _finish(group, bins, min_group_rows=min_group_rows)
        for source, group in sorted(per_source.items())
    }
    language_report = {
        language: _finish(group, bins, min_group_rows=min_group_rows)
        for language, group in sorted(per_language.items())
    }
    unhealthy_sources = [
        source for source, record in source_report.items() if not record["healthy"]
    ]
    evaluated_languages = {
        language: record for language, record in language_report.items() if record["evaluated"]
    }
    unhealthy_languages = [
        language for language, record in evaluated_languages.items() if not record["healthy"]
    ]

    def combined_rank_median(prefix: str) -> tuple[int, float | None]:
        hist = np.zeros(bins + 1, dtype=np.int64)
        rows = 0
        for language, group in per_language.items():
            if language == prefix or language.startswith(prefix + "-"):
                hist += group["rank_hist"]
                rows += int(group["rank_hist"].sum())
        return rows, _quantiles(hist, bins)[2]

    zh_rows, zh_rank_median = combined_rank_median("zh")
    en_rows, en_rank_median = combined_rank_median("en")
    chinese_bias_passed = (
        zh_rows >= min_group_rows
        and en_rows >= min_group_rows
        and zh_rank_median is not None
        and en_rank_median is not None
        and zh_rank_median >= en_rank_median - 0.10
    )
    flags = []
    if file_errors:
        failed_file_count = sum(
            int(error.get("files_in_failed_task", 1)) for error in file_errors
        )
        flags.append(f"{failed_file_count} parquet files/sources could not be audited")
    if unhealthy_sources:
        flags.append(f"unhealthy source distributions: {unhealthy_sources}")
    if unhealthy_languages:
        flags.append(f"unhealthy evaluated language distributions: {unhealthy_languages}")
    if not chinese_bias_passed:
        flags.append(
            "Chinese domain-normalized median is missing or more than 0.10 below English"
        )

    return {
        "schema_version": 1,
        "gate": "C2",
        "status": "PASS" if not flags else "FAIL",
        "score_field": score_field,
        "rank_field": rank_field,
        "method": {
            "scan": "all parquet rows in included sources; projected columns only",
            "files_per_scan_task": files_per_task,
            "histogram_bins": bins,
            "quantile_absolute_resolution": 1 / bins,
            "min_group_rows": min_group_rows,
            "health_thresholds": {
                "endpoint_mass_lte": 0.05,
                "p99_minus_p1_gte": 0.05,
                "chinese_domain_rank_median_vs_english_gte": -0.10,
            },
        },
        "sources_expected": sources,
        "files_scanned": files_scanned,
        "file_errors": file_errors,
        "per_source": source_report,
        "per_language": language_report,
        "language_bias": {
            "zh_rows": zh_rows,
            "zh_domain_rank_median": zh_rank_median,
            "en_rows": en_rows,
            "en_domain_rank_median": en_rank_median,
            "passed": chinese_bias_passed,
        },
        "flags": flags,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s4-root", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--score-field", default="hybrid_v2")
    parser.add_argument("--rank-field", default="domain_quantile")
    parser.add_argument("--bins", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=262_144)
    parser.add_argument("--min-group-rows", type=int, default=1_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--files-per-task", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit_score_distributions(
        args.s4_root.resolve(),
        args.scope.resolve(),
        score_field=args.score_field,
        rank_field=args.rank_field,
        bins=args.bins,
        batch_size=args.batch_size,
        min_group_rows=args.min_group_rows,
        workers=args.workers,
        files_per_task=args.files_per_task,
        progress_every=args.progress_every,
    )
    destination = args.out.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    print(
        json.dumps(
            {
                "status": result["status"],
                "files_scanned": result["files_scanned"],
                "sources": len(result["per_source"]),
                "languages": len(result["per_language"]),
                "flags": result["flags"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
