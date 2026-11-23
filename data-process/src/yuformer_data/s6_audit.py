"""Independent metadata checks for an S6 materialized mixture."""
from __future__ import annotations

from typing import Any

import numpy as np


def _result(passed: bool, detail: str, actual: Any = None) -> dict[str, Any]:
    result = {"passed": bool(passed), "detail": detail}
    if actual is not None:
        result["actual"] = actual
    return result


def audit_dedup_ledger(
    stage_s5: dict[str, Any],
    consistency_audit: dict[str, Any],
    ndv2_global: dict[str, Any] | None = None,
    ndv2_decisions: dict[str, dict[str, Any]] | None = None,
    ndv2_lookup_rows: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    per_domain = stage_s5.get("per_domain") or {}
    decision_summary = consistency_audit.get("decision_parquet_summary") or {}
    s2_rows = (decision_summary.get("s2_decisions") or {}).get("rows_by_parent") or {}
    s3_rows = (decision_summary.get("s3_decisions") or {}).get("rows_by_parent") or {}
    details: dict[str, Any] = {}
    passed = True
    ndv2_domains = (ndv2_global or {}).get("per_domain") or {}
    ndv2_decisions = ndv2_decisions or {}
    ndv2_lookup_rows = ndv2_lookup_rows or {}
    lookup_fingerprints = (
        (stage_s5.get("input_contract") or {}).get("lookup_fingerprint") or {}
    )
    for domain, row in sorted(per_domain.items()):
        drops = {
            key.removeprefix("decisions."): int(value)
            for key, value in row.items()
            if key.startswith("decisions.drop_")
        }
        expected_out = int(row.get("in_rows") or 0) - sum(drops.values())
        actual_out = int(row.get("out_rows") or 0)
        exact_drop = drops.get("drop_s2_exact_dup", 0)
        near_drop = drops.get("drop_s3_near_dup", 0)
        raw_ndv2_drop = None
        lookup_rows = None
        decision_rows = None
        if domain in ndv2_domains:
            raw_ndv2_drop = int((ndv2_domains.get(domain) or {}).get("dropped") or 0)
            lookup_record = ndv2_lookup_rows.get(domain) or {}
            fingerprint = lookup_fingerprints.get(domain) or []
            lookup_rows = int(
                lookup_record.get("ids_rows")
                if "ids_rows" in lookup_record
                else (int(fingerprint[0]) // 16 if fingerprint else -1)
            )
            lookup_code_rows = int(lookup_record.get("codes_rows", lookup_rows))
            decision_rows = int((ndv2_decisions.get(domain) or {}).get("dropped") or -1)
            # S5 applies exact duplicates before near duplicates, so the effective
            # near-drop ledger may be smaller than the raw lookup by at most the
            # exact-drop overlap.  Code currently has no exact drops and must match.
            near_identity_ok = (
                raw_ndv2_drop == lookup_rows == lookup_code_rows == decision_rows
                and 0 <= raw_ndv2_drop - near_drop <= exact_drop
            )
        else:
            near_identity_ok = int(s3_rows.get(domain, 0)) == near_drop
        domain_passed = (
            expected_out == actual_out
            and int(s2_rows.get(domain, 0)) == exact_drop
            and near_identity_ok
        )
        passed &= domain_passed
        details[domain] = {
            "passed": domain_passed,
            "in_rows": int(row.get("in_rows") or 0),
            "out_rows": actual_out,
            "drops": drops,
            "expected_out_rows": expected_out,
            "s2_decision_rows": int(s2_rows.get(domain, 0)),
            "s3_decision_rows": int(s3_rows.get(domain, 0)),
            "ndv2_raw_drop_rows": raw_ndv2_drop,
            "ndv2_lookup_rows": lookup_rows,
            "ndv2_lookup_code_rows": (
                lookup_code_rows if domain in ndv2_domains else None
            ),
            "ndv2_decision_rows": decision_rows,
        }
    expected_domains = set(per_domain)
    passed &= set(s2_rows) == expected_domains and set(s3_rows) == expected_domains
    return {
        "status": "PASS" if passed else "FAIL",
        "detail": "for every domain: S5 row ledger closes; legacy decision rows match effective drops; ndv2 lookup/global/decision raw drops agree and exact-overlap explains the effective S5 near-drop count",
        "per_domain": details,
    }


def audit_quantile_coverage(
    stage_s5: dict[str, Any], quantile_meta: dict[str, Any], tables_root: Any
) -> dict[str, Any]:
    domains = quantile_meta.get("domains") or {}
    expected_domains = set(stage_s5.get("per_domain") or {})
    details: dict[str, Any] = {}
    passed = set(domains) == expected_domains
    for domain, record in sorted(domains.items()):
        path = tables_root / str(record.get("npy"))
        try:
            array = np.load(path, mmap_mode="r")
            expected_rows = int(record.get("n") or 0)
            indices = [int((len(array) - 1) * q) for q in (0.01, 0.25, 0.50, 0.75, 0.99)]
            percentiles = [float(array[index]) for index in indices] if len(array) else []
            domain_passed = (
                array.ndim == 1
                and array.dtype == np.dtype("<f4")
                and len(array) == expected_rows
                and expected_rows
                == int((stage_s5.get("per_domain") or {}).get(domain, {}).get("in_rows") or 0)
                and percentiles == sorted(percentiles)
            )
            details[domain] = {
                "passed": domain_passed,
                "rows": len(array),
                "dtype": str(array.dtype),
                "percentiles_p1_p25_p50_p75_p99": percentiles,
            }
        except Exception as exc:
            domain_passed = False
            details[domain] = {"passed": False, "error": str(exc)}
        passed &= domain_passed
    return {
        "status": "PASS" if passed else "FAIL",
        "score_field": quantile_meta.get("score_field"),
        "detail": "selected score table covers every S5 domain and mmap row counts match",
        "per_domain": details,
    }


def audit_length_distribution(
    stage_s5: dict[str, Any], *, primary_context: int, long_context_min_tokens: int
) -> dict[str, Any]:
    """Validate lossless document/token bucket accounting and long-context supply.

    A high short-document ratio is expected for web corpora and is not itself a
    quality failure. The release-relevant checks are exact accounting plus enough
    source tokens above the primary context for a later long-context phase.
    """
    docs = {str(k): int(v) for k, v in (stage_s5.get("buckets") or {}).items()}
    tokens = {
        str(k): int(v) for k, v in (stage_s5.get("bucket_tokens") or {}).items()
    }
    total_docs = int((stage_s5.get("totals") or {}).get("out_rows") or 0)
    total_tokens = int((stage_s5.get("totals") or {}).get("total_tokens") or 0)
    expected_primary = f"B{int(primary_context) // 1024}K"
    docs_closed = bool(docs) and sum(docs.values()) == total_docs
    tokens_closed = bool(tokens) and sum(tokens.values()) == total_tokens
    long_tokens = sum(value for key, value in tokens.items() if key != expected_primary)
    passed = (
        docs_closed
        and tokens_closed
        and expected_primary in docs
        and long_tokens >= int(long_context_min_tokens)
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "detail": (
            "bucket document/token ledgers close exactly and the >primary-context "
            "pool meets the declared long-context research reserve"
        ),
        "primary_context": int(primary_context),
        "primary_bucket": expected_primary,
        "long_context_min_tokens": int(long_context_min_tokens),
        "long_context_tokens": long_tokens,
        "docs_closed": docs_closed,
        "tokens_closed": tokens_closed,
        "doc_counts": docs,
        "token_counts": tokens,
        "doc_ratios": {
            key: (value / total_docs if total_docs else 0.0) for key, value in docs.items()
        },
        "token_ratios": {
            key: (value / total_tokens if total_tokens else 0.0)
            for key, value in tokens.items()
        },
    }


def audit_s6_metadata(
    summary: dict[str, Any],
    manifest: dict[str, Any],
    val_summary: dict[str, Any],
    *,
    expected_config_echo: str,
) -> dict[str, Any]:
    per_domain = summary.get("per_domain") or {}
    val_domains = val_summary.get("per_domain") or {}
    checks: dict[str, dict[str, Any]] = {}

    selected_sum = sum(int(row.get("selected_tokens") or 0) for row in per_domain.values())
    total_selected = int(summary.get("total_selected_tokens") or 0)
    checks["selected_token_sum"] = _result(
        selected_sum == total_selected,
        "sum(per_domain.selected_tokens) == total_selected_tokens",
        {"sum": selected_sum, "declared": total_selected},
    )
    quota_rows_closed = all(
        int(row.get("selected_tokens") or 0) + int(row.get("quota_shortfall_tokens") or 0)
        == int(row.get("quota_tokens") or 0)
        for row in per_domain.values()
    )
    checks["quota_rows_closed"] = _result(
        quota_rows_closed,
        "selected_tokens + quota_shortfall_tokens == quota_tokens for every domain",
    )
    total_shortfall = sum(
        int(row.get("quota_shortfall_tokens") or 0) for row in per_domain.values()
    )
    declared_shortfall = int(summary.get("total_quota_shortfall_tokens") or 0)
    max_shortfall = int(summary.get("max_total_shortfall_tokens") or 0)
    shortfall_passed = (
        total_shortfall == declared_shortfall and total_shortfall <= max_shortfall
    )
    checks["total_quota_shortfall"] = _result(
        shortfall_passed,
        "aggregate quota shortfall is declared and within the configured document-tail tolerance",
        {
            "sum": total_shortfall,
            "declared": declared_shortfall,
            "maximum": max_shortfall,
        },
    )
    source_rows_closed = all(
        sum(int(value) for value in (row.get("selected_tokens_by_source") or {}).values())
        == int(row.get("selected_tokens") or 0)
        for row in per_domain.values()
    )
    checks["source_token_sum"] = _result(
        source_rows_closed,
        "sum(selected_tokens_by_source) == selected_tokens for every domain",
    )
    web = per_domain.get("web") or {}
    web_quota = int(web.get("quota_tokens") or 0)
    web_shortfall_ratio = (
        int(web.get("quota_shortfall_tokens") or 0) / web_quota if web_quota else 1.0
    )
    checks["web_shortfall_lt_10pct"] = _result(
        web_shortfall_ratio < 0.10,
        "web quota shortfall is below 10%",
        web_shortfall_ratio,
    )

    curriculum = summary.get("curriculum") or {}
    curriculum_tokens = sum(
        int(row.get("tokens") or 0)
        for row in curriculum.values()
        if isinstance(row, dict)
    )
    checks["curriculum_token_sum"] = _result(
        curriculum_tokens == total_selected,
        "sum(curriculum stage tokens) == total_selected_tokens",
        {"sum": curriculum_tokens, "declared": total_selected},
    )

    domain_sets_match = set(per_domain) == set(val_domains)
    checks["val_domain_coverage"] = _result(
        domain_sets_match,
        "validation summary covers exactly the selected domains",
        {"mix": sorted(per_domain), "validation": sorted(val_domains)},
    )
    val_quotas_met = domain_sets_match and all(
        int(val_domains[domain].get("val_tokens") or 0)
        >= int(val_domains[domain].get("quota_tokens") or 0)
        for domain in per_domain
    )
    checks["val_token_quotas"] = _result(
        val_quotas_met,
        "every domain reaches its declared validation token quota",
    )

    observed_echoes = {
        "summary": summary.get("config_echo"),
        "manifest": manifest.get("config_echo"),
        "expected": expected_config_echo,
    }
    checks["config_echo"] = _result(
        len(set(observed_echoes.values())) == 1,
        "S6 summary and manifest config echoes match the current normalized config",
        observed_echoes,
    )
    declared_curriculum = manifest.get("curriculum") or {}
    expected_order_tables = {f"{summary.get('mix_name')}_stage2.parquet"}
    if declared_curriculum.get("enabled"):
        expected_order_tables.add(f"{summary.get('mix_name')}_stage1.parquet")
    checks["order_table_declarations"] = _result(
        set(manifest.get("order_tables") or []) == expected_order_tables,
        "manifest declares exactly the order tables required by the curriculum mode",
        manifest.get("order_tables"),
    )
    checks["upstream_snapshot_bound"] = _result(
        bool(manifest.get("upstream_snapshot_sha256"))
        and isinstance(manifest.get("upstream_snapshot"), dict),
        "manifest binds the content-addressed upstream snapshot",
    )

    buckets = summary.get("length_buckets") or {}
    if not buckets:
        buckets = {}
    total_bucket_docs = sum(int(value) for value in buckets.values())
    bucket_ratios = {
        key: (int(value) / total_bucket_docs if total_bucket_docs else 0.0)
        for key, value in buckets.items()
    }
    checks["length_bucket_rows"] = _result(
        bool(buckets) and total_bucket_docs > 0,
        "length buckets are present and non-empty",
        {"docs": total_bucket_docs, "ratios": bucket_ratios},
    )

    d1_keys = (
        "selected_token_sum",
        "quota_rows_closed",
        "total_quota_shortfall",
        "source_token_sum",
        "web_shortfall_lt_10pct",
    )
    d1_pass = all(checks[key]["passed"] for key in d1_keys)
    metadata_integrity_keys = d1_keys + (
        "curriculum_token_sum",
        "val_domain_coverage",
        "val_token_quotas",
        "config_echo",
        "order_table_declarations",
    )
    return {
        "schema_version": 1,
        "status": "PASS"
        if all(checks[key]["passed"] for key in metadata_integrity_keys)
        else "FAIL",
        "checks": checks,
        "gates": {
            "D1": {"status": "PASS" if d1_pass else "FAIL"},
            "D2": {
                "status": (
                    "PASS"
                    if not declared_curriculum.get("enabled")
                    and declared_curriculum.get("method") == "random"
                    and set(manifest.get("order_tables") or []) == expected_order_tables
                    else "REVIEW"
                ),
                "reason": "production baseline uses one deterministic globally shuffled order table",
            },
            "D3": {
                "status": "PENDING",
                "reason": "token quotas checked; train/validation doc_id disjointness requires a full ID audit",
            },
            "D4": {
                "status": "PENDING",
                "reason": "config echo checked; global doc_id uniqueness requires a full ID audit",
            },
            "E3": {
                "status": "PENDING",
                "reason": "length buckets must be read from stage_s5_global_summary",
            },
            "F2": {
                "status": "PASS"
                if checks["upstream_snapshot_bound"]["passed"]
                else "BLOCKED"
            },
        },
    }
