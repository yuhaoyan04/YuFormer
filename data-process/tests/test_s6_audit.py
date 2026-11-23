import numpy as np

from yuformer_data.s6_audit import (
    audit_dedup_ledger,
    audit_length_distribution,
    audit_quantile_coverage,
    audit_s6_metadata,
)


def test_length_distribution_uses_token_ledger_and_long_context_supply():
    s5 = {
        "totals": {"out_rows": 100, "total_tokens": 2_200_000_000},
        "buckets": {"B4K": 98, "B16K": 1, "B64K": 1},
        "bucket_tokens": {"B4K": 1_100_000_000, "B16K": 500_000_000, "B64K": 600_000_000},
    }
    result = audit_length_distribution(
        s5, primary_context=4096, long_context_min_tokens=1_000_000_000
    )
    assert result["status"] == "PASS"
    assert result["doc_ratios"]["B4K"] == 0.98
    assert result["long_context_tokens"] == 1_100_000_000

    s5["bucket_tokens"]["B64K"] -= 1
    assert audit_length_distribution(
        s5, primary_context=4096, long_context_min_tokens=1_000_000_000
    )["status"] == "FAIL"


def test_s6_metadata_closes_quota_and_reports_unfinished_identity_checks():
    summary = {
        "mix_name": "mix",
        "config_echo": "abc",
        "total_selected_tokens": 91,
        "total_quota_shortfall_tokens": 9,
        "max_total_shortfall_tokens": 10,
        "per_domain": {
            "web": {
                "quota_tokens": 100,
                "quota_shortfall_tokens": 9,
                "selected_tokens": 91,
                "selected_tokens_by_source": {"a": 41, "b": 50},
            }
        },
        "curriculum": {"stage1": {"tokens": 20}, "stage2": {"tokens": 71}},
    }
    manifest = {
        "config_echo": "abc",
        "order_tables": ["mix_stage1.parquet", "mix_stage2.parquet"],
        "curriculum": {"enabled": True, "method": "upstream_score_legacy"},
    }
    val = {"per_domain": {"web": {"quota_tokens": 10, "val_tokens": 11}}}

    result = audit_s6_metadata(summary, manifest, val, expected_config_echo="abc")

    assert result["status"] == "PASS"
    assert result["gates"]["D1"]["status"] == "PASS"
    assert result["gates"]["D3"]["status"] == "PENDING"
    assert result["gates"]["D4"]["status"] == "PENDING"
    assert result["gates"]["F2"]["status"] == "BLOCKED"


def test_s6_metadata_fails_mismatched_accounting():
    summary = {
        "mix_name": "mix",
        "config_echo": "wrong",
        "total_selected_tokens": 91,
        "total_quota_shortfall_tokens": 9,
        "max_total_shortfall_tokens": 10,
        "per_domain": {
            "web": {
                "quota_tokens": 100,
                "quota_shortfall_tokens": 9,
                "selected_tokens": 90,
                "selected_tokens_by_source": {"a": 89},
            }
        },
        "curriculum": {"stage1": {"tokens": 20}, "stage2": {"tokens": 70}},
    }
    manifest = {"config_echo": "wrong", "order_tables": []}
    val = {"per_domain": {}}

    result = audit_s6_metadata(summary, manifest, val, expected_config_echo="abc")

    assert result["status"] == "FAIL"
    assert result["gates"]["D1"]["status"] == "FAIL"


def test_random_baseline_declares_only_stage2_and_passes_d2():
    summary = {
        "mix_name": "mix", "config_echo": "abc", "total_selected_tokens": 91,
        "total_quota_shortfall_tokens": 9, "max_total_shortfall_tokens": 10,
        "per_domain": {"web": {"quota_tokens": 100, "quota_shortfall_tokens": 9,
                                "selected_tokens": 91, "selected_tokens_by_source": {"a": 91}}},
        "curriculum": {"stage2": {"tokens": 91}},
    }
    manifest = {
        "config_echo": "abc", "order_tables": ["mix_stage2.parquet"],
        "curriculum": {"enabled": False, "method": "random", "stage1_fraction": 0.0},
        "upstream_snapshot_sha256": "sha", "upstream_snapshot": {},
    }
    val = {"per_domain": {"web": {"quota_tokens": 10, "val_tokens": 10}}}
    result = audit_s6_metadata(summary, manifest, val, expected_config_echo="abc")
    assert result["status"] == "PASS"
    assert result["gates"]["D2"]["status"] == "PASS"


def test_dedup_ledger_matches_decision_parquet_counts():
    s5 = {
        "per_domain": {
            "web": {
                "in_rows": 100,
                "out_rows": 82,
                "decisions.drop_s2_exact_dup": 10,
                "decisions.drop_s3_near_dup": 3,
                "decisions.drop_decontam": 5,
            }
        }
    }
    consistency = {
        "decision_parquet_summary": {
            "s2_decisions": {"rows_by_parent": {"web": 10}},
            "s3_decisions": {"rows_by_parent": {"web": 3}},
        }
    }

    result = audit_dedup_ledger(s5, consistency)

    assert result["status"] == "PASS"
    s5["per_domain"]["web"]["out_rows"] = 83
    assert audit_dedup_ledger(s5, consistency)["status"] == "FAIL"


def test_dedup_ledger_accepts_ndv2_raw_lookup_with_exact_overlap():
    s5 = {
        "input_contract": {"lookup_fingerprint": {"web": [80, 1, 5, 2]}},
        "per_domain": {
            "web": {
                "in_rows": 100,
                "out_rows": 83,
                "decisions.drop_s2_exact_dup": 10,
                "decisions.drop_s3_near_dup": 4,
                "decisions.drop_decontam": 3,
            }
        },
    }
    consistency = {
        "decision_parquet_summary": {
            "s2_decisions": {"rows_by_parent": {"web": 10}},
            "s3_decisions": {"rows_by_parent": {"web": 0}},
        }
    }
    ndv2 = {"per_domain": {"web": {"dropped": 5}}}
    decisions = {"web": {"dropped": 5}}
    lookup_rows = {"web": {"ids_rows": 5, "codes_rows": 5}}
    result = audit_dedup_ledger(s5, consistency, ndv2, decisions, lookup_rows)
    assert result["status"] == "PASS"
    assert result["per_domain"]["web"]["ndv2_lookup_rows"] == 5

    decisions["web"]["dropped"] = 6
    assert audit_dedup_ledger(
        s5, consistency, ndv2, decisions, lookup_rows
    )["status"] == "FAIL"


def test_quantile_coverage_uses_mmap_shape_and_s5_rows(tmp_path):
    values = np.array([0.1, 0.2, 0.3, 0.4], dtype="<f4")
    np.save(tmp_path / "web.npy", values)
    s5 = {"per_domain": {"web": {"in_rows": 4}}}
    meta = {
        "score_field": "hybrid_v2",
        "domains": {"web": {"n": 4, "npy": "web.npy"}},
    }

    result = audit_quantile_coverage(s5, meta, tmp_path)

    assert result["status"] == "PASS"
    meta["domains"]["web"]["n"] = 3
    assert audit_quantile_coverage(s5, meta, tmp_path)["status"] == "FAIL"
