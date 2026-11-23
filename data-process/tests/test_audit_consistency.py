import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_consistency.py"
SPEC = importlib.util.spec_from_file_location("audit_consistency", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


def _stage(source_rows, *, errors=None):
    per_source = {}
    for source, (shards, in_rows, out_rows) in source_rows.items():
        per_source[source] = {
            "n_shards": shards,
            "in_rows": in_rows,
            "out_rows": out_rows,
            "output_bytes": 1,
            "decisions": {},
        }
    errors = errors or {}
    return {
        "missing_dir": False,
        "per_source": per_source,
        "error_counts_by_source": errors,
    }


def test_reconcile_blocks_in_scope_errors_but_retains_excluded_failures():
    rows = {"active": (2, 10, 10)}
    agg = {stage: _stage(rows) for stage in audit.STAGES}
    agg["s1"]["error_counts_by_source"] = {"active": 1, "retired": 7}

    table, flags, warnings = audit.reconcile(
        agg, ["active"], {"retired": "explicitly retired"}
    )

    assert any("active: s1 has 1 error summaries" in flag for flag in flags)
    assert any("retired: excluded" in warning and "errors=7" in warning for warning in warnings)
    assert {row["source"]: row["disposition"] for row in table} == {
        "active": "included",
        "retired": "excluded",
    }


def test_reconcile_enforces_all_stage_coverage_and_row_conservation():
    agg = {
        "s0": _stage({"active": (2, 10, 10)}),
        "s1": _stage({"active": (2, 10, 9)}),
        "s2": _stage({"active": (2, 8, 8)}),
        "s3": _stage({"active": (2, 8, 8)}),
        "decontam": _stage({}),
    }

    _, flags, _ = audit.reconcile(agg, ["active"], {})

    assert "active: missing successful decontam summaries" in flags
    assert "active: row conservation s1_out=9 vs s2_in=8" in flags


def test_reconcile_passes_a_closed_lineage():
    agg = {
        "s0": _stage({"active": (2, 10, 10)}),
        "s1": _stage({"active": (2, 10, 9)}),
        "s2": _stage({"active": (2, 9, 8)}),
        "s3": _stage({"active": (2, 8, 7)}),
        "decontam": _stage({"active": (2, 7, 7)}),
    }

    _, flags, warnings = audit.reconcile(agg, ["active"], {})

    assert flags == []
    assert warnings == []
