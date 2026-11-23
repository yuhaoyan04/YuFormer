import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_raw_source_completeness.py"
SPEC = importlib.util.spec_from_file_location("audit_raw_source_completeness", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_inventory_comparison_reports_missing_and_truncated_files():
    result = MODULE.compare_inventories(
        [("a", 10), ("b", 20)],
        [("a", 7)],
    )
    assert result["status"] == "FAIL"
    assert result["missing_files"] == ["b"]
    assert result["size_mismatches"] == [
        {"path": "a", "expected_bytes": 10, "observed_bytes": 7}
    ]


def test_unlimited_selection_is_complete_and_sorted():
    config = {
        "name": "source",
        "max_size_gb": 0,
        "include_patterns": ["data/*.jsonl"],
        "exclude_patterns": ["data/debug-*"],
    }
    selected = MODULE._selected_inventory(
        config,
        [("README.md", 1), ("data/z.jsonl", 3), ("data/debug-a.jsonl", 2), ("data/a.jsonl", 4)],
    )
    assert selected == [("data/a.jsonl", 4), ("data/z.jsonl", 3)]
