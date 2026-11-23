import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from yuformer_rl.diagnostics import audit_rollouts


def _row(prompt, index, reward, **extra):
    return {
        "prompt_id": prompt,
        "rollout_id": f"{prompt}-{index}",
        "reward": reward,
        "response_tokens": 10 + index,
        **extra,
    }


def test_audit_selects_only_nonzero_variance_groups():
    rows = [
        _row("variable", 0, 0.0, kl=0.1, importance_ratio=1.1, clip_high=True),
        _row("variable", 1, 1.0, kl=0.2, importance_ratio=0.9, clip_high=False),
        _row("constant", 0, 1.0, truncated=True),
        _row("constant", 1, 1.0, truncated=False),
    ]
    result = audit_rollouts(rows, expected_group_size=2, max_response_tokens=32)
    assert result["status"] == "PASS"
    assert result["eligible_prompt_ids"] == ["variable"]
    assert result["zero_variance_fraction"] == 0.5
    assert result["all_one_groups"] == 1
    assert result["truncate_fraction"] == 0.25
    assert result["clip_high_fraction"] == 0.5
    assert result["clip_high_coverage"] == 0.5


def test_audit_fails_closed_on_incomplete_or_invalid_groups():
    rows = [_row("only", 0, 0.0), _row("only", 1, math.nan)]
    result = audit_rollouts(rows, expected_group_size=2, max_response_tokens=32)
    assert result["status"] == "FAIL"
    assert result["valid_rows"] == 1
    assert any("expected 2" in error for item in result["errors"] for error in item["errors"])


def test_audit_rejects_duplicate_rollout_ids():
    rows = [_row("a", 0, 0.0), _row("b", 0, 1.0)]
    rows[1]["rollout_id"] = rows[0]["rollout_id"]
    result = audit_rollouts(rows, expected_group_size=2, max_response_tokens=32)
    assert result["status"] == "FAIL"
    assert any("globally unique" in error for item in result["errors"] for error in item["errors"])
