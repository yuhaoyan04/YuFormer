import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/select_lr_pilots.py"
SPEC = importlib.util.spec_from_file_location("select_lr_pilots", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_selects_bracketed_lowest_final_validation_loss_per_size(tmp_path):
    paths = []
    for size in MODULE.SIZES:
        for index, level in enumerate(MODULE.LEVELS):
            path = tmp_path / f"{size}-{level}.json"
            path.write_text(json.dumps({
                "status": "PASS", "run_name": f"e1_lr_{size}_{level}",
                "identity_sha256": f"{size}-{level}",
                "hyperparameters": {"learning_rate": 1e-4 * (2 ** index),
                                    "tokens_per_step": MODULE.TOKENS_PER_STEP,
                                    "trained_steps": (MODULE.MATCHED_TRAINED_TOKENS
                                                      // MODULE.TOKENS_PER_STEP),
                                    "trained_tokens": MODULE.MATCHED_TRAINED_TOKENS},
                "metrics": {"validation_metrics/loss": {"last": 1.0 + abs(index - 2)}},
            }), encoding="utf-8")
            paths.append(path)
    result = MODULE.select(paths)
    assert result["status"] == "PASS"
    assert all(value["level"] == "mid" for value in result["selections"].values())
    assert result["boundary_selections"] == []


def test_selection_blocks_on_incomplete_roster(tmp_path):
    path = tmp_path / "one.json"
    path.write_text(json.dumps({
        "status": "PASS", "run_name": "e1_lr_50m_mid",
        "metrics": {"validation_metrics/loss": {"last": 2.0}},
    }), encoding="utf-8")
    result = MODULE.select([path])
    assert result["status"] == "BLOCKED"
    assert len(result["missing"]) == 14


def test_selection_rejects_duplicate_run_identity(tmp_path):
    paths = []
    for size in MODULE.SIZES:
        for level in MODULE.LEVELS:
            path = tmp_path / f"{size}-{level}.json"
            path.write_text(json.dumps({
                "status": "PASS", "run_name": f"e1_lr_{size}_{level}",
                "identity_sha256": f"{size}-{level}",
                "hyperparameters": {"learning_rate": 1e-4,
                                    "tokens_per_step": MODULE.TOKENS_PER_STEP,
                                    "trained_steps": (MODULE.MATCHED_TRAINED_TOKENS
                                                      // MODULE.TOKENS_PER_STEP),
                                    "trained_tokens": MODULE.MATCHED_TRAINED_TOKENS},
                "metrics": {"validation_metrics/loss": {"last": 2.0}},
            }))
            paths.append(path)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(paths[0].read_text())
    result = MODULE.select(sorted(paths + [duplicate]))
    assert result["status"] == "BLOCKED"
    assert len(result["duplicates"]) == 1


def test_selection_blocks_when_optimum_is_on_search_boundary(tmp_path):
    paths = []
    for size in MODULE.SIZES:
        for index, level in enumerate(MODULE.LEVELS):
            path = tmp_path / f"{size}-{level}.json"
            path.write_text(json.dumps({
                "status": "PASS", "run_name": f"e1_lr_{size}_{level}",
                "identity_sha256": f"{size}-{level}",
                "hyperparameters": {"learning_rate": 1e-5 * (2 ** index),
                                    "tokens_per_step": MODULE.TOKENS_PER_STEP,
                                    "trained_steps": (MODULE.MATCHED_TRAINED_TOKENS
                                                      // MODULE.TOKENS_PER_STEP),
                                    "trained_tokens": MODULE.MATCHED_TRAINED_TOKENS},
                "metrics": {"validation_metrics/loss": {"last": 5.0 - index}},
            }))
            paths.append(path)
    result = MODULE.select(paths)
    assert result["status"] == "BLOCKED"
    assert {row["size"] for row in result["boundary_selections"]} == set(MODULE.SIZES)
