#!/usr/bin/env python3
"""Analyze exact E3 results and freeze an evidence-backed mixture candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "pretrain/scripts"))
from performance_evidence import promote_bound_loss_analysis
from research_budget import screen_budget
DOMAINS = {"academic", "books", "chinese", "code", "encyclopedia",
           "math", "multilingual", "qa", "web"}
RESPONSE_ARMS = {
    "e3_code_2x": "code", "e3_math_2x": "math", "e3_chinese_2x": "chinese",
    "e3_academic_2x": "academic", "e3_encyclopedia_2x": "encyclopedia",
}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            old_semantic = {key: item for key, item in previous.items() if key != "observed_at"}
            new_semantic = {key: item for key, item in value.items() if key != "observed_at"}
            if old_semantic == new_semantic:
                return
        except (OSError, ValueError, AttributeError):
            pass
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def analyze(matrix_path: Path, results_root: Path,
            policy_path: Path | None = None) -> dict[str, Any]:
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    policy_path = policy_path or matrix_path.with_name("e3_analysis_policy.yaml")
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    expected = set(matrix["arms"])
    tokens_per_step = 4096 * 4 * 16
    matched_trained_tokens = (
        screen_budget(ROOT, int(matrix['budget_tokens_per_arm'])) // tokens_per_step * tokens_per_step
    )
    rows = []
    unexpected = []
    for path in sorted(results_root.glob("*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        name = result.get("run_name")
        if name not in expected:
            unexpected.append({"path": str(path), "run_name": name})
            continue
        metrics = result.get("metrics", {})
        loss_only = bool(result.get("execution_context", {}).get("loss_only"))
        aggregate = metrics.get("validation_metrics/loss", {}).get("last")
        throughput = metrics.get("throughput(tps)", {}).get("last")
        losses = {domain: metrics.get(f"validation_metrics/domain/{domain}/loss", {}).get("last")
                  for domain in sorted(DOMAINS)}
        issues = []
        if result.get("status") != "PASS": issues.append("result_not_pass")
        if (not isinstance(aggregate, (int, float))
                or not math.isfinite(float(aggregate)) or float(aggregate) <= 0):
            issues.append("aggregate_loss_invalid")
        missing = [domain for domain, value in losses.items()
                   if (not isinstance(value, (int, float))
                       or not math.isfinite(float(value)) or float(value) <= 0)]
        if missing: issues.append("missing_domains:" + ",".join(missing))
        if (not loss_only and (not isinstance(throughput, (int, float))
                or not math.isfinite(float(throughput)) or float(throughput) <= 0)):
            issues.append("throughput_missing_or_invalid")
        if not isinstance(result.get("identity_sha256"), str) or not result["identity_sha256"]:
            issues.append("identity_sha256_missing")
        hyper = result.get("hyperparameters") or {}
        trained_steps = hyper.get("trained_steps")
        if hyper.get("tokens_per_step") != tokens_per_step:
            issues.append("tokens_per_step_mismatch")
        if (not isinstance(trained_steps, int) or isinstance(trained_steps, bool)
                or trained_steps <= 0
                or trained_steps * tokens_per_step != matched_trained_tokens):
            issues.append("trained_steps_mismatch")
        if hyper.get("trained_tokens") != matched_trained_tokens:
            issues.append("trained_tokens_mismatch")
        rows.append({"run_name": name, "status": "PASS" if not issues else "FAIL",
                     "issues": issues, "validation_loss": aggregate, "domain_losses": losses,
                     "throughput_tps_per_device": throughput,
                     "loss_only": loss_only,
                     "performance_metrics_excluded": (
                         ["throughput(tps)", "mfu(%)", "wall_time"] if loss_only else []
                     ),
                     "trained_tokens": matched_trained_tokens,
                     "weights": matrix["arms"][name]["weights"],
                     "identity_sha256": result.get("identity_sha256")})
    observed = {row["run_name"] for row in rows}
    duplicates = sorted(name for name in observed
                        if sum(row["run_name"] == name for row in rows) > 1)
    eligible = [row for row in rows if row["status"] == "PASS"]
    if (observed != expected or len(eligible) != len(expected)
            or duplicates or unexpected):
        return {"schema_version": 1, "status": "BLOCKED",
                "observed_at": datetime.now(timezone.utc).isoformat(),
                "expected_results": len(expected), "observed_results": len(observed),
                "missing": sorted(expected - observed), "duplicates": duplicates,
                "unexpected": unexpected, "runs": rows}
    by_name = {row["run_name"]: row for row in eligible}
    baseline = by_name["e3_m2_base"]
    practical_equivalence_delta = float(
        policy.get("practical_equivalence_absolute_loss", 0.002)
    )
    maximum_domain_regression = float(
        policy.get("maximum_domain_relative_regression_vs_m2", 0.02)
    )
    if practical_equivalence_delta < 0 or maximum_domain_regression < 0:
        return {"schema_version": 1, "status": "FAIL",
                "reason": "invalid_analysis_policy", "runs": rows}
    for row in eligible:
        regressions = {
            domain: ((float(row["domain_losses"][domain])
                      - float(baseline["domain_losses"][domain]))
                     / float(baseline["domain_losses"][domain]))
            for domain in sorted(DOMAINS)
        }
        row["domain_relative_regressions_vs_m2"] = regressions
        row["maximum_domain_relative_regression_vs_m2"] = max(regressions.values())
        row["domain_noninferiority_pass"] = all(
            value <= maximum_domain_regression for value in regressions.values()
        )
    responses = []
    for arm, domain in RESPONSE_ARMS.items():
        row = by_name[arm]
        weight_delta = float(row["weights"][domain]) - float(baseline["weights"][domain])
        loss_delta = float(row["domain_losses"][domain]) - float(baseline["domain_losses"][domain])
        responses.append({"arm": arm, "domain": domain, "weight_delta": weight_delta,
                          "target_loss_delta": loss_delta,
                          "loss_improvement_per_weight_point": -loss_delta / weight_delta,
                          "aggregate_loss_delta": float(row["validation_loss"] - baseline["validation_loss"])})
    ranking = sorted(eligible, key=lambda row: (float(row["validation_loss"]), row["run_name"]))
    safe_ranking = [row for row in ranking if row["domain_noninferiority_pass"]]
    # The M2 baseline has zero regression by definition, so this roster cannot
    # be empty unless the evidence itself is inconsistent.
    if not safe_ranking:
        return {"schema_version": 1, "status": "FAIL",
                "reason": "no_domain_noninferior_mixture", "runs": rows}
    empirical_best = safe_ranking[0]
    runner_up = safe_ranking[1] if len(safe_ranking) > 1 else safe_ranking[0]
    equivalent = [row for row in safe_ranking
                  if float(row["validation_loss"] - empirical_best["validation_loss"])
                  <= practical_equivalence_delta]
    winner = min(
        equivalent,
        key=lambda row: (
            sum(abs(float(row["weights"][domain]) - float(baseline["weights"][domain]))
                for domain in DOMAINS),
            float(row["validation_loss"]), row["run_name"],
        ),
    )
    quality = by_name["e3_m3_quality"]
    return {
        "schema_version": 1,
        "status": ("PASS_PROVISIONAL" if any(row["loss_only"] for row in eligible)
                   else "PASS"),
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "expected_results": len(expected), "observed_results": len(observed),
        "selection": {"run_name": winner["run_name"], "weights": winner["weights"],
                      "empirical_best_run_name": empirical_best["run_name"],
                      "validation_loss": winner["validation_loss"],
                      "throughput_tps_per_device": winner["throughput_tps_per_device"],
                      "identity_sha256": winner["identity_sha256"],
                      "loss_gap_to_empirical_best": float(winner["validation_loss"]
                                                          - empirical_best["validation_loss"]),
                      "empirical_best_margin_to_runner_up": float(
                          runner_up["validation_loss"] - empirical_best["validation_loss"]),
                      "practical_equivalence_delta": practical_equivalence_delta,
                      "maximum_domain_relative_regression_vs_m2":
                          winner["maximum_domain_relative_regression_vs_m2"],
                      "domain_noninferiority_pass": winner["domain_noninferiority_pass"]},
        "ranking": ranking, "response_slopes": responses,
        "domain_noninferiority": {
            "baseline": "e3_m2_base",
            "maximum_relative_regression": maximum_domain_regression,
            "passing_arms": [row["run_name"] for row in safe_ranking],
            "excluded_arms": [row["run_name"] for row in ranking
                              if not row["domain_noninferiority_pass"]],
        },
        "quality_tradeoff": {
            "m3_minus_m2_aggregate_loss": float(quality["validation_loss"] - baseline["validation_loss"]),
            "per_domain_loss_delta": {domain: float(quality["domain_losses"][domain]
                                                       - baseline["domain_losses"][domain])
                                      for domain in sorted(DOMAINS)},
        },
        "selection_rule": ("among arms with <= configured per-domain relative regression versus "
                           "M2, choose the closest-to-M2 arm within the configured aggregate-loss "
                           "equivalence band at matched token budget"),
        "practically_equivalent_arms": [row["run_name"] for row in equivalent],
        "uncertainty_note": (f"E3 is a one-seed {matched_trained_tokens}-token mixture screen; report the winner margin "
                             "and do not interpret small deltas as a statistically resolved effect."),
        "analysis_policy": {"path": str(policy_path),
                            "sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
                            **policy},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path,
                        default=ROOT / "experiments/e3_data_mix/configs/e3_matrix.yaml")
    parser.add_argument("--results-root", type=Path, default=ROOT / "data/experiments/e3")
    parser.add_argument("--policy", type=Path,
                        default=ROOT / "experiments/e3_data_mix/configs/e3_analysis_policy.yaml")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments/e3_data_mix/results/selection.json")
    args = parser.parse_args()
    result = analyze(args.matrix, args.results_root, args.policy)
    result["analysis_tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = promote_bound_loss_analysis(ROOT, "e3", result)
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "observed": result["observed_results"],
                      "selection": result.get("selection")}, sort_keys=True))
    return 0 if result["status"] in {"PASS", "PASS_PROVISIONAL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
