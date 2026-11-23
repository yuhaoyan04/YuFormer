#!/usr/bin/env python3
"""Fit the E1 loss law using three paired seeds at every fixed design point."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

from analyze_e1_scaling import fit_axis, fit_law, non_embedding_params, predict
from audit_experiment_identities import audit, parse_env, sha256, validate_e1_roster

ROOT = Path(__file__).resolve().parents[2]
DOMAINS = ("academic", "books", "chinese", "code", "encyclopedia", "math",
           "multilingual", "qa", "web")
THRESHOLDS = {
    "minimum_loss_r_squared": 0.90,
    "maximum_normalized_rmse": 0.03,
    "maximum_leave_one_out_median_relative_error": 0.05,
    "maximum_leave_one_out_p95_relative_error": 0.10,
    "minimum_throughput_log_r_squared": 0.75,
}
SEED_BOOTSTRAP_RNG_SEED = 20261107
SEED_BOOTSTRAP_POLICY_VALUE = (
    "resample_seed_indices_with_replacement_jointly_across_all_fixed_design_points")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.is_file() and path.read_text() == payload:
        return
    temporary = path.with_name(f".{path.name}.{__import__('os').getpid()}")
    temporary.write_text(payload)
    temporary.replace(path)


def result_for(root: Path, relative_dir: str) -> tuple[dict, Path]:
    path = root / relative_dir / "result.json"
    value = json.loads(path.read_text())
    if value.get("status") != "PASS":
        raise ValueError(f"non-passing result: {path}")
    return value, path


def validate_replication_config(source_name: str, source_config: dict,
                                contract: dict, replication_config: dict,
                                expected_seed: int, expected_run_name: str) -> None:
    expected = dict(source_config)
    expected.update(run_name=expected_run_name, seed=expected_seed)
    source_budget = int(source_config["train"]["budget_tokens"])
    if (contract.get("source_run") != source_name
            or contract.get("name") != expected_run_name
            or int(contract.get("seed", -1)) != expected_seed
            or int(contract.get("budget_tokens", -1)) != source_budget
            or replication_config != expected):
        raise ValueError(f"replication differs from matched source design point: {source_name}")


def validate_paired_seed_design(manifest: dict, source_names: set[str]) -> None:
    expected_seeds = [20261002, 20261003]
    if (manifest.get("paired_seed_design") is not True
            or int(manifest.get("existing_seed", -1)) != 20261001
            or manifest.get("additional_seeds") != expected_seeds
            or int(manifest.get("total_seed_count_per_point", 0)) != 3
            or int(manifest.get("design_points", 0)) != 24):
        raise ValueError("seed replication manifest differs from the paired three-seed design")
    observed: dict[str, list[int]] = {}
    for row in manifest["runs"]:
        source = row.get("source_run")
        if source not in source_names:
            raise ValueError(f"replication references unknown E1 source point: {source}")
        observed.setdefault(source, []).append(int(row["seed"]))
    if (set(observed) != source_names
            or any(sorted(seeds) != expected_seeds for seeds in observed.values())):
        raise ValueError("each E1 point must have exactly the two paired replication seeds")


def metrics_for(result: dict, cfg: dict, expected_name: str) -> dict:
    if result.get("run_name") != expected_name or not result.get("identity_sha256"):
        raise ValueError(f"run identity/name mismatch: {expected_name}")
    hyper = result.get("hyperparameters", {})
    tps = (int(cfg["data"]["seq_len"]) * int(cfg["train"]["micro_batch_size"])
           * int(cfg["train"]["grad_accum"]) * int(cfg["parallel"]["world_size"]))
    expected_iters = int(cfg["train"]["expected_iters"])
    expected_budget = int(cfg["train"]["budget_tokens"])
    expected_tokens = expected_iters * tps
    budget_shortfall = expected_budget - expected_tokens
    if budget_shortfall < 0 or budget_shortfall >= tps:
        raise ValueError(f"configured budget is not aligned to full optimizer steps: {expected_name}")
    if (hyper.get("budget_tokens") != expected_budget
            or hyper.get("tokens_per_step") != tps
            or hyper.get("trained_steps") != expected_iters
            or hyper.get("trained_tokens") != expected_tokens
            or result.get("trained_steps") != expected_iters
            or result.get("trained_tokens") != expected_tokens
            or result.get("configured_budget_shortfall_tokens") != budget_shortfall):
        raise ValueError(f"training accounting mismatch: {expected_name}")
    scalar = result.get("metrics", {})
    def final_metric(key: str) -> float:
        metric = scalar[key]
        if int(metric.get("last_step", -1)) != expected_iters:
            raise ValueError(f"held-out metric is not from the final training step: {expected_name}:{key}")
        return float(metric["last"])

    loss = final_metric("validation_metrics/loss")
    domains = {domain: final_metric(f"validation_metrics/domain/{domain}/loss")
               for domain in DOMAINS}
    if any(not math.isfinite(v) or v <= 0 for v in [loss, *domains.values()]):
        raise ValueError(f"non-finite validation result: {expected_name}")
    return {"loss": loss, "domains": domains, "tokens": expected_tokens,
            "configured_budget_tokens": expected_budget,
            "configured_budget_shortfall_tokens": budget_shortfall,
            "identity_sha256": result["identity_sha256"]}


def paired_seed_bootstrap_mean(seed_arrays: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Resample seed indices once and apply them to every paired design point."""
    if not seed_arrays or any(values.shape != (3,) for values in seed_arrays):
        raise ValueError("paired-seed bootstrap requires three ordered losses at every design point")
    sampled_indices = rng.integers(0, 3, 3)
    return np.asarray([float(np.mean(values[sampled_indices])) for values in seed_arrays])


def compute_optimal_point(theta: np.ndarray, flops: float) -> dict | None:
    """Analytic optimum for the fitted law under the study convention C=6ND."""
    _, log_a, alpha, log_b, beta = (float(x) for x in theta)
    if flops <= 0 or alpha <= 0 or beta <= 0:
        return None
    log_n = (log_a + math.log(alpha) - log_b - math.log(beta)
             + beta * math.log(flops / 6.0)) / (alpha + beta)
    if (not math.isfinite(log_n)
            or log_n > math.log(np.finfo(np.float64).max)
            or log_n < math.log(np.finfo(np.float64).tiny)):
        return None
    n_star = math.exp(log_n)
    d_star = flops / (6.0 * n_star)
    loss_star = float(predict(theta, np.asarray([n_star]), np.asarray([d_star]))[0])
    if not all(math.isfinite(x) and x > 0 for x in (n_star, d_star, loss_star)):
        return None
    return {"n_params": n_star, "tokens": d_star, "tokens_per_param": d_star / n_star,
            "predicted_loss": loss_star}


def _analyze(root: Path, bootstrap_samples: int = 2000) -> dict:
    research_policy = yaml.safe_load((root / "pretrain/configs/research_budget.yaml").read_text())
    replication_policy = research_policy["e1_seed_replication"]
    if replication_policy.get("seed_bootstrap") != SEED_BOOTSTRAP_POLICY_VALUE:
        return {"status": "BLOCKED", "reason": "seed_bootstrap_policy_mismatch",
                "expected_method": SEED_BOOTSTRAP_POLICY_VALUE,
                "observed_method": replication_policy.get("seed_bootstrap")}
    bootstrap_samples = max(bootstrap_samples, int(replication_policy["seed_bootstrap_samples"]))
    manifest_path = root / "data/health/e1_seedrep-contracts.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "PASS" or len(manifest.get("runs", [])) != 48:
        raise ValueError("the complete 48-run E1 replication manifest is required")
    for relative, expected in manifest["inputs"].items():
        if sha256(root / relative) != expected:
            return {"status": "BLOCKED", "reason": f"stale_replication_input:{relative}"}
    for row in manifest["runs"]:
        for key in ("config", "env"):
            if sha256(root / row[key]) != row[f"{key}_sha256"]:
                return {"status": "BLOCKED", "reason": f"replication_contract_drift:{row['name']}"}
    e1_configs = {p.stem: yaml.safe_load(p.read_text())
                  for p in (root / "pretrain/configs/e1").glob("*.yaml")}
    try:
        validate_e1_roster(set(e1_configs))
    except ValueError as exc:
        return {"status": "BLOCKED", "reason": "e1_design_roster_mismatch",
                "observed_base_design_points": len(e1_configs), "detail": str(exc)}
    try:
        validate_paired_seed_design(manifest, set(e1_configs))
    except (ValueError, KeyError, TypeError) as exc:
        return {"status": "BLOCKED", "reason": "seed_replication_design_audit",
                "detail": str(exc)}
    declared_additional_tokens = int(manifest["maximum_additional_tokens"])
    policy_max_additional_tokens = int(replication_policy["maximum_additional_tokens"])
    contract_additional_tokens = sum(int(row["budget_tokens"]) for row in manifest["runs"])
    configured_additional_tokens = 2 * sum(
        int(cfg["train"]["budget_tokens"]) for cfg in e1_configs.values())
    if (declared_additional_tokens != policy_max_additional_tokens
            or contract_additional_tokens != declared_additional_tokens
            or configured_additional_tokens != declared_additional_tokens):
        return {"status": "BLOCKED", "reason": "seed_replication_budget_audit",
                "declared_additional_tokens": declared_additional_tokens,
                "policy_max_additional_tokens": policy_max_additional_tokens,
                "contract_additional_tokens": contract_additional_tokens,
                "configured_additional_tokens": configured_additional_tokens}
    names = {row["name"] for row in manifest["runs"]}
    identity = audit(root, {"e1": set(e1_configs), "e1_seedrep": names})
    if identity.get("status") != "PASS" or identity.get("total") != 72:
        return {"status": "BLOCKED", "reason": "e1_and_replication_identity_audit",
                "identity": identity}

    replication_rows = {row["name"]: row for row in manifest["runs"]}
    grouped: dict[str, list[dict]] = {name: [] for name in e1_configs}
    seed_reports = []
    actual_additional_trained_tokens = 0
    for name, cfg in sorted(e1_configs.items()):
        base_env = parse_env(root / f"pretrain/configs/e1/{name}.env")
        base_result, base_path = result_for(root, str(Path(base_env["YUFORMER_PRETRAIN_OUT"]).relative_to(root)))
        base_metrics = metrics_for(base_result, cfg, name)
        seed_results = [{"seed": int(cfg["seed"]), **base_metrics}]
        seed_reports.append({"run_name": name, "seed": int(cfg["seed"]),
                             "role": "screen", "loss": base_metrics["loss"],
                             "identity_sha256": base_metrics["identity_sha256"]})
        for seed_index, seed in ((2, 20261002), (3, 20261003)):
            replicate_name = f"e1_seedrep_{name}_s{seed_index}"
            contract = replication_rows[replicate_name]
            replicate_cfg = yaml.safe_load((root / contract["config"]).read_text())
            env = parse_env(root / contract["env"])
            result, result_path = result_for(root, str(Path(env["YUFORMER_PRETRAIN_OUT"]).relative_to(root)))
            measured = metrics_for(result, replicate_cfg, replicate_name)
            validate_replication_config(name, cfg, contract, replicate_cfg, seed, replicate_name)
            actual_additional_trained_tokens += measured["tokens"]
            seed_results.append({"seed": seed, **measured})
            seed_reports.append({"run_name": name, "replicate": replicate_name,
                                 "seed": seed, "role": "replication",
                                 "loss": measured["loss"],
                                 "identity_sha256": measured["identity_sha256"],
                                 "result": result_path.relative_to(root).as_posix(),
                                 "result_sha256": sha256(result_path)})
        if sorted(item["seed"] for item in seed_results) != [20261001, 20261002, 20261003]:
            raise ValueError(f"three-seed roster mismatch: {name}")
        losses = np.asarray([item["loss"] for item in seed_results], dtype=np.float64)
        domain_means = {domain: float(np.mean([item["domains"][domain] for item in seed_results]))
                        for domain in DOMAINS}
        configured_budget = int(cfg["train"]["budget_tokens"])
        actual_tokens = int(seed_results[0]["tokens"])
        if any(item["tokens"] != actual_tokens for item in seed_results):
            raise ValueError(f"seed runs consumed different actual token counts: {name}")
        row = {"run_name": name,
               "status": "PASS", "identity_sha256": base_metrics["identity_sha256"],
               "n_params": non_embedding_params(yaml.safe_load(
                   (root / "model/configs/model_configs.yaml").read_text())["presets"][cfg["model"]["preset"]]),
               "configured_budget_tokens": configured_budget,
               "tokens": actual_tokens,
               "configured_budget_shortfall_tokens": int(seed_results[0][
                   "configured_budget_shortfall_tokens"]),
               "validation_loss": float(losses.mean()),
               "seed_losses": [float(value) for value in losses],
               "seed_std_loss": float(losses.std(ddof=1)),
               "seed_cv_loss": float(losses.std(ddof=1) / losses.mean()),
               "domain_mean_losses": domain_means,
               "seed_values": [item["seed"] for item in seed_results]}
        grouped[name] = row

    ordered = [grouped[name] for name in sorted(grouped)]
    n = np.asarray([row["n_params"] for row in ordered], dtype=np.float64)
    d = np.asarray([row["tokens"] for row in ordered], dtype=np.float64)
    losses = np.asarray([row["validation_loss"] for row in ordered], dtype=np.float64)
    theta, rmse = fit_law(n, d, losses)
    fitted = predict(theta, n, d)
    residuals = fitted - losses
    variance = float(np.sum((losses - losses.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(residuals ** 2)) / variance if variance > 0 else float("nan")
    normalized_rmse = rmse / float(losses.mean())

    loo = []
    for i, row in enumerate(ordered):
        keep = np.arange(len(n)) != i
        fitted_loo, _ = fit_law(n[keep], d[keep], losses[keep])
        prediction = float(predict(fitted_loo, n[i:i + 1], d[i:i + 1])[0])
        loo.append({"run_name": row["run_name"], "observed": float(losses[i]),
                    "prediction": prediction,
                    "relative_error": abs(prediction - float(losses[i])) / float(losses[i])})
    loo_median = float(np.median([row["relative_error"] for row in loo]))
    loo_p95 = float(np.percentile([row["relative_error"] for row in loo], 95))

    rng = np.random.default_rng(SEED_BOOTSTRAP_RNG_SEED)
    bootstrap_parameters, bootstrap_nrmse = [], []
    seed_arrays = [np.asarray(row["seed_losses"], dtype=np.float64) for row in ordered]
    attempts = 0
    while len(bootstrap_parameters) < bootstrap_samples and attempts < bootstrap_samples * 4:
        attempts += 1
        # Resample seed indices jointly across design points so paired-seed
        # covariance is retained in each bootstrap fit.
        sampled_means = paired_seed_bootstrap_mean(seed_arrays, rng)
        try:
            boot_theta, boot_rmse = fit_law(n, d, sampled_means)
        except (ValueError, FloatingPointError):
            continue
        if not np.all(np.isfinite(boot_theta)) or not math.isfinite(boot_rmse):
            continue
        bootstrap_parameters.append(boot_theta)
        bootstrap_nrmse.append(boot_rmse / float(sampled_means.mean()))
    if len(bootstrap_parameters) < max(500, bootstrap_samples // 2):
        return {"status": "FAIL", "reason": "insufficient_seed_bootstrap_fits",
                "bootstrap_successful": len(bootstrap_parameters), "runs": ordered}
    boot = np.asarray(bootstrap_parameters)
    parameter_names = ("floor", "log_a", "alpha", "log_b", "beta")
    parameters = {name: {"estimate": float(theta[i]),
                         "ci95": [float(v) for v in np.percentile(boot[:, i], [2.5, 97.5])],
                         "bootstrap_seed_interval": True}
                  for i, name in enumerate(parameter_names)}
    compute_optimal_frontier = []
    for family in ("c0", "c1", "c2"):
        family_rows = [row for row in ordered if row["run_name"].startswith(f"e1_iso_{family}_")]
        if not family_rows:
            continue
        flops = float(np.median([6.0 * row["n_params"] * row["tokens"]
                                 for row in family_rows]))
        point = compute_optimal_point(theta, flops)
        bootstrap_points = [value for sample in boot
                            if (value := compute_optimal_point(sample, flops)) is not None]
        summary = {"family": family, "flops": flops,
                   "budget_definition": "C=6*N*D with N=non_embedding_parameters",
                   "valid_bootstrap_fraction": len(bootstrap_points) / len(boot)}
        if point is not None and len(bootstrap_points) >= max(500, len(boot) // 2):
            summary.update({"status": "PASS", **point,
                            "ci95": {key: [float(x) for x in np.percentile(
                                [sample[key] for sample in bootstrap_points], [2.5, 97.5])]
                                for key in ("n_params", "tokens", "tokens_per_param", "predicted_loss")}})
        else:
            summary["status"] = "UNSTABLE_PARAMETER_BOOTSTRAP"
        compute_optimal_frontier.append(summary)
    controlled = {}
    for axis_name, prefix, x_values in (
        ("L_N_fixed_D", "e1_ln_", [row["n_params"] for row in ordered
                                     if row["run_name"].startswith("e1_ln_")]),
        ("L_D_fixed_300M", "e1_ld_300m_", [row["tokens"] for row in ordered
                                             if row["run_name"].startswith("e1_ld_300m_")]),
    ):
        selected = [row for row in ordered if row["run_name"].startswith(prefix)]
        axis_fit, axis_rmse = fit_axis(np.asarray(x_values, dtype=np.float64),
                                       np.asarray([row["validation_loss"] for row in selected]))
        controlled[axis_name] = {"parameters": axis_fit, "rmse": axis_rmse,
                                 "points": len(selected)}

    screen_fit_path = root / "experiments/e1_scaling/results/scaling_fit_screening_20261107.json"
    screen_fit = json.loads(screen_fit_path.read_text())
    throughput_gate = bool(screen_fit.get("fit_quality_gate", {}).get("checks", {}).get(
        "throughput_log_r_squared", False))
    checks = {
        "loss_r_squared": math.isfinite(r_squared) and r_squared >= THRESHOLDS["minimum_loss_r_squared"],
        "normalized_rmse": math.isfinite(normalized_rmse)
        and normalized_rmse <= THRESHOLDS["maximum_normalized_rmse"],
        "leave_one_out_median": loo_median <= THRESHOLDS["maximum_leave_one_out_median_relative_error"],
        "leave_one_out_p95": loo_p95 <= THRESHOLDS["maximum_leave_one_out_p95_relative_error"],
        "throughput_log_r_squared": throughput_gate,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    if status != "PASS":
        for point in compute_optimal_frontier:
            if point.get("status") == "PASS":
                point["status"] = "WITHHELD_FIT_GATE_FAIL"
    perf_rows = [row for row in ordered if row["run_name"].startswith("e1_ln_")]
    target_tokens = 1_000_000_000_000
    previous_candidates = {row["source_run"]: row
                           for row in screen_fit.get("production_candidates", [])}
    candidates = []
    for row in sorted(perf_rows, key=lambda item: item["n_params"]):
        loss_distribution = (boot[:, 0]
                             + np.exp(boot[:, 1] - boot[:, 2] * np.log(row["n_params"]))
                             + np.exp(boot[:, 3] - boot[:, 4] * np.log(target_tokens)))
        candidate = dict(previous_candidates.get(row["run_name"], {}))
        candidate.update({"source_run": row["run_name"], "n_params": row["n_params"],
                          "predicted_loss_at_1t": float(predict(
                              theta, np.asarray([row["n_params"]]), np.asarray([target_tokens]))[0]),
                          "predicted_loss_at_1t_ci95": [float(x) for x in
                                                         np.percentile(loss_distribution, [2.5, 97.5])]})
        candidates.append(candidate)
    feasible = [row for row in candidates if row.get("feasible_within_45_days")]
    recommendation = (min(feasible, key=lambda row: row["predicted_loss_at_1t"])
                      if feasible else min(candidates, key=lambda row: row.get("predicted_days_4gpu", math.inf)))
    production_decision_status = "READY" if status == "PASS" else "WITHHELD_FIT_GATE_FAIL"
    for candidate in candidates:
        candidate["decision_status"] = production_decision_status
    if status != "PASS":
        recommendation = {"status": production_decision_status,
                          "reason": "seed_aware_scaling_fit_quality_gate"}
    input_snapshot = {**manifest["inputs"], **manifest["source_identity_evidence"],
                      **identity.get("evidence_snapshot", {}),
                      "data/health/e1_seedrep-contracts.json": sha256(manifest_path),
                      "experiments/e1_scaling/results/scaling_fit_screening_20261107.json": sha256(screen_fit_path),
                      "pretrain/scripts/analyze_e1_seedrep.py": sha256(Path(__file__))}
    for row in seed_reports:
        if row.get("result_sha256"):
            input_snapshot[row["result"]] = row["result_sha256"]
    return {
        "schema_version": 2, "status": status,
        "failure_reason": None if status == "PASS" else "seed_aware_scaling_fit_quality_gate",
        "study": "all_24_E1_design_points_with_three_paired_seeds",
        "observed_results": 24, "expected_results": 24, "replication_result_count": 72,
        "additional_tokens": configured_additional_tokens,
        "additional_trained_tokens": actual_additional_trained_tokens,
        "budget_audit": {
            "status": "PASS",
            "policy_max_additional_tokens": policy_max_additional_tokens,
            "declared_additional_tokens": declared_additional_tokens,
            "contract_additional_tokens": contract_additional_tokens,
            "configured_additional_tokens": configured_additional_tokens,
            "actual_additional_trained_tokens": actual_additional_trained_tokens,
            "actual_total_trained_tokens_including_screen": (
                actual_additional_trained_tokens + sum(int(row["tokens"]) for row in ordered)),
            "configured_budget_shortfall_tokens": (
                configured_additional_tokens - actual_additional_trained_tokens),
            "seed_count_per_design_point": 3,
            "design_points": 24,
        },
        "identity_audit": {
            "status": identity["status"], "passed": identity["passed"],
            "total": identity["total"], "roster_exact": identity["roster_exact"],
            "evidence_snapshot_count": identity["evidence_snapshot_count"],
        },
        "seed_design_audit": {
            "status": "PASS", "paired_seed_design": True,
            "design_points": 24, "seed_count_per_design_point": 3,
            "existing_seed": 20261001,
            "additional_seeds": [20261002, 20261003],
        },
        "analysis_method_amendments": manifest.get("analysis_method_amendments", []),
        "model": "loss=E+A/N^alpha+B/D^beta", "parameters": parameters,
        "rmse": rmse, "bootstrap_requested": bootstrap_samples,
        "seed_bootstrap_rng_seed": SEED_BOOTSTRAP_RNG_SEED,
        "bootstrap_successful": len(bootstrap_parameters),
        "seed_bootstrap_normalized_rmse_ci95": [float(x) for x in
                                                 np.percentile(bootstrap_nrmse, [2.5, 97.5])],
        "fit_quality_gate": {"status": status, "thresholds": THRESHOLDS, "checks": checks},
        "throughput_fit": screen_fit["throughput_fit"],
        "independent_performance": screen_fit["independent_performance"],
        "performance_evidence_status": screen_fit.get("performance_evidence_status"),
        "controlled_axis_fits": controlled,
        "compute_optimal_frontier": compute_optimal_frontier,
        "production_decision_status": production_decision_status,
        "production_candidates": candidates, "production_recommendation": recommendation,
        "runs": ordered,
        "production_policy": screen_fit.get("production_policy"),
        "fit_diagnostics": {
            "r_squared": r_squared, "normalized_rmse": normalized_rmse,
            "residuals": [{"run_name": row["run_name"], "observed": float(losses[i]),
                            "predicted": float(fitted[i]), "residual": float(residuals[i])}
                           for i, row in enumerate(ordered)],
            "leave_one_out": loo, "leave_one_out_median_relative_error": loo_median,
            "leave_one_out_p95_relative_error": loo_p95,
            "seed_level": {"seed_count_per_design_point": 3,
                           "bootstrap": "joint_seed_index_resampling_across_design_points",
                           "policy_value": SEED_BOOTSTRAP_POLICY_VALUE,
                           "mean_within_point_std_loss": float(np.mean([row["seed_std_loss"]
                                                                        for row in ordered])),
                           "median_within_point_cv_loss": float(np.median([row["seed_cv_loss"]
                                                                          for row in ordered])),
                           "per_design_point": [{"run_name": row["run_name"],
                                                 "loss_by_seed": row["seed_losses"],
                                                 "std_loss": row["seed_std_loss"],
                                                 "cv_loss": row["seed_cv_loss"]}
                                                for row in ordered]},
            "seed_results": seed_reports,
        },
        "evidence_snapshot": input_snapshot,
        "screening_fit_sha256": sha256(screen_fit_path),
    }


def analyze(root: Path, bootstrap_samples: int = 2000) -> dict:
    try:
        return _analyze(root, bootstrap_samples)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        return {"schema_version": 2, "status": "BLOCKED",
                "reason": f"seed_replication_analysis:{type(exc).__name__}:{exc}",
                "expected_results": 24, "observed_results": 0, "runs": []}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments/e1_scaling/results/seed_replication_fit.json")
    args = parser.parse_args()
    result = analyze(args.root.resolve(), args.bootstrap_samples)
    result["analysis_tool_sha256"] = sha256(Path(__file__))
    output = args.output if args.output.is_absolute() else args.root / args.output
    atomic_json(output, result)
    print(json.dumps({"status": result["status"], "expected_results": result.get("expected_results"),
                      "normalized_rmse": result.get("fit_diagnostics", {}).get("normalized_rmse")}))
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
