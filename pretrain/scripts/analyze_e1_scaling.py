#!/usr/bin/env python3
"""Fit the E1 scaling law, bootstrap uncertainty, and measured throughput law."""
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

import numpy as np
import yaml
from scipy.optimize import least_squares
sys.path.insert(0, str(Path(__file__).resolve().parent))
from performance_evidence import qualified_performance
from audit_experiment_identities import parse_env


EXPECTED_PREFIXES = ("e1_ln_", "e1_ld_", "e1_iso_")


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


def non_embedding_params(preset: dict[str, Any]) -> int:
    h, layers, ffn = (int(preset[key]) for key in ("hidden_size", "num_layers", "intermediate_size"))
    heads, kv_heads = int(preset["num_heads"]), int(preset["num_kv_heads"])
    head_dim = h // heads
    kv_width = kv_heads * head_dim
    full_layers = sum((index + 1) % int(preset["attention_every"]) == 0 for index in range(layers))
    gated_layers = (layers - full_layers) if preset.get("attention_output_gate", False) else 0
    attention = layers * (h * (h + 2 * kv_width) + h * h) + gated_layers * h * h
    mlp = layers * 3 * h * ffn
    norms = 2 * layers * h + h + (2 * layers * head_dim if preset.get("qk_norm", False) else 0)
    return attention + mlp + norms


def predict(theta: np.ndarray, n: np.ndarray, d: np.ndarray) -> np.ndarray:
    floor, log_a, alpha, log_b, beta = theta
    return floor + np.exp(log_a - alpha * np.log(n)) + np.exp(log_b - beta * np.log(d))


def fit_law(n: np.ndarray, d: np.ndarray, loss: np.ndarray) -> tuple[np.ndarray, float]:
    minimum = float(loss.min())
    log_n, log_d = np.log(n), np.log(d)
    ref_n, ref_d = float(np.median(log_n)), float(np.median(log_d))
    def residual(theta):
        return (theta[0] + np.exp(theta[1] - theta[2] * (log_n - ref_n))
                + np.exp(theta[3] - theta[4] * (log_d - ref_d)) - loss)
    lower = np.array([0.0, -30.0, 0.01, -30.0, 0.01])
    upper = np.array([minimum, 50.0, 2.0, 50.0, 2.0])
    best = None
    for floor_fraction, exponent in ((.5, .2), (.9, .3), (.1, .6)):
        coefficient = math.log(max(minimum * (1 - floor_fraction) / 2, 1e-8))
        initial = np.array([minimum * floor_fraction, coefficient, exponent, coefficient, exponent])
        fitted = least_squares(residual, initial, bounds=(lower, upper), x_scale='jac',
                               max_nfev=3000, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        error = float(np.sqrt(np.mean(residual(fitted.x) ** 2)))
        if best is None or error < best[0]:
            theta = fitted.x.copy()
            theta[1] += theta[2] * ref_n
            theta[3] += theta[4] * ref_d
            best = (error, theta)
        if error < 1e-9:
            break
    return best[1], best[0]


def fit_axis(x: np.ndarray, loss: np.ndarray) -> tuple[dict[str, float], float]:
    """Fit E + A/x^alpha for the controlled L(N) or L(D) section."""
    minimum = float(loss.min())
    initial = np.asarray([minimum * 0.9, math.log(max(minimum, 1e-3)) + 5, 0.3])
    fitted = least_squares(
        lambda theta: theta[0] + np.exp(theta[1] - theta[2] * np.log(x)) - loss,
        initial, bounds=([0.0, -30.0, 0.01], [minimum, 50.0, 2.0]), max_nfev=10000,
    )
    predicted = fitted.x[0] + np.exp(fitted.x[1] - fitted.x[2] * np.log(x))
    return ({"floor": float(fitted.x[0]), "log_coefficient": float(fitted.x[1]),
             "exponent": float(fitted.x[2])},
            float(np.sqrt(np.mean((predicted - loss) ** 2))))


def analyze(root: Path, results_root: Path, bootstrap_samples: int = 1000) -> dict[str, Any]:
    # Once all paired seed replications have materialized, promote their
    # seed-aware fit to the canonical E1 result. Until then retain the
    # immutable single-seed screening analysis and its explicit quality status.
    replication_manifest = root / "data/health/e1_seedrep-contracts.json"
    if replication_manifest.is_file():
        try:
            manifest = json.loads(replication_manifest.read_text(encoding="utf-8"))
            replication_complete = (manifest.get("status") == "PASS"
                                    and len(manifest.get("runs", [])) == 48
                                    and all((root / parse_env(root / row["env"])["YUFORMER_PRETRAIN_OUT"]
                                             / "result.json").is_file()
                                            for row in manifest["runs"]))
        except (OSError, ValueError, KeyError, TypeError):
            replication_complete = False
        if replication_complete:
            from analyze_e1_seedrep import analyze as analyze_seedrep
            return analyze_seedrep(root, bootstrap_samples)
    configs = {path.stem: yaml.safe_load(path.read_text(encoding="utf-8"))
               for path in (root / "pretrain/configs/e1").glob("*.yaml")}
    presets = yaml.safe_load((root / "model/configs/model_configs.yaml").read_text(encoding="utf-8"))["presets"]
    expected = set(configs)
    rows, rejected = [], []
    for path in sorted(results_root.glob("*/result.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            rejected.append({"path": str(path), "reason": f"{type(exc).__name__}:{exc}"})
            continue
        name = str(result.get("run_name", ""))
        if name not in expected or not name.startswith(EXPECTED_PREFIXES):
            continue
        loss = result.get("metrics", {}).get("validation_metrics/loss", {}).get("last")
        loss_only = bool(result.get("execution_context", {}).get("loss_only"))
        tps_points = result.get("series", {}).get("throughput(tps)", [])
        tps = [float(point["value"]) for point in tps_points[len(tps_points) // 4:]]
        cfg = configs[name]
        preset = presets[cfg["model"]["preset"]]
        issues = []
        if result.get("status") != "PASS": issues.append("result_not_pass")
        if (not isinstance(loss, (int, float)) or not math.isfinite(float(loss))
                or float(loss) <= 0):
            issues.append("validation_loss_missing_or_invalid")
        if (not loss_only
                and (not tps or any(not math.isfinite(value) or value <= 0 for value in tps))):
            issues.append("throughput_missing_or_invalid")
        if not result.get("identity_sha256"):
            issues.append("identity_sha256_missing")
        hyper = result.get("hyperparameters") or {}
        expected_tps = (int(cfg["data"]["seq_len"])
                        * int(cfg["train"]["micro_batch_size"])
                        * int(cfg["train"]["grad_accum"])
                        * int(cfg["parallel"]["world_size"]))
        expected_trained_tokens = int(cfg["train"]["expected_iters"]) * expected_tps
        if hyper.get("model_preset") != cfg["model"]["preset"]:
            issues.append("model_preset_mismatch")
        if hyper.get("budget_tokens") != int(cfg["train"]["budget_tokens"]):
            issues.append("budget_tokens_mismatch")
        if hyper.get("tokens_per_step") != expected_tps:
            issues.append("tokens_per_step_mismatch")
        if hyper.get("trained_steps") != int(cfg["train"]["expected_iters"]):
            issues.append("trained_steps_mismatch")
        if hyper.get("trained_tokens") != expected_trained_tokens:
            issues.append("trained_tokens_mismatch")
        rows.append({"run_name": name, "result": str(path),
                     "status": "PASS" if not issues else "FAIL", "issues": issues,
                     "n_params": non_embedding_params(preset),
                     "configured_budget_tokens": int(cfg["train"]["budget_tokens"]),
                     "tokens": expected_trained_tokens,
                     "validation_loss": loss,
                     "median_tps_per_device": float(np.median(tps)) if tps else None,
                     "loss_only": loss_only,
                     "performance_metrics_excluded": (
                         ["throughput(tps)", "mfu(%)", "wall_time"] if loss_only else []
                     ),
                     "identity_sha256": result.get("identity_sha256")})
    observed = {row["run_name"] for row in rows}
    eligible = [row for row in rows if row["status"] == "PASS"]
    loss_only_mode = any(row["loss_only"] for row in eligible)
    performance = {"status": "NOT_NEEDED", "runs": [], "snapshot": {}}
    missing = sorted(expected - observed)
    if missing or len(eligible) != len(expected):
        return {"schema_version": 1, "status": "BLOCKED", "observed_at": datetime.now(timezone.utc).isoformat(),
                "expected_results": len(expected), "observed_results": len(observed),
                "missing": missing, "rejected": rejected, "runs": rows}
    if loss_only_mode:
        performance = qualified_performance(root, configs, presets, non_embedding_params)
        loss_only_mode = performance["status"] != "PASS"
    n = np.asarray([row["n_params"] for row in eligible], dtype=np.float64)
    d = np.asarray([row["tokens"] for row in eligible], dtype=np.float64)
    losses = np.asarray([row["validation_loss"] for row in eligible], dtype=np.float64)
    theta, rmse = fit_law(n, d, losses)
    fitted_losses = predict(theta, n, d)
    residual_values = fitted_losses - losses
    loss_variance = float(np.sum((losses - losses.mean()) ** 2))
    r_squared = (1.0 - float(np.sum(residual_values ** 2)) / loss_variance
                 if loss_variance > 0 else float("nan"))
    normalized_rmse = rmse / float(losses.mean())
    ln_rows = [row for row in eligible if row["run_name"].startswith("e1_ln_")]
    ld_rows = [row for row in eligible if row["run_name"].startswith("e1_ld_300m_")]
    if len(ln_rows) != 5 or len(ld_rows) != 5:
        return {"schema_version": 1, "status": "FAIL",
                "reason": "controlled_axis_roster_is_not_5_plus_5", "runs": rows}
    ln_fit, ln_rmse = fit_axis(
        np.asarray([row["n_params"] for row in ln_rows], dtype=np.float64),
        np.asarray([row["validation_loss"] for row in ln_rows], dtype=np.float64),
    )
    ld_fit, ld_rmse = fit_axis(
        np.asarray([row["tokens"] for row in ld_rows], dtype=np.float64),
        np.asarray([row["validation_loss"] for row in ld_rows], dtype=np.float64),
    )
    rng = np.random.default_rng(20261021)
    boot = []
    for _ in range(bootstrap_samples):
        indices = rng.integers(0, len(n), len(n))
        if len(np.unique(n[indices])) < 3 or len(np.unique(d[indices])) < 3:
            continue
        try:
            fitted, _ = fit_law(n[indices], d[indices], losses[indices])
            boot.append(fitted)
        except ValueError:
            continue
    if len(boot) < max(30, bootstrap_samples // 4):
        return {"schema_version": 1, "status": "FAIL", "reason": "insufficient_bootstrap_fits",
                "successful_bootstrap_fits": len(boot), "runs": rows}
    boot_array = np.asarray(boot)
    names = ("floor", "log_a", "alpha", "log_b", "beta")
    parameters = {
        name: {"estimate": float(theta[index]),
               "ci95": [float(x) for x in np.percentile(boot_array[:, index], [2.5, 97.5])]}
        for index, name in enumerate(names)
    }
    throughput_r_squared = float("nan")
    throughput_boot = []
    if not loss_only_mode:
        throughput_rows = performance["runs"] if performance["status"] == "PASS" else eligible
        throughput_n = np.asarray([row["n_params"] for row in throughput_rows], dtype=np.float64)
        log_n = np.log(throughput_n)
        log_tps = np.log(np.asarray([row["median_tps_per_device"] for row in throughput_rows]))
        slope, intercept = np.polyfit(log_n, log_tps, 1)
        fitted_log_tps = intercept + slope * log_n
        throughput_residual_sum = float(np.sum((fitted_log_tps - log_tps) ** 2))
        throughput_variance = float(np.sum((log_tps - log_tps.mean()) ** 2))
        throughput_r_squared = (1.0 - throughput_residual_sum / throughput_variance
                                if throughput_variance > 0 else float("nan"))
        for _ in range(bootstrap_samples):
            indices = rng.integers(0, len(throughput_n), len(throughput_n))
            if len(np.unique(throughput_n[indices])) < 3:
                continue
            boot_slope, boot_intercept = np.polyfit(log_n[indices], log_tps[indices], 1)
            throughput_boot.append((float(boot_intercept), float(boot_slope)))
        throughput_boot_array = np.asarray(throughput_boot)

    leave_one_out = []
    for held_out in range(len(n)):
        mask = np.arange(len(n)) != held_out
        try:
            fitted, _ = fit_law(n[mask], d[mask], losses[mask])
            prediction = float(predict(fitted, n[held_out:held_out + 1],
                                       d[held_out:held_out + 1])[0])
            relative_error = abs(prediction - float(losses[held_out])) / float(losses[held_out])
            leave_one_out.append({"run_name": eligible[held_out]["run_name"],
                                  "prediction": prediction,
                                  "observed": float(losses[held_out]),
                                  "relative_error": relative_error})
        except ValueError:
            continue
    if (len(leave_one_out) != len(eligible)
            or (not loss_only_mode
                and len(throughput_boot) < max(30, bootstrap_samples // 4))):
        return {"schema_version": 1, "status": "FAIL",
                "reason": "insufficient_diagnostic_fits", "runs": rows,
                "leave_one_out_successful": len(leave_one_out),
                "throughput_bootstrap_successful": len(throughput_boot)}
    target_tokens = 1_000_000_000_000
    # E1 arms are single-GPU. Do not silently assume perfect FSDP scaling when
    # turning their throughput into a four-GPU schedule commitment. The final
    # sustained production canary may replace this conservative provisioning
    # factor with measured evidence before pretrain launch.
    distributed_efficiency = 0.70
    candidates = []
    for row in sorted(ln_rows, key=lambda value: value["n_params"]):
        predicted_loss = float(predict(
            theta, np.asarray([row["n_params"]], dtype=np.float64),
            np.asarray([target_tokens], dtype=np.float64),
        )[0])
        loss_distribution = (
            boot_array[:, 0]
            + np.exp(boot_array[:, 1] - boot_array[:, 2] * np.log(row["n_params"]))
            + np.exp(boot_array[:, 3] - boot_array[:, 4] * np.log(target_tokens))
        )
        candidate = {"source_run": row["run_name"], "n_params": row["n_params"],
                           "predicted_loss_at_1t": predicted_loss,
                           "predicted_loss_at_1t_ci95": [float(value) for value in
                                                          np.percentile(loss_distribution, [2.5, 97.5])]}
        if not loss_only_mode:
            predicted_tps = float(np.exp(intercept + slope * np.log(row["n_params"])))
            throughput_distribution = np.exp(
                throughput_boot_array[:, 0]
                + throughput_boot_array[:, 1] * np.log(row["n_params"])
            )
            days_distribution = target_tokens / (
                4 * distributed_efficiency * throughput_distribution * 86400
            )
            days_ci = [float(value) for value in np.percentile(days_distribution, [2.5, 97.5])]
            candidate.update({
                "predicted_tps_per_device": predicted_tps,
                "predicted_days_4gpu": target_tokens / (
                    4 * distributed_efficiency * predicted_tps * 86400
                ),
                "predicted_days_4gpu_ci95": days_ci,
                "feasibility_probability": float(np.mean(days_distribution <= 45.0)),
                "feasible_within_45_days": days_ci[1] <= 45.0,
            })
        candidates.append(candidate)
    if loss_only_mode:
        recommendation = min(candidates, key=lambda row: row["predicted_loss_at_1t"])
    else:
        feasible = [row for row in candidates if row["feasible_within_45_days"]]
        recommendation = (min(feasible, key=lambda row: row["predicted_loss_at_1t"])
                          if feasible else min(candidates, key=lambda row: row["predicted_days_4gpu"]))
    loo_median = float(np.median([row["relative_error"] for row in leave_one_out]))
    loo_p95 = float(np.percentile([row["relative_error"] for row in leave_one_out], 95))
    quality_thresholds = {
        "minimum_loss_r_squared": 0.90,
        "maximum_normalized_rmse": 0.03,
        "maximum_leave_one_out_median_relative_error": 0.05,
        "maximum_leave_one_out_p95_relative_error": 0.10,
        "minimum_throughput_log_r_squared": 0.75,
    }
    quality_checks = {
        "loss_r_squared": math.isfinite(r_squared)
        and r_squared >= quality_thresholds["minimum_loss_r_squared"],
        "normalized_rmse": math.isfinite(normalized_rmse)
        and normalized_rmse <= quality_thresholds["maximum_normalized_rmse"],
        "leave_one_out_median": loo_median
        <= quality_thresholds["maximum_leave_one_out_median_relative_error"],
        "leave_one_out_p95": loo_p95
        <= quality_thresholds["maximum_leave_one_out_p95_relative_error"],
    }
    if not loss_only_mode:
        quality_checks["throughput_log_r_squared"] = (
            math.isfinite(throughput_r_squared)
            and throughput_r_squared >= quality_thresholds["minimum_throughput_log_r_squared"]
        )
    status = (("PASS_PROVISIONAL" if loss_only_mode else "PASS")
              if all(quality_checks.values()) else "FAIL")
    return {"schema_version": 1, "status": status,
            "failure_reason": None if status in {"PASS", "PASS_PROVISIONAL"}
            else "scaling_fit_quality_gate",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "expected_results": len(expected), "observed_results": len(observed),
            "model": "loss=E+A/N^alpha+B/D^beta", "parameters": parameters,
            "rmse": rmse, "bootstrap_requested": bootstrap_samples,
            "bootstrap_successful": len(boot), "throughput_fit": ({
                "formula": "tps_per_device=exp(intercept)*N^slope",
                "intercept": float(intercept), "slope": float(slope),
                "log_r_squared": throughput_r_squared,
                "bootstrap_successful": len(throughput_boot),
            } if not loss_only_mode else {
                "status": "EXCLUDED_CONTENDED_RESOURCE", "bootstrap_successful": 0,
            }), "controlled_axis_fits": {
                "L_N_fixed_D": {"parameters": ln_fit, "rmse": ln_rmse, "points": 5,
                                "configured_budget_tokens": ln_rows[0]['configured_budget_tokens']},
                "L_D_fixed_300M": {"parameters": ld_fit, "rmse": ld_rmse, "points": 5},
            }, "production_candidates": candidates,
            "fit_diagnostics": {
                "r_squared": r_squared, "normalized_rmse": normalized_rmse,
                "residuals": [
                    {"run_name": row["run_name"], "residual": float(residual),
                     "predicted": float(predicted), "observed": float(observed)}
                    for row, residual, predicted, observed in zip(
                        eligible, residual_values, fitted_losses, losses, strict=True)
                ],
                "leave_one_out": leave_one_out,
                "leave_one_out_median_relative_error": loo_median,
                "leave_one_out_p95_relative_error": loo_p95,
            },
            "fit_quality_gate": {"status": "PASS" if all(quality_checks.values()) else "FAIL",
                                 "thresholds": quality_thresholds, "checks": quality_checks},
            "production_recommendation": recommendation,
            "performance_evidence_status": ("EXCLUDED_CONTENDED_RESOURCE"
                                            if loss_only_mode else "MEASURED"),
            "independent_performance": performance,
            "production_policy": {"target_tokens": target_tokens, "gpu_count": 4,
                                  "provisional_distributed_efficiency": distributed_efficiency,
                                  "maximum_days": 45,
                                  "selection": ("lowest predicted loss; performance feasibility deferred"
                                                if loss_only_mode else
                                                "lowest predicted loss among conservatively throughput-feasible scales"),
                                  "final_requirement": "replace provisional efficiency with sustained four-GPU canary evidence before launch"},
            "runs": sorted(rows, key=lambda row: row["run_name"]), "rejected": rejected}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()
    result = analyze(args.root.resolve(), args.results_root, args.bootstrap_samples)
    result["analysis_tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "observed_results": result.get("observed_results", len(result.get("runs", []))),
                      "rmse": result.get("rmse")}, sort_keys=True))
    return 0 if result["status"] in {"PASS", "PASS_PROVISIONAL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
