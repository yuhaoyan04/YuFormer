#!/usr/bin/env python3
"""Rank the exact E2 architecture roster from nine-domain result contracts."""
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


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pretrain/scripts"))
from performance_evidence import promote_bound_loss_analysis


ARCHITECTURES = {
    "e2_l16", "e2_l32", "e2_l48", "e2_ffn2x", "e2_ffn4x",
    "e2_nogate", "e2_noqkn", "e2_fullrope", "e2_swaoff", "e2_untied",
}
EXPECTED = {
    architecture if seed == 1 else f"{architecture}_s{seed}"
    for architecture in ARCHITECTURES for seed in (1, 2, 3)
}
DOMAINS = {
    "academic", "books", "chinese", "code", "encyclopedia",
    "math", "multilingual", "qa", "web",
}


def analysis_status_ready(value: dict[str, Any]) -> bool:
    return value.get("status") in {"PASS", "PASS_PROVISIONAL"}


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
    head_dim, kv_width = h // heads, kv_heads * (h // heads)
    full = sum((index + 1) % int(preset["attention_every"]) == 0 for index in range(layers))
    gated = layers - full if preset.get("attention_output_gate", False) else 0
    norms = 2 * layers * h + h + (2 * layers * head_dim if preset.get("qk_norm", False) else 0)
    return (layers * (h * (h + 2 * kv_width) + h * h) + gated * h * h
            + layers * 3 * h * ffn + norms)


def flops_per_token(preset: dict[str, Any], seq_len: int) -> float:
    layers = int(preset["num_layers"])
    full = sum((index + 1) % int(preset["attention_every"]) == 0 for index in range(layers))
    sliding = layers - full
    attention_context = full * seq_len + sliding * int(preset["window_size"])
    # Dense transformer weights and the vocabulary projection each contribute
    # roughly 6 FLOPs/parameter/token in training (forward + two backward
    # matmuls). Embedding lookup itself is not counted as a dense matmul.
    return (6.0 * non_embedding_params(preset)
            + 6.0 * int(preset["vocab_size"]) * int(preset["hidden_size"])
            + 12.0 * int(preset["hidden_size"]) * attention_context)


def scaling_loss(parameters: dict[str, Any], n_params: int, tokens: int) -> float:
    values = {key: float(parameters[key]["estimate"])
              for key in ("floor", "log_a", "alpha", "log_b", "beta")}
    return (values["floor"] + math.exp(values["log_a"] - values["alpha"] * math.log(n_params))
            + math.exp(values["log_b"] - values["beta"] * math.log(tokens)))


def architecture_name(run_name: str) -> str:
    return run_name[:-3] if run_name.endswith(("_s2", "_s3")) else run_name


def analyze(results_root: Path, root: Path = ROOT,
            scaling: dict[str, Any] | None = None) -> dict[str, Any]:
    configs = {path.stem: yaml.safe_load(path.read_text(encoding="utf-8"))
               for path in (root / "pretrain/configs/e2").glob("*.yaml")}
    presets = yaml.safe_load((root / "model/configs/model_configs.yaml").read_text(
        encoding="utf-8"))["presets"]
    rows, rejected = [], []
    observed = set()
    for path in sorted(results_root.glob("*/result.json")):
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            rejected.append({"path": str(path), "reason": f"{type(exc).__name__}:{exc}"})
            continue
        name = str(result.get("run_name", ""))
        if name not in EXPECTED:
            rejected.append({"path": str(path), "reason": "unexpected_run_name"})
            continue
        observed.add(name)
        metrics = result.get("metrics", {})
        loss_only = bool(result.get("execution_context", {}).get("loss_only"))
        aggregate = metrics.get("validation_metrics/loss", {}).get("last")
        domain_losses = {
            domain: metrics.get(f"validation_metrics/domain/{domain}/loss", {}).get("last")
            for domain in sorted(DOMAINS)
        }
        issues = []
        if result.get("status") != "PASS":
            issues.append("result_not_pass")
        if (not isinstance(aggregate, (int, float)) or not math.isfinite(float(aggregate))
                or float(aggregate) <= 0):
            issues.append("aggregate_validation_loss_missing")
        missing_domains = [key for key, value in domain_losses.items()
                           if (not isinstance(value, (int, float))
                               or not math.isfinite(float(value)) or float(value) <= 0)]
        if missing_domains:
            issues.append("missing_domains:" + ",".join(missing_domains))
        if not result.get("identity_sha256"):
            issues.append("identity_sha256_missing")
        cfg = configs.get(name)
        if cfg is None:
            issues.append("config_missing")
            n_params, budget, trained_tokens = 0, 0, 0
        else:
            preset = presets[cfg["model"]["preset"]]
            n_params = non_embedding_params(preset)
            budget = int(cfg["train"]["budget_tokens"])
            model_flops_per_token = flops_per_token(preset, int(cfg["data"]["seq_len"]))
            expected_tps = (int(cfg["data"]["seq_len"])
                            * int(cfg["train"]["micro_batch_size"])
                            * int(cfg["train"]["grad_accum"])
                            * int(cfg["parallel"]["world_size"]))
            expected_steps = int(cfg["train"]["expected_iters"])
            trained_tokens = expected_steps * expected_tps
            hyper = result.get("hyperparameters") or {}
            if hyper.get("model_preset") != cfg["model"]["preset"]:
                issues.append("model_preset_mismatch")
            if hyper.get("budget_tokens") != budget:
                issues.append("budget_tokens_mismatch")
            if hyper.get("tokens_per_step") != expected_tps:
                issues.append("tokens_per_step_mismatch")
            if hyper.get("trained_steps") != expected_steps:
                issues.append("trained_steps_mismatch")
            if hyper.get("trained_tokens") != trained_tokens:
                issues.append("trained_tokens_mismatch")
        throughput = result.get("metrics", {}).get("throughput(tps)", {}).get("last")
        if (not loss_only and (not isinstance(throughput, (int, float))
                              or not math.isfinite(float(throughput)) or throughput <= 0)):
            issues.append("throughput_missing_or_invalid")
        rows.append({"run_name": name, "result": str(path), "status": "PASS" if not issues else "FAIL",
                     "issues": issues, "validation_loss": aggregate, "domain_losses": domain_losses,
                     "identity_sha256": result.get("identity_sha256"),
                     "n_params": n_params, "budget_tokens": budget,
                     "trained_tokens": trained_tokens,
                     "model_flops_per_token": model_flops_per_token if cfg is not None else None,
                     "training_flops": (model_flops_per_token * trained_tokens
                                        if cfg is not None else None),
                     "throughput_tps_per_device": throughput,
                     "loss_only": loss_only,
                     "performance_metrics_excluded": (
                         ["throughput(tps)", "mfu(%)", "wall_time"] if loss_only else []
                     )})
    missing = sorted(EXPECTED - observed)
    eligible = [row for row in rows if row["status"] == "PASS"]
    if missing or len(eligible) != len(EXPECTED):
        status = "BLOCKED"
    else:
        status = "PENDING_SCALING_FIT"
    if scaling is None:
        try:
            scaling = json.loads((root / "experiments/e1_scaling/results/scaling_fit.json").read_text(
                encoding="utf-8"))
        except (OSError, ValueError):
            scaling = {}
    grouped = []
    for architecture in sorted(ARCHITECTURES):
        members = sorted((row for row in eligible
                          if architecture_name(row["run_name"]) == architecture),
                         key=lambda row: row["run_name"])
        if len(members) != 3:
            continue
        parameter_values = {row["n_params"] for row in members}
        budget_values = {row["trained_tokens"] for row in members}
        if len(parameter_values) != 1 or len(budget_values) != 1:
            continue
        losses = [float(row["validation_loss"]) for row in members]
        throughputs = [float(row["throughput_tps_per_device"]) for row in members
                       if isinstance(row["throughput_tps_per_device"], (int, float))]
        grouped.append({
            "run_name": architecture,
            "seed_runs": [row["run_name"] for row in members],
            "identity_sha256": [row["identity_sha256"] for row in members],
            "n_params": members[0]["n_params"], "budget_tokens": members[0]["budget_tokens"],
            "trained_tokens": members[0]["trained_tokens"],
            "model_flops_per_token": members[0]["model_flops_per_token"],
            "training_flops": members[0]["training_flops"],
            "validation_loss": float(np.mean(losses)),
            "validation_loss_std": float(np.std(losses, ddof=1)),
            "seed_validation_losses": losses,
            "domain_losses": {
                domain: float(np.mean([float(row["domain_losses"][domain]) for row in members]))
                for domain in sorted(DOMAINS)
            },
            "domain_loss_std": {
                domain: float(np.std([float(row["domain_losses"][domain]) for row in members], ddof=1))
                for domain in sorted(DOMAINS)
            },
            "throughput_tps_per_device": (float(np.mean(throughputs))
                                           if len(throughputs) == len(members) else None),
            "throughput_tps_std": (float(np.std(throughputs, ddof=1))
                                   if len(throughputs) == len(members) else None),
        })
    if status == "PENDING_SCALING_FIT" and len(grouped) != len(ARCHITECTURES):
        status = "BLOCKED"

    reference_n = non_embedding_params(presets[configs["e2_l32"]["model"]["preset"]])
    if status == "PENDING_SCALING_FIT" and analysis_status_ready(scaling):
        parameters = scaling["parameters"]
        for row in grouped:
            expected_at_arm = scaling_loss(parameters, row["n_params"], row["trained_tokens"])
            expected_at_reference = scaling_loss(parameters, reference_n, row["trained_tokens"])
            row["parameter_loss_adjustment"] = expected_at_arm - expected_at_reference
            row["normalized_validation_loss"] = (
                float(row["validation_loss"]) - row["parameter_loss_adjustment"]
            )
            row["seed_normalized_validation_losses"] = [
                value - row["parameter_loss_adjustment"]
                for value in row["seed_validation_losses"]
            ]
        grouped.sort(key=lambda row: (row["normalized_validation_loss"], row["run_name"]))
        status = ("PASS_PROVISIONAL"
                  if scaling.get("status") == "PASS_PROVISIONAL"
                  or any(row["loss_only"] for row in eligible) else "PASS")
    else:
        grouped.sort(key=lambda row: (float(row["validation_loss"]), row["run_name"]))
    selection = None
    comparisons = []
    indistinguishable = []
    if status in {"PASS", "PASS_PROVISIONAL"}:
        empirical_best = grouped[0]
        best_losses = np.asarray(empirical_best["seed_normalized_validation_losses"])
        for row in grouped:
            differences = np.asarray(row["seed_normalized_validation_losses"]) - best_losses
            mean_difference = float(np.mean(differences))
            half_width = (4.302652729911275 * float(np.std(differences, ddof=1))
                          / math.sqrt(len(differences)))
            interval = [mean_difference - half_width, mean_difference + half_width]
            resolved_worse = interval[0] > 1e-12
            comparisons.append({
                "run_name": row["run_name"], "against": empirical_best["run_name"],
                "paired_mean_normalized_loss_delta": mean_difference,
                "paired_delta_ci95": interval, "statistically_resolved_worse": resolved_worse,
            })
            if not resolved_worse:
                indistinguishable.append(row)
        selected = min(
            indistinguishable,
            key=lambda row: (float(row["model_flops_per_token"]),
                             float(row["normalized_validation_loss"]), row["run_name"]),
        )
        selection = {
            "run_name": selected["run_name"],
            "empirical_best_run_name": empirical_best["run_name"],
            "validation_loss": selected["validation_loss"],
            "validation_loss_std": selected["validation_loss_std"],
            "normalized_validation_loss": selected["normalized_validation_loss"],
            "identity_sha256": selected["identity_sha256"],
            "seed_runs": selected["seed_runs"],
            "selection_policy": "lowest estimated FLOPs among architectures not resolved worse than empirical best",
        }
    return {
        "schema_version": 1, "status": status,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "expected_results": len(EXPECTED), "observed_results": len(observed),
        "missing": missing, "rejected": rejected, "runs": rows, "ranking": grouped,
        "reference_non_embedding_params": reference_n,
        "selection_rule": ("paired-seed 95% CI, then lowest estimated FLOPs within the "
                           "statistically indistinguishable set"),
        "paired_comparisons_to_empirical_best": comparisons,
        "statistically_indistinguishable": [row["run_name"] for row in indistinguishable],
        "selection": selection,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.results_root, args.root.resolve())
    result["analysis_tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = promote_bound_loss_analysis(args.root.resolve(), "e2", result)
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "observed_results": result["observed_results"],
                      "selection": result["selection"]}, sort_keys=True))
    return 0 if result["status"] in {"PASS", "PASS_PROVISIONAL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
