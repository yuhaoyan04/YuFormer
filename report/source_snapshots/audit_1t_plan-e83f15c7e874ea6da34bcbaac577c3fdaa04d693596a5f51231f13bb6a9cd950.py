#!/usr/bin/env python3
"""Static capacity and accounting audit for the 1T WSD/context plan.

This audit is intentionally usable before the three large S6 releases exist.
It proves that requested domain quotas fit the governed source pool, that strict
top-10 stages fit the measured top-10 candidate ledger, and that pretrain and
context budgets close.  Final-tokenizer stream closure remains BLOCKED until
the corresponding immutable token-stream metadata has been published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        value = yaml.safe_load(fh) if path.suffix in {".yaml", ".yml"} else json.load(fh)
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quota(total: int, weights: dict[str, float]) -> dict[str, int]:
    exact = {key: total * value / sum(weights.values()) for key, value in weights.items()}
    result = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(result.values())
    order = sorted(exact, key=lambda key: (-(exact[key] - result[key]), key))
    for key in order[:remainder]:
        result[key] += 1
    return result


def audit(root: Path, plan_path: Path, s6_path: Path) -> dict[str, Any]:
    plan = _load(plan_path)
    s6 = _load(s6_path)
    baseline_manifest_path = root / "data/processed/mix_order/mix_m3_e1_100b.manifest.json"
    baseline_manifest = _load(baseline_manifest_path)
    observed_effective_top_k = baseline_manifest.get("effective_top_k") or {}
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    tokenizer_gate = _load(root / plan["tokenizer_gate"])
    data_gate = _load(root / plan["data_gate"])
    checks["tokenizer_gate_pass"] = tokenizer_gate.get("status") == "PASS"
    checks["data_gate_pass"] = data_gate.get("status") == "PASS"
    if not checks["tokenizer_gate_pass"]:
        errors.append("tokenizer gate is not PASS")
    if not checks["data_gate_pass"]:
        errors.append("data gate is not PASS")

    pretrain = plan["pretrain"]
    stages = pretrain["stages"]
    target = int(pretrain["target_tokens"])
    stage_total = sum(int(stage["target_tokens"]) for stage in stages)
    fraction_total = sum(float(stage["fraction"]) for stage in stages)
    checks["pretrain_budget"] = {
        "target_tokens": target,
        "stage_tokens": stage_total,
        "fraction_sum": fraction_total,
        "closed": stage_total == target and abs(fraction_total - 1.0) < 1e-12,
    }
    if not checks["pretrain_budget"]["closed"]:
        errors.append("pretrain stage budgets/fractions do not close")

    train_contract_path = root / "pretrain/configs/pretrain_base.yaml"
    train_contract = _load(train_contract_path)
    train_data, train_optim, train_loop = (
        train_contract["data"], train_contract["optim"], train_contract["train"]
    )
    limits = [int(value) for value in train_data.get("stage_sample_limits", [])]
    samples_per_step = (int(train_loop["micro_batch_size"]) * int(train_loop["grad_accum"])
                        * int(train_contract["parallel"]["world_size"]))
    seq_len = int(train_data["seq_len"])
    aligned = (len(limits) == len(stages)
               and all(value > 0 and value % samples_per_step == 0 for value in limits))
    consumed = [value * seq_len for value in limits] if aligned else []
    boundary_close = (aligned and all(abs(actual - int(stage["target_tokens"])) < seq_len * samples_per_step
                                      for actual, stage in zip(consumed, stages, strict=True)))
    schedule_close = (aligned
                      and int(train_optim["warmup_steps"]) == limits[0] // samples_per_step
                      and int(train_optim["decay_steps"]) == limits[-1] // samples_per_step
                      and sum(limits) == int(train_loop["expected_iters"]) * samples_per_step)
    checks["optimizer_aligned_stage_boundaries"] = {
        "stage_sample_limits": limits, "samples_per_optimizer_step": samples_per_step,
        "tokens_per_optimizer_step": samples_per_step * seq_len,
        "consumed_tokens": consumed, "aligned": aligned,
        "within_one_optimizer_step_of_nominal": boundary_close,
        "lr_schedule_boundaries_match": schedule_close,
        "config": str(train_contract_path.relative_to(root)),
        "config_sha256": _sha256(train_contract_path),
    }
    if not aligned or not boundary_close or not schedule_close:
        errors.append("pretrain stage sample limits do not close against WSD/LR boundaries")

    contexts = {int(stage["context_length"]) for stage in stages}
    checks["pretrain_context_isolated"] = contexts == {int(pretrain["context_length"])}
    if not checks["pretrain_context_isolated"]:
        errors.append("context length changes inside the WSD data/LR curriculum")

    formal_domains = s6.get("per_domain") or {}
    roster = set(s6.get("included_sources") or [])
    stage_reports: list[dict[str, Any]] = []
    for stage in stages:
        mix_path = root / stage["mix_config"]
        mix = _load(mix_path)
        total = int(mix["total_tokens"])
        final_target = int(mix.get("target_final_tokens", total))
        if final_target != int(stage["target_tokens"]):
            errors.append(f"{stage['name']}: plan final-token target differs from mix target")
        weights = {domain: float(spec["weight"]) for domain, spec in mix["domains"].items()}
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 1e-12:
            errors.append(f"{stage['name']}: domain weights sum to {weight_sum}")
        mix_roster = set(mix.get("included_sources") or [])
        if mix_roster != roster:
            errors.append(f"{stage['name']}: included source roster differs from formal S6")
        quotas = _quota(total, weights)
        domain_report: dict[str, Any] = {}
        for domain, requested in quotas.items():
            ledger = formal_domains.get(domain)
            if ledger is None:
                errors.append(f"{stage['name']}: domain absent from S6 ledger: {domain}")
                continue
            pool = int(ledger["pool_tokens"]) - int(ledger.get("val_tokens", 0))
            observed = (baseline_manifest.get("selected") or {}).get(domain, {})
            if "candidate_tokens" not in observed:
                errors.append(f"{stage['name']}/{domain}: baseline candidate ledger is missing")
                continue
            candidate = int(observed["candidate_tokens"])
            candidate_is_exact_top10 = abs(
                float(observed_effective_top_k.get(domain, -1.0)) - 0.10
            ) < 1e-12
            top_k = float(mix["domains"][domain].get("top_k", 1.0))
            nominal_preferred_capacity = int(top_k * int(ledger["pool_tokens"]))
            fits_pool = requested <= pool
            fits_nominal_preferred_band = requested <= nominal_preferred_capacity
            fits_measured_top10 = None
            if abs(top_k - 0.10) < 1e-12 and candidate_is_exact_top10:
                fits_measured_top10 = requested <= candidate
            if not fits_pool:
                errors.append(f"{stage['name']}/{domain}: quota exceeds governed pool")
            if (
                mix.get("quota_fill_policy", "strict_top_k") == "deepen_to_quota"
                and not fits_nominal_preferred_band
            ):
                warnings.append(
                    f"{stage['name']}/{domain}: materializer will scan the full pool as deep-fill fallback"
                )
            if fits_measured_top10 is False:
                errors.append(f"{stage['name']}/{domain}: quota exceeds measured top-10% ledger")
            domain_report[domain] = {
                "quota_tokens": requested,
                "available_pool_tokens_after_val": pool,
                "observed_candidate_tokens": candidate,
                "observed_candidate_is_exact_top10": candidate_is_exact_top10,
                "nominal_preferred_capacity_tokens": nominal_preferred_capacity,
                "requested_top_k": top_k,
                "fits_pool": fits_pool,
                "fits_nominal_preferred_band": fits_nominal_preferred_band,
                "fits_measured_top10": fits_measured_top10,
            }
        stage_reports.append({
            "name": stage["name"],
            "mix_config": str(stage["mix_config"]),
            "mix_config_sha256": _sha256(mix_path),
            "target_final_tokens": final_target,
            "selection_reference_tokens": total,
            "domains": domain_report,
        })

    context = plan["context_scaling"]
    expected_tpu = int(context["constant_tokens_per_update"])
    context_arms = {
        "progressive": context["ladder_config"],
        "direct": context["experiment"]["direct_config"],
    }
    context_reports: dict[str, Any] = {}
    for arm, relative_path in context_arms.items():
        arm_path = root / relative_path
        arm_config = _load(arm_path)
        arm_tokens = sum(int(stage["budget_tokens"]) for stage in arm_config["stages"])
        rows = []
        for stage in arm_config["stages"]:
            actual_tpu = (
                int(stage["seq_len"])
                * int(stage["world_size"])
                * int(stage["micro_batch_size"])
                * int(stage["grad_accum"])
            )
            ok = actual_tpu == expected_tpu == int(stage["tokens_per_update"])
            if not ok:
                errors.append(f"{arm}/{stage['name']}: tokens/update does not close")
            if stage.get("init") != "previous_stage_weights_only":
                errors.append(f"{arm}/{stage['name']}: context transition is not weights-only")
            rows.append({"name": stage["name"], "tokens_per_update": actual_tpu, "closed": ok})
        if arm_tokens != int(context["total_extension_tokens"]):
            errors.append(f"{arm}: context extension stage budgets do not close")
        context_reports[arm] = {
            "config": relative_path,
            "config_sha256": _sha256(arm_path),
            "extension_tokens": arm_tokens,
            "ladder": rows,
        }
    if context.get("budget_included_in_pretrain_1t") is not False:
        errors.append("context extension must be accounted outside the 1T base budget")
    checks["context_scaling"] = {
        "expected_extension_tokens": int(context["total_extension_tokens"]),
        "matched_token_arms": context_reports,
    }

    # Static planning can prove capacity, but actual-token closure needs the
    # three final streams.  Keep this explicit rather than silently equating S5
    # reference token counts with the selected tokenizer's token counts.
    warnings.append(
        "final-tokenizer stage closure remains pending until warmup/stable/anneal stream metas exist"
    )
    inputs = {
        "plan": str(plan_path.relative_to(root)),
        "plan_sha256": _sha256(plan_path),
        "s6_ledger": str(s6_path.relative_to(root)),
        "s6_ledger_sha256": _sha256(s6_path),
        "baseline_mix_manifest": str(baseline_manifest_path.relative_to(root)),
        "baseline_mix_manifest_sha256": _sha256(baseline_manifest_path),
        "dataset_id": data_gate.get("dataset_id"),
        "tokenizer_id": tokenizer_gate.get("tokenizer_id"),
    }
    frozen_path = root / "experiments/e4_curriculum/results/frozen_1t_recipe.json"
    if frozen_path.is_file():
        frozen = _load(frozen_path)
        if frozen.get("status") == "PASS":
            freeze_tool = root / "experiments/e4_curriculum/scripts/freeze_1t_recipe.py"
            if frozen.get("freeze_tool_sha256") != _sha256(freeze_tool):
                errors.append("frozen recipe uses a stale freeze tool")
            inputs["frozen_recipe"] = str(frozen_path.relative_to(root))
            inputs["frozen_recipe_sha256"] = _sha256(frozen_path)
    return {
        "schema_version": 1,
        "audit_tool_sha256": _sha256(Path(__file__)),
        "plan_id": plan["plan_id"],
        "status": "PASS_STATIC" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "inputs": inputs,
        "checks": checks,
        "stages": stage_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument(
        "--plan", type=Path,
        default=Path("experiments/e4_curriculum/configs/e4_1t_wsd_context.yaml"),
    )
    parser.add_argument(
        "--s6-ledger", type=Path, default=Path("data/processed/stage_s6_global_summary.json")
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    plan = args.plan if args.plan.is_absolute() else root / args.plan
    ledger = args.s6_ledger if args.s6_ledger.is_absolute() else root / args.s6_ledger
    report = audit(root, plan, ledger)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    raise SystemExit(0 if report["status"] == "PASS_STATIC" else 1)


if __name__ == "__main__":
    main()
