#!/usr/bin/env python3
"""Freeze model, LR, and calibrated 50B/800B/150B recipes from E1/E2/E3."""
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
sys.path.insert(0, str(ROOT / 'pretrain/scripts'))
from research_scope import load_scope, POLICY as SCOPE_POLICY
from training_tokenizer_calibration import load_training_ratios, REPORT as TRAINING_CALIBRATION_REPORT
DOMAINS = ("web", "code", "math", "chinese", "academic", "books",
           "encyclopedia", "qa", "multilingual")
STAGES = {
    "warmup": ("data-process/configs/mix_warmup_50b.yaml", 50_000_000_000,
               {"web": .811, "code": .667, "math": 1.25, "chinese": 2.286,
                "academic": 1.714, "books": 10.0, "encyclopedia": 10.0,
                "qa": 15.0, "multilingual": 1.25}),
    "stable": ("data-process/configs/mix_stable_800b.yaml", 800_000_000_000,
               {domain: 1.0 for domain in DOMAINS}),
    "anneal": ("data-process/configs/mix_anneal.yaml", 150_000_000_000,
               {"web": .800, "code": 1.167, "math": 2.0, "chinese": 1.714,
                "academic": 1.429, "books": 6.25, "encyclopedia": 4.0,
                "qa": 7.0, "multilingual": 1.0}),
}
USER_OVERRIDE = "data/health/pretrain-user-full-pretrain-authorization.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_json_bound(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value, sha256_bytes(raw)


def verify_captured_inputs(root: Path, inputs: dict[str, str]) -> None:
    for relative, expected in inputs.items():
        path = root / relative
        if not path.is_file() or sha256_bytes(path.read_bytes()) != expected:
            raise ValueError(f"freeze input changed during generation: {relative}")


def verify_identity_audit_snapshot(root: Path, audit: dict[str, Any]) -> None:
    """Reject a PASS audit if any evidence it covered has since changed."""
    snapshot = audit.get("evidence_snapshot")
    if not isinstance(snapshot, dict) or not snapshot:
        raise ValueError("experiment identity audit lacks an evidence snapshot")
    if audit.get("evidence_snapshot_count") != len(snapshot):
        raise ValueError("experiment identity audit snapshot count is inconsistent")
    for recorded_path, expected_sha in snapshot.items():
        if not isinstance(recorded_path, str) or not isinstance(expected_sha, str):
            raise ValueError("experiment identity audit snapshot is malformed")
        path = Path(recorded_path)
        if not path.is_absolute():
            path = root / path
        if not path.is_file() or sha256_bytes(path.read_bytes()) != expected_sha:
            raise ValueError(f"stale experiment identity evidence: {recorded_path}")


def source_sha256(root: Path) -> str:
    h = hashlib.sha256()
    roots = [root / "model/src", root / "pretrain/backends/torchtitan"]
    for path in sorted(p for base in roots for p in base.rglob("*")
                       if p.is_file() and "__pycache__" not in p.parts):
        h.update(str(path.relative_to(root)).encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
    for relative in ("pretrain/scripts/run_with_lease.py", "pretrain/scripts/assert_run_lease.py",
                     "pretrain/src/yuformer_pretrain/lease.py"):
        path = root / relative
        if path.is_file():
            h.update(relative.encode("utf-8"))
            h.update(path.read_bytes())
    return h.hexdigest()


def identity_audit_source_drift_only(audit: dict[str, Any]) -> bool:
    rows = audit.get("rows")
    if (audit.get("status") != "BLOCKED" or audit.get("roster_exact") is not True
            or not isinstance(rows, list) or len(rows) != int(audit.get("total", -1))
            or not rows):
        return False
    return all(row.get("status") == "FAIL"
               and row.get("errors") == ["training_source_drift"]
               for row in rows)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == value:
        return
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
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


def scale_architecture(source: dict[str, Any], target_params: int) -> dict[str, Any]:
    layers = int(source["num_layers"])
    ffn_ratio = float(source["intermediate_size"]) / int(source["hidden_size"])
    mha = int(source["num_kv_heads"]) == int(source["num_heads"])
    candidates = []
    for hidden in range(256, 3073, 64):
        heads = hidden // 64
        if not mha and heads % 2:
            continue
        candidate = dict(source)
        candidate.update({"hidden_size": hidden, "num_heads": heads,
                          "num_kv_heads": heads if mha else heads // 2,
                          "intermediate_size": max(64, round(hidden * ffn_ratio / 64) * 64)})
        params = non_embedding_params(candidate)
        candidates.append((abs(params - target_params), params, candidate))
    _, actual, selected = min(candidates, key=lambda row: (row[0], row[1]))
    selected["notes"] = (f"Frozen from E2 architecture; target non-embedding params={target_params}, "
                         f"actual={actual}")
    return selected


def normalized(base: dict[str, float], multipliers: dict[str, float]) -> dict[str, float]:
    raw = {domain: float(base[domain]) * float(multipliers[domain]) for domain in DOMAINS}
    total = sum(raw.values())
    return {domain: raw[domain] / total for domain in DOMAINS}


def optimizer_aligned_stage_steps(stage_tokens: list[int], tokens_per_step: int) -> list[int]:
    """Hamilton-apportion optimizer steps while keeping the 1T total floor exact."""
    if tokens_per_step <= 0 or not stage_tokens or any(value <= 0 for value in stage_tokens):
        raise ValueError("stage tokens and tokens_per_step must be positive")
    total_steps = sum(stage_tokens) // tokens_per_step
    steps = [value // tokens_per_step for value in stage_tokens]
    missing = total_steps - sum(steps)
    remainders = sorted(
        range(len(stage_tokens)),
        key=lambda index: (stage_tokens[index] % tokens_per_step, -index),
        reverse=True,
    )
    for index in remainders[:missing]:
        steps[index] += 1
    if sum(steps) != total_steps:
        raise ValueError("optimizer-step apportionment does not close")
    return steps


def resolved_learning_rate(lr_resolution: dict[str, Any], n_params: int) -> tuple[float, float]:
    fit = lr_resolution["fit"]
    factor = float(lr_resolution["batch_lr_factor"])
    if n_params <= 0 or not math.isfinite(factor) or factor <= 0:
        raise ValueError("LR resolution contains an invalid parameter count or batch LR factor")
    value = (math.exp(float(fit["log_intercept"])
                      + float(fit["exponent"]) * math.log(n_params)) * factor)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("resolved production learning rate is invalid")
    return value, factor


def validation_ratios(audit: dict[str, Any]) -> dict[str, float]:
    ratios = {}
    for row in audit.get("streams", []):
        name = str(row.get("name", ""))
        if not name.startswith("val_") or row.get("status") != "PASS":
            continue
        domain = name.removeprefix("val_")
        expected = int(row["expected_reference_tokens"])
        actual_text = int(row["actual_total_tokens_with_eod"]) - int(row["expected_docs"])
        if expected <= 0 or actual_text <= 0:
            raise ValueError(f"invalid tokenizer calibration row: {name}")
        ratios[domain] = actual_text / expected
    if set(ratios) != set(DOMAINS):
        raise ValueError("tokenizer calibration does not contain exactly nine validation domains")
    return ratios


def freeze(root: Path, capacity_headroom: float = 0.02) -> dict[str, Any]:
    if not 0.0 < capacity_headroom <= 0.05:
        raise ValueError("capacity_headroom must be in (0, 0.05]")
    e1_path = root / "experiments/e1_scaling/results/scaling_fit.json"
    e2_path = root / "experiments/e2_architecture/results/selection.json"
    e3_path = root / "experiments/e3_data_mix/results/selection.json"
    lr_path = root / "experiments/e1_scaling/results/lr_resolution.json"
    audit_path = root / "data/processed/audits/e3_token_streams_20261021.json"
    identity_audit_path = root / "report/experiment_identity_audit_latest.json"
    throughput_canary_path = root / "data/health/full-pretrain-optimization-canary.json"
    waiver_path = root / USER_OVERRIDE
    waiver = json.loads(waiver_path.read_text(encoding="utf-8")) if waiver_path.is_file() else {}
    user_override = waiver.get("status") == "APPROVED_BY_USER"
    if user_override and waiver.get("authorization_scope") != "skip_e1_and_launch_full_pretrain":
        raise ValueError("user pretraining override has an invalid authorization scope")
    scope = load_scope(root)
    fixed_baseline = scope['profile'] == 'scaling_only'
    if user_override and not fixed_baseline:
        raise ValueError("user override currently supports the deferred E2/E3 fixed-baseline scope only")
    input_paths = ((e1_path, audit_path, identity_audit_path, throughput_canary_path) if user_override
                   else (e1_path, lr_path, audit_path, identity_audit_path) if fixed_baseline
                   else (e1_path, e2_path, e3_path, lr_path, audit_path, identity_audit_path))
    loaded = [load_json_bound(path) for path in input_paths]
    if user_override:
        e1, stream_audit, identity_audit, throughput_canary = (value for value, _ in loaded)
        lr_resolution = {"status": "SKIPPED_BY_USER"}
        e2 = {'status': 'DEFERRED'}
        e3 = {'status': 'DEFERRED', 'selection': {
            'status': 'FIXED_BASELINE', 'run_name': None,
            'weights': scope['baseline']['mixture_weights'],
            'choice_origin': scope['baseline']['mixture_origin'],
            'comparative_evidence': False}}
    elif fixed_baseline:
        e1, lr_resolution, stream_audit, identity_audit = (value for value, _ in loaded)
        e2 = {'status': 'DEFERRED'}
        e3 = {'status': 'DEFERRED', 'selection': {
            'status': 'FIXED_BASELINE', 'run_name': None,
            'weights': scope['baseline']['mixture_weights'],
            'choice_origin': scope['baseline']['mixture_origin'],
            'comparative_evidence': False}}
    else:
        e1, e2, e3, lr_resolution, stream_audit, identity_audit = (value for value, _ in loaded)
    captured_inputs = {
        str(path.relative_to(root)): digest
        for path, (_, digest) in zip(input_paths, loaded, strict=True)
    }
    if user_override:
        captured_inputs[USER_OVERRIDE] = sha256_bytes(waiver_path.read_bytes())
    if (root / SCOPE_POLICY).is_file():
        captured_inputs[SCOPE_POLICY] = sha256_bytes((root / SCOPE_POLICY).read_bytes())
        if not user_override and lr_resolution.get('research_scope_sha256') != captured_inputs[SCOPE_POLICY]:
            raise ValueError('LR resolution does not match the current research scope')
    budget_policy = root / 'pretrain/configs/research_budget.yaml'
    if budget_policy.is_file():
        policy = yaml.safe_load(budget_policy.read_text())
        captured_inputs[str(budget_policy.relative_to(root))] = sha256_bytes(budget_policy.read_bytes())
        if not fixed_baseline and policy.get('confirmation', {}).get('required') is True:
            for stage in ('e2_architecture', 'e3_data_mix'):
                confirmation_path = root / f'experiments/{stage}/results/confirmation.json'
                confirmation, confirmation_sha = load_json_bound(confirmation_path)
                if confirmation.get('status') != 'PASS':
                    raise ValueError(f'{stage} longer-budget confirmation must PASS')
                snapshot = confirmation.get('evidence_snapshot') or {}
                if not snapshot:
                    raise ValueError('confirmation evidence snapshot missing')
                verify_captured_inputs(root, snapshot)
                captured_inputs.update(snapshot)
                captured_inputs[str(confirmation_path.relative_to(root))] = confirmation_sha
    required_evidence = ((stream_audit,) if user_override
                         else (e1, lr_resolution, stream_audit, identity_audit))
    if not fixed_baseline and not user_override:
        required_evidence += (e2, e3)
    if any(value.get('status') != 'PASS' for value in required_evidence):
        raise ValueError('All non-waived active research, stream and identity evidence must PASS')
    if user_override:
        if identity_audit.get("status") != "PASS" and not identity_audit_source_drift_only(identity_audit):
            raise ValueError("user override only waives legacy experiment source-drift failures")
        if int(waiver["training"].get("world_size", 4)) >= 8:
            parity_relative = throughput_canary.get("gradient_parity_report")
            if not isinstance(parity_relative, str):
                raise ValueError("eight-GPU freeze requires selected-topology gradient parity")
            parity_path = root / parity_relative
            parity, parity_sha = load_json_bound(parity_path)
            topology = throughput_canary.get("selected_topology") or {}
            if (parity_sha != throughput_canary.get("gradient_parity_sha256")
                    or parity.get("status") != "PASS" or parity.get("world_size") != 8
                    or parity.get("replicate_degree") != topology.get("replicate_degree")
                    or parity.get("shard_degree") != topology.get("shard_degree")):
                raise ValueError("selected-topology gradient parity is absent or mismatched")
            captured_inputs[str(parity_path.relative_to(root))] = parity_sha
        canary_cfg = throughput_canary.get("compile") or {}
        production_attention = yaml.safe_load(
            (root / "pretrain/configs/pretrain_base.yaml").read_text()
        ).get("attention", {"sdpa_backend": "auto"})
        if throughput_canary.get("attention", {"sdpa_backend": "auto"}) != production_attention:
            raise ValueError("measured attention backend differs from production config")
        if (throughput_canary.get("status") != "PASS"
                or int(throughput_canary.get("world_size", -1)) != int(
                    waiver["training"].get("world_size", 4))
                or throughput_canary.get("source_sha256") != source_sha256(root)
                or throughput_canary.get("model_configs_sha256") != sha256_bytes(
                    (root / str(waiver["architecture"]["config"])).read_bytes())
                or int(throughput_canary.get("tokens_per_step", -1)) != int(
                    waiver["training"]["tokens_per_step"])
                or int(throughput_canary.get("micro_batch_size", -1)) != int(
                    waiver["training"]["micro_batch_size"])
                or canary_cfg != waiver["training"].get("compile")):
            raise ValueError("current-source full-pretrain throughput canary does not match authorization")
        captured_inputs[str(throughput_canary_path.relative_to(root))] = sha256_bytes(
            throughput_canary_path.read_bytes())
    performance = e1.get("independent_performance") or {}
    if fixed_baseline and not user_override and performance.get('status') != 'PASS':
        raise ValueError('Scaling-only freeze still requires independent performance PASS')
    if performance.get("status") == "PASS" and not user_override:
        performance_snapshot = performance.get("snapshot") or {}
        if not performance_snapshot:
            raise ValueError("independent performance snapshot is missing")
        verify_captured_inputs(root, performance_snapshot)
        captured_inputs.update(performance_snapshot)
    verify_identity_audit_snapshot(root, identity_audit)
    recommendation = e1.get("production_recommendation") or {}
    if not user_override and recommendation.get("feasible_within_45_days") is not True:
        raise ValueError("no E1 scale is throughput-feasible for 1T within 45 days")
    target_params = int(recommendation["n_params"]) if not user_override else 0
    selected_e2 = None
    if user_override:
        source_preset_name = str(waiver["architecture"]["preset"])
        models_path = root / str(waiver["architecture"]["config"])
    elif fixed_baseline:
        source_preset_name = scope['baseline']['architecture_preset']
    else:
        selected_e2 = e2["selection"]["run_name"]
        e2_cfg_path = root / f"pretrain/configs/e2/{selected_e2}.yaml"
        e2_cfg_raw = e2_cfg_path.read_bytes()
        e2_cfg = yaml.safe_load(e2_cfg_raw)
        captured_inputs[str(e2_cfg_path.relative_to(root))] = sha256_bytes(e2_cfg_raw)
        source_preset_name = e2_cfg["model"]["preset"]
    if not user_override:
        models_path = root / "model/configs/model_configs.yaml"
    production_models_path = root / "model/configs/pretrain_final_model_configs.yaml"
    models_raw = models_path.read_bytes()
    models = yaml.safe_load(models_raw)
    captured_inputs[str(models_path.relative_to(root))] = sha256_bytes(models_raw)
    if user_override:
        if source_preset_name not in models.get("presets", {}):
            raise ValueError(f"user override preset missing from bound config: {source_preset_name}")
        final_preset = dict(models["presets"][source_preset_name])
        target_params = non_embedding_params(final_preset)
        expected_params = int(waiver["architecture"].get("non_embedding_params", target_params))
        if target_params != expected_params:
            raise ValueError("user override architecture parameter count differs from its authorization")
        models = {"presets": {}}
    else:
        final_preset = scale_architecture(models["presets"][source_preset_name], target_params)
    if fixed_baseline or user_override:
        final_preset['notes'] = final_preset['notes'].replace('E2 architecture', 'fixed baseline architecture (E2 deferred)')
    models["presets"]["pretrain-final"] = final_preset

    final_params = non_embedding_params(final_preset)
    if user_override:
        final_lr = float(waiver["training"]["peak_learning_rate"])
        batch_lr_factor = float(waiver["training"].get("batch_lr_factor", 1.0))
    else:
        final_lr, batch_lr_factor = resolved_learning_rate(lr_resolution, final_params)
    base_weights = {domain: float(e3["selection"]["weights"][domain]) for domain in DOMAINS}
    phase_ratios = load_training_ratios(root, captured_inputs)
    stage_documents, stage_reports = {}, {}
    for stage, (relative, target_final, multipliers) in STAGES.items():
        ratios = phase_ratios[stage]
        path = root / relative
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        final_weights = normalized(base_weights, multipliers)
        reference_quotas = {
            domain: target_final * final_weights[domain] / ratios[domain] * (1.0 + capacity_headroom)
            for domain in DOMAINS
        }
        reference_total = int(math.ceil(sum(reference_quotas.values())))
        reference_weights = {domain: reference_quotas[domain] / sum(reference_quotas.values())
                             for domain in DOMAINS}
        cfg["total_tokens"] = reference_total
        cfg["target_final_tokens"] = target_final
        cfg["target_final_domain_weights"] = final_weights
        cfg["tokenizer_capacity_headroom"] = capacity_headroom
        cfg["tokenizer_calibration_source"] = TRAINING_CALIBRATION_REPORT
        stored_weights = {domain: float(f"{reference_weights[domain]:.12g}")
                          for domain in DOMAINS if domain != "web"}
        stored_weights["web"] = 1.0 - sum(stored_weights.values())
        for domain in DOMAINS:
            cfg["domains"][domain]["weight"] = stored_weights[domain]
        stage_documents[path] = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
        stage_reports[stage] = {"config": relative, "target_final_tokens": target_final,
                                "selection_reference_tokens": reference_total,
                                "target_final_domain_weights": final_weights,
                                "selection_reference_weights": reference_weights}

    pretrain_path = root / "pretrain/configs/pretrain_base.yaml"
    pretrain_env_path = root / "pretrain/configs/pretrain_base.env"
    pretrain = yaml.safe_load(pretrain_path.read_text(encoding="utf-8"))
    pretrain["run_name"] = "pretrain-final-1t"
    pretrain["model"]["preset"] = "pretrain-final"
    pretrain["optim"]["lr"] = float(f"{final_lr:.10g}")
    selected_tps = int(waiver["training"]["tokens_per_step"] if user_override
                       else lr_resolution["selected_tokens_per_step"])
    seq_len = int(pretrain["data"]["seq_len"])
    micro_batch = int(waiver["training"]["micro_batch_size"] if user_override
                      else pretrain["train"]["micro_batch_size"])
    pretrain["train"]["micro_batch_size"] = micro_batch
    if user_override:
        pretrain["train"]["activation_checkpointing"] = bool(
            waiver["training"].get("activation_checkpointing", False))
        compile_override = waiver["training"].get("compile", {})
        if compile_override:
            pretrain["compile"] = {
                "enable": bool(compile_override.get("enable", False)),
                "components": list(compile_override.get("components", ["model"])),
                "backend": str(compile_override.get("backend", "inductor")),
            }
    world_size = int(waiver["training"].get("world_size", pretrain["parallel"]["world_size"])
                     if user_override else pretrain["parallel"]["world_size"])
    pretrain["parallel"]["world_size"] = world_size
    base_unit = seq_len * micro_batch * world_size
    if selected_tps <= 0 or selected_tps % base_unit:
        raise ValueError("selected tokens/update is incompatible with production topology")
    pretrain["train"]["grad_accum"] = selected_tps // base_unit
    stage_targets = [target for _, target, _ in STAGES.values()]
    stage_steps = optimizer_aligned_stage_steps(stage_targets, selected_tps)
    samples_per_step = micro_batch * pretrain["train"]["grad_accum"] * world_size
    pretrain["data"]["stage_sample_limits"] = [
        steps * samples_per_step for steps in stage_steps
    ]
    pretrain["train"]["expected_iters"] = sum(stage_steps)
    pretrain["optim"]["warmup_steps"] = stage_steps[0]
    pretrain["optim"]["decay_steps"] = stage_steps[-1]
    env_lines = pretrain_env_path.read_text(encoding="utf-8").splitlines()
    if not any(line.startswith("YUFORMER_MODEL_CONFIGS=") for line in env_lines):
        raise ValueError("pretrain_base.env lacks YUFORMER_MODEL_CONFIGS")
    model_config_line = f"YUFORMER_MODEL_CONFIGS={production_models_path}"
    env_text = "\n".join(
        model_config_line if line.startswith("YUFORMER_MODEL_CONFIGS=") else line
        for line in env_lines
    ) + "\n"
    documents = {
        production_models_path: yaml.safe_dump(models, sort_keys=True, allow_unicode=True),
        pretrain_path: yaml.safe_dump(pretrain, sort_keys=False, allow_unicode=True),
        pretrain_env_path: env_text,
        **stage_documents,
    }
    verify_captured_inputs(root, captured_inputs)
    verify_identity_audit_snapshot(root, identity_audit)
    for path, text in documents.items():
        atomic_text(path, text)
    verify_captured_inputs(root, captured_inputs)
    verify_identity_audit_snapshot(root, identity_audit)
    outputs = {str(path.relative_to(root)): sha256_bytes(text.encode("utf-8"))
               for path, text in documents.items()}
    return {
        "schema_version": 1, "status": "PASS", "frozen_at": datetime.now(timezone.utc).isoformat(),
        "freeze_tool_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "inputs": captured_inputs, "outputs": outputs,
        "scale_evidence": ({"status": "WAIVED_BY_USER", "observed_status": e1.get("status"),
                            "fit_quality_gate": (e1.get("fit_quality_gate") or {}).get("status"),
                            "waiver": USER_OVERRIDE} if user_override else recommendation),
        "architecture_evidence": {"run_name": selected_e2, "source_preset": source_preset_name,
                                  "selection_status": ('USER_FIXED' if user_override else
                                                       'FIXED_BASELINE' if fixed_baseline else 'EXPERIMENT_SELECTED'),
                                  "comparative_evidence": not fixed_baseline,
                                  "final_preset": "pretrain-final", "target_params": target_params,
                                  "actual_params": final_params,
                                  "production_model_configs": str(production_models_path.relative_to(root))},
        "mixture_evidence": e3["selection"], "tokenizer_domain_ratios_by_stage": phase_ratios,
        "research_scope": scope,
        "capacity_headroom": capacity_headroom, "learning_rate": final_lr,
        "batch_lr_factor": batch_lr_factor,
        "user_authorization": waiver if user_override else None,
        "selected_tokens_per_step": selected_tps,
        "optimizer_stage_steps": dict(zip(STAGES, stage_steps, strict=True)),
        "stages": stage_reports,
    }


def publish_recipe(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
        old = {key: value for key, value in previous.items() if key != "frozen_at"}
        new = {key: value for key, value in result.items() if key != "frozen_at"}
        if old == new and isinstance(previous.get("frozen_at"), str):
            result = dict(result, frozen_at=previous["frozen_at"])
    except (OSError, ValueError, AttributeError):
        pass
    atomic_text(path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--capacity-headroom", type=float, default=0.02)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments/e4_curriculum/results/frozen_1t_recipe.json")
    args = parser.parse_args()
    result = freeze(args.root.resolve(), args.capacity_headroom)
    result = publish_recipe(args.output, result)
    print(json.dumps({"status": result["status"], "model": result["architecture_evidence"],
                      "learning_rate": result["learning_rate"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
