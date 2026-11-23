#!/usr/bin/env python3
"""Fail-closed audit for the formal YuFormer pretraining stack.

The report separates repository defects (FAIL) from external/runtime gates
(BLOCKED).  A formal run may start only when the status is READY.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qualification_identity import qualification_identity, report_matches  # noqa: E402
from user_pretrain_authorization import identity_exception_is_bound, load_authorization  # noqa: E402
from run_production_recovery import validate_report as validate_production_recovery


STALE_BUDGET = re.compile(
    r"(?:4\.4\s*T|4_?400_?000_?000_?000|1\.5\s*T|1_?500_?000_?000_?000|"
    r"1\.44\s*T|1_?440_?000_?000_?000)", re.IGNORECASE
)
PREFLIGHT_MAX_AGE_SECONDS = 24 * 60 * 60


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected mapping: {path}")
    return value


def data_recipe_binding_errors(root, release_path, frozen_path, stream_audit):
    release = json.loads(release_path.read_bytes())
    data_sha = release.get('frozen_recipe_sha256')
    errors = []
    if not frozen_path.is_file():
        return ['derived release has no frozen recipe']
    if data_sha != sha256(frozen_path):
        try:
            path = root/'experiments/e4_curriculum/scripts/training_data_origin.py'
            spec = importlib.util.spec_from_file_location('_stack_training_data_origin', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            origin = module.resolve_data_origin(root, release_path, frozen_path)
            if origin['mode'] != 'TRAINING_ONLY_WITH_EXACT_DATA_REUSE' or origin['data_recipe_sha256'] != data_sha:
                raise ValueError('derived release differs from its declared immutable data origin')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append('derived release is stale or has an invalid training data origin: '+str(exc))
    if stream_audit.get('status') == 'PASS' and stream_audit.get('frozen_recipe_sha256') != data_sha:
        errors.append('token stream audit is stale for the immutable data recipe')
    return errors


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def bound_artifact_path(root: Path, recorded: str) -> Path:
    path = Path(recorded)
    return path if path.is_absolute() else root / path


def hash_manifest_errors(root: Path, manifest: Any, label: str) -> list[str]:
    if not isinstance(manifest, dict) or not manifest:
        return [f"{label} hash manifest is absent"]
    errors = []
    for recorded, expected in manifest.items():
        if not isinstance(recorded, str) or not isinstance(expected, str):
            errors.append(f"{label} hash manifest is malformed")
            continue
        path = bound_artifact_path(root, recorded)
        if not path.is_file() or sha256(path) != expected:
            errors.append(f"{label} drift: {recorded}")
    return errors


def stat_manifest_errors(root: Path, manifest: Any, label: str) -> list[str]:
    if not isinstance(manifest, dict) or not manifest:
        return [f"{label} stat manifest is absent"]
    errors = []
    for recorded, expected in manifest.items():
        if not isinstance(recorded, str) or not isinstance(expected, dict):
            errors.append(f"{label} stat manifest is malformed")
            continue
        path = bound_artifact_path(root, recorded)
        if not path.is_file():
            errors.append(f"{label} stat drift: {recorded}")
            continue
        stat = path.stat()
        current = {"bytes": stat.st_size, "device": stat.st_dev,
                   "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns}
        if current != expected:
            errors.append(f"{label} stat drift: {recorded}")
    return errors


def audit_environment_preflight(
    root: Path,
    preflight_path: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Reject old runtime evidence or paths that differ from the launch contract."""
    report = json.loads(preflight_path.read_text(encoding="utf-8"))
    env_path = root / "pretrain/configs/pretrain_base.env"
    env_values = {}
    for line_number, raw in enumerate(env_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"invalid environment line {line_number}: {env_path}")
        key, value = line.split("=", 1)
        env_values[key.strip()] = value.strip()

    actual = report.get("actual") or {}
    current_time = now or datetime.now(timezone.utc)
    report_mtime = datetime.fromtimestamp(preflight_path.stat().st_mtime, timezone.utc)
    age_seconds = (current_time - report_mtime).total_seconds()

    expected_python = env_values.get("YUFORMER_PYTHON", "")
    expected_source = env_values.get("YUFORMER_TORCHTITAN_ROOT", "")
    recorded_python = str(actual.get("executable") or "")
    recorded_source = str(actual.get("torchtitan_source_root") or "")

    def same_path(left: str, right: str) -> bool:
        if not left or not right:
            return False
        return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()

    checks = {
        "status_ready": report.get("status") == "ready",
        "report_file_fresh": 0 <= age_seconds <= PREFLIGHT_MAX_AGE_SECONDS,
        "python_matches_config": same_path(recorded_python, expected_python),
        "python_executable_exists": bool(expected_python)
        and Path(expected_python).expanduser().is_file()
        and os.access(Path(expected_python).expanduser(), os.X_OK),
        "torchtitan_source_matches_config": same_path(recorded_source, expected_source),
        "torchtitan_source_exists": bool(expected_source)
        and Path(expected_source).expanduser().is_dir(),
    }
    return {
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "checks": checks,
        "report_mtime": report_mtime.isoformat(),
        "age_seconds": age_seconds,
        "configured_python": expected_python,
        "configured_torchtitan_source": expected_source,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
    }


def active_yaml(root: Path) -> list[Path]:
    paths = [root / "pretrain/configs/pretrain_base.yaml"]
    paths += sorted((root / "pretrain/configs/e1_pilot").glob("*.yaml"))
    paths += sorted((root / "pretrain/configs/batch_pilot").glob("*.yaml"))
    paths += sorted((root / "pretrain/configs/e1").glob("*.yaml"))
    paths += sorted((root / "pretrain/configs/e2").glob("*.yaml"))
    paths += sorted((root / "pretrain/configs/e3").glob("*.yaml"))
    paths += sorted((root / "pretrain/configs/qualification").glob("*.yaml"))
    paths += sorted((root / "pretrain/configs/production_canary").glob("*.yaml"))
    paths += sorted((root / "experiments/_common/mfu_gate").glob("*.yaml"))
    return paths


def audit(root: Path) -> dict[str, Any]:
    defects: list[str] = []
    blockers: list[str] = []
    checks: dict[str, Any] = {}

    plan_path = root / "experiments/e4_curriculum/configs/e4_1t_wsd_context.yaml"
    plan = load_yaml(plan_path)
    sys.path.insert(0, str(root/'experiments/e4_curriculum/scripts'))
    from total_budget_contract import load_total_budget_contract
    total_contract = None
    try:
        total_contract = load_total_budget_contract(root, plan)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        defects.append('whole-chain budget contract: '+str(exc))
    stages = plan["pretrain"]["stages"]
    stage_tokens = [int(stage["target_tokens"]) for stage in stages]
    checks["canonical_budget"] = {
        "target_tokens": int(plan["pretrain"]["target_tokens"]),
        "stage_tokens": stage_tokens,
        "names": [stage["name"] for stage in stages],
    }
    if total_contract is not None:
        expected_pretrain = total_contract['budget']['stages'][0]
        if (int(plan['pretrain']['target_tokens']) != expected_pretrain['tokens']
                or stage_tokens != [p['tokens'] for p in expected_pretrain['phases']]
                or plan['context_scaling'].get('budget_included_in_pretrain_1t') is not True
                or plan['context_scaling']['total_extension_tokens'] != sum(
                    s['tokens'] for s in total_contract['budget']['stages'][1:])):
            defects.append('canonical WSD/context stages differ from the bound full-chain allocation')
        checks['canonical_budget'].update(total_training_tokens=total_contract['budget']['actual_total_tokens'],
            nominal_target_tokens=total_contract['nominal']['target_total_tokens'],
            full_chain_forecast=total_contract['forecast'])
    else:
        if int(plan["pretrain"]["target_tokens"]) != 1_000_000_000_000:
            defects.append("canonical pretrain target is not 1T")
        if stage_tokens != [50_000_000_000, 800_000_000_000, 150_000_000_000]:
            defects.append("canonical WSD stages are not 50B/800B/150B")

    configs = active_yaml(root)
    config_root = root / "pretrain/configs"
    allowed_root_configs = {"pretrain_base.yaml", "pretrain_base.env", 'research_budget.yaml'}
    unexpected_root_configs = sorted(
        path.name for path in config_root.iterdir()
        if path.is_file() and path.suffix in {".yaml", ".env"}
        and path.name not in allowed_root_configs
    )
    checks["active_root_config_roster"] = {
        "allowed": sorted(allowed_root_configs), "unexpected": unexpected_root_configs,
    }
    if unexpected_root_configs:
        defects.append("unexpected root-level pretrain configs: " + ", ".join(unexpected_root_configs))
    config_rows: list[dict[str, Any]] = []
    for path in configs:
        cfg = load_yaml(path)
        rel = str(path.relative_to(root))
        train = cfg.get("train") or {}
        data = cfg.get("data") or {}
        parallel = cfg.get("parallel") or {}
        budget = int(train.get("budget_tokens", 0))
        tps = (
            int(data.get("seq_len", 0))
            * int(train.get("micro_batch_size", 0))
            * int(train.get("grad_accum", 0))
            * int(parallel.get("world_size", 0))
        )
        expected = int(train.get("expected_iters", -1))
        derived = budget // tps if tps else -1
        group = "probe" if "mfu_gate" in rel else "formal"
        row = {
            "path": rel,
            "group": group,
            "backend": cfg.get("backend"),
            "budget_tokens": budget,
            "tokens_per_step": tps,
            "expected_iters": expected,
            "derived_iters": derived,
            "has_release_gate": bool(data.get("require_release")),
            "has_tokenizer_gate": bool(data.get("tokenizer_gate")),
            "has_token_stream_audit": bool(data.get("token_stream_audit")),
        }
        config_rows.append(row)
        if expected != derived:
            defects.append(f"{rel}: expected_iters does not match token arithmetic")
        if STALE_BUDGET.search(path.read_text(encoding="utf-8")):
            defects.append(f"{rel}: contains retired 4.4T/1.5T/1.44T budget")
        if group == "formal":
            if cfg.get("backend") != "torchtitan-v0.3.0":
                defects.append(f"{rel}: formal config is not pinned to TorchTitan v0.3.0")
            if (not data.get("require_release") or not data.get("tokenizer_gate")
                    or not data.get("token_stream_audit")):
                defects.append(f"{rel}: missing data release, tokenizer gate, or token-stream audit")
            sharding = parallel.get("sharding")
            if sharding not in {"full_shard", "hybrid_shard"}:
                defects.append(f"{rel}: formal config has unsupported FSDP2 sharding policy")
            replicate = parallel.get("replicate_degree", 1)
            shard = parallel.get("shard_degree", -1)
            if (not isinstance(replicate, int) or not isinstance(shard, int)
                    or replicate < 1 or shard == 0 or shard < -1):
                defects.append(f"{rel}: invalid FSDP2 replicate/shard degree")
            elif shard == -1 and parallel.get("world_size", 0) % replicate:
                defects.append(f"{rel}: world_size is not divisible by replicate_degree")
            elif replicate * (parallel.get("world_size", 0) // replicate if shard == -1 else shard) != parallel.get("world_size"):
                defects.append(f"{rel}: replicate_degree * shard_degree does not equal world_size")
        else:
            if cfg.get("experiment_status") != "probe_only":
                defects.append(f"{rel}: performance probe is not marked probe_only")
            if cfg.get("backend") != "pytorch-fsdp1-compat":
                defects.append(f"{rel}: probe backend must be explicit")
            if expected > 200:
                defects.append(f"{rel}: performance probe is not bounded to <=200 steps")
    checks["configs"] = config_rows

    env_paths = [root / "pretrain/configs/pretrain_base.env"]
    env_paths += sorted((root / "pretrain/configs/e1_pilot").glob("*.env"))
    env_paths += sorted((root / "pretrain/configs/batch_pilot").glob("*.env"))
    env_paths += sorted((root / "pretrain/configs/e1").glob("*.env"))
    env_paths += sorted((root / "pretrain/configs/e2").glob("*.env"))
    env_paths += sorted((root / "pretrain/configs/e3").glob("*.env"))
    env_paths += sorted((root / "pretrain/configs/qualification").glob("*.env"))
    env_paths += sorted((root / "pretrain/configs/production_canary").glob("*.env"))
    env_rows = []
    for path in env_paths:
        text = path.read_text(encoding="utf-8")
        rel = str(path.relative_to(root))
        required = [
            "YUFORMER_PYTHON=", "YUFORMER_TORCHTITAN_ROOT=", "YUFORMER_CONFIG=",
            "YUFORMER_TOKEN_STREAM_AUDIT=", "GPUS_PER_NODE=", "MASTER_PORT=",
        ]
        missing = [key[:-1] for key in required if key not in text]
        env_rows.append({"path": rel, "missing": missing})
        if missing:
            defects.append(f"{rel}: incomplete formal launch env: {', '.join(missing)}")
    checks["formal_envs"] = env_rows

    base = load_yaml(root / "pretrain/configs/pretrain_base.yaml")
    if int(base['train']['budget_tokens']) != int(plan['pretrain']['target_tokens']):
        defects.append('pretrain_base budget differs from its canonical allocated 4K stage')
    if len(base["data"].get("stages", [])) != 3:
        defects.append("pretrain_base does not consume three WSD stages")
    if base["optim"].get("lr_schedule") != "wsd":
        defects.append("pretrain_base does not use WSD")
    limits = [int(value) for value in base["data"].get("stage_sample_limits", [])]
    samples_per_step = (int(base["train"]["micro_batch_size"])
                        * int(base["train"]["grad_accum"])
                        * int(base["parallel"]["world_size"]))
    checks["production_stage_limits"] = {
        "limits": limits, "samples_per_step": samples_per_step,
        "sum": sum(limits),
        "expected": int(base["train"]["expected_iters"]) * samples_per_step,
    }
    if (len(limits) != 3 or any(value < 1 or value % samples_per_step for value in limits)
            or sum(limits) != int(base["train"]["expected_iters"]) * samples_per_step):
        defects.append("pretrain_base stage_sample_limits are absent or not optimizer-step aligned")

    backend_dir = root / "pretrain/backends/torchtitan"
    required_backend = [
        "backend.lock.json", "preflight.py", "validate_contract.py", "launch.sh", "yuformer_ext/config_registry.py",
        "yuformer_ext/model.py", "yuformer_ext/data.py", "yuformer_ext/parallelize.py",
        "yuformer_ext/validation.py",
        "numerical_parity.py",
        "lr_schedule_contract.py",
    ]
    missing_backend = [name for name in required_backend if not (backend_dir / name).is_file()]
    checks["torchtitan_adapter"] = {"required": required_backend, "missing": missing_backend}
    if missing_backend:
        defects.append("TorchTitan adapter closure is incomplete: " + ", ".join(missing_backend))
    launch_text = (backend_dir / "launch.sh").read_text(encoding="utf-8")
    checks["torchtitan_cross_pod_lease"] = {
        "uses_generation_lease": "run_with_lease.py" in launch_text,
        "uses_flock": "flock" in launch_text,
    }
    if "run_with_lease.py" not in launch_text or "flock" in launch_text:
        defects.append("TorchTitan launcher is not fenced by the cross-pod generation lease")
    registry_text = (backend_dir / "yuformer_ext/config_registry.py").read_text(encoding="utf-8")
    if 'DebugConfig(' not in registry_text or 'seed=int(cfg["seed"])' not in registry_text:
        defects.append("TorchTitan adapter does not bind the contract seed to runtime RNG")

    required_materialization = [
        "pretrain/scripts/assert_run_lease.py",
        "pretrain/scripts/run_with_lease.py",
        "pretrain/scripts/materialize_tokens.py",
        "pretrain/scripts/merge_token_parts.c",
        "pretrain/scripts/prepare_batch_pilots.py",
        "pretrain/scripts/select_batch_pilots.py",
        "pretrain/scripts/run_production_canary.py",
        "experiments/e3_data_mix/configs/e3_analysis_policy.yaml",
        "experiments/e3_data_mix/scripts/analyze_results.py",
        "experiments/e4_curriculum/scripts/freeze_1t_recipe.py",
        "experiments/e4_curriculum/scripts/audit_1t_plan.py",
        "experiments/e4_curriculum/scripts/materialize_1t_orders.py",
        "experiments/e4_curriculum/scripts/build_1t_order_release.py",
        "experiments/e4_curriculum/scripts/tokenize_1t_orders.py",
        "experiments/e4_curriculum/scripts/audit_1t_token_streams.py",
        "experiments/e4_curriculum/scripts/audit_1t_overlap.py",
        "experiments/e4_curriculum/scripts/overlap_merge.c",
        "experiments/e4_curriculum/scripts/overlap_consumed_merge.c",
        "pretrain/scripts/validate_production_contract.py",
    ]
    missing_materialization = [path for path in required_materialization if not (root / path).is_file()]
    checks["production_materialization_tools"] = {
        "required": required_materialization, "missing": missing_materialization,
    }
    if missing_materialization:
        defects.append("production materialization closure is incomplete: "
                       + ", ".join(missing_materialization))

    qualification_names = {path.stem for path in (root / "pretrain/configs/qualification").glob("*.yaml")}
    expected_qualification = {
        "q1_single_gpu", "q2_fsdp4_dcp", "q3_recovery_reference", "q3_train_interrupt",
        "q3_save_interrupt", "q3_load_interrupt", "q4_reshard_4to2",
        "q5_reshard_4to8",
        'q6_large_accumulation',
    }
    checks["qualification_roster"] = {
        "expected": sorted(expected_qualification), "observed": sorted(qualification_names)
    }
    if qualification_names != expected_qualification:
        defects.append("TorchTitan qualification config roster is not exact")

    for gate_rel in ("data/PRETRAIN_DATA_READY.json", "data/TOKENIZER_READY.json"):
        path = root / gate_rel
        if not path.is_file():
            blockers.append(f"missing runtime gate: {gate_rel}")
            continue
        gate = json.loads(path.read_text(encoding="utf-8"))
        if gate.get("status") != "PASS":
            blockers.append(f"runtime gate is not PASS: {gate_rel}")

    stream_root = root / "data/token_streams/pretrain-1t"
    expected_streams = [
        f"{name}.{suffix}"
        for name in ("warmup", "stable", "anneal")
        for suffix in ("tokens.bin", "meta.json", "doc_lengths.u32")
    ]
    missing_streams = [name for name in expected_streams if not (stream_root / name).is_file()]
    checks["production_streams"] = {"root": str(stream_root.relative_to(root)), "missing": missing_streams}
    if missing_streams:
        blockers.append("final 1T WSD streams are not materialized")

    evidence_gates = {
        "frozen_recipe": (root / "experiments/e4_curriculum/results/frozen_1t_recipe.json", {"PASS"}),
        "post_freeze_plan_audit": (
            root / "experiments/e4_curriculum/results/e4_1t_plan_static_audit.json",
            {"PASS_STATIC"},
        ),
        "production_throughput_canary": (
            root / "data/health/production-throughput-canary.json", {"PASS"},
        ),
        "derived_release": (root / "data/processed/releases/pretrain-1t-frozen.release.json", {"PASS"}),
        "token_stream_audit": (root / "data/processed/audits/pretrain_1t_token_streams.json", {"PASS"}),
        "overlap_audit": (root / "data/processed/audits/pretrain_1t_overlap.json", {"PASS"}),
        "production_contract": (root / "data/health/pretrain-production-contract.json", {"PASS"}),
        "experiment_identity_audit": (root / "report/experiment_identity_audit_latest.json", {"PASS"}),
    }
    if int(base["parallel"]["world_size"]) >= 8:
        evidence_gates["production_recovery"] = (
            root / "data/health/production-recovery-qualification.json", {"PASS"})
        try:
            validate_production_recovery(root)
        except (OSError, ValueError, KeyError) as exc:
            blockers.append(f"production topology recovery is not current: {exc}")
    evidence_status = {}
    evidence_values: dict[str, dict[str, Any]] = {}
    for name, (path, accepted) in evidence_gates.items():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            evidence_values[name] = value
            evidence_status[name] = value.get("status")
            waived_identity_drift = (
                name == "experiment_identity_audit"
                and identity_exception_is_bound(
                    root, evidence_values.get("frozen_recipe", {}), value
                )
            )
            if value.get("status") not in accepted and not waived_identity_drift:
                blockers.append(f"production evidence is not PASS: {name}")
            if name == "experiment_identity_audit":
                checks["experiment_identity_waiver"] = (
                    "WAIVED_BY_USER_SOURCE_DRIFT_ONLY" if waived_identity_drift else "NOT_USED"
                )
        except (OSError, ValueError):
            evidence_status[name] = "MISSING"
            blockers.append(f"production evidence is absent: {name}")
    checks["production_evidence"] = evidence_status
    evidence_tool_bindings = {
        "post_freeze_plan_audit": (
            "audit_tool_sha256", root / "experiments/e4_curriculum/scripts/audit_1t_plan.py"),
        "production_throughput_canary": (
            "runner_tool_sha256", root / "pretrain/scripts/run_production_canary.py"),
        "derived_release": (
            "builder_tool_sha256", root / "experiments/e4_curriculum/scripts/build_1t_order_release.py"),
        "token_stream_audit": (
            "audit_tool_sha256", root / "experiments/e4_curriculum/scripts/audit_1t_token_streams.py"),
        "overlap_audit": (
            "audit_tool_sha256", root / "experiments/e4_curriculum/scripts/audit_1t_overlap.py"),
        "production_contract": (
            "validator_tool_sha256", root / "pretrain/scripts/validate_production_contract.py"),
    }
    for name, (field, tool_path) in evidence_tool_bindings.items():
        value = evidence_values.get(name, {})
        if (name == "post_freeze_plan_audit"
                and evidence_values.get("frozen_recipe", {}).get("status") != "PASS"):
            continue
        if value.get("status") in evidence_gates[name][1]:
            if value.get(field) != sha256(tool_path):
                blockers.append(f"production evidence uses a stale tool: {name}")
    overlap_path = evidence_gates["overlap_audit"][0]
    if evidence_status["overlap_audit"] == "PASS":
        overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
        if (int(overlap.get("schema_version", 0)) < 2
                or overlap.get("checks", {}).get("configured_consumption_exact") is not True
                or overlap.get("checks", {}).get("consumed_domain_mixture_within_tolerance") is not True
                or not all((overlap.get("checks") or {}).values())):
            defects.append("PASS overlap audit contains a failed check")
        overlap_snapshot = overlap.get("evidence_snapshot")
        if overlap.get("evidence_snapshot_count") != (
                len(overlap_snapshot) if isinstance(overlap_snapshot, dict) else -1):
            defects.append("overlap evidence snapshot count is inconsistent")
        defects.extend(hash_manifest_errors(root, overlap_snapshot,
                                            "overlap evidence"))
        defects.extend(stat_manifest_errors(
            root, overlap.get("large_artifact_stat_identities"),
            "overlap large artifact"
        ))
        defects.extend(hash_manifest_errors(root, overlap.get("merge_tool_sha256"),
                                            "overlap merge tool"))
    stream_audit = evidence_values.get("token_stream_audit", {})
    if (stream_audit.get("status") == "PASS"
            and stream_audit.get("train_doc_lengths_audited") is not True):
        defects.append("PASS token stream audit does not close production document lengths")
    if stream_audit.get("status") == "PASS":
        shared_auditor = root / "pretrain/scripts/audit_token_stream_release.py"
        if stream_audit.get("shared_auditor_sha256") != sha256(shared_auditor):
            defects.append("token stream audit uses a stale shared auditor")

    derived_release = evidence_values.get("derived_release", {})
    if derived_release.get("status") == "PASS":
        frozen_path = evidence_gates["frozen_recipe"][0]
        defects.extend(data_recipe_binding_errors(root, evidence_gates['derived_release'][0],
                                                  frozen_path, stream_audit))

    identity_audit = evidence_values.get("experiment_identity_audit", {})
    identity_waived = identity_exception_is_bound(
        root, evidence_values.get("frozen_recipe", {}), identity_audit
    )
    if identity_audit.get("status") == "PASS" or identity_waived:
        snapshot = identity_audit.get("evidence_snapshot")
        if identity_audit.get("evidence_snapshot_count") != (
                len(snapshot) if isinstance(snapshot, dict) else -1):
            defects.append("experiment identity evidence snapshot count is inconsistent")
        defects.extend(hash_manifest_errors(
            root, snapshot, "experiment identity evidence"
        ))

    frozen = evidence_values.get("frozen_recipe", {})
    if frozen.get("status") == "PASS":
        freeze_tool = root / "experiments/e4_curriculum/scripts/freeze_1t_recipe.py"
        if frozen.get("freeze_tool_sha256") != sha256(freeze_tool):
            defects.append("frozen recipe was generated by a stale freeze tool")
        defects.extend(hash_manifest_errors(root, frozen.get("inputs"),
                                            "frozen recipe input"))
        defects.extend(hash_manifest_errors(root, frozen.get("outputs"),
                                            "frozen recipe output"))
        actual_tps = (int(base["data"]["seq_len"])
                      * int(base["train"]["micro_batch_size"])
                      * int(base["train"]["grad_accum"])
                      * int(base["parallel"]["world_size"]))
        if int(frozen.get("selected_tokens_per_step", -1)) != actual_tps:
            defects.append("frozen batch selection differs from pretrain_base tokens/update")
        if float(frozen.get("learning_rate", -1)) != float(base["optim"]["lr"]):
            defects.append("frozen learning rate differs from pretrain_base")

    plan_audit = evidence_values.get("post_freeze_plan_audit", {})
    # The same report path also holds the pre-freeze capacity audit.  It becomes
    # production evidence only after a frozen recipe exists.
    if frozen.get("status") != "PASS" and plan_audit.get("status") == "PASS_STATIC":
        evidence_status["post_freeze_plan_audit"] = "PRE_FREEZE_ONLY"
    if frozen.get("status") == "PASS" and plan_audit.get("status") == "PASS_STATIC":
        plan_inputs = plan_audit.get("inputs") or {}
        frozen_path = evidence_gates["frozen_recipe"][0]
        if (plan_inputs.get("frozen_recipe") != str(frozen_path.relative_to(root))
                or plan_inputs.get("frozen_recipe_sha256") != sha256(frozen_path)):
            defects.append("post-freeze plan audit is not bound to the frozen recipe")
        bound_files = []
        if plan_inputs.get("plan") and plan_inputs.get("plan_sha256"):
            bound_files.append((plan_inputs["plan"], plan_inputs["plan_sha256"]))
        boundary = (plan_audit.get("checks") or {}).get("optimizer_aligned_stage_boundaries") or {}
        if boundary.get("config") and boundary.get("config_sha256"):
            bound_files.append((boundary["config"], boundary["config_sha256"]))
        bound_files.extend((row.get("mix_config"), row.get("mix_config_sha256"))
                           for row in plan_audit.get("stages", []))
        if len(bound_files) != 5:
            defects.append("post-freeze plan audit does not bind the plan, train config, and three mixes")
        for relative, expected_sha in bound_files:
            path = root / str(relative)
            if not relative or not expected_sha or not path.is_file() or sha256(path) != expected_sha:
                defects.append(f"post-freeze plan audit input drift: {relative}")

    canary = evidence_values.get("production_throughput_canary", {})
    if canary.get("status") == "PASS":
        if not report_matches(canary, qualification_identity(root)):
            blockers.append("production throughput canary PASS is stale for current training inputs")
        production_world = int(base["parallel"]["world_size"])
        canary_config = root / f"pretrain/configs/production_canary/production_{production_world}gpu.yaml"
        canary_env = root / f"pretrain/configs/production_canary/production_{production_world}gpu.env"
        evaluation = canary.get("evaluation") or {}
        if (frozen.get('training_adoption', {}).get('global_training_budget') is not None
                and evaluation.get('runtime_scope') != 'TOTAL_4K_16K_64K'):
            defects.append('production canary omits the required full-chain runtime allocation')
        frozen_path = evidence_gates["frozen_recipe"][0]
        if (not frozen_path.is_file()
                or canary.get("frozen_recipe_sha256") != sha256(frozen_path)):
            defects.append("production throughput canary is not bound to frozen recipe")
        if (not canary_config.is_file()
                or canary.get("config_sha256") != sha256(canary_config)):
            defects.append("production throughput canary config drift")
        if not canary_env.is_file() or canary.get("env_sha256") != sha256(canary_env):
            defects.append("production throughput canary env drift")
        try:
            runtime_ceiling_days = float(
                load_authorization(root)["training"].get("launch_runtime_ceiling_days", 45.0)
            )
        except (OSError, ValueError, TypeError, KeyError):
            runtime_ceiling_days = 45.0
        if (evaluation.get("status") != "PASS"
                or float(evaluation.get("maximum_days", -1)) != runtime_ceiling_days
                or int(evaluation.get("world_size", -1)) != production_world
                or float(evaluation.get("projected_total_training_days", evaluation.get("projected_1t_days_from_p10", float("inf"))))
                > runtime_ceiling_days):
            defects.append(
                f"production throughput canary exceeds or does not bind the {runtime_ceiling_days:g}-day gate"
            )

    production_contract = evidence_values.get("production_contract", {})
    if production_contract.get("status") == "PASS":
        runtime_evidence = production_contract.get("runtime_evidence")
        if production_contract.get("runtime_evidence_count") != (
                len(runtime_evidence) if isinstance(runtime_evidence, dict) else -1):
            defects.append("production contract runtime evidence count is inconsistent")
        defects.extend(hash_manifest_errors(
            root, runtime_evidence, "production contract runtime evidence"
        ))
        frozen_evidence = production_contract.get("frozen_evidence")
        expected_frozen = {
            "frozen_recipe_sha256": evidence_gates["frozen_recipe"][0],
            "post_freeze_plan_audit_sha256": evidence_gates["post_freeze_plan_audit"][0],
            "overlap_audit_sha256": evidence_gates["overlap_audit"][0],
            "production_canary_sha256": evidence_gates["production_throughput_canary"][0],
        }
        if int(base["parallel"]["world_size"]) >= 8:
            expected_frozen["production_recovery_sha256"] = evidence_gates["production_recovery"][0]
        if not isinstance(frozen_evidence, dict):
            defects.append("production contract frozen evidence is absent")
        else:
            for key, path in expected_frozen.items():
                if not path.is_file() or frozen_evidence.get(key) != sha256(path):
                    defects.append(f"production contract frozen evidence drift: {key}")

    preflight = root / "data/health/torchtitan-preflight.json"
    if not preflight.is_file():
        blockers.append("TorchTitan environment preflight report is absent")
    else:
        preflight_audit = audit_environment_preflight(root, preflight)
        checks["environment_preflight"] = preflight_audit
        if preflight_audit["status"] != "PASS":
            blockers.append(
                "TorchTitan environment preflight is stale or does not match "
                "the configured runtime: " + ", ".join(preflight_audit["failed_checks"])
            )

    qualification = root / "data/health/torchtitan-qualification.json"
    if not qualification.is_file():
        blockers.append("TorchTitan numerical/recovery/reshard qualification is absent")
    else:
        result = json.loads(qualification.read_text(encoding="utf-8"))
        checks["torchtitan_qualification"] = result.get("status")
        identity_current = qualification_identity(root)
        identity_matches = report_matches(result, identity_current)
        checks["torchtitan_qualification_identity"] = {
            "matches": identity_matches,
            "current_sha256": identity_current["sha256"],
            "recorded_sha256": (result.get("qualification_identity") or {}).get("sha256"),
        }
        if result.get("status") != "PASS":
            blockers.append("TorchTitan numerical/recovery/reshard qualification is not PASS")
        elif not identity_matches:
            blockers.append("TorchTitan qualification PASS is stale for current inputs")

    status = "FAIL" if defects else ("BLOCKED" if blockers else "READY")
    return {
        "schema_version": 1,
        "status": status,
        "defects": defects,
        "blockers": blockers,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    result = audit(root)
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        out = args.output if args.output.is_absolute() else root / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["status"] in {"READY", "BLOCKED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
