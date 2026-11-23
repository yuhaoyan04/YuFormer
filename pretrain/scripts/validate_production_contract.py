#!/usr/bin/env python3
"""Validate the frozen production contract without publishing a run identity."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qualification_identity import qualification_identity, report_matches
from research_scope import scope_binding_current
from user_pretrain_authorization import identity_exception_is_bound, load_authorization
from run_production_recovery import validate_report as validate_production_recovery
import source_document_binding


ROOT = Path(__file__).resolve().parents[2]
ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_hash_manifest(root: Path, manifest: Any, label: str) -> None:
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError(f"{label} hash manifest is absent")
    for recorded, expected in manifest.items():
        if not isinstance(recorded, str) or not isinstance(expected, str):
            raise ValueError(f"{label} hash manifest is malformed")
        path = Path(recorded)
        if not path.is_absolute():
            path = root / path
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"{label} drift: {recorded}")


def validate_stat_manifest(root: Path, manifest: Any, label: str) -> None:
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError(f"{label} stat manifest is absent")
    for recorded, expected in manifest.items():
        path = Path(recorded)
        if not path.is_absolute():
            path = root / path
        if not isinstance(expected, dict) or not path.is_file():
            raise ValueError(f"{label} stat manifest is malformed: {recorded}")
        stat = path.stat()
        current = {"bytes": stat.st_size, "device": stat.st_dev,
                   "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns}
        if current != expected:
            raise ValueError(f"{label} stat drift: {recorded}")


def validate_frozen_evidence(root: Path) -> dict[str, str]:
    qualification_path = root / "data/health/torchtitan-qualification.json"
    qualification = load_json(qualification_path)
    if (qualification.get("status") != "PASS"
            or not report_matches(qualification, qualification_identity(root))):
        raise ValueError("production requires current numerical/recovery/reshard qualification PASS")
    frozen_path = root / "experiments/e4_curriculum/results/frozen_1t_recipe.json"
    plan_path = root / "experiments/e4_curriculum/results/e4_1t_plan_static_audit.json"
    overlap_path = root / "data/processed/audits/pretrain_1t_overlap.json"
    canary_path = root / "data/health/production-throughput-canary.json"
    production_world = int(yaml.safe_load((root / "pretrain/configs/pretrain_base.yaml").read_text())["parallel"]["world_size"])
    if production_world >= 8 and qualification.get("stages", {}).get("reshard_4to8", {}).get("status") != "PASS":
        raise ValueError("eight-GPU production requires 4-to-8 reshard qualification PASS")
    if production_world >= 8:
        validate_production_recovery(root)
        streams = load_json(root / "data/processed/audits/pretrain_1t_token_streams.json")
        if (streams.get("status") != "PASS" or streams.get("train_source_rows_verified") is not True
                or streams.get("source_binding_tool_sha256") != sha256(Path(source_document_binding.__file__))
                or streams.get("audit_tool_sha256") != sha256(root / "experiments/e4_curriculum/scripts/audit_1t_token_streams.py")
                or streams.get("shared_auditor_sha256") != sha256(root / "pretrain/scripts/audit_token_stream_release.py")):
            raise ValueError("production training streams lack current source document-ID verification")
    canary_config = root / f"pretrain/configs/production_canary/production_{production_world}gpu.yaml"
    canary_env = root / f"pretrain/configs/production_canary/production_{production_world}gpu.env"
    identity_audit_path = root / "report/experiment_identity_audit_latest.json"
    frozen, plan, overlap, canary = (
        load_json(path) for path in (frozen_path, plan_path, overlap_path, canary_path)
    )
    if frozen.get("status") != "PASS":
        raise ValueError("frozen 1T recipe is not PASS")
    if not scope_binding_current(root, frozen):
        raise ValueError('frozen 1T recipe is not bound to the current research scope')
    freeze_tool = root / "experiments/e4_curriculum/scripts/freeze_1t_recipe.py"
    if frozen.get("freeze_tool_sha256") != sha256(freeze_tool):
        raise ValueError("frozen 1T recipe was generated by a stale freeze tool")
    if plan.get("status") != "PASS_STATIC" or plan.get("errors"):
        raise ValueError("post-freeze 1T plan audit is not PASS_STATIC")
    plan_tool = root / "experiments/e4_curriculum/scripts/audit_1t_plan.py"
    if plan.get("audit_tool_sha256") != sha256(plan_tool):
        raise ValueError("post-freeze 1T plan audit uses a stale tool")
    plan_inputs = plan.get("inputs") or {}
    if (plan_inputs.get("frozen_recipe") != str(frozen_path.relative_to(root))
            or plan_inputs.get("frozen_recipe_sha256") != sha256(frozen_path)):
        raise ValueError("post-freeze 1T plan audit is not bound to the frozen recipe")
    if (overlap.get("status") != "PASS"
            or int(overlap.get("schema_version", 0)) < 2
            or overlap.get("checks", {}).get("configured_consumption_exact") is not True
            or overlap.get("checks", {}).get("consumed_domain_mixture_within_tolerance") is not True
            or not all((overlap.get("checks") or {}).values())):
        raise ValueError("1T overlap audit is not PASS")
    overlap_snapshot = overlap.get("evidence_snapshot")
    if overlap.get("evidence_snapshot_count") != (
            len(overlap_snapshot) if isinstance(overlap_snapshot, dict) else -1):
        raise ValueError("1T overlap evidence snapshot count is inconsistent")
    validate_hash_manifest(root, overlap_snapshot, "1T overlap evidence")
    validate_stat_manifest(root, overlap.get("large_artifact_stat_identities"),
                           "1T overlap large artifact")
    overlap_tool = root / "experiments/e4_curriculum/scripts/audit_1t_overlap.py"
    if overlap.get("audit_tool_sha256") != sha256(overlap_tool):
        raise ValueError("1T overlap audit uses a stale tool")
    validate_hash_manifest(root, overlap.get("merge_tool_sha256"), "1T overlap merge tool")
    canary_evaluation = canary.get("evaluation") or {}
    if (frozen.get('training_adoption', {}).get('global_training_budget') is not None
            and canary_evaluation.get('runtime_scope') != 'TOTAL_4K_16K_64K'):
        raise ValueError('production canary must include the full4K/16K/64K runtime allocation')
    if (canary.get("status") != "PASS" or canary_evaluation.get("status") != "PASS"
            or int(canary_evaluation.get("world_size", -1)) != production_world
            or float(canary_evaluation.get("planned_training_duty_cycle", 1.0)) != float(
                load_authorization(root)["training"].get("planned_training_duty_cycle", 1.0))
            or not report_matches(canary, qualification_identity(root))
            or canary.get("runner_tool_sha256") != sha256(
                root / "pretrain/scripts/run_production_canary.py")
            or canary.get("frozen_recipe_sha256") != sha256(frozen_path)
            or not canary_config.is_file() or canary.get("config_sha256") != sha256(canary_config)
            or not canary_env.is_file() or canary.get("env_sha256") != sha256(canary_env)
            or float(canary_evaluation.get("maximum_days", -1)) != float(
                load_authorization(root)["training"].get("launch_runtime_ceiling_days", 45.0))
            or float(canary_evaluation.get("projected_total_training_days", canary_evaluation.get("projected_1t_days_from_p10", math.inf))) > float(
                load_authorization(root)["training"].get("launch_runtime_ceiling_days", 45.0))):
        raise ValueError("frozen production throughput canary is absent, stale, or too slow")
    validate_hash_manifest(root, frozen.get("inputs"), "frozen recipe input")
    validate_hash_manifest(root, frozen.get("outputs"), "frozen recipe output")
    identity_audit = load_json(identity_audit_path)
    if (identity_audit.get("status") != "PASS"
            and not identity_exception_is_bound(root, frozen, identity_audit)):
        raise ValueError("experiment identity audit is neither PASS nor covered by the explicit E1 waiver")
    snapshot = identity_audit.get("evidence_snapshot")
    if identity_audit.get("evidence_snapshot_count") != (
            len(snapshot) if isinstance(snapshot, dict) else -1):
        raise ValueError("experiment identity evidence snapshot count is inconsistent")
    validate_hash_manifest(root, snapshot, "experiment identity evidence")
    train_path = root / "pretrain/configs/pretrain_base.yaml"
    train = yaml.safe_load(train_path.read_text(encoding="utf-8"))
    actual_tps = (int(train["data"]["seq_len"])
                  * int(train["train"]["micro_batch_size"])
                  * int(train["train"]["grad_accum"])
                  * int(train["parallel"]["world_size"]))
    if int(frozen.get("selected_tokens_per_step", -1)) != actual_tps:
        raise ValueError("frozen batch selection differs from production tokens/update")
    if not math.isclose(float(frozen.get("learning_rate", -1)),
                        float(train["optim"]["lr"]), rel_tol=1e-12, abs_tol=0.0):
        raise ValueError("frozen learning rate differs from production config")
    bound = []
    inputs = plan.get("inputs") or {}
    bound.append((inputs.get("plan"), inputs.get("plan_sha256")))
    boundary = (plan.get("checks") or {}).get("optimizer_aligned_stage_boundaries") or {}
    if int(boundary.get("tokens_per_optimizer_step", -1)) != actual_tps:
        raise ValueError("post-freeze audit tokens/update differs from frozen production config")
    bound.append((boundary.get("config"), boundary.get("config_sha256")))
    bound.extend((row.get("mix_config"), row.get("mix_config_sha256"))
                 for row in plan.get("stages", []))
    if len(bound) != 5:
        raise ValueError("post-freeze plan audit has an incomplete bound file roster")
    for relative, expected in bound:
        path = root / str(relative)
        if not relative or not expected or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"post-freeze plan audit input drift: {relative}")
    evidence = {
        "qualification_sha256": sha256(qualification_path),
        "frozen_recipe_sha256": sha256(frozen_path),
        "post_freeze_plan_audit_sha256": sha256(plan_path),
        "overlap_audit_sha256": sha256(overlap_path),
        "production_canary_sha256": sha256(canary_path),
    }
    if production_world >= 8:
        evidence["production_recovery_sha256"] = sha256(root / "data/health/production-recovery-qualification.json")
    return evidence


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"invalid env line {number}: {path}")
        key, value = line.split("=", 1)
        if not ENV_KEY.fullmatch(key) or any(token in value for token in ("$(`", "`", "$(", "${")):
            raise ValueError(f"unsafe env line {number}: {path}")
        values[key] = value
    return values


def run(root: Path, env_path: Path) -> dict[str, Any]:
    frozen_evidence = validate_frozen_evidence(root)
    values = load_env(env_path)
    config_value = values.get("YUFORMER_CONFIG")
    if not config_value:
        raise ValueError("YUFORMER_CONFIG is absent")
    config_path = Path(config_value)
    if not config_path.is_absolute():
        config_path = root / "pretrain" / config_path
    module_path = root / "pretrain/backends/torchtitan/validate_contract.py"
    spec = importlib.util.spec_from_file_location("production_contract_validator", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load validator: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        validated = module.validate(root, config_path.resolve(), publish_identity=False,
                                    require_production_order=True)
    finally:
        for key, old in previous.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
    identity = validated["identity"]
    runtime_paths = {
        env_path,
        config_path.resolve(),
        Path(identity["model_configs"]),
        Path(identity["tokenizer_gate"]),
        Path(identity["release_manifest"]),
        Path(identity["token_stream_audit"]),
        Path(identity["tokenizer_path"]),
    }
    runtime_evidence = {}
    for path in sorted(runtime_paths, key=lambda value: str(value.resolve())):
        resolved = path.resolve()
        try:
            recorded = resolved.relative_to(root).as_posix()
        except ValueError:
            recorded = str(resolved)
        runtime_evidence[recorded] = sha256(resolved)
    return {
        "schema_version": 1,
        "status": "PASS",
        "validator_tool_sha256": sha256(Path(__file__)),
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.resolve().relative_to(root)),
        "run_name": identity["run_name"],
        "identity_sha256": identity["identity_sha256"],
        "frozen_evidence": frozen_evidence,
        "runtime_evidence": runtime_evidence,
        "runtime_evidence_count": len(runtime_evidence),
        "stage_sample_limits": [row["sample_limit"] for row in identity["streams"]],
        "stage_capacity_samples": [row["capacity_samples"] for row in identity["streams"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--env", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    env_path = (args.env or root / "pretrain/configs/pretrain_base.env").resolve()
    output = (args.output or root / "data/health/pretrain-production-contract.json").resolve()
    try:
        result = run(root, env_path)
    except Exception as exc:
        result = {"schema_version": 1, "status": "FAIL", "validated_at": datetime.now(timezone.utc).isoformat(),
                  "error": f"{type(exc).__name__}: {exc}"}
    atomic_json(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
