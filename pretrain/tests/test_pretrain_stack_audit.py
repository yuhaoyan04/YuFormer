from pathlib import Path
import importlib.util
import json
import os
import sys
from datetime import datetime, timedelta, timezone


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "pretrain/scripts/audit_pretrain_stack.py"
SPEC = importlib.util.spec_from_file_location("audit_pretrain_stack", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

VALIDATOR_SCRIPT = ROOT / "pretrain/scripts/validate_production_contract.py"
VALIDATOR_SPEC = importlib.util.spec_from_file_location("validate_production_contract", VALIDATOR_SCRIPT)
VALIDATOR = importlib.util.module_from_spec(VALIDATOR_SPEC)
assert VALIDATOR_SPEC and VALIDATOR_SPEC.loader
VALIDATOR_SPEC.loader.exec_module(VALIDATOR)

CANARY_SCRIPT = ROOT / "pretrain/scripts/run_production_canary.py"
CANARY_SPEC = importlib.util.spec_from_file_location("run_production_canary", CANARY_SCRIPT)
CANARY = importlib.util.module_from_spec(CANARY_SPEC)
assert CANARY_SPEC and CANARY_SPEC.loader
CANARY_SPEC.loader.exec_module(CANARY)


def test_repository_has_no_active_config_defects():
    result = MODULE.audit(ROOT)
    # Runtime streams/environment may remain blocked, but checked-in contracts
    # and adapter code must always be internally closed.
    assert result["defects"] == []
    assert result["status"] in {"BLOCKED", "READY"}


def test_canonical_budget_and_stage_split():
    result = MODULE.audit(ROOT)
    budget = result["checks"]["canonical_budget"]
    effective = json.loads((ROOT/'experiments/e4_curriculum/results/effective_total_training_budget.json').read_bytes())
    assert budget['target_tokens'] == effective['stages'][0]['tokens']
    assert budget['stage_tokens'] == [phase['tokens'] for phase in effective['stages'][0]['phases']]
    assert budget['total_training_tokens'] == effective['actual_total_tokens']
    assert result["checks"]["active_root_config_roster"]["unexpected"] == []


def test_environment_preflight_must_be_fresh_and_match_configured_runtime(tmp_path):
    env_path = tmp_path / "pretrain/configs/pretrain_base.env"
    env_path.parent.mkdir(parents=True)
    source_root = tmp_path / "torchtitan"
    source_root.mkdir()
    env_path.write_text(
        f"YUFORMER_PYTHON={sys.executable}\nYUFORMER_TORCHTITAN_ROOT={source_root}\n",
        encoding="utf-8",
    )
    preflight_path = tmp_path / "preflight.json"
    now = datetime.now(timezone.utc)
    preflight_path.write_text(json.dumps({
        "status": "ready",
        "actual": {
            "executable": sys.executable,
            "torchtitan_source_root": str(source_root),
        },
    }), encoding="utf-8")

    report = MODULE.audit_environment_preflight(
        tmp_path, preflight_path, now=now + timedelta(seconds=1)
    )
    assert report["status"] == "PASS"

    env_path.write_text(
        f"YUFORMER_PYTHON={sys.executable}.stale\nYUFORMER_TORCHTITAN_ROOT={source_root}\n",
        encoding="utf-8",
    )
    mismatched = MODULE.audit_environment_preflight(
        tmp_path, preflight_path, now=now + timedelta(seconds=1)
    )
    assert "python_matches_config" in mismatched["failed_checks"]
    env_path.write_text(
        f"YUFORMER_PYTHON={sys.executable}\nYUFORMER_TORCHTITAN_ROOT={source_root}\n",
        encoding="utf-8",
    )

    os.utime(preflight_path, (now.timestamp() - 25 * 60 * 60,) * 2)
    stale = MODULE.audit_environment_preflight(
        tmp_path, preflight_path, now=now,
    )
    assert stale["status"] == "BLOCKED"
    assert "report_file_fresh" in stale["failed_checks"]
    assert stale["failed_checks"] == ["report_file_fresh"]


def test_frozen_production_evidence_rejects_post_audit_drift(tmp_path):
    relative_files = [
        "experiments/e4_curriculum/configs/e4_1t_wsd_context.yaml",
        "pretrain/configs/pretrain_base.yaml",
        "data-process/configs/mix_warmup_50b.yaml",
        "data-process/configs/mix_stable_800b.yaml",
        "data-process/configs/mix_anneal.yaml",
    ]
    for relative in relative_files:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    production_tps = 4096 * 4 * 32 * 4
    (tmp_path / relative_files[1]).write_text(
        "data:\n  seq_len: 4096\ntrain:\n  micro_batch_size: 4\n  grad_accum: 32\n"
        "parallel:\n  world_size: 4\noptim:\n  lr: 0.0002\n",
        encoding="utf-8",
    )
    frozen_path = tmp_path / "experiments/e4_curriculum/results/frozen_1t_recipe.json"
    plan_path = tmp_path / "experiments/e4_curriculum/results/e4_1t_plan_static_audit.json"
    overlap_path = tmp_path / "data/processed/audits/pretrain_1t_overlap.json"
    canary_path = tmp_path / "data/health/production-throughput-canary.json"
    canary_config = tmp_path / "pretrain/configs/production_canary/production_4gpu.yaml"
    canary_env = tmp_path / "pretrain/configs/production_canary/production_4gpu.env"
    identity_audit_path = tmp_path / "report/experiment_identity_audit_latest.json"
    authorization_path = tmp_path / "data/health/pretrain-user-full-pretrain-authorization.json"
    freeze_tool = tmp_path / "experiments/e4_curriculum/scripts/freeze_1t_recipe.py"
    plan_tool = tmp_path / "experiments/e4_curriculum/scripts/audit_1t_plan.py"
    overlap_tool = tmp_path / "experiments/e4_curriculum/scripts/audit_1t_overlap.py"
    canary_tool = tmp_path / "pretrain/scripts/run_production_canary.py"
    merge_tool = tmp_path / "experiments/e4_curriculum/scripts/overlap_merge.c"
    consumed_merge_tool = tmp_path / "experiments/e4_curriculum/scripts/overlap_consumed_merge.c"
    for path in (frozen_path, plan_path, overlap_path, canary_path, canary_config, canary_env,
                 identity_audit_path, authorization_path, freeze_tool, plan_tool, overlap_tool, canary_tool,
                 merge_tool, consumed_merge_tool):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    for path in (freeze_tool, plan_tool, overlap_tool, canary_tool, merge_tool, consumed_merge_tool):
        path.write_text(f"# {path.name}\n", encoding="utf-8")
    outputs = {relative: VALIDATOR.sha256(tmp_path / relative) for relative in relative_files[1:]}
    identity_evidence = tmp_path / "data/experiments/e1/result.json"
    identity_evidence.parent.mkdir(parents=True, exist_ok=True)
    identity_evidence.write_text('{"status":"PASS"}\n', encoding="utf-8")
    authorization_path.write_text(json.dumps({
        "status": "APPROVED_BY_USER",
        "authorization_scope": "skip_e1_and_launch_full_pretrain",
        "research_decision": "explicitly waive historical E1 source drift",
        "training": {"launch_runtime_ceiling_days": 45.0},
    }), encoding="utf-8")
    identity_audit_path.write_text(json.dumps({
        "status": "BLOCKED", "roster_exact": True, "total": 1,
        "rows": [{"status": "FAIL", "errors": ["training_source_drift"]}],
        "evidence_snapshot_count": 1,
        "evidence_snapshot": {
            str(identity_evidence.relative_to(tmp_path)): VALIDATOR.sha256(identity_evidence)
        },
    }), encoding="utf-8")
    inputs = {str(identity_audit_path.relative_to(tmp_path)): VALIDATOR.sha256(identity_audit_path),
              str(authorization_path.relative_to(tmp_path)): VALIDATOR.sha256(authorization_path)}
    frozen_path.write_text(json.dumps({"status": "PASS", "inputs": inputs, "outputs": outputs,
                                       "scale_evidence": {"status": "WAIVED_BY_USER"},
                                       "freeze_tool_sha256": VALIDATOR.sha256(freeze_tool),
                                       "selected_tokens_per_step": production_tps,
                                       "learning_rate": 0.0002}))
    plan_path.write_text(json.dumps({
        "status": "PASS_STATIC", "errors": [],
        "audit_tool_sha256": VALIDATOR.sha256(plan_tool),
        "inputs": {"plan": relative_files[0],
                   "plan_sha256": VALIDATOR.sha256(tmp_path / relative_files[0]),
                   "frozen_recipe": str(frozen_path.relative_to(tmp_path)),
                   "frozen_recipe_sha256": VALIDATOR.sha256(frozen_path)},
        "checks": {"optimizer_aligned_stage_boundaries": {
            "config": relative_files[1],
            "config_sha256": VALIDATOR.sha256(tmp_path / relative_files[1]),
            "tokens_per_optimizer_step": production_tps,
        }},
        "stages": [{"mix_config": relative,
                    "mix_config_sha256": VALIDATOR.sha256(tmp_path / relative)}
                   for relative in relative_files[2:]],
    }))
    overlap_path.write_text(json.dumps({
        "schema_version": 2, "status": "PASS",
        "checks": {"closed": True, "configured_consumption_exact": True,
                   "consumed_domain_mixture_within_tolerance": True},
        "evidence_snapshot_count": 1,
        "evidence_snapshot": {relative_files[0]: VALIDATOR.sha256(tmp_path / relative_files[0])},
        "large_artifact_stat_identities": {relative_files[0]: {
            "bytes": (tmp_path / relative_files[0]).stat().st_size,
            "device": (tmp_path / relative_files[0]).stat().st_dev,
            "inode": (tmp_path / relative_files[0]).stat().st_ino,
            "mtime_ns": (tmp_path / relative_files[0]).stat().st_mtime_ns,
        }},
        "audit_tool_sha256": VALIDATOR.sha256(overlap_tool),
        "merge_tool_sha256": {
            str(merge_tool.relative_to(tmp_path)): VALIDATOR.sha256(merge_tool),
            str(consumed_merge_tool.relative_to(tmp_path)): VALIDATOR.sha256(consumed_merge_tool),
        },
    }))
    canary_config.write_text("canary", encoding="utf-8")
    canary_env.write_text("CANARY=1\n", encoding="utf-8")
    canary_path.write_text(json.dumps({
        "status": "PASS", "frozen_recipe_sha256": VALIDATOR.sha256(frozen_path),
        "runner_tool_sha256": VALIDATOR.sha256(canary_tool),
        "config_sha256": VALIDATOR.sha256(canary_config),
        "env_sha256": VALIDATOR.sha256(canary_env),
        "evaluation": {"status": "PASS", "maximum_days": 45.0,
                       "projected_1t_days_from_p10": 30.0, "world_size": 4},
    }))
    qualification_path = tmp_path / "data/health/torchtitan-qualification.json"
    qualification_path.write_text(json.dumps({
        "status": "PASS", "qualification_identity": VALIDATOR.qualification_identity(tmp_path),
    }))
    canary = json.loads(canary_path.read_text())
    canary["qualification_identity"] = VALIDATOR.qualification_identity(tmp_path)
    canary_path.write_text(json.dumps(canary))
    assert set(VALIDATOR.validate_frozen_evidence(tmp_path)) == {
        "frozen_recipe_sha256", "post_freeze_plan_audit_sha256", "overlap_audit_sha256",
        "production_canary_sha256", "qualification_sha256",
    }
    del canary["qualification_identity"]
    canary_path.write_text(json.dumps(canary))
    try:
        VALIDATOR.validate_frozen_evidence(tmp_path)
    except ValueError as exc:
        assert "throughput canary" in str(exc)
    else:
        raise AssertionError("canary without current training identity was accepted")
    canary["qualification_identity"] = VALIDATOR.qualification_identity(tmp_path)
    canary_path.write_text(json.dumps(canary))
    (tmp_path / relative_files[-1]).write_text("drift", encoding="utf-8")
    try:
        VALIDATOR.validate_frozen_evidence(tmp_path)
    except ValueError as exc:
        assert "drift" in str(exc)
    else:
        raise AssertionError("post-audit config drift was accepted")


def test_production_canary_uses_p10_throughput_and_exact_training_contract():
    contract = {"tokens_per_step": 2_097_152, "steps": 200}
    result = {
        "status": "PASS",
        "hyperparameters": {"tokens_per_step": contract["tokens_per_step"],
                            "trained_steps": 200,
                            "trained_tokens": 200 * contract["tokens_per_step"]},
        "series": {
            "throughput(tps)": [{"step": i, "value": 100_000.0 + i} for i in range(40)],
            "mfu(%)": [{"step": i, "value": 40.0} for i in range(40)],
        },
    }
    report = CANARY.evaluate_result(result, contract)
    assert report["status"] == "PASS"
    assert report["p10_tps_per_device"] <= report["median_tps_per_device"]
    assert report["projected_1t_days_from_p10"] < 45
    result["hyperparameters"]["trained_tokens"] -= 1
    assert CANARY.evaluate_result(result, contract)["status"] == "FAIL"


def test_production_canary_rejects_stale_frozen_input(tmp_path):
    source = tmp_path / "selection.json"
    output = tmp_path / "pretrain.yaml"
    evidence = tmp_path / "result.json"
    for path in (source, output, evidence):
        path.write_text(path.name, encoding="utf-8")
    identity_path = tmp_path / "report/experiment_identity_audit_latest.json"
    identity_path.parent.mkdir(parents=True)
    identity_path.write_text(json.dumps({
        "status": "PASS", "evidence_snapshot_count": 1,
        "evidence_snapshot": {"result.json": CANARY.sha256(evidence)},
    }), encoding="utf-8")
    freeze_tool = tmp_path / "experiments/e4_curriculum/scripts/freeze_1t_recipe.py"
    freeze_tool.parent.mkdir(parents=True)
    freeze_tool.write_text("# freeze tool\n", encoding="utf-8")
    frozen = {
        "freeze_tool_sha256": CANARY.sha256(freeze_tool),
        "inputs": {"selection.json": CANARY.sha256(source)},
        "outputs": {"pretrain.yaml": CANARY.sha256(output)},
    }
    CANARY.validate_frozen_bindings(tmp_path, frozen)
    source.write_text("changed", encoding="utf-8")
    try:
        CANARY.validate_frozen_bindings(tmp_path, frozen)
    except ValueError as exc:
        assert "frozen input drift" in str(exc)
    else:
        raise AssertionError("stale frozen input was accepted by production canary")


def test_overlap_stat_binding_detects_replaced_order(tmp_path):
    order = tmp_path / "order.parquet"
    order.write_bytes(b"original")
    stat = order.stat()
    manifest = {"order.parquet": {
        "bytes": stat.st_size, "device": stat.st_dev,
        "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns,
    }}
    VALIDATOR.validate_stat_manifest(tmp_path, manifest, "order")
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"replaced")
    replacement.replace(order)
    try:
        VALIDATOR.validate_stat_manifest(tmp_path, manifest, "order")
    except ValueError as exc:
        assert "stat drift" in str(exc)
    else:
        raise AssertionError("replaced order retained stale overlap evidence")
