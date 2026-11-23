import importlib.util
import json
from pathlib import Path

import yaml
import pytest


ROOT = Path(__file__).resolve().parents[2]


def _module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


CONTRACTS = _module("formal_contracts", "pretrain/scripts/validate_formal_contracts.py")
WATCH = _module("pretrain_watch", "pretrain/scripts/pretrain_readiness_watch.py")
QUEUE = _module("experiment_queue", "pretrain/scripts/run_experiment_queue.py")
IDENTITIES = _module("experiment_identities", "pretrain/scripts/audit_experiment_identities.py")


@pytest.fixture(autouse=True)
def isolate_live_allocation_files(monkeypatch, tmp_path):
    monkeypatch.setattr(WATCH, "HEALTH", tmp_path / "health")
    import gpu_resource_policy as RESOURCES
    monkeypatch.setattr(RESOURCES, "visible_gpu_indices", lambda: set(range(8)))
    monkeypatch.setattr(WATCH, "allowed_gpu_indices", lambda _: set(range(8)))
    monkeypatch.setattr(WATCH, "resource_policy", lambda _: {
        "status": "ACTIVE", "allowed_gpu_indices": list(range(8)),
        "max_concurrent_jobs": 8, "allow_busy_gpu_sharing": True,
    })


def test_env_parser_rejects_shell_expressions(tmp_path):
    path = tmp_path / "bad.env"
    path.write_text("SAFE=value\nBAD=$(touch /tmp/nope)\n", encoding="utf-8")
    try:
        CONTRACTS.parse_env(path)
    except ValueError as exc:
        assert "shell expressions" in str(exc)
    else:
        raise AssertionError("unsafe env value was accepted")


def test_formal_contract_roster_adds_e3_only_after_atomic_generation_pass(tmp_path):
    assert CONTRACTS.expected_roster(tmp_path)["e3"] == 0
    report = tmp_path / "experiments/e3_data_mix/results/config_generation.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"status": "IN_PROGRESS"}))
    assert CONTRACTS.expected_roster(tmp_path)["e3"] == 0
    report.write_text(json.dumps({"status": "PASS"}))
    assert CONTRACTS.expected_roster(tmp_path)["e3"] == 8


def test_formal_contract_report_does_not_churn_for_timestamp_only(tmp_path):
    output = tmp_path / "contracts.json"
    CONTRACTS.atomic_write(output, json.dumps({"status": "PASS", "observed_at": "a"}))
    first = output.read_bytes()
    first_mtime = output.stat().st_mtime_ns
    CONTRACTS.atomic_write(output, json.dumps({"status": "PASS", "observed_at": "b"}))
    assert output.read_bytes() == first
    assert output.stat().st_mtime_ns == first_mtime


def test_phase_readiness_uses_config_hashes_instead_of_report_timestamp(tmp_path, monkeypatch):
    import os
    monkeypatch.setattr(WATCH, 'ROOT', tmp_path)
    cfg = tmp_path / 'pretrain/configs/batch_pilot/run.yaml'
    cfg.parent.mkdir(parents=True)
    cfg.write_text('lr: 0.0012\n')
    env = cfg.with_suffix('.env'); env.write_text('GPU=1\n')
    name = str(cfg.relative_to(tmp_path)); digest = WATCH.sha256(cfg)
    report = tmp_path / 'contracts.json'
    value = {'status': 'PASS', 'roster_exact': True, 'rows': [{
        'status': 'PASS', 'config': name, 'env': str(env.relative_to(tmp_path)),
        'config_sha256': digest, 'env_sha256': WATCH.sha256(env)}]}
    report.write_text(json.dumps(value)); os.utime(report, (1, 1))
    assert WATCH.contracts_cover_current_configs(report, {name: digest})
    cfg.write_text('lr: 0.0024\n')
    os.utime(report, None)
    assert not WATCH.contracts_cover_current_configs(report, {name: WATCH.sha256(cfg)})
    cfg.write_text('lr: 0.0012\n'); env.write_text('GPU=2\n')
    assert not WATCH.contracts_cover_current_configs(report, {name: digest})


def test_formal_experiment_names_and_e2_budget_are_exact(monkeypatch):
    monkeypatch.setattr(CONTRACTS, 'group_enabled', lambda *_: True)
    names = CONTRACTS.expected_name_roster(ROOT)
    counts = {group: len(values) for group, values in names.items()}
    assert counts == CONTRACTS.expected_roster(ROOT)
    configs = sorted((ROOT / "pretrain/configs/e2").glob("*.yaml"))
    assert {path.stem for path in configs} == names["e2"]
    for path in configs:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert value["train"]["budget_tokens"] == 1_000_000_000


def test_short_batch_pilots_have_validation_and_proportional_warmup():
    for path in sorted((ROOT / "pretrain/configs/batch_pilot").glob("*.yaml")):
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        steps = value["train"]["expected_iters"]
        warmup = value["optim"]["warmup_steps"]
        assert value["train"]["eval_every"] <= steps
        assert 5 <= warmup <= max(5, steps // 50)


def test_external_driver_blocker_does_not_consume_retry(monkeypatch, tmp_path):
    state_path = tmp_path / "actions.json"
    monkeypatch.setattr(WATCH, "ACTION_STATE", state_path)
    monkeypatch.setattr(WATCH.shutil, "which", lambda name: "/usr/bin/python3.12")
    events = WATCH.run_bootstrap_if_eligible(
        {"status": "blocked", "checks": {"nvidia_driver": False}}, True
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    action = state["actions"]["bootstrap_torchtitan_env"]
    assert events[0]["decision"] == "wait_for_driver"
    assert action["attempts"] == 0
    assert action["status"] == "WAITING_EXTERNAL"


def test_outer_queue_restart_renews_exhausted_job_but_preserves_total_attempts():
    job = {"status": "RETRY_EXHAUSTED", "attempts": len(QUEUE.BACKOFF),
           "total_attempts": 3, "next_retry_epoch": 1, "next_retry_at": "old"}
    assert QUEUE.renew_retry_generation(job)
    assert job["status"] == "PENDING"
    assert job["attempts"] == 0
    assert job["retry_generation"] == 1
    assert job["total_attempts"] == 3
    assert "next_retry_epoch" not in job


def test_outer_queue_restart_never_renews_passed_job():
    job = {"status": "PASS", "attempts": len(QUEUE.BACKOFF)}
    assert not QUEUE.renew_retry_generation(job)


def test_restarted_queue_reserves_gpu_of_adopted_live_launcher(monkeypatch):
    jobs = {
        "adopted": {"status": "RUNNING", "pid": 10, "gpu": 2, "env": "/tmp/a.env"},
        "local": {"status": "RUNNING", "pid": 11, "gpu": 3, "env": "/tmp/b.env"},
        "stale": {"status": "RUNNING", "pid": 12, "gpu": 1, "env": "/tmp/c.env"},
    }
    monkeypatch.setattr(QUEUE, "launch_process_matches", lambda pid, env: pid == 10)
    assert QUEUE.adopted_gpu_reservations(jobs, {"local"}) == {2}


def test_restarted_queue_forwards_signals_to_adopted_process_groups(monkeypatch):
    jobs = {
        "adopted": {"status": "RUNNING", "pid": 41, "gpu": 2,
                    "env": "/tmp/adopted.env"},
        "local": {"status": "RUNNING", "pid": 42, "gpu": 3,
                  "env": "/tmp/local.env"},
        "stale": {"status": "RUNNING", "pid": 43, "gpu": 4,
                  "env": "/tmp/stale.env"},
    }
    monkeypatch.setattr(
        QUEUE, "launch_process_matches",
        lambda pid, env: pid in {41, 42},
    )
    assert QUEUE.adopted_process_groups(jobs, {"local"}) == {41}


def test_experiment_scheduler_uses_up_to_all_eight_idle_gpus():
    gpus = [{"index": index, "memory_used_mb": 0} for index in range(10)]
    assert WATCH.select_experiment_gpus(gpus) == list(range(8))
    assert WATCH.select_experiment_gpus(gpus, {1, 3}) == [0, 2, 4, 5, 6, 7]


def test_shared_gpu_policy_is_fail_closed(monkeypatch, tmp_path):
    policy_path = tmp_path / "shared.json"
    monkeypatch.setattr(WATCH, "SHARED_GPU_POLICY", policy_path)
    valid = {
        "status": "ACTIVE_PROVISIONAL",
        "controls": {
            "gpu_indices": [7, 0, 6],
            "formal_acceptance_requires_later_torchtitan_qualification_pass": True,
            "single_gpu_contracts_only": True,
            "tokens_per_update_preserved": 262144,
            "selection_metrics_allowed": ["validation_loss"],
                "performance_metrics_excluded": [
                    "throughput(tps)", "mfu(%)", "wall_time",
                ],
                "max_concurrent_jobs": 3,
        },
    }
    policy_path.write_text(json.dumps(valid), encoding="utf-8")
    assert WATCH.active_shared_gpu_policy()["gpu_indices"] == [0, 6, 7]
    valid["controls"]["selection_metrics_allowed"] = ["throughput(tps)"]
    policy_path.write_text(json.dumps(valid), encoding="utf-8")
    assert WATCH.active_shared_gpu_policy() is None


def test_queue_allocation_uses_shared_lane_before_qualification():
    gpus = [{"index": index, "memory_used_mb": 40000} for index in range(8)]
    policy = {"gpu_indices": [0, 6, 7]}
    allocated, mode = WATCH.queue_gpu_allocation(gpus, False, policy)
    assert allocated == [0, 6, 7]
    assert mode == "provisional_shared"
    allocated, mode = WATCH.queue_gpu_allocation(gpus, False, policy, {6})
    assert allocated == [0, 7]
    assert mode == "provisional_shared"


def test_queue_allocation_prefers_formal_idle_lane_after_qualification():
    gpus = [{"index": index, "memory_used_mb": 0} for index in range(8)]
    allocated, mode = WATCH.queue_gpu_allocation(
        gpus, True, {"gpu_indices": [0, 6, 7]}
    )
    assert allocated == list(range(8))
    assert mode == "formal_idle"


def test_provisional_experiment_dag_preserves_result_dependencies():
    pending_qualification = {"status": "PENDING"}
    ready = WATCH.experiment_group_prerequisites(
        pending_qualification, True, {"status": "BLOCKED"}, True
    )
    assert ready["e1"][0] is True
    assert ready["e2"][0] is True
    ready = WATCH.experiment_group_prerequisites(
        pending_qualification, True, {"status": "PASS"}, True
    )
    assert ready["e2"][0] is True


def test_final_eight_gpu_gate_handoffs_only_checkpointed_shared_queue(monkeypatch, tmp_path):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "checkpoint/step-250"
    checkpoint.mkdir(parents=True)
    (checkpoint / ".metadata").write_text("complete")
    (checkpoint / "__0_0.distcp").write_text("state")
    env_path = tmp_path / "arm.env"
    env_path.write_text(f"YUFORMER_PRETRAIN_OUT={run_dir}\n")
    queue_path = tmp_path / "e1_pilot-queue.json"
    queue_path.write_text(json.dumps({
        "jobs": {"arm": {"status": "RUNNING", "env": str(env_path)}}
    }))
    monkeypatch.setattr(WATCH, "HEALTH", tmp_path)
    monkeypatch.setattr(
        WATCH, "discover_group_runner",
        lambda group, policy=None: 4242 if group == "e1_pilot" else 0,
    )
    signals = []
    monkeypatch.setattr(WATCH.os, "kill", lambda pid, sig: signals.append((pid, sig)))
    qualification = {"stages": {
        stage: {"status": "PASS"} for stage in WATCH.QUALIFICATION_PARTIAL_STAGES
    }}
    policy = {"gpu_indices": [0, 6, 7]}
    gpus = [{"index": index, "memory_used_mb": (40000 if index in {0, 6, 7} else 0)}
            for index in range(8)]
    event = WATCH.checkpointed_shared_handoff(qualification, gpus, policy, True)
    assert event["decision"] == "started"
    assert signals == [(4242, WATCH.signal.SIGTERM)]

    (checkpoint / ".metadata").unlink()
    event = WATCH.checkpointed_shared_handoff(qualification, gpus, policy, True)
    assert event["decision"] == "wait_for_checkpoints"


def test_adopted_completion_must_match_current_attempt(tmp_path):
    (tmp_path / "current_attempt.json").write_text(json.dumps({"attempt_id": "new"}))
    (tmp_path / "training_complete.json").write_text(json.dumps({"attempt_id": "old"}))
    assert not QUEUE.current_attempt_matches(tmp_path, "new")
    (tmp_path / "training_complete.json").write_text(json.dumps({"attempt_id": "new"}))
    assert QUEUE.current_attempt_matches(tmp_path, "new")


def test_queue_accepts_pass_only_from_current_attempt(tmp_path):
    (tmp_path / "current_attempt.json").write_text(json.dumps({"attempt_id": "new"}))
    (tmp_path / "training_complete.json").write_text(json.dumps({"attempt_id": "new"}))
    assert QUEUE.result_is_current_pass(tmp_path, {"status": "PASS", "attempt_id": "new"})
    assert not QUEUE.result_is_current_pass(
        tmp_path, {"status": "PASS", "attempt_id": "old"}
    )


def test_experiment_identity_audit_closes_attempt_evidence():
    identity = {"identity_sha256": "identity"}
    attempt = {"attempt_id": "attempt", "identity_sha256": "identity", "status": "PASS"}
    completion = {"attempt_id": "attempt"}
    result = {"attempt_id": "attempt"}
    assert IDENTITIES.attempt_binding_errors(identity, attempt, completion, result) == []
    result["attempt_id"] = "old"
    assert "attempt_id_mismatch" in IDENTITIES.attempt_binding_errors(
        identity, attempt, completion, result
    )


def test_experiment_identity_snapshot_detects_mutated_evidence(tmp_path):
    result = tmp_path / "result.json"
    result.write_text('{"status":"PASS"}\n', encoding="utf-8")
    snapshot = IDENTITIES.snapshot_artifacts(tmp_path, {result})
    assert snapshot == {"result.json": IDENTITIES.sha256(result)}
    result.write_text('{"status":"FAIL"}\n', encoding="utf-8")
    assert snapshot["result.json"] != IDENTITIES.sha256(result)


def test_identity_report_does_not_churn_hash_for_timestamp_only(tmp_path):
    output = tmp_path / "identity.json"
    IDENTITIES.atomic_json(output, {"status": "PASS", "observed_at": "first", "rows": []})
    first = output.read_bytes()
    first_mtime = output.stat().st_mtime_ns
    IDENTITIES.atomic_json(output, {"status": "PASS", "observed_at": "second", "rows": []})
    assert output.read_bytes() == first
    assert output.stat().st_mtime_ns == first_mtime
    IDENTITIES.atomic_json(output, {"status": "BLOCKED", "observed_at": "third", "rows": []})
    assert output.read_bytes() != first


def test_identity_failure_invalidates_derived_passes():
    pilot = {"status": "PASS"}
    analyses = {"e1": {"status": "PASS"}, "e2": {"status": "PASS"},
                "e3": {"status": "PASS"}}
    WATCH.invalidate_unbound_analyses(
        {"rows": [{"group": "e1_pilot", "status": "FAIL"},
                  {"group": "e2", "status": "FAIL"}]}, pilot, analyses
    )
    assert pilot == {"status": "BLOCKED", "blocker": "experiment_identity_audit"}
    assert analyses["e2"]["status"] == "BLOCKED"
    assert analyses["e1"]["status"] == "PASS"


def test_e2_training_can_overlap_e1_before_scaling_analysis():
    waiting = WATCH.experiment_group_prerequisites({"status": "PASS"}, True,
                                                   {"status": "BLOCKED"})
    assert waiting["e1"] == (True, "current_lr_resolution")
    assert waiting["e2"] == (True, "current_lr_resolution")

    ready = WATCH.experiment_group_prerequisites({"status": "PASS"}, True,
                                                 {"status": "PASS"})
    assert ready["e2"] == (True, "current_lr_resolution")


def test_monitor_cadence_is_sparse_only_for_stable_long_actions():
    assert WATCH.monitor_interval_seconds({"actions": [
        {"name": "e3_tokenization", "decision": "running"}
    ]}, 900, 7200) == 7200
    assert WATCH.monitor_interval_seconds({"actions": [
        {"name": "e3_tokenization", "decision": "started"}
    ]}, 900, 7200) == 900
    assert WATCH.monitor_interval_seconds({"actions": [
        {"name": "e3_tokenization", "decision": "stalled_suspect"}
    ]}, 900, 7200) == 900
    assert WATCH.monitor_interval_seconds({"actions": [
        {"name": "torchtitan_qualification", "decision": "running"}
    ]}, 900, 7200) == 900


def test_prefreeze_static_audit_cannot_unlock_final_materialization(monkeypatch, tmp_path):
    freeze_path = tmp_path / "freeze.json"
    audit_path = tmp_path / "audit.json"
    monkeypatch.setattr(WATCH, "ROOT", tmp_path)
    monkeypatch.setattr(WATCH, "FROZEN_RECIPE", freeze_path)
    monkeypatch.setattr(WATCH, "FINAL_PLAN_AUDIT", audit_path)
    audit_path.write_text("{}", encoding="utf-8")
    audit = {"status": "PASS_STATIC", "errors": [], "checks": {}, "stages": []}
    assert not WATCH.post_freeze_plan_audit_current({"status": "PENDING"}, audit)


def test_postfreeze_static_audit_must_bind_exact_frozen_outputs(monkeypatch, tmp_path):
    freeze_path = tmp_path / "freeze.json"
    audit_path = tmp_path / "audit.json"
    monkeypatch.setattr(WATCH, "ROOT", tmp_path)
    monkeypatch.setattr(WATCH, "FROZEN_RECIPE", freeze_path)
    monkeypatch.setattr(WATCH, "FINAL_PLAN_AUDIT", audit_path)
    paths = ["pretrain/configs/pretrain_base.yaml",
             "data-process/configs/mix_warmup_50b.yaml",
             "data-process/configs/mix_stable_800b.yaml",
             "data-process/configs/mix_anneal.yaml"]
    outputs = {}
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
        outputs[relative] = WATCH.sha256(path)
    freeze_path.write_text("{}", encoding="utf-8")
    audit_path.write_text("{}", encoding="utf-8")
    frozen = {"status": "PASS", "outputs": outputs}
    audit = {
        "status": "PASS_STATIC", "errors": [],
        "inputs": {"frozen_recipe": "freeze.json",
                   "frozen_recipe_sha256": WATCH.sha256(freeze_path)},
        "checks": {"optimizer_aligned_stage_boundaries": {
            "config": paths[0], "config_sha256": outputs[paths[0]]}},
        "stages": [{"mix_config": path, "mix_config_sha256": outputs[path]}
                   for path in paths[1:]],
    }
    assert WATCH.post_freeze_plan_audit_current(frozen, audit)
    audit["stages"][0]["mix_config_sha256"] = "stale"
    assert not WATCH.post_freeze_plan_audit_current(frozen, audit)


def test_identity_drift_is_quarantined_inside_experiment_root(monkeypatch, tmp_path):
    run_dir = tmp_path / "data/experiments/e2/e2_l16"
    run_dir.mkdir(parents=True)
    (run_dir / "result.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(WATCH, "ROOT", tmp_path)
    events = WATCH.quarantine_identity_drift({"rows": [{
        "group": "e2", "name": "e2_l16", "run_dir": str(run_dir),
        "errors": ["resolved_config_drift"],
    }]}, True)
    assert events[0]["decision"] == "archived"
    assert not run_dir.exists()
    assert Path(events[0]["archive"]).is_dir()


def test_transient_run_failure_is_not_quarantined(monkeypatch, tmp_path):
    run_dir = tmp_path / "data/experiments/e1_pilot/arm"
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoint").mkdir()
    monkeypatch.setattr(WATCH, "ROOT", tmp_path)
    events = WATCH.quarantine_identity_drift({"rows": [{
        "group": "e1_pilot", "name": "arm", "run_dir": str(run_dir),
        "errors": ["attempt_not_pass", "run_not_pass"],
    }]}, True)
    assert events == []
    assert run_dir.is_dir()


def test_identity_quarantine_defers_a_live_training_lease(monkeypatch, tmp_path):
    monkeypatch.setattr(WATCH, "ROOT", tmp_path)
    run_dir = tmp_path / "data/experiments/e1/live"
    lease = WATCH.RunLease(run_dir, "training").acquire()
    try:
        events = WATCH.quarantine_identity_drift({"rows": [{
            "run_dir": str(run_dir), "errors": ["training_source_drift"],
        }]}, True)
        assert events[0]["decision"] == "deferred_active_lease"
        lease.assert_owner()
    finally:
        lease.release()


def test_scheduler_reserves_gpus_for_exact_live_other_actions(monkeypatch, tmp_path):
    path = tmp_path / "actions.json"
    monkeypatch.setattr(WATCH, "ACTION_STATE", path)
    monkeypatch.setattr(WATCH, "runner_matches", lambda pid, script: pid in {10, 11})
    path.write_text(json.dumps({"actions": {
        "e1_queue": {"status": "RUNNING", "pid": 10, "gpus": [0, 6]},
        "torchtitan_qualification": {"status": "RUNNING", "pid": 11, "gpus": [1, 2, 3, 4]},
        "e2_queue": {"status": "RUNNING", "pid": 12, "gpus": [7]},
    }}))
    assert WATCH.active_gpu_reservations("e1_queue") == {1, 2, 3, 4}
    assert WATCH.active_gpu_reservations() == {0, 1, 2, 3, 4, 6}


def test_performance_queue_can_use_one_idle_gpu_and_rejects_shared_slots(monkeypatch, tmp_path):
    from types import SimpleNamespace
    monkeypatch.setattr(WATCH, "ACTION_STATE", tmp_path / "actions.json")
    monkeypatch.setattr(WATCH, "PERFORMANCE_QUEUE_REPORT", tmp_path / "performance.json")
    monkeypatch.setattr(WATCH, "discover_group_runner", lambda *args: 0)
    monkeypatch.setattr(WATCH, "active_gpu_reservations", lambda *args: set())
    commands = []

    def popen(argv, **kwargs):
        commands.append(argv)
        return SimpleNamespace(pid=12345)

    monkeypatch.setattr(WATCH.subprocess, "Popen", popen)
    _, event = WATCH.start_group_if_eligible(
        "e1_performance", True, "resolved", {"status": "PENDING"},
        [{"index": 0, "memory_used_mb": 40000}, {"index": 2, "memory_used_mb": 0}], True,
        shared_policy={"gpu_indices": [0]}, qualification_passed=True)
    assert event["decision"] == "started" and event["gpus"] == [2]
    assert "--provisional-loss-only" not in commands[0]


def test_queue_pass_requires_current_result_roster(monkeypatch, tmp_path):
    env_dir = tmp_path / "pretrain/configs/e1"
    run_dir = tmp_path / "runs/arm"
    env_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (env_dir / "arm.env").write_text(
        f"YUFORMER_PRETRAIN_OUT={run_dir}\n", encoding="utf-8"
    )
    monkeypatch.setattr(WATCH, "ROOT", tmp_path)
    assert not WATCH.experiment_result_roster_complete("e1")
    (run_dir / "result.json").write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    assert WATCH.experiment_result_roster_complete("e1")
    (run_dir / "result.json").unlink()
    assert not WATCH.experiment_result_roster_complete("e1")


def test_stale_queue_pass_is_reopened_for_bounded_retry(tmp_path):
    report = tmp_path / "queue.json"
    record = {"status": "PASS", "attempts": 3, "pid": 999,
              "next_retry_epoch": 123, "stale_observations": 2}
    queue = WATCH.invalidate_stale_queue_pass(
        report, {"status": "PASS", "jobs": {"arm": {"status": "PASS"}}}, record
    )
    assert queue["status"] == "STALE_RESULT_EVIDENCE"
    assert record["status"] == "RETRY_TRIGGERED"
    assert record["attempts"] == 0
    assert "pid" not in record and "next_retry_epoch" not in record
    assert json.loads(report.read_text())["status"] == "STALE_RESULT_EVIDENCE"


def test_identity_auditor_expands_only_declared_environment(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("path: ${BOUND}\n", encoding="utf-8")
    assert IDENTITIES.expand_config(config, {"BOUND": "/exact"}) == "path: /exact\n"
    try:
        IDENTITIES.expand_config(config, {})
    except ValueError as exc:
        assert "unresolved variables" in str(exc)
    else:
        raise AssertionError("unresolved experiment config was accepted")


def test_identity_auditor_uses_exact_named_rosters(monkeypatch):
    monkeypatch.setattr(IDENTITIES, 'group_enabled', lambda *_: True)
    names = IDENTITIES.expected_names(ROOT)
    assert {group: len(names[group]) for group in IDENTITIES.EXPECTED} == IDENTITIES.EXPECTED
    declared = json.loads((ROOT / 'pretrain/configs/lr_pilot_search.json').read_text())['extensions']
    assert names['e1_pilot_extension'] == {row['run_name'] for row in declared}
    assert {'e1_lr_200m_ext1800', 'e1_lr_200m_ext2400', 'e1_lr_200m_ext4800'}.issubset(names['e1_pilot_extension'])
    assert "e1_ld_300m_9p85b" in names["e1"]
    assert "e2_l32_s3" in names["e2"]
    assert names["e3"] == {
        "e3_m2_base", "e3_code_2x", "e3_math_2x", "e3_chinese_2x",
        "e3_academic_2x", "e3_encyclopedia_2x", "e3_m1_native", "e3_m3_quality",
    }


def test_stale_qualification_pass_is_not_reused(monkeypatch, tmp_path):
    action_state = tmp_path / "actions.json"
    report_path = tmp_path / "qualification.json"
    action_state.write_text(json.dumps({
        "actions": {"torchtitan_qualification": {
            "status": "PASS", "attempts": 3, "pid": 999999999,
            "next_retry_epoch": 123,
        }}
    }))
    report_path.write_text(json.dumps({
        "status": "PASS", "stages": {},
        "qualification_identity": {"schema_version": 1, "sha256": "old"},
    }))
    current = {"schema_version": 1, "sha256": "new", "files": []}
    monkeypatch.setattr(WATCH, "ACTION_STATE", action_state)
    monkeypatch.setattr(WATCH, "QUALIFICATION_REPORT", report_path)
    monkeypatch.setattr(WATCH, "qualification_identity", lambda root: current)
    report, event = WATCH.start_qualification_if_eligible(
        {"status": "ready"}, {"status": "PASS"}, [], False
    )
    assert report["status"] == "STALE"
    assert event["decision"] == "disabled"
    record = json.loads(action_state.read_text())["actions"]["torchtitan_qualification"]
    assert record["status"] == "STALE"
    assert record["attempts"] == 0
    assert "pid" not in record


def test_qualification_starts_partial_gates_with_four_idle_gpus(monkeypatch, tmp_path):
    action_state = tmp_path / "actions.json"
    report_path = tmp_path / "qualification.json"
    report_path.write_text(json.dumps({"status": "PENDING", "stages": {}}))
    monkeypatch.setattr(WATCH, "ACTION_STATE", action_state)
    monkeypatch.setattr(WATCH, "QUALIFICATION_REPORT", report_path)
    monkeypatch.setattr(WATCH, "qualification_identity",
                        lambda root: {"schema_version": 1, "sha256": "current", "files": []})

    class Process:
        pid = 1234

    monkeypatch.setattr(WATCH.subprocess, "Popen", lambda *args, **kwargs: Process())
    gpus = [{"index": index, "memory_used_mb": 0} for index in range(6)]
    _, event = WATCH.start_qualification_if_eligible(
        {"status": "ready"}, {"status": "PASS"}, gpus, True
    )
    assert event["decision"] == "started"
    assert event["required_gpus"] == 4
    assert event["gpus"] == [0, 1, 2, 3]


def test_qualification_reserves_eight_gpus_only_for_final_reshard(monkeypatch, tmp_path):
    action_state = tmp_path / "actions.json"
    report_path = tmp_path / "qualification.json"
    partial = {
        stage: {"status": "PASS"} for stage in (
            "lr_schedule_contract", "numerical_parity", "distributed_gradient_parity", "single_gpu", "fsdp4_dcp",
            "recovery_reference", "train_interrupt_resume", "save_interrupt_resume",
            "load_interrupt_resume", "reshard_4to2",
        )
    }
    report_path.write_text(json.dumps({"status": "IN_PROGRESS", "stages": partial}))
    monkeypatch.setattr(WATCH, "ACTION_STATE", action_state)
    monkeypatch.setattr(WATCH, "QUALIFICATION_REPORT", report_path)
    monkeypatch.setattr(WATCH, "qualification_identity",
                        lambda root: {"schema_version": 1, "sha256": "current", "files": []})
    gpus = [{"index": index, "memory_used_mb": 0} for index in range(6)]
    _, event = WATCH.start_qualification_if_eligible(
        {"status": "ready"}, {"status": "PASS"}, gpus, True
    )
    assert event["decision"] == "wait_for_idle_gpus"
    record = json.loads(action_state.read_text())["actions"]["torchtitan_qualification"]
    assert record["required_gpus"] == 8
    assert record["blocker"] == "eight_idle_gpus"


def test_partial_qualification_receives_all_eight_when_already_idle(monkeypatch, tmp_path):
    action_state = tmp_path / "actions.json"
    report_path = tmp_path / "qualification.json"
    report_path.write_text(json.dumps({"status": "PENDING", "stages": {}}))
    monkeypatch.setattr(WATCH, "ACTION_STATE", action_state)
    monkeypatch.setattr(WATCH, "QUALIFICATION_REPORT", report_path)
    monkeypatch.setattr(WATCH, "qualification_identity",
                        lambda root: {"schema_version": 1, "sha256": "current", "files": []})

    class Process:
        pid = 1234

    monkeypatch.setattr(WATCH.subprocess, "Popen", lambda *args, **kwargs: Process())
    gpus = [{"index": index, "memory_used_mb": 0} for index in range(8)]
    _, event = WATCH.start_qualification_if_eligible(
        {"status": "ready"}, {"status": "PASS"}, gpus, True
    )
    assert event["required_gpus"] == 4
    assert event["gpus"] == list(range(8))


def test_live_reused_pid_is_not_accepted_as_qualification_runner(monkeypatch, tmp_path):
    action_state = tmp_path / "actions.json"
    report_path = tmp_path / "qualification.json"
    action_state.write_text(json.dumps({
        "actions": {"torchtitan_qualification": {
            "status": "RUNNING", "attempts": 1, "pid": 42,
        }}
    }))
    report_path.write_text(json.dumps({"status": "PENDING", "stages": {}}))
    monkeypatch.setattr(WATCH, "ACTION_STATE", action_state)
    monkeypatch.setattr(WATCH, "QUALIFICATION_REPORT", report_path)
    monkeypatch.setattr(WATCH, "qualification_identity",
                        lambda root: {"schema_version": 1, "sha256": "current", "files": []})
    monkeypatch.setattr(WATCH, "runner_matches", lambda pid, script: False)
    _, event = WATCH.start_qualification_if_eligible(
        {"status": "blocked"}, {"status": "PASS"}, [], False
    )
    assert event["decision"] == "disabled"
    record = json.loads(action_state.read_text())["actions"]["torchtitan_qualification"]
    assert "pid" not in record


def test_final_tokenization_requires_all_streams_and_current_repair():
    names = [f"stream-{index}" for index in range(12)]
    value = {
        "status": "PASS", "expected_jobs": names,
        "jobs": {name: {"status": "PASS"} for name in names},
        "repaired_audit_sha256": "audit-b",
    }
    assert WATCH.final_tokenization_complete(value, None)
    assert WATCH.final_tokenization_complete(value, "audit-b")
    assert not WATCH.final_tokenization_complete(value, "audit-a")
    value["jobs"][names[-1]]["status"] = "FAIL"
    assert not WATCH.final_tokenization_complete(value, None)

    e3_names = [f"e3-stream-{index}" for index in range(17)]
    e3 = {"status": "PASS", "expected_jobs": e3_names,
          "jobs": {name: {"status": "PASS"} for name in e3_names}}
    assert WATCH.tokenization_complete(e3, 17, None)
    assert not WATCH.tokenization_complete(e3, 12, None)


def test_background_retry_trigger_resets_exhausted_generation(monkeypatch, tmp_path):
    state_path = tmp_path / "actions.json"
    pid_path = tmp_path / "action.pid"
    state_path.write_text(json.dumps({"actions": {"repairable": {
        "status": "RETRY_EXHAUSTED", "attempts": WATCH.MAX_ATTEMPTS,
        "retry_trigger": "old", "next_retry_epoch": 9999999999,
    }}}))
    monkeypatch.setattr(WATCH, "ACTION_STATE", state_path)
    completion, event = WATCH.start_background_action(
        action_name="repairable", prerequisite_ready=True, blocker="none",
        completion={"status": "FAIL"},
        completion_predicate=lambda value: value.get("status") == "PASS",
        pid_path=pid_path, argv=["python", "/tmp/repair.py"],
        log_path=tmp_path / "repair.log", enabled=False, retry_trigger="new",
    )
    assert completion["status"] == "FAIL"
    assert event["decision"] == "disabled"
    record = json.loads(state_path.read_text())["actions"]["repairable"]
    assert record["attempts"] == 0
    assert record["status"] == "RETRY_TRIGGERED"
    assert record["retry_trigger"] == "new"


def test_new_dependency_trigger_invalidates_old_pass(monkeypatch, tmp_path):
    state_path = tmp_path / "actions.json"
    state_path.write_text(json.dumps({"actions": {"derived": {
        "status": "PASS", "attempts": 1, "retry_trigger": "old",
    }}}))
    monkeypatch.setattr(WATCH, "ACTION_STATE", state_path)
    completion, event = WATCH.start_background_action(
        action_name="derived", prerequisite_ready=True, blocker="source",
        completion={"status": "PASS"},
        completion_predicate=lambda value: value.get("status") == "PASS",
        pid_path=tmp_path / "derived.pid", argv=["python", "/tmp/derive.py"],
        log_path=tmp_path / "derive.log", enabled=False, retry_trigger="new",
    )
    assert completion["status"] == "PASS"
    assert event["decision"] == "disabled"
    record = json.loads(state_path.read_text())["actions"]["derived"]
    assert record["status"] == "RETRY_TRIGGERED"
    assert record["attempts"] == 0


def test_plan_reports_contract_count(monkeypatch, tmp_path):
    monkeypatch.setattr(WATCH, "PLAN_PATH", tmp_path / "plan.json")
    monkeypatch.setattr(WATCH, "ROOT", tmp_path)
    monkeypatch.setattr(WATCH, "STREAM_AUDIT", tmp_path / "audit.json")
    WATCH.write_plan(
        {"status": "PASS", "roster_exact": True},
        {"status": "PASS", "passed": 78, "total": 78, "roster_exact": True},
        {"status": "blocked", "failed_checks": ["nvidia_driver"]},
        [],
        {"status": "BLOCKED", "observed_results": 0},
        {"status": "PENDING", "stages": {}},
        {"status": "PENDING", "jobs": {}},
        {"status": "BLOCKED", "observed_results": 0},
        {"status": "PENDING", "jobs": {}},
        {"status": "PASS", "arms": [{}] * 8},
        {"status": "BLOCKED", "observed_results": 0},
        {"status": "PENDING", "jobs": {}},
        {"status": "BLOCKED", "observed_results": 0},
        {"status": "PENDING", "jobs": {}},
        {"status": "BLOCKED", "observed_results": 0},
        {"status": "PENDING", "jobs": {}},
        {"status": "PENDING"},
        {"status": "PENDING"},
        {"status": "PASS_STATIC", "errors": []},
        {"status": "PENDING", "jobs": {}},
        {"status": "BLOCKED_STORAGE", "jobs": {},
         "storage_preflight": {"status": "BLOCKED", "required_free_bytes": 100}},
        {"status": "PENDING", "streams": []},
        {"status": "PENDING", "checks": {}},
        {"status": "PENDING"},
    )
    plan = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
    assert plan["stages"][1]["status"] == "PASS_STATIC"
    assert plan["stages"][1]["checks"]["formal_contracts_pass"] == 78
    by_id = {stage["id"]: stage for stage in plan["stages"]}
    assert by_id["P5"]["training_depends_on"] == ["P3"]
    assert by_id["P5"]["analysis_depends_on"] == ["P4"]
    assert by_id["P7"]["status"] == "PASS_STATIC"
    assert by_id["I1"]["depends_on"] == ["P2", "P6"]
    assert by_id["P8-P10"]["depends_on"] == ["P7", "I1"]
    assert by_id["P8-P10"]["status"] == "BLOCKED_STORAGE"
    assert by_id["P8-P10"]["storage_preflight"]["required_free_bytes"] == 100
