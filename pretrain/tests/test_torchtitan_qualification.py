import importlib.util
import json
import time
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_torchtitan_qualification.py"
SPEC = importlib.util.spec_from_file_location("run_torchtitan_qualification", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_runner_completes_fault_and_reshard_publication(monkeypatch, tmp_path):
    identity = {"schema_version": 1, "sha256": "test", "files": []}
    initial = {"qualification_identity": identity, "stages": {
        name: {"status": "PASS"} for name in ("lr_schedule_contract", "numerical_parity", "attention_backend_parity",
        "distributed_gradient_parity", "single_gpu", "fsdp4_dcp", "recovery_reference")}}
    monkeypatch.setattr(MODULE, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(MODULE, "REPORT", tmp_path / "report.json")
    monkeypatch.setattr(MODULE, "load_report", lambda: initial)
    monkeypatch.setattr(MODULE, "qualification_identity", lambda root: identity)
    monkeypatch.setattr(MODULE, "acquire_runner_lease", lambda: SimpleNamespace(
        release=lambda: None, assert_owner=lambda: None))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--gpus", "0,1,2,3,4,5,6,7"])
    monkeypatch.setattr(MODULE, "launch", lambda name, gpus, **kwargs: (0, {
        "killed_at_step": 4 if kwargs.get("inject_mode") else None,
        "pending_checkpoint_step": 8 if kwargs.get("inject_mode") == "saving" else None,
        "pending_checkpoint_complete_after_kill": False,
        "injection_mode": kwargs.get("inject_mode"), "log": "test.log"}))
    monkeypatch.setattr(MODULE, "collect", lambda name: (0, {"status": "PASS"}))
    monkeypatch.setattr(MODULE, "verify_cursor", lambda *args, **kwargs: {"status": "PASS"})
    monkeypatch.setattr(MODULE, "compare_loss", lambda *args, **kwargs: {"status": "PASS"})
    assert MODULE.main() == 0
    report = json.loads(MODULE.REPORT.read_text())
    assert report["status"] == "PASS"
    assert len(report["stages"]) == 13
    assert report["stages"]["train_interrupt_resume"]["injection_mode"] == "training"


def test_fault_injection_requires_a_complete_checkpoint():
    assert MODULE.injection_decision("checkpoint", [], "")[0] is False
    assert MODULE.injection_decision("checkpoint", [4], "")[0] is True
    assert MODULE.injection_decision("loading", [], "Loading the checkpoint from x")[0] is False


def test_training_interrupt_occurs_after_saved_step():
    assert MODULE.injection_decision("training", [4], "step: 4")[0] is False
    assert MODULE.injection_decision("training", [4], "step: 5")[0] is True


def test_save_interrupt_ignores_repeated_rank_staging_and_completed_requests():
    text = "Staging the checkpoint.\n" * 8
    text += "Received async checkpoint request with id=/run/checkpoint/step-4\n" * 8
    assert MODULE.injection_decision("saving", [4], text)[0] is False


def test_save_interrupt_requires_new_incomplete_async_request():
    text = "Received async checkpoint request with id=/run/checkpoint/step-8\n"
    assert MODULE.injection_decision("saving", [], text)[0] is False
    assert MODULE.injection_decision("saving", [4], text) == (True, 4)
    assert MODULE.injection_decision("saving", [4, 8], text) == (False, 8)


def test_save_interrupt_rejects_metadata_completion_race():
    evidence = {"killed_at_step": 4, "pending_checkpoint_step": 8,
                "pending_checkpoint_complete_after_kill": False}
    assert MODULE.saving_interrupt_observed(evidence)
    evidence["pending_checkpoint_complete_after_kill"] = True
    assert not MODULE.saving_interrupt_observed(evidence)
    evidence.pop("pending_checkpoint_complete_after_kill")
    assert not MODULE.saving_interrupt_observed(evidence)


def test_completed_request_log_cannot_trigger_save_interrupt():
    text = "Completed checkpoint save request for checkpoint_id='/run/checkpoint/step-8'\n"
    assert MODULE.injection_decision("saving", [4], text) == (False, 4)


def test_explicit_writer_requires_actual_new_shards_and_a_complete_baseline(tmp_path):
    text = "Received async checkpoint request with id=None\n"
    directory = tmp_path/"checkpoint/step-8"
    directory.mkdir(parents=True)
    shard = directory/"__0_0.distcp"
    shard.write_bytes(b"old incomplete shard")
    initial = MODULE.shard_write_snapshot(tmp_path)
    assert MODULE.active_write_steps(tmp_path, initial) == ()
    assert MODULE.injection_decision("saving", [4], text) == (False, 4)
    shard.write_bytes(b"actually writing a new shard payload")
    writing = MODULE.active_write_steps(tmp_path, initial)
    assert writing == (8,)
    assert MODULE.injection_decision("saving", [], text, writing) == (False, None)
    assert MODULE.injection_decision("saving", [4], text, writing) == (True, 4)
    assert MODULE.injection_decision("saving", [4], "", writing) == (False, 4)
    (directory/".metadata").write_bytes(b"published")
    assert MODULE.active_write_steps(tmp_path, initial) == ()
    assert MODULE.injection_decision("saving", [4, 8], text, writing) == (False, 8)


def test_empty_shards_cannot_be_mistaken_for_active_save(tmp_path):
    directory = tmp_path/"checkpoint/step-8"
    directory.mkdir(parents=True)
    (directory/"__0_0.distcp").touch()
    assert MODULE.active_write_steps(tmp_path, {}) == ()


def test_load_interrupt_detects_restore_after_checkpoint_exists():
    decision, step = MODULE.injection_decision(
        "loading", [4, 8], "Loading the checkpoint from checkpoint/step-8."
    )
    assert decision is True
    assert step == 8


def test_cursor_evidence_requires_every_resumed_rank(monkeypatch, tmp_path):
    monkeypatch.setattr(MODULE, "RUNS", tmp_path)
    audit = tmp_path / "run/cursor_audit"
    audit.mkdir(parents=True)
    for rank in range(2):
        (audit / f"load-16-world2-rank{rank}.json").write_text(json.dumps({
            "consumed_samples": 16, "dp_world_size": 2, "dp_rank": rank,
            "local_batch_size": 1, "resumed_dataloader_step": 8, "seq_len": 4096,
        }))
    result = MODULE.verify_cursor(
        "run", source_world=4, resumed_world=2, checkpoint_step=4
    )
    assert result["status"] == "PASS"
    (audit / "load-16-world2-rank1.json").unlink()
    assert MODULE.verify_cursor(
        "run", source_world=4, resumed_world=2, checkpoint_step=4
    )["status"] == "FAIL"


def test_recovery_loss_parity_uses_post_checkpoint_overlap(monkeypatch, tmp_path):
    monkeypatch.setattr(MODULE, "RUNS", tmp_path)
    key = "loss_metrics/global_avg_loss"
    for name, offset in (("reference", 0.0), ("resumed", 0.001)):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "result.json").write_text(json.dumps({
            "series": {key: [{"step": step, "value": 3.0 - step * 0.01 + offset}
                              for step in range(1, 13)]}
        }))
    result = MODULE.compare_loss("reference", "resumed", after_step=4)
    assert result["status"] == "PASS"
    assert result["steps"] == list(range(5, 13))


@pytest.mark.parametrize('fault', ['missing_middle', 'truncated_both', 'only_one_common_step', 'nan', 'infinity'])
def test_recovery_loss_parity_rejects_incomplete_or_nonfinite_trajectories(monkeypatch, tmp_path, fault):
    monkeypatch.setattr(MODULE, 'RUNS', tmp_path)
    points = {name: [{'step': step, 'value': 3.0-step*.01} for step in range(1, 25)]
              for name in ('reference', 'resumed')}
    if fault == 'missing_middle':
        points['resumed'] = [p for p in points['resumed'] if p['step'] != 16]
    elif fault == 'truncated_both':
        points = {name: values[:23] for name, values in points.items()}
    elif fault == 'only_one_common_step':
        points['resumed'] = [points['resumed'][-1]]
    else:
        points['resumed'][15]['value'] = float('nan' if fault == 'nan' else 'inf')
    for name, values in points.items():
        directory = tmp_path/name
        directory.mkdir()
        (directory/'result.json').write_text(json.dumps({'series': {'loss_metrics/global_avg_loss': values}}))
    result = MODULE.compare_loss('reference', 'resumed', after_step=8, through_step=24)
    assert result['status'] == 'FAIL'
    assert result['expected_steps'] == list(range(9, 25))
    if fault in {'nan', 'infinity'}:
        assert result['all_losses_finite'] is False
        assert result['max_abs_loss_delta'] is None
    else:
        assert result['missing_resumed_steps']


def test_recovery_loss_parity_requires_observations_after_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(MODULE, 'RUNS', tmp_path)
    result = MODULE.compare_loss('reference', 'resumed', after_step=24, through_step=24)
    assert result['status'] == 'FAIL'
    assert result['expected_steps'] == []


def test_cursor_accounts_for_production_microbatch_and_accumulation(monkeypatch, tmp_path):
    monkeypatch.setattr(MODULE, "RUNS", tmp_path)
    audit = tmp_path / "run/cursor_audit"
    audit.mkdir(parents=True)
    for rank in range(8):
        (audit / f"load-512-world8-rank{rank}.json").write_text(json.dumps({
            "consumed_samples": 512, "dp_world_size": 8, "dp_rank": rank,
            "local_batch_size": 8, "resumed_dataloader_step": 8,
        }))
    result = MODULE.verify_cursor("run", source_world=8, resumed_world=8,
                                  checkpoint_step=4, local_batch_size=8, grad_accum=2)
    assert result["status"] == "PASS"
    assert result["expected_consumed_samples"] == 512
    assert result["expected_resumed_dataloader_step"] == 8


def test_recovery_fault_contracts_match_uninterrupted_reference():
    reference = yaml.safe_load(
        (MODULE.CONFIGS / "q3_recovery_reference.yaml").read_text()
    )
    for name in ("q3_train_interrupt", "q3_save_interrupt", "q3_load_interrupt"):
        candidate = yaml.safe_load((MODULE.CONFIGS / f"{name}.yaml").read_text())
        for key in ("seed", "model", "data", "optim", "train", "parallel"):
            assert candidate[key] == reference[key], f"{name} differs at {key}"


def test_runner_lease_fences_a_second_qualification_writer(monkeypatch, tmp_path):
    monkeypatch.setattr(MODULE, "RUNS", tmp_path)
    first = MODULE.acquire_runner_lease()
    try:
        with pytest.raises(SystemExit, match="already active"):
            MODULE.acquire_runner_lease()
    finally:
        first.release()

    successor = MODULE.acquire_runner_lease()
    successor.release()


def test_runner_lease_fences_report_publish_after_takeover(monkeypatch, tmp_path):
    monkeypatch.setattr(MODULE, "RUNS", tmp_path / "runs")
    lease = MODULE.acquire_runner_lease()
    monkeypatch.setattr(MODULE, "_RUNNER_LEASE", lease)
    try:
        successor = lease.lock_root / "generation-00000000000000000002"
        successor.mkdir()
        token = "successor"
        (successor / "owner.json").write_text(json.dumps({
            "token": token, "started_unix": time.time(),
        }))
        (successor / f"heartbeat-{token}").touch()
        with pytest.raises(MODULE.ConfigError, match="generation fence"):
            MODULE.atomic_json(tmp_path / "report.json", {"status": "PASS"})
        assert not (tmp_path / "report.json").exists()
    finally:
        lease.release()


def test_qualification_identity_changes_with_source(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    backend = root / "pretrain/backends/torchtitan"
    scripts = root / "pretrain/scripts"
    configs = root / "pretrain/configs/qualification"
    model_src = root / "model/src"
    model_configs = root / "model/configs"
    for directory in (backend, scripts, configs, model_src, model_configs):
        directory.mkdir(parents=True, exist_ok=True)
    for name in ("run_torchtitan_qualification.py", "collect_torchtitan_result.py",
                 "run_with_lease.py", "assert_run_lease.py", "qualification_identity.py"):
        (scripts / name).write_text(f"# {name}\n")
    (backend / "adapter.py").write_text("VERSION = 1\n")
    (model_src / "model.py").write_text("MODEL = 1\n")
    (model_configs / "model_configs.yaml").write_text("models: {}\n")
    first = MODULE.qualification_identity(root)
    assert MODULE.report_matches({"qualification_identity": first}, first)
    (backend / "adapter.py").write_text("VERSION = 2\n")
    second = MODULE.qualification_identity(root)
    assert first["sha256"] != second["sha256"]
    assert not MODULE.report_matches({"qualification_identity": first}, second)
