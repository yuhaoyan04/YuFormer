import json
from types import SimpleNamespace

import pytest

import run_authorized_pretrain as worker


@pytest.mark.parametrize("errors,allowed", [(["training_source_drift"], True),
                                            (["tokenizer_hash_mismatch"], False)])
def test_identity_exit_two_allows_only_the_recorded_source_drift_waiver(tmp_path, monkeypatch, errors, allowed):
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    path = tmp_path / "report/experiment_identity_audit_latest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"status": "BLOCKED", "roster_exact": True,
                                "total": 1, "rows": [{"status": "FAIL", "errors": errors}]}))
    pipeline = object.__new__(worker.Pipeline)
    calls = []
    pipeline.script = lambda *args, **kwargs: calls.append(kwargs)
    if allowed:
        pipeline.audit_identity()
    else:
        with pytest.raises(RuntimeError, match="outside the explicit source-drift waiver"):
            pipeline.audit_identity()
    assert calls[0]["accepted"] == (0, 2)


def test_runtime_binary_does_not_bypass_in_progress_environment_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    binary = tmp_path / "bin/python"
    binary.parent.mkdir()
    binary.touch()
    monkeypatch.setattr(worker, "PYTHON", binary)
    monkeypatch.setattr(worker.time, "sleep", lambda seconds: None)
    state = tmp_path / "data/health/torchtitan-env-prepare.json"
    state.parent.mkdir(parents=True)
    pipeline = object.__new__(worker.Pipeline)
    calls = []

    def command(stage, argv, **kwargs):
        calls.append(stage)
        if stage == "prepare_environment":
            status = "PASS" if calls.count(stage) == 2 else "RUNNING"
            state.write_text(json.dumps({"status": status}))

    pipeline.command = command
    pipeline.script = lambda stage, *args, **kwargs: calls.append(stage)
    pipeline.environment()
    assert calls == ["prepare_environment", "prepare_environment", "userspace", "preflight"]


def test_resume_preserves_valid_frozen_recipe_and_materialized_data(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(worker, "load_authorization", lambda root: {"training": {"world_size": 8}})
    pipeline = object.__new__(worker.Pipeline)
    pipeline.state = {}
    events = []
    pipeline.publish = lambda: None
    pipeline.environment = lambda: None
    pipeline.qualify = lambda label: events.append(label)
    pipeline.frozen_recipe_current = lambda: True
    pipeline.production_assets_current = lambda: True
    pipeline.production_canary_current = lambda: True
    pipeline.probes = lambda auth: pytest.fail("a valid frozen recipe must not be replaced")
    pipeline.wait_for_gpus = lambda: None
    pipeline.require = lambda path, status: {"status": status}
    pipeline.script = lambda stage, *a, **k: events.append(stage)
    pipeline.command = lambda stage, *a, **k: events.append(stage)
    pipeline.run()
    assert pipeline.state["frozen_recipe_reused"] is True
    assert pipeline.state["production_assets_reused"] is True
    assert pipeline.state["production_canary_reused"] is True
    assert not set(events) & {"freeze", "production_canary", "orders", "release", "tokens", "stream_audit", "overlap_and_mixture"}
    assert events[-1] == "full_pretrain"


def test_existing_checkpoint_blocks_refreeze_when_recipe_bindings_changed(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(worker, "load_authorization", lambda root: {"training": {"world_size": 8}})
    env = tmp_path / "pretrain/configs/pretrain_base.env"
    env.parent.mkdir(parents=True)
    env.write_text("YUFORMER_PRETRAIN_OUT=runs/full\n")
    metadata = tmp_path / "runs/full/checkpoint/step-100/.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_bytes(b"existing checkpoint")
    pipeline = object.__new__(worker.Pipeline)
    pipeline.state = {}
    pipeline.environment = lambda: None
    pipeline.qualify = lambda label: None
    pipeline.frozen_recipe_current = lambda: False
    pipeline.probes = lambda auth: pytest.fail("cannot mutate evidence for an existing checkpoint")
    with pytest.raises(RuntimeError, match="refusing to refreeze or rebuild data"):
        pipeline.run()
    assert metadata.read_bytes() == b"existing checkpoint"


@pytest.mark.parametrize("tamper", [False, True])
def test_canary_reuse_requires_current_bound_contract_and_result(tmp_path, monkeypatch, tamper):
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(worker, "allowed_gpu_indices", lambda _: set(range(8)))
    result = tmp_path / "canary/result.json"
    result.parent.mkdir()
    result.write_text('{"status":"PASS"}')
    report = tmp_path / "data/health/production-throughput-canary.json"
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({"status": "PASS", "bound": "current", "gpu_indices": list(range(8)),
                                  "result_sha256": worker.digest(result), "evaluation": {"status": "PASS"}}))
    canary = SimpleNamespace(ROOT=tmp_path, RUN_DIR=result.parent,
                             build_contract=lambda root: {"bound": "current"},
                             evaluate_result=lambda *args: {"status": "PASS"})
    monkeypatch.setattr(worker, "module", lambda *args: canary)
    if tamper:
        result.write_text('{"status":"changed"}')
    assert object.__new__(worker.Pipeline).production_canary_current() is not tamper


@pytest.mark.parametrize("stack_status,recovery_status", [("BLOCKED", "PASS"), ("READY", "PASS"), ("READY", "FAIL")])
def test_full_launch_requires_every_phase_and_explicit_ready(tmp_path, monkeypatch, stack_status, recovery_status):
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    monkeypatch.setattr(worker, "load_authorization", lambda root: {"training": {"world_size": 8}})
    pipeline = object.__new__(worker.Pipeline)
    pipeline.state = {}
    events = []
    pipeline.publish = lambda: None
    pipeline.environment = lambda: events.append("environment")
    pipeline.qualify = lambda label: events.append(label)
    pipeline.probes = lambda auth: events.append("probes")
    pipeline.prepare_training_calibration = lambda: events.append("training_tokenizer_calibration")
    pipeline.wait_for_gpus = lambda: None

    def write(relative, status):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": status}))

    def script(stage, *args, **kwargs):
        events.append(stage)
        outputs = {
            "experiment_identity": "report/experiment_identity_audit_latest.json",
            "freeze": "experiments/e4_curriculum/results/frozen_1t_recipe.json",
            "production_recovery": "data/health/production-recovery-qualification.json",
            "production_canary": "data/health/production-throughput-canary.json",
            "production_contract": "data/health/pretrain-production-contract.json",
            "stack": "data/health/pretrain-stack-audit.json",
        }
        if stage in outputs:
            status = stack_status if stage == "stack" else recovery_status if stage == "production_recovery" else "PASS"
            write(outputs[stage], status)

    pipeline.script = script
    pipeline.command = lambda stage, argv: events.append(stage)
    if stack_status == "BLOCKED" or recovery_status == "FAIL":
        with pytest.raises(RuntimeError, match="expected READY" if stack_status == "BLOCKED" else "expected PASS"):
            pipeline.run()
        assert "full_pretrain" not in events
    else:
        pipeline.run()
        assert events[-1] == "full_pretrain"
        assert pipeline.state["status"] == "COMPLETE"
    assert events.index("qualification_after_freeze") < events.index("production_recovery")
    assert events.index("training_tokenizer_calibration") < events.index("freeze")
    if recovery_status == "PASS":
        assert events.index("overlap_and_mixture") < events.index("production_contract")
        assert events.index("plan_after_streams") < events.index("final_preflight") < events.index("production_contract")
        assert events.index("qualification_after_freeze") < events.index("production_canary")
        assert events.index("production_recovery") < events.index("production_canary")
    else:
        assert "production_canary" not in events


def test_current_training_calibration_is_reused_without_encoding(monkeypatch):
    monkeypatch.setattr(worker, "load_training_ratios", lambda *_args: {})
    pipeline = object.__new__(worker.Pipeline)
    pipeline.state = {}
    pipeline.script = lambda *_a, **_k: pytest.fail("current calibration must not be re-encoded")
    pipeline.prepare_training_calibration()
    assert pipeline.state["training_tokenizer_calibration_reused"] is True


def test_missing_selected_release_blocks_calibration_bootstrap(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    def missing(*_args):
        raise FileNotFoundError("missing report")
    monkeypatch.setattr(worker, "load_training_ratios", missing)
    pipeline = object.__new__(worker.Pipeline)
    pipeline.state = {}
    pipeline.script = lambda *_a, **_k: pytest.fail("cannot calibrate unselected data")
    with pytest.raises(RuntimeError, match="draft order release"):
        pipeline.prepare_training_calibration()


def test_stale_calibration_is_measured_then_revalidated_before_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    release = tmp_path / "data/processed/releases/pretrain-1t-frozen.release.json"
    release.parent.mkdir(parents=True)
    release.write_text(json.dumps({"status": "PASS", "stages": {k: {} for k in ("warmup", "stable", "anneal")}}))
    events = []
    def validate(*_args):
        events.append("validate")
        if len(events) == 1:
            raise ValueError("stale")
        return {}
    monkeypatch.setattr(worker, "load_training_ratios", validate)
    pipeline = object.__new__(worker.Pipeline)
    pipeline.state = {}
    pipeline.script = lambda stage, *args, **kwargs: events.append(stage)
    pipeline.prepare_training_calibration()
    assert events == ["validate", "training_tokenizer_calibration", "validate"]
    assert pipeline.state["training_tokenizer_calibration_reused"] is False


@pytest.mark.parametrize('stale', ['recipe', 'canary', 'assets'])
def test_prepared_launch_never_rebuilds_when_any_evidence_is_stale(tmp_path, monkeypatch, stale):
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    monkeypatch.setattr(worker, 'load_authorization', lambda root: {'training': {'world_size': 8}})
    pipeline = object.__new__(worker.Pipeline)
    pipeline.state = {}
    pipeline.reuse_prepared_production = True
    pipeline.frozen_recipe_current = lambda: stale != 'recipe'
    pipeline.production_canary_current = lambda: stale != 'canary'
    pipeline.production_assets_current = lambda: stale != 'assets'
    pipeline.script = lambda *a, **k: pytest.fail('prepared launch cannot rebuild or repair artifacts')
    pipeline.command = lambda *a, **k: pytest.fail('stale production cannot launch')
    with pytest.raises(RuntimeError, match='rebuilding is forbidden'):
        pipeline.run()


@pytest.mark.parametrize('stack_ready', [True, False])
def test_prepared_launch_retains_final_contract_and_ready_interlocks(tmp_path, monkeypatch, stack_ready):
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    monkeypatch.setattr(worker, 'load_authorization', lambda root: {'training': {'world_size': 8}})
    pipeline = object.__new__(worker.Pipeline)
    pipeline.state = {}
    pipeline.reuse_prepared_production = True
    pipeline.frozen_recipe_current = pipeline.production_canary_current = pipeline.production_assets_current = lambda: True
    events = []
    pipeline.wait_for_gpus = pipeline.publish = lambda: None
    pipeline.environment = lambda: pytest.fail('prepared launch must retain its qualified environment')
    pipeline.script = lambda stage, *a, **k: events.append(stage)
    pipeline.command = lambda stage, *a, **k: events.append(stage)
    def require(relative, expected):
        if relative.endswith('pretrain-stack-audit.json') and not stack_ready:
            raise RuntimeError('stack not READY')
        return {'status': expected}
    pipeline.require = require
    if stack_ready:
        pipeline.run()
        assert events == ['final_preflight', 'production_contract', 'stack', 'full_pretrain']
        assert pipeline.state['production_assets_reused'] is True
    else:
        with pytest.raises(RuntimeError, match='not READY'):
            pipeline.run()
        assert 'full_pretrain' not in events


def test_adopted_recipe_reuse_binds_new_qualification_without_legacy_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    path = tmp_path/'experiments/e4_curriculum/results/frozen_1t_recipe.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'status': 'PASS',
        'data_origin': {'kind': 'TRAINING_ONLY_WITH_EXACT_DATA_REUSE'},
        'training_adoption': {'qualification_identity': 'new-identity'}}))
    monkeypatch.setattr(worker, 'validate_frozen_bindings', lambda *a: None)
    calls = []
    def module(name, relative):
        if relative.endswith('training_data_origin.py'):
            return SimpleNamespace(resolve_data_origin=lambda *a: calls.append('exact-data-origin'))
        return SimpleNamespace(qualification_identity=lambda *a: {'sha256': 'new-identity'})
    monkeypatch.setattr(worker, 'module', module)
    assert object.__new__(worker.Pipeline).frozen_recipe_current() is True
    assert calls == ['exact-data-origin']
    assert not (tmp_path/'data/health/full-pretrain-optimization-canary.json').exists()


def test_adopted_recipe_forces_reuse_only_even_without_cli_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    monkeypatch.setattr(worker, 'load_authorization', lambda root: {'training': {'world_size': 8}})
    path = tmp_path/'experiments/e4_curriculum/results/frozen_1t_recipe.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'data_origin': {'kind': 'TRAINING_ONLY_WITH_EXACT_DATA_REUSE'}}))
    pipeline = object.__new__(worker.Pipeline)
    pipeline.state = {}
    pipeline.reuse_prepared_production = False
    pipeline.frozen_recipe_current = lambda: False
    pipeline.environment = pipeline.probes = pipeline.script = lambda *a, **k: pytest.fail('adopted data cannot be generically rebuilt')
    with pytest.raises(RuntimeError, match='rebuilding is forbidden'):
        pipeline.run()
    assert pipeline.state['adopted_recipe_automatically_forces_reuse_only'] is True


@pytest.mark.parametrize('explicit_reuse', [False, True])
def test_active_model_adoption_hold_blocks_coordinator_without_touching_data(tmp_path, monkeypatch, explicit_reuse):
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    monkeypatch.setattr(worker, 'load_authorization', lambda root: {'training': {'world_size': 8}})
    path = tmp_path/'data/health/pretrain-model-order-adoption.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'status': 'PRELAUNCH_BLOCKED_REMEDIATION_RUNNING'}))
    pipeline = object.__new__(worker.Pipeline)
    pipeline.state = {}
    pipeline.reuse_prepared_production = explicit_reuse
    pipeline.environment = pipeline.script = pipeline.command = lambda *a, **k: pytest.fail('active model adoption cannot refreeze or launch')
    with pytest.raises(RuntimeError, match='preprocessing continues independently'):
        pipeline.run()
