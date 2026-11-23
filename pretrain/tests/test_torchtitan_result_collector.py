import importlib.util
import json
from pathlib import Path

import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/collect_torchtitan_result.py"
SPEC = importlib.util.spec_from_file_location("collect_torchtitan_result", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_collector_excludes_archived_attempt_events(tmp_path):
    active = tmp_path / "tensorboard/events.out.tfevents.active"
    archived = tmp_path / "attempts/old/events/events.out.tfevents.old"
    active.parent.mkdir(parents=True)
    archived.parent.mkdir(parents=True)
    active.touch()
    archived.touch()
    assert MODULE.active_event_files(tmp_path) == [active]


def test_exact_training_metric_wins_over_validation_suffix():
    series = {
        "throughput(tps)": [{"step": 1, "value": 100.0}],
        "validation_metrics/throughput(tps)": [{"step": 1, "value": 200.0}],
    }
    assert MODULE.resolve_tag(series, "throughput(tps)") == "throughput(tps)"


def _run(tmp_path: Path, monkeypatch, validation_step: int, domain_value: float = 1.2,
         domains=None):
    (tmp_path / "current_attempt.json").write_text(json.dumps({
        "attempt_id": "attempt-00000001", "identity_sha256": "identity",
    }))
    (tmp_path / "run_identity.json").write_text(json.dumps({
        "run_name": "test", "identity_sha256": "identity",
    }))
    (tmp_path / "training_complete.json").write_text(json.dumps({
        "status": "PASS", "returncode": 0, "identity_sha256": "identity",
        "attempt_id": "attempt-00000001",
    }))
    config = {
        "model": {"preset": "tiny"}, "optim": {"lr": 1e-3},
        "data": {"seq_len": 4},
        "train": {"expected_iters": 4, "budget_tokens": 16,
                  "micro_batch_size": 1, "grad_accum": 1, "log_every": 2},
        "parallel": {"world_size": 1},
    }
    (tmp_path / "resolved_config.yaml").write_text(yaml.safe_dump(config))
    event = tmp_path / "events.out.tfevents.test"
    event.touch()
    point = lambda step, value: [{"step": step, "value": value, "wall_time": 1.0}]
    series = {
        "loss_metrics/global_avg_loss": point(4, 1.0),
        "validation_metrics/loss": point(validation_step, 1.1),
        "throughput(tps)": point(4, 100.0),
        "mfu(%)": point(4, 40.0),
    }
    for domain in domains or MODULE.EXPECTED_VALIDATION_DOMAINS:
        series[f"validation_metrics/domain/{domain}/loss"] = point(validation_step, domain_value)
    monkeypatch.setattr(MODULE, "scalar_series", lambda paths: series)
    return MODULE.summarize(tmp_path)


def test_collector_requires_validation_at_exact_final_training_step(tmp_path, monkeypatch):
    assert _run(tmp_path, monkeypatch, 4)["status"] == "PASS"
    result = _run(tmp_path, monkeypatch, 2)
    assert result["status"] == "FAIL"
    assert any(issue.startswith("final_validation_missing") for issue in result["issues"])
    assert any(issue.startswith("final_domain_validation_missing") for issue in result["issues"])


def test_collector_requires_actual_runtime_token_and_sample_accounting(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, 4)
    path = tmp_path / "run_identity.json"
    identity = json.loads(path.read_text())
    identity["runtime_state_schema_version"] = 2
    path.write_text(json.dumps(identity))
    assert "runtime_accounting_missing_or_unbound" in MODULE.summarize(tmp_path)["issues"]
    accounting = {"status": "PASS", "attempt_id": "attempt-00000001", "step": 4,
                  "global_tokens_seen": 16, "consumed_samples": 4}
    path = tmp_path / "training_accounting.json"
    path.write_text(json.dumps(accounting))
    assert MODULE.summarize(tmp_path)["status"] == "PASS"
    accounting["global_tokens_seen"] = 12
    path.write_text(json.dumps(accounting))
    assert "runtime_token_count_mismatch" in MODULE.summarize(tmp_path)["issues"]


def test_collector_rejects_non_finite_domain_loss(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, 4, domain_value=float("nan"))
    assert result["status"] == "FAIL"
    assert any(issue.startswith("non_finite_metric:validation_metrics/domain/")
               for issue in result["issues"])


def test_collector_rejects_wrong_nine_domain_roster(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, 4, domains={f"d{index}" for index in range(9)})
    assert result["status"] == "FAIL"
    assert any(issue.startswith("validation_domain_roster:") for issue in result["issues"])


def test_collector_rejects_completion_from_prior_attempt(tmp_path, monkeypatch):
    result = _run(tmp_path, monkeypatch, 4)
    completion = json.loads((tmp_path / "training_complete.json").read_text())
    completion["attempt_id"] = "attempt-prior000"
    (tmp_path / "training_complete.json").write_text(json.dumps(completion))
    result = MODULE.summarize(tmp_path)
    assert result["status"] == "FAIL"
    assert "completion_attempt_mismatch" in result["issues"]


def test_collector_binds_execution_context_to_current_attempt(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, 4)
    (tmp_path / "execution_context.json").write_text(json.dumps({
        "attempt_id": "attempt-prior000", "loss_only": True,
        "performance_metrics_excluded": ["throughput(tps)"],
    }))
    result = MODULE.summarize(tmp_path)
    assert result["status"] == "FAIL"
    assert "execution_context_attempt_mismatch" in result["issues"]


def test_resume_event_lineage_clips_abandoned_future_steps(tmp_path, monkeypatch):
    current = tmp_path / "tb/events.out.tfevents.current"
    archived = tmp_path / "attempts/superseded-before-new/events/tb/events.out.tfevents.old"
    for path in (current, archived):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (tmp_path / "attempts/old.json").write_text(json.dumps({
        "attempt_id": "old", "identity_sha256": "identity",
    }))
    attempt = {"attempt_id": "new", "parent_attempt_id": "old",
               "identity_sha256": "identity", "resumed_from_step": 4}

    def series(paths):
        steps = [2, 4, 6] if archived in paths else [5, 6, 8]
        return {"loss": [{"step": step, "value": -step if archived in paths else step,
                           "wall_time": 1.0} for step in steps]}

    monkeypatch.setattr(MODULE, "scalar_series", series)
    _, observed = MODULE.lineage_series(tmp_path, attempt)
    assert [(p["step"], p["value"]) for p in observed["loss"]] == [
        (2, -2), (4, -4), (5, 5), (6, 6), (8, 8),
    ]
