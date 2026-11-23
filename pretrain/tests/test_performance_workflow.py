import importlib.util
import json
import shutil
import sys
import signal
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pretrain/scripts"))
import performance_evidence as EVIDENCE
import run_experiment_queue as QUEUE
from prepare_performance_canaries import prepare, SIZES
from audit_experiment_identities import expand_config, parse_env


def source_fixture(tmp_path):
    target = tmp_path / "pretrain/configs/e1"
    target.mkdir(parents=True)
    model = tmp_path / "model.yaml"
    shutil.copyfile(ROOT / "model/configs/model_configs.yaml", model)
    configs = {}
    for size in SIZES:
        name = f"e1_ln_{size}"
        source = ROOT / f"pretrain/configs/e1/{name}.yaml"
        shutil.copyfile(source, target / source.name)
        values = parse_env(source.with_suffix(".env"))
        values["YUFORMER_MODEL_CONFIGS"] = str(model)
        (target / f"{name}.env").write_text("".join(f"{k}={v}\n" for k, v in values.items()))
        configs[name] = yaml.safe_load(source.read_text())
    path = tmp_path / "experiments/e1_scaling/results/lr_resolution.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"status":"PASS"}')
    return configs, yaml.safe_load(model.read_text())["presets"]


def test_performance_contracts_preserve_selected_shape_and_token_batch(tmp_path):
    configs, _ = source_fixture(tmp_path)
    report = prepare(tmp_path)
    assert report["status"] == "PASS" and len(report["generated"]) == 5
    for row in report["generated"]:
        cfg = yaml.safe_load((tmp_path / row["config"]).read_text())
        original = configs[row["name"].replace("e1_perf_", "e1_ln_")]
        assert cfg["train"]["expected_iters"] == 32
        assert cfg["train"]["budget_tokens"] == 32 * row["tokens_per_update"]
        for key in ("micro_batch_size", "grad_accum", "bf16", "activation_checkpointing"):
            assert cfg["train"][key] == original["train"][key]
    first = (tmp_path / report["generated"][0]["config"]).stat().st_mtime_ns
    prepare(tmp_path)
    assert (tmp_path / report["generated"][0]["config"]).stat().st_mtime_ns == first


def qualify_fixture(tmp_path, monkeypatch):
    path = tmp_path / "data/health/torchtitan-qualification.json"
    path.parent.mkdir(parents=True)
    identity = {"schema_version": 1, "sha256": "qualified"}
    path.write_text(json.dumps({"status": "PASS", "qualification_identity": identity}))
    monkeypatch.setattr(EVIDENCE, "qualification_identity", lambda root: identity)
    monkeypatch.setattr(EVIDENCE, "current_source_sha", lambda root: "source")


def test_started_probe_preserves_exact_provenance_on_reference_only_change(tmp_path):
    source_fixture(tmp_path)
    prepare(tmp_path)
    target = tmp_path / "pretrain/configs/e1_performance/e1_perf_50m.yaml"
    original = target.read_bytes()
    run = tmp_path / "data/experiments/e1_performance/e1_perf_50m"
    run.mkdir(parents=True)
    (run / "run_identity.json").write_text('{}')
    source = tmp_path / "pretrain/configs/e1/e1_ln_50m.yaml"
    cfg = yaml.safe_load(source.read_text())
    cfg["optim"]["lr_resolution"] = {"batch_selection_sha256": "new-report-reference"}
    source.write_text(yaml.safe_dump(cfg, sort_keys=False))
    report = prepare(tmp_path)
    assert target.read_bytes() == original
    assert report["generated"][0]["config_sha256"] == EVIDENCE.sha256(target)


@pytest.mark.parametrize("section,key,value", [
    ("optim", "lr", 0.123), ("train", "grad_accum", 123),
    ("data", "seq_len", 2048), ("train", "activation_checkpointing", "changed"),
])
def test_started_probe_rejects_training_changes_without_partial_writes(tmp_path, section, key, value):
    source_fixture(tmp_path)
    prepare(tmp_path)
    folder = tmp_path / "pretrain/configs/e1_performance"
    original = {p: p.read_bytes() for p in folder.iterdir()}
    # Change an unstarted earlier arm too: a later rejection must leave it intact.
    first = tmp_path / "pretrain/configs/e1/e1_ln_50m.yaml"
    first_cfg = yaml.safe_load(first.read_text())
    first_cfg["optim"]["lr"] = 0.111
    first.write_text(yaml.safe_dump(first_cfg, sort_keys=False))
    run = tmp_path / "data/experiments/e1_performance/e1_perf_800m"
    run.mkdir(parents=True)
    (run / "run_identity.json").write_text('{}')
    source = tmp_path / "pretrain/configs/e1/e1_ln_800m.yaml"
    cfg = yaml.safe_load(source.read_text())
    cfg[section][key] = value
    source.write_text(yaml.safe_dump(cfg, sort_keys=False))
    with pytest.raises(ValueError, match="started performance contract"):
        prepare(tmp_path)
    assert all(p.read_bytes() == before for p, before in original.items())


def performance_fixture(tmp_path, monkeypatch):
    configs, presets = source_fixture(tmp_path)
    prepare(tmp_path)
    qualify_fixture(tmp_path, monkeypatch)
    for size in SIZES:
        name = f"e1_perf_{size}"
        run = tmp_path / f"data/experiments/e1_performance/{name}"
        run.mkdir(parents=True)
        cfg_path = tmp_path / f"pretrain/configs/e1_performance/{name}.yaml"
        resolved = expand_config(cfg_path, parse_env(cfg_path.with_suffix(".env")))
        (run / "resolved_config.yaml").write_text(resolved)
        objects = {
            "run_identity.json": {"identity_sha256": name, "source_sha256": "source",
                "resolved_config_sha256": EVIDENCE.sha256(run / "resolved_config.yaml"),
                "model_configs_sha256": EVIDENCE.sha256(tmp_path / "model.yaml")},
            "current_attempt.json": {"status": "PASS", "attempt_id": name, "identity_sha256": name},
            "training_complete.json": {"status": "PASS", "attempt_id": name, "identity_sha256": name},
            "execution_context.json": {"attempt_id": name, "loss_only": False,
                                       "performance_probe": True, "idle_admission": True,
                                       "contention_detected": False, "contention_checks": 10},
            "result.json": {"status": "PASS", "attempt_id": name, "identity_sha256": name,
                            "series": {"throughput(tps)": [{"step": step, "value": 1234.0}
                                                           for step in range(1, 33)]}},
        }
        for filename, obj in objects.items():
            (run / filename).write_text(json.dumps(obj))
    return configs, presets


def test_throughput_requires_current_identity_and_idle_admission(tmp_path, monkeypatch):
    configs, presets = performance_fixture(tmp_path, monkeypatch)
    report = EVIDENCE.qualified_performance(tmp_path, configs, presets, lambda preset: 100)
    assert report["status"] == "PASS" and len(report["runs"]) == 5
    assert all(row["steady_points"] == 23 for row in report["runs"])
    path = tmp_path / "data/experiments/e1_performance/e1_perf_800m/execution_context.json"
    context = json.loads(path.read_text())
    context["loss_only"] = True
    path.write_text(json.dumps(context))
    assert EVIDENCE.qualified_performance(tmp_path, configs, presets, lambda preset: 100)["status"] == "BLOCKED"


def test_throughput_rejects_different_selected_batch(tmp_path, monkeypatch):
    configs, presets = performance_fixture(tmp_path, monkeypatch)
    configs["e1_ln_50m"]["train"]["grad_accum"] *= 2
    report = EVIDENCE.qualified_performance(tmp_path, configs, presets, lambda preset: 100)
    assert report["status"] == "BLOCKED"
    assert any("shape/batch" in issue for issue in report["issues"])


def test_loss_promotion_requires_current_complete_identity_audit(tmp_path, monkeypatch):
    qualify_fixture(tmp_path, monkeypatch)
    rows, snapshot, ranking = [], {}, []
    for index in range(8):
        name = f"arm{index}"
        rows.append({"group": "e3", "name": name, "status": "PASS"})
        ranking.append({"run_name": name, "identity_sha256": name})
        for filename in ("result.json", "run_identity.json", "current_attempt.json", "training_complete.json"):
            path = tmp_path / f"data/experiments/e3/{name}/{filename}"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
            snapshot[str(path.relative_to(tmp_path))] = EVIDENCE.sha256(path)
    path = tmp_path / "report/experiment_identity_audit_latest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"source_sha256": "source", "rows": rows, "evidence_snapshot": snapshot}))
    analysis = {"status": "PASS_PROVISIONAL", "ranking": ranking}
    promoted = EVIDENCE.promote_bound_loss_analysis(tmp_path, "e3", analysis)
    assert promoted["status"] == "PASS"
    assert "remain excluded" in promoted["loss_evidence_qualification"]["scope"]
    (tmp_path / next(iter(snapshot))).write_text("drift")
    assert EVIDENCE.promote_bound_loss_analysis(tmp_path, "e3", analysis)["status"] == "PASS_PROVISIONAL"


def test_performance_queue_aborts_and_archives_a_contended_measurement(tmp_path, monkeypatch):
    folder = tmp_path / "pretrain/configs/e1_performance"
    folder.mkdir(parents=True)
    run = tmp_path / "data/experiments/e1_performance/probe"
    (folder / "probe.env").write_text(f"YUFORMER_PRETRAIN_OUT={run}\n")
    launcher = tmp_path / "launch.sh"
    launcher.write_text("#!/bin/bash\nexec sleep 60\n")
    monkeypatch.setattr(QUEUE, "ROOT", tmp_path)
    policy = tmp_path / "data/health/gpu-resource-policy.json"
    policy.parent.mkdir(parents=True, exist_ok=True)
    policy.write_text(json.dumps({
        "status": "ACTIVE", "allowed_gpu_indices": [0],
        "max_concurrent_jobs": 1, "allow_busy_gpu_sharing": False,
    }))
    monkeypatch.setattr(QUEUE, "LAUNCH", launcher)
    monkeypatch.setattr(QUEUE, "training_command", lambda gpu, name, env: ["bash", str(launcher), str(env)])
    monkeypatch.setattr(QUEUE, "BACKOFF", (0,))
    monkeypatch.setattr(QUEUE, "_QUEUE_LEASE", None)
    probes = iter((set(), {123, 456}))
    monkeypatch.setattr(QUEUE, "gpu_compute_pids", lambda gpu: next(probes))
    monkeypatch.setattr(sys, "argv", ["queue", "--group", "e1_performance", "--gpus", "0"])
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        assert QUEUE.main() == 1
        report = json.loads((tmp_path / "data/health/e1_performance-queue.json").read_text())
        assert report["status"] == "FAIL"
        job = report["jobs"]["probe"]
        assert job["performance_contaminated"] is True
        assert not QUEUE.process_alive(job["pid"])
        context = json.loads((Path(job["contaminated_archive"]) / "execution_context.json").read_text())
        assert context["contention_detected"] is True
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if QUEUE._QUEUE_LEASE is not None:
            QUEUE._QUEUE_LEASE.release()


def test_batch_loss_qualification_keeps_shared_performance_excluded(tmp_path, monkeypatch):
    import batch_pilot_contract as contract
    qualify_fixture(tmp_path, monkeypatch)
    rows, snapshot, curves = [], {}, {batch: [] for batch in contract.BATCHES}
    for batch in contract.BATCHES:
        for level in contract.LEVELS:
            name = f'batch_{batch}_{level}'
            rows.append({'group': 'batch_pilot', 'name': name, 'status': 'PASS'})
            curves[batch].append({'run_name': name, 'identity_sha256': name, 'loss_only': True,
                                  'throughput_tps_per_device': None})
            for filename in ['result.json', 'run_identity.json', 'current_attempt.json', 'training_complete.json']:
                path = tmp_path / f'data/experiments/batch_pilot/{name}/{filename}'
                path.parent.mkdir(parents=True, exist_ok=True); path.write_text('{}')
                snapshot[str(path.relative_to(tmp_path))] = EVIDENCE.sha256(path)
    path = tmp_path / 'report/experiment_identity_audit_latest.json'; path.parent.mkdir()
    path.write_text(json.dumps({'source_sha256': 'source', 'rows': rows, 'evidence_snapshot': snapshot}))
    analysis = {'status': 'PASS_PROVISIONAL', 'expected_results': 9, 'observed_results': 9,
                'selection': {'loss_only': True}, 'curves': curves}
    promoted = EVIDENCE.promote_bound_loss_analysis(tmp_path, 'batch_pilot', analysis)
    assert promoted['status'] == 'PASS'
    assert promoted['selection']['loss_only'] is True
    assert all(row['throughput_tps_per_device'] is None for curve in promoted['curves'].values() for row in curve)
    (tmp_path / next(iter(snapshot))).write_text('drift')
    assert EVIDENCE.promote_bound_loss_analysis(tmp_path, 'batch_pilot', analysis)['status'] == 'PASS_PROVISIONAL'
