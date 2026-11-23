import importlib.util
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pretrain/scripts"))
import queue_control as CONTROL
import run_experiment_queue as QUEUE
from process_control import terminate_process_tree
from yuformer_pretrain.lease import RunLease
from yuformer_pretrain.config import ConfigError

spec = importlib.util.spec_from_file_location("allocation_watch", ROOT / "pretrain/scripts/pretrain_readiness_watch.py")
WATCH = importlib.util.module_from_spec(spec)
spec.loader.exec_module(WATCH)


@pytest.fixture(autouse=True)
def isolate_runtime_gpu_policy(tmp_path, monkeypatch):
    monkeypatch.setattr(WATCH, 'HEALTH', tmp_path / 'health')
    monkeypatch.setattr(WATCH, 'allowed_gpu_indices', lambda _: set(range(8)))
    monkeypatch.setattr(WATCH, 'resource_policy', lambda _: {
        'status': 'ACTIVE', 'allowed_gpu_indices': list(range(8)),
        'max_concurrent_jobs': 8, 'allow_busy_gpu_sharing': True,
    })
    monkeypatch.setattr(WATCH, 'resource_policy', lambda _: {
        'status': 'ACTIVE', 'allowed_gpu_indices': list(range(8)),
        'max_concurrent_jobs': 8, 'allow_busy_gpu_sharing': True,
    })
    monkeypatch.setattr(QUEUE, 'resource_policy', lambda _: {
        'allowed_gpu_indices': list(range(8)), 'allow_busy_gpu_sharing': True})


def test_allocation_can_grow_and_malformed_reassignment_is_rejected(tmp_path):
    path = tmp_path / "allocation.json"
    value = {"group": "e1", "gpus": [0, 1], "shared_gpus": [], "max_parallel": 2}
    path.write_text(json.dumps(value))
    assert CONTROL.read_allocation(path, "e1")["gpus"] == [0, 1]
    value.update(gpus=list(range(8)), max_parallel=8)
    path.write_text(json.dumps(value))
    assert CONTROL.read_allocation(path, "e1")["max_parallel"] == 8
    value["shared_gpus"] = [9]
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        CONTROL.read_allocation(path, "e1")


def test_e1_e2_balance_and_running_jobs_survive_allocation_refresh(tmp_path, monkeypatch):
    monkeypatch.setattr(WATCH, "ROOT", tmp_path)
    monkeypatch.setattr(WATCH, "HEALTH", tmp_path / "health")
    monkeypatch.setattr(WATCH, "ACTION_STATE", tmp_path / "actions.json")
    monkeypatch.setattr(WATCH, "experiment_result_roster_complete", lambda group: False)
    monkeypatch.setattr(WATCH, "runner_matches", lambda pid, script: pid == 123)
    WATCH.HEALTH.mkdir()
    (WATCH.HEALTH / "e1-queue.json").write_text(json.dumps({"jobs": {
        "live": {"status": "RUNNING", "gpu": 6, "pid": 123}}}))
    record = {}
    pool = WATCH.publish_queue_allocation("e1", record, [0, 1, 2, 3, 4, 5, 7], "formal_idle", None)
    assert pool == [6, 0, 1, 2]
    WATCH.ACTION_STATE.write_text(json.dumps({"actions": {"e1_queue": {
        "status": "RUNNING", "pid": 123, "gpus": pool}}}))
    other_pool = WATCH.publish_queue_allocation("e2", {}, [3, 4, 5, 7], "formal_idle", None)
    assert len(other_pool) == 4 and not set(pool) & set(other_pool)
    monkeypatch.setattr(WATCH, "experiment_result_roster_complete", lambda group: True)
    pool = WATCH.publish_queue_allocation("e1", record, list(range(8)), "formal_idle", None)
    assert len(pool) == 8 and pool[0] == 6


def test_three_available_gpus_are_split_between_both_families(tmp_path, monkeypatch):
    monkeypatch.setattr(WATCH, "ROOT", tmp_path)
    monkeypatch.setattr(WATCH, "HEALTH", tmp_path / "health")
    monkeypatch.setattr(WATCH, "ACTION_STATE", tmp_path / "actions.json")
    monkeypatch.setattr(WATCH, "experiment_result_roster_complete", lambda group: False)
    monkeypatch.setattr(WATCH, "runner_matches", lambda pid, script: pid == 123)
    first = WATCH.publish_queue_allocation("e1", {}, [0, 1, 2], "formal_idle", None)
    WATCH.ACTION_STATE.write_text(json.dumps({"actions": {"e1_queue": {
        "status": "RUNNING", "pid": 123, "gpus": first}}}))
    second = WATCH.publish_queue_allocation("e2", {}, [2], "formal_idle", None)
    assert first == [0, 1] and second == [2]


def test_admission_skips_busy_gpu_but_can_use_another_idle_slot(monkeypatch):
    monkeypatch.setattr(QUEUE, "gpu_slot_busy", lambda gpu: gpu == 0)
    monkeypatch.setattr(QUEUE, "gpu_compute_pids", lambda gpu: set() if gpu == 1 else {123})
    allocation = {"gpus": [0, 1, 2], "shared_gpus": []}
    assert [gpu for gpu in allocation["gpus"] if QUEUE.launch_admitted(gpu, "e1", allocation)] == [1]
    assert QUEUE.launch_admitted(2, "e1", {"shared_gpus": [2]})
    assert not QUEUE.launch_admitted(2, "e1_performance", {"shared_gpus": [2]})


def test_gpu_lease_and_training_lease_have_distinct_lifetimes(tmp_path, monkeypatch):
    slot, inner, receipt = tmp_path / "slot", tmp_path / "inner", tmp_path / "ready.json"
    monkeypatch.setattr(QUEUE, "gpu_slot_root", lambda gpu: slot)
    monkeypatch.setattr(QUEUE, "PYTHON", Path(sys.executable))
    launcher = tmp_path / "launch.sh"
    wrapper = ROOT / "pretrain/scripts/run_with_lease.py"
    program = ("import json,os,time; "
               f"open({str(receipt)!r},'w').write(json.dumps(dict(os.environ))); time.sleep(60)")
    inner_command = [sys.executable, str(wrapper), "--out-dir", str(inner),
                     "--run-name", "inner", "--", sys.executable, "-c", program]
    launcher.write_text('#!/bin/bash\n[[ -z "${YUFORMER_RUN_LEASE_HELD+x}" ]] || exit 9\nexec '
                        + shlex.join(inner_command) + '\n')
    monkeypatch.setattr(QUEUE, "LAUNCH", launcher)
    with (tmp_path / "process.log").open("w") as output:
        process = subprocess.Popen(QUEUE.training_command(2, "test", tmp_path / "unused.env"),
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 15
            while not receipt.exists() and time.monotonic() < deadline:
                if process.poll() is not None:
                    pytest.fail((tmp_path / "process.log").read_text())
                time.sleep(.05)
            environment = json.loads(receipt.read_text())
            assert QUEUE.gpu_slot_busy(2)
            assert str(inner / ".run.lease") in environment["YUFORMER_RUN_LEASE_GENERATION"]
            assert environment["YUFORMER_RUN_LEASE_TOKEN"] != json.loads((slot / "run.lock").read_text())["token"]
            with pytest.raises(ConfigError):
                RunLease(slot, "collision").acquire()
        finally:
            terminate_process_tree(process.pid)
            process.wait(timeout=10)
    assert not QUEUE.gpu_slot_busy(2)
    successor = RunLease(slot, "recovered").acquire()
    successor.release()


def test_wakeup_only_signals_a_matching_watcher(tmp_path, monkeypatch):
    record = tmp_path / "data/health/pretrain-watch-wakeup.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"pid": 123, "signal": "SIGUSR1"}))
    monkeypatch.setattr(Path, "read_bytes", lambda path: b"python\0pretrain/scripts/pretrain_readiness_watch.py\0")
    signalled = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: signalled.append((pid, sig)))
    assert CONTROL.notify_watcher(tmp_path)
    assert signalled == [(123, signal.SIGUSR1)]
    monkeypatch.setattr(Path, "read_bytes", lambda path: b"python\0another-user-job.py\0")
    assert not CONTROL.notify_watcher(tmp_path)
    assert len(signalled) == 1


def test_checkpoint_transport_requires_bound_concurrency_qualification(tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setattr(QUEUE, 'ROOT', tmp_path)
    assert QUEUE.checkpoint_transport_environment()[0] == {}


    evidence = []
    for name in ['single', 'multi']:
        p = tmp_path / f'{name}.json'; p.write_text('{"status":"PASS"}')
        evidence.append({'path': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    report = tmp_path / 'data/health/dcp-transport-qualification.json'
    report.parent.mkdir(parents=True)
    report.write_text(json.dumps({'status': 'PASS', 'mode': 'existing_rendezvous_prefix_store',
                                 'concurrent_single_rank_jobs': 8, 'multi_rank_world_size': 2,
                                 'evidence': evidence}))
    environment, context = QUEUE.checkpoint_transport_environment()
    assert environment == {'DCP_USE_PREFIX_STORE': '1'}
    assert context['qualification_sha256'] == hashlib.sha256(report.read_bytes()).hexdigest()
    (tmp_path / 'single.json').write_text('{"status":"FAIL"}')
    assert QUEUE.checkpoint_transport_environment()[0] == {}


def test_lr_retries_reserve_next_shared_slot_without_interrupting_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(WATCH, 'ROOT', tmp_path)
    monkeypatch.setattr(WATCH, 'HEALTH', tmp_path / 'health')
    monkeypatch.setattr(WATCH, 'ACTION_STATE', tmp_path / 'actions.json')
    monkeypatch.setattr(WATCH, 'runner_matches', lambda pid, script: pid in {123, 456})
    WATCH.HEALTH.mkdir()
    (WATCH.HEALTH / 'e1_pilot_extension-queue.json').write_text(json.dumps({'jobs': {'retry': {'status': 'RETRY_WAIT'}}}))
    batch = {'jobs': {'running': {'status': 'RUNNING', 'gpu': 6, 'pid': 456}}}
    (WATCH.HEALTH / 'batch_pilot-queue.json').write_text(json.dumps(batch))
    record = {'pid': 123, 'status': 'RUNNING'}
    pool = WATCH.publish_queue_allocation('batch_pilot', record, [4, 5, 7], 'provisional_shared', {'gpu_indices': [0, 6, 7]})
    allocation = json.loads((WATCH.HEALTH / 'gpu-allocations/batch_pilot.json').read_text())
    assert set(pool) == {4, 5, 6, 7} and allocation['max_parallel'] == 2
    WATCH.ACTION_STATE.write_text(json.dumps({'actions': {'batch_pilot_queue': record}}))
    assert WATCH.active_gpu_reservations() == {4, 5, 6}
    batch['jobs']['running']['status'] = 'PASS'
    (WATCH.HEALTH / 'batch_pilot-queue.json').write_text(json.dumps(batch))
    assert WATCH.active_gpu_reservations() == {4, 5}
