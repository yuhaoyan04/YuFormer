import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from process_watchdog import observe_progress, cleanup_attempt_processes
from pipeline_guard import restart_allowed


def test_logging_heartbeat_cannot_hide_stalled_training(tmp_path):
    job = {'attempt_id': 'a' * 32}
    log = tmp_path / 'queue.log'
    log.write_text('step: 50\n')
    assert observe_progress(job, tmp_path, now=100, poll_seconds=0, stall_seconds=60) is None
    log.write_text('step: 50\nstill alive\n' * 10)
    assert observe_progress(job, tmp_path, now=161, poll_seconds=0, stall_seconds=60) == 'training_progress_timeout'
    log.write_text('step: 100\n')
    assert observe_progress(job, tmp_path, now=162, poll_seconds=0, stall_seconds=60) is None


def test_large_token_batches_get_adaptive_timeout(tmp_path):
    (tmp_path / 'resolved_config.yaml').write_text(yaml.safe_dump({
        'data': {'seq_len': 4096}, 'parallel': {'world_size': 1},
        'train': {'micro_batch_size': 4, 'grad_accum': 256, 'log_every': 50}}))
    (tmp_path / 'queue.log').write_text('step: 50 tps: 60,000\n')
    job = {'attempt_id': 'b' * 32}
    observe_progress(job, tmp_path, now=100, poll_seconds=0)
    assert job['watchdog']['timeout_seconds'] > 10000
    assert observe_progress(job, tmp_path, now=4000, poll_seconds=0) is None


def test_old_attempt_steps_cannot_mask_failed_new_startup(tmp_path):
    job = {'attempt_id': 'c' * 32}
    (tmp_path / 'queue.log').write_text('step: 5000\nYUFORMER_QUEUE_ATTEMPT attempt=' + job['attempt_id'] + '\nstarting\n')
    observe_progress(job, tmp_path, now=100, poll_seconds=0, startup_seconds=60)
    assert observe_progress(job, tmp_path, now=161, poll_seconds=0, startup_seconds=60) == 'startup_timeout'


def test_checkpoint_daemon_failure_is_detected_despite_training_progress(tmp_path):
    job = {'attempt_id': 'f' * 32}
    (tmp_path / 'queue.log').write_text(
        f'YUFORMER_QUEUE_ATTEMPT attempt={job["attempt_id"]}\nstep: 250\n'
        'Checkpoint background process failed during initialization: EADDRINUSE\nstep: 300\n')
    assert observe_progress(job, tmp_path, now=100, poll_seconds=0) == 'checkpoint_background_initialization_failure'


def test_previous_attempt_checkpoint_error_does_not_kill_recovery(tmp_path):
    job = {'attempt_id': 'e' * 32}
    (tmp_path / 'queue.log').write_text(
        'Checkpoint background process failed during initialization: EADDRINUSE\n'
        f'YUFORMER_QUEUE_ATTEMPT attempt={job["attempt_id"]}\nstep: 1\n')
    assert observe_progress(job, tmp_path, now=100, poll_seconds=0) is None


def test_cleanup_requires_both_exact_attempt_and_output_directory(tmp_path):
    attempt = 'd' * 32
    matching = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
        env={**os.environ, 'YUFORMER_ATTEMPT_ID': attempt, 'YUFORMER_PRETRAIN_OUT': str(tmp_path)})
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
        env={**os.environ, 'YUFORMER_ATTEMPT_ID': attempt, 'YUFORMER_PRETRAIN_OUT': str(tmp_path / 'other')})
    try:
        killed = cleanup_attempt_processes({'attempt_id': attempt}, tmp_path)
        assert matching.pid in killed and unrelated.pid not in killed
        matching.wait(timeout=5)
        assert unrelated.poll() is None
    finally:
        for process in (matching, unrelated):
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)


def test_subreaper_removes_cross_session_grandchild_zombie():
    scripts = str(Path(__file__).resolve().parents[1] / 'scripts')
    program = '''
import json, os, subprocess, sys, time
from process_watchdog import enable_subreaper, reap_descendants, reap_orphan_children
from process_control import terminate_process_tree
assert enable_subreaper()
p = subprocess.Popen([sys.executable, '-u', '-c', "import subprocess,sys,time; c=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True);print(c.pid,flush=True);time.sleep(60)"], stdout=subprocess.PIPE, text=True, start_new_session=True)
child = int(p.stdout.readline())
victims = terminate_process_tree(p.pid)
p.wait(timeout=5)
pending = [pid for pid in victims if pid != p.pid]
deadline = time.monotonic() + 5
while pending and time.monotonic() < deadline:
    reap_orphan_children(set())
    pending = [pid for pid in pending if os.path.exists('/proc/' + str(pid))]
    time.sleep(.02)
assert not pending
assert not os.path.exists('/proc/' + str(child))
print('reaped')
'''
    result = subprocess.run([sys.executable, '-c', program], timeout=10,
                            env={**os.environ, 'PYTHONPATH': scripts}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'reaped'


def test_supervisor_restarts_have_daily_limit_and_backoff():
    record = {'restart_epochs': [1000, 1500, 2000]}
    assert not restart_allowed(record, 2500)
    assert restart_allowed(record, 90000)
    record = {'restart_epochs': [1000]}
    assert not restart_allowed(record, 1100)


def test_nonfinite_failures_do_not_get_another_outer_retry_generation():
    from run_experiment_queue import renew_retry_generation
    job = {'status': 'RETRY_EXHAUSTED', 'attempts': 3, 'failure_kind': 'nonfinite_gradient'}
    assert renew_retry_generation(job) is False


@pytest.mark.parametrize('training_started', [False, True])
def test_queue_times_out_real_stuck_process_and_exhausts_bounded_retries(tmp_path, monkeypatch, training_started):
    import types
    import run_experiment_queue as queue
    policy = tmp_path / 'data/health/gpu-resource-policy.json'
    policy.parent.mkdir(parents=True, exist_ok=True)
    policy.write_text(json.dumps({
        'status': 'ACTIVE', 'allowed_gpu_indices': [0],
        'max_concurrent_jobs': 1, 'allow_busy_gpu_sharing': False,
    }))
    folder = tmp_path / 'pretrain/configs/e2'
    folder.mkdir(parents=True)
    run = tmp_path / 'data/experiments/e2/stuck'
    (folder / 'stuck.env').write_text(f'YUFORMER_PRETRAIN_OUT={run}\n')
    launcher = tmp_path / 'launch.sh'
    if training_started:
        launcher.write_text(f'#!/bin/bash\nexec {sys.executable} -u -c \'import time; print("step: 1",flush=True); time.sleep(60)\'\n')
    else:
        launcher.write_text('#!/bin/bash\nexec sleep 60\n')
    monkeypatch.setattr(queue, 'ROOT', tmp_path)
    monkeypatch.setattr(queue, 'LAUNCH', launcher)
    monkeypatch.setattr(queue, 'training_command', lambda gpu, name, env: ['bash', str(launcher), str(env)])
    monkeypatch.setattr(queue, 'launch_admitted', lambda *args: True)
    monkeypatch.setattr(queue, 'BACKOFF', (0, 0, 0))
    monkeypatch.setattr(queue, '_QUEUE_LEASE', None)
    monkeypatch.setattr(queue, 'time', types.SimpleNamespace(time=time.time, time_ns=time.time_ns,
                                                          sleep=lambda seconds: time.sleep(.03)))
    monkeypatch.setattr(queue, 'observe_progress', lambda job, path: observe_progress(
        job, path, startup_seconds=5 if training_started else .1, stall_seconds=.1, poll_seconds=0))
    monkeypatch.setattr(sys, 'argv', ['queue', '--group', 'e2', '--gpus', '0'])
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        assert queue.main() == 1
        state = json.loads((tmp_path / 'data/health/e2-queue.json').read_text())
        job = state['jobs']['stuck']
        assert job['attempts'] == 3
        assert job['status'] == 'RETRY_EXHAUSTED'
        assert job['recovery_reason'] == ('training_progress_timeout' if training_started else 'startup_timeout')
        assert not Path(f'/proc/{job["pid"]}').exists()
        assert state['pending_reap_pids'] == []
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if queue._QUEUE_LEASE is not None:
            queue._QUEUE_LEASE.release()
