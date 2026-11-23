import subprocess
import sys
from pathlib import Path

import drain_current_experiments as drain
import json


def test_live_launcher_and_pending_collection_both_delay_shutdown(monkeypatch):
    captured = {'pid': 123, 'env': '/run.env', 'attempt_id': 'original'}
    request = {'running_at_request': {'batch_pilot': {'run': captured}}}
    monkeypatch.setattr(drain, 'exact_process', lambda *_: True)
    assert drain.unfinished(request, {}, {}) == ['run']
    monkeypatch.setattr(drain, 'exact_process', lambda *_: False)
    queues = {'batch_pilot': {'jobs': {'run': {'status': 'RUNNING', 'attempt_id': 'original'}}}}
    assert drain.unfinished(request, queues, {'batch_pilot': True}) == ['run']
    queues['batch_pilot']['jobs']['run']['status'] = 'RETRY_WAIT'
    assert drain.unfinished(request, queues, {'batch_pilot': True}) == []


def test_pid_reuse_cannot_match_a_different_command():
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    try:
        assert not drain.exact_process(process.pid, ['/unrelated/launch.sh', '/run.env'])
        assert drain.exact_process(process.pid, ['-c', 'import time; time.sleep(30)'])
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_old_attempt_cannot_hold_drain_after_its_launcher_exits(monkeypatch):
    monkeypatch.setattr(drain, 'exact_process', lambda *_: False)
    request = {'running_at_request': {'batch_pilot': {'run': {
        'pid': 123, 'env': '/run.env', 'attempt_id': 'old'}}}}
    queues = {'batch_pilot': {'jobs': {'run': {'status': 'RUNNING', 'attempt_id': 'different'}}}}
    assert drain.unfinished(request, queues, {'batch_pilot': True}) == []


def test_completed_drain_persists_pause_and_exits_without_restarting_jobs(tmp_path, monkeypatch):
    health = tmp_path / 'health'
    health.mkdir()
    monkeypatch.setattr(drain, 'HEALTH', health)
    monkeypatch.setattr(drain, 'exact_process', lambda *_: False)
    (health / 'pretrain-user-pause.json').write_text('{"status":"DRAINING"}')
    (health / 'gpu-allocations').mkdir()
    (health / 'gpu-allocations/batch_pilot.json').write_text('{"max_parallel":8}')
    (health / 'batch_pilot-queue.json').write_text(json.dumps({'jobs': {
        'run': {'status': 'PASS', 'attempt_id': 'original'}, 'next': {'status': 'PENDING'}}}))
    request = health / 'request.json'
    request.write_text(json.dumps({'running_at_request': {'batch_pilot': {'run': {
        'pid': 123, 'env': '/run.env', 'attempt_id': 'original'}}},
        'queue_coordinators': {'batch_pilot': 234}}))
    monkeypatch.setattr(sys, 'argv', ['drain', '--request', str(request)])
    assert drain.main() == 0
    assert json.loads((health / 'pretrain-user-pause.json').read_text())['status'] == 'PAUSED'
    assert json.loads(request.read_text())['status'] == 'DRAINED_AND_PAUSED'
    assert json.loads((health / 'gpu-allocations/batch_pilot.json').read_text())['max_parallel'] == 0
