import json
import signal

import gpu_resource_policy as resources
import pipeline_guard as guard


def test_new_watcher_is_not_signalled_before_handler_registration(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, 'HEALTH', tmp_path)
    monkeypatch.setattr(guard, 'exact_process', lambda *_: True)
    signalled = []
    monkeypatch.setattr(guard.os, 'kill', lambda *args: signalled.append(args))
    registration = tmp_path / 'pretrain-watch-wakeup.json'
    assert not guard.wake_registered_watcher(200)
    registration.write_text(json.dumps({'pid': 100, 'signal': 'SIGUSR1'}))
    assert not guard.wake_registered_watcher(200)
    assert signalled == []
    registration.write_text(json.dumps({'pid': 200, 'signal': 'SIGUSR1'}))
    assert guard.wake_registered_watcher(200)
    assert signalled == [(200, signal.SIGUSR1)]


def test_reused_pid_or_exited_watcher_is_not_woken(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, 'HEALTH', tmp_path)
    (tmp_path / 'pretrain-watch-wakeup.json').write_text(json.dumps({'pid': 200, 'signal': 'SIGUSR1'}))
    monkeypatch.setattr(guard, 'exact_process', lambda *_: False)
    monkeypatch.setattr(guard.os, 'kill', lambda *_: (_ for _ in ()).throw(AssertionError('wrong process')))
    assert not guard.wake_registered_watcher(200)


def test_idle_runnable_queues_wake_monitor_after_five_minutes_with_cooldown(monkeypatch):
    monkeypatch.setattr(guard, 'queues_waiting_for_admission', lambda _: ['batch_pilot_queue'])
    monkeypatch.setattr(guard, 'live_training_gpus', lambda _: set())
    monkeypatch.setattr(guard, 'admitted_gpu_is_idle', lambda *_: True)
    wakeups = []
    monkeypatch.setattr(guard, 'wake_registered_watcher', lambda pid: wakeups.append(pid) or True)
    state = {'events': []}
    assert not guard.wake_starved_queues(state, {}, 200, 1000)
    assert not guard.wake_starved_queues(state, {}, 200, 1299)
    assert guard.wake_starved_queues(state, {}, 200, 1300)
    assert not guard.wake_starved_queues(state, {}, 200, 1350)
    assert wakeups == [200]
    monkeypatch.setattr(guard, 'admitted_gpu_is_idle', lambda *_: False)
    assert not guard.wake_starved_queues(state, {}, 200, 1600)
    assert 'idle_queue_since_epoch' not in state


def test_all_admitted_gpus_owned_prevent_idle_queue_alarm(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, 'HEALTH', tmp_path)
    monkeypatch.setattr(guard, 'exact_process', lambda *_: True)
    monkeypatch.setattr(resources, 'visible_gpu_indices', lambda: {0, 1, 2, 3, 4})
    (tmp_path / 'gpu-resource-policy.json').write_text(json.dumps({
        'status': 'ACTIVE', 'allowed_gpu_indices': [1, 4], 'max_concurrent_jobs': 2,
        'allow_busy_gpu_sharing': False}))
    (tmp_path / 'batch_pilot-queue.json').write_text(json.dumps({'jobs': {
        'pending': {'status': 'PENDING'}}}))
    (tmp_path / 'e1-queue.json').write_text(json.dumps({'jobs': {
        'active': {'status': 'RUNNING', 'pid': 123, 'gpu': 1},
        'other': {'status': 'RUNNING', 'pid': 124, 'gpu': 4}}}))
    actions = {'batch_pilot_queue': {'status': 'RUNNING', 'pid': 100},
               'e1_queue': {'status': 'RUNNING', 'pid': 101}}
    assert guard.queues_waiting_for_admission(actions) == []


def test_one_active_gpu_does_not_hide_retryable_lr_waiting_for_idle_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, 'HEALTH', tmp_path)
    monkeypatch.setattr(guard, 'exact_process', lambda pid, _: pid == 123)
    monkeypatch.setattr(resources, 'visible_gpu_indices', lambda: {0, 1, 2, 3, 4})
    (tmp_path / 'gpu-resource-policy.json').write_text(json.dumps({
        'status': 'ACTIVE', 'allowed_gpu_indices': [1, 4], 'max_concurrent_jobs': 2,
        'allow_busy_gpu_sharing': False}))
    (tmp_path / 'batch_pilot-queue.json').write_text(json.dumps({'jobs': {
        'active': {'status': 'RUNNING', 'pid': 123, 'gpu': 4}}}))
    (tmp_path / 'e1_pilot_extension-queue.json').write_text(json.dumps({'jobs': {
        'stopped': {'status': 'RETRY_EXHAUSTED'}}}))
    actions = {'batch_pilot_queue': {'status': 'RUNNING', 'pid': 123},
               'e1_pilot_extension_queue': {'status': 'WAITING_EXTERNAL', 'attempts': 2,
                                           'blocker': 'experiment_gpu_allocation'}}
    assert guard.queues_waiting_for_admission(actions) == ['e1_pilot_extension_queue']
