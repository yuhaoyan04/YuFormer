import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import gpu_resource_policy as resources
import pretrain_readiness_watch as watch
import queue_control
import run_experiment_queue as queue
import run_production_canary as canary


@pytest.fixture
def restricted(tmp_path, monkeypatch):
    monkeypatch.setattr(resources, 'visible_gpu_indices', lambda: set(range(8)))
    health = tmp_path / 'data/health'
    health.mkdir(parents=True)
    policy = {'status': 'ACTIVE', 'allowed_gpu_indices': [1, 4],
              'max_concurrent_jobs': 2, 'allow_busy_gpu_sharing': False}
    (health / 'gpu-resource-policy.json').write_text(json.dumps(policy))
    monkeypatch.setattr(watch, 'HEALTH', health)
    monkeypatch.setattr(watch, 'ROOT', tmp_path)
    monkeypatch.setattr(watch, 'ACTION_STATE', health / 'actions.json')
    monkeypatch.setattr(queue, 'ROOT', tmp_path)
    return health, policy


def test_idle_and_legacy_shared_fallback_cannot_escape_two_gpu_scope(restricted):
    rows = [{'index': i, 'memory_used_mb': 0} for i in range(8)]
    assert watch.select_experiment_gpus(rows) == [1, 4]
    assert watch.select_experiment_gpus(rows, {1}) == [4]
    stale_shared = {'gpu_indices': [0, 6, 7]}
    assert watch.queue_gpu_allocation(rows, False, stale_shared) == ([], 'provisional_shared')
    assert watch.active_shared_gpu_policy() is None


def test_allocation_writer_and_reader_reject_outside_gpu(restricted, monkeypatch):
    health, _ = restricted
    monkeypatch.setattr(watch, 'experiment_result_roster_complete', lambda _: True)
    assert watch.publish_queue_allocation('e1', {}, list(range(8)), 'formal_idle', None) == [1, 4]
    path = health / 'gpu-allocations/e1.json'
    assert queue_control.read_allocation(path, 'e1', {1, 4})['gpus'] == [1, 4]
    value = json.loads(path.read_text()); value['gpus'] = [0, 1]; path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='resource policy'):
        queue_control.read_allocation(path, 'e1', {1, 4})
    (health / 'e1-queue.json').write_text(json.dumps({'jobs': {
        'foreign_slot': {'status': 'RUNNING', 'pid': 123, 'gpu': 6}}}))
    monkeypatch.setattr(watch, 'runner_matches', lambda *_: True)
    with pytest.raises(ValueError, match='outside'):
        watch.publish_queue_allocation('e1', {}, [1, 4], 'formal_idle', None)


def test_launch_admission_checks_policy_even_with_shared_or_no_allocation(restricted, monkeypatch):
    monkeypatch.setattr(queue, 'gpu_slot_busy', lambda _: False)
    monkeypatch.setattr(queue, 'gpu_compute_pids', lambda _: set())
    assert queue.launch_admitted(1, 'e1', None)
    assert not queue.launch_admitted(6, 'e1', {'shared_gpus': [6]})
    monkeypatch.setattr(queue, 'gpu_compute_pids', lambda _: {123})
    assert not queue.launch_admitted(1, 'e1', None)
    assert not queue.launch_admitted(4, 'e1', {'shared_gpus': [4]})


def test_malformed_resource_policy_fails_closed(restricted):
    health, policy = restricted
    for key, value in [('allowed_gpu_indices', [1, 1]), ('max_concurrent_jobs', 8),
                       ('allowed_gpu_indices', [True, 4]), ('status', 'DISABLED')]:
        (health / 'gpu-resource-policy.json').write_text(json.dumps({**policy, key: value}))
        with pytest.raises(ValueError):
            resources.allowed_gpu_indices(health)


def test_missing_resource_policy_fails_closed(tmp_path):
    with pytest.raises(ValueError, match='resource policy is missing'):
        resources.allowed_gpu_indices(tmp_path / 'health')


def test_policy_fails_closed_when_a_previously_authorized_gpu_disappears(restricted, monkeypatch):
    health, _ = restricted
    monkeypatch.setattr(resources, 'visible_gpu_indices', lambda: {0, 1, 2, 3})
    with pytest.raises(ValueError, match='unavailable device indices \\[4\\]'):
        resources.allowed_gpu_indices(health)


def test_policy_fails_closed_when_gpu_inventory_cannot_be_verified(restricted, monkeypatch):
    health, _ = restricted
    monkeypatch.setattr(resources, 'visible_gpu_indices', lambda: None)
    with pytest.raises(ValueError, match='Cannot verify GPU resource policy'):
        resources.allowed_gpu_indices(health)


def test_restricted_batch_releases_next_slot_for_interrupted_lr(restricted, monkeypatch):
    health, _ = restricted
    (health / 'e1_pilot_extension-queue.json').write_text(json.dumps({'jobs': {
        'interrupted': {'status': 'RETRY_EXHAUSTED', 'attempts': 3}}}))
    (health / 'batch_pilot-queue.json').write_text(json.dumps({'jobs': {
        'batch': {'status': 'RUNNING', 'pid': 123, 'gpu': 4}}}))
    monkeypatch.setattr(watch, 'runner_matches', lambda *_: True)
    assert watch.publish_queue_allocation('batch_pilot', {}, [1, 4], 'formal_idle', None) == [4, 1]
    allocation = json.loads((health / 'gpu-allocations/batch_pilot.json').read_text())
    assert allocation['max_parallel'] == 0
    assert allocation['gpus'] == [4, 1]
    watch.ACTION_STATE.write_text(json.dumps({'actions': {
        'e1_pilot_extension_queue': {'attempts': watch.MAX_ATTEMPTS}}}))
    watch.publish_queue_allocation('batch_pilot', {}, [1, 4], 'formal_idle', None)
    allocation = json.loads((health / 'gpu-allocations/batch_pilot.json').read_text())
    assert allocation['max_parallel'] == 2


def test_exhausted_lr_restarts_and_second_gpu_launches_batch_without_deadlock(restricted, monkeypatch):
    health, _ = restricted
    root = watch.ROOT
    for group, name in [('e1_pilot_extension', 'lr'), ('batch_pilot', 'batch')]:
        cfg = root / 'pretrain/configs' / group / f'{name}.env'
        cfg.parent.mkdir(parents=True)
        cfg.write_text('DUMMY=1\n')
    (health / 'e1_pilot_extension-queue.json').write_text(json.dumps({'status': 'FAIL', 'jobs': {
        'lr': {'status': 'RETRY_EXHAUSTED', 'attempts': 3}}}))
    watch.ACTION_STATE.write_text(json.dumps({'actions': {
        'e1_pilot_extension_queue': {'status': 'WAITING_EXTERNAL', 'attempts': 2}}}))
    monkeypatch.setattr(watch, 'BATCH_QUEUE_REPORT', health / 'batch_pilot-queue.json')
    monkeypatch.setattr(watch, 'experiment_result_roster_complete', lambda _: False)
    monkeypatch.setattr(watch, 'discover_group_runner', lambda *_: 0)
    monkeypatch.setattr(watch, 'runner_matches', lambda pid, script: pid in {123, 124})
    launched = []
    def launch(argv, **kwargs):
        launched.append(argv)
        return SimpleNamespace(pid=122 + len(launched))
    monkeypatch.setattr(watch.subprocess, 'Popen', launch)
    rows = [{'index': i, 'memory_used_mb': 0} for i in (1, 4)]
    _, lr_event = watch.start_group_if_eligible('e1_pilot_extension', True, 'qualified',
        {'status': 'PENDING'}, rows, True, qualification_passed=True)
    assert lr_event['decision'] == 'started'
    assert lr_event['gpus'] == [1]
    # The new queue has not yet rewritten the old RETRY_EXHAUSTED job record.
    # Its assigned launch slot already covers that pending renewal.
    _, batch_event = watch.start_group_if_eligible('batch_pilot', True, 'qualified',
        {'status': 'PENDING'}, rows, True, qualification_passed=True)
    assert batch_event['decision'] == 'started'
    assert batch_event['gpus'] == [4]
    assert json.loads((health / 'gpu-allocations/batch_pilot.json').read_text())['max_parallel'] == 1
    assert len(launched) == 2


def test_exhausted_lr_cannot_get_slots_beyond_outer_retry_budget(restricted):
    health, _ = restricted
    cfg = watch.ROOT / 'pretrain/configs/e1_pilot_extension/lr.env'
    cfg.parent.mkdir(parents=True); cfg.write_text('DUMMY=1\n')
    (health / 'e1_pilot_extension-queue.json').write_text(json.dumps({'jobs': {
        'lr': {'status': 'RETRY_EXHAUSTED', 'attempts': 3}}}))
    assert watch.publish_queue_allocation('e1_pilot_extension', {'attempts': watch.MAX_ATTEMPTS},
        [1, 4], 'formal_idle', None) == []


def test_four_gpu_expansion_admits_only_requested_cards_and_allows_four_e1_arms(restricted, monkeypatch):
    health, policy = restricted
    policy.update(allowed_gpu_indices=[0, 1, 2, 4], max_concurrent_jobs=4)
    (health / 'gpu-resource-policy.json').write_text(json.dumps(policy))
    rows = [{'index': i, 'memory_used_mb': 0} for i in range(8)]
    assert watch.select_experiment_gpus(rows) == [0, 1, 2, 4]
    monkeypatch.setattr(watch, 'group_enabled', lambda _, group: group != 'e2')
    monkeypatch.setattr(watch, 'experiment_result_roster_complete', lambda _: False)
    assert watch.publish_queue_allocation('e1', {}, list(range(8)), 'formal_idle', None) == [0, 1, 2, 4]
    monkeypatch.setattr(queue, 'gpu_slot_busy', lambda _: False)
    monkeypatch.setattr(queue, 'gpu_compute_pids', lambda _: set())
    assert queue.launch_admitted(0, 'e1', None)
    assert queue.launch_admitted(2, 'e1', None)
    assert not queue.launch_admitted(3, 'e1', None)


def test_production_canary_overrides_inherited_two_gpu_visibility(restricted, monkeypatch):
    health, policy = restricted
    policy.update(allowed_gpu_indices=list(range(8)), max_concurrent_jobs=8)
    (health / 'gpu-resource-policy.json').write_text(json.dumps(policy))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1,4')
    env, devices = canary.launch_environment(queue.ROOT)
    assert devices == list(range(8))
    assert env['CUDA_VISIBLE_DEVICES'] == '0,1,2,3,4,5,6,7'


def test_production_canary_cannot_escape_two_gpu_policy(restricted):
    with pytest.raises(ValueError, match='eight authorized'):
        canary.launch_environment(queue.ROOT)
