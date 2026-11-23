import json
from pathlib import Path

import pytest

from yuformer_eval.context_pipeline import ContextPipeline, formal_completion_state


def write(path, value):
    path.write_text(json.dumps(value))


def test_terminal_marker_write_order_cannot_start_next_stage_prematurely(tmp_path):
    attempt = {'status': 'RUNNING', 'attempt_id': 'current', 'identity_sha256': 'id'}
    write(tmp_path/'current_attempt.json', attempt)
    assert formal_completion_state(tmp_path) == 'WAITING_FOR_ACTUAL_4K_COMPLETION'
    complete = {'status': 'PASS', 'attempt_id': 'old', 'identity_sha256': 'id', 'returncode': 0}
    write(tmp_path/'training_complete.json', complete)
    assert formal_completion_state(tmp_path) == 'WAITING_FOR_ACTUAL_4K_COMPLETION'
    complete['attempt_id'] = 'current'
    write(tmp_path/'training_complete.json', complete)
    assert formal_completion_state(tmp_path) == 'WAITING_FOR_ACTUAL_4K_COMPLETION'
    attempt['status'] = 'PASS'
    write(tmp_path/'current_attempt.json', attempt)
    assert formal_completion_state(tmp_path) == 'ACTUAL_4K_PROCESS_COMPLETED_REQUIRES_RESULT_AND_NATIVE_AUDIT'


@pytest.mark.parametrize('fault', ['failure', 'stale_attempt', 'missing_complete'])
def test_failed_or_incomplete_parent_cannot_advance(tmp_path, fault):
    attempt = {'status': 'PASS', 'attempt_id': 'current', 'identity_sha256': 'id'}
    complete = {**attempt, 'returncode': 0}
    if fault == 'failure':
        complete['status'] = 'FAIL'
    elif fault == 'stale_attempt':
        complete['attempt_id'] = 'old'
    write(tmp_path/'current_attempt.json', attempt)
    if fault != 'missing_complete':
        write(tmp_path/'training_complete.json', complete)
    with pytest.raises(ValueError):
        formal_completion_state(tmp_path)


def test_eval_and_long_stage_children_do_not_inherit_formal_cursor_or_lease(monkeypatch):
    pipeline = object.__new__(ContextPipeline)
    pipeline.root = Path('/declared/repo')
    pipeline.base_env = {'YUFORMER_PYTHON': '/declared/python',
                         'YUFORMER_TRAINING_BLOCK_MANIFEST': '/formal/4k/blocks.json',
                         'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7'}
    monkeypatch.setenv('YUFORMER_RUN_LEASE_HELD', '1')
    monkeypatch.setenv('YUFORMER_ATTEMPT_ID', 'formal-attempt')
    monkeypatch.setenv('PYTORCH_CUDA_ALLOC_CONF', 'inherited allocator')
    env = pipeline.environment(cuda='5')
    assert env['CUDA_VISIBLE_DEVICES'] == '5'
    for key in ('YUFORMER_RUN_LEASE_HELD', 'YUFORMER_ATTEMPT_ID',
                'YUFORMER_TRAINING_BLOCK_MANIFEST', 'PYTORCH_CUDA_ALLOC_CONF'):
        assert key not in env
    assert pipeline.base_env['CUDA_VISIBLE_DEVICES'] == '0,1,2,3,4,5,6,7'


def test_running_formal_pretrain_only_waits_and_never_starts_auxiliary_or_gpu_work(tmp_path, monkeypatch):
    write(tmp_path/'current_attempt.json', {'status': 'RUNNING', 'attempt_id': 'actual'})
    pipeline = object.__new__(ContextPipeline)
    pipeline.config = {'formal_run': str(tmp_path)}
    calls = []
    pipeline.guard = lambda: calls.append('guard')
    pipeline.publish = lambda *args, **kwargs: calls.append('publish')
    pipeline.audit_checkpoint = lambda *args: pytest.fail('must not touch checkpoint finalization while pretrain runs')
    def yield_wait(seconds):
        assert seconds == 30
        raise InterruptedError('unit test stops the wait, not training')
    monkeypatch.setattr('yuformer_eval.context_pipeline.time.sleep', yield_wait)
    with pytest.raises(InterruptedError):
        pipeline.run()
    assert calls == ['publish', 'guard', 'publish']


def test_failed_actual_16k_admission_prevents_any_64k_training(tmp_path, monkeypatch):
    complete = {'status': 'PASS', 'attempt_id': 'actual', 'identity_sha256': 'id', 'returncode': 0}
    write(tmp_path/'current_attempt.json', complete)
    write(tmp_path/'training_complete.json', complete)
    pipeline = object.__new__(ContextPipeline)
    pipeline.config = {'formal_run': str(tmp_path)}
    pipeline.state = {'stages': {}}
    pipeline.publish = lambda *args, **kwargs: None
    pipeline.audit_checkpoint = lambda *args: ({'file': 'unit-baseline-audit'}, {'seq_len': 4096})
    trained = []
    def train(seq, parent, **kwargs):
        trained.append(seq)
        return {'file': 'unit-candidate-audit'}, {'seq_len': seq}
    pipeline.train_long = train
    pipeline.measure = lambda *args: {'unit': 'measurements only'}
    def fail_gate(*args):
        raise ValueError('actual 16k capability gate failed')
    pipeline.admit = fail_gate
    with pytest.raises(ValueError, match='gate failed'):
        pipeline.run()
    assert trained == [16384]
