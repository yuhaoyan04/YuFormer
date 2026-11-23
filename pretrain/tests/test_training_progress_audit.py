import copy
import math

import pytest

from audit_training_progress import expected_lr, verify_checkpoint, verify_scalar_series


@pytest.fixture
def config():
    return {'optim': {'lr': .0001, 'lr_schedule': 'wsd', 'warmup_steps': 8,
                      'decay_steps': 8, 'min_lr_ratio': .1},
            'train': {'expected_iters': 24, 'micro_batch_size': 1, 'grad_accum': 1},
            'parallel': {'world_size': 8},
            'data': {'seq_len': 4, 'stage_sample_limits': [64, 64, 64],
                     'sampling_policy': 'block_permutation'}}


def test_logged_lr_uses_update_minus_one_while_checkpoint_stores_next_lr(config):
    series = {'loss_metrics/global_avg_loss': [{'step': i, 'value': 6.} for i in range(1, 25)]}
    for group in (0, 1):
        series[f'lr/AdamW/{group}'] = [{'step': i, 'value': expected_lr(config, i-1)} for i in range(1, 25)]
    result = verify_scalar_series(config, series)
    assert result['checked_lr_points'] == 48
    assert result['status'] == 'PASS_OBSERVED_PREFIX'
    assert expected_lr(config, 0) == .0001/8
    assert expected_lr(config, 7) == expected_lr(config, 8) == expected_lr(config, 16) == .0001
    assert expected_lr(config, 23) > expected_lr(config, 24)
    assert math.isclose(expected_lr(config, 24), .00001)


@pytest.mark.parametrize('fault', ['wrong_lr', 'off_by_one_lr', 'nonfinite_loss', 'missing_lr'])
def test_logged_actual_optimizer_state_cannot_hide_bad_warmup(config, fault):
    series = {'loss_metrics/global_avg_loss': [{'step': 2, 'value': 6.}],
              'lr/AdamW/0': [{'step': 2, 'value': expected_lr(config, 1)}]}
    if fault == 'wrong_lr': series['lr/AdamW/0'][0]['value'] *= 2
    elif fault == 'off_by_one_lr': series['lr/AdamW/0'][0]['value'] = expected_lr(config, 2)
    elif fault == 'nonfinite_loss': series['loss_metrics/global_avg_loss'][0]['value'] = float('nan')
    else: series.pop('lr/AdamW/0')
    with pytest.raises(ValueError):
        verify_scalar_series(config, series)


def state(config, step):
    return {'train_state.step': step, 'lr_scheduler.last_epoch': step,
        'train_state.samples_consumed': step*8, 'dataloader.consumed_samples': step*8,
        'train_state.global_tokens_seen': step*32, 'dataloader.seq_len': 4,
        'dataloader.sampling_policy': 'block_permutation',
        'dataloader.block_permutation_manifest_sha256': 'bound-production-order',
        'optimizer.param_groups.first.lr': expected_lr(config, step),
        'optimizer.param_groups.second.lr': expected_lr(config, step)}


@pytest.mark.parametrize('step,next_stage', [(0, 0), (8, 1), (16, 2), (24, None)])
def test_checkpoint_lr_tokens_and_stage_cursor_agree_at_every_wsd_boundary(config, step, next_stage):
    result = verify_checkpoint(config, state(config, step), {'manifest_sha256': 'bound-production-order'})
    assert result['status'] == 'PASS'
    assert result['next_logical_stage_index'] == next_stage
    assert result['optimizer_groups_checked'] == 2


@pytest.mark.parametrize('field,value', [('lr_scheduler.last_epoch', 9),
    ('train_state.samples_consumed', 63), ('dataloader.consumed_samples', 72),
    ('train_state.global_tokens_seen', 255), ('dataloader.seq_len', 8),
    ('dataloader.sampling_policy', 'sequential'),
    ('dataloader.block_permutation_manifest_sha256', 'different-order'),
    ('optimizer.param_groups.second.lr', .0002)])
def test_checkpoint_cannot_resume_another_lr_cursor_or_order(config, field, value):
    values = state(config, 8)
    values[field] = value
    with pytest.raises(ValueError):
        verify_checkpoint(config, values, {'manifest_sha256': 'bound-production-order'})
