import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments/e4_curriculum/scripts'))
from global_training_budget import forecast_global_training
from adopt_training_recipe import require_global_runtime, verify_global_runtime_evidence
from runtime_limited_plan import derive_runtime_limited_plan


@pytest.fixture
def inputs():
    budget = json.loads((ROOT/'data/health/user-total-pretrain-and-length-budget.json').read_bytes())
    rates = {s['name']: {'seq_len': s['seq_len'], 'world_size': 8,
        'parameters': budget['parameters'], 'p10_tps_per_device': 55000} for s in budget['stages']}
    return budget, rates


def test_entire_chain_is_accounted_once(inputs):
    result = forecast_global_training(*inputs)
    assert result['status'] == 'WITHIN_APPROVED_TOTAL_RUNTIME'
    assert result['total_target_labels'] == 999997571072
    assert result['production_qualified'] is False
    assert result['total_days'] == pytest.approx(999997571072/(8*55000*.9*86400))


def test_fast_4k_cannot_hide_slow_length_extension(inputs):
    budget, rates = inputs
    rates['64k-midtrain']['p10_tps_per_device'] = 1000
    assert forecast_global_training(budget, rates)['status'] == 'EXCEEDS_APPROVED_TOTAL_RUNTIME'


def test_current_accepted_quality_policy_cannot_fall_back_to_4k_only_days():
    with pytest.raises(ValueError, match='entire 4K/16K/64K'):
        require_global_runtime({'production_user_tradeoff_review': {}}, {})


def test_inline_forecast_without_bound_actual_budget_is_rejected(tmp_path, inputs):
    budget, rates = inputs
    selection = {'global_training_budget': {'approved_budget': budget, 'measurements': rates,
        'forecast': forecast_global_training(budget, rates)}}
    with pytest.raises(ValueError, match='actual user-approved'):
        verify_global_runtime_evidence(tmp_path, selection, {})


def test_formal_4k_consumption_must_reserve_the_long_stage_budget(inputs):
    budget, rates = inputs
    selection = {'global_training_budget': {'approved_budget': budget, 'measurements': rates,
        'forecast': forecast_global_training(budget, rates)},
        'sustained_measurement': {'p10_tps_per_device': 55000}}
    base = {'train': {'expected_iters': 238418}, 'data': {'seq_len': 4096,
        'stage_sample_limits': [p['samples'] for p in budget['stages'][0]['phases']]}}
    with pytest.raises(ValueError, match='prepared 4K consumption'):
        require_global_runtime(selection, base)


def test_actual_small_runtime_adjustment_is_gated_as_a_full_chain_budget(inputs):
    budget, rates = inputs
    for row, value in zip(rates.values(), (48858.0234375, 40894.80859375, 24933.76953125)):
        row['p10_tps_per_device'] = value
    proposal = derive_runtime_limited_plan(budget, rates)
    effective = proposal['proposed_budget']
    selection = {'global_training_budget': {'approved_budget': budget, 'measurements': rates,
        'runtime_execution_plan': proposal, 'forecast': proposal['forecast']},
        'sustained_measurement': {'p10_tps_per_device': 48858.0234375}}
    base = {'train': {'expected_iters': effective['stages'][0]['updates']}, 'data': {'seq_len': 4096,
        'stage_sample_limits': [p['samples'] for p in effective['stages'][0]['phases']]}}
    assert require_global_runtime(selection, base)['total_days'] <= 33


@pytest.mark.parametrize('fault', ['missing_length', 'extra_stage', 'wrong_world', 'wrong_model',
    'wrong_length', 'nonfinite_rate', 'negative_rate', 'boolean_rate', 'token_closure', 'phase_closure'])
def test_partial_or_mismatched_chain_is_rejected(inputs, fault):
    budget, rates = copy.deepcopy(inputs)
    if fault == 'missing_length': rates.pop('64k-midtrain')
    elif fault == 'extra_stage': rates['extra'] = rates['64k-midtrain']
    elif fault == 'wrong_world': rates['16k-midtrain']['world_size'] = 4
    elif fault == 'wrong_model': rates['16k-midtrain']['parameters'] = 530107904
    elif fault == 'wrong_length': rates['64k-midtrain']['seq_len'] = 4096
    elif fault == 'nonfinite_rate': rates['64k-midtrain']['p10_tps_per_device'] = float('nan')
    elif fault == 'negative_rate': rates['64k-midtrain']['p10_tps_per_device'] = -1
    elif fault == 'boolean_rate': rates['64k-midtrain']['p10_tps_per_device'] = True
    elif fault == 'token_closure': budget['stages'][1]['tokens'] += 1
    else: budget['stages'][0]['phases'][1]['updates'] += 1
    with pytest.raises(ValueError):
        forecast_global_training(budget, rates)
