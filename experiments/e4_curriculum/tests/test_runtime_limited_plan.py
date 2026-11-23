import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments/e4_curriculum/scripts'))
from runtime_limited_plan import derive_runtime_limited_plan, resolve_effective_budget


def inputs():
    budget = json.loads((ROOT/'data/health/user-total-pretrain-and-length-budget.json').read_bytes())
    rates = (48858.0234375, 40894.80859375, 24933.76953125)
    measurements = {stage['name']: {'seq_len': stage['seq_len'], 'world_size': 8,
        'parameters': budget['parameters'], 'p10_tps_per_device': rate}
        for stage, rate in zip(budget['stages'], rates)}
    return budget, measurements


def test_time_priority_keeps_larger_model_and_long_budgets_without_approving_a_launch():
    budget, rates = inputs()
    original = copy.deepcopy(budget)
    result = derive_runtime_limited_plan(budget, rates)
    assert budget == original
    assert result['production_qualified'] is False
    assert .96e12 < result['proposed_total_tokens'] < 1e12
    assert result['total_days'] <= result['planning_ceiling_days'] < 33
    plan = result['proposed_budget']
    assert plan['parameters'] == 1516600064
    assert plan['stages'][1:] == budget['stages'][1:]
    assert plan['stages'][0]['phases'][0] == budget['stages'][0]['phases'][0]
    assert plan['stages'][0]['phases'][2] == budget['stages'][0]['phases'][2]
    assert result['total_days'] + 4194304/(8*rates['4k-pretrain']['p10_tps_per_device']*.9*86400) > result['planning_ceiling_days']


def test_fast_execution_does_not_increase_the_authorized_nominal_total():
    budget, rates = inputs()
    for row in rates.values():
        row['p10_tps_per_device'] = 100000
    result = derive_runtime_limited_plan(budget, rates)
    assert result['proposed_total_tokens'] == budget['actual_total_tokens']


def test_impossible_long_stage_cost_is_not_hidden_by_cutting_away_wsd():
    budget, rates = inputs()
    rates['64k-midtrain']['p10_tps_per_device'] = 100
    with pytest.raises(ValueError, match='cannot fit'):
        derive_runtime_limited_plan(budget, rates)


def test_only_the_independently_reproducible_runtime_allocation_is_accepted():
    budget, rates = inputs()
    proposal = derive_runtime_limited_plan(budget, rates)
    actual, forecast = resolve_effective_budget(budget, proposal)
    assert actual['actual_total_tokens'] == proposal['proposed_total_tokens']
    assert forecast['total_days'] <= 33
    proposal['proposed_budget']['stages'][0]['phases'][1]['samples'] += 8
    with pytest.raises(ValueError, match='does not reproduce'):
        resolve_effective_budget(budget, proposal)
