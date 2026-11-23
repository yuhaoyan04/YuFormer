"""Propose the maximum token count under the user's prioritized full-chain time."""
import copy
import importlib.util
import math
from pathlib import Path

PLANNING_CEILING_DAYS = 32.8

_spec = importlib.util.spec_from_file_location('_runtime_budget_structure',
    Path(__file__).with_name('global_training_budget.py'))
_budget = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_budget)
validate_global_budget = _budget.validate_global_budget
forecast_global_training = _budget.forecast_global_training


def derive_runtime_limited_plan(nominal, measurements):
    """Keep both long stages, architecture and WSD warmup/decay; shorten stable.

    The result is a proposal, never a launch admission. Actual paired quality,
    source-bound speed, new consumed-domain/overlap/order audits and the updated
    production budget must all pass before it can be frozen.
    """
    validate_global_budget(nominal)
    original_forecast = forecast_global_training(nominal, measurements)
    result = copy.deepcopy(nominal)
    long_days = sum(row['days'] for row in original_forecast['stages'][1:])
    rate = measurements['4k-pretrain']['p10_tps_per_device']
    # Keep a small additional allowance for measured canary variation and stage
    # handoffs, on top of the existing .9 duty factor. Launch remains gated at33.
    available = math.floor((PLANNING_CEILING_DAYS-long_days)*86400*8*rate*.9/nominal['tokens_per_update'])
    updates = min(nominal['stages'][0]['updates'], available)
    base = result['stages'][0]
    stable = updates-base['phases'][0]['updates']-base['phases'][2]['updates']
    if stable <= 0:
        raise ValueError('33-day budget cannot fit the retained warmup/decay and both long stages')
    base.update(updates=updates, tokens=updates*nominal['tokens_per_update'])
    base['phases'][1].update(updates=stable, tokens=stable*nominal['tokens_per_update'], samples=stable*1024)
    result['actual_total_tokens'] = sum(row['tokens'] for row in result['stages'])
    result['total_optimizer_updates'] = sum(row['updates'] for row in result['stages'])
    days = sum(row['tokens']/(8*measurements[row['name']]['p10_tps_per_device']*.9*86400)
               for row in result['stages'])
    rows = [{'name': row['name'], 'tokens': row['tokens'], 'updates': row['updates'],
             'p10_tps_per_device': measurements[row['name']]['p10_tps_per_device'],
             'days': row['tokens']/(8*measurements[row['name']]['p10_tps_per_device']*.9*86400)}
            for row in result['stages']]
    return {'status': 'PROPOSAL_REQUIRES_BOUND_SPEED_QUALITY_AND_NEW_DATA_ADMISSION',
            'production_qualified': False, 'nominal_total_tokens': nominal['actual_total_tokens'],
            'proposed_total_tokens': result['actual_total_tokens'],
            'token_reduction_fraction': 1-result['actual_total_tokens']/nominal['actual_total_tokens'],
            'total_days': days, 'next_4k_update_would_exceed_ceiling': updates == available,
            'planning_ceiling_days': PLANNING_CEILING_DAYS,
            'retained_long_budgets': True, 'retained_warmup_decay_steps': True,
            'measurements': copy.deepcopy(measurements),
            'forecast': {'status': 'WITHIN_APPROVED_TOTAL_RUNTIME' if days <= 33 else 'EXCEEDS_APPROVED_TOTAL_RUNTIME',
                'production_qualified': False, 'requires_bound_actual_measurements': True,
                'duty_factor': .9, 'ceiling_days': 33, 'total_days': days,
                'total_target_labels': result['actual_total_tokens'], 'stages': rows},
            'proposed_budget': result}


def resolve_effective_budget(nominal, proposal):
    expected = derive_runtime_limited_plan(nominal, proposal.get('measurements', {}))
    if expected != proposal:
        raise ValueError('runtime execution budget does not reproduce the maximum feasible declared allocation')
    # "Approximately1T, prioritize33days" permits the observed small adjustment;
    # it must not become an implicit authorization for a materially smaller run.
    if not 0 <= expected['token_reduction_fraction'] <= .04:
        raise ValueError('runtime ceiling requires more than a small adjustment to the approximate1T target')
    return expected['proposed_budget'], expected['forecast']
