"""Account for pretrain and both context stages in one wall-time budget.

This arithmetic review does not qualify its inputs. The adopter must bind each
measurement to the actual result, execution configuration and source identity.
"""
import math


def validate_global_budget(budget):
    if (budget.get('status') != 'ACTIVE_USER_APPROVED_TOTAL_1T_INCLUDING_LONG_CONTEXT'
            or budget.get('world_size') != 8
            or budget.get('launch_total_runtime_ceiling_days') != 33
            or budget.get('target_total_tokens') != 10**12
            or budget.get('tokens_per_update') != 4194304):
        raise ValueError('global budget must preserve the approved eight-GPU total-1T/33-day policy')
    stages = budget.get('stages', [])
    expected = [('4k-pretrain', 4096), ('16k-midtrain', 16384), ('64k-midtrain', 65536)]
    if [(s.get('name'), s.get('seq_len')) for s in stages] != expected:
        raise ValueError('global budget must include 4K, 16K and 64K exactly once')
    for stage in stages:
        updates, tokens = stage.get('updates'), stage.get('tokens')
        if (type(updates) is not int or updates <= 0 or type(tokens) is not int
                or tokens != updates*budget['tokens_per_update']):
            raise ValueError('stage label budget must close exact optimizer updates')
        if stage['seq_len'] > 4096 and stage.get('short_replay_fraction') != .25:
            raise ValueError('long stages must retain the approved 25% short replay')
    total_tokens = sum(row['tokens'] for row in stages)
    if (total_tokens != budget.get('actual_total_tokens')
            or sum(row['updates'] for row in stages) != budget.get('total_optimizer_updates')
            or not 0 <= budget['target_total_tokens']-total_tokens < budget['tokens_per_update']):
        raise ValueError('global accounting does not close the approved total label budget')
    phases = stages[0].get('phases', [])
    if ([p.get('name') for p in phases] != ['warmup', 'stable', 'decay']
            or any(type(p.get('updates')) is not int or p['updates'] <= 0
                or type(p.get('samples')) is not int or p['samples'] <= 0
                or p.get('tokens') != p['updates']*budget['tokens_per_update']
                or p.get('samples')*4096 != p['tokens'] for p in phases)
            or sum(p['updates'] for p in phases) != stages[0]['updates']):
        raise ValueError('4K warmup/stable/decay consumption does not close its allocated budget')
    return stages


def forecast_global_training(budget, measurements):
    stages = validate_global_budget(budget)
    if not isinstance(measurements, dict) or set(measurements) != {s['name'] for s in stages}:
        raise ValueError('all three stage throughput measurements are required')
    rows = []
    for stage in stages:
        measured = measurements[stage['name']]
        p10 = measured.get('p10_tps_per_device')
        if (type(p10) not in (int, float) or not math.isfinite(p10) or p10 <= 0
                or measured.get('world_size') != 8 or measured.get('seq_len') != stage['seq_len']
                or measured.get('parameters') != budget.get('parameters')):
            raise ValueError('stage throughput belongs to another model, length or world size')
        rows.append({'name': stage['name'], 'tokens': stage['tokens'], 'updates': stage['updates'],
                     'p10_tps_per_device': p10, 'days': stage['tokens']/(8*p10*.9*86400)})
    days = sum(row['days'] for row in rows)
    return {'status': 'WITHIN_APPROVED_TOTAL_RUNTIME' if days <= 33 else 'EXCEEDS_APPROVED_TOTAL_RUNTIME',
            'production_qualified': False, 'requires_bound_actual_measurements': True,
            'duty_factor': .9, 'ceiling_days': 33, 'total_days': days,
            'total_target_labels': sum(row['tokens'] for row in rows), 'stages': rows}
