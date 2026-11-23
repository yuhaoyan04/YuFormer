"""Independently recompute the whole fixed production architecture/LR grid."""
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'experiments/e2_architecture/scripts'),str(ROOT/'pretrain/scripts')]
from development_validation_review import load_evaluation, compare
from freeze_1t_recipe import source_sha256
from audit_training_progress import audit, verify_reference_schedule
import benchmark_model_sizes as bench


def review(root, protocol_path):
    root, protocol_path = Path(root).resolve(), Path(protocol_path)
    if not protocol_path.is_absolute():
        protocol_path = root/protocol_path
    protocol_path = protocol_path.resolve()
    if not protocol_path.is_relative_to(root):
        raise ValueError('shape quality protocol must be inside the repository')
    protocol = json.loads(protocol_path.read_bytes())
    if (protocol.get('status') != 'PREDECLARED_BEFORE_ANY_NEW_GRID_TRAINING'
            or protocol.get('training_source_sha256') != source_sha256(root)
            or protocol.get('updates') != 200 or protocol.get('target_tokens') != 838860800
            or protocol.get('gate') != dict(primary_all_nine_strict=True,expanded_all_nine_strict=True,
                 aggregate_strict=True,noninferiority_margin=0,forecast_days_ceiling=33,
                 finite_all_updates=True,allocator_retries=0)):
        raise ValueError('shape grid needs the current source and unchanged strict predeclared endpoints')
    rows = protocol.get('cases', [])
    if (len(rows) != 4 or len({row['name'] for row in rows}) != 4
            or sorted((row['parameters'],row['peak_lr']) for row in rows) !=
               [(1002005248,.0001),(1002005248,.00015),(1219045632,.0001),(1219045632,.00015)]):
        raise ValueError('the exact fixed four-case authorized grid is required')
    folder = protocol_path.parent
    state_path = folder/'review.json'
    state = json.loads(state_path.read_bytes())
    if (state.get('status') != 'COMPLETE_BOUNDED_GRID_REQUIRES_SELECTION_REVIEW'
            or state.get('protocol_sha256') != bench.digest(protocol_path)
            or set(state.get('cases', {})) != {row['name'] for row in rows}):
        raise ValueError('the full actually completed bounded grid is required')
    inputs = dict(protocol['source_inputs'])
    for name, expected in inputs.items():
        if bench.digest(root/name) != expected:
            raise ValueError('shape-grid measured evidence changed: '+name)
    inputs[str(protocol_path.relative_to(root))] = bench.digest(protocol_path)
    inputs[str(state_path.relative_to(root))] = bench.digest(state_path)
    reference_protocol, reference, reference_inputs = load_evaluation(
        root,root/protocol['reference_plan'],protocol['reference_case'])
    inputs.update(reference_inputs)
    reference_primary = {d:reference['domains'][d]['original_primary_loss'] for d in reference_protocol['domains']}
    reference_macro = float(np.mean(list(reference_primary.values())))
    actual_cases = {}
    eligible = []
    for row in rows:
        name = row['name']
        measured = state['cases'][name]
        variant = row['variant']
        env_path = Path(variant['env'])
        env = dict(line.split('=',1) for line in env_path.read_text().splitlines()
                   if line.strip() and not line.lstrip().startswith('#'))
        run = Path(env['YUFORMER_PRETRAIN_OUT'])
        if not run.is_absolute():
            run = root/run
        if not run.resolve().is_relative_to(root) or run.name != name:
            raise ValueError('shape training result belongs to another run')
        result_path = run/'result.json'
        result = json.loads(result_path.read_bytes())
        identity = json.loads((run/'run_identity.json').read_bytes())
        config = yaml.safe_load((run/'resolved_config.yaml').read_bytes())
        if (measured.get('result_sha256') != bench.digest(result_path)
                or result.get('identity_sha256') != identity.get('identity_sha256')
                or identity.get('source_sha256') != protocol['training_source_sha256']
                or config['optim']['lr'] != row['peak_lr']
                or config['data'].get('stage_sample_limits') != [204800]
                or config['data'].get('sampling_policy') != 'block_permutation'
                or config['data'].get('stages') != [str(root/'data/token_streams/pretrain-1t/warmup.tokens.bin')]):
            raise ValueError('shape result/config/source/actual production budget differs: '+name)
        evaluated = bench.evaluate(result,steps=200,skip_steps=50)
        if (evaluated != measured['evaluation'] or evaluated['status'] != 'PASS_SCREENING'
                or not evaluated['within_33_days']):
            raise ValueError('shape actual system/throughput result did not pass: '+name)
        runtime = audit(run,check_checkpoint=True)
        runtime['reference_scheduler'] = verify_reference_schedule(config)
        runtime_path = folder/(name+'-runtime.json')
        if (json.loads(runtime_path.read_bytes()) != runtime or runtime['checked_lr_points'] != 400
                or runtime['checkpoint']['status'] != 'PASS' or runtime['checkpoint']['step'] != 200
                or runtime['checkpoint']['samples_consumed'] != 204800
                or runtime['checkpoint']['global_tokens_seen'] != 838860800
                or runtime['checkpoint']['optimizer_groups_checked'] != row['optimizer_groups']
                or runtime['reference_scheduler']['scheduler_epochs_checked'] != 201):
            raise ValueError('shape actual LR/checkpoint/runtime does not independently reproduce: '+name)
        primary = {d:dict(candidate=result['metrics'][f'validation_metrics/domain/{d}/loss']['last'],
                         baseline=reference_primary[d]) for d in reference_primary}
        if any(not math.isfinite(v['candidate']) for v in primary.values()):
            raise ValueError('shape primary metrics must all be finite')
        for value in primary.values():
            value['delta'] = value['candidate']-value['baseline']
        primary_ok = (all(v['delta'] < 0 for v in primary.values())
                      and result['metrics']['validation_metrics/loss']['last'] < reference_macro)
        if primary != measured['primary_comparison'] or primary_ok != measured['primary_passed']:
            raise ValueError('shape primary quality flags do not match actual metrics')
        actual = dict(case=name,parameters=row['parameters'],peak_lr=row['peak_lr'],
                      run_dir=str(run.relative_to(root)),primary_comparison=primary,
                      primary_all_nine_passed=bool(primary_ok),status='FAIL_PRIMARY_ALL_NINE')
        if primary_ok:
            eligible.append(name)
            plan_path = folder/'validation/plan.json'
            p, candidate, evaluation_inputs = load_evaluation(root,plan_path,name)
            if any(p[k] != protocol['expanded_window'][k] for k in protocol['expanded_window']):
                raise ValueError('shape actual expanded window differs from its predeclaration')
            comparison = compare(p,candidate,reference_protocol,reference,np.random.default_rng(p['uncertainty']['seed']))
            passed = (all(v['mean_strictly_improves'] for v in comparison.values())
                      and candidate['aggregate_mean_loss'] < reference['aggregate_mean_loss'])
            actual.update(expanded_comparison=comparison,expanded_macro_loss=candidate['aggregate_mean_loss'],
                          status='PASS_PRIMARY_AND_EXPANDED_ALL_NINE' if passed else 'FAIL_EXPANDED_ALL_NINE')
            inputs.update(evaluation_inputs)
            if (measured['expanded_comparison'] != comparison
                    or measured['expanded_macro_loss'] != candidate['aggregate_mean_loss']):
                raise ValueError('shape stored expanded comparison does not reproduce')
        if measured['quality_status'] != actual['status']:
            raise ValueError('shape stored quality status does not reproduce')
        for path in (result_path,run/'run_identity.json',run/'resolved_config.yaml',env_path,runtime_path,
                     Path(variant['config']),Path(row['model_path']),Path(measured['log'])):
            inputs[str(path.relative_to(root))] = bench.digest(path)
        actual_cases[name] = actual
    if eligible:
        plan = json.loads((folder/'validation/plan.json').read_bytes())
        if set(plan['cases']) != set(eligible):
            raise ValueError('shape expanded roster must include every primary-passing candidate')
    passing = [v for v in actual_cases.values() if v['status']=='PASS_PRIMARY_AND_EXPANDED_ALL_NINE']
    passing.sort(key=lambda v:(-v['parameters'],v['expanded_macro_loss'],v['case']))
    preferred = passing[0]['case'] if passing else None
    if preferred != state.get('preferred_case'):
        raise ValueError('shape selection does not match the predeclared largest-model ranking')
    inputs[str(Path(__file__).relative_to(ROOT))] = bench.digest(__file__)
    return dict(status='PASS_INDEPENDENT_PRODUCTION_SHAPE_GRID' if passing else 'FAIL_ALL_SHAPE_CANDIDATES',
                preferred_case=preferred,cases=actual_cases,source_inputs=inputs,
                training_source_sha256=protocol['training_source_sha256'],
                interpretation='Fixed bounded development grid; all earlier failures preserved; no unseen-test or global optimum claim.',
                production_qualified=False,formal_pretrain_started=False)
