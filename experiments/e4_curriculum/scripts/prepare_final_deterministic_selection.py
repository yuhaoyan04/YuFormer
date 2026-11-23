"""Publish a selection only from the completed, strict actual-production pair."""
import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'experiments/e2_architecture/scripts'),str(ROOT/'pretrain/scripts')]
import benchmark_model_sizes as bench
import run_final_deterministic_production_controls as controls
from review_production_matched_controls import review
from benchmark_budget_lr_controls import require_matched_contract
from checkpoint_durability_evidence import require_checkpoint_directory_durability
from production_selection_evidence import require_production_quality_review
from adopt_training_recipe import require_selection, verify_sustained_result, source_sha256, training_execution_contract
from runtime_limited_plan import resolve_effective_budget
from audit_training_progress import audit, verify_reference_schedule
from allocator_retry_review import require_no_allocator_retries
from sustained_memory_review import review_memory_log
from frozen_training_evidence import snapshot_files, publish_checked, production_mutation_lease, production_output_dir
import prepare_production_model as original_preparer

BASE = 'experiments/e2_architecture/results/final_deterministic_production_controls_20261115'
RUNS = 'data/experiments/final_deterministic_production_controls_20261115'
SELECTION = 'experiments/e2_architecture/results/model-scale-production-selection.json'
ASSESSMENT = 'experiments/e2_architecture/results/model-scale-scientific-assessment.json'


def encoded(value):
    return (json.dumps(value,sort_keys=True,indent=2)+'\n').encode()


def prepare(root, runtime_path, block_manifest):
    root = Path(root).resolve()
    health_name = 'data/health/final-deterministic-production-controls.json'
    review_name = 'data/health/final-deterministic-production-review.json'
    if not (root/health_name).is_file() or not (root/review_name).is_file():
        raise ValueError('actual production matched training and strict quality proof are required')
    health = json.loads((root/health_name).read_bytes())
    review_health = json.loads((root/review_name).read_bytes())
    if (health['status'] != 'COMPLETE_REQUIRES_PAIRED_PRIMARY_AND_EXPANDED_QUALITY_REVIEW'
            or review_health['status'] != 'PASS_PRODUCTION_PRIMARY_AND_EXPANDED_ALL_NINE'):
        raise ValueError('actual production all-nine quality review has not passed')
    if controls.input_identity(health['probe'],health['loaded_source_inputs']) != health['source_qualification_identity']:
        raise ValueError('actual production matched training inputs changed')
    declaration = json.loads((root/BASE/'protocol.json').read_bytes())
    controls.require_declared_source(declaration)
    plan_path = root/BASE/'validation/plan.json'
    quality_path = root/BASE/'validation/review.json'
    actual_quality = review(plan_path)
    if (actual_quality != json.loads(quality_path.read_bytes())
            or actual_quality['status'] != 'PASS_PRODUCTION_PRIMARY_AND_EXPANDED_ALL_NINE'
            or bench.digest(quality_path) != review_health['report_sha256']):
        raise ValueError('actual production quality evidence does not independently reproduce')
    current_source = source_sha256(root)
    runtime_name = str(runtime_path.relative_to(root))
    runtime_document = json.loads(runtime_path.read_bytes())
    if runtime_document['training_source_sha256'] != current_source:
        raise ValueError('measured runtime allocation belongs to another training source')
    nominal_path = root/'data/health/user-total-pretrain-and-length-budget.json'
    nominal = json.loads(nominal_path.read_bytes())
    if bench.digest(nominal_path) != runtime_document['nominal_budget_sha256']:
        raise ValueError('runtime allocation has another nominal user budget')
    effective, global_forecast = resolve_effective_budget(nominal, runtime_document['proposal'])
    inputs = dict(actual_quality['source_inputs'])
    for name, expected in runtime_document['source_inputs'].items():
        if bench.digest(root/name) != expected:
            raise ValueError('actual measured runtime input changed: '+name)
        inputs[name] = expected
    inputs[runtime_name] = bench.digest(runtime_path)
    inputs[str(nominal_path.relative_to(root))] = bench.digest(nominal_path)
    global_bundle = dict(approved_budget=nominal, approved_budget_sha256=bench.digest(nominal_path),
        measurements=runtime_document['proposal']['measurements'], forecast=global_forecast,
        runtime_execution_plan=runtime_document['proposal'], runtime_execution_plan_file=runtime_name,
        runtime_execution_plan_sha256=bench.digest(runtime_path))
    inputs.update(require_checkpoint_directory_durability(root))
    for path in (root/health_name,root/review_name,quality_path,root/BASE/'protocol.json',Path(__file__),
                 Path(controls.__file__),Path(original_preparer.__file__),
                 root/'experiments/e4_curriculum/scripts/production_selection_evidence.py',
                 root/'experiments/e4_curriculum/scripts/sustained_memory_review.py',
                 root/'experiments/e4_curriculum/scripts/allocator_retry_review.py',
                 root/'pretrain/scripts/audit_training_progress.py'):
        inputs[str(path.relative_to(root))] = bench.digest(path)
    measurements,configs,runtimes = {},{},{}
    for case in ('matched_baseline_b8','larger_lr1e4_b8'):
        run = root/RUNS/case
        row = health['cases'][case]
        result = json.loads((run/'result.json').read_bytes())
        identity = json.loads((run/'run_identity.json').read_bytes())
        config = yaml.safe_load((run/'resolved_config.yaml').read_bytes())
        evaluated = bench.evaluate(result,steps=200,skip_steps=50)
        if (bench.digest(run/'result.json') != row['result_sha256'] or identity['source_sha256'] != current_source
                or result['identity_sha256'] != identity['identity_sha256'] or evaluated != row['evaluation']
                or evaluated['status'] != 'PASS_SCREENING'):
            raise ValueError('actual production sustained result/source/throughput does not close')
        data = config['data']
        if (data.get('sampling_policy') != 'block_permutation'
                or data.get('stage_sample_limits') != [204800]
                or data.get('meta') != [str(root/'data/token_streams/pretrain-1t/warmup.meta.json')]
                or data.get('stages') != [str(root/'data/token_streams/pretrain-1t/warmup.tokens.bin')]
                or data.get('block_permutation_manifest') != health['probe']['probe_order_manifest']):
            raise ValueError('quality comparison did not consume the actual declared production samples')
        model_path = Path(identity['model_configs'])
        if bench.digest(model_path) != identity['model_configs_sha256']:
            raise ValueError('actually trained production-control model changed')
        paths = [run/'result.json',run/'run_identity.json',run/'resolved_config.yaml',model_path,Path(row['log'])]
        for key in ('config','env'):
            path = Path(row['variant'][key])
            if bench.digest(path) != row['variant'][key+'_sha256']:
                raise ValueError('actual production-control execution contract changed')
            paths.append(path)
        inputs.update({str(p.relative_to(root)):bench.digest(p) for p in paths})
        raw_log = Path(row['log']).read_bytes()
        memory = review_memory_log(raw_log,steps=200,skip_steps=50)
        require_no_allocator_retries(raw_log)
        runtime = audit(run,check_checkpoint=True)
        runtime['reference_scheduler'] = verify_reference_schedule(config)
        groups = 146 if case == 'matched_baseline_b8' else 164
        if (runtime['status'] != 'PASS_OBSERVED_PREFIX' or runtime['latest_logged_step'] != 200
                or runtime['checked_lr_points'] != 400 or runtime['checkpoint']['status'] != 'PASS'
                or runtime['checkpoint']['step'] != 200 or runtime['checkpoint']['samples_consumed'] != 204800
                or runtime['checkpoint']['global_tokens_seen'] != 838860800
                or runtime['checkpoint']['optimizer_groups_checked'] != groups
                or runtime['reference_scheduler']['status'] != 'PASS'
                or runtime['reference_scheduler']['scheduler_epochs_checked'] != 201):
            raise ValueError('actual production LR/checkpoint/reference schedule proof is incomplete')
        name = BASE+'/'+case+'-runtime-review.json'
        runtimes[name] = encoded(runtime)
        inputs[name] = hashlib.sha256(runtimes[name]).hexdigest()
        measurements[case] = dict(parameters=row['parameters'],peak_lr=config['optim']['lr'],
            p10_tps_per_device=evaluated['p10_tps_per_device'],
            projected_1t_days_duty_09=evaluated['projected_1t_days_duty_09'],
            original_primary_macro_loss=result['metrics']['validation_metrics/loss']['last'],
            memory_review=memory,actual_runtime_review=name)
        config = copy.deepcopy(config)
        config['out_dir'] = '<independent-production-control-output>'
        configs[case] = config
    require_matched_contract(configs['larger_lr1e4_b8'],configs['matched_baseline_b8'])
    candidate = health['cases']['larger_lr1e4_b8']
    base = yaml.safe_load((root/'pretrain/configs/pretrain_base.yaml').read_bytes())
    measured_config = yaml.safe_load((root/RUNS/'larger_lr1e4_b8/resolved_config.yaml').read_bytes())
    measured_model = yaml.safe_load(Path(measured_config['model']['configs']).read_bytes())['presets'][measured_config['model']['preset']]
    model = yaml.safe_load((root/'model/configs/pretrain_final_model_configs.yaml').read_bytes())['presets'][base['model']['preset']]
    if measured_model != model or base['optim']['lr'] != .0001 or measured_config['optim']['lr'] != .0001:
        raise ValueError('actual production pair does not confirm the unchanged prepared candidate')
    env_path = root/'pretrain/configs/pretrain_base.env'
    auth_path = root/'data/health/pretrain-user-full-pretrain-authorization.json'
    base, environment, prepared_authorization = original_preparer.production_documents(
        base, original_preparer.parse_env(env_path), json.loads(auth_path.read_bytes()),
        model, measured_config, original_preparer.parse_env(Path(candidate['variant']['env'])),
        model_path=root/'model/configs/pretrain_final_model_configs.yaml', block_manifest=block_manifest)
    stage = effective['stages'][0]
    base['train'].update(budget_tokens=stage['tokens'], expected_iters=stage['updates'])
    base['data']['stage_sample_limits'] = [phase['samples'] for phase in stage['phases']]
    base['optim'].update(warmup_steps=stage['phases'][0]['updates'],decay_steps=stage['phases'][2]['updates'])
    history_paths = [SELECTION,ASSESSMENT,'data/health/pretrain-user-full-pretrain-authorization.json',
        'experiments/e2_architecture/results/rope_source_controls_20261114/disjoint_validation_attempt2/review.json',
        'experiments/e2_architecture/results/rope_source_controls_20261114/lower_lr_controls/quality_review.json',
        'data/health/nearby-lr-development-review.json',
        'data/health/research-vs-production-prefix-exposure.json',
        'data/health/research-exposure-column-check.json']
    history = snapshot_files(root,{name:bench.digest(root/name) for name in history_paths})
    inputs.update(history['dependencies'])
    assessment = dict(schema_version=1,status='REVIEWED_FOR_QUALIFICATION',selected_variant='larger_lr1e4_b8',
        control_health_sha256=bench.digest(root/health_name),measurements=measurements,
        training_source_sha256=current_source,runtime_ceiling_days=33,duty_factor=.9,formal_token_budget=effective['actual_total_tokens'],
        formal_pretrain_tokens=stage['tokens'], global_runtime_forecast=global_forecast,
        user_preference='允许接近 30 天，优先选择更大模型',reviewer='source-bound actual-production distribution review',
        quality_review=str(quality_path.relative_to(root)),prior_research_failures_preserved=True,
        rationale=('The predeclared current-backend pair consumed identical actual production warmup samples '
            'and passed strict all-nine original and expanded development endpoints. The old research prefix '
            'was 92.53% web/1.58% Chinese; actual production warmup is 60.04% web/7.94% Chinese. Old failed '
            'research endpoints remain failed and archived. This confirms the unchanged 1.5166B LR1e-4 '
            'candidate on the actual training distribution after checkpoint durability hardening. '
            'It is bounded early-training development evidence, not an unseen test, a global architecture '
            'optimum or a final1T capability claim. Full current-source GPU/recovery qualification, frozen '
            'production canary and all data/identity/launch gates remain mandatory.'),production_qualified=False)
    inputs[ASSESSMENT] = hashlib.sha256(encoded(assessment)).hexdigest()
    selection = dict(schema_version=1,status='SELECTED_FOR_PRODUCTION_QUALIFICATION',production_qualified=False,
        model=model,parameters=candidate['parameters'],optimizer=base['optim'],attention=base['attention'],
        parallel=base['parallel'],scientific_assessment=assessment,evidence_sha256=inputs,
        execution_contract=training_execution_contract(base),global_training_budget=global_bundle,
        historical_control_inputs=history['manifest'],preparation_tool_sha256=bench.digest(original_preparer.__file__),
        production_confirmation_tool_sha256=bench.digest(__file__),
        production_validation_review=dict(case='larger_lr1e4_b8',plan=str(plan_path.relative_to(root)),report=str(quality_path.relative_to(root))),
        sustained_measurement=dict(steps=200,trained_tokens=838860800,steady_points=150,world_size=8,
            parameters=candidate['parameters'],quality_review_completed=True,
            p10_tps_per_device=candidate['evaluation']['p10_tps_per_device'],
            memory_log=str(Path(candidate['log']).relative_to(root)),run_dir=RUNS+'/larger_lr1e4_b8'))
    require_selection(selection,model,base,json.loads((root/'data/health/model-scale-user-preference.json').read_bytes()))
    verify_sustained_result(root,selection,inputs)
    require_production_quality_review(root,selection)
    if controls.input_identity(health['probe'],health['loaded_source_inputs']) != health['source_qualification_identity']:
        raise ValueError('actual production measurement inputs changed during review')
    auth_name = 'data/health/pretrain-user-full-pretrain-authorization.json'
    authorization = prepared_authorization
    authorization['model_scale_selection'].update(measured_variant='larger_lr1e4_b8',
        measured_run_dir=RUNS+'/larger_lr1e4_b8',quality_assessment=ASSESSMENT,
        measured_parameters=candidate['parameters'],confirmed_training_source_sha256=current_source)
    authorization['training'].update(pretrain_stage_tokens=stage['tokens'],
        total_planned_training_tokens=effective['actual_total_tokens'],
        total_budget_includes_context_extension=True,
        throughput_note=f"Bound actual4K/16K/64K speeds; effective total={effective['actual_total_tokens']} labels; full-chain forecast={global_forecast['total_days']} days at duty=.9, planning ceiling32.8, launch ceiling33. Fresh canonical GPU/recovery, data/order/freeze/canary gates remain mandatory.")
    effective_name = 'experiments/e4_curriculum/results/effective_total_training_budget.json'
    effective = copy.deepcopy(effective)
    effective.update(applied_to_production_configuration=True,
        runtime_budget_derivation={'nominal_budget_file':str(nominal_path.relative_to(root)),
            'nominal_budget_sha256':bench.digest(nominal_path),'execution_plan_file':runtime_name,
            'execution_plan_sha256':bench.digest(runtime_path)},
        allocation_role='PREPARED_EXECUTION_REQUIRES_FINAL_PRODUCTION_GATES')
    inputs[effective_name] = hashlib.sha256(encoded(effective)).hexdigest()
    return {**runtimes,ASSESSMENT:encoded(assessment),SELECTION:encoded(selection),auth_name:encoded(authorization),
        'pretrain/configs/pretrain_base.yaml':yaml.safe_dump(base,sort_keys=False).encode(),
        'pretrain/configs/pretrain_base.env':''.join(f'{k}={v}\n' for k,v in environment.items()).encode(),
        'model/configs/pretrain_final_model_configs.yaml':yaml.safe_dump({'presets':{'pretrain-final':model}},sort_keys=True).encode(),
        effective_name:encoded(effective)}



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--runtime-plan',type=Path,required=True)
    parser.add_argument('--block-manifest',type=Path,required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    with production_mutation_lease(root,'confirm-actual-production-model-selection') if args.publish else nullcontext():
        if any(production_output_dir(root).glob('checkpoint/step-*/.metadata')):
            raise ValueError('formal checkpoints already exist; prelaunch selection refresh forbidden')
        if args.publish:
            bench.require_idle()
        documents = prepare(root,args.runtime_plan.resolve(),args.block_manifest.resolve())
        for name,raw in documents.items():
            path = root/name if args.publish else root/'report/patches/final-deterministic-selection'/name
            path.parent.mkdir(parents=True,exist_ok=True)
            publish_checked(path,raw)
    print(json.dumps(dict(status='CONFIRMED_REQUIRES_FRESH_PRODUCTION_QUALIFICATION' if args.publish else 'VALIDATED_PREVIEW',
                         production_qualified=False,files={n:hashlib.sha256(v).hexdigest() for n,v in documents.items()})))
