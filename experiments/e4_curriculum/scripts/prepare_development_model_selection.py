"""Prepare a strict, reproducible selection from the completed nearby-LR study."""
import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'experiments/e2_architecture/scripts'), str(ROOT/'pretrain/scripts')]
import benchmark_model_sizes as bench
from benchmark_budget_lr_controls import require_matched_contract
from development_validation_review import review
from research_control_identity import research_control_identity
from adopt_training_recipe import require_selection, verify_sustained_result, source_sha256
from allocator_retry_review import require_no_allocator_retries
from audit_training_progress import audit, verify_reference_schedule
from sustained_memory_review import review_memory_log
import prepare_production_model as original_preparer
from frozen_training_evidence import snapshot_files, publish_checked, production_mutation_lease, production_output_dir

BASE = 'experiments/e2_architecture/results/rope_source_controls_20261114'
GRID = BASE+'/nearby_lr_controls'
SELECTION = 'experiments/e2_architecture/results/model-scale-production-selection.json'
ASSESSMENT = 'experiments/e2_architecture/results/model-scale-scientific-assessment.json'


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2)+'\n').encode()


def prepare(root):
    root = Path(root).resolve()
    health_name = 'data/health/nearby-lr-controls.json'
    review_name = 'data/health/nearby-lr-development-review.json'
    health = json.loads((root/health_name).read_bytes())
    completed_review = json.loads((root/review_name).read_bytes())
    if (health.get('status') != 'COMPLETE_REQUIRES_PRIMARY_AND_LARGER_DEVELOPMENT_REVIEW'
            or completed_review.get('status') != 'PASS_REQUIRES_SCIENTIFIC_SELECTION_AND_PRODUCTION_QUALIFICATION'):
        raise ValueError('actual completed strict primary and expanded quality reviews are required')
    config_dir = root/'experiments/e2_architecture/configs/rope_source_controls_20261114/nearby_lr_controls'
    current = research_control_identity(root, config_dir)
    if current != health['source_qualification_identity']:
        raise ValueError('actual nearby controls no longer match the current prepared inputs')
    for path, expected in {**health['loaded_source_inputs'], **completed_review['loaded_source_inputs']}.items():
        if bench.digest(path) != expected:
            raise ValueError('actual development producer changed: '+path)
    protocol = json.loads((root/GRID/'protocol.json').read_bytes())
    expected_cases = {row['case'] for row in protocol['controls']}
    if set(health['cases']) != expected_cases or set(completed_review['cases']) != expected_cases:
        raise ValueError('fixed nearby-LR roster is incomplete')
    inputs = {health_name:bench.digest(root/health_name), review_name:bench.digest(root/review_name),
              GRID+'/protocol.json':bench.digest(root/GRID/'protocol.json')}
    actual_reviews = {}
    for case in protocol['controls']:
        name = case['case']
        row = completed_review['cases'][name]
        if row['status'] == 'FAIL_ORIGINAL_ALL_NINE_PRIMARY':
            continue
        path = Path(row['report'])
        actual = review(root, path.with_name('plan.json'), name, root/BASE/'disjoint_validation_attempt2/plan.json')
        stored = json.loads(path.read_bytes())
        if bench.digest(path) != row['report_sha256'] or actual != stored:
            raise ValueError('stored development review differs from independently recomputed evidence')
        inputs.update(actual['source_inputs'])
        inputs[str(path.relative_to(root))] = bench.digest(path)
        if actual['status'] == 'PASS_PRIMARY_AND_EXPANDED_ALL_NINE':
            actual_reviews[name] = actual
    if not actual_reviews:
        raise ValueError('no candidate actually passed both unchanged all-nine endpoints')
    selected_case = min(actual_reviews, key=lambda name:actual_reviews[name]['candidate_aggregate_mean_loss'])
    if selected_case != completed_review.get('best_qualifying_candidate'):
        raise ValueError('selected candidate differs from the declared minimum development loss rule')
    summaries = {}
    configs = {}
    sources = [(name, 'data/experiments/rope_source_controls_20261114/'+name,
                json.loads((root/'data/health/model-rope-source-controls.json').read_bytes())['cases'][name])
               for name in ('matched_baseline_b8', 'larger_lr1e4_b8', 'larger_lr2e4_b8')]
    sources += [('larger_lr8e5_b8', 'data/experiments/rope_source_controls_20261114/lower_lr_controls/larger_lr8e5_b8',
                 json.loads((root/'data/health/lower-lr-control.json').read_bytes())['cases']['larger_lr8e5_b8'])]
    sources += [(name, 'data/experiments/rope_source_controls_20261114/nearby_lr_controls/'+name, health['cases'][name])
                for name in sorted(expected_cases)]
    for name, directory, measured in sources:
        run = root/directory
        result = json.loads((run/'result.json').read_bytes())
        identity = json.loads((run/'run_identity.json').read_bytes())
        config = yaml.safe_load((run/'resolved_config.yaml').read_bytes())
        evaluated = bench.evaluate(result, steps=200, skip_steps=50)
        if (bench.digest(run/'result.json') != measured['result_sha256']
                or identity['source_sha256'] != source_sha256(root)
                or result['identity_sha256'] != identity['identity_sha256']
                or evaluated != measured['evaluation'] or evaluated['status'] != 'PASS_SCREENING'):
            raise ValueError('actual training identity/accounting does not close: '+name)
        log = Path(measured['log'])
        raw = log.read_bytes()
        memory = review_memory_log(raw, steps=200, skip_steps=50)
        require_no_allocator_retries(raw)
        model_path = Path(identity['model_configs'])
        if bench.digest(model_path) != identity['model_configs_sha256']:
            raise ValueError('actual measured model changed')
        paths = [run/'result.json', run/'run_identity.json', run/'resolved_config.yaml', model_path, log]
        for key in ('config', 'env'):
            path = Path(measured['variant'][key])
            if bench.digest(path) != measured['variant'][key+'_sha256']:
                raise ValueError('actual training execution contract changed')
            paths.append(path)
        inputs.update({str(path.relative_to(root)):bench.digest(path) for path in paths})
        summaries[name] = dict(parameters=measured['parameters'], peak_lr=config['optim']['lr'],
            p10_tps_per_device=evaluated['p10_tps_per_device'],
            projected_1t_days_duty_09=evaluated['projected_1t_days_duty_09'],
            validation_loss_step_200=result['metrics']['validation_metrics/loss']['last'], memory_review=memory,
            domains_step_200={d:result['metrics'][f'validation_metrics/domain/{d}/loss']['last']
                              for d in protocol['expanded_window']['domains']})
        config = copy.deepcopy(config)
        config['out_dir'] = '<independent-matched-control-output>'
        configs[name] = config
    for config in configs.values():
        require_matched_contract(configs[selected_case], config)
    selected_measurement = health['cases'][selected_case]
    measured_config = yaml.safe_load(Path(selected_measurement['variant']['config']).read_bytes())
    run = root/'data/experiments/rope_source_controls_20261114/nearby_lr_controls'/selected_case
    runtime = audit(run, check_checkpoint=True)
    runtime['reference_scheduler'] = verify_reference_schedule(yaml.safe_load((run/'resolved_config.yaml').read_bytes()))
    if (runtime['status'] != 'PASS_OBSERVED_PREFIX' or runtime['latest_logged_step'] != 200
            or runtime['checked_lr_points'] != 400 or runtime['checkpoint']['status'] != 'PASS'
            or runtime['checkpoint']['step'] != 200 or runtime['checkpoint']['samples_consumed'] != 204800
            or runtime['checkpoint']['global_tokens_seen'] != 838860800
            or runtime['checkpoint']['optimizer_groups_checked'] != 164
            or runtime['reference_scheduler']['status'] != 'PASS'
            or runtime['reference_scheduler']['scheduler_epochs_checked'] != 201):
        raise ValueError('selected actual LR/checkpoint/reference runtime proof is incomplete')
    runtime_name = GRID+'/selected-runtime-review.json'
    inputs[runtime_name] = hashlib.sha256(encoded(runtime)).hexdigest()
    inputs['pretrain/scripts/audit_training_progress.py'] = bench.digest(root/'pretrain/scripts/audit_training_progress.py')
    measured_model = yaml.safe_load(Path(measured_config['model']['configs']).read_bytes())['presets'][measured_config['model']['preset']]
    base_path, env_path = root/'pretrain/configs/pretrain_base.yaml', root/'pretrain/configs/pretrain_base.env'
    auth_path = root/'data/health/pretrain-user-full-pretrain-authorization.json'
    model_path = root/'model/configs/pretrain_final_model_configs.yaml'
    environment = original_preparer.parse_env(env_path)
    base, env, authorization = original_preparer.production_documents(yaml.safe_load(base_path.read_bytes()),
        environment, json.loads(auth_path.read_bytes()), measured_model, measured_config,
        original_preparer.parse_env(selected_measurement['variant']['env']), model_path=model_path,
        block_manifest=Path(environment['YUFORMER_TRAINING_BLOCK_MANIFEST']))
    if yaml.safe_load(model_path.read_bytes())['presets'][base['model']['preset']] != measured_model:
        raise ValueError('this nearby-LR selection may not silently replace the prepared architecture')
    history = snapshot_files(root, {name:bench.digest(root/name) for name in
        (SELECTION, ASSESSMENT, 'pretrain/configs/pretrain_base.yaml', 'pretrain/configs/pretrain_base.env',
         'model/configs/pretrain_final_model_configs.yaml', 'data/health/pretrain-user-full-pretrain-authorization.json')})
    inputs.update(history['dependencies'])
    for relative in ('experiments/e4_curriculum/scripts/prepare_development_model_selection.py',
                     'experiments/e4_curriculum/scripts/prepare_production_model.py',
                     'experiments/e4_curriculum/scripts/sustained_memory_review.py',
                     'experiments/e4_curriculum/scripts/allocator_retry_review.py',
                     BASE+'/disjoint_validation_attempt2/review.json',
                     BASE+'/lower_lr_controls/quality_review.json'):
        inputs[relative] = bench.digest(root/relative)
    assessment = dict(schema_version=1, status='REVIEWED_FOR_QUALIFICATION', selected_variant=selected_case,
        control_health_sha256=bench.digest(root/health_name), user_preference='允许接近 30 天，优先选择更大模型',
        runtime_ceiling_days=33, duty_factor=.9, formal_token_budget=10**12, production_qualified=False,
        reviewer='source-bound repository development review', training_source_sha256=source_sha256(root),
        measurements=summaries, development_validation_review=completed_review['cases'][selected_case]['report'],
        prior_failed_quality_endpoints_preserved=True,
        rationale=('The fixed-order development LR search passed both original all-nine and expanded all-nine '
                   'zero-margin endpoints with the same current-source 200-update checkpoints. All prior failed '
                   'LR1e-4/LR8e-5 measurements are retained. Selection uses minimum expanded macro loss among '
                   'qualifying nearby-LR controls. This is bounded early-training development evidence, not an '
                   'unseen test, a global architecture optimum, or proof of final1T quality. Full current GPU '
                   'qualification, physical production-window canary and actual fault recovery remain required.'))
    inputs[ASSESSMENT] = hashlib.sha256(encoded(assessment)).hexdigest()
    selection = dict(schema_version=1, status='SELECTED_FOR_PRODUCTION_QUALIFICATION', production_qualified=False,
        model=measured_model, parameters=selected_measurement['parameters'], optimizer=base['optim'],
        attention=base['attention'], parallel=base['parallel'], scientific_assessment=assessment,
        development_validation_review=dict(case=selected_case,
            candidate_plan=str((root/GRID/'validation'/selected_case/'plan.json').relative_to(root)),
            reference_plan=BASE+'/disjoint_validation_attempt2/plan.json',
            report=str(Path(completed_review['cases'][selected_case]['report']).relative_to(root))),
        evidence_sha256=inputs, historical_control_inputs=history['manifest'],
        preparation_tool_sha256=bench.digest(original_preparer.__file__),
        development_selection_tool_sha256=bench.digest(__file__),
        sustained_measurement=dict(steps=200, trained_tokens=838860800, steady_points=150, world_size=8,
            parameters=selected_measurement['parameters'], quality_review_completed=True,
            p10_tps_per_device=selected_measurement['evaluation']['p10_tps_per_device'],
            memory_log=str(Path(selected_measurement['log']).relative_to(root)),
            run_dir='data/experiments/rope_source_controls_20261114/nearby_lr_controls/'+selected_case))
    require_selection(selection, measured_model, base,
                      json.loads((root/'data/health/model-scale-user-preference.json').read_bytes()))
    verify_sustained_result(root, selection, inputs)
    authorization['model_scale_selection'].update(measured_variant=selected_case,
        quality_assessment=ASSESSMENT, measured_parameters=selected_measurement['parameters'])
    authorization['training']['throughput_note'] = (
        f"Current-source p10={selected_measurement['evaluation']['p10_tps_per_device']} tokens/s/GPU; "
        f"1T forecast={selected_measurement['evaluation']['projected_1t_days_duty_09']} days at duty=.9; "
        'production data, numerical, recovery and canary gates remain required.')
    if research_control_identity(root, config_dir) != current:
        raise ValueError('actual research inputs changed during scientific preparation')
    return {runtime_name:encoded(runtime), ASSESSMENT:encoded(assessment), SELECTION:encoded(selection),
            'pretrain/configs/pretrain_base.yaml':yaml.safe_dump(base, sort_keys=False).encode(),
            'pretrain/configs/pretrain_base.env':''.join(f'{k}={v}\n' for k, v in env.items()).encode(),
            'data/health/pretrain-user-full-pretrain-authorization.json':encoded(authorization)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    with production_mutation_lease(root, 'prepare-development-selection') if args.publish else nullcontext():
        if any(production_output_dir(root).glob('checkpoint/step-*/.metadata')):
            raise ValueError('formal checkpoints already exist; prelaunch selection changes are forbidden')
        if args.publish:
            bench.require_idle()
        documents = prepare(root)
        for name, raw in documents.items():
            path = root/name if args.publish else root/'report/patches/development-production-selection'/name
            publish_checked(path, raw)
    print(json.dumps(dict(status='PREPARED_REQUIRES_FRESH_QUALIFICATION' if args.publish else 'VALIDATED_PREVIEW',
                          production_qualified=False,
                          files={name:hashlib.sha256(raw).hexdigest() for name, raw in documents.items()})))
