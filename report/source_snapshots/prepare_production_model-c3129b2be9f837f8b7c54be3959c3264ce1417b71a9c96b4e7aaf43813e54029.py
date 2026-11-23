#!/usr/bin/env python3
"""Prepare measured model/LR/backend bytes before fresh production qualification."""
import argparse
import copy
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'pretrain/scripts'), str(Path(__file__).resolve().parent)]
from adopt_training_recipe import require_selection, verify_sustained_result, non_embedding_params
from frozen_training_evidence import snapshot_files, production_output_dir, production_mutation_lease


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def production_documents(base, environment, authorization, measured_model, measured_config,
                         measured_environment, *, model_path, block_manifest):
    for key in ('seq_len',):
        if base['data'][key] != measured_config['data'][key]:
            raise ValueError('measured sequence geometry differs from production')
    for key in ('micro_batch_size', 'grad_accum', 'activation_checkpointing', 'bf16', 'cuda_graphs'):
        if base['train'][key] != measured_config['train'][key]:
            raise ValueError('measured execution shape differs from production: '+key)
    if base['parallel'] != measured_config['parallel'] or base['compile'] != measured_config['compile']:
        raise ValueError('measured topology/compiler differs from production')
    result = copy.deepcopy(base)
    result['model'] = {'preset': 'pretrain-final', 'configs': '${YUFORMER_MODEL_CONFIGS}'}
    result['optim']['lr'] = measured_config['optim']['lr']
    for key in set(result['optim']) | set(measured_config['optim']):
        if key not in {'lr_schedule', 'warmup_steps', 'decay_steps', 'min_lr_ratio'}:
            if result['optim'].get(key) != measured_config['optim'].get(key):
                raise ValueError('measured optimizer differs from production: '+key)
    result['attention'] = copy.deepcopy(measured_config['attention'])
    result['data'].update(sampling_policy='block_permutation',
                           block_permutation_manifest='${YUFORMER_TRAINING_BLOCK_MANIFEST}')
    env = dict(environment, YUFORMER_MODEL_CONFIGS=str(model_path),
               YUFORMER_TRAINING_BLOCK_MANIFEST=str(block_manifest))
    for key in ('OMP_NUM_THREADS', 'TORCHINDUCTOR_COMPILE_THREADS', 'TORCHINDUCTOR_CACHE_DIR'):
        if not measured_environment.get(key):
            raise ValueError('measured compiler environment is incomplete: '+key)
        env[key] = measured_environment[key]
    authorized = copy.deepcopy(authorization)
    authorized['architecture'].update(config='model/configs/pretrain_final_model_configs.yaml',
        preset='pretrain-final', non_embedding_params=non_embedding_params(measured_model),
        parameter_count_method='freeze_1t_recipe.non_embedding_params; embeddings excluded')
    authorized['training']['peak_learning_rate'] = measured_config['optim']['lr']
    return result, env, authorized


def parse_env(path):
    return dict(line.split('=', 1) for line in Path(path).read_text().splitlines()
                if line.strip() and not line.lstrip().startswith('#'))


def prepare(root, variant, assessment_path, block_manifest):
    health_path = root/'data/health/model-flex-sustained-controls.json'
    health = json.loads(health_path.read_bytes())
    if health.get('status') != 'COMPLETE_SUSTAINED_CONTROLS':
        raise ValueError('all immutable matched controls must finish before changing production bytes')
    assessment = json.loads(assessment_path.read_bytes())
    if (assessment.get('selected_variant') != variant or assessment.get('status') != 'REVIEWED_FOR_QUALIFICATION'
            or assessment.get('control_health_sha256') != digest(health_path)
            or not isinstance(assessment.get('rationale'), str) or not assessment['rationale'].strip()):
        raise ValueError('scientific assessment is absent or belongs to another measured control')
    for name, expected in health['pinned_inputs'].items():
        if digest(name) != expected:
            raise ValueError('measured control input changed before production preparation: '+name)
    historical_inputs = snapshot_files(root, health['pinned_inputs'])
    selected = health['cases'][variant]
    if selected['evaluation']['status'] != 'PASS_SCREENING' or not selected['evaluation']['within_33_days']:
        raise ValueError('selected sustained model fails the authorized runtime ceiling')
    config = yaml.safe_load(Path(selected['variant']['config']).read_bytes())
    if digest(selected['variant']['config']) != selected['variant']['config_sha256']:
        raise ValueError('selected control YAML changed')
    model_source = Path(config['model']['configs'])
    measured_model = yaml.safe_load(model_source.read_bytes())['presets'][config['model']['preset']]
    base_path, env_path = root/'pretrain/configs/pretrain_base.yaml', root/'pretrain/configs/pretrain_base.env'
    auth_path = root/'data/health/pretrain-user-full-pretrain-authorization.json'
    model_path = root/'model/configs/pretrain_final_model_configs.yaml'
    base, environment, authorization = production_documents(yaml.safe_load(base_path.read_bytes()),
        parse_env(env_path), json.loads(auth_path.read_bytes()), measured_model, config,
        parse_env(selected['variant']['env']), model_path=model_path, block_manifest=block_manifest)
    evidence = {str(health_path.relative_to(root)): digest(health_path),
                str(assessment_path.relative_to(root)): digest(assessment_path)}
    evidence.update(historical_inputs['dependencies'])
    for case in health['cases'].values():
        run = root/'data/experiments/flex_sustained_20261114'/case['variant']['name']
        paths = [run/name for name in ('result.json', 'run_identity.json', 'resolved_config.yaml')]
        paths.append(Path(case['log']))
        paths.append(Path(yaml.safe_load((run/'resolved_config.yaml').read_bytes())['model']['configs']))
        for path in paths:
            evidence[str(path.relative_to(root))] = digest(path)
        if digest(run/'result.json') != case['result_sha256']:
            raise ValueError('completed control result changed before model preparation')
    evaluation = selected['evaluation']
    selection = {'schema_version': 1, 'status': 'SELECTED_FOR_PRODUCTION_QUALIFICATION',
        'production_qualified': False, 'model': measured_model, 'parameters': selected['parameters'],
        'optimizer': base['optim'], 'attention': base['attention'], 'parallel': base['parallel'],
        'sustained_measurement': {'steps': 200, 'trained_tokens': 838860800, 'steady_points': 150,
            'world_size': 8, 'parameters': selected['parameters'], 'quality_review_completed': True,
            'p10_tps_per_device': evaluation['p10_tps_per_device'],
            'memory_log': str(Path(selected['log']).relative_to(root)),
            'run_dir': f'data/experiments/flex_sustained_20261114/{variant}'},
        'scientific_assessment': assessment, 'evidence_sha256': evidence,
        'historical_control_inputs': historical_inputs['manifest'],
        'preparation_tool_sha256': digest(__file__)}
    preference_path = root/'data/health/model-scale-user-preference.json'
    require_selection(selection, measured_model, base, json.loads(preference_path.read_bytes()))
    verify_sustained_result(root, selection, evidence)
    authorization['model_scale_selection'] = {'user_preference_sha256': digest(preference_path),
        'user_reply': '允许接近 30 天，优先选择更大模型', 'measured_variant': variant,
        'measured_parameters': selected['parameters'], 'quality_assessment': str(assessment_path.relative_to(root)),
        'formal_data_numeric_recovery_and_canary_gates_still_required': True}
    authorization['training']['throughput_note'] = (
        f"Selected larger model sustained p10={evaluation['p10_tps_per_device']} tokens/s/GPU; "
        f"1T forecast={evaluation['projected_1t_days_duty_09']} days at duty=.9. Fresh production-data qualification remains required.")
    return {base_path: yaml.safe_dump(base, sort_keys=False),
            env_path: ''.join(f'{key}={value}\n' for key, value in environment.items()),
            auth_path: json.dumps(authorization, sort_keys=True, indent=2)+'\n',
            model_path: yaml.safe_dump({'presets': {'pretrain-final': measured_model}}, sort_keys=True),
            root/'experiments/e2_architecture/results/model-scale-production-selection.json':
                json.dumps(selection, sort_keys=True, indent=2)+'\n'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--assessment', type=Path, required=True)
    parser.add_argument('--block-manifest', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    lease = production_mutation_lease(root, 'prepare-production-model-contract') if args.publish else nullcontext()
    with lease:
        if any(production_output_dir(root).glob('checkpoint/step-*/.metadata')):
            raise ValueError('formal checkpoints already exist; production model replacement is forbidden')
        documents = prepare(root, args.variant, args.assessment.resolve(), args.block_manifest.resolve())
        preview = root/'report/patches/selected-production-contracts'
        for path, content in documents.items():
            output = path if args.publish else preview/path.relative_to(root)
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(output.name+f'.tmp-{os.getpid()}')
            temporary.write_text(content)
            os.replace(temporary, output)
    print(json.dumps({'status': 'PREPARED_REQUIRES_FRESH_QUALIFICATION' if args.publish else 'REVIEWABLE_PREVIEW',
        'production_qualified': False, 'files': {str(p.relative_to(root)): hashlib.sha256(v.encode()).hexdigest()
                                              for p, v in documents.items()}}, sort_keys=True))


if __name__ == '__main__':
    main()
