#!/usr/bin/env python3
"""Refresh actual current-source evidence while preserving prepared model bytes."""
import argparse
from contextlib import nullcontext
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'pretrain/scripts'), str(ROOT/'experiments/e2_architecture/scripts')]
from qualification_identity import qualification_identity
from benchmark_budget_lr_controls import require_matched_contract
from benchmark_model_sizes import evaluate
from adopt_training_recipe import require_selection, verify_sustained_result, source_sha256
from frozen_training_evidence import snapshot_files, publish_checked, production_mutation_lease, production_output_dir
from training_data_origin import read_bound
from sustained_memory_review import review_memory_log

HEALTH = 'data/health/model-window-source-controls.json'
SELECTION = 'experiments/e2_architecture/results/model-scale-production-selection.json'
ASSESSMENT = 'experiments/e2_architecture/results/model-scale-scientific-assessment.json'
CASES = ('matched_baseline_b8', 'larger_lr1e4_b8', 'larger_lr2e4_b8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(document):
    return (json.dumps(document, sort_keys=True, indent=2)+'\n').encode()


def refresh(root):
    root = Path(root).resolve()
    current = qualification_identity(root)
    health = json.loads((root/HEALTH).read_bytes())
    if (health.get('status') != 'COMPLETE_SUSTAINED_CONTROLS'
            or health.get('source_qualification_identity') != current
            or set(health.get('cases', {})) != set(CASES)):
        raise ValueError('all three actual current-source controls must complete before selection refresh')
    protected = {str(p.relative_to(root)): digest(p) for p in (
        root/'pretrain/configs/pretrain_base.yaml', root/'pretrain/configs/pretrain_base.env',
        root/'model/configs/pretrain_final_model_configs.yaml',
        root/'data/health/pretrain-user-full-pretrain-authorization.json')}
    for name, expected in health['pinned_inputs'].items():
        if digest(name) != expected:
            raise ValueError('current control measurement input changed before review: '+name)
    history = snapshot_files(root, {SELECTION:digest(root/SELECTION), ASSESSMENT:digest(root/ASSESSMENT),
                                    **health['pinned_inputs']})
    base = yaml.safe_load((root/'pretrain/configs/pretrain_base.yaml').read_bytes())
    model = yaml.safe_load((root/'model/configs/pretrain_final_model_configs.yaml').read_bytes())['presets'][base['model']['preset']]
    inputs = {HEALTH:digest(root/HEALTH), **history['dependencies']}
    configs, results, measurements = {}, {}, {}
    for name in CASES:
        case = health['cases'][name]
        run = root/'data/experiments/window_source_controls_20261114'/name
        result = json.loads((run/'result.json').read_bytes())
        identity = json.loads((run/'run_identity.json').read_bytes())
        config = yaml.safe_load((run/'resolved_config.yaml').read_bytes())
        actual = evaluate(result, steps=200, skip_steps=50)
        log_path = Path(case['log'])
        raw_log, log_sha = read_bound(log_path)
        memory_review = review_memory_log(raw_log, steps=200, skip_steps=50)
        inputs[str(log_path.relative_to(root))] = log_sha
        if (digest(run/'result.json') != case['result_sha256']
                or identity.get('source_sha256') != source_sha256(root)
                or result.get('identity_sha256') != identity.get('identity_sha256')
                or actual != case['evaluation'] or actual['status'] != 'PASS_SCREENING'
                or case.get('requires_smaller_batch_comparison') is not False):
            raise ValueError('actual current-source result/identity/throughput does not close: '+name)
        for kind in ('config', 'env'):
            path = Path(case['variant'][kind])
            if digest(path) != case['variant'][kind+'_sha256']:
                raise ValueError('actual control execution contract changed: '+name)
            inputs[str(path.relative_to(root))] = digest(path)
        measured_model = Path(identity['model_configs'])
        if digest(measured_model) != identity['model_configs_sha256']:
            raise ValueError('actual measured model changed')
        for path in (run/'result.json', run/'run_identity.json', run/'resolved_config.yaml', measured_model):
            inputs[str(path.relative_to(root))] = digest(path)
        domains = {domain:result['metrics'][f'validation_metrics/domain/{domain}/loss']['last'] for domain in
                   ('web','code','math','chinese','academic','books','encyclopedia','qa','multilingual')}
        if not all(math.isfinite(value) for value in [*domains.values(), result['metrics']['validation_metrics/loss']['last']]):
            raise ValueError('every actual control requires finite aggregate and nine-domain validation')
        measurements[name] = dict(parameters=case['parameters'],
            p10_tps_per_device=actual['p10_tps_per_device'],
            projected_1t_days_duty_09=actual['projected_1t_days_duty_09'],
            validation_loss_step_200=result['metrics']['validation_metrics/loss']['last'],
            domains_step_200=domains, memory_review=memory_review)
        results[name], configs[name] = result, config
    normalized = copy.deepcopy(configs)
    for config in normalized.values():
        config['out_dir'] = '<independent-matched-control-output>'
    require_matched_contract(normalized[CASES[1]], normalized[CASES[2]])
    require_matched_contract(normalized[CASES[1]], normalized[CASES[0]])
    candidate = measurements[CASES[1]]
    if (candidate['projected_1t_days_duty_09'] > 33
            or any(candidate['validation_loss_step_200'] >= measurements[other]['validation_loss_step_200']
                   or any(candidate['domains_step_200'][d] >= measurements[other]['domains_step_200'][d]
                          for d in candidate['domains_step_200']) for other in (CASES[0], CASES[2]))):
        raise ValueError('new measurements require a new scientific decision; prepared candidate was not silently changed')
    selected = copy.deepcopy(json.loads((root/SELECTION).read_bytes()))
    assessment = dict(schema_version=1, status='REVIEWED_FOR_QUALIFICATION',
        selected_variant=CASES[1], control_health_sha256=digest(root/HEALTH),
        user_preference='允许接近 30 天，优先选择更大模型',
        runtime_ceiling_days=33, duty_factor=.9, formal_token_budget=10**12,
        production_qualified=False, reviewer='repository engineering research review',
        measurements=measurements, training_source_sha256=source_sha256(root),
        rationale=(f'Actual fresh current-source 200-update matched controls retain the prepared '
            f'{candidate["parameters"]}-parameter LR1e-4 model: aggregate and all nine domain losses '
            f'improve against both the 0.53B control and the same larger model at LR2e-4. '
            f'Actual conservative sustained p10={candidate["p10_tps_per_device"]} tokens/s/device '
            f'projects {candidate["projected_1t_days_duty_09"]} days for 1T at duty .9, within 33 days. '
            'These are bounded early-training observations. Original 1B–5B frontier measurements '
            'remain historical; this refresh does not claim a global architecture optimum or final '
            '1T quality. Production-window data, numerical, recovery and sustained canary gates remain required.'))
    inputs[ASSESSMENT] = hashlib.sha256(encoded(assessment)).hexdigest()
    inputs[str(Path(__file__).relative_to(root))] = digest(__file__)
    inputs['experiments/e4_curriculum/scripts/sustained_memory_review.py'] = digest(
        root/'experiments/e4_curriculum/scripts/sustained_memory_review.py')
    selected.update(scientific_assessment=assessment, evidence_sha256=inputs,
        historical_control_inputs=history['manifest'],
        measurement_refresh_tool_sha256=digest(__file__), production_qualified=False)
    selected['sustained_measurement'].update(p10_tps_per_device=candidate['p10_tps_per_device'],
        run_dir=f'data/experiments/window_source_controls_20261114/{CASES[1]}',
        memory_log=str(Path(health['cases'][CASES[1]]['log']).relative_to(root)))
    require_selection(selected, model, base, json.loads((root/'data/health/model-scale-user-preference.json').read_bytes()))
    verify_sustained_result(root, selected, inputs)
    if qualification_identity(root) != current or any(digest(root/name) != sha for name,sha in protected.items()):
        raise ValueError('prepared production bytes or current-source measurement identity changed during review')
    return {ASSESSMENT:assessment, SELECTION:selected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    lease = production_mutation_lease(root, 'refresh-current-source-model-selection') if args.publish else nullcontext()
    with lease:
        if any(production_output_dir(root).glob('checkpoint/step-*/.metadata')):
            raise ValueError('formal checkpoints already exist; prelaunch selection refresh is forbidden')
        documents = refresh(root)
        for name, document in documents.items():
            path = root/name if args.publish else root/'report/patches/current-source-selection'/name
            path.parent.mkdir(parents=True, exist_ok=True)
            publish_checked(path, encoded(document))
    print(json.dumps({'status':'REFRESHED_REQUIRES_PRODUCTION_QUALIFICATION' if args.publish else 'VALIDATED_PREVIEW',
                      'sustained_measurement':documents[SELECTION]['sustained_measurement']}, sort_keys=True))


if __name__ == '__main__':
    main()
