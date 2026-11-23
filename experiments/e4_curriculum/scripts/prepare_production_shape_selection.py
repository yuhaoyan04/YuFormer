"""Prepare production bytes only after the independently reviewed fixed grid."""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'experiments/e2_architecture/scripts'),str(ROOT/'pretrain/scripts')]
import benchmark_model_sizes as bench
import run_production_shape_lr_controls as controls
from production_shape_quality_review import review
from shape_selection_evidence import require_shape_quality_review
from adopt_training_recipe import require_selection, verify_sustained_result, source_sha256
from checkpoint_durability_evidence import require_checkpoint_directory_durability
from frozen_training_evidence import snapshot_files, publish_checked, production_mutation_lease, production_output_dir
import prepare_production_model as original

SELECTION = 'experiments/e2_architecture/results/model-scale-production-selection.json'
ASSESSMENT = 'experiments/e2_architecture/results/model-scale-scientific-assessment.json'
AUTHORIZATION = 'data/health/pretrain-user-full-pretrain-authorization.json'


def encoded(value):
    return (json.dumps(value,sort_keys=True,indent=2)+'\n').encode()


def prepare(root):
    root = Path(root).resolve()
    if root != controls.ROOT:
        raise ValueError('production grid evidence is bound to this actual repository')
    protocol = json.loads(controls.PROTOCOL.read_bytes())
    controls.require_inputs(protocol)
    independent = review(root,controls.PROTOCOL)
    if independent['status'] != 'PASS_INDEPENDENT_PRODUCTION_SHAPE_GRID':
        raise ValueError('no actual strictly passing shape is available')
    name = independent['preferred_case']
    state = json.loads((controls.RESULTS/'review.json').read_bytes())
    selected = state['cases'][name]
    declaration = next(row for row in protocol['cases'] if row['name']==name)
    config = yaml.safe_load((controls.RUNS/name/'resolved_config.yaml').read_bytes())
    model = yaml.safe_load(Path(declaration['model_path']).read_bytes())['presets'][config['model']['preset']]
    report_path = controls.RESULTS/'independent-selection-review.json'
    if not report_path.is_file() or json.loads(report_path.read_bytes()) != independent:
        raise ValueError('publish the independently reproduced shape review receipt before production preview')
    evidence = dict(independent['source_inputs'])
    evidence[str(report_path.relative_to(root))] = bench.digest(report_path)
    evidence.update(require_checkpoint_directory_durability(root))
    for path in (Path(__file__),Path(original.__file__),
                 root/'experiments/e4_curriculum/scripts/shape_selection_evidence.py',
                 root/'data/health/production-shape-lr-contract-preview.json'):
        evidence[str(path.relative_to(root))] = bench.digest(path)
    # The production model/base will change. Preserve their original qualification
    # identity as immutable history rather than requiring those old live bytes.
    historical = {row['path']:row['sha256'] for row in protocol['qualification_identity']['files']}
    for path in (root/SELECTION,root/ASSESSMENT,root/AUTHORIZATION,
                 root/'experiments/e2_architecture/results/production_matched_controls_20261115/validation/review.json',
                 root/'data/health/production-matched-controls.json',
                 root/'data/health/nearby-lr-development-review.json'):
        historical[str(path.relative_to(root))] = bench.digest(path)
    archive = snapshot_files(root,historical)
    evidence.update(archive['dependencies'])
    old_base = yaml.safe_load((root/'pretrain/configs/pretrain_base.yaml').read_bytes())
    old_env = original.parse_env(root/'pretrain/configs/pretrain_base.env')
    old_auth = json.loads((root/AUTHORIZATION).read_bytes())
    model_path = root/'model/configs/pretrain_final_model_configs.yaml'
    manifest = Path(old_env['YUFORMER_TRAINING_BLOCK_MANIFEST'])
    base,env,authorization = original.production_documents(old_base,old_env,old_auth,model,config,
        original.parse_env(declaration['variant']['env']),model_path=model_path,block_manifest=manifest)
    if base['train'] != old_base['train'] or base['data'] != old_base['data']:
        raise ValueError('shape adoption must preserve the complete actual data order and formal token budget')
    assessment = dict(schema_version=1,status='REVIEWED_FOR_QUALIFICATION',selected_variant=name,
        training_source_sha256=source_sha256(root),runtime_ceiling_days=33,duty_factor=.9,
        formal_token_budget=10**12,user_preference='允许接近 30 天，优先选择更大模型',
        quality_review=str(report_path.relative_to(root)),measurements=independent['cases'],
        prior_research_failures_preserved=True,
        rationale=('The completed predeclared four-case grid consumed identical actual production warmup '
                   'samples with two existing1B+ architectures and two fixed learning rates. Independent '
                   'review reproduces every strict primary and expanded endpoint and selects the largest '
                   'strictly passing model. All previous1.52B Chinese failures remain failed. This is '
                   'bounded early-training development evidence, not an unseen test or global optimum. '
                   'Changed production model/LR requires fresh complete GPU, actual-model recovery, '
                   'gradient/component parity, training-only ADOPT and final frozen production canary.'),
        production_qualified=False)
    evidence[ASSESSMENT] = hashlib.sha256(encoded(assessment)).hexdigest()
    evaluation = selected['evaluation']
    selection = dict(schema_version=1,status='SELECTED_FOR_PRODUCTION_QUALIFICATION',production_qualified=False,
        model=model,parameters=selected['parameters'],optimizer=base['optim'],attention=base['attention'],
        parallel=base['parallel'],scientific_assessment=assessment,evidence_sha256=evidence,
        historical_control_inputs=archive['manifest'],preparation_tool_sha256=bench.digest(original.__file__),
        shape_confirmation_tool_sha256=bench.digest(__file__),
        production_shape_validation_review=dict(case=name,protocol=str(controls.PROTOCOL.relative_to(root)),
                                               report=str(report_path.relative_to(root))),
        sustained_measurement=dict(steps=200,trained_tokens=838860800,steady_points=150,world_size=8,
            parameters=selected['parameters'],quality_review_completed=True,
            p10_tps_per_device=evaluation['p10_tps_per_device'],
            memory_log=str(Path(selected['log']).relative_to(root)),
            run_dir=str((controls.RUNS/name).relative_to(root))))
    preference = root/'data/health/model-scale-user-preference.json'
    require_selection(selection,model,base,json.loads(preference.read_bytes()))
    verify_sustained_result(root,selection,evidence)
    require_shape_quality_review(root,selection)
    authorization['model_scale_selection'].update(measured_variant=name,
        measured_run_dir=str((controls.RUNS/name).relative_to(root)),
        measured_parameters=selected['parameters'],quality_assessment=ASSESSMENT,
        user_preference_sha256=bench.digest(preference),confirmed_training_source_sha256=source_sha256(root))
    authorization['training']['throughput_note'] = (
        f"Actual production grid p10={evaluation['p10_tps_per_device']} tokens/s/GPU; "
        f"1T forecast={evaluation['projected_1t_days_duty_09']} days at duty=.9. "
        'Fresh actual-model GPU/recovery/parity and frozen production canary remain mandatory.')
    controls.require_inputs(protocol)
    return {ASSESSMENT:encoded(assessment),SELECTION:encoded(selection),AUTHORIZATION:encoded(authorization),
        'pretrain/configs/pretrain_base.yaml':yaml.safe_dump(base,sort_keys=False).encode(),
        'pretrain/configs/pretrain_base.env':''.join(f'{k}={v}\n' for k,v in env.items()).encode(),
        'model/configs/pretrain_final_model_configs.yaml':yaml.safe_dump({'presets':{'pretrain-final':model}},sort_keys=True).encode()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish',action='store_true')
    args = parser.parse_args()
    with production_mutation_lease(ROOT,'prepare-qualified-production-shape') if args.publish else nullcontext():
        if any(production_output_dir(ROOT).glob('checkpoint/step-*/.metadata')):
            raise ValueError('formal checkpoints already exist; prelaunch model replacement is forbidden')
        if args.publish:
            bench.require_idle()
        documents = prepare(ROOT)
        for name,raw in documents.items():
            path = ROOT/name if args.publish else ROOT/'report/patches/production-shape-selection'/name
            path.parent.mkdir(parents=True,exist_ok=True)
            publish_checked(path,raw)
    print(json.dumps(dict(status='PREPARED_REQUIRES_FRESH_QUALIFICATION' if args.publish else 'VALIDATED_PREVIEW',
                         production_qualified=False,files={n:hashlib.sha256(v).hexdigest() for n,v in documents.items()})))
