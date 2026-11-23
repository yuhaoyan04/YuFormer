"""Prepare the predeclared development window for an eligible nearby-LR case."""
import argparse
import copy
import json
import math
from pathlib import Path

import yaml

import benchmark_model_sizes as bench
from evaluate_disjoint_validation import checkpoint_identity

ROOT = bench.ROOT
BASE = ROOT/'experiments/e2_architecture/results/rope_source_controls_20261114'


def prepare(case):
    health_path = ROOT/'data/health/nearby-lr-controls.json'
    health = json.loads(health_path.read_bytes())
    if health['status'] != 'COMPLETE_REQUIRES_PRIMARY_AND_LARGER_DEVELOPMENT_REVIEW':
        raise ValueError('both fixed-order nearby controls must complete before development evaluation')
    grid_protocol_path = BASE/'nearby_lr_controls/protocol.json'
    grid = json.loads(grid_protocol_path.read_bytes())
    rows = {row['case']:row for row in grid['controls']}
    if case not in rows or case not in health['cases']:
        raise ValueError('case was not in the predeclared nearby-LR study')
    old_plan_path = BASE/'disjoint_validation_attempt2/plan.json'
    plan = copy.deepcopy(json.loads(old_plan_path.read_bytes()))
    old_protocol = json.loads((ROOT/plan['protocol']).read_bytes())
    if any(old_protocol[key] != value for key, value in grid['expanded_window'].items()):
        raise ValueError('the predeclared development window changed')
    source = ROOT/'data/experiments/rope_source_controls_20261114/nearby_lr_controls'/case
    identity = json.loads((source/'run_identity.json').read_bytes())
    result = json.loads((source/'result.json').read_bytes())
    config = yaml.safe_load((source/'resolved_config.yaml').read_bytes())
    if (result['status'] != 'PASS' or result['hyperparameters']['trained_steps'] != 200
            or identity['source_sha256'] != grid['training_source_sha256']
            or config['optim']['lr'] != rows[case]['peak_lr']
            or bench.digest(source/'result.json') != health['cases'][case]['result_sha256']):
        raise ValueError('actual nearby-LR checkpoint differs from its declared training contract')
    baseline = json.loads((ROOT/plan['cases']['matched_baseline_b8']['primary_result']).read_bytes())
    pairs = [(result['metrics'][f'validation_metrics/domain/{d}/loss']['last'],
              baseline['metrics'][f'validation_metrics/domain/{d}/loss']['last']) for d in grid['expanded_window']['domains']]
    if any(not math.isfinite(a) or not math.isfinite(b) or a >= b for a, b in pairs):
        raise ValueError('candidate failed original all-nine primary endpoint; preserve the failure')
    folder = BASE/'nearby_lr_controls/validation'/case
    config_folder = ROOT/'experiments/e2_architecture/configs/rope_source_controls_20261114/nearby_lr_controls/validation'
    output = ROOT/'data/experiments/rope_source_controls_20261114/nearby_lr_controls/validation'/case
    config_path = config_folder/(case+'.yaml')
    if folder.exists() or config_path.exists() or output.exists():
        raise ValueError('nearby-LR validation must preserve all earlier attempts')
    folder.mkdir(parents=True)
    config_folder.mkdir(parents=True, exist_ok=True)
    protocol = copy.deepcopy(old_protocol)
    protocol.update(cases=[*old_protocol['cases'], case], target_candidate=case,
                    reference_cases=['matched_baseline_b8', 'larger_lr2e4_b8'],
                    purpose='PREDECLARED_NEARBY_LR_DEVELOPMENT_WINDOW_NOT_UNSEEN_TEST',
                    original_primary_64_all_nine_improvement_required=True,
                    nearby_training_protocol_sha256=bench.digest(grid_protocol_path))
    protocol_path = folder/'protocol.json'
    bench.publish(protocol_path, protocol)
    config.update(out_dir=str(output), run_name=case+'_larger_development_validation')
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    env_path = ROOT/'experiments/e2_architecture/configs/rope_source_controls_20261114/nearby_lr_controls'/(case+'.env')
    variables = dict(line.split('=', 1) for line in env_path.read_text().splitlines()
                     if line.strip() and not line.lstrip().startswith('#'))
    runtime = {key:variables[key] for key in ('TORCHINDUCTOR_CACHE_DIR','TORCHINDUCTOR_COMPILE_THREADS',
               'OMP_NUM_THREADS','PYTORCH_CUDA_ALLOC_CONF','CUBLAS_WORKSPACE_CONFIG',
               'NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE') if key in variables}
    if runtime.get('TORCHINDUCTOR_CACHE_DIR') != '/tmp/yuformer-inductor-cudnn-qualified-v3':
        raise ValueError('nearby-LR validation must inherit its actual original numeric environment')
    checkpoint = source/'checkpoint/step-200'
    plan['protocol'] = str(protocol_path.relative_to(ROOT))
    plan['cases'][case] = dict(config=str(config_path.relative_to(ROOT)), runtime_environment=runtime,
        checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_stat=checkpoint_identity(checkpoint),
        checkpoint_metadata_sha256=bench.digest(checkpoint/'.metadata'),
        primary_result=str((source/'result.json').relative_to(ROOT)), report=str((output/'scores.json').relative_to(ROOT)))
    for path in (Path(__file__), grid_protocol_path, old_plan_path, protocol_path, health_path,
                 source/'result.json', source/'run_identity.json', source/'resolved_config.yaml',
                 checkpoint/'.metadata', Path(identity['model_configs']), env_path, config_path):
        plan['source_inputs'][str(path.relative_to(ROOT))] = bench.digest(path)
    plan_path = folder/'plan.json'
    bench.publish(plan_path, plan)
    return plan_path, plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    args = parser.parse_args()
    path, plan = prepare(args.case)
    print(json.dumps(dict(plan=str(path), plan_sha256=bench.digest(path))))
