"""Bind the lower-LR checkpoint to the same larger development window."""
from pathlib import Path
import copy
import json
import sys

import yaml

import benchmark_model_sizes as bench
from evaluate_disjoint_validation import checkpoint_identity

ROOT=bench.ROOT
BASE=ROOT/'experiments/e2_architecture/results/rope_source_controls_20261114'
CASE='larger_lr8e5_b8'


def prepare():
    state=json.loads((ROOT/'data/health/lower-lr-control.json').read_bytes())
    if state['status']!='COMPLETE_REQUIRES_PRIMARY_AND_LARGER_HELDOUT_REVIEW':
        raise ValueError('the exact lower-LR training run must complete before checkpoint evaluation')
    original=BASE/'disjoint_validation_attempt2/plan.json'
    plan=copy.deepcopy(json.loads(original.read_bytes()))
    protocol=BASE/'lower_lr_controls/validation_protocol.json'
    if not protocol.is_file():
        raise ValueError('lower-LR development validation must be predeclared')
    declared=json.loads(protocol.read_bytes())
    old=json.loads((ROOT/plan['protocol']).read_bytes())
    for key in ('domains','seq_len','world_size','local_batch_size','global_batch_size',
                'source_sample_offset','samples_per_domain','global_batches_per_domain','target_interval',
                'validation_metadata_sha256','uncertainty'):
        if declared[key]!=old[key]:
            raise ValueError('lower-LR review changed the declared development window: '+key)
    source=ROOT/'data/experiments/rope_source_controls_20261114/lower_lr_controls'/CASE
    identity=json.loads((source/'run_identity.json').read_bytes())
    result=json.loads((source/'result.json').read_bytes())
    if (identity['source_sha256']!=declared['training_source_sha256'] or result['status']!='PASS'
            or result['hyperparameters']['trained_steps']!=200):
        raise ValueError('lower-LR actual source/checkpoint budget does not close')
    config=yaml.safe_load((source/'resolved_config.yaml').read_bytes())
    if config['optim']['lr']!=.00008:
        raise ValueError('the additional checkpoint is not the declared 8e-5 control')
    folder=ROOT/'experiments/e2_architecture/configs/rope_source_controls_20261114/lower_lr_controls/validation'
    folder.mkdir(parents=True,exist_ok=True)
    config_path=folder/(CASE+'.yaml')
    output=ROOT/'data/experiments/rope_source_controls_20261114/lower_lr_controls/validation'/CASE
    if config_path.exists() or output.exists():
        raise ValueError('additional validation must preserve all previous attempts')
    config['out_dir']=str(output)
    config['run_name']=CASE+'_larger_development_validation'
    config_path.write_text(yaml.safe_dump(config,sort_keys=False))
    checkpoint=source/'checkpoint/step-200'
    env_path=ROOT/'experiments/e2_architecture/configs/rope_source_controls_20261114/lower_lr_controls'/(CASE+'.env')
    variables=dict(line.split('=',1) for line in env_path.read_text().splitlines()
                   if line.strip() and not line.lstrip().startswith('#'))
    runtime={key:variables[key] for key in ('TORCHINDUCTOR_CACHE_DIR','TORCHINDUCTOR_COMPILE_THREADS',
            'OMP_NUM_THREADS','PYTORCH_CUDA_ALLOC_CONF','CUBLAS_WORKSPACE_CONFIG',
            'NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE') if key in variables}
    if runtime.get('TORCHINDUCTOR_CACHE_DIR')!='/tmp/yuformer-inductor-cudnn-qualified-v3':
        raise ValueError('additional validation must use the actual training compiler cache')
    plan['protocol']=str(protocol.relative_to(ROOT))
    plan['cases'][CASE]=dict(config=str(config_path.relative_to(ROOT)),runtime_environment=runtime,
        checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_stat=checkpoint_identity(checkpoint),
        checkpoint_metadata_sha256=bench.digest(checkpoint/'.metadata'),
        primary_result=str((source/'result.json').relative_to(ROOT)),report=str((output/'scores.json').relative_to(ROOT)))
    for path in (Path(__file__),original,protocol,BASE/'lower_lr_controls/lower_lr_protocol.json',
                 source/'result.json',source/'run_identity.json',source/'resolved_config.yaml',
                 checkpoint/'.metadata',Path(identity['model_configs']),env_path,config_path):
        plan['source_inputs'][str(path.relative_to(ROOT))]=bench.digest(path)
    path=BASE/'lower_lr_controls/validation_plan.json'
    if path.exists():raise ValueError('additional validation plan already exists')
    bench.publish(path,plan)
    return path,plan['cases'][CASE]


if __name__=='__main__':
    path,record=prepare()
    print(json.dumps(dict(plan=str(path),plan_sha256=bench.digest(path),case=CASE,
                         runtime_environment=record['runtime_environment']),sort_keys=True))
