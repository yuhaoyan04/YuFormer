#!/usr/bin/env python3
"""Render a production context stage only from completed, admitted parent assets."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import yaml

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'midtrain/src'),str(ROOT/'pretrain/scripts')]
from yuformer_midtrain.registry import validate_midtrain_contract
from yuformer_midtrain.preflight import seal_bytes
from qualification_identity import _parse_env


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(root,*,plan_file,plan_sha256,seq_len,handoff_file,window_release,output,run_dir,evaluation_admission=None):
    root=Path(root).resolve();plan_file=Path(plan_file).resolve()
    if digest(plan_file)!=plan_sha256:raise ValueError('long-stage hyperparameter plan changed after pretrain launch')
    plan=yaml.safe_load(plan_file.read_bytes())
    if plan.get('scope')!='PRODUCTION' or plan.get('execution_order')!=[4096,16384,65536]:
        raise ValueError('only the frozen production4K→16K→64K ladder may be rendered')
    selected=[s for s in plan['stages'] if s['seq_len']==seq_len]
    if len(selected)!=1:raise ValueError('stage is absent or duplicated in the frozen plan')
    stage=selected[0];budget_binding=plan['approved_effective_budget'];frozen_binding=plan['pretrain_frozen_recipe']
    budget_file=root/budget_binding['file'];frozen_file=root/frozen_binding['file']
    if digest(budget_file)!=budget_binding['sha256'] or digest(frozen_file)!=frozen_binding['sha256']:
        raise ValueError('stage allocation or qualified pretrain recipe changed')
    model_file=root/stage['qualified_model']['file']
    if digest(model_file)!=stage['qualified_model']['sha256']:raise ValueError('qualified long-stage model bytes changed')
    release_file=Path(window_release).resolve();release=json.loads(release_file.read_bytes())
    if (release.get('scope')!='PRODUCTION' or release.get('seq_len')!=seq_len
            or release.get('target_labels')!=stage['target_labels']):
        raise ValueError('production stage must bind its exact complete long-window data release')
    handoff_file=Path(handoff_file).resolve();handoff=json.loads(handoff_file.read_bytes())
    checkpoint=Path(handoff['checkpoint']);identity=checkpoint.parent.parent/'run_identity.json'
    step=int(checkpoint.name.removeprefix('step-'))
    parent={'path':str(checkpoint),'step':step,'metadata_sha256':handoff['metadata_sha256'],
        'run_identity':str(identity),'run_identity_sha256':digest(identity),
        'handoff_file':str(handoff_file),'handoff_sha256':digest(handoff_file)}
    if evaluation_admission is not None:
        admission=Path(evaluation_admission).resolve()
        parent['evaluation_admission']={'file':str(admission),'sha256':digest(admission)}
    env=_parse_env(root/'pretrain/configs/pretrain_base.env')
    rendered=(root/'pretrain/configs/pretrain_base.yaml').read_text()
    for key,value in env.items():rendered=rendered.replace('${'+key+'}',value)
    if '${' in rendered:raise ValueError('unresolved base asset dependency')
    cfg=yaml.safe_load(rendered)
    name='pretrain-final-context-'+str(seq_len)
    cfg.update(run_name=name,seed=stage['seed'],out_dir=str(Path(run_dir).resolve()),experiment_status='production_context_stage')
    cfg.pop('research_compute',None)
    cfg['model']={'preset':stage['qualified_model']['preset'],'configs':str(model_file)}
    cfg['data'].update(seq_len=seq_len,stages=[],meta=[],stage_sample_limits=[],epochs=1,
        val_samples_per_domain=stage['validation_samples_per_domain'])
    for key in ('sampling_policy','block_permutation_manifest'):cfg['data'].pop(key,None)
    cfg['train'].update(budget_tokens=stage['target_labels'],expected_iters=stage['updates'],
        micro_batch_size=stage['micro_batch_size'],grad_accum=stage['grad_accum'],
        save_every=stage['checkpoint_every_updates'],eval_every=stage['validation_every_updates'],
        log_every=stage['log_every_updates'],keep_last_n=stage['keep_last_n'],keep_every=stage['keep_every_updates'],
        activation_checkpointing=stage['activation_checkpointing'],bf16=stage['bf16'],
        cuda_graphs=stage['cuda_graphs'],deterministic=stage['deterministic'])
    cfg['optim']={'optimizer':stage['optimizer'],'lr':stage['peak_learning_rate'],'lr_schedule':stage['lr_schedule'],
        'warmup_steps':stage['warmup_steps'],'min_lr_ratio':stage['min_lr_ratio'],'adam_betas':stage['adam_betas'],
        'adam_eps':stage['adam_eps'],'weight_decay':stage['weight_decay'],'clip_grad':stage['clip_grad']}
    cfg['parallel']={'world_size':8,'sharding':stage['sharding'],'replicate_degree':stage['replicate_degree'],
        'shard_degree':stage['shard_degree'],'fsdp_reshard_after_forward':stage['fsdp_reshard_after_forward']}
    cfg['attention']={'sdpa_backend':stage['attention']}
    cfg['compile']={'enable':stage['compile_model'],'components':['model'],'backend':'inductor'}
    cfg['loss']={'num_chunks':stage['loss_num_chunks']}
    cfg['midtrain']={'scope':'PRODUCTION','approved_budget_file':str(budget_file),
        'approved_budget_sha256':budget_binding['sha256'],'window_release':str(release_file),
        'window_release_sha256':digest(release_file),'source_root':str(root),
        'initial_checkpoint':parent,'optimizer_transition':'reset','initial_load_model_only':True,
        'selective_recompute':stage['selective_recompute'],
        'evaluation':{'long_context':True,'short_regression':True},
        'production_stage_plan':{'file':str(plan_file),'sha256':plan_sha256}}
    allocator=stage['cuda_allocator_conf']
    if allocator is not None:cfg['midtrain']['cuda_allocator_conf']=allocator
    old_allocator=os.environ.get('PYTORCH_CUDA_ALLOC_CONF')
    try:
        if allocator is not None:os.environ['PYTORCH_CUDA_ALLOC_CONF']=allocator
        # Real current-byte validation, native parent step/cursor/scheduler,
        # whole-budget coverage, and required prior-stage evaluations all run
        # BEFORE any output configuration is published.
        validate_midtrain_contract(cfg)
    finally:
        if old_allocator is None:os.environ.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        else:os.environ['PYTORCH_CUDA_ALLOC_CONF']=old_allocator
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    cp=output/(name+'.yaml');ep=cp.with_suffix('.env')
    env.update(YUFORMER_CONFIG=str(cp),YUFORMER_MODEL_CONFIGS=str(model_file),YUFORMER_PRETRAIN_OUT=cfg['out_dir'],
        YUFORMER_REPO_ROOT=str(root),YUFORMER_SHARED_BACKEND_ROOT=str(root/'pretrain/backends/torchtitan'),
        YUFORMER_BUDGET_CONTRACT_ROOT=str(root/'experiments/e4_curriculum/scripts'),GPUS_PER_NODE='8',
        CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',MASTER_PORT='29527' if seq_len==16384 else '29528')
    env.pop('YUFORMER_TRAINING_BLOCK_MANIFEST',None)
    if allocator is not None:env['PYTORCH_CUDA_ALLOC_CONF']=allocator
    else:env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    seal_bytes(cp,yaml.safe_dump(cfg,sort_keys=False).encode())
    seal_bytes(ep,''.join(f'{key}={value}\n' for key,value in env.items()).encode())
    receipt={'status':'PASS_PREPARED_PRODUCTION_CONTEXT_STAGE','seq_len':seq_len,'target_labels':stage['target_labels'],
        'config':str(cp),'config_sha256':digest(cp),'env':str(ep),'env_sha256':digest(ep),
        'source_inputs':{str(p):digest(p) for p in [Path(__file__),plan_file,budget_file,frozen_file,model_file,
            release_file,handoff_file,identity]}}
    seal_bytes(output/(name+'.preparation.json'),(json.dumps(receipt,sort_keys=True,indent=2)+'\n').encode())
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--seq-len',type=int,choices=[16384,65536],required=True)
    parser.add_argument('--parent-handoff',type=Path,required=True);parser.add_argument('--window-release',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--evaluation-admission',type=Path)
    args=parser.parse_args();result=render(args.root,plan_file=args.plan,plan_sha256=args.plan_sha256,
        seq_len=args.seq_len,handoff_file=args.parent_handoff,window_release=args.window_release,
        output=args.output,run_dir=args.run_dir,evaluation_admission=args.evaluation_admission)
    print(json.dumps(result,sort_keys=True))
