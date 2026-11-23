"""Read-only actual LR/native-counter observer for the declared future stages."""
import argparse
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path('/opt/huawei/explorer-env/dataset/yyh_arcagi/YUFormer/YuFormer')
CONFIG=ROOT/'eval/configs/authorized_context_chain_20261115_v1.json'
PLAN=ROOT/'midtrain/configs/production_total966b_stage_plan.yaml'
HEALTH=ROOT/'data/health/long-stage-actual-lr-native-counters-live-20261115.json'
PROOF=ROOT/'data/health/long-stage-lr-reference-and-actual-qualification-proof-20261115.json'
sys.path[:0]=[str(ROOT/'pretrain/src'),str(ROOT/'midtrain/src')]

def digest(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()

def expected_lr(config,update):
    optim=config['optim'];total=config['train']['expected_iters'];warm=optim['warmup_steps']
    if not 1<=update<=total+1 or optim['lr_schedule']!='cosine':raise ValueError('invalid cosine observation step')
    decay=total-warm;stable_end=warm+1
    if update<=warm:factor=update/warm
    elif update<=stable_end:factor=1.
    else:factor=optim['min_lr_ratio']+(1-optim['min_lr_ratio'])*.5*(1+math.cos(math.pi*(update-stable_end)/decay))
    return optim['lr']*factor

def actual_lr_check(events,config):
    count=0;last={};mismatches=[]
    for tag in events.Tags()['scalars']:
        if not tag.startswith('lr/'):continue
        points=events.Scalars(tag)
        for point in points:
            count+=1;expected=expected_lr(config,point.step)
            if not math.isfinite(point.value) or not math.isclose(point.value,expected,rel_tol=1e-6,abs_tol=1e-14):
                mismatches.append({'tag':tag,'step':point.step,'actual':point.value,'expected':expected})
        if points:
            point=points[-1];last[tag]={'step':point.step,'actual':point.value,'expected':expected_lr(config,point.step)}
    return {'status':'FAIL' if mismatches else 'PASS_ALL_OBSERVED_COSINE_LEARNING_RATES' if count else 'PENDING_FIRST_LR_SCALAR',
            'observations_checked':count,'latest':last,'mismatches':mismatches}

def verify():
    import torch
    from torchtitan.components.optimizer import LRSchedulersContainer
    plan=yaml.safe_load(PLAN.read_bytes());boundaries=[];actual=[]
    for stage in plan['stages']:
        total,warm=stage['updates'],stage['warmup_steps']
        config={'train':{'expected_iters':total},'optim':{'lr':stage['peak_learning_rate'],
            'warmup_steps':warm,'min_lr_ratio':stage['min_lr_ratio'],'lr_schedule':stage['lr_schedule']}}
        optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))],lr=stage['peak_learning_rate'])
        scheduler=LRSchedulersContainer.Config(total_steps=total,warmup_steps=warm,decay_ratio=None,
            decay_type='cosine',min_lr_factor=stage['min_lr_ratio']).build(optimizers=[optimizer],training_steps=total)
        for update in sorted({1,2,warm-1,warm,warm+1,warm+2,total//2,total-1,total,total+1}):
            scheduler.load_state_dict({'last_epoch':update-1})
            value=list(scheduler.get_metrics().values())[0];expected=expected_lr(config,update)
            if not math.isclose(value,expected,rel_tol=1e-12,abs_tol=1e-20):raise ValueError('independent long cosine formula differs from pinned scheduler')
            boundaries.append({'seq_len':stage['seq_len'],'update':update,'actual':value,'expected':expected})
    qualified=json.loads((ROOT/'data/health/midtrain-actual-eight-gpu-stage-qualification.json').read_bytes())
    for name,case in qualified['cases'].items():
        if not case.get('run_dir'):continue
        run=Path(case['run_dir']);config=yaml.safe_load((run/'resolved_config.yaml').read_bytes())
        for path in sorted((run/'tb').glob('*/events.out.tfevents.*')):
            events=EventAccumulator(str(path),size_guidance={'scalars':0});events.Reload()
            check=actual_lr_check(events,config)
            if check['status']!='PASS_ALL_OBSERVED_COSINE_LEARNING_RATES':raise ValueError((name,check))
            actual.append({'case':name,'events_sha256':digest(path),**check})
    if len(actual)!=4:raise ValueError('must check all four real long-stage reference/resume runs')
    report={'status':'PASS_PRODUCTION_COSINE_BOUNDARIES_AND_FOUR_REAL_STAGE_QUALIFICATION_LR_TRACES',
        'scope':'OBSERVER_MATH_VERIFICATION_NOT_NEW_PRODUCTION_MODEL_CAPABILITY',
        'boundaries':boundaries,'actual_qualification_cases':actual,
        'source_inputs':{str(p):digest(p) for p in (Path(__file__).resolve(),CONFIG,PLAN,
            ROOT/'data/health/midtrain-actual-eight-gpu-stage-qualification.json',
            Path(sys.modules[LRSchedulersContainer.__module__].__file__).resolve())}}
    with PROOF.open('xb') as handle:
        handle.write((json.dumps(report,sort_keys=True,indent=2)+'\n').encode());handle.flush();os.fsync(handle.fileno())
    print(json.dumps({'status':report['status'],'boundary_points':len(boundaries),
                      'actual_qualification_runs':len(actual),'actual_lr_scalars':sum(x['observations_checked'] for x in actual)},sort_keys=True))

def native_check(path,config):
    import torch
    from yuformer_midtrain.checkpoint_contract import checkpoint_scalars
    from yuformer_midtrain.window_release import stat_identity
    raw=(path/'.metadata').read_bytes();metadata=pickle.loads(raw);step=int(path.name[5:])
    pins={p.name:stat_identity(p.stat()) for p in path.iterdir() if p.is_file()}
    files={'.metadata'}
    for info in metadata.storage_data.values():
        name=str(info.relative_path)
        if Path(name).name!=name or name not in pins or info.offset<0 or info.length<=0 or info.offset+info.length>pins[name]['bytes']:
            raise ValueError('long native shard roster/bounds invalid')
        files.add(name)
    if files!=set(pins):raise ValueError('long native shard roster incomplete')
    values=checkpoint_scalars(path,hashlib.sha256(raw).hexdigest());seq=config['data']['seq_len']
    samples=step*8*config['train']['micro_batch_size']*config['train']['grad_accum']
    expected={'train_state.step':step,'lr_scheduler.last_epoch':step,'dataloader.seq_len':seq,
        'train_state.samples_consumed':samples,'dataloader.consumed_samples':samples,'train_state.global_tokens_seen':samples*seq}
    if values!=expected:raise ValueError('long native step/scheduler/cursor/labels mismatch')
    def payload(name):
        infos=[info for index,info in metadata.storage_data.items() if index.fqn==name]
        if len(infos)!=1:raise ValueError('ambiguous long native control payload')
        info=infos[0]
        with (path/info.relative_path).open('rb') as handle:
            handle.seek(info.offset);return torch.load(io.BytesIO(handle.read(info.length)),map_location='cpu',weights_only=True)
    steps=[key for key in metadata.state_dict_metadata if key.startswith('optimizer.state.') and key.endswith('.step')]
    lrs=[key for key in metadata.state_dict_metadata if key.startswith('optimizer.param_groups.') and key.endswith('.lr')]
    if len(steps)!=164 or len(lrs)!=164 or any(float(payload(key))!=step for key in steps):
        raise ValueError('long native Adam counter/group roster mismatch')
    if any(not math.isclose(float(payload(key)),expected_lr(config,step+1),rel_tol=1e-12,abs_tol=1e-20) for key in lrs):
        raise ValueError('long native saved next-update LR mismatch')
    if any(stat_identity((path/name).stat())!=stat for name,stat in pins.items()) or (path/'.metadata').read_bytes()!=raw:
        raise ValueError('long native checkpoint changed during observation')
    return {'status':'PASS_ACTUAL_LONG_NATIVE_COUNTERS_AND_ADAM_LR','step':step,'native_scalars':values,
            'adam_groups':len(steps),'saved_next_update_lr':expected_lr(config,step+1),
            'scope':'STORAGE_BOUNDS_AND_CONTROL_STATE_NOT_REPEATED_FULL_TENSOR_FINITE_SCAN'}

def observe():
    from yuformer_pretrain.lease import RunLease
    cfg=json.loads(CONFIG.read_bytes());plan=yaml.safe_load(PLAN.read_bytes());proof=json.loads(PROOF.read_bytes())
    if proof['status']!='PASS_PRODUCTION_COSINE_BOUNDARIES_AND_FOUR_REAL_STAGE_QUALIFICATION_LR_TRACES':
        raise ValueError('actual long LR observer verification must pass first')
    pins={str(p):digest(p) for p in (Path(__file__).resolve(),CONFIG,PLAN,PROOF,
        ROOT/'midtrain/src/yuformer_midtrain/checkpoint_contract.py',ROOT/'midtrain/src/yuformer_midtrain/window_release.py')}
    accumulators={};checked={};lease=RunLease(ROOT/'data/health/readonly-long-lr-native-observer',
        'readonly-long-lr-native-observer').acquire()
    while True:
        state={'status':'WAITING_FOR_DECLARED_PRODUCTION_CONTEXT_STAGES','pid':os.getpid(),
            'observational_only':True,'source_inputs':pins,'alerts':[],'stages':{},
            'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        try:
            lease.assert_owner()
            if any(digest(p)!=sha for p,sha in pins.items()):raise ValueError('long observer approved inputs changed')
            for stage in plan['stages']:
                seq=stage['seq_len'];run=Path(cfg['output'])/('training-'+str(seq));resolved=run/'resolved_config.yaml'
                if not resolved.exists():state['stages'][str(seq)]={'status':'NOT_STARTED'};continue
                actual=yaml.safe_load(resolved.read_bytes());o=actual['optim'];t=actual['train']
                if (actual['data']['seq_len']!=seq or actual['midtrain']['scope']!='PRODUCTION'
                        or t['expected_iters']!=stage['updates'] or t['budget_tokens']!=stage['target_labels']
                        or t['micro_batch_size']!=stage['micro_batch_size'] or t['grad_accum']!=stage['grad_accum']
                        or actual['parallel']['world_size']!=8 or o['lr']!=stage['peak_learning_rate']
                        or o['warmup_steps']!=stage['warmup_steps'] or o['lr_schedule']!='cosine'
                        or o['min_lr_ratio']!=stage['min_lr_ratio']):raise ValueError('actual long runtime configuration differs from frozen approved stage')
                checks=[];losses=[]
                for path in sorted((run/'tb').glob('*/events.out.tfevents.*')):
                    events=accumulators.setdefault(path,EventAccumulator(str(path),size_guidance={'scalars':0}));events.Reload()
                    checks.append(actual_lr_check(events,actual))
                    if 'loss_metrics/global_avg_loss' in events.Tags()['scalars']:losses+=events.Scalars('loss_metrics/global_avg_loss')
                if any(c['status']=='FAIL' for c in checks):state['alerts'].append(str(seq)+'_ACTUAL_COSINE_LR_MISMATCH')
                row={'status':'PRODUCTION_CONTEXT_STAGE_OBSERVED','actual_lr_checks':checks}
                if losses:
                    last=max(losses,key=lambda x:(x.step,x.wall_time));row.update(step=last.step,loss=last.value,last_metric_age_seconds=time.time()-last.wall_time)
                    if not math.isfinite(last.value):state['alerts'].append(str(seq)+'_NONFINITE_LOSS')
                paths=sorted((run/'checkpoint').glob('step-*/.metadata'),key=lambda p:int(p.parent.name[5:]))
                if paths:
                    path=paths[-1].parent;key=(seq,path.name)
                    if key not in checked:checked[key]=native_check(path,actual)
                    row['latest_native_control_check']=checked[key]
                state['stages'][str(seq)]=row;state['status']='OBSERVING_ACTUAL_PRODUCTION_CONTEXT_STAGE_LR_AND_NATIVE_COUNTERS'
        except Exception as error:
            state.update(status='LONG_STAGE_OBSERVER_ERROR_REQUIRES_REVIEW',error=repr(error),traceback=traceback.format_exc())
        tmp=HEALTH.with_suffix('.tmp');tmp.write_text(json.dumps(state,sort_keys=True,indent=2)+'\n');os.replace(tmp,HEALTH)
        if state['alerts'] or 'error' in state:print(json.dumps(state,sort_keys=True),flush=True)
        time.sleep(30)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    verify() if args.verify else observe()
