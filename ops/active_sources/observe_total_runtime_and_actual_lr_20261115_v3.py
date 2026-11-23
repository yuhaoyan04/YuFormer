"""Observe complete actual stage progress; no training or configuration writes."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path('/opt/huawei/explorer-env/dataset/yyh_arcagi/YUFormer/YuFormer')
FORMAL=ROOT/'data/runs/pretrain-final-1t-8gpu'
HEALTH=ROOT/'data/health/total-pretrain-runtime-and-actual-lr-live.json'
CONFIG=ROOT/'eval/configs/authorized_context_chain_20261115_v1.json'
PLAN=ROOT/'midtrain/configs/production_total966b_stage_plan.yaml'
CANARY=ROOT/'data/health/production-throughput-canary.json'
BUDGET=ROOT/'experiments/e4_curriculum/results/effective_total_training_budget.json'

def digest(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()

def expected_lr(config,step):
    optim=config['optim'];total=config['train']['expected_iters'];warm=optim['warmup_steps']
    decay=optim['decay_steps'] if optim['lr_schedule']=='wsd' else total-warm
    stable=total+1-warm-decay
    if not 1<=step<=total or stable<0:raise ValueError('LR observation outside actual training horizon')
    if step<=warm:factor=step/warm
    elif step<=warm+stable:factor=1.
    else:factor=optim['min_lr_ratio']+(1-optim['min_lr_ratio'])*.5*(1+math.cos(math.pi*(step-warm-stable)/decay))
    return optim['lr']*factor

def q10(values):
    values=sorted(values);position=(len(values)-1)*.1;left=int(position);fraction=position-left
    return values[left]*(1-fraction)+values[min(left+1,len(values)-1)]*fraction

def actual_lr_check(events,config):
    count=0;mismatches=[];latest={}
    for tag in events.Tags()['scalars']:
        if not tag.startswith('lr/'):continue
        points=events.Scalars(tag)
        for point in points:
            count+=1;expected=expected_lr(config,point.step)
            if not math.isfinite(point.value) or not math.isclose(point.value,expected,rel_tol=1e-6,abs_tol=1e-14):
                mismatches.append({'tag':tag,'step':point.step,'actual':point.value,'expected':expected})
        if points:
            point=points[-1];latest[tag]={'step':point.step,'actual':point.value,'expected':expected_lr(config,point.step)}
    return {'status':'FAIL' if mismatches else 'PASS_ALL_OBSERVED_LEARNING_RATES' if count else 'PENDING_FIRST_LR_SCALAR',
            'observations_checked':count,'mismatches':mismatches,'latest':latest}

def effective_progress(config,logged_step,attempt,completion,accounting,identity):
    total=config['train']['expected_iters'];target=config['train']['budget_tokens']
    if not 0<=logged_step<=total:raise ValueError('logged step exceeds actual frozen stage horizon')
    matched=(completion.get('attempt_id')==attempt.get('attempt_id') and bool(attempt.get('attempt_id'))
             and completion.get('identity_sha256')==identity.get('identity_sha256'))
    if matched and completion.get('status')=='FAIL':raise ValueError('actual current production training attempt failed')
    finished=(matched and completion.get('status')=='PASS' and completion.get('returncode')==0
              and attempt.get('status')=='PASS' and accounting.get('status')=='PASS'
              and accounting.get('attempt_id')==attempt['attempt_id'] and accounting.get('step')==total
              and accounting.get('global_tokens_seen')==target
              and type(accounting.get('consumed_samples'))is int
              and accounting.get('consumed_samples')*config['data']['seq_len']==target)
    if finished:return total,True,'ACTUAL_FINAL_PROCESS_AND_BUDGET_ACCOUNTING'
    resumed=attempt.get('resumed_from_step',0)
    if type(resumed)is not int or not 0<=resumed<=total:raise ValueError('invalid actual resumed step')
    return max(logged_step,resumed),False,'ACTUAL_CURRENT_ATTEMPT_LOG_OR_RESTORED_CURSOR'

def load_json(path):
    return json.loads(path.read_bytes()) if path.exists() else {}

def read_stage(run,stage,*,accumulators,qualified_rate,now,expected_optim):
    resolved=run/'resolved_config.yaml'
    if not resolved.exists():return {'seq_len':stage['seq_len'],'step':0,'completed':False,'status':'NOT_STARTED',
        'remaining_labels':stage['tokens'],'p10_tps_per_card':qualified_rate,'lr_checks':[],'alerts':[]}
    config=yaml.safe_load(resolved.read_bytes());identity=load_json(run/'run_identity.json')
    if (config['data']['seq_len']!=stage['seq_len'] or config['train']['expected_iters']!=stage['updates']
            or config['train']['budget_tokens']!=stage['tokens'] or config['parallel']['world_size']!=8
            or digest(resolved)!=identity.get('resolved_config_sha256')
            or stage['seq_len']!=4096 and config.get('midtrain',{}).get('scope')!='PRODUCTION'):
        raise ValueError('actual resolved stage differs from its identity or approved budget')
    if any(config['optim'].get(key)!=value for key,value in expected_optim.items()):
        raise ValueError('actual learning-rate schedule differs from the frozen intended stage')
    paths=set((run/'tb').glob('*/events.out.tfevents.*'));points=[];losses=[];checks=[]
    for path in sorted(paths):
        events=accumulators.setdefault(path,EventAccumulator(str(path),size_guidance={'scalars':0}));events.Reload()
        checks.append(actual_lr_check(events,config));tags=events.Tags()['scalars']
        if 'throughput(tps)' in tags:points+=events.Scalars('throughput(tps)')
        if 'loss_metrics/global_avg_loss' in tags:losses+=events.Scalars('loss_metrics/global_avg_loss')
    attempt=load_json(run/'current_attempt.json');complete=load_json(run/'training_complete.json');accounting=load_json(run/'training_accounting.json')
    last=max(losses,key=lambda x:(x.step,x.wall_time)) if losses else None
    step,finished,basis=effective_progress(config,last.step if last else 0,attempt,complete,accounting,identity)
    warm=config['optim']['warmup_steps'];steady=[p for p in sorted(points,key=lambda p:p.wall_time)
        if p.step>100 and math.isfinite(p.value) and p.value>0][-100:]
    speed=q10([p.value for p in steady]) if len(steady)>=10 else qualified_rate
    alerts=[]
    if any(check['status']=='FAIL' for check in checks):alerts.append(str(stage['seq_len'])+'_ACTUAL_LR_MISMATCH')
    if last and not math.isfinite(last.value):alerts.append(str(stage['seq_len'])+'_NONFINITE_LOSS')
    if last and not finished and now-last.wall_time>1800:alerts.append(str(stage['seq_len'])+'_NO_FRESH_TRAINING_METRIC_FOR1800S')
    row={'seq_len':stage['seq_len'],'step':step,'logged_step':last.step if last else None,'completed':finished,
        'status':'TRAINING_BUDGET_COMPLETED_VERIFY_NATIVE_AND_CAPABILITY' if finished else 'CURRENT_ATTEMPT_OBSERVED',
        'progress_basis':basis,'remaining_labels':max(0,stage['tokens']-step*4194304),
        'p10_tps_per_card':speed,'lr_checks':checks,'alerts':alerts,
        'latest_loss':last.value if last and math.isfinite(last.value) else None,
        'latest_loss_is_finite':math.isfinite(last.value) if last else None,
        'last_metric_age_seconds':now-last.wall_time if last else None,
        'rolling_steady_throughput_points':len(steady),
        'throughput_window_start_step':steady[0].step if steady else None,
        'throughput_window_end_step':steady[-1].step if steady else None}
    row['phase']='completed' if finished else 'warmup' if step<=warm else 'stable' if config['optim']['lr_schedule']=='wsd' and step<=config['train']['expected_iters']-config['optim']['decay_steps'] else 'decay'
    return row

def build_state(accumulators):
    cfg=load_json(CONFIG);budget=load_json(BUDGET);forecast=load_json(CANARY)['evaluation'];duty=forecast['planned_training_duty_cycle'];now=time.time()
    if digest(FORMAL/'run_identity.json')!=cfg['formal_run_identity_sha256']:
        raise ValueError('formal run identity differs from the authorized actual production run')
    base=yaml.safe_load((ROOT/'pretrain/configs/pretrain_base.yaml').read_bytes());plan=yaml.safe_load(PLAN.read_bytes())
    long_stages={s['seq_len']:s for s in plan['stages']}
    by_name={s['name']:s for s in forecast['stage_forecasts']};rows=[];remaining=[];alerts=[];active_events=set()
    for stage in budget['stages']:
        seq=stage['seq_len'];run=FORMAL if seq==4096 else Path(cfg['output'])/('training-'+str(seq))
        expected={key:base['optim'][key] for key in ('lr','warmup_steps','lr_schedule','min_lr_ratio','decay_steps')} if seq==4096 else {
            'lr':long_stages[seq]['peak_learning_rate'],'warmup_steps':long_stages[seq]['warmup_steps'],
            'lr_schedule':'cosine','min_lr_ratio':long_stages[seq]['min_lr_ratio']}
        row=read_stage(run,stage,accumulators=accumulators,qualified_rate=by_name[stage['name']]['p10_tps_per_device'],now=now,expected_optim=expected)
        rows.append(row);alerts+=row['alerts'];remaining.append({'stage':stage['name'],'remaining_labels':row['remaining_labels'],
            'p10_tps_per_card':row['p10_tps_per_card'],'days':row['remaining_labels']/(row['p10_tps_per_card']*8*duty*86400)})
        active_events.update((run/'tb').glob('*/events.out.tfevents.*'))
    for stale in set(accumulators)-active_events:del accumulators[stale]
    starts=[d['started_at_ns']/1e9 for path in (FORMAL/'attempts').glob('*.json') for d in [load_json(path)] if d.get('started_at_ns')]
    if not starts:raise ValueError('actual formal training start time is absent')
    elapsed=(now-min(starts))/86400;days=sum(s['days'] for s in remaining);chain=load_json(Path(cfg['health']))
    all_trained=all(row['completed'] for row in rows);all_done=chain.get('status')=='PASS_COMPLETE_4K_WSD_16K_64K_TRAINING_AND_ACTUAL_CAPABILITY_GATES'
    if all_done and not all_trained:raise ValueError('chain completion contradicts actual final stage accounting')
    final_eval_pending=all_trained and not all_done
    if chain.get('status')=='STOPPED_REQUIRES_DIAGNOSIS_NO_STAGE_GATE_BYPASS':alerts.append('CONTEXT_CHAIN_STOPPED_REQUIRES_REVIEW')
    projected=elapsed+days
    if projected>33:alerts.append('PROJECTED_TOTAL_RUNTIME_ABOVE33D_NO_AUTOMATIC_BUDGET_CHANGE')
    status='FORMAL_4K_WSD_RUNNING' if not rows[0]['completed'] else 'COMPLETE_ACTUAL_TRAINING_AND_CAPABILITY_CHAIN' if all_done else 'TRAINING_BUDGET_COMPLETE_FINAL_CAPABILITY_EVALUATION_PENDING' if final_eval_pending else chain.get('status','WAITING_VERIFIED_CONTEXT_PROGRESSION')
    return {'status':status,'observational_only':True,'pid':os.getpid(),'runtime_scope':'TOTAL_4K_WSD_PLUS_16K_PLUS_64K',
        'scalar_history_policy':'ALL_SCALARS_RETAINED_LAST100_ACTUAL_LOGGED_TPS_POINTS',
        'qualified_initial_forecast_days':forecast['projected_total_training_days'],'runtime_ceiling_days':33,
        'planned_training_duty_cycle':duty,'actual_elapsed_days':elapsed,'remaining_training_days':days,
        'whole_chain_remaining_days':None if final_eval_pending else days,
        'projected_total_days_including_elapsed':None if final_eval_pending else projected,
        'projected_completion_utc':None if final_eval_pending else datetime.datetime.fromtimestamp(now+days*86400,datetime.timezone.utc).isoformat(),
        'forecast_note':'Remaining training with declared90% duty allowance; final capability evaluation has no zero-time completion claim.',
        'all_training_budgets_completed':all_trained,'actual_full_chain_completed':all_done,
        'final_capability_evaluation_pending':final_eval_pending,'chain_status':chain.get('status'),
        'remaining_stage_forecasts':remaining,'stages':rows,'actual_lr_contracts':rows[0]['lr_checks'],
        'step':rows[0]['step'],'phase':rows[0].get('phase','not_started'),'latest_loss':rows[0].get('latest_loss'),
        'latest_loss_is_finite':rows[0].get('latest_loss_is_finite'),'last_metric_age_seconds':rows[0].get('last_metric_age_seconds'),
        'rolling_4k_steady_throughput_points':rows[0].get('rolling_steady_throughput_points',0),
        'throughput_basis':'LIVE_LAST100_ACTUAL_LOGGED_POINTS_P10','alerts':alerts,
        'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}

def verify_actual():
    checks=[];runs=[ROOT/'data/experiments/production_canary/production_8gpu']
    qualified=load_json(ROOT/'data/health/midtrain-actual-eight-gpu-stage-qualification.json')
    runs += [Path(case['run_dir']) for case in qualified['cases'].values() if case.get('run_dir')]
    for run in runs:
        config=yaml.safe_load((run/'resolved_config.yaml').read_bytes())
        for path in sorted((run/'tb').glob('*/events.out.tfevents.*')):
            events=EventAccumulator(str(path),size_guidance={'scalars':0});events.Reload();check=actual_lr_check(events,config)
            if check['status']!='PASS_ALL_OBSERVED_LEARNING_RATES':raise ValueError((run,check))
            checks.append({'run':str(run),**check})
    print(json.dumps({'status':'PASS_OBSERVER_ACTUAL_MAIN_CANARY_AND_FOUR_LONG_STAGE_LR_TRACES',
        'runs':len(checks),'actual_lr_scalars':sum(c['observations_checked'] for c in checks),'checks':checks},sort_keys=True))

def observe():
    sys.path.insert(0,str(ROOT/'pretrain/src'))
    from yuformer_pretrain.lease import RunLease
    files=[Path(__file__),CONFIG,PLAN,CANARY,BUDGET,ROOT/'pretrain/configs/pretrain_base.yaml',ROOT/'pretrain/configs/pretrain_base.env']
    pins={str(path):digest(path) for path in files};accumulators={}
    lease=RunLease(ROOT/'data/health/total-pretrain-runtime-lr-observer','read-only-total-runtime-lr-observer').acquire()
    while True:
        try:
            lease.assert_owner()
            if any(digest(path)!=sha for path,sha in pins.items()):raise ValueError('approved observer scope/source changed')
            state=build_state(accumulators);state['source_inputs']=pins
        except Exception as error:
            state={'status':'OBSERVER_ERROR_FAIL_CLOSED','observational_only':True,'pid':os.getpid(),'alerts':[],
                'source_inputs':pins,'error':repr(error),'traceback':traceback.format_exc(),
                'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
        temporary=HEALTH.with_suffix('.tmp');temporary.write_text(json.dumps(state,sort_keys=True,indent=2)+'\n');os.replace(temporary,HEALTH)
        if state['alerts'] or state['status']=='OBSERVER_ERROR_FAIL_CLOSED':print(json.dumps(state,sort_keys=True),flush=True)
        time.sleep(30)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify-actual',action='store_true');parser.add_argument('--snapshot-only',action='store_true');args=parser.parse_args()
    if args.verify_actual:verify_actual()
    elif args.snapshot_only:print(json.dumps(build_state({}),sort_keys=True))
    else:observe()
