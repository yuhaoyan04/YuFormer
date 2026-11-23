"""Read-only recurring validation integrity and immutable event-prefix receipts."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import secrets
import sys
import time
import traceback

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path('/opt/huawei/explorer-env/dataset/yyh_arcagi/YUFormer/YuFormer')
CONFIG=ROOT/'eval/configs/authorized_context_chain_20261115_v1.json'
HELPER=Path('/opt/yuformer-build/audit_first_formal_validation_2000_20261115.py')
LIVE=ROOT/'data/health/all-production-nine-domain-validation-live-20261115.json'
HISTORY=ROOT/'data/health/all-production-nine-domain-validation-receipts-20261115'
DOMAINS={'academic','books','chinese','code','encyclopedia','math','multilingual','qa','web'}
TAGS={'validation_metrics/domain/'+name+'/loss' for name in DOMAINS}|{'validation_metrics/loss'}
spec=importlib.util.spec_from_file_location('fixed_first_validation_prefix_helper',HELPER)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)

def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()

def groups_from_events(paths,accumulators):
    groups={}
    for path in paths:
        events=accumulators.setdefault(path,EventAccumulator(str(path),size_guidance={'scalars':0}));events.Reload()
        for tag in events.Tags()['scalars']:
            if tag not in TAGS:continue
            for point in events.Scalars(tag):
                if not math.isfinite(point.value) or point.value<0:raise ValueError('nonfinite or negative actual validation '+tag)
                row=groups.setdefault(point.step,{})
                if tag in row and row[tag]['value']!=point.value:raise ValueError('ambiguous repeated actual validation '+tag)
                row[tag]={'step':point.step,'value':point.value,'wall_time':point.wall_time}
    return groups

def validate_group(group):
    if set(group)!=TAGS:raise ValueError('validation receipt requires all nine domains and aggregate')
    macro=math.fsum(group['validation_metrics/domain/'+d+'/loss']['value'] for d in DOMAINS)/9
    aggregate=group['validation_metrics/loss']['value']
    if not math.isclose(macro,aggregate,rel_tol=1e-6,abs_tol=1e-7):
        raise ValueError('actual equal-domain validation aggregation mismatch')
    return macro,aggregate

def scheduled(step,cfg):
    return 1 <= step <= cfg['train']['expected_iters'] and (step == 1 or step % cfg['train']['eval_every'] == 0 or step == cfg['train']['expected_iters'])

def verify():
    cases=[ROOT/'data/experiments/production_canary/production_8gpu',ROOT/'data/runs/pretrain-final-1t-8gpu']
    q=json.loads((ROOT/'data/health/midtrain-actual-eight-gpu-stage-qualification.json').read_bytes())
    cases += [Path(case['run_dir']) for case in q['cases'].values() if case.get('run_dir')]
    result=[]
    for run in cases:
        cfg=yaml.safe_load((run/'resolved_config.yaml').read_bytes())
        groups=groups_from_events(sorted((run/'tb').glob('*/events.out.tfevents.*')),{});complete=0
        for step,group in groups.items():
            if not scheduled(step,cfg):raise ValueError('actual reference validation schedule mismatch')
            validate_group(group);complete+=1
        if not complete:raise ValueError('actual reference run has no complete nine-domain validation')
        result.append({'run':str(run),'complete_nine_domain_observations':complete,'steps':sorted(groups)})
    print(json.dumps({'status':'PASS_RECURRING_VALIDATION_AUDITOR_ON_ACTUAL_FORMAL_CANARY_AND_FOUR_LONG_REFERENCE_RUNS',
        'scope':'OBSERVER_SYSTEM_VERIFICATION_NOT_FUTURE_CAPABILITY','runs':len(result),'observations':sum(r['complete_nine_domain_observations'] for r in result),'cases':result},sort_keys=True))

def observe():
    config=json.loads(CONFIG.read_bytes());parent=Path(config['output']);runs={4096:ROOT/'data/runs/pretrain-final-1t-8gpu',
        16384:parent/'training-16384',65536:parent/'training-65536'}
    pins={str(path):sha(path) for path in (Path(__file__).resolve(),CONFIG,HELPER,
        ROOT/'pretrain/backends/torchtitan/yuformer_ext/stream_math.py',
        ROOT/'pretrain/backends/torchtitan/yuformer_ext/validation.py',
        ROOT/'pretrain/backends/torchtitan/yuformer_ext/config_registry.py')}
    sys.path.insert(0,str(ROOT/'pretrain/src'));from yuformer_pretrain.lease import RunLease
    accumulators={};checked={};HISTORY.mkdir(exist_ok=True)
    with RunLease(ROOT/'data/health/readonly-all-nine-domain-validation-observer','readonly-all-nine-domain-validation-observer') as lease:
        while True:
            state={'status':'WAITING_ACTUAL_PRODUCTION_VALIDATION','pid':os.getpid(),'observational_only':True,
                'scope':'ALL_ACTUAL_VALIDATION_DOMAIN_INTEGRITY_NOT_FINAL_CAPABILITY','source_inputs':pins,'stages':{},'alerts':[],
                'updated_at':datetime.now(timezone.utc).isoformat()}
            try:
                lease.assert_owner()
                if any(sha(path)!=digest for path,digest in pins.items()):raise ValueError('approved validation observer source changed')
                active=set()
                for seq,run in runs.items():
                    resolved=run/'resolved_config.yaml'
                    if not resolved.exists():state['stages'][str(seq)]={'status':'NOT_STARTED'};continue
                    cfg=yaml.safe_load(resolved.read_bytes());identity_path=run/'run_identity.json';identity=json.loads(identity_path.read_bytes())
                    if (cfg['data']['seq_len']!=seq or sha(resolved)!=identity['resolved_config_sha256']
                            or seq!=4096 and cfg.get('midtrain',{}).get('scope')!='PRODUCTION'):
                        raise ValueError('actual validation stage identity/context mismatch')
                    attempt_path=run/'current_attempt.json'
                    if not attempt_path.exists():state['stages'][str(seq)]={'status':'WAITING_FIRST_ATTEMPT'};continue
                    attempt=json.loads(attempt_path.read_bytes());attempt_id=attempt['attempt_id']
                    if attempt['identity_sha256']!=identity['identity_sha256']:raise ValueError('validation attempt identity mismatch')
                    paths=sorted((run/'tb').glob('*/events.out.tfevents.*'));active.update(paths)
                    groups=groups_from_events(paths,accumulators);pending=[];complete=[]
                    for step,group in sorted(groups.items()):
                        if not scheduled(step,cfg):
                            raise ValueError('actual validation step differs from frozen validation interval/final boundary')
                        if set(group)!=TAGS:
                            pending.append(step)
                            if time.time()-max(row['wall_time'] for row in group.values())>1800:
                                state['alerts'].append(str(seq)+'_MISSING_VALIDATION_DOMAIN_OR_AGGREGATE_AT_'+str(step))
                            continue
                        macro,aggregate=validate_group(group);key=(seq,attempt_id,step);record=HISTORY/(str(seq)+'-'+attempt_id+'-step-'+str(step)+'.json')
                        if key not in checked:
                            if record.exists():
                                old=json.loads(record.read_bytes())
                                if old.get('metrics')!=group or old.get('run_identity_sha256')!=sha(identity_path):raise ValueError('existing validation receipt differs from actual current observation')
                                if any(sha(p)!=h for p,h in old['source_inputs'].items()):raise ValueError('existing immutable validation receipt changed')
                            else:
                                evidence=HISTORY/(str(seq)+'-'+attempt_id+'-step-'+str(step)+'-events-'+secrets.token_hex(6));evidence.mkdir()
                                snapshots=[helper.snapshot_prefix(path,evidence/('events.out.tfevents.snapshot.'+str(i))) for i,path in enumerate(paths)]
                                copied,missing=helper.metrics([Path(row['snapshot']) for row in snapshots],step)
                                if copied is None or missing or copied['metrics']!=group:raise ValueError('immutable recurring validation prefix differs from actual measurements')
                                inputs={**pins,str(resolved):sha(resolved),str(identity_path):sha(identity_path),
                                        **{row['snapshot']:row['sha256'] for row in snapshots}}
                                receipt={'status':'PASS_ACTUAL_NINE_DOMAIN_VALIDATION_AND_IMMUTABLE_EVENT_PREFIX',
                                    'scope':'INTERMEDIATE_VALIDATION_INTEGRITY_ONLY_NOT_CHECKPOINT_OR_CAPABILITY_ADMISSION',
                                    'seq_len':seq,'step':step,'observed_during_attempt_id':attempt_id,
                                    'attempt_scope':'RECEIPT_OBSERVED_DURING_CURRENT_ATTEMPT_EVENT_HISTORY_MAY_INCLUDE_PRIOR_SAME_RUN_ATTEMPTS',
                                    'run_identity_sha256':sha(identity_path),
                                    'validation_samples_per_domain':cfg['data']['val_samples_per_domain'],
                                    'validation_target_tokens_per_domain':cfg['data']['val_samples_per_domain']*seq,
                                    'metrics':group,'aggregate':aggregate,'equal_domain_macro':macro,
                                    'immutable_event_prefixes':snapshots,'source_inputs':inputs}
                                lease.assert_owner()
                                if any(sha(p)!=h for p,h in inputs.items()):raise ValueError('validation receipt source changed before publication')
                                with record.open('xb') as handle:
                                    handle.write((json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+'\n').encode());handle.flush();os.fsync(handle.fileno())
                            checked[key]={'step':step,'aggregate':aggregate,'receipt':str(record),'receipt_sha256':sha(record)}
                        complete.append(checked[key])
                    state['stages'][str(seq)]={'status':'PASS_ALL_OBSERVED_NINE_DOMAIN_VALIDATIONS' if complete and not pending else 'WAITING_COMPLETE_VALIDATION',
                        'attempt_id':attempt_id,'completed_validation_steps':[r['step'] for r in complete],
                        'latest':complete[-1] if complete else None,'pending_metric_flush_steps':pending}
                    if complete:state['status']='OBSERVING_ACTUAL_PRODUCTION_VALIDATION_ALL_COMPLETED_GROUPS_PASS'
                for stale in set(accumulators)-active:del accumulators[stale]
            except Exception as error:
                state.update(status='VALIDATION_OBSERVER_ERROR_FAIL_CLOSED',error=repr(error),traceback=traceback.format_exc())
            temp=LIVE.with_suffix('.tmp');temp.write_text(json.dumps(state,sort_keys=True,indent=2,allow_nan=False)+'\n');os.replace(temp,LIVE)
            if state['alerts'] or 'error' in state:print(json.dumps(state,sort_keys=True),flush=True)
            time.sleep(30)

if __name__=='__main__':verify() if '--verify-actual' in sys.argv else observe()
