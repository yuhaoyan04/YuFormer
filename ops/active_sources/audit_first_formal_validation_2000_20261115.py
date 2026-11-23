"""Snapshot first formal nine-domain validation without collecting a final result."""
from datetime import datetime, timezone
import hashlib
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
RUN=ROOT/'data/runs/pretrain-final-1t-8gpu'
LIVE=ROOT/'data/health/first-formal-nine-domain-validation-2000-live-20261115.json'
REPORT=ROOT/'data/health/first-formal-nine-domain-validation-2000-admission-20261115.json'
DOMAINS={'academic','books','chinese','code','encyclopedia','math','multilingual','qa','web'}
STEP=2000

def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()

def metrics(paths,step):
    required={'validation_metrics/domain/'+domain+'/loss' for domain in DOMAINS}|{'validation_metrics/loss'}
    found={}
    for path in paths:
        events=EventAccumulator(str(path),size_guidance={'scalars':0});events.Reload()
        for tag in events.Tags()['scalars']:
            if tag not in required:continue
            for point in events.Scalars(tag):
                if point.step==step:
                    if not math.isfinite(point.value) or point.value<0:raise ValueError('nonfinite/negative first validation metric '+tag)
                    if tag in found and found[tag]['value']!=point.value:raise ValueError('ambiguous repeated validation metric '+tag)
                    found[tag]={'step':point.step,'value':point.value,'wall_time':point.wall_time}
    if set(found)!=required:return None,sorted(required-set(found))
    macro=math.fsum(found['validation_metrics/domain/'+domain+'/loss']['value'] for domain in DOMAINS)/len(DOMAINS)
    aggregate=found['validation_metrics/loss']['value']
    if not math.isclose(macro,aggregate,rel_tol=1e-6,abs_tol=1e-7):
        raise ValueError('equal-domain first validation aggregation differs from actual logged global loss')
    return {'metrics':found,'equal_domain_macro':macro,'aggregate':aggregate},[]

def snapshot_prefix(source,output):
    with Path(source).open('rb') as read,Path(output).open('xb') as write:
        stat=os.fstat(read.fileno());remaining=stat.st_size;digest=hashlib.sha256()
        while remaining:
            block=read.read(min(8<<20,remaining))
            if not block:raise ValueError('active event prefix truncated while snapshotting')
            digest.update(block);write.write(block);remaining-=len(block)
        write.flush();os.fsync(write.fileno())
        after=os.fstat(read.fileno())
        if (after.st_ino,after.st_dev)!=(stat.st_ino,stat.st_dev):raise ValueError('active event file changed identity')
    with Path(source).open('rb') as read:
        remaining=stat.st_size;again=hashlib.sha256()
        while remaining:
            block=read.read(min(8<<20,remaining))
            if not block:raise ValueError('active event prefix disappeared')
            again.update(block);remaining-=len(block)
    if again.hexdigest()!=digest.hexdigest() or sha(output)!=digest.hexdigest():raise ValueError('active event prefix mutated')
    return {'original_file':str(source),'snapshot':str(output),'bytes':stat.st_size,'sha256':digest.hexdigest()}

def publish(value):
    value.update(pid=os.getpid(),observational_only=True,updated_at=datetime.now(timezone.utc).isoformat())
    temp=LIVE.with_suffix('.tmp');temp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n');os.replace(temp,LIVE)

def verify_canary():
    import tempfile
    run=ROOT/'data/experiments/production_canary/production_8gpu'
    paths=sorted((run/'tb').glob('*/events.out.tfevents.*'));original,missing=metrics(paths,200)
    if original is None:raise ValueError('actual completed canary lacks required validation: '+repr(missing))
    with tempfile.TemporaryDirectory(prefix='yuformer-val-prefix-unit-') as directory:
        copies=[snapshot_prefix(path,Path(directory)/('events.out.tfevents.snapshot.'+str(index))) for index,path in enumerate(paths)]
        restored,missing=metrics([Path(row['snapshot']) for row in copies],200)
        if restored!=original or missing:raise ValueError('immutable event-prefix re-read differs')
    print(json.dumps({'status':'PASS_READONLY_VALIDATION_AND_EVENT_PREFIX_AUDITOR_ON_ACTUAL_CANARY200',
        'scope':'AUDITOR_SYSTEM_VERIFICATION_NOT_FORMAL_STEP2000_RESULT','domains':len(DOMAINS),
        'aggregate':original['aggregate'],'equal_domain_macro':original['equal_domain_macro']},sort_keys=True))

def observe():
    if REPORT.exists():raise ValueError('preserve the immutable existing first formal validation admission')
    config=yaml.safe_load((RUN/'resolved_config.yaml').read_bytes());identity=json.loads((RUN/'run_identity.json').read_bytes())
    initial=json.loads((RUN/'current_attempt.json').read_bytes())
    if config['data']['val_samples_per_domain']!=64 or config['data']['seq_len']!=4096:
        raise ValueError('first formal validation auditor requires the declared exact64 samples/domain at4K')
    inputs=[Path(__file__).resolve(),RUN/'run_identity.json',RUN/'resolved_config.yaml',
        ROOT/'pretrain/backends/torchtitan/yuformer_ext/validation.py',ROOT/'pretrain/backends/torchtitan/yuformer_ext/config_registry.py',
        ROOT/'data/processed/audits/pretrain_1t_token_streams.json']
    pins={str(path):sha(path) for path in inputs}
    sys.path.insert(0,str(ROOT/'pretrain/src'));from yuformer_pretrain.lease import RunLease
    with RunLease(ROOT/'data/health/readonly-first-formal-validation-2000-observer','readonly-first-formal-validation-2000') as lease:
        try:
            while True:
                lease.assert_owner()
                if any(sha(path)!=digest for path,digest in pins.items()):raise ValueError('formal validation identity/source changed')
                attempt=json.loads((RUN/'current_attempt.json').read_bytes())
                if attempt['attempt_id']!=initial['attempt_id'] or attempt['identity_sha256']!=identity['identity_sha256']:
                    raise ValueError('review first validation observer after actual attempt/identity transition')
                paths=sorted((RUN/'tb').glob('*/events.out.tfevents.*'));value,missing=metrics(paths,STEP)
                native=ROOT/'data/health/formal-native-checkpoint-state-admissions-20261115'/f'step-{STEP}.json'
                if value is None or not native.exists():
                    publish({'status':'WAITING_FOR_ACTUAL_STEP2000_NINE_DOMAIN_VALIDATION_AND_NATIVE_CHECKPOINT',
                             'step':STEP,'missing_metric_tags':missing,'native_checkpoint_control_admission_present':native.exists()})
                    time.sleep(15);continue
                control=json.loads(native.read_bytes())
                if (control['status']!='PASS_ACTUAL_NATIVE_CHECKPOINT_STRUCTURAL_AND_FULL_CONTROL_STATE'
                        or control['step']!=STEP or control['run_identity_sha256']!=sha(RUN/'run_identity.json')
                        or control['actual_scalars']['train_state.global_tokens_seen']!=STEP*4194304):
                    raise ValueError('first validation checkpoint control admission mismatch')
                evidence=ROOT/'data/health'/('first-formal-validation-2000-evidence-20261115-'+secrets.token_hex(6));evidence.mkdir()
                prefixes=[snapshot_prefix(path,evidence/('events.out.tfevents.snapshot.'+str(index))) for index,path in enumerate(paths)]
                copied,missing=metrics([Path(row['snapshot']) for row in prefixes],STEP)
                if copied!=value or missing:raise ValueError('first formal validation event-prefix snapshot mismatch')
                for row in prefixes:pins[row['snapshot']]=row['sha256']
                pins[str(native)]=sha(native)
                report={'status':'PASS_FIRST_FORMAL_STEP2000_NINE_DOMAIN_VALIDATION_AND_MATCHING_NATIVE_CONTROL',
                    'scope':'EARLY_VALIDATION_PROGRESS_NOT_COMPLETED_PRETRAIN_OR_FINAL_CAPABILITY',
                    'step':STEP,'attempt_id':attempt['attempt_id'],'run_identity_sha256':sha(RUN/'run_identity.json'),
                    'internal_run_identity_sha256':identity['identity_sha256'],'validation_samples_per_domain':64,
                    'validation_target_tokens_per_domain':64*4096,'domain_count':9,**value,
                    'checkpoint_control':control['actual_scalars'],'native_checkpoint_admission':str(native),
                    'immutable_event_prefix_snapshots':prefixes,'source_inputs':pins,
                    'observed_at':datetime.now(timezone.utc).isoformat()}
                if any(sha(path)!=digest for path,digest in pins.items()):raise ValueError('validation evidence changed before admission')
                lease.assert_owner()
                with REPORT.open('xb') as handle:
                    handle.write((json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n').encode());handle.flush();os.fsync(handle.fileno())
                publish({'status':report['status'],'scope':report['scope'],'admission':str(REPORT),'admission_sha256':sha(REPORT),
                         'step':STEP,'domain_count':9,'aggregate':value['aggregate']})
                print(json.dumps({'status':report['status'],'aggregate':value['aggregate'],'admission':str(REPORT)},sort_keys=True),flush=True)
                return
        except Exception as error:
            publish({'status':'FIRST_VALIDATION_AUDITOR_REQUIRES_REVIEW','error':repr(error),'traceback':traceback.format_exc()})
            raise

if __name__=='__main__':verify_canary() if '--verify-canary' in sys.argv else observe()
