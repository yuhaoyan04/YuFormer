"""CPU-only on-demand operational review; never writes active training output."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
from itertools import accumulate
from bisect import bisect_right
import hashlib, importlib.util, json, math, os, subprocess
import yaml

ROOT=Path('/opt/huawei/explorer-env/dataset/yyh_arcagi/YUFormer/YuFormer')
HEALTH=ROOT/'data/health'
RUN=ROOT/'data/runs/pretrain-final-1t-8gpu'
HELPER=Path('/opt/yuformer-build/audit_first_formal_validation_2000_20261115.py')
LR_HELPER=Path('/opt/yuformer-build/observe_total_runtime_and_actual_lr_20261115_v3.py')
EXPECTED_IDENTITY='8eeae4ac539490c01ed9987b027434fafc82d8f24df473efb3350a19cff05b5d'
NAMES=['authorized-pretrain-live-monitor.json','total-pretrain-runtime-and-actual-lr-live.json','formal-native-checkpoint-state-live-20261115.json','all-production-nine-domain-validation-live-20261115.json','authorized-total-pretrain-context-chain-20261115-v1.json','long-stage-actual-lr-native-counters-live-20261115.json','readonly-future-block-prefetch-live-20261115.json']

def read(p): return json.loads(Path(p).read_bytes())
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def review():
    now=datetime.now(timezone.utc);sources={};hash_cache={}
    def pin(p):
        p=str(Path(p).resolve())
        if p not in hash_cache:
            with open(p,'rb') as handle: hash_cache[p]=hashlib.file_digest(handle,'sha256').hexdigest()
        sources[p]=hash_cache[p];return sources[p]
    assert pin(HELPER)=='7739c2bac4d8b4dab04b3fa3422844be7aa711d80cf178286c21817270cf206d'
    assert pin(LR_HELPER)=='9ad6d868d607a5299eb7bef654d2ef520a0e3fbc0562e8a2d14ec50b6c8a3b46'
    pin(__file__);helper=module('fixed_validation_reader',HELPER);lr=module('fixed_lr_formula',LR_HELPER)
    states={name:read(HEALTH/name) for name in NAMES}
    for name,d in states.items():
        age=(now-datetime.fromisoformat(d.get('updated_at',d.get('time')))).total_seconds()
        assert -60<age<120,(name,'stale heartbeat',age)
        assert not d.get('error') and not d.get('alerts'),(name,d.get('error'),d.get('alerts'))
        raw=Path(f'/proc/{d["pid"]}/stat').read_text();assert raw[raw.rfind(')')+2:].split()[0] not in ('Z','X'),(name,'process missing or zombie')
    live,eta,control,validation,chain,long,prefetch=[states[n] for n in NAMES]
    recovery=read(HEALTH/'live-pretrain-supervisor-recovery-20261118.json')
    assert recovery['status']=='RECOVERED_SAME_LIVE_ATTEMPT_SUPERVISING_BACKGROUND_GUARDS' and not recovery.get('alerts') and not recovery.get('error'),('recovery guardian',recovery)
    assert 0<=(now-datetime.fromisoformat(recovery['updated_at'])).total_seconds()<120,'stale recovery guardian'
    assert pin('/opt/yuformer-build/recover_live_pretrain_supervisors_20261118.py')==recovery['script_sha256']
    for pid,expected in recovery['original_processes'].items():
        raw=Path(f'/proc/{pid}/stat').read_text();fields=raw[raw.rfind(')')+2:].split()
        assert fields[0] not in ('Z','X') and fields[19]==expected['start_ticks'],('original training process missing/replaced',pid)
    assert live.get('alive') is True,'actual pipeline guardian is not live'
    metric_time=datetime.strptime(live['training_metrics']['log_timestamp_utc'],'%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    assert -60<(now-metric_time).total_seconds()<1800,'training metrics stopped advancing'
    assert live['status']=='RUNNING',('PHASE_OR_PROCESS_CHANGE_REQUIRES_MANUAL_REVIEW',live['status'])
    assert chain['status']=='WAITING_FOR_ACTUAL_4K_COMPLETION',('CONTEXT_CHAIN_REQUIRES_MANUAL_REVIEW',chain['status'])
    cfg_path=RUN/'resolved_config.yaml';cfg=yaml.safe_load(cfg_path.read_bytes());identity=read(RUN/'run_identity.json');attempt=read(RUN/'current_attempt.json')
    assert pin(RUN/'run_identity.json')==EXPECTED_IDENTITY
    assert pin(cfg_path)==identity['resolved_config_sha256'] and attempt['identity_sha256']==identity['identity_sha256']
    total=cfg['train']['expected_iters'];seq=cfg['data']['seq_len'];tokens_per_update=cfg['train']['budget_tokens']//total
    assert tokens_per_update==4194304 and seq==4096
    assert attempt['attempt_id']==recovery['attempt_id'],'recovery does not match actual training attempt'
    batch=tokens_per_update//seq;latest_step=max(control['checked_steps'])
    # A restarted observer rechecks retained checkpoints. Earlier immutable
    # admissions survive normal retention and must also be revalidated here.
    steps=sorted(int(path.stem.removeprefix('step-')) for path in (HEALTH/'formal-native-checkpoint-state-admissions-20261115').glob('step-*.json') if int(path.stem.removeprefix('step-'))<=latest_step)
    assert steps==list(range(250,latest_step+1,250)),'missing historical periodic checkpoint admission'
    assert latest_step<total,'TERMINAL_4K_BUDGET_REQUIRES_CONTEXT_CHAIN_REVIEW'
    last=None
    for step in steps:
        path=HEALTH/f'formal-native-checkpoint-state-admissions-20261115/step-{step}.json';d=read(path);pin(path)
        assert d['status']=='PASS_ACTUAL_NATIVE_CHECKPOINT_STRUCTURAL_AND_FULL_CONTROL_STATE'
        assert d['run_identity_sha256']==EXPECTED_IDENTITY
        assert d['actual_scalars']=={'dataloader.consumed_samples':step*batch,'dataloader.seq_len':seq,'lr_scheduler.last_epoch':step,'train_state.global_tokens_seen':step*tokens_per_update,'train_state.samples_consumed':step*batch,'train_state.step':step}
        assert d['adam_counter_groups']==d['optimizer_lr_groups']==164 and d['rng_ranks']==8
        assert math.isclose(d['saved_next_update_lr'],lr.expected_lr(cfg,step+1),rel_tol=1e-12)
        for p,h in d['source_inputs'].items(): assert pin(p)==h,p
        if step==latest_step:
            last=d;assert pin(Path(d['checkpoint'])/'.metadata')==d['metadata_sha256']
    limits=cfg['data']['stage_sample_limits'];cursor=last['actual_scalars']['dataloader.consumed_samples'];phase_index=bisect_right(list(accumulate(limits)),cursor-1)
    recent=validation['stages']['4096']['completed_validation_steps'][-3:];values={}
    for step in recent:
        path=HEALTH/f'all-production-nine-domain-validation-receipts-20261115/4096-{attempt["attempt_id"]}-step-{step}.json';d=read(path);pin(path)
        assert d['status']=='PASS_ACTUAL_NINE_DOMAIN_VALIDATION_AND_IMMUTABLE_EVENT_PREFIX'
        assert d['run_identity_sha256']==EXPECTED_IDENTITY
        for p,h in d['source_inputs'].items(): assert pin(p)==h,p
        copied,missing=helper.metrics([Path(v['snapshot']) for v in d['immutable_event_prefixes']],step)
        assert not missing and copied['metrics']==d['metrics']
        assert d['validation_samples_per_domain']==64 and d['validation_target_tokens_per_domain']==262144
        values[str(step)]={k:v['value'] for k,v in d['metrics'].items()}
    first,last_validation=values[str(recent[0])],values[str(recent[-1])]
    changes={tag:100*(last_validation[tag]/first[tag]-1) for tag in first}
    regressions={tag:change for tag,change in changes.items() if '/domain/' in tag and change>0}
    assert all(d['status']=='PASS_ALL_OBSERVED_LEARNING_RATES' and not d['mismatches'] for d in eta['actual_lr_contracts'])
    hardware=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,temperature.gpu,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],text=True)
    gpu=[dict(zip(['index','util_percent','memory_mib','temperature_c','uncorrected_ecc'],map(int,line.split(',')))) for line in hardware.strip().splitlines()]
    assert len(gpu)==8 and {d['index'] for d in gpu}==set(range(8))
    assert all(d['uncorrected_ecc']==0 for d in gpu),gpu
    failcnt=int(Path('/sys/fs/cgroup/memory/memory.failcnt').read_text());assert failcnt==4,('new memory allocation failures',failcnt)
    cadence=read(HEALTH/'assistant-monitoring-cadence.json');due=datetime.fromisoformat(cadence['next_review_at'])
    assert now>=due-timedelta(minutes=5),'NOT_YET_DUE'
    next_review=due+timedelta(hours=12)
    while next_review<=now: next_review+=timedelta(hours=12)
    tag=due.strftime('%Y%m%dT%H%M');out=HEALTH/f'assistant-half-day-pretrain-review-{tag}.json';report=ROOT/f'report/PRETRAIN_HALF_DAY_REVIEW_{tag}.md'
    assert not out.exists() and not report.exists(),'REVIEW_ALREADY_EXISTS_DO_NOT_OVERWRITE'
    metrics=live['training_metrics'];assert metrics['loss_is_finite'] and math.isfinite(metrics['grad_norm'])
    result={'status':'PASS_HALF_DAY_OPERATIONAL_REVIEW' if not regressions else 'HALF_DAY_REVIEW_DOMAIN_REGRESSION_REQUIRES_ATTENTION','scope':'ACTUAL_CONTROL_AND_FIXED_HELDOUT_PROGRESS_NOT_FINAL_CAPABILITY_NOT_REPEATED_FULL_TENSOR_FINITE_SCAN','reviewed_at':now.isoformat(),'supervisor_recovery_observation':recovery,'next_review_at':next_review.isoformat(),'attempt_id':attempt['attempt_id'],'training_metrics':metrics,'labels_at_last_logged_step':metrics['step']*tokens_per_update,'phase':eta['phase'],'data_cursor_phase':{'latest_native_step':latest_step,'consumed_samples':cursor,'phase_index':phase_index,'stream':cfg['data']['stages'][phase_index],'frozen_sample_limits':limits},'checkpoint_receipts_rechecked':steps,'validation_losses':values,'validation_relative_changes_percent':changes,'regressed_domains':regressions,'lr_contracts':eta['actual_lr_contracts'],'gpu':gpu,'memory_failcnt':failcnt,'projected_total_days':eta['projected_total_days_including_elapsed'],'whole_chain_remaining_days':eta['whole_chain_remaining_days'],'projected_completion_utc':eta['projected_completion_utc'],'source_inputs':sources}
    with out.open('xb') as handle:handle.write((json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode());handle.flush();os.fsync(handle.fileno())
    proof_sha=hashlib.sha256(out.read_bytes()).hexdigest();n_lr=sum(d['observations_checked'] for d in eta['actual_lr_contracts'])
    lines=['# 半日预训练运行检查','',f'时间：{now.isoformat()}，同attempt {attempt["attempt_id"]}/sourceAF54。正式训练与后台控制器运行中。','',f'最新{metrics["step"]}步/{result["labels_at_last_logged_step"]:,}labels，phase {eta["phase"]}，train loss{metrics["loss"]}，TPS{metrics["tps_per_device"]}每卡，CUDA峰值{metrics["memory_gib"]}GiB。最新原生游标映射到{Path(cfg["data"]["stages"][phase_index]).name}，数据reader和冻结配方未改。','',f'{len(steps)}份250..{latest_step}原生结构/完整控制状态验收独立重读：六计数、164Adam计数/LR、8rank RNG一致，所有source pins及最新metadata SHA通过。实际{n_lr}个LR标量匹配；本项不是重复全浮点finite扫描，首250全数值扫描及归档另有证据。','', '| 域 | '+' | '.join(str(v) for v in recent)+' |', '|---|'+'---:|'*len(recent)]
    for metric in sorted(first):
        label=metric.split('/')[-2] if '/domain/' in metric else 'aggregate';lines.append('| '+label+' | '+' | '.join(f'{values[str(s)][metric]:.6f}' for s in recent)+' |')
    lines+=['',f'固定heldout事件prefix逐一SHA及TensorBoard重读通过。最新相对首列aggregate变化{changes["validation_metrics/loss"]:+.2f}%；域回退观测：{json.dumps(regressions,ensure_ascii=False)}。这是早期点估计进展，不代表最终能力。','',f'八卡util范围{min(v["util_percent"] for v in gpu)}–{max(v["util_percent"] for v in gpu)}%，温度{min(v["temperature_c"] for v in gpu)}–{max(v["temperature_c"] for v in gpu)}°C，ECC0，memory.failcnt历史4未增。全链路966254395392labels包含4K WSD/16K/64K，90%duty预测总工期{result["projected_total_days"]:.3f}天，剩余{result["whole_chain_remaining_days"]:.3f}天，预计{result["projected_completion_utc"]}。长阶段等待真实完整4K。','',f'机器证据 `{out.relative_to(ROOT)}`，SHA256 `{proof_sha}`。下一检查{next_review.isoformat()}，期间无对话轮询。',''];report.write_text('\n'.join(lines))
    header=f'''当前运行状态（{now.isoformat()}）：**正式八卡1.5166B pretrain半日检查{result['status']}。** 最新{metrics['step']}步/loss{metrics['loss']}，累计{result['labels_at_last_logged_step']:,}labels，phase={eta['phase']}，TPS{metrics['tps_per_device']}每卡，CUDA峰值{metrics['memory_gib']}GiB。原生数据游标对应{Path(cfg['data']['stages'][phase_index]).name}。

{len(steps)}份250..{latest_step}原生结构/完整控制状态验收独立复核，{n_lr}个实际LR标量匹配，无数值/源/控制告警。最新{recent[-1]}固定九域aggregate{last_validation['validation_metrics/loss']:.9f}，相对{recent[0]}变化{changes['validation_metrics/loss']:+.2f}%；域回退观测{json.dumps(regressions,ensure_ascii=False)}。事件prefix独立SHA及重读PASS。报告{report.name}，机器证据{out.name}，SHA{proof_sha}。当前是早期heldout进展，不是最终能力完成。

八卡ECC0、显存读数详见本次机器证据，历史memory.failcnt4未增，同c5bc attempt/sourceAF54。总预算966254395392labels含4K WSD/16K/64K，90%duty保守总工期{result['projected_total_days']:.3f}天、剩余{result['whole_chain_remaining_days']:.3f}天，预计{result['projected_completion_utc']}完成，目标33天。114冻结输入coordinator仍等待完整4K，长阶段未开始；严禁热改执行源或在active run调用finalcollector。按用户每12h汇总，下次{next_review.isoformat()}。

以下为历史时点记录；以上及实时health为准。

---

'''
    current=ROOT/'report/PRETRAIN_CURRENT_GATE_STATUS_20261114.md';current.write_text(header+current.read_text())
    with (ROOT/'report/ACTIVE_PRETRAIN_OPERATIONS.md').open('a') as handle:handle.write(f'\n\n### {now.isoformat()} —12h review {result["status"]}\n\nActual{metrics["step"]}steps/{result["labels_at_last_logged_step"]:,}labels/phase{eta["phase"]}/loss{metrics["loss"]}/TPS{metrics["tps_per_device"]}. {len(steps)}nativecontrolreceipts250..{latest_step} independentlyverified;{n_lr}LRscalarsmatched;actualcursorstream{Path(cfg["data"]["stages"][phase_index]).name}. Fixedvalidation{recent},latestaggregate{last_validation["validation_metrics/loss"]:.9f},change{changes["validation_metrics/loss"]:+.2f}%,domainregressions{json.dumps(regressions)};prefixesSHAandindependentreadPASS.8GPUECC0,memoryfailcnt4,sourceAF54 unchanged. Forecast{result["projected_total_days"]:.4f}days/remaining{result["whole_chain_remaining_days"]:.4f}days includes4KWSD/16K/64K. Evidence `{out.relative_to(ROOT)}`,SHA{proof_sha};report `{report.relative_to(ROOT)}`. Next{next_review.isoformat()},no pollingbetween.\n')
    cadence.update(status='WAITING_FOR_NEXT_12_HOUR_ASSISTANT_REVIEW',updated_at=now.isoformat(),last_review_evidence=str(out),last_review_status=result['status'],next_review_at=next_review.isoformat())
    path=HEALTH/'assistant-monitoring-cadence.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(cadence,indent=2)+'\n');os.replace(tmp,path)
    print(json.dumps({'status':result['status'],'step':metrics['step'],'labels':result['labels_at_last_logged_step'],'phase':eta['phase'],'loss':metrics['loss'],'tps_per_card':metrics['tps_per_device'],'checkpoint_count':len(steps),'latest_validation_step':recent[-1],'aggregate_loss':last_validation['validation_metrics/loss'],'aggregate_relative_change_percent':changes['validation_metrics/loss'],'regressed_domains':regressions,'total_days':result['projected_total_days'],'remaining_days':result['whole_chain_remaining_days'],'next_review_at':next_review.isoformat(),'wait_ms':max(1,int((next_review-datetime.now(timezone.utc)).total_seconds()*1000)),'evidence':str(out)},ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:review()
    except Exception as error:
        print(json.dumps({'status':'HALF_DAY_REVIEW_REQUIRES_MANUAL_ATTENTION','error':repr(error),'scope':'READONLY_HELPER_FAILURE_NO_TRAINING_CONTROL_ACTION'},ensure_ascii=False),flush=True)
        raise
