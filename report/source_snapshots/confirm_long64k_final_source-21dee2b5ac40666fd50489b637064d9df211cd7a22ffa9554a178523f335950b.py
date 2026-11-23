from pathlib import Path
import json,os,sys,time,yaml
root=Path('/opt/huawei/explorer-env/dataset/yyh_arcagi/YUFormer/YuFormer')
sys.path.insert(0,str(root/'experiments/e2_architecture/scripts'));sys.path.insert(0,str(root/'pretrain/src'))
import benchmark_model_sizes as bench
from analyze_model_size_sweep import memory_pressure
from yuformer_pretrain.lease import RunLease
bench.CONFIGS=root/'experiments/e2_architecture/configs/long_context_final_source_confirm_20261114';bench.RESULTS=root/'experiments/e2_architecture/results/long_context_final_source_confirm_20261114';bench.RUNS=root/'data/experiments/long_context_final_source_confirm_20261114';bench.HEALTH=root/'data/health/long64k-final-source-confirmation.json'
bench.CONFIGS.mkdir(parents=True,exist_ok=True);bench.RESULTS.mkdir(parents=True,exist_ok=True)
state=dict(pid=os.getpid(),status='WAITING_ACTUAL_DIAGNOSTIC_PASS',production_qualified=False,long_capability_qualified=False,purpose='CONFIRM_ACTUAL_64K_SUPPORTED_CHECKPOINT_CONFIG',cases={})
bench.publish(bench.HEALTH,state)
with RunLease(bench.RESULTS,'confirm-supported-64k-checkpoint-config'):
 try:
  deadline=time.monotonic()+600
  while True:
   parent=json.loads((root/'data/health/final-source-checkpoint-diagnostics.json').read_bytes())
   case=parent.get('cases',{}).get('context64k_mlp_ac_gc')
   if case:
    if case['evaluation']['status']!='PASS_SCREENING':raise ValueError('actual64kMLPdiagnosticdidnotpass')
    break
   if time.monotonic()>=deadline:raise TimeoutError('waitingforactual64kMLPdiagnostic')
   time.sleep(5)
  source=root/'experiments/e2_architecture/configs/final_source_checkpoint_diagnostics_20261114/context64k_mlp_ac_gc.yaml';cfg=yaml.safe_load(source.read_bytes());name='context64k_mlp_gc_chunks32_24'
  assert cfg['data']['seq_len']==65536 and cfg['train']['activation_checkpointing'] is False and cfg['loss']['num_chunks']==32
  model=yaml.safe_load(Path(cfg['model']['configs']).read_bytes())['presets'][cfg['model']['preset']];assert model['selective_recompute']=='mlp'
  mp=bench.CONFIGS/(name+'.model.yaml');mp.write_text(yaml.safe_dump({'presets':{'supported-64k':model}},sort_keys=True));cfg['model']={'preset':'supported-64k','configs':str(mp)};cfg['run_name']=name
  cfg['train'].update(expected_iters=24,budget_tokens=24*bench.TOKENS_PER_STEP,eval_every=24,save_every=24,keep_last_n=2);cfg['optim']['warmup_steps']=4
  cp=bench.CONFIGS/(name+'.yaml');cp.write_text(yaml.safe_dump(cfg,sort_keys=False));env=dict(line.split('=',1) for line in source.with_suffix('.env').read_text().splitlines() if line.strip() and not line.lstrip().startswith('#'));env.update(YUFORMER_CONFIG=str(cp),YUFORMER_MODEL_CONFIGS=str(mp),YUFORMER_PRETRAIN_OUT=str(bench.RUNS/name),MASTER_PORT='29883')
  ep=cp.with_suffix('.env');ep.write_text(''.join(f'{k}={v}\n' for k,v in env.items()));candidate=dict(name=name,parameters=1516600064,model_path=str(mp),model_sha256=bench.digest(mp));variant=dict(name=name,config=str(cp),env=str(ep),config_sha256=bench.digest(cp),env_sha256=bench.digest(ep),microbatch=1,grad_accum=8)
  state.update(status='RUNNING',source_qualification_identity=bench.qualification_identity(root),wrapper_tool_sha256=bench.digest(__file__))
  result=bench.run_variant(candidate,variant,state);result.update(memory_pressure(result['log'],skip_steps=8,final_step=25));state['cases'][name]=result
  import re
  raw_log=Path(result['log']).read_text(errors='replace')
  if re.search(r'\b[1-9]\d* CUDA memory allocation retries',raw_log) or 'memory allocation failed with OOM' in raw_log:raise ValueError('actual allocator retry observed; no long64k admission')
  if result['evaluation']['status']!='PASS_SCREENING' or result['requires_smaller_batch_comparison']:raise ValueError('actual24step64kconfirmationfailed')
  if bench.digest(__file__)!=state['wrapper_tool_sha256']:raise ValueError('confirmationsourcechanged')
  state.update(status='PASS_64K_SYSTEM_CONFIRMATION_NOT_CAPABILITY',updated_at=bench.now())
 except Exception as error:state.update(status='FAIL_CLOSED_64K_CONFIRMATION',error=repr(error),updated_at=bench.now());raise
 finally:bench.publish(bench.HEALTH,state)
