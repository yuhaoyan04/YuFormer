"""Read-only recurring structural/native state validation, independently of training."""
import datetime,hashlib,io,json,math,os,pickle,sys,time,traceback
from pathlib import Path
import torch,yaml
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torchtitan.components.optimizer import LRSchedulersContainer
root=Path('/opt/huawei/explorer-env/dataset/yyh_arcagi/YUFormer/YuFormer');run=root/'data/runs/pretrain-final-1t-8gpu';sys.path[:0]=[str(root/'midtrain/src'),str(root/'pretrain/src')]
from yuformer_midtrain.checkpoint_contract import checkpoint_scalars
from yuformer_midtrain.window_release import stat_identity
from yuformer_pretrain.lease import RunLease
config=yaml.safe_load((run/'resolved_config.yaml').read_bytes());identity=json.loads((run/'run_identity.json').read_bytes())
reference=root/'data/experiments/production_canary/production_8gpu/checkpoint/step-200/.metadata'
ref=pickle.loads(reference.read_bytes())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def expected_next_lr(step,cfg):
 optim=cfg['optim'];total=cfg['train']['expected_iters'];warm=optim['warmup_steps'];decay=optim['decay_steps'] if optim['lr_schedule']=='wsd' else total-warm;stable_end=total+1-decay
 if step<warm:factor=(step+1)/warm
 elif step<stable_end:factor=1.
 else:factor=optim['min_lr_ratio']+(1-optim['min_lr_ratio'])*.5*(1+math.cos(math.pi*(step+1-stable_end)/decay))
 return optim['lr']*factor
# Compare the independent state observer against the actual pinned scheduler at
# every boundary, including its virtual terminal step, without touching training.
math_proof=[]
for cfg in [config,{'train':{'expected_iters':24},'optim':{'lr_schedule':'wsd','lr':1e-4,'warmup_steps':8,'decay_steps':8,'min_lr_ratio':.1}}]:
 total=cfg['train']['expected_iters'];o=cfg['optim'];optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))],lr=o['lr'])
 sched=LRSchedulersContainer.Config(total_steps=total,warmup_steps=o['warmup_steps'],decay_ratio=o['decay_steps']/total,decay_type='cosine',min_lr_factor=o['min_lr_ratio']).build(optimizers=[optimizer],training_steps=total)
 points=sorted({0,1,o['warmup_steps']-1,o['warmup_steps'],total-o['decay_steps']-1,total-o['decay_steps'],total-o['decay_steps']+1,total-1,total})
 for step in points:
  sched.load_state_dict({'last_epoch':step});actual=list(sched.get_metrics().values())[0];expected=expected_next_lr(step,cfg)
  if not math.isclose(actual,expected,rel_tol=1e-12,abs_tol=1e-20):raise ValueError('checkpoint LR observer differs from actual pinned scheduler')
  math_proof.append({'total_steps':total,'checkpoint_step':step,'actual_next_lr':actual,'expected_next_lr':expected})
inputs=[Path(__file__),run/'run_identity.json',run/'resolved_config.yaml',reference,root/'midtrain/src/yuformer_midtrain/checkpoint_contract.py',root/'midtrain/src/yuformer_midtrain/window_release.py']
pins={str(p):digest(p) for p in inputs};out=root/'data/health/formal-native-checkpoint-state-live-20261115.json';history=root/'data/health/formal-native-checkpoint-state-admissions-20261115';history.mkdir(exist_ok=True)
lease=RunLease(root/'data/health/formal-native-checkpoint-state-observer','readonly-native-state-observer').acquire()
state={'status':'WAITING_COMPLETE_NATIVE_CHECKPOINT','observational_only':True,'pid':os.getpid(),'scope':'ALL_SHARD_BOUNDS_TENSOR_GEOMETRY_SCALARS_ADAM_LR_COUNTERS_RNG_NOT_REPEATED_FULL_TENSOR_FINITE_SCAN','source_inputs':pins,'lr_boundary_proof':math_proof,'checked_steps':[],'alerts':[]}
def publish():
 state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(state,sort_keys=True,indent=2)+'\n');os.replace(tmp,out)
def payload(metadata,path,name):
 matches=[info for index,info in metadata.storage_data.items() if index.fqn==name]
 if len(matches)!=1:raise ValueError('native control payload roster changed: '+name)
 info=matches[0]
 with (path/info.relative_path).open('rb') as h:h.seek(info.offset);return torch.load(io.BytesIO(h.read(info.length)),map_location='cpu',weights_only=True)
def check(path):
 step=int(path.name[5:]);raw=(path/'.metadata').read_bytes();sha=hashlib.sha256(raw).hexdigest();m=pickle.loads(raw)
 if set(m.state_dict_metadata)!=set(ref.state_dict_metadata):raise ValueError('native key roster differs from qualified model')
 files={'.metadata'};stats={p.name:stat_identity(p.stat()) for p in path.iterdir() if p.is_file()}
 for index,info in m.storage_data.items():
  name=str(info.relative_path)
  if Path(name).name!=name or info.offset<0 or info.length<=0 or name not in stats or info.offset+info.length>stats[name]['bytes'] or getattr(info,'transform_descriptors',None):raise ValueError('native storage shard is missing/truncated/unsupported')
  files.add(name);actual=m.state_dict_metadata[index.fqn];expected=ref.state_dict_metadata[index.fqn]
  if isinstance(actual,TensorStorageMetadata) and (not isinstance(expected,TensorStorageMetadata) or actual.properties.dtype!=expected.properties.dtype or tuple(actual.size)!=tuple(expected.size)):raise ValueError('native tensor global geometry changed')
 if files!=set(stats):raise ValueError('native file roster has unreferenced/incomplete artifacts')
 values=checkpoint_scalars(path,sha);samples=step*8*config['train']['micro_batch_size']*config['train']['grad_accum']
 expected={'train_state.step':step,'lr_scheduler.last_epoch':step,'dataloader.seq_len':4096,'train_state.samples_consumed':samples,'dataloader.consumed_samples':samples,'train_state.global_tokens_seen':samples*4096}
 if values!=expected:raise ValueError('native checkpoint trainer/scheduler/global labels/cursor diverged')
 lrs=[name for name in m.state_dict_metadata if name.startswith('optimizer.param_groups.') and name.endswith('.lr')];steps=[name for name in m.state_dict_metadata if name.startswith('optimizer.state.') and name.endswith('.step')]
 if not lrs or not steps or any(not math.isclose(float(payload(m,path,name)),expected_next_lr(step,config),rel_tol=1e-12,abs_tol=1e-20) for name in lrs):raise ValueError('native Adam LR does not match actual next scheduler update')
 if any(float(payload(m,path,name))!=step for name in steps):raise ValueError('native Adam update counter differs from trainer')
 rng=pickle.loads(payload(m,path,'train_state.rng_payload'))
 if rng['world_size']!=8 or len(rng['states'])!=8 or any(set(r)!={'python','numpy','torch_cpu','torch_cuda'} or any(not isinstance(r[k],torch.Tensor) or r[k].dtype!=torch.uint8 or not r[k].numel() for k in ['torch_cpu','torch_cuda']) for r in rng['states']):raise ValueError('all-rank RNG state is incomplete')
 if digest(path/'.metadata')!=sha or any(stat_identity((path/name).stat())!=value for name,value in stats.items()):raise ValueError('native files changed during inspection')
 return {'status':'PASS_ACTUAL_NATIVE_CHECKPOINT_STRUCTURAL_AND_FULL_CONTROL_STATE','step':step,'metadata_sha256':sha,'checkpoint':str(path),'files':stats,'actual_scalars':values,'optimizer_lr_groups':len(lrs),'adam_counter_groups':len(steps),'saved_next_update_lr':expected_next_lr(step,config),'rng_ranks':8,'run_identity_sha256':digest(run/'run_identity.json'),'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_inputs':pins,'scope':state['scope']}
try:
 publish()
 while True:
  lease.assert_owner()
  if any(digest(p)!=sha for p,sha in pins.items()):raise ValueError('observer source/qualified reference/run configuration changed')
  paths=sorted((run/'checkpoint').glob('step-*/.metadata'),key=lambda p:int(p.parent.name[5:]))
  for metadata in paths:
   step=int(metadata.parent.name[5:])
   if step in state['checked_steps']:continue
   result=check(metadata.parent);target=history/f'step-{step}.json'
   if target.exists():
    old=json.loads(target.read_bytes())
    if old['metadata_sha256']!=result['metadata_sha256']:raise ValueError('previously admitted checkpoint metadata was overwritten')
   else:
    with target.open('x') as h:h.write(json.dumps(result,sort_keys=True,indent=2)+'\n');h.flush();os.fsync(h.fileno())
   state.update(status='PASS_LATEST_COMPLETED_NATIVE_CHECKPOINT_CONTROL_STATE',latest_complete_step=step,latest_admission=str(target),last_control_state=result['actual_scalars']);state['checked_steps'].append(step);publish();print('PASS_NATIVE_STATE',step,flush=True)
  publish();time.sleep(30)
except Exception as exc:state.update(status='FAIL_CLOSED_NATIVE_CHECKPOINT_STATE_OBSERVER',error=repr(exc),traceback=traceback.format_exc(),alerts=['NATIVE_CHECKPOINT_STATE_REQUIRES_REVIEW']);publish();raise
finally:lease.release()
