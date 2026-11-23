"""Warm exact admitted future blocks using readonly file descriptors, outside training."""
import bisect,datetime,hashlib,json,os,sys,time,traceback
from pathlib import Path
import numpy as np,yaml
root=Path('/opt/huawei/explorer-env/dataset/yyh_arcagi/YUFormer/YuFormer');run=root/'data/runs/pretrain-final-1t-8gpu'
sys.path[:0]=[str(root/'pretrain/backends/torchtitan'),str(root/'model/src')]
from yuformer_ext.data import PretokenizedDataLoader,_file_identity
from yuformer_ext.block_permutation import map_sample
config=yaml.safe_load((run/'resolved_config.yaml').read_bytes());data=config['data']
keys=['stages','meta','stage_sample_limits','tokenizer_gate','release_manifest','epochs','sampling_policy','block_permutation_manifest']
reader=PretokenizedDataLoader(PretokenizedDataLoader.Config(**{k:data[k] for k in keys}),dp_world_size=8,dp_rank=0,tokenizer=None,seq_len=data['seq_len'],local_batch_size=config['train']['micro_batch_size'])
if data['sampling_policy']!='block_permutation' or reader._block_permutation['block_samples']!=8 or data['seq_len']!=4096:raise ValueError('prefetch only supports this exact qualified4K/B8order')
fds=[os.open(p,os.O_RDONLY) for p in data['stages']];stats=[_file_identity(os.fstat(fd)) for fd in fds]
paths=[Path(__file__),run/'run_identity.json',run/'resolved_config.yaml',Path(data['block_permutation_manifest']),*[Path(p) for p in data['meta']],*[root/'pretrain/backends/torchtitan/yuformer_ext'/name for name in ['data.py','block_permutation.py','stream_math.py']]]
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={str(p):digest(p) for p in paths};total=sum(reader._stage_samples);bs=8;seq=4096

def guard():
 if any(digest(p)!=sha for p,sha in pins.items()):raise ValueError('frozen runtime/order/config changed')
 for fd,path,expected in zip(fds,data['stages'],stats,strict=True):
  if _file_identity(os.fstat(fd))!=expected or _file_identity(Path(path).stat())!=expected:raise ValueError('frozen token source changed')

def read_block(global_sample):
 if global_sample%bs or not 0<=global_sample<total:raise ValueError('prefetch needs exact global B8block alignment')
 stage=bisect.bisect_right(reader._ends,global_sample);base=0 if stage==0 else reader._ends[stage-1]
 local=map_sample(global_sample-base,reader._block_permutation['arrays'][stage],bs)
 if local%bs:raise ValueError('mapped physical block is not B8aligned')
 dtype=reader._arrays[stage].dtype;length=(bs*seq+1)*dtype.itemsize;offset=local*seq*dtype.itemsize
 raw=os.pread(fds[stage],length,offset)
 if len(raw)!=length:raise ValueError('mapped future block exceeds exact admitted S+1source bounds')
 return stage,np.frombuffer(raw,dtype=dtype),raw

proof=[]
starts=[0,*reader._ends[:-1]]
for start,end in zip(starts,reader._ends,strict=True):
 for global_sample in [start,end-bs]:
  stage,values,raw=read_block(global_sample)
  for item in [0,bs-1]:
   x,y,actual_stage=reader._sample(global_sample+item);left=item*seq
   if actual_stage!=stage or not np.array_equal(x,values[left:left+seq]) or not np.array_equal(y,values[left+1:left+seq+1]):raise ValueError('readonly prefetch bytes differ from actual qualified runtime samples')
  proof.append({'global_block_first_sample':global_sample,'stage':stage,'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)})
if reader._step!=0:raise ValueError('verification unexpectedly advanced the independent reader cursor')
guard();proof_file=root/'data/health/readonly-future-block-prefetch-proof-20261115.json'
proof_file.write_text(json.dumps({'status':'PASS_READONLY_BLOCKS_MATCH_ACTUAL_RUNTIME_AT_ALL_THREE_PHASE_BOUNDARIES','scope':'NO_TRAINING_STATE_OR_DATA_WRITES','source_inputs':pins,'blocks':proof,'independent_reader_step':reader._step},sort_keys=True,indent=2)+'\n')
if '--verify-only' in sys.argv:print(json.dumps({'status':'PASS','blocks':len(proof)},sort_keys=True));raise SystemExit(0)
health=root/'data/health/readonly-future-block-prefetch-live-20261115.json'
state={'status':'RUNNING_READONLY_PAGECACHE_PREFETCH','pid':os.getpid(),'source_inputs':pins,'proof_file':str(proof_file),'lookahead_updates':128,'maximum_read_mib_per_second':20,'training_or_dataset_writes':False,'cache_side_effect':'OS page cache only','bytes_read':0,'blocks_read':0}
def publish():
 state['updated_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=health.with_suffix('.tmp');tmp.write_text(json.dumps(state,sort_keys=True,indent=2)+'\n');os.replace(tmp,health)
seen_until=0;attempt=None;global_batch=8*config['train']['micro_batch_size']*config['train']['grad_accum']
try:
 while True:
  guard();a=json.loads((run/'current_attempt.json').read_bytes());live=json.loads((root/'data/health/authorized-pretrain-live-monitor.json').read_bytes())
  if live.get('status')=='BLOCKED':raise ValueError('formal run failed; prefetch stopped')
  if live.get('stage')!='full_pretrain' or live.get('alive') is not True:state['status']='WAITING_CURRENT_FORMAL_PROCESS';publish();time.sleep(30);continue
  if a['attempt_id']!=attempt:attempt=a['attempt_id'];seen_until=0
  step=(live.get('training_metrics')or{}).get('step',0);left=step*global_batch;right=min(total,(step+128)*global_batch)
  cursor=max(left,seen_until);started=time.monotonic();batch_bytes=0
  state.update(status='RUNNING_READONLY_PAGECACHE_PREFETCH',attempt_id=attempt,observed_step=step,target_global_sample_exclusive=right);publish()
  while cursor<right:
   _,_,raw=read_block(cursor);batch_bytes+=len(raw);state['bytes_read']+=len(raw);state['blocks_read']+=1;cursor+=bs
   delay=batch_bytes/(20*1024*1024)-(time.monotonic()-started)
   if delay>0:time.sleep(min(delay,.1))
   if state['blocks_read']%1024==0:state['prefetched_global_sample_exclusive']=cursor;publish()
  seen_until=right;state['prefetched_global_sample_exclusive']=right;guard();publish();time.sleep(10)
except Exception as exc:state.update(status='STOPPED_PREFETCH_FAIL_CLOSED_TRAINING_UNCHANGED',error=repr(exc),traceback=traceback.format_exc());publish();raise
finally:
 for fd in fds:os.close(fd)
