import argparse,json,os,sys,time,subprocess,hashlib
from pathlib import Path
ROOT=Path('/opt/huawei/explorer-env/dataset/yyh_arcagi/YUFormer/YuFormer')
OUT=ROOT/'data/health/rank-bound-stager-device-probe-v3'
parser=argparse.ArgumentParser();parser.add_argument('--worker',type=int);parser.add_argument('--mode');args=parser.parse_args()
if args.worker is not None:
 import torch
 torch.cuda.set_device(args.worker)
 from torch.distributed.checkpoint.staging import DefaultStager,StagingOptions
 sys.path[:0]=[str(ROOT/'model/src'),str(ROOT/'midtrain/src'),str(ROOT/'pretrain/backends/torchtitan')]
 from yuformer_midtrain.checkpoint import RankBoundStager
 options=StagingOptions(use_async_staging=True,use_non_blocking_copy=True)
 data=torch.arange(4096,device=f'cuda:{args.worker}',dtype=torch.float32)
 directory=OUT/args.mode
 (directory/f'ready{args.worker}.json').write_text(json.dumps({'rank':args.worker,'pid':os.getpid()}))
 while not (directory/'start').exists():time.sleep(.1)
 stager=DefaultStager(options) if args.mode=='original' else RankBoundStager(options,device=args.worker)
 actual_stage=stager._state_dict_stager.stage
 recorded=[]
 def record_device(*a,**k):
  recorded.append(torch.cuda.current_device())
  return actual_stage(*a,**k)
 stager._state_dict_stager.stage=record_device
 staged=stager.stage({'tensor':data}).result(timeout=60)
 assert recorded==[args.worker]

 assert staged['tensor'].device.type=='cpu' and torch.equal(staged['tensor'],torch.arange(4096,dtype=torch.float32))
 directory=OUT/args.mode
 (directory/f'rank{args.worker}.json').write_text(json.dumps({'rank':args.worker,'pid':os.getpid(),'roundtrip':'PASS','actual_copy_device':recorded[0]}))
 while not (directory/'release').exists():time.sleep(.1)
 stager.close();raise SystemExit(0)
sys.path.insert(0,str(ROOT/'experiments/e2_architecture/scripts'));import benchmark_model_sizes as bench
bench.require_idle()
if OUT.exists():raise ValueError('preserve prior probe')
OUT.mkdir(parents=True)
report={'status':'RUNNING_ACTUAL_ALL8_STAGER_DEVICE_PROBE','modes':{},'production_qualified':False}
for mode in ['original','rank_bound']:
 directory=OUT/mode;directory.mkdir();processes=[];handles=[]
 try:
  for rank in range(8):
   log=(directory/f'rank{rank}.log').open('w');handles.append(log)
   env=dict(os.environ,PYTHONPATH=str(ROOT/'model/src')+':'+str(ROOT/'midtrain/src')+':'+str(ROOT/'pretrain/backends/torchtitan'),CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7')
   processes.append(subprocess.Popen([sys.executable,__file__,'--worker',str(rank),'--mode',mode],env=env,stdout=log,stderr=subprocess.STDOUT))
  deadline=time.monotonic()+180
  while len(list(directory.glob('ready*.json')))<8:
   if any(p.poll() is not None for p in processes) or time.monotonic()>deadline:raise ValueError('stager worker initialization failed')
   time.sleep(.5)
  before=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
  (directory/'start').touch()
  while len(list(directory.glob('rank*.json')))<8:
   if any(p.poll() is not None for p in processes) or time.monotonic()>deadline:raise ValueError('stager worker failed/timeout; see logs')
   time.sleep(.5)
  rows=[json.loads((directory/f'rank{rank}.json').read_bytes()) for rank in range(8)]
  uuids=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
  gpu_ids={line.split(',')[1].strip():int(line.split(',')[0]) for line in uuids.splitlines()}
  raw=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
  ownership={r['pid']:[] for r in rows}
  for line in raw.splitlines():
   uuid,pid,memory=[s.strip() for s in line.split(',')]
   if int(pid) in ownership:ownership[int(pid)].append({'gpu':gpu_ids[uuid],'memory_mib':memory})
  report['modes'][mode]={'roundtrip':'PASS_ALL8','workers':rows,'actual_device_contexts':ownership,'nvml_before_staging':before,'nvml_after_staging':raw}
  (OUT/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
  # Existing backend imports may already own a CUDA0 context; retain both
  # NVML snapshots instead of attributing all foreign contexts to staging.
 finally:
  (directory/'release').touch()
  for p in processes:
   try:p.wait(timeout=15)
   except subprocess.TimeoutExpired:p.kill();p.wait(timeout=15)
  for h in handles:h.close()
report['status']='PASS_ACTUAL_ALL8_RANK_BOUND_STAGING_AND_EXACT_CPU_ROUNDTRIP'
report['source_inputs']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'midtrain/src/yuformer_midtrain/checkpoint.py']}
(OUT/'report.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps(report,sort_keys=True))
