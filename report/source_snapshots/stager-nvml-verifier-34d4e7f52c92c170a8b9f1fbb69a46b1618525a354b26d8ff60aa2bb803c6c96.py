import json,hashlib,subprocess
from pathlib import Path
root=Path('/opt/huawei/explorer-env/dataset/yyh_arcagi/YUFormer/YuFormer');path=root/'data/health/rank-bound-stager-device-probe-v3/report.json';report=json.loads(path.read_bytes())
raw=subprocess.run(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
uuid_to_rank={line.split(',')[1].strip():int(line.split(',')[0]) for line in raw.splitlines()}
def parse(text):return [(uuid_to_rank[x.strip()],int(y.strip()),int(z.strip())) for x,y,z in (line.split(',') for line in text.splitlines())]
result={}
for mode,body in report['modes'].items():
 before=parse(body['nvml_before_staging']);after=parse(body['nvml_after_staging'])
 assert len(before)==8 and {x[0] for x in before}==set(range(8))
 host_to_rank={pid:rank for rank,pid,_ in before}
 assert len(host_to_rank)==8
 contexts={r:[] for r in range(8)}
 for gpu,pid,mem in after:
  assert pid in host_to_rank
  contexts[host_to_rank[pid]].append({'gpu':gpu,'memory_mib':mem,'nvml_host_pid':pid})
 if mode=='rank_bound':assert all(len(ds)==1 and ds[0]['gpu']==rank for rank,ds in contexts.items())
 assert all(w['roundtrip']=='PASS' and w['actual_copy_device']==w['rank'] for w in body['workers'])
 result[mode]={'contexts_by_rank':contexts,'total_contexts':len(after),'extra_gpu0_contexts':sum(gpu==0 and host_to_rank[pid]!=0 for gpu,pid,_ in after)}
assert result['original']['extra_gpu0_contexts']==7 and result['rank_bound']['extra_gpu0_contexts']==0
out={'status':'PASS_INDEPENDENT_NVML_NAMESPACE_MAPPING_15_TO_8_CONTEXTS_AND_ALL8_EXACT_ROUNDTRIP',
 'pid_namespace_note':'nvidia-smi returns host PIDs; worker os.getpid returns container PIDs. Match through the unique one-context-per-rank baseline.',
 'observations':result,'source_inputs':{str(path):hashlib.sha256(path.read_bytes()).hexdigest(),str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
(root/'data/health/rank-bound-stager-nvml-admission.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n');print(out['status'])
