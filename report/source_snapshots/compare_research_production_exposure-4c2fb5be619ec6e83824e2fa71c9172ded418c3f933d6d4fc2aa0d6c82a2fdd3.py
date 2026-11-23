import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path('/opt/huawei/explorer-env/dataset/yyh_arcagi/YUFormer/YuFormer')
OUTPUT = ROOT/'data/health/research-vs-production-prefix-exposure.json'
DOMAINS = ['academic','books','chinese','code','encyclopedia','math','multilingual','qa','web']
LABELS = 838860800
END = LABELS+1


def sha(path):
    result=hashlib.sha256()
    with path.open('rb') as handle:
        while chunk:=handle.read(8<<20):result.update(chunk)
    return result.hexdigest()


def identity(path):
    s=path.stat()
    return dict(bytes=s.st_size,device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns)


def main():
    meta_path=ROOT/'data/token_streams/m3_20261020_yf64k/mix_m3_e1_100b_stage2.meta.json'
    meta=json.loads(meta_path.read_bytes())
    prior_path=ROOT/'data/processed/audits/formal_token_stream_release_mix_m3_e1_100b_m3_20261020.json'
    prior=json.loads(prior_path.read_bytes())
    old=next(row for row in prior['streams'] if row['name']==meta['name'])
    if (prior['status']!='PASS' or old['status']!='PASS' or not old['checks']['eod_count_exact']
            or old['actual_sha256']!=meta['sha256']):raise ValueError('prior whole research stream/EOD proof is required')
    binary=meta_path.with_name(meta['name']+'.tokens.bin')
    order=ROOT/meta['source'].removeprefix('s6:')
    protected=[meta_path,prior_path,binary,order]
    before={str(p):identity(p) for p in protected}
    if sha(order)!=meta['order_sha256']:raise ValueError('research document order changed')
    tokens=np.memmap(binary,mode='r',dtype='<u2')
    ends=[]
    prefix_hash=hashlib.sha256()
    for start in range(0,END,4<<20):
        values=tokens[start:min(start+(4<<20),END)]
        prefix_hash.update(values.tobytes())
        ends.extend((np.flatnonzero(values==0).astype(np.int64)+start+1).tolist())
    boundaries=[0,*ends]
    partial=boundaries[-1]<END
    if partial:boundaries.append(END)
    lengths=np.diff(np.asarray(boundaries,dtype=np.int64))
    lengths[0]-=1
    if lengths.sum()!=LABELS:raise ValueError('research target/EOD accounting did not close')
    counts={domain:0 for domain in DOMAINS}
    cursor=0
    for batch in pq.ParquetFile(order).iter_batches(batch_size=131072,columns=['position','domain']):
        n=min(batch.num_rows,len(lengths)-cursor)
        if not n:break
        row=batch.slice(0,n)
        positions=row.column(0).to_numpy()
        if not np.array_equal(positions,np.arange(cursor,cursor+n)):raise ValueError('research source positions are not sequential')
        names=row.column(1).to_pylist()
        for domain in DOMAINS:
            indices=np.fromiter((name==domain for name in names),dtype=np.bool_,count=n)
            counts[domain]+=int(lengths[cursor:cursor+n][indices].sum())
        cursor+=n
    if cursor!=len(lengths) or sum(counts.values())!=LABELS:raise ValueError('research prefix domain membership incomplete')
    diagnostic_path=ROOT/'data/health/warmup-training-block-permutation.json'
    diagnostic=json.loads(diagnostic_path.read_bytes())
    master_path=ROOT/'data/processed/training_permutations/pretrain-1t-dd284efd7c715544-b8/manifest.json'
    master=json.loads(master_path.read_bytes())
    row=master['stages'][0]
    array=master_path.parent/row['permutation']['file']
    matrix=ROOT/diagnostic['block_domain_tokens_file']
    # Matrix columns use the auditor's explicit order, never an inferred alphabetical order.
    names=diagnostic.get('domains')
    if names is None:
        raise ValueError('actual block-domain matrix must declare its column order')
    if sha(array)!=row['permutation']['sha256'] or sha(matrix)!=diagnostic['block_domain_tokens_sha256']:
        raise ValueError('actual production order/domain matrix changed')
    order_values=np.load(array,mmap_mode='r')
    domain_tokens=np.load(matrix,mmap_mode='r')
    samples=LABELS//4096
    if samples%master['block_samples']:raise ValueError('comparison needs whole actual causal blocks')
    selected=order_values[:samples//master['block_samples']]
    sums=domain_tokens[selected].sum(axis=0,dtype=np.uint64)
    current={name:int(sums[i]) for i,name in enumerate(names)}
    if sum(current.values())!=LABELS:raise ValueError('production prefix target accounting did not close')
    if any(identity(p)!=before[str(p)] for p in protected):raise ValueError('research source changed while auditing')
    report=dict(status='PASS_READONLY_EXPOSURE_COMPARISON',target_tokens_per_case=LABELS,
        purpose='DIAGNOSTIC_DIFFERENT_RESEARCH_AND_PRODUCTION_DATA_DISTRIBUTIONS_NOT_QUALITY_ADMISSION',
        research=dict(domain_target_tokens=counts,domain_fractions={d:counts[d]/LABELS for d in DOMAINS},
            whole_stream_sha256_from_prior_audit=meta['sha256'],actual_prefix_including_context_sha256=prefix_hash.hexdigest(),
            completed_documents=len(ends),partial_last_document=partial,target_interval=[1,END],source_stat=before),
        production=dict(domain_target_tokens=current,domain_fractions={d:current[d]/LABELS for d in DOMAINS},
            master_sha256=sha(master_path),permutation_sha256=sha(array),block_domain_tokens_sha256=sha(matrix),
            logical_sample_interval=[0,samples],physical_source_sample_offset=row['source_sample_offset']),
        source_inputs={str(p.relative_to(ROOT)):sha(p) for p in [meta_path,prior_path,diagnostic_path,master_path]},
        producer_sha256=sha(Path(__file__)),original_files_modified=False,production_qualified=False)
    OUTPUT.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],research=report['research']['domain_fractions'],production=report['production']['domain_fractions'])),flush=True)


if __name__=='__main__':main()
