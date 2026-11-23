#!/usr/bin/env python3
"""Prepare source-index windows and short replay from audited encoded documents.

Publication of a training release is a separate independent audit operation.
No tokenizer or model weights are changed by this producer.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from yuformer_midtrain.preflight import seal_bytes, digest
from yuformer_midtrain.window_release import stat_identity, edge_sha256

PHASES = ('warmup','stable','anneal')


def publish(path, value):
    seal_bytes(path,(json.dumps(value,sort_keys=True,indent=2)+'\n').encode())


def key(seed, value):
    return hashlib.blake2b((seed+'\0'+value).encode(),digest_size=16).digest()


def source_row(root, relative, dtype, role, sha=None):
    path = root/relative
    with path.open('rb') as handle:
        before = stat_identity(os.fstat(handle.fileno()))
        edge = edge_sha256(handle)
    return {'file':relative,'dtype':dtype,'tokens':before['bytes']//np.dtype(dtype).itemsize,
            'role':role,'sha256':sha or digest(path),'edge_sha256':edge,'stat':before}


def prepare(root, folder, *, scope, updates):
    root,folder = Path(root).resolve(),Path(folder).resolve()
    if not folder.is_relative_to(root) or folder.exists():
        raise ValueError('window preparation requires a new immutable folder inside the declared source root')
    gate = json.loads((root/'data/TOKENIZER_READY.json').read_bytes())
    budget_path = root/'experiments/e4_curriculum/results/effective_total_training_budget.json'
    budget = json.loads(budget_path.read_bytes())
    catalog_report_path = root/'data/health/unique-encoded-long-doc-catalog.json'
    report = json.loads(catalog_report_path.read_bytes())
    catalog = root/report['candidate_catalog']
    audit_path = root/'data/processed/audits/pretrain_1t_token_streams.json'
    audit = json.loads(audit_path.read_bytes())
    if (gate['status']!='PASS' or audit['status']!='PASS'
            or digest(audit_path)!=report['physical_audit_sha256']
            or digest(catalog)!=report['candidate_catalog_sha256']
            or report['status']!='COMPLETE_UNIQUE_ENCODED_LONG_DOC_CATALOG_NOT_RELEASE'):
        raise ValueError('window preparation needs the actual complete encoded source catalog and byte audit')
    stage_updates = {s['seq_len']:(s['updates'] if scope=='PRODUCTION' else updates) for s in budget['stages'][1:]}
    if scope not in ('QUALIFICATION','PRODUCTION') or any(type(n)is not int or n<=0 for n in stage_updates.values()):
        raise ValueError('invalid window release scope or probe budget')
    mixture_path = root/'data/health/pretrain-1t-anneal-mixture-preflight.json'
    mixture = json.loads(mixture_path.read_bytes())
    if mixture['status']!='PASS':
        raise ValueError('short replay must use the admitted high-quality anneal domain mixture')
    weights = {d:r['target_weight'] for d,r in mixture['mixture']['domains'].items()}
    if not abs(sum(weights.values())-1)<1e-8:
        raise ValueError('short replay domain weights do not sum to one')
    folder.mkdir(parents=True)
    pins = {str(p.relative_to(root)):digest(p) for p in
        (catalog_report_path,catalog,audit_path,budget_path,mixture_path,root/'data/TOKENIZER_READY.json',Path(__file__).resolve())}
    heldout = set()
    release = json.loads((root/'data/processed/releases/pretrain-1t-frozen.release.json').read_bytes())
    release_assets = {r['path']:r for r in release['artifacts']}
    for domain in sorted(weights):
        name = f'data/processed/val_sets/{domain}.parquet'
        if digest(root/name)!=release_assets[name]['sha256']:
            raise ValueError('heldout document roster changed')
        heldout.update(pq.read_table(root/name,columns=['doc_id'])['doc_id'].to_pylist())
        pins[name] = release_assets[name]['sha256']
    policy = {'schema_version':1,'scope':scope,'seed':'yuformer-long-window-v1-20261115',
        'stage_updates':{str(k):v for k,v in stage_updates.items()},'world_size':8,
        'tokens_per_update':4194304,'short_replay_fraction':.25,'short_max_encoded_doc_tokens':4096,
        'short_target_domain_weights':weights,'short_domain_absolute_tolerance':.002,
        'short_domain_relative_tolerance':.1,'long_selection':'one-per-domain-then-independent-blake2b-ranked-unique-aligned-windows',
        'allow_64k_end_window_if_needed':True,'maximum_extra_end_windows_per_document':1,
        'maximum_repeated_fraction_of_long_targets':.012,
        'long_domain_policy':'capacity-proportional-random-window-selection-with-nine-domain-coverage',
        'data_exposure_scope':'long curriculum and short replay intentionally reuse previous-stage documents; stage tokens are labels, not unique new tokens',
        'inputs':pins}
    publish(folder/'policy.json',policy)  # Fixed BEFORE any source selection/copy.
    sources = []
    for phase in PHASES:
        row = report['stages'][phase]
        stream = root/row['stream']
        if stat_identity(stream.stat())!=row['stream_stat']:
            raise ValueError('long-document source changed since its full byte audit')
        sources.append(source_row(root,row['stream'],'<u2','long_document_source',row['stream_sha256']))
    table = pq.read_table(catalog)
    docs = table.to_pylist()
    if len({r['doc_id'] for r in docs})!=len(docs) or heldout.intersection(r['doc_id'] for r in docs):
        raise ValueError('long document catalog is duplicated or overlaps heldout documents')
    plans = {}
    for seq,nsteps in stage_updates.items():
        samples=nsteps*4194304//seq
        needed=samples*3//4
        candidates=[]
        tails=[]
        for row in docs:
            n=(row['encoded_tokens_with_eod']-1)//seq
            for j in range(n):
                candidates.append((key(policy['seed'],f'{seq}:{row["doc_id"]}:{j*seq}'),
                    PHASES.index(row['parent_stage']),row['parent_token_offset']+j*seq,row['domain'],row['doc_id'],j*seq))
            tail=(row['encoded_tokens_with_eod']-1)%seq
            if seq==65536 and n and tail:
                tails.append((-tail,key(policy['seed'],row['doc_id']),PHASES.index(row['parent_stage']),
                    row['parent_token_offset']+row['encoded_tokens_with_eod']-seq-1,row['domain'],row['doc_id'],
                    row['encoded_tokens_with_eod']-seq-1))
        candidates.sort()
        chosen=[];covered=set();chosen_ids=set()
        for c in candidates:
            if c[3] not in covered:
                chosen.append(c);covered.add(c[3]);chosen_ids.add((c[4],c[5]))
        if covered!=set(weights) or needed<len(chosen):
            raise ValueError('long window capacity does not cover the frozen domain roster')
        for c in candidates:
            if len(chosen)>=needed:break
            if (c[4],c[5]) not in chosen_ids:
                chosen.append(c);chosen_ids.add((c[4],c[5]))
        repeats=0
        if len(chosen)<needed:
            if seq!=65536:raise ValueError('16K unique aligned long-window supply is insufficient')
            tails.sort()
            for negative_tail,h,source,offset,domain,doc_id,relative in tails[:needed-len(chosen)]:
                chosen.append((h,source,offset,domain,doc_id,relative));repeats+=seq+negative_tail
        if len(chosen)!=needed or repeats/(needed*seq)>policy['maximum_repeated_fraction_of_long_targets']:
            raise ValueError('long supply requires more than the frozen bounded target overlap')
        chosen.sort(key=lambda c:key(policy['seed'],f'shuffle:{seq}:{c[4]}:{c[5]}'))
        plans[seq]={'samples':samples,'long':chosen,'repeated_long_targets':repeats}
    del candidates,tails,docs,table
    # Pack short documents once, preserving every original EOD. Target-domain
    # quotas are exact up to at most one short document per domain.
    source=report['stages']['anneal']; stream=np.memmap(root/source['stream'],dtype='<u2',mode='r')
    lengths=np.memmap(root/source['lengths_file'],dtype='<u4',mode='r')
    eos=json.loads((root/source['stream'].replace('.tokens.bin','.meta.json')).read_bytes())['document_boundary']['token_id']
    packed=folder/'short-replay.tokens.bin'
    segments=[];position=0;base=0;ordinal=0
    sequence_order=list(plans);active=0;all_done=False
    counts=dict.fromkeys(weights,0)
    def stage_quotas(seq):
        total=plans[seq]['samples']//4*seq+1
        result={d:int(total*w) for d,w in weights.items()}
        result['web']+=total-sum(result.values())
        return result
    quotas=stage_quotas(sequence_order[active]);plans[sequence_order[active]]['short_start']=0
    with packed.open('xb') as output, (root/source['order_file']).open('rb') as order_handle:
        for batch in pq.ParquetFile(order_handle).iter_batches(batch_size=262144,columns=['doc_id','domain'],use_threads=False):
            values=np.asarray(lengths[ordinal:ordinal+batch.num_rows],dtype=np.uint64)
            offsets=np.cumsum(values,dtype=np.uint64)-values+base
            chunks=[]
            for i,(doc_id,domain) in enumerate(zip(batch.column(0).to_pylist(),batch.column(1).to_pylist(),strict=True)):
                length=int(values[i])
                if length>4096 or counts[domain]>=quotas[domain]:continue
                if doc_id in heldout:raise ValueError('short replay overlaps heldout documents')
                start=int(offsets[i]);part=np.asarray(stream[start:start+length])
                if int(part[-1])!=eos:
                    raise ValueError('packed short document is missing its audited EOD')
                chunks.append(part)
                segments.append((position,start,length,domain,doc_id,ordinal+i))
                position+=length;counts[domain]+=length
                if all(counts[d]>=quotas[d] for d in counts):
                    active+=1
                    if active==len(sequence_order):all_done=True;break
                    counts=dict.fromkeys(weights,0);quotas=stage_quotas(sequence_order[active])
                    plans[sequence_order[active]]['short_start']=position
            if chunks:output.write(np.concatenate(chunks).tobytes())
            ordinal+=batch.num_rows;base+=int(values.sum())
            if all_done:break
        output.flush();os.fsync(output.fileno())
    if not all_done:
        raise ValueError('short-document source cannot satisfy all frozen stage/domain quotas')
    segment_table=pa.table({'packed_start':[r[0] for r in segments],'source_start':[r[1] for r in segments],
        'encoded_tokens':[r[2] for r in segments],'domain':[r[3] for r in segments],
        'doc_id':[r[4] for r in segments],'source_document_index':[r[5] for r in segments]})
    pq.write_table(segment_table,folder/'short-replay-documents.parquet',compression='zstd')
    with (folder/'short-replay-documents.parquet').open('rb') as f:os.fsync(f.fileno())
    sources.append(source_row(root,str(packed.relative_to(root)),'<u2','packed_short_replay'))
    publish(folder/'source-byte-audit.json',{'status':'PASS_LONG_SOURCE_BYTE_BINDING','artifacts':sources,
        'complete_existing_physical_audit':{'file':str(audit_path.relative_to(root)),'sha256':digest(audit_path)},
        'new_packed_source_sha256_computed':True})
    proposals=[]
    for seq,plan in plans.items():
        short_offset=plan['short_start']
        index=np.empty((plan['samples'],3),dtype='<u8');longpos=0;shortpos=0
        for i in range(plan['samples']):
            if i%8 in (1,6):
                index[i]=(3,short_offset+shortpos*seq,1);shortpos+=1
            else:
                c=plan['long'][longpos];index[i]=(c[1],c[2],0);longpos+=1
        path=folder/f'context-{seq}.index.u64';index.tofile(path)
        with path.open('rb') as f:os.fsync(f.fileno())
        provenance=pa.table({'source_id':[r[1] for r in plan['long']], 'token_offset':[r[2] for r in plan['long']],
            'domain':[r[3] for r in plan['long']],'doc_id':[r[4] for r in plan['long']],
            'document_offset':[r[5] for r in plan['long']]})
        prov=folder/f'context-{seq}.long-documents.parquet';pq.write_table(provenance,prov,compression='zstd')
        with prov.open('rb') as f:os.fsync(f.fileno())
        proposal={'schema_version':1,'status':'PREPARED_REQUIRES_INDEPENDENT_WINDOW_AUDIT','scope':scope,
            'seq_len':seq,'samples':plan['samples'],'target_labels':plan['samples']*seq,
            'short_start':short_offset,'short_labels':shortpos*seq,'repeated_long_targets':plan['repeated_long_targets'],
            'policy':{'file':str((folder/'policy.json').relative_to(root)),'sha256':digest(folder/'policy.json')},
            'record_index':{'file':str(path.relative_to(root)),'sha256':digest(path),'bytes':path.stat().st_size,
                'dtype':'<u8','columns':['source_id','token_offset','short_replay']},
            'long_provenance':{'file':str(prov.relative_to(root)),'sha256':digest(prov)},
            'short_provenance':{'file':str((folder/'short-replay-documents.parquet').relative_to(root)),
                'sha256':digest(folder/'short-replay-documents.parquet')},
            'source_byte_audit':{'file':str((folder/'source-byte-audit.json').relative_to(root)),
                'sha256':digest(folder/'source-byte-audit.json')},'sources':sources,'tokenizer_sha256':gate['tokenizer_sha256']}
        publish(folder/f'context-{seq}.proposal.json',proposal);proposals.append(proposal)
    for name,sha in pins.items():
        if digest(root/name)!=sha:raise ValueError('preparation source changed: '+name)
    for row in sources:
        if stat_identity((root/row['file']).stat())!=row['stat']:raise ValueError('source changed during preparation')
    return {'status':'PREPARED_LONG_WINDOWS_REQUIRES_INDEPENDENT_ADMISSION','scope':scope,
            'folder':str(folder),'packed_short_tokens':position,'stages':[p['target_labels'] for p in proposals]}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--scope',choices=['QUALIFICATION','PRODUCTION'],required=True)
    parser.add_argument('--updates',type=int,default=24)
    args=parser.parse_args()
    print(json.dumps(prepare(args.root,args.folder,scope=args.scope,updates=args.updates),sort_keys=True))
