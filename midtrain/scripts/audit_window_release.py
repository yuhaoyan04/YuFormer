#!/usr/bin/env python3
"""Independently admit every long window and every copied short document."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from yuformer_midtrain.preflight import digest, seal_bytes
from yuformer_midtrain.window_release import bound_json, stat_identity, edge_sha256, CHECKS, dataset_id, load_window_release

PHASES=('warmup','stable','anneal')


def publish(path,value):
    seal_bytes(path,(json.dumps(value,sort_keys=True,indent=2)+'\n').encode())


def audit(root, folder):
    root,folder=Path(root).resolve(),Path(folder).resolve()
    policy_path=folder/'policy.json';policy=json.loads(policy_path.read_bytes())
    pins={str(policy_path.relative_to(root)):digest(policy_path),
          str(Path(__file__).resolve().relative_to(root)):digest(__file__)}
    for name,sha in policy['inputs'].items():
        if digest(root/name)!=sha:raise ValueError('frozen window preparation input changed: '+name)
        pins[name]=sha
    if policy['scope'] not in ('PRODUCTION','QUALIFICATION'):
        raise ValueError('long-window scope must be explicit')
    catalog_report=json.loads((root/'data/health/unique-encoded-long-doc-catalog.json').read_bytes())
    catalog=pq.read_table(root/catalog_report['candidate_catalog']).to_pylist()
    lookup={r['doc_id']:r for r in catalog}
    if len(lookup)!=len(catalog):raise ValueError('catalog document IDs are not unique')
    del catalog
    heldout=set()
    for name in policy['inputs']:
        if name.startswith('data/processed/val_sets/') and name.endswith('.parquet'):
            heldout.update(pq.read_table(root/name,columns=['doc_id'])['doc_id'].to_pylist())
    if not heldout:raise ValueError('independent heldout ID roster is absent')
    physical=json.loads((root/'data/processed/audits/pretrain_1t_token_streams.json').read_bytes())
    audited={r['name']:r for r in physical['streams']}
    source_audit=json.loads((folder/'source-byte-audit.json').read_bytes())
    sources=source_audit['artifacts'];arrays=[]
    if source_audit['complete_existing_physical_audit']['sha256']!=digest(root/'data/processed/audits/pretrain_1t_token_streams.json'):
        raise ValueError('source-byte projection does not bind the actual complete physical audit')
    for i,row in enumerate(sources):
        path=root/row['file']
        with path.open('rb') as f:
            if stat_identity(os.fstat(f.fileno()))!=row['stat'] or edge_sha256(f)!=row['edge_sha256']:
                raise ValueError('window source bytes/stat changed')
        if i<3:
            actual=audited[PHASES[i]]
            if (row['sha256']!=actual['actual_sha256'] or row['stat']!={k:actual['actual_'+k] for k in ('bytes','device','inode','mtime_ns')}
                    or row['file']!=str(Path(actual['artifacts']['stream']).relative_to(root))):
                raise ValueError('long source does not equal an actual fully audited physical stream')
        elif i!=3 or digest(path)!=row['sha256']:
            raise ValueError('packed replay full byte checksum differs')
        arrays.append(np.memmap(path,mode='r',dtype=row['dtype']))
    if len(sources)!=4:raise ValueError('exact three long sources plus one packed short source are required')
    segment_path=folder/'short-replay-documents.parquet'
    segments=pq.read_table(segment_path).to_pylist()
    starts=np.array([r['packed_start'] for r in segments],dtype=np.uint64)
    lengths=np.array([r['encoded_tokens'] for r in segments],dtype=np.uint64)
    if (not len(segments) or starts[0]!=0 or not np.array_equal(starts[1:],np.cumsum(lengths)[:-1])
            or int(lengths.sum())!=len(arrays[3]) or np.any(lengths>4096) or np.any(lengths==0)
            or len({r['doc_id'] for r in segments})!=len(segments)):
        raise ValueError('short document packing boundaries/uniqueness/capacity do not close')
    if heldout.intersection(r['doc_id'] for r in segments):raise ValueError('packed replay contains heldout documents')
    eos=json.loads((root/'data/token_streams/pretrain-1t/anneal.meta.json').read_bytes())['document_boundary']['token_id']
    parent_lengths=np.memmap(root/catalog_report['stages']['anneal']['lengths_file'],mode='r',dtype='<u4')
    last=max(r['source_document_index'] for r in segments)
    cumulative=np.cumsum(parent_lengths[:last+1],dtype=np.uint64)
    for r in segments:
        index=r['source_document_index'];offset=0 if index==0 else int(cumulative[index-1])
        if (r['source_start']!=offset or r['encoded_tokens']!=int(parent_lengths[index])
                or int(arrays[3][r['packed_start']+r['encoded_tokens']-1])!=eos
                or not np.array_equal(arrays[3][r['packed_start']:r['packed_start']+r['encoded_tokens']],
                                      arrays[2][offset:offset+r['encoded_tokens']])):
            raise ValueError('independent short document byte/EOD/offset verification failed')
    # Independently join every copied ID and domain to its actual physical
    # order row; the producer's provenance alone cannot assert these fields.
    needed={r['source_document_index']:(r['doc_id'],r['domain']) for r in segments}
    wanted=np.array(sorted(needed),dtype=np.int64)
    position=0;seen=0
    order=root/catalog_report['stages']['anneal']['order_file']
    with order.open('rb') as handle:
        for batch in pq.ParquetFile(handle).iter_batches(batch_size=262144,columns=['doc_id','domain'],use_threads=False):
            left,right=np.searchsorted(wanted,[position,position+batch.num_rows])
            indices=(wanted[left:right]-position).tolist()
            if indices:
                import pyarrow as pa
                rows=batch.take(pa.array(indices)).to_pylist()
                for local,r in zip(indices,rows,strict=True):
                    if needed[position+local]!=(r['doc_id'],r['domain']):raise ValueError('short source document ID/domain mismatch')
                    seen+=1
            position+=batch.num_rows
            if position>last:break
    if seen!=len(segments):raise ValueError('short source identity join is incomplete')
    del cumulative,needed
    results=[]
    for seq_s,nsteps in policy['stage_updates'].items():
        seq=int(seq_s);proposal_path=folder/f'context-{seq}.proposal.json'
        proposal=json.loads(proposal_path.read_bytes())
        pins[str(proposal_path.relative_to(root))]=digest(proposal_path)
        for binding in ('record_index','long_provenance','short_provenance','source_byte_audit','policy'):
            row=proposal[binding]
            if digest(root/row['file'])!=row['sha256']:raise ValueError('window proposal dependency changed: '+binding)
            pins[row['file']]=row['sha256']
        if (proposal['scope']!=policy['scope'] or proposal['samples']*seq!=nsteps*4194304
                or proposal['sources']!=sources or proposal['target_labels']!=nsteps*4194304):
            raise ValueError('window proposal differs from its frozen scope/budget/source policy')
        index=np.fromfile(root/proposal['record_index']['file'],dtype='<u8').reshape(-1,3)
        if len(index)!=proposal['samples'] or not np.all(index[:,2].reshape(-1,8).sum(axis=1)==2):
            raise ValueError('window index fails its exact 25% replay groups')
        long=index[index[:,2]==0];short=index[index[:,2]==1]
        expected_short=proposal['short_start']+np.arange(len(short),dtype=np.uint64)*seq
        if not np.all(short[:,0]==3) or not np.array_equal(short[:,1],expected_short):
            raise ValueError('short replay windows omit/duplicate/misalign target positions')
        records=pq.read_table(root/proposal['long_provenance']['file']).to_pylist()
        if len(records)!=len(long):raise ValueError('long window provenance row count mismatch')
        seen_windows=set();by_doc=defaultdict(list);end_counts=defaultdict(int);domains=defaultdict(int)
        for actual,r in zip(long,records,strict=True):
            parent=lookup.get(r['doc_id'])
            if parent is None or r['doc_id'] in heldout:raise ValueError('long source document is absent or heldout')
            source=PHASES.index(parent['parent_stage']);relative=r['document_offset'];length=parent['encoded_tokens_with_eod']
            if (source!=r['source_id'] or relative<0 or relative+seq+1>length
                    or r['token_offset']!=parent['parent_token_offset']+relative
                    or not np.array_equal(actual,np.array([source,r['token_offset'],0],dtype='<u8'))
                    or r['domain']!=parent['domain']):
                raise ValueError('long window source/document/S+1 byte bounds mismatch')
            if relative%seq:
                if seq!=65536 or relative!=length-seq-1:raise ValueError('unapproved overlapping long window')
                end_counts[r['doc_id']]+=1
            token=(r['doc_id'],relative)
            if token in seen_windows:raise ValueError('duplicate long context window')
            seen_windows.add(token);by_doc[r['doc_id']].append((relative+1,relative+seq+1));domains[r['domain']]+=seq
        if any(n>1 for n in end_counts.values()) or set(domains)!=set(policy['short_target_domain_weights']):
            raise ValueError('long-window extra-tail cap/domain coverage fails')
        unique=0
        for intervals in by_doc.values():
            previous=-1
            for left,right in sorted(intervals):
                unique+=max(0,right-max(left,previous));previous=max(previous,right)
        repeats=len(long)*seq-unique
        if repeats!=proposal['repeated_long_targets'] or repeats/(len(long)*seq)>policy['maximum_repeated_fraction_of_long_targets']:
            raise ValueError('actual long target interval union exceeds the frozen repetition policy')
        left=proposal['short_start']+1;right=left+len(short)*seq
        if right>len(arrays[3]):raise ValueError('short target range exceeds packed capacity')
        counts=dict.fromkeys(policy['short_target_domain_weights'],0)
        for r in segments:
            overlap=max(0,min(right,r['packed_start']+r['encoded_tokens'])-max(left,r['packed_start']))
            counts[r['domain']]+=overlap
        if sum(counts.values())!=len(short)*seq:raise ValueError('short target-domain accounting is incomplete')
        for domain,weight in policy['short_target_domain_weights'].items():
            error=abs(counts[domain]/(len(short)*seq)-weight)
            if error>policy['short_domain_absolute_tolerance'] or error/weight>policy['short_domain_relative_tolerance']:
                raise ValueError('short replay domain floor/cap failed: '+domain)
        # Every published generated dependency is flushed before admission.
        for path in folder.iterdir():
            if path.is_file():
                with path.open('rb') as handle:os.fsync(handle.fileno())
        fd=os.open(folder,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        admission={'schema_version':1,'status':'PASS_LONG_WINDOW_DATA_ADMISSION','scope':policy['scope'],
            'checks':dict.fromkeys(CHECKS,'PASS'),'seq_len':seq,'record_index_sha256':proposal['record_index']['sha256'],
            'target_labels':proposal['target_labels'],'sources':sources,'inputs':pins,
            'long_domain_tokens':dict(domains),'short_domain_tokens':counts,
            'long_repeated_target_labels':repeats,'long_unique_target_labels':unique,
            'repeated_fraction_of_total_stage':repeats/proposal['target_labels'],
            'heldout_document_ids_checked':len(heldout),'short_documents_compared_byte_for_byte':len(segments)}
        admission_path=folder/f'context-{seq}.admission.json';publish(admission_path,admission)
        release={'schema_version':1,'status':'PASS_LONG_CONTEXT_WINDOW_RELEASE','scope':policy['scope'],
            'layout':'source-index-s-plus-one-v1','seq_len':seq,'tokenizer_sha256':proposal['tokenizer_sha256'],
            'short_replay_fraction':.25,'samples':proposal['samples'],'target_labels':proposal['target_labels'],
            'record_index':proposal['record_index'],'sources':sources,'source_byte_audit':proposal['source_byte_audit'],
            'admission':{'file':str(admission_path.relative_to(root)),'sha256':digest(admission_path)}}
        release['dataset_id']=dataset_id(release)
        release_path=folder/f'context-{seq}.release.json';publish(release_path,release)
        load_window_release(release_path,expected_sha256=digest(release_path),source_root=root,seq_len=seq,
                            tokenizer_sha256=proposal['tokenizer_sha256'])
        results.append({'seq_len':seq,'scope':policy['scope'],'samples':len(index),'release':str(release_path),
                        'release_sha256':digest(release_path),'repeated_target_labels':repeats})
    for name,sha in pins.items():
        if digest(root/name)!=sha:raise ValueError('independent admission input changed: '+name)
    for row in sources:
        if stat_identity((root/row['file']).stat())!=row['stat']:raise ValueError('admitted source changed')
    return {'status':'PASS_LONG_WINDOW_RELEASES','scope':policy['scope'],'stages':results}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--folder',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(audit(args.root,args.folder),sort_keys=True))
