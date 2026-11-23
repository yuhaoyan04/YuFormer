#!/usr/bin/env python3
"""Read-only early consumption counts from a completed, unpublished ledger."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml


def identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=('warmup', 'stable', 'anneal'), required=True)
    parser.add_argument('--producer-pid', type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root/'experiments/e4_curriculum/scripts'))
    from audit_1t_overlap import audit_consumed_mixture
    domains = ('web', 'code', 'math', 'chinese', 'academic', 'books', 'encyclopedia', 'qa', 'multilingual')
    command = Path(f'/proc/{args.producer_pid}/cmdline').read_bytes().split(b'\0')
    if (os.fsencode(root/'pretrain/scripts/materialize_tokens.py') not in command
            or b'--name' not in command or command[command.index(b'--name')+1] != args.phase.encode()):
        raise ValueError('staging diagnostic requires the actual phase producer')
    children = Path(f'/proc/{args.producer_pid}/task/{args.producer_pid}/children').read_text().split()
    for child in children:
        proc = Path('/proc')/child
        try:
            if b'merge_token_parts' in (proc/'cmdline').read_bytes():
                raise ValueError('native merge must finish before staging diagnostic')
        except FileNotFoundError:
            pass
    release_path = root/'data/processed/releases/pretrain-1t-frozen.release.json'
    frozen_path = root/'experiments/e4_curriculum/results/frozen_1t_recipe.json'
    base_path = root/'pretrain/configs/pretrain_base.yaml'
    pinned = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (release_path, frozen_path, base_path)}
    release, frozen = [json.loads(p.read_bytes()) for p in (release_path, frozen_path)]
    if release['frozen_recipe_sha256'] != pinned[str(frozen_path)]:
        raise ValueError('original selected data contract changed')
    config = yaml.safe_load(base_path.read_bytes())
    total = config['data']['seq_len']*config['data']['stage_sample_limits'][('warmup','stable','anneal').index(args.phase)]
    ledger_path = root/f'data/token_streams/pretrain-1t/.{args.phase}.doc_lengths.u32.tmp-{args.producer_pid}'
    order_path = root/release['stages'][args.phase]['order']
    expected_docs = release['stages'][args.phase]['selection_docs']
    with ledger_path.open('rb') as ledger_handle, order_path.open('rb') as order_handle:
        ledger_identity, order_identity = [identity(os.fstat(h.fileno())) for h in (ledger_handle, order_handle)]
        if ledger_identity[2] != expected_docs*4:
            raise ValueError('complete staging document ledger required')
        lengths = np.memmap(ledger_handle, dtype='<u4', mode='r')
        stop, offset, last = total+1, 0, None
        for start in range(0, len(lengths), 1_000_000):
            values = np.asarray(lengths[start:start+1_000_000], dtype=np.uint64)
            if np.any(values == 0):
                raise ValueError('zero staging document length')
            cumulative = np.cumsum(values, dtype=np.uint64)
            if offset+int(cumulative[-1]) >= stop:
                local = int(np.searchsorted(cumulative, stop-offset, side='left'))
                last = start+local
                before_last = offset+(int(cumulative[local-1]) if local else 0)
                break
            offset += int(cumulative[-1])
        if last is None:
            raise ValueError('staging stream cannot cover the actual training budget')
        seen = np.zeros(last+1, dtype=np.bool_)
        counts = np.zeros(len(domains), dtype=np.uint64)
        rows = selected_rows = 0
        for batch in pq.ParquetFile(order_handle).iter_batches(batch_size=1_000_000, columns=['position','domain']):
            positions = batch.column(0).to_numpy(zero_copy_only=False)
            if np.any(positions < 0) or np.any(positions >= expected_docs):
                raise ValueError('staging order position outside exact population')
            mask = positions <= last
            selected = positions[mask]
            if np.any(seen[selected]):
                raise ValueError('duplicate staging consumed position')
            seen[selected] = True
            weights = np.asarray(lengths[selected], dtype=np.int64)
            weights[selected == 0] -= 1
            weights[selected == last] -= int(lengths[last])-(stop-before_last)
            if np.any(weights < 0):
                raise ValueError('invalid clipped target interval')
            codes = pc.index_in(pc.cast(batch.column(1).filter(pa.array(mask)), pa.string()), value_set=pa.array(domains))
            if codes.null_count:
                raise ValueError('unknown staging domain')
            counts += np.bincount(codes.to_numpy(zero_copy_only=False), weights=weights, minlength=len(domains)).astype(np.uint64)
            rows += batch.num_rows
            selected_rows += len(selected)
            if rows % 50_000_000 == 0:
                print(json.dumps({'observed_order_rows':rows,'expected_docs':expected_docs}),flush=True)
        if rows != expected_docs or selected_rows != last+1 or not np.all(seen) or int(counts.sum()) != total:
            raise ValueError('exact staging consumed position/token coverage does not close')
        for handle, expected in ((ledger_handle, ledger_identity),(order_handle, order_identity)):
            if identity(os.fstat(handle.fileno())) != expected:
                raise ValueError('staging input changed during read-only diagnostic')
        if identity(order_path.stat()) != order_identity:
            raise ValueError('staging source order replaced')
        current_ledger = ledger_path if ledger_path.exists() else ledger_path.parent/f'{args.phase}.doc_lengths.u32'
        if identity(current_ledger.stat()) != ledger_identity:
            raise ValueError('staging ledger replaced')
    for name, expected in pinned.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError('staging diagnostic contract changed')
    values = dict(zip(domains, map(int, counts)))
    mixture = audit_consumed_mixture(values, frozen['stages'][args.phase]['target_final_domain_weights'], absolute_tolerance=.002, relative_tolerance=.1)
    result = {'status':'PASS_STAGING_MIXTURE_DIAGNOSTIC' if mixture['status']=='PASS' else 'FAIL_STAGING_MIXTURE_DIAGNOSTIC',
              'scope':'UNPUBLISHED_STAGING_OBSERVATION_NOT_CANONICAL_GATE','phase':args.phase,'mixture':mixture,
              'observed_at':datetime.now(timezone.utc).isoformat(),'configured_target_tokens':total,'order_rows':rows,
              'ledger_stat_identity':ledger_identity,'order_stat_identity':order_identity,'evidence_sha256':pinned,
              'source_bytes_modified':False,'producer_interrupted':False,'tool_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output=root/f'data/health/{args.phase}-staging-consumption-diagnostic.json'
    temporary=output.with_name(output.name+f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');os.replace(temporary,output)
    print(json.dumps({'status':result['status'],'mixture':mixture},sort_keys=True),flush=True)


if __name__ == '__main__':
    main()
