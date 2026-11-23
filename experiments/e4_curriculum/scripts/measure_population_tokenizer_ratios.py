#!/usr/bin/env python3
"""Exact tokenizer ratios of each complete, immutable selected phase population."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import calibrate_training_tokens as calibration

ROOT = Path(__file__).resolve().parents[3]


def measure(phase):
    release_path = ROOT/'data/processed/releases/pretrain-1t-frozen.release.json'
    release = json.loads(release_path.read_bytes())
    row = release['stages'][phase]
    order, manifest_path = ROOT/row['order'], ROOT/row['manifest']
    artifacts = {r['path']: r for r in release['artifacts']}
    meta_path = ROOT/f'data/token_streams/pretrain-1t/{phase}.meta.json'
    meta = json.loads(meta_path.read_bytes())
    if (release.get('status') != 'PASS' or not calibration.verified_source_document_binding(meta)
            or meta.get('order_sha256') != artifacts[row['order']]['sha256']
            or meta.get('data_snapshot_id') != release.get('snapshot_id')):
        raise ValueError('full-population measurement source/data/order binding mismatch')
    ledger = meta_path.parent/meta['doc_lengths']['file']
    counts = np.zeros((len(calibration.DOMAINS), 3), dtype=np.int64)
    with order.open('rb') as oh, ledger.open('rb') as lh, manifest_path.open('rb') as mh:
        handles = [(order, oh, artifacts[row['order']]['sha256']),
                   (ledger, lh, meta['doc_lengths']['sha256']),
                   (manifest_path, mh, artifacts[row['manifest']]['sha256'])]
        identities = []
        for path, handle, expected in handles:
            identities.append(calibration.materializer._stat_identity(os.fstat(handle.fileno())))
            if calibration.materializer._hash_open_file(handle) != expected:
                raise ValueError('full-population measurement input SHA mismatch')
        manifest = json.load(mh)
        lengths = np.fromfile(lh, dtype='<u4')
        if len(lengths) != meta['n_docs'] or np.any(lengths == 0):
            raise ValueError('full-population measurement ledger is incomplete')
        offset = 0
        for batch in pq.ParquetFile(oh).iter_batches(batch_size=262144, columns=['domain', 'n_tokens']):
            codes = pc.index_in(pc.cast(batch.column(0), pa.string()), value_set=pa.array(calibration.DOMAINS))
            if codes.null_count or batch.column(1).null_count:
                raise ValueError('full-population measurement has an invalid domain/reference')
            codes = codes.to_numpy(zero_copy_only=False)
            refs = batch.column(1).to_numpy(zero_copy_only=False)
            actual = lengths[offset:offset+batch.num_rows]
            if len(actual) != batch.num_rows or np.any(refs <= 0):
                raise ValueError('full-population ledger and order do not close')
            for index, domain in enumerate(calibration.DOMAINS):
                mask = codes == index
                counts[index] += [int(mask.sum()), int(refs[mask].sum(dtype=np.int64)),
                                   int(actual[mask].sum(dtype=np.uint64))]
            offset += batch.num_rows
        for (path, handle, _), identity in zip(handles, identities):
            if (calibration.materializer._stat_identity(os.fstat(handle.fileno())) != identity or
                    calibration.materializer.source_file_identity(path) != identity):
                raise ValueError('full-population measurement input changed')
    if offset != len(lengths) or int(counts[:, 2].sum()) != meta['total_tokens']:
        raise ValueError('full-population actual token accounting mismatch')
    domains = {}
    for index, domain in enumerate(calibration.DOMAINS):
        docs, reference, actual = map(int, counts[index])
        if (docs != manifest['selected'][domain]['selected_docs']
                or reference != manifest['selected'][domain]['selected_tokens'] or docs < 100 or reference <= 0):
            raise ValueError('full-population domain counts differ from selected manifest: '+domain)
        domains[domain] = {'docs': docs, 'reference_tokens': reference,
                           'actual_tokens_with_eod': actual, 'ratio': actual/reference}
    report = {'schema_version': 1, 'status': 'PASS_FULL_POPULATION_MEASUREMENT',
        'purpose': 'EXACT_SELECTED_POPULATION_TOKENIZER_MEASUREMENT', 'phase': phase,
        'sampling': 'CENSUS_NO_SAMPLING', 'production_qualified': False,
        'includes_appended_eod': True, 'domains': domains,
        'full_population_docs': offset, 'full_population_actual_tokens': meta['total_tokens'],
        'domain_weights': {d: v['actual_tokens_with_eod']/meta['total_tokens'] for d, v in domains.items()},
        'does_not_replace_configured_consumption_gate': True,
        'source_roster_fingerprint': meta['source_document_binding']['roster_fingerprint'],
        'evidence_sha256': {str(p.relative_to(ROOT)): calibration.digest(p)
            for p in (release_path, meta_path, manifest_path)},
        'order_sha256': artifacts[row['order']]['sha256'], 'ledger_sha256': meta['doc_lengths']['sha256'],
        'tool_sha256': calibration.digest(__file__)}
    calibration.atomic_json(ROOT/f'data/health/{phase}-population-tokenizer-census.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('warmup', 'stable', 'anneal'), required=True)
    result = measure(parser.parse_args().phase)
    print(json.dumps({k: result[k] for k in ('status', 'phase', 'full_population_docs',
                                           'full_population_actual_tokens', 'domain_weights')}), flush=True)
