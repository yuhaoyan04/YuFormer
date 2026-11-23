#!/usr/bin/env python3
"""Measure uniform sample lengths against an already verified complete population."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import calibrate_training_tokens as calibration
from analyze_uniform_calibration import ratio_statistics

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT/'data/processed/tokenizer_calibration/post-selection-11326994-uniform-v2'


def selected_positions(parent_handle, sample):
    ids = sample['doc_id'].combine_chunks()
    if ids.null_count or not len(ids):
        raise ValueError('uniform sample contains no valid IDs')
    if len(ids) > 1 and pc.any(pc.greater_equal(ids.slice(0, len(ids)-1), ids.slice(1))).as_py():
        raise ValueError('uniform sample IDs are not strictly sorted')
    positions = np.full(len(ids), -1, dtype=np.int64)
    seen = np.zeros(len(ids), dtype=bool)
    row = 0
    for batch in pq.ParquetFile(parent_handle).iter_batches(
            batch_size=262144, columns=['doc_id', 'domain', 'n_tokens']):
        # Both arrays are sorted, so only sample IDs within this batch's range participate.
        mask = pc.and_(pc.greater_equal(ids, batch.column(0)[0]), pc.less_equal(ids, batch.column(0)[-1]))
        selected = np.flatnonzero(mask.to_numpy(zero_copy_only=False))
        if len(selected):
            matches = pc.index_in(ids.take(pa.array(selected)), value_set=batch.column(0))
            if matches.null_count or np.any(seen[selected]):
                raise ValueError('uniform sample ID missing/duplicated in selected population')
            local = matches.to_numpy(zero_copy_only=False).astype(np.int64)
            for name, col in [('domain', 1), ('n_tokens', 2)]:
                if not pc.all(pc.equal(sample[name].take(pa.array(selected)),
                                       batch.column(col).take(pa.array(local)))).as_py():
                    raise ValueError('uniform sample domain/reference differs from selected population')
            positions[selected] = row+local
            seen[selected] = True
        row += batch.num_rows
    if not np.all(seen):
        raise ValueError('uniform sample does not belong to the complete selected population')
    return positions, row


def measure(phase):
    directory = WORK/phase
    selection_path, sample_path = directory/'selection.json', directory/'sample.parquet'
    selection = json.loads(selection_path.read_bytes())
    release_path = ROOT/'data/processed/releases/pretrain-1t-frozen.release.json'
    release = json.loads(release_path.read_bytes())
    order = ROOT/release['stages'][phase]['order']
    record = next(r for r in release['artifacts'] if r['path'] == str(order.relative_to(ROOT)))
    meta_path = ROOT/f'data/token_streams/pretrain-1t/{phase}.meta.json'
    meta = json.loads(meta_path.read_bytes())
    if (not calibration.verified_source_document_binding(meta)
            or meta['order_sha256'] != record['sha256']
            or selection['parent_order_sha256'] != record['sha256']
            or meta['data_snapshot_id'] != release['snapshot_id']):
        raise ValueError('uniform/full population source provenance mismatch')
    ledger = meta_path.parent/meta['doc_lengths']['file']
    with order.open('rb') as ph, sample_path.open('rb') as sh, ledger.open('rb') as lh:
        handles = [(order, ph, record['sha256']), (sample_path, sh, selection['sample_sha256']),
                   (ledger, lh, meta['doc_lengths']['sha256'])]
        identities = []
        for path, handle, expected in handles:
            identities.append(calibration.materializer._stat_identity(os.fstat(handle.fileno())))
            if calibration.materializer._hash_open_file(handle) != expected:
                raise ValueError('uniform/full population byte SHA mismatch')
        sample = pq.read_table(sh, columns=['doc_id', 'domain', 'n_tokens'])
        positions, population_rows = selected_positions(ph, sample)
        lengths = np.fromfile(lh, dtype='<u4')
        if (len(lengths) != population_rows or population_rows != meta['n_docs'] or np.any(lengths == 0)
                or int(lengths.sum(dtype=np.uint64)) != meta['total_tokens']):
            raise ValueError('full population length ledger does not close')
        actual = lengths[positions]
        for (path, handle, _), identity in zip(handles, identities):
            if (calibration.materializer._stat_identity(os.fstat(handle.fileno())) != identity or
                    calibration.materializer.source_file_identity(path) != identity):
                raise ValueError('uniform/full population inputs changed')
    domains = np.asarray(sample['domain'].to_pylist())
    refs = sample['n_tokens'].combine_chunks().to_numpy(zero_copy_only=False)
    measurements = {d: ratio_statistics(refs[domains == d], actual[domains == d],
                        selection['population_docs_by_domain'][d]) for d in calibration.DOMAINS}
    predicted = {d: selection['population_reference_tokens_by_domain'][d]*v['ratio']
                 for d, v in measurements.items()}
    total = sum(predicted.values())
    report = {'status': 'PASS_INDEPENDENT_POPULATION_LEDGER_DIAGNOSTIC', 'production_qualified': False,
        'phase': phase, 'sampling': selection['sampling'], 'seed': selection['seed'],
        'measurement_origin': 'actual per-document lengths of the complete source-verified population',
        'sample_docs': len(actual), 'population_docs': len(lengths), 'domains': measurements,
        'predicted_actual_tokens': total, 'full_actual_tokens': meta['total_tokens'],
        'relative_total_prediction_error': total/meta['total_tokens']-1,
        'predicted_weights': {d: v/total for d, v in predicted.items()},
        'sample_positions_sha256': hashlib.sha256(positions.astype('<i8').tobytes()).hexdigest(),
        'sample_actual_lengths_sha256': hashlib.sha256(actual.astype('<u4').tobytes()).hexdigest(),
        'evidence_sha256': {str(p.relative_to(ROOT)): calibration.digest(p)
            for p in (selection_path, sample_path, release_path, meta_path)},
        'order_sha256': record['sha256'], 'ledger_sha256': meta['doc_lengths']['sha256'],
        'tool_sha256': calibration.digest(__file__)}
    calibration.atomic_json(ROOT/f'data/health/uniform-{phase}-population-ledger-measurement.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('warmup', 'stable', 'anneal'), required=True)
    value = measure(parser.parse_args().phase)
    print(json.dumps({k: value[k] for k in ('status', 'phase', 'predicted_actual_tokens',
                                          'full_actual_tokens', 'relative_total_prediction_error')}), flush=True)
