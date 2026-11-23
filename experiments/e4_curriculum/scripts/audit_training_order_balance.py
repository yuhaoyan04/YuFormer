#!/usr/bin/env python3
"""Measure actual training-token domain balance over consecutive time windows."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

import calibrate_training_tokens as calibration

ROOT = Path(__file__).resolve().parents[3]


def audit(root, phase, windows=10):
    frozen_path = root / 'experiments/e4_curriculum/results/frozen_1t_recipe.json'
    release_path = root / 'data/processed/releases/pretrain-1t-frozen.release.json'
    meta_path = root / f'data/token_streams/pretrain-1t/{phase}.meta.json'
    config_path = root / 'pretrain/configs/pretrain_base.yaml'
    frozen, release, meta = [json.loads(p.read_text()) for p in (frozen_path, release_path, meta_path)]
    config = yaml.safe_load(config_path.read_text())
    i = ('warmup', 'stable', 'anneal').index(phase)
    total = int(config['data']['stage_sample_limits'][i]) * int(config['data']['seq_len'])
    order = root / release['stages'][phase]['order']
    record = next(r for r in release['artifacts'] if r['path'] == str(order.relative_to(root)))
    ledger = meta_path.parent / meta['doc_lengths']['file']
    edges = np.array([1 + total*k//windows for k in range(windows+1)], dtype=np.int64)
    counts = np.zeros((windows, len(calibration.DOMAINS)), dtype=np.int64)
    with order.open('rb') as order_handle, ledger.open('rb') as ledger_handle:
        handles = ((order, order_handle, record['sha256']),
                   (ledger, ledger_handle, meta['doc_lengths']['sha256']))
        identities = []
        for path, handle, expected in handles:
            identities.append(calibration.materializer._stat_identity(os.fstat(handle.fileno())))
            if calibration.materializer._hash_open_file(handle) != expected:
                raise ValueError('training order balance input SHA mismatch: ' + str(path))
        lengths = np.fromfile(ledger_handle, dtype='<u4')
        position = offset = 0
        for batch in pq.ParquetFile(order_handle).iter_batches(batch_size=262144, columns=['domain']):
            end = position + batch.num_rows
            values = lengths[position:end].astype(np.int64)
            if len(values) != batch.num_rows or np.any(values <= 0):
                raise ValueError('training balance ledger does not close')
            ends = np.cumsum(values) + offset
            starts = ends - values
            codes = pc.index_in(pc.cast(batch.column(0), pa.string()),
                                value_set=pa.array(calibration.DOMAINS))
            if codes.null_count:
                raise ValueError('invalid balance domain')
            codes = codes.to_numpy(zero_copy_only=False)
            for window in range(windows):
                if offset >= edges[window+1] or ends[-1] <= edges[window]:
                    continue
                overlap = np.maximum(0, np.minimum(ends, edges[window+1]) - np.maximum(starts, edges[window]))
                for domain in range(len(calibration.DOMAINS)):
                    counts[window, domain] += int(overlap[codes == domain].sum(dtype=np.int64))
            position = end
            offset = int(ends[-1])
        if position != len(lengths) or offset != meta['total_tokens']:
            raise ValueError('training balance full ledger counts mismatch')
        for (path, handle, _), identity in zip(handles, identities):
            if (calibration.materializer._stat_identity(os.fstat(handle.fileno())) != identity or
                    calibration.materializer.source_file_identity(path) != identity):
                raise ValueError('training balance input changed')
    if not np.array_equal(counts.sum(axis=1), np.diff(edges)):
        raise ValueError('training time window coverage does not close')
    weights = counts / counts.sum(axis=1, keepdims=True)
    expected = np.array([frozen['stages'][phase]['target_final_domain_weights'][d]
                         for d in calibration.DOMAINS])
    return {'purpose': 'TRAINING_ORDER_BALANCE_DIAGNOSTIC', 'phase': phase,
        'status': 'MEASURED', 'windows': windows, 'total_trained_tokens': total,
        'window_target_intervals': list(zip(edges[:-1].tolist(), edges[1:].tolist())),
        'domain_counts': {d: counts[:, j].tolist() for j, d in enumerate(calibration.DOMAINS)},
        'domain_weights': {d: weights[:, j].tolist() for j, d in enumerate(calibration.DOMAINS)},
        'target_weights': dict(zip(calibration.DOMAINS, expected.tolist())),
        'max_window_domain_absolute_error': float(np.max(np.abs(weights-expected))),
        'whole_consumed_domain_counts': dict(zip(calibration.DOMAINS, counts.sum(axis=0).tolist())),
        'evidence_sha256': {str(p.relative_to(root)): calibration.digest(p)
            for p in (frozen_path, release_path, meta_path, config_path)},
        'order_sha256': record['sha256'], 'ledger_sha256': meta['doc_lengths']['sha256'],
        'audit_tool_sha256': calibration.digest(__file__)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('warmup', 'stable', 'anneal'), required=True)
    args = parser.parse_args()
    value = audit(ROOT, args.phase)
    destination = ROOT / f'data/health/{args.phase}-training-order-balance.json'
    calibration.atomic_json(destination, value)
    print(json.dumps({'phase': args.phase, 'max_window_error': value['max_window_domain_absolute_error'],
                      'report': str(destination)}), flush=True)
