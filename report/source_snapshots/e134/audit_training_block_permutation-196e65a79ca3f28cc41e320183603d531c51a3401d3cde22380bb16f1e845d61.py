#!/usr/bin/env python3
"""Verify independent causal-block order on actual immutable training lengths."""
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
import training_block_permutation as blocks

ROOT = Path(__file__).resolve().parents[3]
SEED = 'yuformer-e134-independent-training-order-20261114-v1'
WORK = ROOT / 'data/processed/training_permutations/e134-11326994-v1'


def audit(phase, block_samples):
    release_path = ROOT / 'data/processed/releases/pretrain-1t-frozen.release.json'
    config_path = ROOT / 'pretrain/configs/pretrain_base.yaml'
    frozen_path = ROOT / 'experiments/e4_curriculum/results/frozen_1t_recipe.json'
    meta_path = ROOT / f'data/token_streams/pretrain-1t/{phase}.meta.json'
    release, frozen, meta = [json.loads(p.read_text()) for p in (release_path, frozen_path, meta_path)]
    config = yaml.safe_load(config_path.read_text())
    seq_len = int(config['data']['seq_len'])
    samples = int(config['data']['stage_sample_limits'][('warmup', 'stable', 'anneal').index(phase)])
    permutation = blocks.build_permutation(samples, block_samples, seed=SEED, phase=phase)
    block_count, block_tokens = len(permutation), block_samples*seq_len
    total = samples*seq_len
    order = ROOT / release['stages'][phase]['order']
    record = next(r for r in release['artifacts'] if r['path'] == str(order.relative_to(ROOT)))
    ledger = meta_path.parent / meta['doc_lengths']['file']
    counts = np.zeros((block_count, len(calibration.DOMAINS)), dtype=np.int64)
    difference = np.zeros((block_count+1, len(calibration.DOMAINS)), dtype=np.int64)
    with order.open('rb') as order_handle, ledger.open('rb') as ledger_handle:
        handles = ((order, order_handle, record['sha256']),
                   (ledger, ledger_handle, meta['doc_lengths']['sha256']))
        identities = []
        for path, handle, expected in handles:
            identities.append(calibration.materializer._stat_identity(os.fstat(handle.fileno())))
            if calibration.materializer._hash_open_file(handle) != expected:
                raise ValueError('causal block input SHA mismatch: ' + str(path))
        lengths = np.fromfile(ledger_handle, dtype='<u4')
        position = offset = 0
        for batch in pq.ParquetFile(order_handle).iter_batches(batch_size=262144, columns=['domain']):
            end_position = position+batch.num_rows
            values = lengths[position:end_position].astype(np.int64)
            if len(values) != batch.num_rows or np.any(values <= 0):
                raise ValueError('causal block ledger does not cover population')
            ends = np.cumsum(values)+offset
            starts = ends-values
            codes = pc.index_in(pc.cast(batch.column(0), pa.string()),
                                value_set=pa.array(calibration.DOMAINS))
            if codes.null_count:
                raise ValueError('invalid block domain')
            codes = codes.to_numpy(zero_copy_only=False)
            begins, stops = np.maximum(starts, 1)-1, np.minimum(ends, total+1)-1
            for domain in range(len(calibration.DOMAINS)):
                mask = (codes == domain) & (begins < stops)
                blocks.add_document_intervals(counts, difference, domain,
                    begins[mask], stops[mask], block_tokens=block_tokens)
            position, offset = end_position, int(ends[-1])
        if position != len(lengths) or offset != meta['total_tokens']:
            raise ValueError('causal block full-population ledger accounting mismatch')
        for (path, handle, _), identity in zip(handles, identities):
            if (calibration.materializer._stat_identity(os.fstat(handle.fileno())) != identity or
                    calibration.materializer.source_file_identity(path) != identity):
                raise ValueError('causal block input changed during audit')
    counts += np.cumsum(difference, axis=0)[:-1]
    if np.any(counts < 0) or not np.all(counts.sum(axis=1) == block_tokens):
        raise ValueError('causal block coverage does not close exactly')
    ordered = counts[permutation]
    if not np.array_equal(ordered.sum(axis=0), counts.sum(axis=0)):
        raise ValueError('permutation changed full consumed domain counts')
    edges = [block_count*k//10 for k in range(11)]
    before = np.array([counts[a:b].sum(axis=0) for a, b in zip(edges[:-1], edges[1:])])
    after = np.array([ordered[a:b].sum(axis=0) for a, b in zip(edges[:-1], edges[1:])])
    before_weights = before/before.sum(axis=1, keepdims=True)
    after_weights = after/after.sum(axis=1, keepdims=True)
    expected = np.array([frozen['stages'][phase]['target_final_domain_weights'][d]
                         for d in calibration.DOMAINS])
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / f'{phase}.blocks.u32'
    temporary = path.with_name(path.name+f'.tmp-{os.getpid()}')
    permutation.tofile(temporary)
    os.replace(temporary, path)
    stats = WORK / f'{phase}.block-domain-tokens.npy'
    with stats.open('wb') as handle:
        np.save(handle, counts)
    result = {'status': 'PASS_DIAGNOSTIC', 'purpose': 'EXACT_CAUSAL_BLOCK_ORDER_DIAGNOSTIC',
        'production_qualified': False, 'phase': phase, 'seed': SEED, 'algorithm': blocks.ALGORITHM,
        'block_samples': block_samples, 'block_tokens': block_tokens, 'block_count': block_count,
        'seq_len': seq_len, 'stage_sample_limit': samples, 'trained_target_tokens': total,
        'target_interval_unchanged': [1, total+1], 'headroom_excluded': True,
        'every_causal_block_exactly_once': True, 'whole_domain_token_counts_unchanged': True,
        'windows_are_whole_blocks': True, 'window_block_boundaries': edges,
        'whole_domain_token_counts': dict(zip(calibration.DOMAINS, counts.sum(axis=0).tolist())),
        'target_weights': dict(zip(calibration.DOMAINS, expected.tolist())),
        'original_window_weights': {d: before_weights[:, i].tolist() for i, d in enumerate(calibration.DOMAINS)},
        'permuted_window_weights': {d: after_weights[:, i].tolist() for i, d in enumerate(calibration.DOMAINS)},
        'original_max_window_absolute_error': float(np.max(np.abs(before_weights-expected))),
        'permuted_max_window_absolute_error': float(np.max(np.abs(after_weights-expected))),
        'permutation_file': str(path.relative_to(ROOT)), 'permutation_sha256': calibration.digest(path),
        'block_domain_tokens_file': str(stats.relative_to(ROOT)), 'block_domain_tokens_sha256': calibration.digest(stats),
        'data_snapshot_id': release['snapshot_id'], 'dataset_id': release['dataset_id'],
        'source_order_sha256': record['sha256'], 'doc_lengths_sha256': meta['doc_lengths']['sha256'],
        'evidence_sha256': {str(p.relative_to(ROOT)): calibration.digest(p)
            for p in (release_path, config_path, frozen_path, meta_path)},
        'audit_tool_sha256': calibration.digest(__file__),
        'permutation_tool_sha256': calibration.digest(blocks.__file__)}
    calibration.atomic_json(WORK / f'{phase}.diagnostic.json', result)
    calibration.atomic_json(ROOT / f'data/health/{phase}-training-block-permutation.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('warmup', 'stable', 'anneal'), required=True)
    parser.add_argument('--block-samples', type=int, required=True)
    args = parser.parse_args()
    result = audit(args.phase, args.block_samples)
    print(json.dumps({k: result[k] for k in ('phase', 'original_max_window_absolute_error',
                                           'permuted_max_window_absolute_error')}), flush=True)
