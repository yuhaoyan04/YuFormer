#!/usr/bin/env python3
"""Verify independent causal-block order on actual immutable training lengths."""
import argparse
import hashlib
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
from training_source_windows import require_window_admission

ROOT = Path(__file__).resolve().parents[3]
SEED = 'yuformer-e134-independent-training-order-20261114-v1'
WORK = ROOT / 'data/processed/training_permutations/e134-11326994-v1'


def temporal_mixing_check(counts, window_weights, edges):
    population = len(counts)
    totals = counts.sum(axis=1)
    sizes = np.diff(edges)
    if (population < 2 or len(sizes) != len(window_weights) or np.any(sizes <= 0)
            or np.any(totals <= 0) or not np.all(totals == totals[0])):
        raise ValueError('temporal mixing requires nonempty equal-token causal blocks/windows')
    token_unit = int(totals[0])
    mean = counts.sum(axis=0)/(population*token_unit)
    variance = np.array([(counts[:, i].astype(np.float64)/token_unit).var()
                          for i in range(counts.shape[1])])
    deviation = np.abs(window_weights-mean)
    standard_error = np.sqrt(variance[None, :]*(population-sizes[:, None])
                              /((population-1)*sizes[:, None]))
    bands = 6*standard_error+1/(sizes[:, None]*token_unit)+1e-12
    return {'pass': bool(np.all(np.isfinite(deviation)) and np.all(deviation <= bands)),
            'reference': 'actual admitted whole-window domain weights',
            'criterion': 'six finite-population standard errors plus one target-token quantum',
            'sigma_multiplier': 6, 'deviation': deviation.tolist(), 'allowed_deviation': bands.tolist(),
            'global_domain_tolerances_unchanged': True}


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
    mixture_path = ROOT/f'data/health/pretrain-1t-{phase}-mixture-preflight.json'
    mixture = json.loads(mixture_path.read_bytes())
    admitted = require_window_admission(ROOT, mixture, phase=phase, stage_samples=samples,
        seq_len=seq_len, release_sha256=calibration.digest(release_path),
        stream_meta_sha256=calibration.digest(meta_path),
        data_frozen_recipe_sha256=release['frozen_recipe_sha256'])
    source_sample_offset = admitted['source_sample_offset']
    if admitted['block_samples'] != block_samples:
        raise ValueError('temporal recount block size differs from its admitted source window')
    window_start = 1+source_sample_offset*seq_len
    window_stop = window_start+total
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
        for batch in pq.ParquetFile(order_handle).iter_batches(batch_size=262144, columns=['position','domain']):
            end_position = position+batch.num_rows
            if not np.array_equal(batch.column(0).to_numpy(zero_copy_only=False), np.arange(position,end_position)):
                raise ValueError('temporal recount source rows must match actual ledger positions')
            values = lengths[position:end_position].astype(np.int64)
            if len(values) != batch.num_rows or np.any(values <= 0):
                raise ValueError('causal block ledger does not cover population')
            ends = np.cumsum(values)+offset
            starts = ends-values
            codes = pc.index_in(pc.cast(batch.column(1), pa.string()),
                                value_set=pa.array(calibration.DOMAINS))
            if codes.null_count:
                raise ValueError('invalid block domain')
            codes = codes.to_numpy(zero_copy_only=False)
            begins, stops = np.maximum(starts, window_start)-window_start, np.minimum(ends, window_stop)-window_start
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
    mixing = temporal_mixing_check(counts, after_weights, edges)
    original_mixing = temporal_mixing_check(counts, before_weights, edges)
    expected = np.array([frozen['stages'][phase]['target_final_domain_weights'][d]
                         for d in calibration.DOMAINS])
    base_permutation_sha = hashlib.sha256(permutation.tobytes()).hexdigest()
    physical_permutation = (permutation.astype(np.uint64)+source_sample_offset//block_samples).astype('<u4')
    directory = WORK / f'block-{block_samples}' / f'window-{source_sample_offset}'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{phase}.blocks.u32'
    temporary = path.with_name(path.name+f'.tmp-{os.getpid()}')
    physical_permutation.tofile(temporary)
    os.replace(temporary, path)
    stats = directory / f'{phase}.block-domain-tokens.npy'
    with stats.open('wb') as handle:
        np.save(handle, counts)
    result = {'status': 'PASS_DIAGNOSTIC', 'purpose': 'EXACT_CAUSAL_BLOCK_ORDER_DIAGNOSTIC',
        'production_qualified': False, 'phase': phase, 'seed': SEED, 'algorithm': blocks.ALGORITHM,
        'block_samples': block_samples, 'block_tokens': block_tokens, 'block_count': block_count,
        'seq_len': seq_len, 'stage_sample_limit': samples, 'trained_target_tokens': total,
        'target_interval': [window_start, window_stop], 'source_sample_offset': source_sample_offset,
        'supervised_prefix_unchanged': source_sample_offset == 0, 'source_window': mixture['source_window'],
        'base_permutation_sha256': base_permutation_sha,
        'temporal_mixing': mixing, 'original_temporal_mixing': original_mixing,
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
            for p in (release_path, config_path, frozen_path, meta_path, mixture_path)},
        'audit_tool_sha256': calibration.digest(__file__),
        'permutation_tool_sha256': calibration.digest(blocks.__file__)}
    calibration.atomic_json(directory / f'{phase}.diagnostic.json', result)
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
