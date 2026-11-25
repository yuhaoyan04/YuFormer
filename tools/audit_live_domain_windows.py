#!/usr/bin/env python3
"""Count actual consumed domain labels from SHA-bound order and block counts.

This diagnostic reads no weights, initializes no CUDA and never rewrites training
assets. Cached counts are reused only after binding their source data and current
runtime permutation. Its output is a census, not a new quality qualification.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

DOMAINS = ('web', 'code', 'math', 'chinese', 'academic', 'books', 'encyclopedia', 'qa', 'multilingual')


def identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def bound_file(path, expected):
    with path.open('rb') as handle:
        before = identity(os.fstat(handle.fileno()))
        sha = hashlib.sha256()
        while block := handle.read(8 << 20):
            sha.update(block)
        if sha.hexdigest() != expected:
            raise ValueError('asset SHA mismatch: '+str(path))
        if identity(os.fstat(handle.fileno())) != before or identity(path.stat()) != before:
            raise ValueError('asset changed while hashing: '+str(path))
    return before


def count_window(counts, permutation, begin, end, *, source_first_block):
    if (type(begin) is not int or type(end) is not int or begin < 0 or end <= begin
            or end > len(permutation) or counts.ndim != 2 or len(counts) != len(permutation)):
        raise ValueError('invalid whole-block window geometry')
    result = np.zeros(counts.shape[1], dtype=np.int64)
    for start in range(begin, end, 16384):
        physical = np.asarray(permutation[start:min(end, start+16384)], dtype=np.int64)
        local = physical-source_first_block
        if np.any(local < 0) or np.any(local >= len(counts)):
            raise ValueError('physical block lies outside bound source window')
        value = counts[local]
        if np.any(value < 0):
            raise ValueError('negative domain token count')
        result += value.sum(axis=0, dtype=np.int64)
    return result


def audit(root, review_path, steps):
    sources = {}

    def load(path):
        path = Path(path)
        if not path.is_absolute(): path = root/path
        raw = path.read_bytes()
        sources[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    review = load(review_path)
    runtime = load('data/runs/pretrain-final-1t-8gpu/run_identity.json')
    attempt = load('data/runs/pretrain-final-1t-8gpu/current_attempt.json')
    proof = load('data/health/stable-training-block-permutation.json')
    meta = load('data/token_streams/pretrain-1t/stable.meta.json')
    cursor = review['data_cursor_phase']
    if (review['attempt_id'] != attempt['attempt_id'] or attempt['identity_sha256'] != runtime['identity_sha256']
            or review['phase'] != 'stable' or cursor['phase_index'] != 1):
        raise ValueError('review is not the current stable attempt')
    order = runtime['training_sample_order']
    artifact = order['artifacts'][1]
    unit, block_samples = proof['block_tokens'], order['block_samples']
    population = order['population_stage_samples'][1]
    offset = artifact['source_sample_offset']
    if (block_samples != proof['block_samples'] or offset != proof['source_sample_offset']
            or offset % block_samples or population != proof['stage_sample_limit']
            or artifact['stage_sample_limit'] != population or artifact['stage_index'] != 1
            or artifact['sha256'] != proof['permutation_sha256']
            or runtime['dataset_id'] != proof['dataset_id'] or runtime['snapshot_id'] != proof['data_snapshot_id']
            or proof['doc_lengths_sha256'] != meta['doc_lengths']['sha256']
            or proof['source_order_sha256'] != meta['order_sha256']
            or runtime['streams'][1]['meta_sha256'] != sources['data/token_streams/pretrain-1t/stable.meta.json']
            or set(proof['whole_domain_token_counts']) != set(DOMAINS)):
        raise ValueError('cached counts do not close current data/order identity')
    per_update, remainder = divmod(cursor['consumed_samples'], cursor['latest_native_step'])
    if remainder or per_update != 1024 or unit != block_samples*4096:
        raise ValueError('unexpected actual 4K batch geometry')
    if len(steps) < 2 or steps != sorted(set(steps)) or steps[-1] > cursor['latest_native_step']:
        raise ValueError('requested step windows are not completely consumed')
    path_counts = root/proof['block_domain_tokens_file']
    path_order = Path(artifact['file'])
    before_counts = bound_file(path_counts, proof['block_domain_tokens_sha256'])
    before_order = bound_file(path_order, artifact['sha256'])
    counts = np.load(path_counts, mmap_mode='r', allow_pickle=False)
    permutation = np.memmap(path_order, dtype='<u4', mode='r')
    if (counts.shape != (population//block_samples, len(DOMAINS)) or counts.dtype != np.dtype('<i8')
            or len(permutation) != len(counts) or path_order.stat().st_size != artifact['bytes']):
        raise ValueError('bound asset geometry mismatch')
    reference_total = sum(proof['whole_domain_token_counts'].values())
    if reference_total != population*4096:
        raise ValueError('reference domain counts do not close full population')
    reference = {d: proof['whole_domain_token_counts'][d]/reference_total for d in DOMAINS}
    windows = []
    warmup_samples = order['population_stage_samples'][0]
    for first, last in zip(steps[:-1], steps[1:]):
        begin_sample, end_sample = first*per_update-warmup_samples, last*per_update-warmup_samples
        if begin_sample < 0 or begin_sample % block_samples or end_sample % block_samples:
            raise ValueError('step window crosses a phase or partial block')
        values = count_window(counts, permutation, begin_sample//block_samples, end_sample//block_samples,
                              source_first_block=offset//block_samples)
        total = (end_sample-begin_sample)*4096
        if int(values.sum()) != total:
            raise ValueError('actual domain counts do not close consumed labels')
        weights = {d: int(v)/total for d, v in zip(DOMAINS, values)}
        windows.append({'after_step': first, 'through_step': last, 'stable_sample_range': [begin_sample, end_sample],
            'training_labels': total, 'domain_labels': dict(zip(DOMAINS, map(int, values))),
            'domain_weights': weights,
            'difference_from_stage_reference_percentage_points': {d: 100*(weights[d]-reference[d]) for d in DOMAINS}})
    if identity(path_counts.stat()) != before_counts or identity(path_order.stat()) != before_order:
        raise ValueError('bound asset changed while counting')
    sources[str(path_counts.relative_to(root))] = proof['block_domain_tokens_sha256']
    sources[str(path_order.relative_to(root))] = artifact['sha256']
    return {'status': 'EXACT_READ_ONLY_CONSUMED_DOMAIN_CENSUS', 'reviewed_at': review['reviewed_at'],
        'attempt_id': review['attempt_id'], 'original_review_status': review['status'],
        'sources_sha256': sources, 'whole_stage_domain_weights': reference, 'windows': windows,
        'source_offset_samples': offset, 'block_samples': block_samples,
        'scope': 'Rehashed cached domain-count matrix and actual runtime permutation; checked data/order/source-window geometry. No repeated raw-token byte scan, no old recipe qualification reused, no post-hoc mixing pass threshold, no claim about within-domain language or quality balance or cause of validation changes.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--steps', nargs='+', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve(), args.review, args.steps)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)+'\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output),
        'multilingual_weights': [w['domain_weights']['multilingual'] for w in result['windows']]}))
