"""Approximate document-sampling uncertainty for independent tokenizer pilots."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

import calibrate_training_tokens as calibration

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / 'data/processed/tokenizer_calibration/post-selection-11326994-uniform-v2'


def ratio_statistics(reference, actual, population_docs):
    reference, actual = np.asarray(reference, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    n = len(reference)
    if (reference.ndim != 1 or actual.shape != reference.shape or n < 2 or population_docs < n or
            not np.all(np.isfinite(reference)) or not np.all(np.isfinite(actual)) or
            np.any(reference <= 0) or np.any(actual <= 0)):
        raise ValueError('invalid independent document ratio sample')
    ratio = float(actual.sum()/reference.sum())
    residual = actual-ratio*reference
    variance = float((1-n/population_docs)*n/(n-1)*np.dot(residual, residual)/reference.sum()**2)
    standard_error = math.sqrt(variance)
    return {'ratio': ratio, 'variance': variance, 'standard_error': standard_error,
            'approximate_95pct_interval': [ratio-1.96*standard_error, ratio+1.96*standard_error],
            'sample_docs': n, 'population_docs': population_docs,
            'reference_tokens': int(reference.sum()), 'actual_tokens_with_eod': int(actual.sum()),
            'assumption': 'independent uniform document ranks; finite-population ratio linearization; not a formal gate'}


def analyze(phase):
    directory = WORK / phase
    order, ledger = directory/'sample.parquet', directory/'sample.doc_lengths.u32'
    meta_path, selection_path = directory/'sample.meta.json', directory/'selection.json'
    meta, selection = [json.loads(p.read_text()) for p in (meta_path, selection_path)]
    if not calibration.verified_source_document_binding(meta):
        raise ValueError('uniform pilot source binding is incomplete')
    measured = calibration.measure(order, ledger, expected_order_sha256=meta['order_sha256'],
                                   expected_lengths_sha256=meta['doc_lengths']['sha256'])
    # Read the same attested inodes for variance, too.
    with order.open('rb') as oh, ledger.open('rb') as lh:
        handles = [(order, oh, meta['order_sha256']), (ledger, lh, meta['doc_lengths']['sha256'])]
        identities = []
        import os
        for path, handle, expected in handles:
            identities.append(calibration.materializer._stat_identity(os.fstat(handle.fileno())))
            if calibration.materializer._hash_open_file(handle) != expected:
                raise ValueError('uniform uncertainty input SHA mismatch')
        table = pq.read_table(oh, columns=['domain', 'n_tokens'])
        actual = np.fromfile(lh, dtype='<u4')
        for (path, handle, _), identity in zip(handles, identities):
            if (calibration.materializer._stat_identity(os.fstat(handle.fileno())) != identity or
                    calibration.materializer.source_file_identity(path) != identity):
                raise ValueError('uniform uncertainty input changed')
    domains = np.asarray(table['domain'].to_pylist())
    reference = table['n_tokens'].combine_chunks().to_numpy(zero_copy_only=False)
    if len(actual) != len(domains) or sum(v['actual_tokens_with_eod'] for v in measured.values()) != meta['total_tokens']:
        raise ValueError('uniform uncertainty sample counts do not close')
    statistics = {d: ratio_statistics(reference[domains == d], actual[domains == d],
                                    selection['population_docs_by_domain'][d]) for d in calibration.DOMAINS}
    population = selection['population_reference_tokens_by_domain']
    predicted = {d: population[d]*statistics[d]['ratio'] for d in calibration.DOMAINS}
    variances = {d: population[d]**2*statistics[d]['variance'] for d in calibration.DOMAINS}
    total, total_variance = sum(predicted.values()), sum(variances.values())
    weights = {}
    frozen = json.loads((ROOT/'experiments/e4_curriculum/results/frozen_1t_recipe.json').read_text())
    for d in calibration.DOMAINS:
        weight = predicted[d]/total
        variance = ((1-weight)**2*variances[d]+weight**2*(total_variance-variances[d]))/total**2
        se = math.sqrt(variance)
        weights[d] = {'predicted_weight': weight, 'standard_error': se,
                      'approximate_95pct_interval': [weight-1.96*se, weight+1.96*se],
                      'formal_target': frozen['stages'][phase]['target_final_domain_weights'][d]}
    report = {'status': 'MEASURED_DIAGNOSTIC', 'phase': phase, 'domains': statistics,
              'predicted_actual_tokens': total, 'total_standard_error': math.sqrt(total_variance),
              'predicted_weights': weights, 'does_not_replace_formal_full_stream_gates': True,
              'evidence_sha256': {str(p.relative_to(ROOT)): calibration.digest(p)
                                 for p in (order, ledger, meta_path, selection_path)},
              'analysis_tool_sha256': calibration.digest(__file__)}
    calibration.atomic_json(directory/'uncertainty.json', report)
    calibration.atomic_json(ROOT/f'data/health/uniform-calibration-{phase}-uncertainty.json', report)
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('warmup', 'stable', 'anneal'), required=True)
    result = analyze(parser.parse_args().phase)
    print(json.dumps({'phase': result['phase'], 'predicted_actual_tokens': result['predicted_actual_tokens'],
                      'total_standard_error': result['total_standard_error']}), flush=True)
