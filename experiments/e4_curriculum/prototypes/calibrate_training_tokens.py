#!/usr/bin/env python3
"""Measure tokenizer expansion separately on each selected training quality band."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import errno

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT / 'pretrain/scripts'), str(ROOT / 'pretrain/src')]
import materialize_tokens as materializer
from source_document_binding import verified_source_document_binding
from yuformer_pretrain.lease import RunLease

DOMAINS = ('web', 'code', 'math', 'chinese', 'academic', 'books', 'encyclopedia', 'qa', 'multilingual')
REPORT = 'data/processed/audits/training_tokenizer_calibration.json'
MIN_REFERENCE_TOKENS = 100_000_000


def digest(path):
    return materializer.file_sha256(Path(path))


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def quality_policy(config):
    return {**{key: config.get(key) for key in (
        'score_field', 'seed', 'included_sources', 'quota_fill_policy', 'curriculum', 'val')},
        'domain_quality': {domain: {key: value for key, value in config['domains'][domain].items()
                                  if key != 'weight'} for domain in DOMAINS}}


def sample_order(parent, target, destination, *, population, expected_parent_sha256, seed, phase):
    from uniform_selected_sample import sample_order as independent_sample
    return independent_sample(parent, target, destination, population=population,
        expected_parent_sha256=expected_parent_sha256, seed=seed, phase=phase)


def selected_population(root, release, row):
    manifest_path = root/row['manifest']
    record = next(item for item in release['artifacts'] if item['path'] == row['manifest'])
    with manifest_path.open('rb') as handle:
        before = materializer._stat_identity(os.fstat(handle.fileno()))
        if materializer._hash_open_file(handle) != record['sha256']:
            raise ValueError('selected calibration population manifest differs from release')
        population = json.load(handle)['selected']
        if (materializer._stat_identity(os.fstat(handle.fileno())) != before or
                materializer.source_file_identity(manifest_path) != before):
            raise ValueError('calibration population manifest changed while reading')
    return population


def measure(order, lengths_path, *, expected_order_sha256=None, expected_lengths_sha256=None):
    with order.open('rb') as handle:
        identity = materializer._stat_identity(os.fstat(handle.fileno()))
        actual = materializer._hash_open_file(handle)
        if expected_order_sha256 is not None and actual != expected_order_sha256:
            raise ValueError('calibration measurement order SHA-256 mismatch')
        table = pq.read_table(handle, columns=['domain', 'n_tokens'])
        if (materializer._stat_identity(os.fstat(handle.fileno())) != identity
                or materializer.source_file_identity(order) != identity):
            raise ValueError('calibration measurement order changed while reading')
    with lengths_path.open('rb') as handle:
        identity = materializer._stat_identity(os.fstat(handle.fileno()))
        actual = materializer._hash_open_file(handle)
        if expected_lengths_sha256 is not None and actual != expected_lengths_sha256:
            raise ValueError('calibration measurement ledger SHA-256 mismatch')
        handle.seek(0)
        lengths = np.fromfile(handle, dtype='<u4')
        if (materializer._stat_identity(os.fstat(handle.fileno())) != identity
                or materializer.source_file_identity(lengths_path) != identity):
            raise ValueError('calibration measurement ledger changed while reading')
    if len(lengths) != table.num_rows or np.any(lengths == 0):
        raise ValueError('calibration length ledger does not cover every document')
    domains = np.asarray(table['domain'].to_pylist())
    refs = table['n_tokens'].combine_chunks().to_numpy(zero_copy_only=False)
    if set(domains) != set(DOMAINS) or np.any(refs <= 0):
        raise ValueError('invalid calibration domain roster or reference lengths')
    values = {}
    for domain in DOMAINS:
        mask = domains == domain
        expected, actual = int(refs[mask].sum(dtype=np.uint64)), int(lengths[mask].sum(dtype=np.uint64))
        values[domain] = {'reference_tokens': expected, 'actual_tokens_with_eod': actual,
                          'docs': int(mask.sum()), 'ratio': actual / expected}
    return values


def build(root, work, tokenizer, workers, target, parent_release=None, *,
          seed='yuformer-e133-independent-calibration-20261114-v1'):
    release_path = parent_release or root / 'data/processed/releases/pretrain-1t-frozen.release.json'
    release = json.loads(release_path.read_text())
    if release.get('status') != 'PASS':
        raise ValueError('calibration requires a PASS selected training release')
    work.mkdir(parents=True, exist_ok=True)
    archived_release = work / 'parent.release.json'
    if archived_release.exists() and archived_release.read_bytes() != release_path.read_bytes():
        raise ValueError('calibration work directory belongs to another training release')
    if not archived_release.exists():
        archived_release.write_bytes(release_path.read_bytes())
    from uniform_selected_sample import SAMPLING
    sampler_path = root/'experiments/e4_curriculum/scripts/uniform_selected_sample.py'
    if not isinstance(seed, str) or not seed:
        raise ValueError('independent calibration requires an explicit seed')
    report = {'schema_version': 2, 'status': 'RUNNING', 'purpose': 'TRAINING_TOKENIZER_CALIBRATION',
              'includes_appended_eod': True, 'sampling': SAMPLING, 'seed': seed,
              'sampler_tool_sha256': digest(sampler_path),
              'minimum_reference_tokens_per_domain': target,
              'tool_sha256': digest(Path(__file__)), 'tokenizer_sha256': digest(tokenizer),
              'parent_release': str(archived_release.relative_to(root)),
              'parent_release_sha256': digest(archived_release), 'stages': {}}
    for stage in ('warmup', 'stable', 'anneal'):
        row = release['stages'][stage]
        directory = work / stage
        directory.mkdir(exist_ok=True)
        parent = directory / 'parent-order.parquet'
        if not parent.exists():
            try:
                os.link(root / row['order'], parent)
            except OSError as error:
                if error.errno not in (errno.EPERM, errno.EXDEV, errno.EOPNOTSUPP):
                    raise
                temporary = parent.with_name(f'.{parent.name}.{os.getpid()}')
                shutil.copyfile(root / row['order'], temporary)
                os.replace(temporary, parent)
        config = yaml.safe_load((root / row['config']).read_text())
        config_path = directory / 'parent-config.yaml'
        if not config_path.exists():
            config_path.write_text(yaml.safe_dump(config, sort_keys=True))
        if quality_policy(config) != quality_policy(yaml.safe_load(config_path.read_text())):
            raise ValueError('training quality policy changed during calibration')
        order = directory / 'sample.parquet'
        selection_path = directory / 'selection.json'
        population = selected_population(root, release, row)
        original = next(item for item in release['artifacts'] if item['path'] == row['order'])
        if not selection_path.exists():
            selection = sample_order(parent, target, order, population=population,
                expected_parent_sha256=original['sha256'], seed=seed, phase=stage)
            if selection['parent_order_sha256'] != original['sha256']:
                raise ValueError('selected parent order differs from frozen release')
            selection['sample_sha256'] = digest(order)
            selection['sample_stat'] = materializer.source_file_identity(order)
            selection['minimum_reference_tokens_per_domain'] = target
            atomic_json(selection_path, selection)
        selection = json.loads(selection_path.read_text())
        if (selection.get('sampling') != SAMPLING or selection.get('seed') != seed
                or selection.get('phase') != stage
                or selection.get('sampler_tool_sha256') != report['sampler_tool_sha256']
                or selection.get('parent_order_sha256') != original['sha256']
                or selection.get('population_docs_by_domain') != {d: population[d]['selected_docs'] for d in DOMAINS}
                or selection.get('population_reference_tokens_by_domain') != {d: population[d]['selected_tokens'] for d in DOMAINS}
                or selection['minimum_reference_tokens_per_domain'] != target
                or digest(order) != selection['sample_sha256']
                or materializer.source_file_identity(order) != selection['sample_stat']):
            raise ValueError('calibration sample identity changed')
        snapshot = directory / 'sample.release.json'
        atomic_json(snapshot, {'schema_version': 1, 'status': 'PASS',
                              'snapshot_id': f'calibration-{selection["sample_sha256"]}',
                              'release_name': 'calibration-only',
                              'artifacts': [{'path': str(order.relative_to(root)),
                                             'sha256': selection['sample_sha256'],
                                             **selection['sample_stat']}]})
        meta_path = directory / 'sample.meta.json'
        if not meta_path.exists():
            command = [sys.executable, str(root / 'pretrain/scripts/materialize_tokens.py'),
                       '--order', str(order), '--s3-root', str(root / 'data/processed/s3'),
                       '--tokenizer', str(tokenizer), '--tokenizer-gate', str(root / 'data/TOKENIZER_READY.json'),
                       '--data-snapshot', str(snapshot), '--out-dir', str(directory), '--name', 'sample',
                       '--workers', str(workers), '--write-doc-lengths']
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
                               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
            print(f'calibrating {stage}: {selection["docs"]} documents', flush=True)
            subprocess.run(command, env=environment, check=True)
        meta = json.loads(meta_path.read_text())
        lengths = directory / 'sample.doc_lengths.u32'
        if (not verified_source_document_binding(meta)
                or meta['tokenizer_sha256'] != report['tokenizer_sha256']
                or meta['order_sha256'] != selection['sample_sha256']
                or digest(lengths) != meta['doc_lengths']['sha256']
                or materializer.source_file_identity(directory / 'sample.tokens.bin') != meta['stream_stat_identity']):
            raise ValueError('calibration token stream provenance mismatch')
        measurements = measure(order, lengths, expected_order_sha256=meta['order_sha256'],
                               expected_lengths_sha256=meta['doc_lengths']['sha256'])
        if sum(v['actual_tokens_with_eod'] for v in measurements.values()) != meta['total_tokens']:
            raise ValueError('calibration measured token counts do not close')
        if any(v['reference_tokens'] < target for v in measurements.values()):
            raise ValueError('insufficient measured training calibration tokens')
        stage_report = {'status': 'PASS', 'quality_policy': quality_policy(config),
                        'domains': measurements, 'source_roster_fingerprint':
                            meta['source_document_binding']['roster_fingerprint'],
                        'selection': selection, 'evidence': {}}
        for evidence in (order, lengths, meta_path, selection_path, snapshot, config_path):
            stage_report['evidence'][str(evidence.relative_to(root))] = digest(evidence)
        report['stages'][stage] = stage_report
        atomic_json(work / 'progress.json', report)
    report['status'] = 'PASS'
    atomic_json(root / REPORT, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--parent-release', type=Path,
                        help='Selected training order release; archived immutable parents are accepted')
    parser.add_argument('--seed', default='yuformer-e133-independent-calibration-20261114-v1')
    parser.add_argument('--workers', type=int, default=128)
    parser.add_argument('--reference-tokens-per-domain', type=int, default=MIN_REFERENCE_TOKENS)
    args = parser.parse_args()
    if args.reference_tokens_per_domain < MIN_REFERENCE_TOKENS:
        parser.error('at least 100M reference tokens per phase and domain are required')
    with RunLease(args.work_dir, 'training-tokenizer-calibration'):
        result = build(args.root.resolve(), args.work_dir.resolve(), args.tokenizer.resolve(),
                       args.workers, args.reference_tokens_per_domain,
                       args.parent_release.resolve() if args.parent_release else None, seed=args.seed)
    print(json.dumps({'status': result['status'], 'report': REPORT}), flush=True)


if __name__ == '__main__':
    main()
