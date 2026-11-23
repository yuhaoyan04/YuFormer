#!/usr/bin/env python3
"""Freeze identical 4K heldout targets for paired evaluation at every context."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

DOMAINS = ('academic', 'books', 'chinese', 'code', 'encyclopedia', 'math',
           'multilingual', 'qa', 'web')


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def identity(stat):
    return dict(bytes=stat.st_size, device=stat.st_dev, inode=stat.st_ino, mtime_ns=stat.st_mtime_ns)


def seal(path, value):
    with path.open('xb') as handle:
        handle.write((json.dumps(value, sort_keys=True, indent=2)+'\n').encode())
        handle.flush()
        os.fsync(handle.fileno())


def prepare(*, root, audit_file, gate_file, release_file, output, samples, contexts):
    root, audit_file, gate_file, release_file, output = [Path(p).resolve()
        for p in (root, audit_file, gate_file, release_file, output)]
    if (output.exists() or type(samples) is not int or samples <= 0 or not contexts
            or len(set(contexts)) != len(contexts) or any(s not in (4096, 16384, 65536) for s in contexts)):
        raise ValueError('new output, positive sample count and an explicit unique context roster are required')
    audit, gate, release = [json.loads(p.read_bytes()) for p in (audit_file, gate_file, release_file)]
    if (audit.get('status') != 'PASS' or gate.get('status') != 'PASS' or release.get('status') != 'PASS'
            or release.get('dataset_id') != gate.get('dataset_id') or release.get('snapshot_id') != gate.get('snapshot_id')):
        raise ValueError('paired short core requires the actual admitted heldout streams and tokenizer/data release')
    tokenizer = Path(gate['tokenizer_path'])
    if not tokenizer.is_absolute():
        tokenizer = root/tokenizer
    if digest(tokenizer) != gate['tokenizer_sha256']:
        raise ValueError('actual heldout tokenizer changed')
    pins = {str(p): digest(p) for p in [Path(__file__).resolve(), audit_file, gate_file, release_file, tokenizer]}
    streams = {row['name']: row for row in audit['streams']}
    sources = []
    for domain in DOMAINS:
        row = streams['val_'+domain]
        stream, meta, order = [Path(row['artifacts'][k]) for k in ('stream', 'meta', 'order')]
        metadata = json.loads(meta.read_bytes())
        expected = {k: row['actual_'+k] for k in ('bytes', 'device', 'inode', 'mtime_ns')}
        if (row.get('status') != 'PASS' or row.get('errors') or not all(row['checks'].values())
                or any(not p.is_absolute() for p in (stream, meta, order))
                or metadata['dtype'] not in ('<u2', '<u4')
                or metadata['total_tokens']*np.dtype(metadata['dtype']).itemsize != expected['bytes']
                or metadata['total_tokens'] != row['actual_total_tokens_with_eod']
                or identity(stream.stat()) != expected or digest(stream) != row['actual_sha256']
                or metadata['sha256'] != row['actual_sha256']
                or metadata['tokenizer_sha256'] != gate['tokenizer_sha256']
                or digest(order) != metadata['order_sha256']
                or metadata['total_tokens'] < samples*4096+1):
            raise ValueError('actual full-byte heldout binding/capacity failed: '+domain)
        pins.update({str(p): digest(p) for p in (stream, meta, order)})
        sources.append((domain, stream, metadata, expected))
    output.mkdir(parents=True)
    manifest = {'schema_version': 1, 'status': 'PREPARING_IMMUTABLE_PAIRED_NLL_CASE_MANIFEST',
        'scope': 'ALL_NINE_SHORT_CORE_IDENTICAL_4K_TARGETS_AT_EACH_NATIVE_CONTEXT',
        'tokenizer_sha256': gate['tokenizer_sha256'], 'dataset_id': gate['dataset_id'],
        'samples_per_domain': samples, 'target_labels_per_case': 4096, 'contexts': contexts,
        'source_inputs': pins, 'files': [], 'runtime': {'python': sys.version, 'numpy': np.__version__},
        'metric_note': 'Token NLL on the exact original heldout stream prefix. This suite is not byte-normalized BPB.'}
    for domain_index, (domain, stream, meta, expected) in enumerate(sources):
        path = output/('short-core-'+domain+'.jsonl')
        with stream.open('rb') as source, path.open('x') as handle:
            if identity(os.fstat(source.fileno())) != expected:
                raise ValueError('heldout stream changed before mapping')
            tokens = np.memmap(source, mode='r', dtype=meta['dtype'])
            for local in range(samples):
                offset = local*4096
                values = tokens[offset:offset+4097]
                if len(values) != 4097 or int(values.max()) >= gate['vocab_size']:
                    raise ValueError('short-core target context/vocabulary mismatch')
                row = {'index': domain_index*samples+local, 'domain': domain,
                    'source_token_offset': offset, 'input_seq_len': 4096, 'token_ids': values.tolist()}
                handle.write(json.dumps(row, sort_keys=True, separators=(',', ':'))+'\n')
            handle.flush()
            os.fsync(handle.fileno())
            if identity(os.fstat(source.fileno())) != expected or identity(stream.stat()) != expected:
                raise ValueError('heldout stream changed during materialization')
        sha = digest(path)
        for context in contexts:
            manifest['files'].append({'file': str(path), 'sha256': sha, 'suite': 'short_core',
                'domain': domain, 'seq_len': context, 'input_seq_len': 4096, 'samples': samples})
    if any(digest(p) != sha for p, sha in pins.items()):
        raise ValueError('heldout source or producer changed before publication')
    manifest['status'] = 'PASS_IMMUTABLE_PAIRED_NLL_CASE_MANIFEST'
    seal(output/'manifest.json', manifest)
    descriptor = os.open(output, os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'stream-audit', 'tokenizer-gate', 'data-release', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--samples-per-domain', type=int, required=True)
    parser.add_argument('--contexts', type=int, nargs='+', required=True)
    args = parser.parse_args()
    result = prepare(root=args.root, audit_file=args.stream_audit, gate_file=args.tokenizer_gate,
        release_file=args.data_release, output=args.output, samples=args.samples_per_domain, contexts=args.contexts)
    print(json.dumps({'status': result['status'], 'scope': result['scope'], 'files': len(result['files'])}, sort_keys=True))
