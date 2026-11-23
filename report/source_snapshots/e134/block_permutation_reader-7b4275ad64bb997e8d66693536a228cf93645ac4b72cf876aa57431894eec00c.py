"""Candidate runtime reader for fully bound, immutable causal-block indices."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ALGORITHM = 'blake2b-128-independent-causal-block-sort-v1'


def _identity(stat):
    return {'bytes': stat.st_size, 'device': stat.st_dev,
            'inode': stat.st_ino, 'mtime_ns': stat.st_mtime_ns}


def _read_bound_json(path):
    with path.open('rb') as handle:
        before = _identity(os.fstat(handle.fileno()))
        raw = handle.read()
        value = json.loads(raw)
        if _identity(os.fstat(handle.fileno())) != before or _identity(path.stat()) != before:
            raise ValueError('block permutation manifest changed while reading')
    return value, hashlib.sha256(raw).hexdigest()


def load_permutations(manifest_path, *, seq_len, stage_samples, stream_meta_sha256,
                      release_sha256, dataset_id, snapshot_id):
    path = Path(manifest_path)
    manifest, manifest_sha = _read_bound_json(path)
    if (manifest.get('schema_version') != 1 or manifest.get('status') != 'PASS'
            or manifest.get('purpose') != 'TRAINING_CAUSAL_BLOCK_PERMUTATION'
            or manifest.get('algorithm') != ALGORITHM or not isinstance(manifest.get('seed'), str)
            or not manifest['seed'] or manifest.get('seq_len') != seq_len
            or manifest.get('release_manifest_sha256') != release_sha256
            or manifest.get('dataset_id') != dataset_id or manifest.get('snapshot_id') != snapshot_id):
        raise ValueError('block permutation manifest data/algorithm binding mismatch')
    block_samples = manifest.get('block_samples')
    if type(block_samples) is not int or block_samples < 1:
        raise ValueError('block permutation block_samples must be an explicit positive integer')
    rows = manifest.get('stages')
    if not isinstance(rows, list) or len(rows) != len(stage_samples) or len(rows) != len(stream_meta_sha256):
        raise ValueError('block permutation stage roster mismatch')
    arrays, evidence = [], []
    for index, row in enumerate(rows):
        samples = stage_samples[index]
        if (type(samples) is not int or samples < 1 or samples % block_samples
                or row.get('stage_index') != index or row.get('stage_sample_limit') != samples
                or row.get('stream_meta_sha256') != stream_meta_sha256[index]):
            raise ValueError('block permutation stage capacity/meta binding mismatch')
        blocks = samples//block_samples
        artifact = row.get('permutation') or {}
        name = artifact.get('file')
        if (not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts
                or artifact.get('bytes') != blocks*4 or artifact.get('dtype') != '<u4'):
            raise ValueError('block permutation artifact geometry/path mismatch')
        artifact_path = path.parent/name
        with artifact_path.open('rb') as handle:
            before = _identity(os.fstat(handle.fileno()))
            if before['bytes'] != blocks*4:
                raise ValueError('block permutation artifact size mismatch')
            digest = hashlib.sha256()
            while value := handle.read(8 << 20):
                digest.update(value)
            if digest.hexdigest() != artifact.get('sha256'):
                raise ValueError('block permutation artifact SHA-256 mismatch')
            handle.seek(0)
            array = np.fromfile(handle, dtype='<u4')
            if (_identity(os.fstat(handle.fileno())) != before
                    or _identity(artifact_path.stat()) != before):
                raise ValueError('block permutation artifact changed while reading')
        if not np.array_equal(np.sort(array), np.arange(blocks, dtype='<u4')):
            raise ValueError('block permutation must cover every physical block exactly once')
        array.flags.writeable = False
        arrays.append(array)
        evidence.append({'file': str(artifact_path.resolve()), 'sha256': artifact['sha256'],
                         'bytes': blocks*4, 'stage_index': index, 'stage_sample_limit': samples})
    return {'manifest_sha256': manifest_sha, 'block_samples': block_samples,
            'arrays': arrays, 'artifacts': evidence, 'seed': manifest['seed'], 'algorithm': ALGORITHM}


def map_sample(local_sample, array, block_samples):
    if local_sample < 0 or block_samples < 1:
        raise ValueError('invalid logical sample/block size')
    block, within = divmod(local_sample, block_samples)
    if block >= len(array):
        raise IndexError('block permutation stage capacity exhausted')
    return int(array[block])*block_samples+within
