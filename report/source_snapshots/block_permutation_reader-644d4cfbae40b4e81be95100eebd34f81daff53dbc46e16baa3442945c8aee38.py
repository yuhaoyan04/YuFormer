"""Candidate runtime reader for fully bound, immutable causal-block indices."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

ALGORITHM = 'blake2b-128-independent-causal-block-sort-v1'
_UNBOUND = object()


def _identity(stat):
    return {'bytes': stat.st_size, 'device': stat.st_dev,
            'inode': stat.st_ino, 'mtime_ns': stat.st_mtime_ns}


def _read_bound_json(path, expected_sha256=_UNBOUND):
    with path.open('rb') as handle:
        before = _identity(os.fstat(handle.fileno()))
        raw = handle.read()
        value = json.loads(raw)
        actual = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not _UNBOUND and (not isinstance(expected_sha256, str) or
                len(expected_sha256) != 64 or actual != expected_sha256):
            raise ValueError('parent block permutation manifest SHA-256 mismatch')
        if _identity(os.fstat(handle.fileno())) != before or _identity(path.stat()) != before:
            raise ValueError('block permutation manifest changed while reading')
    return value, actual


def load_permutations(manifest_path, *, seq_len, stage_samples, stage_capacities, stream_meta_sha256,
                      release_sha256, dataset_id, snapshot_id):
    path = Path(manifest_path)
    manifest, manifest_sha = _read_bound_json(path)
    if (manifest.get('schema_version') != 2 or manifest.get('status') != 'PASS'
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
    if (not isinstance(rows, list) or len(rows) != len(stage_samples) or not rows
            or len(rows) != len(stream_meta_sha256) or len(rows) != len(stage_capacities)):
        raise ValueError('block permutation stage roster mismatch')
    scope = manifest.get('scope')
    parent = None
    if scope == 'PRODUCTION':
        if manifest.get('parent_manifest'):
            raise ValueError('production permutation must not hide a probe parent')
    elif scope == 'PREFIX_PROBE':
        binding = manifest.get('parent_manifest') or {}
        name = binding.get('file')
        if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('prefix probe requires a bound production parent manifest')
        parent, _ = _read_bound_json(path.parent/name, binding.get('sha256'))
        common = ('schema_version', 'status', 'purpose', 'algorithm', 'seed', 'seq_len',
                  'block_samples', 'release_manifest_sha256', 'dataset_id', 'snapshot_id')
        if (parent.get('scope') != 'PRODUCTION' or parent.get('parent_manifest') or
                any(parent.get(key) != manifest.get(key) for key in common)):
            raise ValueError('prefix probe differs from its production data/order contract')
    else:
        raise ValueError('block permutation scope must be explicit PRODUCTION or PREFIX_PROBE')
    arrays, evidence = [], []
    populations = []
    previous_parent = -1
    for index, row in enumerate(rows):
        samples = stage_samples[index]
        if (type(samples) is not int or samples < 1 or samples % block_samples
                or row.get('stage_index') != index or row.get('stage_sample_limit') != samples
                or row.get('stream_meta_sha256') != stream_meta_sha256[index]):
            raise ValueError('block permutation stage capacity/meta binding mismatch')
        population = row.get('permutation_population_samples')
        capacity = stage_capacities[index]
        if (type(population) is not int or type(capacity) is not int or population < samples or
                population > capacity or population % block_samples or population//block_samples >= 2**32):
            raise ValueError('physical permutation population exceeds the exact stream capacity')
        if scope == 'PRODUCTION' and population != samples:
            raise ValueError('production permutation must cover exactly the configured consumed sample set')
        if parent is not None:
            parent_index = row.get('parent_stage_index')
            parent_rows = parent.get('stages')
            if (type(parent_index) is not int or parent_index <= previous_parent or
                    not isinstance(parent_rows, list) or parent_index >= len(parent_rows)):
                raise ValueError('prefix probe has an invalid production parent stage')
            original = parent_rows[parent_index]
            if (original.get('stage_index') != parent_index or original.get('stage_sample_limit') != population
                    or original.get('permutation_population_samples') != population
                    or original.get('stream_meta_sha256') != row.get('stream_meta_sha256')
                    or original.get('permutation') != row.get('permutation')):
                raise ValueError('prefix probe changed its production permutation or physical population')
            previous_parent = parent_index
        blocks = population//block_samples
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
        populations.append(population)
        evidence.append({'file': str(artifact_path.resolve()), 'sha256': artifact['sha256'],
                         'bytes': blocks*4, 'stage_index': index, 'stage_sample_limit': samples})
    return {'manifest_sha256': manifest_sha, 'block_samples': block_samples,
            'arrays': arrays, 'artifacts': evidence, 'seed': manifest['seed'], 'algorithm': ALGORITHM,
            'manifest_scope': scope, 'population_stage_samples': populations}


def map_sample(local_sample, array, block_samples):
    if local_sample < 0 or block_samples < 1:
        raise ValueError('invalid logical sample/block size')
    block, within = divmod(local_sample, block_samples)
    if block >= len(array):
        raise IndexError('block permutation stage capacity exhausted')
    return int(array[block])*block_samples+within


def configuration_order_identity(config, *, require_production=False):
    data = config['data']
    policy = data.get('sampling_policy', 'sequential')
    if policy == 'sequential':
        if data.get('block_permutation_manifest') or require_production:
            raise ValueError('production training requires its explicit independent block-order manifest')
        return None
    if policy != 'block_permutation' or not data.get('block_permutation_manifest'):
        raise ValueError('unsupported or incomplete training sample order policy')
    sequence = data['seq_len']
    limits = data.get('stage_sample_limits')
    if (type(sequence) is not int or sequence <= 0 or not isinstance(limits, list) or not limits
            or any(type(n) is not int or n <= 0 for n in limits)
            or len(data.get('meta', [])) != len(limits)):
        raise ValueError('training block order requires exact sequence and stage geometry')
    release, release_sha = _read_bound_json(Path(data['release_manifest']))
    if release.get('status') != 'PASS':
        raise ValueError('block order release must PASS')
    capacities, meta_hashes = [], []
    for name in data['meta']:
        meta, digest = _read_bound_json(Path(name))
        if meta.get('data_snapshot_id') != release.get('snapshot_id'):
            raise ValueError('block order stream snapshot differs from its release')
        total = meta.get('total_tokens')
        if type(total) is not int or total < sequence+1:
            raise ValueError('block order stream has invalid causal capacity')
        capacities.append((total-1)//sequence)
        meta_hashes.append(digest)
    value = load_permutations(data['block_permutation_manifest'], seq_len=sequence,
        stage_samples=limits, stage_capacities=capacities, stream_meta_sha256=meta_hashes,
        release_sha256=release_sha, dataset_id=release['dataset_id'], snapshot_id=release['snapshot_id'])
    if require_production and value['manifest_scope'] != 'PRODUCTION':
        raise ValueError('full production training cannot use a prefix-probe manifest')
    return {key: value[key] for key in ('manifest_sha256', 'manifest_scope', 'algorithm', 'seed',
                                       'block_samples', 'population_stage_samples', 'artifacts')}
