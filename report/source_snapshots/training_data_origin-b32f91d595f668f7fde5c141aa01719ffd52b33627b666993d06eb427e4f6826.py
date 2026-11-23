"""Candidate fail-closed data provenance validation for training-only adoption."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml

FROZEN = 'experiments/e4_curriculum/results/frozen_1t_recipe.json'
BASE = 'pretrain/configs/pretrain_base.yaml'
STAGES = ('warmup', 'stable', 'anneal')
_UNBOUND = object()


def read_bound(path, expected=_UNBOUND):
    if expected is not _UNBOUND and (not isinstance(expected, str) or len(expected) != 64 or
            any(c not in '0123456789abcdef' for c in expected)):
        raise ValueError('missing or invalid required data-origin SHA-256')
    def identity(stat):
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns
    with path.open('rb') as handle:
        before = identity(os.fstat(handle.fileno()))
        raw = handle.read()
        digest = hashlib.sha256(raw).hexdigest()
        if expected is not _UNBOUND and digest != expected:
            raise ValueError('data-origin byte identity mismatch: ' + str(path))
        if identity(os.fstat(handle.fileno())) != before or identity(path.stat()) != before:
            raise ValueError('data-origin file changed during read')
    return raw, digest


def resolve_data_origin(root, release_path, frozen_path=None):
    root, release_path = Path(root), Path(release_path)
    frozen_path = Path(frozen_path) if frozen_path else root/FROZEN
    release_raw, release_sha = read_bound(release_path)
    frozen_raw, frozen_sha = read_bound(frozen_path)
    release, frozen = json.loads(release_raw), json.loads(frozen_raw)
    if release.get('status') != 'PASS' or frozen.get('status') != 'PASS':
        raise ValueError('data release and current training recipe must PASS')
    if release.get('frozen_recipe_sha256') == frozen_sha:
        if frozen.get('data_origin') is not None:
            raise ValueError('direct frozen data release must not hide a training adoption')
        return {'mode': 'DIRECT_FROZEN_RECIPE', 'data_recipe': str(frozen_path),
                'data_recipe_sha256': frozen_sha, 'release_sha256': release_sha}
    origin = frozen.get('data_origin') or {}
    if (origin.get('schema_version') != 1 or origin.get('kind') != 'TRAINING_ONLY_WITH_EXACT_DATA_REUSE'
            or origin.get('release_manifest_sha256') != release_sha
            or origin.get('parent_frozen_recipe_sha256') != release.get('frozen_recipe_sha256')):
        raise ValueError('training adoption has no exact immutable data-origin binding')
    relative = origin.get('archive_manifest')
    if not isinstance(relative, str) or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('invalid data-origin archive path')
    archive_path = root/relative
    archive_raw, archive_sha = read_bound(archive_path, origin.get('archive_manifest_sha256'))
    archive = json.loads(archive_raw)
    if (archive.get('schema_version') != 1 or archive.get('status') != 'PASS_IMMUTABLE_ORIGIN_ARCHIVE'
            or archive.get('purpose') != 'PRESERVE_ORIGINAL_RECIPE_BYTES_NOT_PRODUCTION_ADOPTION'
            or archive.get('frozen_recipe_sha256') != release.get('frozen_recipe_sha256')):
        raise ValueError('invalid original recipe archive')
    records = archive.get('files')
    if not isinstance(records, dict) or not records:
        raise ValueError('original recipe archive has no objects')
    contents = {}
    for name, row in records.items():
        asset = row.get('archive')
        if not isinstance(asset, str) or Path(asset).is_absolute() or '..' in Path(asset).parts:
            raise ValueError('invalid archived object path')
        contents[name], _ = read_bound(root/asset, row.get('sha256'))
        if len(contents[name]) != row.get('bytes'):
            raise ValueError('archived object size mismatch')
    if FROZEN not in records or records[FROZEN]['sha256'] != release['frozen_recipe_sha256']:
        raise ValueError('original recipe bytes are missing from archive')
    parent = json.loads(contents[FROZEN])
    if parent.get('status') != 'PASS' or parent.get('data_origin') is not None:
        raise ValueError('only one explicit training adoption from the original data recipe is supported')
    original_bindings = dict(parent.get('inputs') or {})
    for name, expected in (parent.get('outputs') or {}).items():
        if name in original_bindings and original_bindings[name] != expected:
            raise ValueError('conflicting original recipe bindings')
        original_bindings[name] = expected
    for name, expected in original_bindings.items():
        if name not in records or records[name]['sha256'] != expected:
            raise ValueError('incomplete original recipe archive: ' + name)
    release_name = str(release_path.relative_to(root))
    if contents.get(release_name) != release_raw:
        raise ValueError('the data release itself changed during training-only adoption')
    if (frozen.get('stages') != parent.get('stages') or set(frozen.get('stages', {})) != set(STAGES)
            or frozen.get('selected_tokens_per_step') != parent.get('selected_tokens_per_step')
            or frozen.get('optimizer_stage_steps') != parent.get('optimizer_stage_steps')
            or frozen.get('mixture_evidence') != parent.get('mixture_evidence')):
        raise ValueError('training adoption changed selected data quotas, mixture, or WSD stages')
    for phase in STAGES:
        name = parent['stages'][phase]['config']
        expected = parent['outputs'].get(name)
        if expected is None or frozen.get('outputs', {}).get(name) != expected:
            raise ValueError('training adoption changed a frozen data config binding')
        read_bound(root/name, expected)
    old_base = yaml.safe_load(contents[BASE])
    current_raw, _ = read_bound(root/BASE, frozen.get('outputs', {}).get(BASE))
    current_base = yaml.safe_load(current_raw)
    old_data = dict(old_base['data'])
    current_data = dict(current_base['data'])
    if old_data.pop('sampling_policy', 'sequential') != 'sequential':
        raise ValueError('original recipe must use sequential causal consumption')
    if current_data.pop('sampling_policy', None) != 'block_permutation':
        raise ValueError('training adoption must explicitly use independent causal-block order')
    permutation_manifest = current_data.pop('block_permutation_manifest', None)
    if not isinstance(permutation_manifest, str) or not permutation_manifest:
        raise ValueError('training adoption has no independent order manifest')
    if current_data != old_data:
        raise ValueError('training adoption changed data identity or the exact consumed sample set')
    if current_base['parallel'] != old_base['parallel']:
        raise ValueError('training-only adoption changed the qualified parallel topology')
    for key in ('budget_tokens', 'expected_iters', 'micro_batch_size', 'grad_accum'):
        if current_base['train'].get(key) != old_base['train'].get(key):
            raise ValueError('training adoption changed token/update capacity: ' + key)
    for key in ('lr_schedule', 'warmup_steps', 'decay_steps'):
        if current_base['optim'].get(key) != old_base['optim'].get(key):
            raise ValueError('training adoption changed the WSD boundaries: ' + key)
    return {'mode': 'TRAINING_ONLY_WITH_EXACT_DATA_REUSE',
            'data_recipe': str(root/records[FROZEN]['archive']),
            'data_recipe_sha256': release['frozen_recipe_sha256'],
            'release_sha256': release_sha, 'archive_manifest_sha256': archive_sha,
            'training_recipe_sha256': frozen_sha,
            'permutation_manifest': permutation_manifest,
            'note': 'Token/ledger/mixture/overlap audits and new-model qualification are separately mandatory.'}
