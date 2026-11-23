import hashlib
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import block_permutation_reader as reader
from derive_block_prefix_probe import derive


@pytest.fixture
def manifest(tmp_path):
    values = np.array([2, 0, 3, 1], dtype='<u4')
    path = tmp_path/'blocks.u32'
    values.tofile(path)
    document = {'schema_version': 2, 'scope': 'PRODUCTION', 'status': 'PASS',
        'purpose': 'TRAINING_CAUSAL_BLOCK_PERMUTATION', 'algorithm': reader.ALGORITHM,
        'seed': 'test', 'seq_len': 8, 'block_samples': 4, 'release_manifest_sha256': 'release',
        'dataset_id': 'dataset', 'snapshot_id': 'snapshot',
        'stages': [{'stage_index': 0, 'stage_sample_limit': 16, 'permutation_population_samples': 16, 'stream_meta_sha256': 'meta',
                    'permutation': {'file': path.name, 'bytes': 16, 'dtype': '<u4',
                                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}]}
    manifest_path = tmp_path/'manifest.json'
    manifest_path.write_text(json.dumps(document))
    return manifest_path, document, path


def load(path):
    return reader.load_permutations(path, seq_len=8, stage_samples=[16], stage_capacities=[16],
        stream_meta_sha256=['meta'], release_sha256='release', dataset_id='dataset', snapshot_id='snapshot')


def test_bound_immutable_arrays_cover_exact_physical_samples(manifest):
    path, _, _ = manifest
    result = load(path)
    observed = [reader.map_sample(i, result['arrays'][0], 4) for i in range(16)]
    assert observed == list(range(8, 12))+list(range(4))+list(range(12, 16))+list(range(4, 8))
    assert sorted(observed) == list(range(16))
    assert result['manifest_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not result['arrays'][0].flags.writeable


@pytest.mark.parametrize('field,value', [('seed', ''), ('status', 'DIAGNOSTIC'), ('seq_len', 9),
    ('block_samples', 3), ('release_manifest_sha256', 'wrong'), ('dataset_id', 'wrong'),
    ('snapshot_id', 'wrong'), ('algorithm', 'prefix-sort')])
def test_mismatched_contract_is_rejected(manifest, field, value):
    path, document, _ = manifest
    document[field] = value
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        load(path)


@pytest.mark.parametrize('field,value', [('stage_index', 1), ('stage_sample_limit', 20),
                                       ('stream_meta_sha256', 'wrong')])
def test_stage_binding_mismatch_is_rejected(manifest, field, value):
    path, document, _ = manifest
    document['stages'][0][field] = value
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='stage capacity/meta'):
        load(path)


def test_duplicate_block_is_rejected_even_if_its_hash_is_bound(manifest):
    path, document, artifact = manifest
    np.array([0, 1, 1, 3], dtype='<u4').tofile(artifact)
    document['stages'][0]['permutation']['sha256'] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='exactly once'):
        load(path)


def test_unbound_bytes_are_rejected(manifest):
    path, _, artifact = manifest
    np.array([0, 1, 2, 3], dtype='<u4').tofile(artifact)
    with pytest.raises(ValueError, match='SHA-256'):
        load(path)


def test_artifact_hash_and_numpy_read_share_fd_during_path_rollback(manifest, monkeypatch):
    path, _, artifact = manifest
    alternate, saved = artifact.with_name('alternate'), artifact.with_name('saved')
    np.zeros(4, dtype='<u4').tofile(alternate)
    original = reader.np.fromfile
    def rollback(handle, *args, **kwargs):
        assert hasattr(handle, 'fileno')
        artifact.rename(saved)
        alternate.rename(artifact)
        try:
            return original(handle, *args, **kwargs)
        finally:
            artifact.rename(alternate)
            saved.rename(artifact)
    monkeypatch.setattr(reader.np, 'fromfile', rollback)
    result = load(path)
    assert result['arrays'][0].tolist() == [2, 0, 3, 1]


def probe_manifest(manifest):
    path, document, artifact = manifest
    target = path.with_name('probe.json')
    probe = derive(path, target, stage_limits=[8], parent_stage_indices=[0])
    return target, probe


def load_probe(path, capacity=16):
    return reader.load_permutations(path, seq_len=8, stage_samples=[8], stage_capacities=[capacity],
        stream_meta_sha256=['meta'], release_sha256='release', dataset_id='dataset', snapshot_id='snapshot')


def test_probe_uses_exact_production_prefix_without_renumbering_physical_samples(manifest):
    path, document = probe_manifest(manifest)
    value = load_probe(path)
    assert value['manifest_scope'] == 'PREFIX_PROBE'
    assert value['population_stage_samples'] == [16]
    # The first physical block is 2, beyond a probe's own logical block range.
    assert [reader.map_sample(i, value['arrays'][0], 4) for i in range(8)] == list(range(8, 12))+list(range(4))


def test_probe_missing_parent_hash_cannot_disable_validation(manifest):
    path, document = probe_manifest(manifest)
    del document['parent_manifest']['sha256']
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='parent.*SHA-256'):
        load_probe(path)


def test_probe_must_fit_the_actual_stream_physical_capacity(manifest):
    path, _ = probe_manifest(manifest)
    with pytest.raises(ValueError, match='physical.*capacity'):
        load_probe(path, capacity=12)


def test_probe_cannot_substitute_another_valid_permutation(manifest):
    path, document = probe_manifest(manifest)
    alternate = path.with_name('different.u32')
    np.arange(4, dtype='<u4').tofile(alternate)
    document['stages'][0]['permutation'].update(file=alternate.name,
        sha256=hashlib.sha256(alternate.read_bytes()).hexdigest())
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='changed its production permutation'):
        load_probe(path)


def test_full_production_scope_cannot_hide_a_smaller_consumption_budget(manifest):
    path, document, _ = manifest
    document['stages'][0]['stage_sample_limit'] = 8
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='exactly the configured consumed'):
        load_probe(path)


def test_probe_derivation_is_idempotent_and_refuses_contract_overwrite(manifest):
    path, _ = probe_manifest(manifest)
    parent_path, document, _ = manifest
    original = path.read_bytes()
    derive(parent_path, path, stage_limits=[8], parent_stage_indices=[0])
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match='different immutable contract'):
        derive(parent_path, path, stage_limits=[4], parent_stage_indices=[0])


@pytest.mark.parametrize('limits,indices', [([20], [0]), ([7], [0]), ([8, 8], [0, 0]), ([8], [1])])
def test_probe_derivation_cannot_exceed_or_reorder_production(manifest, limits, indices):
    parent_path, _, _ = manifest
    with pytest.raises(ValueError):
        derive(parent_path, parent_path.with_name('invalid-probe.json'),
               stage_limits=limits, parent_stage_indices=indices)


def shifted_manifest(manifest):
    path, document, artifact = manifest
    document['schema_version'] = 3
    document['stages'][0]['source_sample_offset'] = 4
    (np.fromfile(artifact, dtype='<u4')+1).tofile(artifact)
    document['stages'][0]['permutation']['sha256'] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    path.write_text(json.dumps(document))
    return path, document, artifact


def load_shifted(path, capacity=20, samples=16):
    return reader.load_permutations(path, seq_len=8, stage_samples=[samples], stage_capacities=[capacity],
        stream_meta_sha256=['meta'], release_sha256='release', dataset_id='dataset', snapshot_id='snapshot')


def test_shifted_source_window_keeps_exact_contexts_and_supervised_budget(manifest):
    path, _, _ = shifted_manifest(manifest)
    result = load_shifted(path)
    observed = [reader.map_sample(i, result['arrays'][0], 4) for i in range(16)]
    assert sorted(observed) == list(range(4, 20))
    assert result['source_sample_offsets'] == [4]
    source = np.arange(161)
    labels = np.concatenate([source[i*8+1:(i+1)*8+1] for i in observed])
    np.testing.assert_array_equal(np.sort(labels), np.arange(33, 161))
    for first in range(0, 16, 4):
        assert observed[first:first+4] == list(range(observed[first], observed[first]+4))


@pytest.mark.parametrize('offset', [-4, 2, True, None])
def test_source_window_offset_must_be_explicit_nonnegative_and_block_aligned(manifest, offset):
    path, document, _ = shifted_manifest(manifest)
    document['stages'][0]['source_sample_offset'] = offset
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='whole-block offset'):
        load_shifted(path)


def test_shifted_window_cannot_exceed_physical_capacity(manifest):
    path, _, _ = shifted_manifest(manifest)
    with pytest.raises(ValueError, match='physical.*capacity'):
        load_shifted(path, capacity=19)


def test_schema_two_cannot_hide_a_nonzero_source_window(manifest):
    path, document, _ = shifted_manifest(manifest)
    document['schema_version'] = 2
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='schema 3'):
        load_shifted(path)


def test_shifted_probe_inherits_absolute_source_blocks_without_rebasing(manifest):
    parent_path, _, _ = shifted_manifest(manifest)
    target = parent_path.with_name('shifted-probe.json')
    derive(parent_path, target, stage_limits=[8], parent_stage_indices=[0])
    parent, probe = load_shifted(parent_path), load_shifted(target, samples=8)
    expected = [reader.map_sample(i, parent['arrays'][0], 4) for i in range(8)]
    assert [reader.map_sample(i, probe['arrays'][0], 4) for i in range(8)] == expected
    assert min(expected) >= 4
    assert probe['source_sample_offsets'] == [4]


def test_probe_cannot_change_its_parent_source_window(manifest):
    parent_path, _, _ = shifted_manifest(manifest)
    target = parent_path.with_name('shifted-probe.json')
    document = derive(parent_path, target, stage_limits=[8], parent_stage_indices=[0])
    document['stages'][0]['source_sample_offset'] = 0
    target.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='changed its production permutation'):
        load_shifted(target, samples=8)


@pytest.mark.parametrize('prototype,production', [
    ('block_permutation_reader.py', 'pretrain/backends/torchtitan/yuformer_ext/block_permutation.py'),
    ('derive_block_prefix_probe.py', 'pretrain/scripts/derive_block_prefix_probe.py')])
def test_diagnostic_and_actual_runtime_implementations_cannot_diverge(prototype, production):
    root = Path(__file__).resolve().parents[3]
    assert (root/'experiments/e4_curriculum/scripts'/prototype).read_bytes() == (root/production).read_bytes()
