import hashlib
import copy
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import training_block_permutation as blocks
from block_permutation_reader import configuration_order_identity
from derive_block_prefix_probe import derive

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(params=['prototype', 'production'])
def reader(monkeypatch, request):
    package = types.ModuleType('_block_reader_prototype')
    runtime = ROOT/'pretrain/backends/torchtitan/yuformer_ext'
    package.__path__ = ([str(ROOT/'experiments/e4_curriculum/prototypes/yuformer_ext'), str(runtime)]
                        if request.param == 'prototype' else [str(runtime)])
    monkeypatch.setitem(sys.modules, package.__name__, package)
    abstract = types.ModuleType('torchtitan.components.dataloader')
    class Base:
        @dataclass
        class Config:
            pass
    class Exhausted(Exception):
        pass
    abstract.BaseDataLoader, abstract.DataloaderExhaustedError = Base, Exhausted
    monkeypatch.setitem(sys.modules, abstract.__name__, abstract)
    name = package.__name__+'.data'
    spec = importlib.util.spec_from_file_location(name, Path(package.__path__[0])/'data.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    yield module.PretokenizedDataLoader, Exhausted
    for suffix in ('.stream_math', '.block_permutation'):
        monkeypatch.delitem(sys.modules, package.__name__+suffix, raising=False)


@pytest.fixture
def assets(tmp_path):
    gate = {'status': 'PASS', 'dataset_id': 'd', 'snapshot_id': 's',
            'tokenizer_sha256': 'tok', 'vocab_size': 10000}
    for name in ('gate.json', 'release.json'):
        (tmp_path/name).write_text(json.dumps(gate))
    limits, stages, metas, rows, expected = [16, 24, 16], [], [], [], []
    for i, (phase, limit) in enumerate(zip(('warmup', 'stable', 'anneal'), limits)):
        tokens = np.arange((limit+3)*8+1, dtype='<u2')+i*1000
        binary, meta_path, index = tmp_path/f'{phase}.bin', tmp_path/f'{phase}.json', tmp_path/f'{phase}.u32'
        tokens.tofile(binary)
        meta = {'dtype': '<u2', 'total_tokens': len(tokens), 'n_docs': 1,
                'tokenizer_sha256': 'tok', 'vocab_size': 10000, 'data_snapshot_id': 's',
                'document_boundary': {'policy': 'append_eod', 'count': 1}}
        meta_path.write_text(json.dumps(meta))
        permutation = blocks.build_permutation(limit, 4, seed='test-e134', phase=phase)
        permutation.tofile(index)
        rows.append({'stage_index': i, 'stage_sample_limit': limit, 'permutation_population_samples': limit,
            'stream_meta_sha256': hashlib.sha256(meta_path.read_bytes()).hexdigest(),
            'permutation': {'file': index.name, 'sha256': hashlib.sha256(index.read_bytes()).hexdigest(),
                            'bytes': index.stat().st_size, 'dtype': '<u4'}})
        stages.append(str(binary)); metas.append(str(meta_path))
        for sample in range(limit):
            physical = blocks.physical_sample(sample, permutation, 4)
            expected.extend(tokens[physical*8+1:physical*8+9].tolist())
    manifest = {'schema_version': 2, 'scope': 'PRODUCTION', 'status': 'PASS', 'purpose': 'TRAINING_CAUSAL_BLOCK_PERMUTATION',
        'algorithm': blocks.ALGORITHM, 'seed': 'test-e134', 'seq_len': 8, 'block_samples': 4,
        'dataset_id': 'd', 'snapshot_id': 's',
        'release_manifest_sha256': hashlib.sha256((tmp_path/'release.json').read_bytes()).hexdigest(),
        'stages': rows}
    manifest_path = tmp_path/'permutation.json'
    manifest_path.write_text(json.dumps(manifest))
    return {'stages': stages, 'meta': metas, 'stage_sample_limits': limits,
        'tokenizer_gate': str(tmp_path/'gate.json'), 'release_manifest': str(tmp_path/'release.json'),
        'sampling_policy': 'block_permutation', 'block_permutation_manifest': str(manifest_path),
        'emit_stage_id': True}, expected


def make_readers(reader, assets, world=2, batch=2):
    cls, _ = reader
    kwargs, _ = assets
    config = cls.Config(**kwargs)
    return [cls(config, dp_world_size=world, dp_rank=rank, tokenizer=None,
                seq_len=8, local_batch_size=batch) for rank in range(world)]


def test_actual_loader_preserves_targets_stage_boundaries_and_resharded_resume(reader, assets):
    loaders = make_readers(reader, assets)
    iterators = [iter(loader) for loader in loaders]
    observed, states = [], []
    for micro in range(14):
        for iterator in iterators:
            inputs, targets = next(iterator)
            assert inputs['_yuformer_stage_id'].tolist() == [0 if micro < 4 else 1 if micro < 10 else 2]*2
            observed.extend(targets.flatten().tolist())
        if micro % 2 == 1:
            states.append((len(observed), loaders[0].state_dict()))
    assert observed == assets[1]
    assert sorted(observed) == sorted(assets[1])
    for offset, state in states:
        assert len(state['block_permutation_manifest_sha256']) == 64
        for world, batch in ((1, 4), (4, 1)):
            resumed = make_readers(reader, assets, world=world, batch=batch)
            for loader in resumed:
                loader.load_state_dict(state)
            iterators = [iter(loader) for loader in resumed]
            remaining = []
            for _ in range((56-state['consumed_samples'])//4):
                for iterator in iterators:
                    _, targets = next(iterator)
                    remaining.extend(targets.flatten().tolist())
            assert remaining == observed[offset:]
            with pytest.raises(reader[1]):
                next(iterators[0])


def test_checkpoint_cannot_resume_with_changed_permutation_or_legacy_policy(reader, assets):
    loader = make_readers(reader, assets)[0]
    state = loader.state_dict()
    manifest = Path(assets[0]['block_permutation_manifest'])
    value = json.loads(manifest.read_bytes()); value['seed'] = 'changed-independent-order'
    manifest.write_text(json.dumps(value))
    changed = make_readers(reader, assets)[0]
    with pytest.raises(ValueError, match='different block permutation manifest'):
        changed.load_state_dict(state)
    with pytest.raises(ValueError, match='different sampling policy'):
        loader.load_state_dict({'consumed_samples': 0, 'seq_len': 8})


def test_cursor_outside_capacity_is_rejected_before_reading(reader, assets):
    loader = make_readers(reader, assets)[0]
    state = loader.state_dict(); state['consumed_samples'] = 1000
    with pytest.raises(ValueError, match='capacity'):
        loader.load_state_dict(state)


def test_three_phase_recovery_probe_reads_the_same_production_prefix_and_resumes(reader, assets):
    kwargs, full_expected = assets
    parent_path = Path(kwargs['block_permutation_manifest'])
    parent = json.loads(parent_path.read_bytes())
    child = copy.deepcopy(parent)
    child.update(scope='PREFIX_PROBE', parent_manifest={'file': parent_path.name,
        'sha256': hashlib.sha256(parent_path.read_bytes()).hexdigest()})
    for index, row in enumerate(child['stages']):
        row.update(stage_sample_limit=8, parent_stage_index=index)
    probe_path = parent_path.with_name('three-phase-probe.json')
    probe_path.write_text(json.dumps(child))
    probe_kwargs = dict(kwargs, stage_sample_limits=[8, 8, 8], block_permutation_manifest=str(probe_path))
    expected = full_expected[:64]+full_expected[128:192]+full_expected[320:384]
    probe_assets = probe_kwargs, expected
    loaders = make_readers(reader, probe_assets)
    iterators = [iter(loader) for loader in loaders]
    observed = []
    saved = None
    for micro in range(6):
        for iterator in iterators:
            inputs, targets = next(iterator)
            assert inputs['_yuformer_stage_id'].tolist() == [micro//2]*2
            observed.extend(targets.flatten().tolist())
        if micro == 1:
            saved = loaders[0].state_dict()
    assert observed == expected
    resumed = make_readers(reader, probe_assets, world=4, batch=1)
    for loader in resumed:
        loader.load_state_dict(saved)
    remaining = []
    iterators = [iter(loader) for loader in resumed]
    for _ in range(4):
        for iterator in iterators:
            _, targets = next(iterator)
            remaining.extend(targets.flatten().tolist())
    assert remaining == expected[64:]


def test_validator_bridge_binds_real_metadata_and_rejects_probe_in_full_production(assets):
    kwargs, _ = assets
    data = dict(kwargs, seq_len=8)
    full = configuration_order_identity({'data': data}, require_production=True)
    assert full['manifest_scope'] == 'PRODUCTION'
    parent_path = Path(kwargs['block_permutation_manifest'])
    child_path = parent_path.with_name('validator-probe.json')
    derive(parent_path, child_path, stage_limits=[8], parent_stage_indices=[0])
    data.update(meta=[kwargs['meta'][0]], stages=[kwargs['stages'][0]], stage_sample_limits=[8],
                block_permutation_manifest=str(child_path))
    assert configuration_order_identity({'data': data})['manifest_scope'] == 'PREFIX_PROBE'
    with pytest.raises(ValueError, match='cannot use a prefix-probe'):
        configuration_order_identity({'data': data}, require_production=True)


def test_full_production_bridge_cannot_fall_back_to_sequential(assets):
    kwargs, _ = assets
    data = dict(kwargs, seq_len=8, sampling_policy='sequential', block_permutation_manifest='')
    assert configuration_order_identity({'data': data}) is None
    with pytest.raises(ValueError, match='requires its explicit independent'):
        configuration_order_identity({'data': data}, require_production=True)
