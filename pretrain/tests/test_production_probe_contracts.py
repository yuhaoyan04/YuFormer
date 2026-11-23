import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from production_probe_contracts import build_probe


@pytest.fixture
def recipe(tmp_path):
    source = Path(__file__).resolve().parents[2]
    base = yaml.safe_load((source/'pretrain/configs/pretrain_base.yaml').read_text())
    base['data'].update(sampling_policy='block_permutation',
        block_permutation_manifest=str(tmp_path/'manifest.json'))
    parent = {'schema_version': 2, 'scope': 'PRODUCTION', 'status': 'PASS',
        'purpose': 'TRAINING_CAUSAL_BLOCK_PERMUTATION',
        'algorithm': 'blake2b-128-independent-causal-block-sort-v1',
        'block_samples': 8, 'seed': 'independent', 'seq_len': 4096,
        'release_manifest_sha256': 'r', 'dataset_id': 'd', 'snapshot_id': 's',
        'stages': [{'stage_index': i, 'stage_sample_limit': n,
            'permutation_population_samples': n, 'stream_meta_sha256': str(i),
            'permutation': {'file': f'{i}.u32', 'sha256': str(i)}}
            for i, n in enumerate(base['data']['stage_sample_limits'])]}
    (tmp_path/'manifest.json').write_text(json.dumps(parent))
    return base, {'YUFORMER_MODEL_CONFIGS': str(tmp_path/'model.yaml')}, parent


@pytest.mark.parametrize('role,limits,schedule', [
    ('recovery', [8192, 8192, 8192], 'wsd'), ('canary', [204800], 'cosine')])
def test_probe_uses_actual_model_optimizer_seed_and_parent_order(tmp_path, recipe, role, limits, schedule):
    base, environment, parent = recipe
    before = copy.deepcopy(base)
    config, identity = build_probe(tmp_path, base, environment, role=role)
    assert base == before
    assert config['seed'] == base['seed']
    for key in ('model', 'parallel', 'attention', 'compile'):
        assert config[key] == base[key]
    for key in ('optimizer', 'lr', 'weight_decay', 'adam_betas', 'clip_grad'):
        assert config['optim'][key] == base['optim'][key]
    assert config['optim']['lr_schedule'] == schedule
    assert config['data']['stage_sample_limits'] == limits
    assert config['train']['expected_iters']*1024 == sum(limits)
    assert config['train']['budget_tokens'] == sum(limits)*4096
    assert identity['production_optimizer'] == base['optim']
    child_path = Path(identity['probe_order_manifest'])
    assert not child_path.exists()
    published, bound = build_probe(tmp_path, base, environment, role=role, publish=True)
    assert published == config and bound == identity
    assert hashlib.sha256(child_path.read_bytes()).hexdigest() == identity['probe_order_manifest_sha256']
    child = json.loads(child_path.read_bytes())
    assert child['parent_manifest']['sha256'] == hashlib.sha256((tmp_path/'manifest.json').read_bytes()).hexdigest()
    for row in child['stages']:
        original = parent['stages'][row['parent_stage_index']]
        assert row['permutation'] == original['permutation']
        assert row['permutation_population_samples'] == original['stage_sample_limit']
    assert build_probe(tmp_path, base, environment, role=role) == (config, identity)


def test_readonly_validation_rejects_changed_existing_probe(tmp_path, recipe):
    base, environment, _ = recipe
    _, identity = build_probe(tmp_path, base, environment, role='recovery', publish=True)
    path = Path(identity['probe_order_manifest'])
    child = json.loads(path.read_bytes())
    child['seed'] = 'different-order'
    path.write_text(json.dumps(child))
    with pytest.raises(ValueError, match='different immutable contract'):
        build_probe(tmp_path, base, environment, role='recovery')


def test_sequential_template_cannot_qualify_block_order(tmp_path, recipe):
    base, environment, _ = recipe
    base['data']['sampling_policy'] = 'sequential'
    with pytest.raises(ValueError, match='three-stage WSD block-order'):
        build_probe(tmp_path, base, environment, role='recovery')
