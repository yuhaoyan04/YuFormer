import copy
import json
from pathlib import Path

import pytest

from yuformer_midtrain.stage_contract import validate_stage_geometry


@pytest.fixture
def config():
    root = Path(__file__).resolve().parents[2]
    budget = json.loads((root/'data/health/user-total-pretrain-and-length-budget.json').read_bytes())
    stage = budget['stages'][2]
    config = {'data': {'seq_len': 65536},
        'train': {'expected_iters': stage['updates'], 'budget_tokens': stage['tokens'],
            'micro_batch_size': 1, 'grad_accum': 8, 'deterministic': True, 'bf16': True,
            'activation_checkpointing': False}, 'parallel': {'world_size': 8,
                'sharding':'hybrid_shard','replicate_degree':2,'shard_degree':4,
                'fsdp_reshard_after_forward':'never'},
        'attention': {'sdpa_backend': 'flash_flex_swa'}, 'compile': {'enable': True, 'components': ['model']},
        'optim': {'lr_schedule': 'cosine','optimizer':'adamw','lr':2e-5,'warmup_steps':10,
                  'min_lr_ratio':.1}, 'loss': {'num_chunks': 32},
        'midtrain': {'scope': 'PRODUCTION', 'selective_recompute': 'mlp',
            'cuda_allocator_conf': 'expandable_segments:True,garbage_collection_threshold:0.7',
            'optimizer_transition': 'reset', 'initial_load_model_only': True,
            'evaluation': {'long_context': True, 'short_regression': True}}}
    return config, budget


def test_production_stage_consumes_exact_approved_allocation(config):
    result = validate_stage_geometry(*config)
    assert result == {'seq_len': 65536, 'updates': 4768, 'tokens': 19998441472,
                      'samples': 305152, 'scope': 'PRODUCTION'}


@pytest.mark.parametrize('fault', ['wrong_world', 'variable_token_update', 'extra_budget', 'full_ac',
    'wrong_recompute', 'large_ce', 'wrong_allocator', 'nondeterministic', 'optimizer_resume',
    'model_only_false', 'missing_short_gate', 'missing_long_gate'])
def test_known_unsafe_stage_contracts_fail_before_training(config, fault):
    config, budget = copy.deepcopy(config)
    if fault == 'wrong_world': config['parallel']['world_size'] = 4
    elif fault == 'variable_token_update': config['train']['grad_accum'] = 16
    elif fault == 'extra_budget':
        config['train']['expected_iters'] += 1
        config['train']['budget_tokens'] += 4194304
    elif fault == 'full_ac': config['train']['activation_checkpointing'] = True
    elif fault == 'wrong_recompute': config['midtrain']['selective_recompute'] = 'none'
    elif fault == 'large_ce': config['loss']['num_chunks'] = 8
    elif fault == 'wrong_allocator': config['midtrain'].pop('cuda_allocator_conf')
    elif fault == 'nondeterministic': config['train']['deterministic'] = False
    elif fault == 'optimizer_resume': config['midtrain']['optimizer_transition'] = 'preserve'
    elif fault == 'model_only_false': config['midtrain']['initial_load_model_only'] = False
    elif fault == 'missing_short_gate': config['midtrain']['evaluation']['short_regression'] = False
    else: config['midtrain']['evaluation']['long_context'] = False
    with pytest.raises(ValueError):
        validate_stage_geometry(config, budget)


def test_short_probe_budget_cannot_masquerade_as_production(config):
    config, budget = config
    config['train'].update(expected_iters=24, budget_tokens=24*4194304)
    with pytest.raises(ValueError, match='production allocation'):
        validate_stage_geometry(config, budget)
    config['midtrain']['scope'] = 'QUALIFICATION'
    assert validate_stage_geometry(config, budget)['scope'] == 'QUALIFICATION'
