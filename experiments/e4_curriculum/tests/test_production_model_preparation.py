import copy
from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments/e4_curriculum/scripts'))
from prepare_production_model import production_documents


@pytest.fixture
def recipe():
    base = yaml.safe_load((ROOT/'pretrain/configs/pretrain_base.yaml').read_bytes())
    base['data'].pop('sampling_policy', None)
    base['data'].pop('block_permutation_manifest', None)
    measured = copy.deepcopy(base)
    measured['data'].update(stages=['research-100b-only'], meta=['research-only-meta'],
                            stage_sample_limits=[204800])
    measured['optim'].update(lr=.0001, lr_schedule='cosine', warmup_steps=10)
    measured['optim'].pop('decay_steps')
    measured['attention'] = {'sdpa_backend': 'cudnn_flex_swa'}
    measured['train'].update(expected_iters=200, budget_tokens=838860800)
    environment = {'YUFORMER_TOKEN_STREAMS': '/immutable/production-1t',
                   'YUFORMER_VAL_STREAMS': '/immutable/nine-validation-domains'}
    measured_env = {'OMP_NUM_THREADS': '1', 'TORCHINDUCTOR_COMPILE_THREADS': '2',
                    'TORCHINDUCTOR_CACHE_DIR': '/tmp/measured-cache'}
    model = dict(vocab_size=65536, hidden_size=2560, intermediate_size=6912,
        num_layers=18, num_heads=40, num_kv_heads=10, attention_every=6,
        attention_output_gate=True, qk_norm=True, tie_word_embeddings=True)
    auth = {'architecture': {}, 'training': {'peak_learning_rate': .0002}}
    return base, environment, auth, model, measured, measured_env


def test_model_preparation_preserves_1t_data_wsd_and_batch_while_declaring_measured_compile_environment(recipe, tmp_path):
    before = copy.deepcopy(recipe)
    base, environment, auth = production_documents(*recipe, model_path=tmp_path/'model.yaml',
                                                   block_manifest=tmp_path/'manifest.json')
    assert recipe == before
    original = recipe[0]
    assert base['train'] == original['train']
    assert base['train']['budget_tokens'] == original['train']['budget_tokens']
    assert base['train']['expected_iters']*4194304 == base['train']['budget_tokens']
    assert base['data']['stages'] == original['data']['stages']
    assert base['data']['meta'] == original['data']['meta']
    assert base['data']['stage_sample_limits'] == original['data']['stage_sample_limits']
    assert base['data']['val_dir'] == original['data']['val_dir']
    for key in ('lr_schedule', 'warmup_steps', 'decay_steps'):
        assert base['optim'][key] == original['optim'][key]
    assert base['optim']['lr'] == .0001
    assert base['attention'] == {'sdpa_backend': 'cudnn_flex_swa'}
    assert base['data']['sampling_policy'] == 'block_permutation'
    assert environment['YUFORMER_TOKEN_STREAMS'] == '/immutable/production-1t'
    for key, value in recipe[-1].items():
        assert environment[key] == value
    assert auth['architecture']['preset'] == 'pretrain-final'
    assert auth['training']['peak_learning_rate'] == .0001


@pytest.mark.parametrize('fault', ['microbatch', 'sequence', 'topology', 'compiler', 'optimizer', 'missing_compile_environment'])
def test_unmeasured_execution_contract_cannot_be_prepared(recipe, tmp_path, fault):
    *_, measured, environment = recipe
    if fault == 'microbatch': measured['train'].update(micro_batch_size=4, grad_accum=32)
    elif fault == 'sequence': measured['data']['seq_len'] = 8192
    elif fault == 'topology': measured['parallel'].update(replicate_degree=2, shard_degree=8)
    elif fault == 'compiler': measured['compile']['enable'] = False
    elif fault == 'optimizer': measured['optim']['weight_decay'] = .01
    else: environment.pop('TORCHINDUCTOR_COMPILE_THREADS')
    with pytest.raises(ValueError):
        production_documents(*recipe, model_path=tmp_path/'model.yaml', block_manifest=tmp_path/'manifest.json')


def test_selected_determinism_loss_and_reshard_policy_survive_preparation(recipe, tmp_path):
    measured = recipe[-2]
    measured['train']['deterministic'] = True
    measured['loss'] = {'num_chunks': 4}
    measured['parallel']['fsdp_reshard_after_forward'] = 'never'
    result, _, _ = production_documents(*recipe, model_path=tmp_path/'model.yaml',
                                        block_manifest=tmp_path/'manifest.json')
    assert result['train']['deterministic'] is True
    assert result['loss'] == {'num_chunks': 4}
    assert result['parallel']['fsdp_reshard_after_forward'] == 'never'


def test_unmeasured_numeric_environment_is_rejected(recipe, tmp_path):
    recipe[1]['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    with pytest.raises(ValueError, match='numeric/allocator environment'):
        production_documents(*recipe, model_path=tmp_path/'model.yaml', block_manifest=tmp_path/'manifest.json')


def test_actual_measured_full_shard_eight_card_geometry_can_be_prepared(recipe, tmp_path):
    recipe[-2]['parallel'].update(sharding='full_shard', replicate_degree=1, shard_degree=8,
                                  fsdp_reshard_after_forward='never')
    result, _, _ = production_documents(*recipe, model_path=tmp_path/'model.yaml',
                                        block_manifest=tmp_path/'manifest.json')
    assert result['parallel'] == recipe[-2]['parallel']
