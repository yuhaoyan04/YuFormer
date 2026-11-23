import sys
from pathlib import Path

import yaml
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from yuformer_midtrain.contract import audit_ladder


def _config():
    return {
        "schema_version": 1,
        "backend": "torchtitan-v0.3.0",
        "pretrain_context": 4096,
        "base_checkpoint": {"manifest_sha256": "a" * 64},
        "dataset_id": "yuformer-dataset-" + "b" * 20,
        "optimizer_transition": "reset",
        "target_tokens_per_update": 65536,
        "stages": [
            {"name": "16k", "seq_len": 16384, "tokens_per_update": 65536,
             "world_size": 1, "micro_batch_size": 1, "grad_accum": 4,
             "budget_tokens": 10000, "short_replay_fraction": 0.25,
             "attention_kernel": "flash_attention_3", "init": "previous_stage_weights_only",
             "evaluation": {"long_context": True, "short_regression": True}},
            {"name": "64k", "seq_len": 65536, "tokens_per_update": 65536,
             "world_size": 1, "micro_batch_size": 1, "grad_accum": 1,
             "budget_tokens": 10000, "short_replay_fraction": 0.25,
             "attention_kernel": "flash_attention_3", "init": "previous_stage_weights_only",
             "evaluation": {"long_context": True, "short_regression": True}},
        ],
    }


def test_valid_progressive_ladder_passes():
    assert audit_ladder(_config())["status"] == "PASS"


def test_nonprogressive_or_variable_update_ladder_fails():
    config = _config()
    config["stages"][1]["seq_len"] = 8192
    config["stages"][1]["tokens_per_update"] = 512
    result = audit_ladder(config)
    assert result["status"] == "FAIL"
    assert any("strictly increase" in error for error in result["errors"])
    assert any("tokens_per_update" in error for error in result["errors"])


def test_unlocked_assets_block_an_otherwise_valid_ladder():
    config = yaml.safe_load(
        (Path(__file__).parents[1] / "configs" / "ladder_4k_16k_64k.yaml").read_text()
    )
    result = audit_ladder(config)
    assert result["status"] == "BLOCKED"
    assert len(result["blockers"]) == 2


def test_production_ladders_preserve_eight_gpus_and_fixed_token_updates():
    for name in ('ladder_4k_16k_64k.yaml', 'direct_4k_64k.yaml'):
        path = Path(__file__).parents[1]/'configs'/name
        config = yaml.safe_load(path.read_text())
        assert config['required_world_size'] == 8
        for stage in config['stages']:
            assert stage['world_size'] == 8
            assert stage['seq_len']*stage['micro_batch_size']*stage['grad_accum']*8 == 4194304
        # Halving world size and doubling accumulation preserves arithmetic but
        # still violates the user's actual hardware constraint.
        config['stages'][0]['world_size'] = 4
        config['stages'][0]['grad_accum'] *= 2
        report = audit_ladder(config)
        assert report['status'] == 'FAIL'
        assert any('required_world_size=8' in error for error in report['errors'])


@pytest.mark.parametrize('fault',['full_checkpoint','no_mlp','large_loss_chunk','no_allocator_headroom','no_compile'])
def test_known_failed_64k_paths_cannot_pass_the_stage_contract(fault):
    path = Path(__file__).parents[1]/'configs'/'direct_4k_64k.yaml'
    config = yaml.safe_load(path.read_text())
    stage = config['stages'][0]
    if fault == 'full_checkpoint': stage['activation_checkpointing'] = True
    elif fault == 'no_mlp': stage['selective_recompute'] = 'none'
    elif fault == 'large_loss_chunk': stage['loss_num_chunks'] = 8
    elif fault == 'no_allocator_headroom': stage.pop('cuda_allocator_conf')
    else: stage['compile_model'] = False
    assert audit_ladder(config)['status'] == 'FAIL'
