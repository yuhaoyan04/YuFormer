import importlib.util
import json
from pathlib import Path

import yaml
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/resolve_experiment_lrs.py"
SPEC = importlib.util.spec_from_file_location("resolve_experiment_lrs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize('supplemental', [False, True])
def test_resolves_all_configs_from_power_law(tmp_path, supplemental):
    presets = {
        "e1-50m": {"hidden_size": 16, "num_layers": 1, "intermediate_size": 32,
                    "num_heads": 2, "num_kv_heads": 1, "attention_every": 1},
        "e1-200m": {"hidden_size": 32, "num_layers": 1, "intermediate_size": 64,
                     "num_heads": 4, "num_kv_heads": 2, "attention_every": 1},
        "e1-800m": {"hidden_size": 64, "num_layers": 1, "intermediate_size": 128,
                     "num_heads": 8, "num_kv_heads": 4, "attention_every": 1},
    }
    model_path = tmp_path / "model/configs"
    model_path.mkdir(parents=True)
    (model_path / "model_configs.yaml").write_text(yaml.safe_dump({"presets": presets}))
    for group, preset in (("e1", "e1-50m"), ("e2", "e1-200m")):
        directory = tmp_path / "pretrain/configs" / group
        directory.mkdir(parents=True)
        (directory / f"{group}.yaml").write_text(yaml.safe_dump({
            "model": {"preset": preset}, "data": {"seq_len": 4096},
            "optim": {"lr": 0.0002, "warmup_steps": 100},
            "train": {"budget_tokens": 2_000_000_000,
                      "micro_batch_size": 4, "grad_accum": 16, "expected_iters": 7629},
            "parallel": {"world_size": 1},
        }))
    selection = tmp_path / "selection.json"
    value = {
        "status": "PASS", "expected_results": 15, "observed_results": 15,
        "boundary_selections": [], "selections": {
            "50m": {"learning_rate": 8e-4},
            "200m": {"learning_rate": 4e-4},
            "800m": {"learning_rate": 2e-4},
        }
    }
    if supplemental:
        from lr_pilot_contract import CONTRACT, contract_sha256, expected_names
        manifest = tmp_path / CONTRACT
        manifest.write_text(json.dumps({'schema_version': 1, 'extensions': [
            {'size': '200m', 'level': f'ext{number}', 'run_name': f'e1_lr_200m_ext{number}',
             'learning_rate': number / 1_000_000, 'group': 'e1_pilot_extension'}
            for number in (1800, 2400, 4800)]}))
        value.update(expected_results=18, observed_results=18,
                     search_contract_sha256=contract_sha256(tmp_path),
                     expected_run_names=sorted(expected_names(tmp_path)))
    selection.write_text(json.dumps(value))
    batch_selection = tmp_path / "batch-selection.json"
    batch_selection.write_text(json.dumps({
        "status": "PASS", "expected_results": 9, "observed_results": 9,
        "boundary_selections": [],
        "selection": {"tokens_per_step": 1_048_576, "learning_rate": 8e-4},
    }))
    if supplemental:
        import batch_pilot_contract as batch_contract
        path = tmp_path / batch_contract.CONTRACT
        path.write_text(json.dumps({'schema_version': 1, 'extensions': [
            {'batch': '1m', 'level': f'ext{number}', 'run_name': f'batch_1m_ext{number}',
             'learning_rate': number / 1_000_000, 'group': batch_contract.GROUP}
            for number in (100, 200)]}))
        value = json.loads(batch_selection.read_text())
        value.update(expected_results=11, observed_results=11,
                     search_contract_sha256=batch_contract.contract_sha256(tmp_path),
                     expected_run_names=sorted(batch_contract.expected_names(tmp_path)))
        batch_selection.write_text(json.dumps(value))
    result = MODULE.resolve(tmp_path, selection, batch_selection)
    assert result["status"] == "PASS"
    assert result["configs_updated"] == 2
    assert result["fit"]["exponent"] < 0
    rewritten = yaml.safe_load((tmp_path / "pretrain/configs/e2/e2.yaml").read_text())
    assert rewritten["train"]["grad_accum"] == 64
    assert rewritten["train"]["expected_iters"] == 1907
    assert rewritten["optim"]["lr"] > 0
    assert rewritten["optim"]["lr_resolution"]["pilot_selection_sha256"] == MODULE.sha256(selection)
    qualified_loss = json.loads(batch_selection.read_text())
    qualified_loss['selection']['loss_only'] = True
    batch_selection.write_text(json.dumps(qualified_loss))
    qualified_resolution = MODULE.resolve(tmp_path, selection, batch_selection)
    assert qualified_resolution['status'] == 'PASS'
    assert qualified_resolution['evidence_mode'] == 'loss_only'

    provisional_batch = json.loads(batch_selection.read_text())
    provisional_batch["status"] = "PASS_PROVISIONAL"
    batch_selection.write_text(json.dumps(provisional_batch))
    provisional = MODULE.resolve(tmp_path, selection, batch_selection)
    assert provisional["status"] == "PASS_PROVISIONAL"
    assert provisional["evidence_mode"] == "loss_only"
    policy = tmp_path / 'pretrain/configs/research_budget.yaml'
    policy.write_text(yaml.safe_dump({'screen_tokens_per_update_cap': 262144,
                                    'minimum_screen_updates': 128}))
    capped = MODULE.resolve(tmp_path, selection, batch_selection)
    rewritten = yaml.safe_load((tmp_path / 'pretrain/configs/e2/e2.yaml').read_text())
    assert capped['selected_tokens_per_step'] == 1048576
    assert capped['screen_tokens_per_step'] == 262144
    assert rewritten['train']['grad_accum'] == 16
    assert rewritten['optim']['lr'] == 4e-4
    assert capped['budget_policy_sha256'] == MODULE.sha256(policy)
