import importlib.util
import json
from pathlib import Path

import yaml
import pytest
import batch_pilot_contract as contract
from research_scope import POLICY, scope_sha256


ROOT = Path(__file__).resolve().parents[2]


def _module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


SELECT = _module("select_batch_pilots", "pretrain/scripts/select_batch_pilots.py")
PREPARE = _module("prepare_batch_pilots", "pretrain/scripts/prepare_batch_pilots.py")


def test_promotion_refresh_preserves_observation_time_but_rechecks_qualification(tmp_path):
    path = tmp_path / 'selection.json'
    base = {'status': 'PASS_PROVISIONAL', 'observed_at': 'original', 'selection': {'lr': .001}}
    promoted = {**base, 'status': 'PASS', 'loss_evidence_qualification': {'identity': 'qualified'}}
    SELECT.atomic_json(path, promoted)
    SELECT.atomic_json(path, {**base, 'observed_at': 'refresh'})
    refreshed = json.loads(path.read_text())
    assert refreshed == base  # qualification must still be recomputed
    SELECT.atomic_json(path, {**refreshed, 'status': 'PASS',
                              'loss_evidence_qualification': {'identity': 'qualified'}})
    assert json.loads(path.read_text()) == promoted
    SELECT.atomic_json(path, {**base, 'observed_at': 'changed', 'selection': {'lr': .002}})
    assert json.loads(path.read_text())['observed_at'] == 'changed'


@pytest.mark.parametrize('change', ['provenance', 'lr', 'steps', 'data'])
def test_resolver_keeps_started_contract_and_rejects_recipe_changes(tmp_path, change):
    from resolve_experiment_lrs import immutable_started_config
    path = tmp_path / 'arm.yaml';run = tmp_path / 'run';run.mkdir()
    (run / 'run_identity.json').write_text('{}')
    path.with_suffix('.env').write_text(f'YUFORMER_PRETRAIN_OUT={run}\n')
    old = {'optim': {'lr': .001, 'lr_resolution': {'reference': 'original'}},
           'train': {'expected_iters': 100}, 'data': {'source': 'A'}}
    text = yaml.safe_dump(old, sort_keys=False);path.write_text(text)
    cfg = yaml.safe_load(text)
    if change == 'provenance':
        cfg['optim']['lr_resolution'] = {'reference': 'updated'}
        assert immutable_started_config(path, cfg) == text
    else:
        if change == 'lr': cfg['optim']['lr'] = .002
        if change == 'steps': cfg['train']['expected_iters'] = 200
        if change == 'data': cfg['data']['source'] = 'B'
        with pytest.raises(ValueError, match='started research contract'):
            immutable_started_config(path, cfg)
    assert path.read_text() == text


def _results(tmp_path: Path, boundary: bool = False) -> list[Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for batch_index, batch in enumerate(SELECT.BATCHES):
        for level_index, level in enumerate(SELECT.LEVELS):
            loss = 2.0 + batch_index / 100 + (2 - level_index if boundary else abs(level_index - 1))
            path = tmp_path / f"{batch}-{level}.json"
            tokens_per_step = 1_048_576 * (2 ** batch_index)
            path.write_text(json.dumps({
                "status": "PASS", "run_name": f"batch_{batch}_{level}",
                "identity_sha256": f"{batch}-{level}",
                "hyperparameters": {"learning_rate": 1e-4 * (2 ** level_index),
                                    "tokens_per_step": tokens_per_step,
                                    "trained_steps": (SELECT.MATCHED_TRAINED_TOKENS
                                                      // tokens_per_step),
                                    "trained_tokens": SELECT.MATCHED_TRAINED_TOKENS},
                "metrics": {"validation_metrics/loss": {"last": loss},
                            "throughput(tps)": {"last": 1000.0}},
            }))
            paths.append(path)
    return paths


def test_selects_joint_batch_lr_only_when_each_lr_curve_is_bracketed(tmp_path):
    report = SELECT.select(_results(tmp_path))
    assert report["status"] == "PASS"
    assert report["selection"]["batch"] == "1m"
    assert all(row["level"] == "mid" for row in report["per_batch_optima"])

    blocked = SELECT.select(_results(tmp_path / "boundary", boundary=True))
    assert blocked["status"] == "BLOCKED"
    assert len(blocked["boundary_selections"]) == 3

    provisional_paths = _results(tmp_path / "provisional")
    for path in provisional_paths:
        value = json.loads(path.read_text())
        value["execution_context"] = {"loss_only": True}
        value["metrics"].pop("throughput(tps)")
        path.write_text(json.dumps(value))
    provisional = SELECT.select(provisional_paths)
    assert provisional["status"] == "PASS_PROVISIONAL"
    assert provisional["selection"]["loss_only"] is True


def test_prepare_batch_grid_binds_selected_lr_and_sqrt_batch_rule(tmp_path):
    selection = tmp_path / "experiments/e1_scaling/results/lr_selection.json"
    selection.parent.mkdir(parents=True)
    selection.write_text(json.dumps({
        "status": "PASS", "expected_results": 15, "observed_results": 15,
        "boundary_selections": [], "selections": {
            "50m": {"learning_rate": 6e-4}, "200m": {"learning_rate": 3e-4},
            "800m": {"learning_rate": 1.5e-4}},
    }))
    config_dir = tmp_path / "pretrain/configs/batch_pilot"
    config_dir.mkdir(parents=True)
    for batch, accum in (("1m", 64), ("2m", 128), ("4m", 256)):
        for level in PREPARE.LEVEL_MULTIPLIER:
            (config_dir / f"batch_{batch}_{level}.yaml").write_text(yaml.safe_dump({
                "data": {"seq_len": 4096},
                "train": {"micro_batch_size": 4, "grad_accum": accum},
                "parallel": {"world_size": 1}, "optim": {"lr": 1e-4},
            }))
    report = PREPARE.prepare(tmp_path, selection)
    assert report["status"] == "PASS"
    rows = {row["run_name"]: row for row in report["configs"]}
    assert rows["batch_4m_mid"]["learning_rate"] == 0.0012
    assert rows["batch_1m_mid"]["learning_rate"] == 0.0006


FIXED_BATCH = {'mode': 'fixed_baseline', 'batch': '1m', 'tokens_per_step': 1048576,
               'claim_limit': 'No globally optimal batch-size claim'}


def _nonchosen_boundaries(tmp_path):
    paths = _results(tmp_path)
    for name in ('2m-high.json', '4m-low.json'):
        path = tmp_path / name
        value = json.loads(path.read_text()); value['metrics']['validation_metrics/loss']['last'] = 1.0
        path.write_text(json.dumps(value))
    return paths


def test_fixed_batch_is_bracketed_baseline_without_claiming_global_optimality(tmp_path):
    paths = _nonchosen_boundaries(tmp_path)
    assert SELECT.select(paths)['status'] == 'BLOCKED'
    report = SELECT.select(paths, selection_policy=FIXED_BATCH)
    assert report['status'] == 'PASS'
    assert report['selection']['batch'] == '1m'
    assert report['selection']['level'] == 'mid'
    assert {edge['batch'] for edge in report['boundary_selections']} == {'2m', '4m'}
    assert report['blocking_boundary_selections'] == []
    assert 'no globally optimal' in report['selection_claim']
    assert min(row['validation_loss'] for row in report['per_batch_optima']) < report['selection']['validation_loss']


def test_fixed_batch_keeps_selected_boundary_missing_and_invalid_pilots_blocking(tmp_path):
    paths = _results(tmp_path, boundary=True)
    assert SELECT.select(paths, selection_policy=FIXED_BATCH)['status'] == 'BLOCKED'
    paths = _nonchosen_boundaries(tmp_path / 'bracketed')
    assert SELECT.select(paths[:-1], selection_policy=FIXED_BATCH)['status'] == 'BLOCKED'
    assert SELECT.select(paths + [paths[-1]], selection_policy=FIXED_BATCH)['status'] == 'BLOCKED'
    value = json.loads(paths[-1].read_text()); value['hyperparameters']['trained_tokens'] -= 1
    paths[-1].write_text(json.dumps(value))
    assert SELECT.select(paths, selection_policy=FIXED_BATCH)['status'] == 'BLOCKED'


def test_fixed_batch_selection_must_be_bound_to_current_declared_scope(tmp_path):
    paths = _nonchosen_boundaries(tmp_path / 'results')
    policy_path = tmp_path / POLICY; policy_path.parent.mkdir(parents=True)
    policy_path.write_text(json.dumps({'profile': 'full', 'batch_selection_policy': FIXED_BATCH}))
    report = SELECT.select(paths, selection_policy=FIXED_BATCH)
    assert not contract.selection_is_current(tmp_path, report)
    report['research_scope_sha256'] = scope_sha256(tmp_path)
    assert contract.selection_is_current(tmp_path, report)
    curve = report['curves']['1m']
    report['curves']['1m'] = [row for row in curve if row['learning_rate'] <= report['selection']['learning_rate']]
    assert not contract.selection_is_current(tmp_path, report)
    report['curves']['1m'] = curve
    policy_path.write_text(json.dumps({'profile': 'full'}))
    assert not contract.selection_is_current(tmp_path, report)
