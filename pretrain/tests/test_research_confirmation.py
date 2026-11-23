import json
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import research_confirmation as CONFIRM


def fixture(root, stage='e2', selected='e2_l16'):
    policy = root / 'pretrain/configs/research_budget.yaml'
    policy.parent.mkdir(parents=True)
    policy.write_text(yaml.safe_dump({'confirmation': {'required': True, 'budget_tokens': 3_000_000_000,
        'seeds': [1, 2, 3], 'maximum_domain_relative_regression': .02,
        'practical_equivalence_absolute_loss': .01}}))
    path = CONFIRM.result_path(root, stage).with_name('selection.json')
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'status': 'PASS', 'selection': {'run_name': selected}}))
    baseline = 'e2_l32' if stage == 'e2' else 'e3_m2_base'
    for name in {selected, baseline}:
        config = root / f'pretrain/configs/{stage}/{name}.yaml'
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(yaml.safe_dump({'run_name': name, 'seed': 0,
            'model': {'preset': 'pretrain-0.5b'}, 'data': {'seq_len': 4096},
            'parallel': {'world_size': 1}, 'optim': {'warmup_steps': 100, 'lr': .0002},
            'train': {'micro_batch_size': 4, 'grad_accum': 16, 'budget_tokens': 1_000_000_000}}))
        config.with_suffix('.env').write_text(f'YUFORMER_CONFIG=configs/{stage}/{name}.yaml\nYUFORMER_ARM_DATA={name}\n')


def test_confirmation_is_fresh_three_seed_training_and_idempotent(tmp_path):
    fixture(tmp_path)
    report = CONFIRM.prepare(tmp_path, 'e2')
    assert len(report['runs']) == 6
    for row in report['runs']:
        cfg = yaml.safe_load((tmp_path / f'pretrain/configs/e2_confirm/{row["name"]}.yaml').read_text())
        assert cfg['train']['budget_tokens'] == 3_000_000_000
        assert cfg['seed'] == row['seed']
        assert cfg['optim']['warmup_steps'] == cfg['train']['expected_iters'] // 100
        assert 'resume' not in cfg
    path = tmp_path / 'data/health/e2_confirm-contracts.json'
    before = path.stat().st_mtime_ns
    assert CONFIRM.prepare(tmp_path, 'e2') == report
    assert path.stat().st_mtime_ns == before


def test_baseline_winner_deduplicates_identical_confirmation(tmp_path):
    fixture(tmp_path, 'e3', 'e3_m2_base')
    assert len(CONFIRM.prepare(tmp_path, 'e3')['runs']) == 3


def test_running_confirmation_contract_cannot_change(tmp_path):
    fixture(tmp_path)
    report = CONFIRM.prepare(tmp_path, 'e2')
    run = tmp_path / f'data/experiments/e2_confirm/{report["runs"][0]["name"]}'
    run.mkdir(parents=True)
    (run / 'current_attempt.json').write_text('{}')
    policy = tmp_path / 'pretrain/configs/research_budget.yaml'
    doc = yaml.safe_load(policy.read_text())
    doc['confirmation']['budget_tokens'] = 5_000_000_000
    policy.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match='drift after launch'):
        CONFIRM.prepare(tmp_path, 'e2')


def test_paired_interval_keeps_seed_uncertainty():
    steady = CONFIRM.paired_interval([.005] * 3)
    assert steady['ci95'][1] == pytest.approx(.005)
    uncertain = CONFIRM.paired_interval([-.05, 0, .05])
    assert uncertain['ci95'][1] > .01
    with pytest.raises(ValueError):
        CONFIRM.paired_interval([0., float('nan'), 0.])


def test_stale_confirmation_source_selection_blocks(tmp_path):
    fixture(tmp_path)
    CONFIRM.prepare(tmp_path, 'e2')
    selection = CONFIRM.result_path(tmp_path, 'e2').with_name('selection.json')
    selection.write_text('{"status":"FAIL"}')
    assert CONFIRM.analyze(tmp_path, 'e2')['reason'].startswith('stale_confirmation_input:')


def test_longer_budget_gate_rejects_domain_regression_and_incomplete_budget(tmp_path, monkeypatch):
    fixture(tmp_path, 'e3', 'e3_math_2x')
    manifest = CONFIRM.prepare(tmp_path, 'e3')
    models = tmp_path / 'model/configs/model_configs.yaml'
    models.parent.mkdir(parents=True)
    models.write_text('presets: {}\n')
    qualification = tmp_path / 'data/health/torchtitan-qualification.json'
    qualification.write_text('{"status":"PASS"}')
    monkeypatch.setattr(CONFIRM, 'qualification_identity', lambda root: {})
    monkeypatch.setattr(CONFIRM, 'report_matches', lambda report, identity: True)
    monkeypatch.setattr(CONFIRM, 'audit', lambda root, groups: {
        'status': 'PASS', 'evidence_snapshot': {str(models.relative_to(tmp_path)): CONFIRM.sha256(models)}})
    for row in manifest['runs']:
        cfg = yaml.safe_load((tmp_path / f'pretrain/configs/e3_confirm/{row["name"]}.yaml').read_text())
        run = tmp_path / f'data/experiments/e3_confirm/{row["name"]}'
        run.mkdir(parents=True)
        metrics = {'validation_metrics/loss': {'last': 1. if row['role'] == 'baseline' else .99}}
        metrics.update({f'validation_metrics/domain/{domain}/loss': {'last': 1.} for domain in CONFIRM.DOMAINS})
        result = {'hyperparameters': {'budget_tokens': 3_000_000_000, 'tokens_per_step': 262144,
                    'trained_steps': cfg['train']['expected_iters'],
                    'trained_tokens': cfg['train']['expected_iters'] * 262144}, 'metrics': metrics}
        (run / 'result.json').write_text(json.dumps(result))
    assert CONFIRM.analyze(tmp_path, 'e3')['status'] == 'PASS'
    row = next(row for row in manifest['runs'] if row['role'] == 'selected')
    result_path = tmp_path / f'data/experiments/e3_confirm/{row["name"]}/result.json'
    result = json.loads(result_path.read_text())
    result['metrics']['validation_metrics/domain/web/loss']['last'] = 1.2
    result_path.write_text(json.dumps(result))
    rejected = CONFIRM.analyze(tmp_path, 'e3')
    assert rejected['status'] == 'FAIL'
    assert rejected['relative_domain_deltas']['web']['ci95'][1] > .02
    result['hyperparameters']['trained_tokens'] -= 262144
    result_path.write_text(json.dumps(result))
    assert CONFIRM.analyze(tmp_path, 'e3')['reason'] == 'confirmation_training_accounting_mismatch'
