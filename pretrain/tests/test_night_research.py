import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_experiment_identities as identities
import lr_pilot_contract as contract
import night_research as night
import prepare_batch_pilots as batch
from select_lr_pilots import atomic_json


@pytest.fixture
def anchor_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(identities, 'source_sha256', lambda root: 'source')
    manifest = tmp_path / contract.CONTRACT
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'schema_version': 1, 'extensions': []}))
    curves, rows, snapshot = [], [], {}
    for index, level in enumerate(contract.LEVELS):
        name = f'e1_lr_200m_{level}'
        env = tmp_path / f'pretrain/configs/e1_pilot/{name}.env'
        cfg = env.with_suffix('.yaml')
        run = tmp_path / f'data/experiments/e1_pilot/{name}'
        env.parent.mkdir(parents=True, exist_ok=True); run.mkdir(parents=True)
        env.write_text(f'run={name}'); cfg.write_text(f'run={name}')
        for filename in ['run_identity.json', 'current_attempt.json', 'training_complete.json', 'result.json']:
            (run / filename).write_text('{}')
        for p in [env, cfg, *(run / f for f in ['run_identity.json', 'current_attempt.json', 'training_complete.json', 'result.json'])]:
            snapshot[str(p.relative_to(tmp_path))] = night.sha256(p)
        rows.append({'name': name, 'group': 'e1_pilot', 'env': str(env.relative_to(tmp_path)),
                     'config': str(cfg), 'run_dir': str(run), 'status': 'PASS'})
        curves.append({'run_name': name, 'learning_rate': 0.000075 * 2 ** index,
                       'validation_loss': 4.0 + abs(index - 3)})
    winner = copy.deepcopy(curves[3])
    selection = {'status': 'BLOCKED', 'missing': ['800m:mid'], 'curves': {'200m': curves},
                 'selections': {'200m': winner}, 'boundary_selections': [],
                 'duplicates': [], 'rejected': []}
    audit = {'rows': rows, 'source_sha256': 'source', 'evidence_snapshot': snapshot}
    return tmp_path, selection, audit


def test_batch_anchor_can_pass_while_other_sizes_are_incomplete(anchor_evidence):
    root, selection, audit = anchor_evidence
    anchor = night.build_batch_anchor(root, selection, audit)
    assert anchor['status'] == 'PASS'
    assert night.batch_anchor_is_current(root, anchor)
    assert not contract.selection_is_current(root, anchor)


def test_other_model_search_expansion_does_not_invalidate_batch_anchor(anchor_evidence):
    root, selection, audit = anchor_evidence
    anchor = night.build_batch_anchor(root, selection, audit)
    path = root / contract.CONTRACT
    path.write_text(json.dumps({'schema_version': 1, 'extensions': [
        {'run_name': 'e1_lr_800m_ext1200', 'size': '800m', 'level': 'ext1200',
         'learning_rate': 0.0012, 'group': contract.EXTENSION_GROUP}]}))
    assert night.batch_anchor_is_current(root, anchor)


def test_same_model_search_expansion_invalidates_batch_anchor(anchor_evidence):
    root, selection, audit = anchor_evidence
    anchor = night.build_batch_anchor(root, selection, audit)
    (root / contract.CONTRACT).write_text(json.dumps({'schema_version': 1, 'extensions': [
        {'run_name': 'e1_lr_200m_ext1800', 'size': '200m', 'level': 'ext1800',
         'learning_rate': 0.0018, 'group': contract.EXTENSION_GROUP}]}))
    assert not night.batch_anchor_is_current(root, anchor)


@pytest.mark.parametrize('failure', ['missing', 'failed_identity', 'boundary', 'changed_after_audit'])
def test_batch_anchor_preserves_scientific_and_identity_gates(anchor_evidence, failure):
    root, selection, audit = anchor_evidence
    if failure == 'missing':
        selection['curves']['200m'].pop()
    elif failure == 'failed_identity':
        audit['rows'][0]['status'] = 'FAIL'
    elif failure == 'boundary':
        selection['curves']['200m'][-1]['validation_loss'] = 1.0
        selection['selections']['200m'] = selection['curves']['200m'][-1]
    else:
        Path(audit['rows'][0]['config']).write_text('mutated')
    assert night.build_batch_anchor(root, selection, audit)['status'] == 'BLOCKED'


def test_anchor_evidence_and_source_drift_are_rejected(anchor_evidence, monkeypatch):
    root, selection, audit = anchor_evidence
    anchor = night.build_batch_anchor(root, selection, audit)
    monkeypatch.setattr(identities, 'source_sha256', lambda root: 'changed')
    assert not night.batch_anchor_is_current(root, anchor)
    monkeypatch.setattr(identities, 'source_sha256', lambda root: 'source')
    Path(audit['rows'][0]['config']).write_text('changed')
    assert not night.batch_anchor_is_current(root, anchor)


def test_anchor_file_stays_stable_as_unrelated_results_arrive(anchor_evidence):
    root, selection, audit = anchor_evidence
    path = root / night.ANCHOR
    atomic_json(path, night.build_batch_anchor(root, selection, audit))
    before = path.read_bytes()
    selection['selections']['800m'] = {'learning_rate': 1e-4}
    atomic_json(path, night.build_batch_anchor(root, selection, audit))
    assert path.read_bytes() == before


def batch_configs(root):
    directory = root / 'pretrain/configs/batch_pilot'
    directory.mkdir(parents=True)
    for batch_name, ga in [('1m', 64), ('2m', 128), ('4m', 256)]:
        for level in batch.LEVEL_MULTIPLIER:
            name = f'batch_{batch_name}_{level}'
            path = directory / f'{name}.yaml'
            path.write_text(yaml.safe_dump({'optim': {'lr': 1e-4}, 'data': {'seq_len': 4096},
                                           'train': {'micro_batch_size': 4, 'grad_accum': ga},
                                           'parallel': {'world_size': 1}}))
            path.with_suffix('.env').write_text(f'YUFORMER_PRETRAIN_OUT={root}/data/experiments/batch_pilot/{name}\n')
    return directory


def test_batch_contracts_prepare_from_scoped_anchor(anchor_evidence):
    root, selection, audit = anchor_evidence
    anchor = night.build_batch_anchor(root, selection, audit)
    path = root / night.ANCHOR; atomic_json(path, anchor)
    directory = batch_configs(root)
    result = batch.prepare(root, path)
    assert result['status'] == 'PASS'
    assert len(result['configs']) == 9
    assert yaml.safe_load((directory / 'batch_1m_mid.yaml').read_text())['optim']['lr'] == 0.0012


def test_batch_preparation_refuses_to_rewrite_started_runs_atomically(tmp_path):
    directory = batch_configs(tmp_path)
    value = {'status': 'PASS', 'expected_results': 15, 'observed_results': 15,
             'selections': {size: {'learning_rate': 0.0003} for size in contract.SIZES}}
    path = tmp_path / 'selection.json'; path.write_text(json.dumps(value))
    batch.prepare(tmp_path, path)
    before = {p.name: p.read_bytes() for p in directory.glob('*.yaml')}
    run = tmp_path / 'data/experiments/batch_pilot/batch_4m_mid'
    run.mkdir(parents=True); (run / 'run_identity.json').write_text('{}')
    value['selections']['200m']['learning_rate'] *= 2; path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='already started'):
        batch.prepare(tmp_path, path)
    assert {p.name: p.read_bytes() for p in directory.glob('*.yaml')} == before


def boundary_selection(selection):
    result = copy.deepcopy(selection)
    curve = result['curves']['200m']
    curve[-1]['validation_loss'] = 1.0
    result['selections']['200m'] = curve[-1]
    result['boundary_selections'] = [{'size': '200m', 'level': 'vhigh', 'learning_rate': curve[-1]['learning_rate']}]
    return result


def test_automatic_expansion_is_matched_and_budget_bounded(anchor_evidence):
    root, selection, _ = anchor_evidence
    policy = json.loads((Path(__file__).resolve().parents[1] / 'configs/night_research_policy.json').read_text())
    result = night.plan_lr_extensions(root, boundary_selection(selection), policy)
    assert result['status'] == 'EXPAND'
    assert [row['learning_rate'] for row in result['new_arms']] == [0.0018, 0.0024, 0.0048]
    assert result['additional_tokens'] == 1_500_000_000
    policy['automatic_lr_extensions']['maximum_additional_tokens'] = 500_000_000
    result = night.plan_lr_extensions(root, boundary_selection(selection), policy)
    assert len(result['new_arms']) == 1
    assert result['limits']


def test_incomplete_curve_never_triggers_expansion(anchor_evidence):
    root, selection, _ = anchor_evidence
    policy = json.loads((Path(__file__).resolve().parents[1] / 'configs/night_research_policy.json').read_text())
    selection = boundary_selection(selection); selection['curves']['200m'].pop(0)
    assert night.plan_lr_extensions(root, selection, policy)['new_arms'] == []


def test_live_extension_queue_prevents_roster_mutation(anchor_evidence):
    root, selection, audit = anchor_evidence
    policy = (Path(__file__).resolve().parents[1] / 'configs/night_research_policy.json').read_text()
    (root / night.POLICY).write_text(policy)
    before = (root / contract.CONTRACT).read_bytes()
    plan = night.expand_lr_search(root, boundary_selection(selection), True, audit)
    assert plan['status'] == 'WAIT_FOR_EXTENSION_QUEUE_COMPLETION'
    assert (root / contract.CONTRACT).read_bytes() == before
