import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import batch_pilot_contract as contract
import night_batch_search as search
import night_research
import select_batch_pilots as selector


@pytest.fixture
def grid(tmp_path):
    path = tmp_path / contract.CONTRACT
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'schema_version': 1, 'extensions': []}))
    (tmp_path / night_research.POLICY).write_text(
        (Path(__file__).resolve().parents[1] / 'configs/night_research_policy.json').read_text())
    paths, audit = [], {'rows': []}
    for index, batch in enumerate(contract.BATCHES):
        tpu = 1048576 * 2 ** index
        for level_index, level in enumerate(contract.LEVELS):
            name = f'batch_{batch}_{level}'
            loss = 3.0 + (level_index if batch == '1m' else abs(level_index - 1))
            value = {'status': 'PASS', 'run_name': name, 'identity_sha256': name,
                     'execution_context': {'loss_only': True},
                     'hyperparameters': {'learning_rate': 0.001 * 2 ** level_index,
                                         'tokens_per_step': tpu,
                                         'trained_tokens': contract.MATCHED_TRAINED_TOKENS,
                                         'trained_steps': contract.MATCHED_TRAINED_TOKENS // tpu},
                     'metrics': {'validation_metrics/loss': {'last': loss}}}
            p = tmp_path / f'{name}.json'; p.write_text(json.dumps(value)); paths.append(p)
            audit['rows'].append({'group': 'batch_pilot', 'name': name, 'status': 'PASS'})
    directory = tmp_path / 'pretrain/configs/batch_pilot'; directory.mkdir()
    for batch in contract.BATCHES:
        name = f'batch_{batch}_mid'
        cfg = {'run_name': name, 'seed': 20261001, 'optim': {'lr': 0.002,
               'batch_pilot_resolution': {'level_multiplier': 1.0}},
               'train': {'budget_tokens': contract.MATCHED_TRAINED_TOKENS}}
        (directory / f'{name}.yaml').write_text(yaml.safe_dump(cfg))
        (directory / f'{name}.env').write_text('YUFORMER_CONFIG=base.yaml\nYUFORMER_PRETRAIN_OUT=/old\nMASTER_PORT=29900\n')
    return tmp_path, paths, audit


def test_matched_batch_extensions_close_boundary_and_preserve_roster(grid):
    root, paths, audit = grid
    initial = selector.select(paths)
    assert initial['status'] == 'BLOCKED'
    result = search.expand(root, initial, audit, False)
    assert result['status'] == 'EXPAND'
    extra = contract.extensions(root)
    assert [r['learning_rate'] for r in extra] == [0.0005, 0.00025]
    assert result['additional_tokens'] == 2 * contract.MATCHED_TRAINED_TOKENS
    for row in extra:
        value = json.loads(paths[0].read_text())
        value['run_name'] = row['run_name']; value['identity_sha256'] = row['run_name']
        value['hyperparameters']['learning_rate'] = row['learning_rate']
        value['metrics']['validation_metrics/loss']['last'] = 4.0
        p = root / f'{row["run_name"]}.json'; p.write_text(json.dumps(value)); paths.append(p)
        cfg = yaml.safe_load((root / f'pretrain/configs/{contract.GROUP}/{row["run_name"]}.yaml').read_text())
        assert cfg['seed'] == 20261001 and cfg['train']['budget_tokens'] == contract.MATCHED_TRAINED_TOKENS
        assert cfg['optim']['batch_pilot_resolution']['level_multiplier'] == row['learning_rate'] / 0.002
    selected = selector.select(paths, extra)
    assert selected['status'] == 'PASS_PROVISIONAL'
    assert selected['expected_results'] == selected['observed_results'] == 11
    selected['search_contract_sha256'] = contract.contract_sha256(root)
    assert contract.selection_is_current(root, selected)
    assert search.expand(root, initial, audit, False)['new_arms'] == []
    assert len(contract.extensions(root)) == 2


def test_incomplete_or_unqualified_batch_curve_cannot_expand(grid):
    root, paths, audit = grid
    policy = json.loads((root / night_research.POLICY).read_text())
    selection = selector.select(paths)
    audit['rows'][0]['status'] = 'FAIL'
    assert search.plan(root, selection, audit, policy)['new_arms'] == []
    audit['rows'][0]['status'] = 'PASS'; selection['curves']['1m'].pop()
    assert search.plan(root, selection, audit, policy)['new_arms'] == []


def test_batch_extensions_obey_limits_and_do_not_mutate_live_queue(grid):
    root, paths, audit = grid
    selection = selector.select(paths)
    assert search.expand(root, selection, audit, True)['status'] == 'WAIT_FOR_EXTENSION_QUEUE_COMPLETION'
    assert contract.extensions(root) == []
    policy = json.loads((root / night_research.POLICY).read_text())
    policy['automatic_batch_extensions']['max_new_arms'] = 0
    result = search.plan(root, selection, audit, policy)
    assert result['status'] == 'LIMIT_REACHED' and result['new_arms'] == []


def test_missing_or_mismatched_supplemental_batch_results_block_selection(grid):
    root, paths, audit = grid
    search.expand(root, selector.select(paths), audit, False)
    extra = contract.extensions(root)
    assert selector.select(paths, extra)['status'] == 'BLOCKED'
    value = json.loads(paths[0].read_text()); value['run_name'] = extra[0]['run_name']
    value['hyperparameters']['learning_rate'] = 9.0
    p = root / 'bad.json'; p.write_text(json.dumps(value))
    assert selector.select(paths + [p], extra)['invalid']


def test_selector_rejects_batch_size_label_mismatch(grid):
    _, paths, _ = grid
    p = paths[0]; value = json.loads(p.read_text()); value['hyperparameters'].update(
        tokens_per_step=2097152, trained_steps=contract.MATCHED_TRAINED_TOKENS // 2097152)
    p.write_text(json.dumps(value))
    assert selector.select(paths)['invalid']


def test_fixed_baseline_does_not_expand_unselected_batch_boundaries(grid):
    root, paths, audit = grid
    from research_scope import POLICY as SCOPE
    (root / SCOPE).write_text(json.dumps({'profile': 'full', 'batch_selection_policy': {
        'mode': 'fixed_baseline', 'batch': '1m', 'tokens_per_step': 1048576}}))
    selection = {'boundary_selections': [{'batch': '2m', 'learning_rate': .004}], 'curves': {}}
    policy = json.loads((root / night_research.POLICY).read_text())
    planned = search.plan(root, selection, audit, policy)
    assert planned['status'] == 'FIXED_BASELINE_POLICY'
    assert planned['new_arms'] == []
