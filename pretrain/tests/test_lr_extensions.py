import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import lr_pilot_contract as contract
import prepare_lr_extensions
import select_lr_pilots as selector


@pytest.fixture
def search(tmp_path):
    rows = [dict(size='200m', level=f'ext{lr}', learning_rate=lr / 1_000_000,
                 run_name=f'e1_lr_200m_ext{lr}', group=contract.EXTENSION_GROUP)
            for lr in (1800, 2400, 4800)]
    manifest = tmp_path / contract.CONTRACT
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'schema_version': 1, 'extensions': rows}))
    paths = []
    for size in contract.SIZES:
        entries = [(level, 0.000075 * 2 ** i, 5.0 + abs(i - 2))
                   for i, level in enumerate(contract.LEVELS)]
        if size == '200m':
            entries = [(level, lr, 5.0 - i * 0.1) for i, (level, lr, _) in enumerate(entries)]
            entries += [(r['level'], r['learning_rate'], 4.8 + i) for i, r in enumerate(rows)]
        for level, lr, loss in entries:
            result = {'status': 'PASS', 'run_name': f'e1_lr_{size}_{level}',
                      'identity_sha256': f'{size}-{level}',
                      'hyperparameters': {'learning_rate': lr, 'tokens_per_step': selector.TOKENS_PER_STEP,
                                          'trained_steps': selector.MATCHED_TRAINED_TOKENS // selector.TOKENS_PER_STEP,
                                          'trained_tokens': selector.MATCHED_TRAINED_TOKENS},
                      'metrics': {'validation_metrics/loss': {'last': loss}}}
            path = tmp_path / f'{result["run_name"]}.json'
            path.write_text(json.dumps(result))
            paths.append(path)
    return tmp_path, rows, paths


def test_extensions_bracket_original_upper_boundary_by_actual_lr(search):
    root, rows, paths = search
    result = selector.select(paths, rows)
    assert result['status'] == 'PASS'
    assert result['expected_results'] == result['observed_results'] == 18
    assert result['selections']['200m']['level'] == 'vhigh'
    assert result['boundary_selections'] == []
    result['search_contract_sha256'] = contract.contract_sha256(root)
    assert contract.selection_is_current(root, result)
    (root / contract.CONTRACT).write_text(json.dumps({'schema_version': 1, 'extensions': rows[:2]}))
    assert not contract.selection_is_current(root, result)


def test_extension_new_upper_boundary_still_blocks(search):
    _, rows, paths = search
    path = next(p for p in paths if 'ext4800' in p.name)
    d = json.loads(path.read_text())
    d['metrics']['validation_metrics/loss']['last'] = 1.0
    path.write_text(json.dumps(d))
    result = selector.select(paths, rows)
    assert result['status'] == 'BLOCKED'
    assert result['boundary_selections'][0]['level'] == 'ext4800'


def test_declared_missing_extension_cannot_be_ignored(search):
    _, rows, paths = search
    result = selector.select([p for p in paths if 'ext4800' not in p.name], rows)
    assert result['status'] == 'BLOCKED'
    assert '200m:ext4800' in result['missing']


def test_undeclared_extension_cannot_influence_selection(search):
    _, rows, paths = search
    result = selector.select(paths, rows[:2])
    assert result['status'] == 'BLOCKED'
    assert result['rejected'][0]['reason'] == 'undeclared_run_name'


def test_extension_lr_must_match_declared_value(search):
    _, rows, paths = search
    p = next(p for p in paths if 'ext2400' in p.name)
    d = json.loads(p.read_text()); d['hyperparameters']['learning_rate'] *= 2
    p.write_text(json.dumps(d))
    assert selector.select(paths, rows)['status'] == 'BLOCKED'


def test_extension_contracts_preserve_baseline_and_refuse_mutation(search):
    root, rows, _ = search
    base = root / 'pretrain/configs/e1_pilot/e1_lr_200m_vhigh'
    base.parent.mkdir(parents=True)
    config = {'run_name': base.stem, 'seed': 20261001,
              'optim': {'lr': 0.0012}, 'train': {'budget_tokens': 500_000_000}}
    base.with_suffix('.yaml').write_text(yaml.safe_dump(config))
    base.with_suffix('.env').write_text('YUFORMER_CONFIG=base.yaml\nYUFORMER_PRETRAIN_OUT=/old\nMASTER_PORT=29600\n')
    prepare_lr_extensions.prepare(root)
    prepare_lr_extensions.prepare(root)
    for row in rows:
        path = root / f'pretrain/configs/{contract.EXTENSION_GROUP}/{row["run_name"]}.yaml'
        expected = copy.deepcopy(config)
        expected['run_name'], expected['optim']['lr'] = row['run_name'], row['learning_rate']
        assert yaml.safe_load(path.read_text()) == expected
    config['seed'] = 2
    base.with_suffix('.yaml').write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='refusing to change'):
        prepare_lr_extensions.prepare(root)


def test_duplicate_extension_contract_rejected(search):
    root, rows, _ = search
    (root / contract.CONTRACT).write_text(json.dumps({'schema_version': 1, 'extensions': rows + rows[:1]}))
    with pytest.raises(ValueError, match='duplicate'):
        contract.extensions(root)
