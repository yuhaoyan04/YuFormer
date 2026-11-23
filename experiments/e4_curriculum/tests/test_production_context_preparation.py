"""Reject changed budgets and qualification-only data before publishing a stage."""
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    'production_context_preparation', ROOT/'experiments/e4_curriculum/scripts/prepare_production_context_stage.py')
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


@pytest.fixture
def assets(tmp_path):
    budget = tmp_path/'budget.json'
    frozen = tmp_path/'frozen.json'
    model = tmp_path/'model.yaml'
    for path in (budget, frozen, model):
        path.write_text('{}\n')
    plan = {
        'scope': 'PRODUCTION', 'execution_order': [4096, 16384, 65536],
        'approved_effective_budget': {'file': budget.name, 'sha256': PREPARE.digest(budget)},
        'pretrain_frozen_recipe': {'file': frozen.name, 'sha256': PREPARE.digest(frozen)},
        'stages': [{'seq_len': 16384, 'target_labels': 29997662208,
                    'qualified_model': {'file': model.name, 'sha256': PREPARE.digest(model)}}],
    }
    plan_file = tmp_path/'plan.yaml'
    plan_file.write_text(yaml.safe_dump(plan))
    release = tmp_path/'release.json'
    release.write_text(json.dumps({'scope': 'PRODUCTION', 'seq_len': 16384,
                                   'target_labels': 29997662208}))
    return dict(root=tmp_path, plan_file=plan_file, plan_sha256=PREPARE.digest(plan_file),
                seq_len=16384, handoff_file=tmp_path/'not-yet-completed-parent.json',
                window_release=release, output=tmp_path/'output', run_dir=tmp_path/'run')


@pytest.mark.parametrize('asset', ['plan.yaml', 'budget.json', 'frozen.json', 'model.yaml'])
def test_changed_frozen_stage_inputs_are_rejected_before_output(assets, asset):
    (assets['root']/asset).write_text('{}\n# changed\n')
    with pytest.raises(ValueError, match='changed'):
        PREPARE.render(**assets)
    assert not assets['output'].exists()


@pytest.mark.parametrize('key,value', [('scope', 'QUALIFICATION'), ('seq_len', 65536),
                                      ('target_labels', 100663296)])
def test_small_or_wrong_context_release_cannot_become_production(assets, key, value):
    release = json.loads(assets['window_release'].read_text())
    release[key] = value
    assets['window_release'].write_text(json.dumps(release))
    with pytest.raises(ValueError, match='exact complete long-window'):
        PREPARE.render(**assets)
    assert not assets['output'].exists()


def test_stage_cannot_be_published_without_a_completed_parent(assets):
    with pytest.raises(FileNotFoundError):
        PREPARE.render(**assets)
    assert not assets['output'].exists()
