import hashlib
import importlib.util
import json
import math
import shutil
from pathlib import Path

import pytest
import yaml

import research_scope as scope
import pretrain_readiness_watch as watch
import audit_experiment_identities as identities
import validate_formal_contracts as contracts
import research_confirmation as confirmation
import run_experiment_queue as queue

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('scaling_only_freeze', ROOT / 'experiments/e4_curriculum/scripts/freeze_1t_recipe.py')
FREEZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FREEZE)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


@pytest.fixture
def scoped(tmp_path):
    write_json(tmp_path / scope.POLICY, json.loads((ROOT / scope.POLICY).read_text()))
    return tmp_path


def test_deferred_studies_cannot_launch_or_become_pass(scoped, monkeypatch):
    monkeypatch.setattr(watch, 'ROOT', scoped)
    monkeypatch.setattr(queue, 'ROOT', scoped)
    for group in ('e2', 'e3', 'e2_confirm', 'e3_confirm'):
        assert not scope.group_enabled(scoped, group)
        value, event = watch.start_group_if_eligible(group, True, 'ready', {'status': 'PASS'}, [], True)
        assert value['status'] == 'DEFERRED'
        assert event['decision'] == 'deferred_by_research_scope'
        assert not queue.launch_admitted(1, group, None)
    assert confirmation.prepare(scoped, 'e2')['status'] == 'DEFERRED'
    assert confirmation.prepare(scoped, 'e3')['status'] == 'DEFERRED'
    evidence = [{'status': 'DEFERRED'}] * 4
    assert scope.ablation_requirements_satisfied(scoped, *evidence)
    assert all(value['status'] == 'DEFERRED' for value in evidence)


def test_active_roster_excludes_deferred_artifacts_and_e1_uses_both_slots(scoped, monkeypatch):
    names = identities.expected_names(scoped)
    assert 'e2' not in names and 'e3' not in names
    assert len(names['e1']) == 24
    assert contracts.expected_roster(scoped)['e2'] == 0
    assert contracts.expected_roster(scoped)['e3'] == 0
    monkeypatch.setattr(watch, 'ROOT', scoped)
    monkeypatch.setattr(watch, 'HEALTH', scoped / 'health')
    write_json(scoped / 'health/gpu-resource-policy.json', {
        'status': 'ACTIVE', 'allowed_gpu_indices': [1, 4],
        'max_concurrent_jobs': 2, 'allow_busy_gpu_sharing': False,
    })
    import gpu_resource_policy as resources
    monkeypatch.setattr(resources, 'visible_gpu_indices', lambda: {0, 1, 2, 3, 4})
    monkeypatch.setattr(watch, 'experiment_result_roster_complete', lambda _: False)
    assert watch.publish_queue_allocation('e1', {}, [1, 4], 'formal_idle', None) == [1, 4]


@pytest.fixture
def freeze_ready(scoped, monkeypatch):
    monkeypatch.setattr(FREEZE, 'load_training_ratios', lambda root, inputs: {
        stage: {domain: 1.0 for domain in FREEZE.DOMAINS} for stage in FREEZE.STAGES})
    for relative in ('model/configs/model_configs.yaml', 'pretrain/configs/pretrain_base.yaml',
                     'pretrain/configs/pretrain_base.env', 'pretrain/configs/research_budget.yaml',
                     'data-process/configs/mix_warmup_50b.yaml', 'data-process/configs/mix_stable_800b.yaml',
                     'data-process/configs/mix_anneal.yaml'):
        destination = scoped / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    evidence = scoped / 'evidence.json'; evidence.write_text('{}')
    snapshot = {'evidence.json': hashlib.sha256(evidence.read_bytes()).hexdigest()}
    write_json(scoped / 'experiments/e1_scaling/results/scaling_fit.json', {
        'status': 'PASS', 'independent_performance': {'status': 'PASS', 'snapshot': snapshot},
        'production_recommendation': {'feasible_within_45_days': True, 'n_params': 200_000_000}})
    write_json(scoped / 'experiments/e1_scaling/results/lr_resolution.json', {
        'status': 'PASS', 'fit': {'log_intercept': math.log(.001), 'exponent': 0.},
        'batch_lr_factor': 1., 'selected_tokens_per_step': 1048576,
        'research_scope_sha256': scope.scope_sha256(scoped)})
    write_json(scoped / 'report/experiment_identity_audit_latest.json', {
        'status': 'PASS', 'evidence_snapshot': snapshot, 'evidence_snapshot_count': 1})
    write_json(scoped / 'data/processed/audits/e3_token_streams_20261021.json', {
        'status': 'PASS', 'streams': [{'name': f'val_{domain}', 'status': 'PASS',
        'expected_reference_tokens': 1000, 'actual_total_tokens_with_eod': 1001, 'expected_docs': 1}
        for domain in FREEZE.DOMAINS]})
    return scoped


def test_freeze_uses_fixed_baselines_without_any_e2_e3_results(freeze_ready):
    root = freeze_ready
    assert not (root / 'experiments/e2_architecture/results/selection.json').exists()
    assert not (root / 'experiments/e3_data_mix/results/selection.json').exists()
    result = FREEZE.freeze(root)
    assert result['status'] == 'PASS'
    assert result['architecture_evidence']['selection_status'] == 'FIXED_BASELINE'
    assert result['architecture_evidence']['run_name'] is None
    assert result['architecture_evidence']['comparative_evidence'] is False
    assert result['mixture_evidence']['status'] == 'FIXED_BASELINE'
    assert result['mixture_evidence']['comparative_evidence'] is False
    assert result['stages']['stable']['target_final_tokens'] == 800_000_000_000
    assert scope.scope_binding_current(root, result)
    assert not any('/e2/' in path or 'confirmation.json' in path for path in result['inputs'])
    policy = json.loads((root / scope.POLICY).read_text()); policy['profile'] = 'full'
    write_json(root / scope.POLICY, policy)
    assert not scope.scope_binding_current(root, result)


@pytest.mark.parametrize('relative', ['experiments/e1_scaling/results/scaling_fit.json',
                                    'experiments/e1_scaling/results/lr_resolution.json',
                                    'report/experiment_identity_audit_latest.json'])
def test_scaling_only_freeze_still_requires_real_evidence(freeze_ready, relative):
    path = freeze_ready / relative
    value = json.loads(path.read_text()); value['status'] = 'BLOCKED'; write_json(path, value)
    with pytest.raises(ValueError, match='must PASS'):
        FREEZE.freeze(freeze_ready)


def test_scaling_only_freeze_cannot_skip_independent_performance(freeze_ready):
    path = freeze_ready / 'experiments/e1_scaling/results/scaling_fit.json'
    value = json.loads(path.read_text()); value['independent_performance'] = {'status': 'PENDING'}
    write_json(path, value)
    with pytest.raises(ValueError, match='independent performance'):
        FREEZE.freeze(freeze_ready)


def test_full_scope_keeps_architecture_mixture_and_confirmation_requirements(freeze_ready):
    path = freeze_ready / scope.POLICY
    policy = json.loads(path.read_text()); policy['profile'] = 'full'; write_json(path, policy)
    assert not scope.ablation_requirements_satisfied(freeze_ready, {'status': 'DEFERRED'})
    with pytest.raises(FileNotFoundError):
        FREEZE.freeze(freeze_ready)


def test_freeze_budgets_each_phase_from_its_training_measurement(freeze_ready, monkeypatch):
    ratios = dict(zip(FREEZE.STAGES, [.8, 1.2, 1.5], strict=True))
    monkeypatch.setattr(FREEZE, 'load_training_ratios', lambda root, inputs: {
        stage: {domain: value for domain in FREEZE.DOMAINS} for stage, value in ratios.items()})
    result = FREEZE.freeze(freeze_ready)
    for stage, row in result['stages'].items():
        expected = row['target_final_tokens'] * 1.02 / ratios[stage]
        assert abs(row['selection_reference_tokens'] - expected) <= 1
        assert result['tokenizer_domain_ratios_by_stage'][stage]['web'] == ratios[stage]


def test_lr_readiness_checks_only_active_e1_configs(scoped, monkeypatch):
    for relative in ('pretrain/configs/e1/active.yaml', 'pretrain/configs/e2/deferred.yaml'):
        path = scoped / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('x: 1\n')
        path.with_suffix('.env').write_text('X=1\n')
    selection = scoped / 'selection.json'; selection.write_text('{}')
    resolver = scoped / 'resolver.py'; resolver.write_text('pass\n')
    resolution = scoped / 'resolution.json'; report = scoped / 'contracts.json'
    cfg = scoped / 'pretrain/configs/e1/active.yaml'; relative = str(cfg.relative_to(scoped))
    write_json(resolution, {'status': 'PASS', 'resolver_tool_sha256': watch.sha256(resolver),
        'research_scope_sha256': scope.scope_sha256(scoped), 'selection_sha256': watch.sha256(selection),
        'batch_selection_sha256': watch.sha256(selection),
        'configs': [{'config': relative, 'config_sha256': watch.sha256(cfg)}]})
    write_json(report, {'status': 'PASS', 'roster_exact': True, 'rows': [{'config': relative,
        'status': 'PASS', 'config_sha256': watch.sha256(cfg),
        'env': str(cfg.with_suffix('.env').relative_to(scoped)),
        'env_sha256': watch.sha256(cfg.with_suffix('.env'))}]})
    for name, value in [('ROOT', scoped), ('LR_RESOLUTION', resolution), ('LR_SELECTION', selection),
                        ('BATCH_SELECTION', selection), ('LR_RESOLVER', resolver)]:
        monkeypatch.setattr(watch, name, value)
    monkeypatch.setattr(watch, 'selection_is_current', lambda *_: True)
    assert watch.lr_resolution_is_current({'status': 'PASS'}, report)
