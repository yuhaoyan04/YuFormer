import copy
import hashlib
import json
from pathlib import Path
import sys
import importlib.util
import shutil

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import training_data_origin as origin
import audit_1t_overlap as overlap

STACK_PATH = Path(__file__).resolve().parents[3]/'pretrain/scripts/audit_pretrain_stack.py'
STACK_SPEC = importlib.util.spec_from_file_location('_data_origin_stack', STACK_PATH)
STACK = importlib.util.module_from_spec(STACK_SPEC)
STACK_SPEC.loader.exec_module(STACK)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(root, name, value):
    path = root/name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value))
    return path


@pytest.fixture
def adoption(tmp_path):
    base = {'data': {'stages': ['a', 'b', 'c'], 'meta': ['x', 'y', 'z'],
                     'seq_len': 4096, 'stage_sample_limits': [8, 16, 8], 'epochs': 1},
            'parallel': {'world_size': 8, 'replicate_degree': 2, 'shard_degree': 4},
            'model': {'preset': 'old'},
            'train': {'budget_tokens': 32*4096, 'expected_iters': 4, 'micro_batch_size': 1, 'grad_accum': 1},
            'optim': {'lr_schedule': 'wsd', 'warmup_steps': 1, 'decay_steps': 1}}
    outputs = {origin.BASE: sha(write(tmp_path, origin.BASE, yaml.safe_dump(base).encode()))}
    stages = {}
    for phase in origin.STAGES:
        name = f'data-process/configs/{phase}.yaml'
        outputs[name] = sha(write(tmp_path, name, b'fixed original selected data config'))
        stages[phase] = {'config': name, 'target_final_domain_weights': {'web': 1}}
    parent = {'status': 'PASS', 'inputs': {}, 'outputs': outputs, 'stages': stages,
              'selected_tokens_per_step': 32768, 'optimizer_stage_steps': {'warmup': 1, 'stable': 2, 'anneal': 1},
              'mixture_evidence': {'web': 1}}
    frozen_path = write(tmp_path, origin.FROZEN, parent)
    old_frozen_sha = sha(frozen_path)
    release = write(tmp_path, 'release.json', {'status': 'PASS', 'dataset_id': 'same',
        'snapshot_id': 'same', 'frozen_recipe_sha256': old_frozen_sha})
    bindings = dict(outputs, **{origin.FROZEN: old_frozen_sha, 'release.json': sha(release)})
    records = {}
    for name, digest in bindings.items():
        blob = write(tmp_path, 'archive/objects/'+digest, (tmp_path/name).read_bytes())
        records[name] = {'archive': str(blob.relative_to(tmp_path)), 'sha256': digest,
                         'bytes': blob.stat().st_size}
    archive = write(tmp_path, 'archive/manifest.json', {'schema_version': 1,
        'status': 'PASS_IMMUTABLE_ORIGIN_ARCHIVE',
        'purpose': 'PRESERVE_ORIGINAL_RECIPE_BYTES_NOT_PRODUCTION_ADOPTION',
        'frozen_recipe_sha256': old_frozen_sha, 'files': records})
    current = copy.deepcopy(parent)
    base['model']['preset'] = 'larger-model'
    base['data'].update(sampling_policy='block_permutation', block_permutation_manifest='permutation.json')
    write(tmp_path, origin.BASE, yaml.safe_dump(base).encode())
    current['outputs'][origin.BASE] = sha(tmp_path/origin.BASE)
    current['data_origin'] = {'schema_version': 1, 'kind': 'TRAINING_ONLY_WITH_EXACT_DATA_REUSE',
        'release_manifest_sha256': sha(release), 'parent_frozen_recipe_sha256': old_frozen_sha,
        'archive_manifest': 'archive/manifest.json', 'archive_manifest_sha256': sha(archive)}
    write(tmp_path, origin.FROZEN, current)
    return tmp_path, release, current, base


def test_larger_model_reuses_exact_original_data_without_rebinding_release(adoption):
    root, release, current, _ = adoption
    result = origin.resolve_data_origin(root, release)
    assert result['mode'] == 'TRAINING_ONLY_WITH_EXACT_DATA_REUSE'
    assert sha(Path(result['data_recipe'])) == json.loads(release.read_text())['frozen_recipe_sha256']
    assert result['release_sha256'] == sha(release)


def test_overlap_requires_the_exact_archived_data_origin_for_a_new_training_recipe(adoption):
    root, release, current, _ = adoption
    frozen, selected, binding = overlap.validated_data_contract(root, release, root/origin.FROZEN)
    assert frozen == current
    assert selected['frozen_recipe_sha256'] == binding['data_recipe_sha256']
    assert binding['training_recipe_sha256'] == sha(root/origin.FROZEN)
    current['data_origin']['archive_manifest_sha256'] = '0'*64
    write(root, origin.FROZEN, current)
    with pytest.raises(ValueError, match='byte identity'):
        overlap.validated_data_contract(root, release, root/origin.FROZEN)


def test_overlap_rejects_atomic_release_replacement_during_origin_validation(adoption, monkeypatch):
    root, release, _, _ = adoption
    resolve = origin.resolve_data_origin
    def changed(*args):
        result = resolve(*args)
        write(root, 'release.json', {'status': 'PASS', 'different': True})
        return result
    monkeypatch.setattr(origin, 'resolve_data_origin', changed)
    with pytest.raises(ValueError, match='byte identity'):
        overlap.validated_data_contract(root, release, root/origin.FROZEN)


def test_final_stack_validates_archived_data_recipe_separately_from_new_training_recipe(adoption):
    root, release, _, _ = adoption
    helper = root/'experiments/e4_curriculum/scripts/training_data_origin.py'
    helper.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(origin.__file__), helper)
    data_sha = json.loads(release.read_bytes())['frozen_recipe_sha256']
    assert STACK.data_recipe_binding_errors(root, release, root/origin.FROZEN,
        {'status': 'PASS', 'frozen_recipe_sha256': data_sha}) == []
    assert STACK.data_recipe_binding_errors(root, release, root/origin.FROZEN,
        {'status': 'PASS', 'frozen_recipe_sha256': sha(root/origin.FROZEN)})
    changed = json.loads((root/origin.FROZEN).read_bytes())
    changed['data_origin']['archive_manifest_sha256'] = '0'*64
    (root/origin.FROZEN).write_text(json.dumps(changed))
    assert STACK.data_recipe_binding_errors(root, release, root/origin.FROZEN,
        {'status': 'PASS', 'frozen_recipe_sha256': data_sha})


def test_changed_selected_data_config_is_rejected(adoption):
    root, release, current, _ = adoption
    write(root, current['stages']['stable']['config'], b'new selection')
    with pytest.raises(ValueError, match='byte identity'):
        origin.resolve_data_origin(root, release)


def test_changed_consumed_sample_set_is_rejected_even_with_new_output_hash(adoption):
    root, release, current, base = adoption
    base['data']['stage_sample_limits'][1] += 8
    write(root, origin.BASE, yaml.safe_dump(base).encode())
    current['outputs'][origin.BASE] = sha(root/origin.BASE)
    write(root, origin.FROZEN, current)
    with pytest.raises(ValueError, match='consumed sample set'):
        origin.resolve_data_origin(root, release)


def test_mixture_change_is_rejected(adoption):
    root, release, current, _ = adoption
    current['stages']['warmup']['target_final_domain_weights'] = {'web': .9, 'math': .1}
    write(root, origin.FROZEN, current)
    with pytest.raises(ValueError, match='quotas, mixture'):
        origin.resolve_data_origin(root, release)


def test_missing_archive_hash_cannot_disable_byte_validation(adoption):
    root, release, current, _ = adoption
    del current['data_origin']['archive_manifest_sha256']
    write(root, origin.FROZEN, current)
    with pytest.raises(ValueError, match='required data-origin SHA'):
        origin.resolve_data_origin(root, release)


def test_release_cannot_be_rebound_to_a_different_dataset(adoption):
    root, release, current, _ = adoption
    document = json.loads(release.read_bytes())
    document['dataset_id'] = 'different'
    write(root, 'release.json', document)
    current['data_origin']['release_manifest_sha256'] = sha(release)
    write(root, origin.FROZEN, current)
    with pytest.raises(ValueError, match='release itself changed'):
        origin.resolve_data_origin(root, release)


@pytest.fixture
def repartition(adoption):
    root, release, current, base = adoption
    repository = Path(__file__).resolve().parents[3]
    budget = json.loads((repository/'data/health/user-total-pretrain-and-length-budget.json').read_bytes())
    old = yaml.safe_load((repository/'pretrain/configs/pretrain_base.yaml').read_bytes())
    old['data'].pop('sampling_policy', None)
    old['data'].pop('block_permutation_manifest', None)
    # Rebuild a small immutable archive with the actual original batch geometry.
    parent = copy.deepcopy(current)
    parent.pop('data_origin')
    parent['selected_tokens_per_step'] = 4194304
    parent['optimizer_stage_steps'] = {'warmup': 11921, 'stable': 190735, 'anneal': 35762}
    parent['outputs'][origin.BASE] = sha(write(root, origin.BASE, yaml.safe_dump(old).encode()))
    write(root, origin.FROZEN, parent)
    parent_sha = sha(root/origin.FROZEN)
    write(root, 'release.json', {'status': 'PASS', 'dataset_id': 'same', 'snapshot_id': 'same',
        'frozen_recipe_sha256': parent_sha})
    bindings = dict(parent['outputs'], **{origin.FROZEN: parent_sha, 'release.json': sha(release)})
    records = {}
    for name, digest in bindings.items():
        blob = write(root, 'new-archive/objects/'+digest, (root/name).read_bytes())
        records[name] = {'archive': str(blob.relative_to(root)), 'sha256': digest, 'bytes': blob.stat().st_size}
    archive = write(root, 'new-archive/manifest.json', {'schema_version': 1,
        'status': 'PASS_IMMUTABLE_ORIGIN_ARCHIVE',
        'purpose': 'PRESERVE_ORIGINAL_RECIPE_BYTES_NOT_PRODUCTION_ADOPTION',
        'frozen_recipe_sha256': parent_sha, 'files': records})
    budget_path = write(root, 'data/health/user-total-pretrain-and-length-budget.json', budget)
    validator = write(root, 'experiments/e4_curriculum/scripts/global_training_budget.py',
        (repository/'experiments/e4_curriculum/scripts/global_training_budget.py').read_bytes())
    base = copy.deepcopy(old)
    base['data'].update(sampling_policy='block_permutation', block_permutation_manifest='new-order.json',
        stage_sample_limits=[p['samples'] for p in budget['stages'][0]['phases']])
    base['train'].update(expected_iters=budget['stages'][0]['updates'], budget_tokens=budget['stages'][0]['tokens'])
    base['parallel']['fsdp_reshard_after_forward'] = 'never'
    write(root, origin.BASE, yaml.safe_dump(base).encode())
    current = copy.deepcopy(parent)
    current['optimizer_stage_steps'] = dict(zip(origin.STAGES, (p['updates'] for p in budget['stages'][0]['phases'])))
    current['outputs'][origin.BASE] = sha(root/origin.BASE)
    current['data_origin'] = {'schema_version': 1, 'kind': 'TRAINING_ONLY_WITH_EXACT_DATA_REUSE',
        'release_manifest_sha256': sha(release), 'parent_frozen_recipe_sha256': parent_sha,
        'archive_manifest': str(archive.relative_to(root)), 'archive_manifest_sha256': sha(archive),
        'approved_budget_repartition': {'file': str(budget_path.relative_to(root)), 'sha256': sha(budget_path),
            'validator_sha256': sha(validator)}}
    write(root, origin.FROZEN, current)
    return root, release, current, base


def test_explicit_global_budget_repartitions_training_without_rewriting_physical_recipe(repartition):
    root, release, _, _ = repartition
    result = origin.resolve_data_origin(root, release)
    assert result['mode'] == 'TRAINING_ONLY_WITH_EXACT_DATA_REUSE'
    assert sha(Path(result['data_recipe'])) == json.loads(release.read_bytes())['frozen_recipe_sha256']


@pytest.mark.parametrize('fault', ['samples', 'wsd', 'budget', 'world', 'physical_stream', 'missing_budget_binding'])
def test_repartition_cannot_hide_unapproved_consumption_or_data_drift(repartition, fault):
    root, release, current, base = repartition
    if fault == 'samples': base['data']['stage_sample_limits'][1] += 1024
    elif fault == 'wsd': base['optim']['decay_steps'] -= 1
    elif fault == 'budget': base['train']['budget_tokens'] = 10**12
    elif fault == 'world': base['parallel']['world_size'] = 4
    elif fault == 'physical_stream': base['data']['stages'][0] = '/unrelated/source'
    else: current['data_origin']['approved_budget_repartition'].pop('sha256')
    write(root, origin.BASE, yaml.safe_dump(base).encode())
    current['outputs'][origin.BASE] = sha(root/origin.BASE)
    write(root, origin.FROZEN, current)
    with pytest.raises(ValueError):
        origin.resolve_data_origin(root, release)


def test_runtime_limited_consumption_still_preserves_all_physical_data_bindings(repartition):
    from runtime_limited_plan import derive_runtime_limited_plan
    root, release, current, base = repartition
    bp = root/'data/health/user-total-pretrain-and-length-budget.json'
    nominal = json.loads(bp.read_bytes())
    rates = {s['name']: {'seq_len': s['seq_len'], 'world_size': 8, 'parameters': nominal['parameters'],
        'p10_tps_per_device': p10} for s, p10 in zip(nominal['stages'], (48858.0234375,40894.80859375,24933.76953125))}
    proposal = derive_runtime_limited_plan(nominal, rates)
    document = {'nominal_budget_sha256': sha(bp), 'proposal': proposal}
    path = write(root, 'allocation.json', document)
    producer = Path(__file__).resolve().parents[3]/'experiments/e4_curriculum/scripts/runtime_limited_plan.py'
    validator = write(root, 'experiments/e4_curriculum/scripts/runtime_limited_plan.py', producer.read_bytes())
    current['data_origin']['approved_budget_repartition'].update(
        runtime_execution_plan={'file': 'allocation.json', 'sha256': sha(path)},
        runtime_validator_sha256=sha(validator))
    effective = proposal['proposed_budget']['stages'][0]
    base['train'].update(expected_iters=effective['updates'], budget_tokens=effective['tokens'])
    base['data']['stage_sample_limits'] = [p['samples'] for p in effective['phases']]
    current['optimizer_stage_steps'] = dict(zip(origin.STAGES, (p['updates'] for p in effective['phases'])))
    write(root, origin.BASE, yaml.safe_dump(base).encode())
    current['outputs'][origin.BASE] = sha(root/origin.BASE)
    write(root, origin.FROZEN, current)
    assert origin.resolve_data_origin(root, release)['mode'] == 'TRAINING_ONLY_WITH_EXACT_DATA_REUSE'
    document['proposal']['proposed_budget']['stages'][0]['tokens'] += 4194304
    write(root, 'allocation.json', document)
    current['data_origin']['approved_budget_repartition']['runtime_execution_plan']['sha256'] = sha(path)
    write(root, origin.FROZEN, current)
    with pytest.raises(ValueError, match='does not reproduce'):
        origin.resolve_data_origin(root, release)
