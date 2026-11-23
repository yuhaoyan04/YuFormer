import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from adopt_training_recipe import require_selection, non_embedding_params, verify_sustained_result, source_sha256
from adopt_training_recipe import training_execution_contract, require_execution_contract


@pytest.fixture
def selection():
    model = dict(vocab_size=65536, hidden_size=2560, intermediate_size=6912,
        num_layers=18, num_heads=40, num_kv_heads=10, attention_every=6,
        qk_norm=True, attention_output_gate=True, tie_word_embeddings=True)
    base = {'optim': {'lr': .0001}, 'attention': {'sdpa_backend': 'cudnn_flex_swa'},
            'parallel': {'world_size': 8, 'replicate_degree': 2, 'shard_degree': 4}}
    preference = dict(status='ACTIVE_USER_PREFERENCE', world_size=8,
                      target_training_tokens=10**12, launch_runtime_ceiling_days=33)
    document = dict(status='SELECTED_FOR_PRODUCTION_QUALIFICATION', model=copy.deepcopy(model),
        optimizer=copy.deepcopy(base['optim']), attention=copy.deepcopy(base['attention']),
        parallel=copy.deepcopy(base['parallel']), parameters=non_embedding_params(model)+65536*2560,
        sustained_measurement=dict(steps=200, trained_tokens=838860800, steady_points=150,
            parameters=non_embedding_params(model)+65536*2560,
            world_size=8, quality_review_completed=True, p10_tps_per_device=53000),
        evidence_sha256={'result.json': 'a'*64})
    return document, model, base, preference


def test_selection_requires_measured_larger_model_and_retains_budget(selection):
    count, days = require_selection(*selection)
    assert count == 1516600064
    assert 30 < days < 33


@pytest.mark.parametrize('drift', ['deterministic', 'loss', 'batch', 'compiler', 'missing'])
def test_adoption_binds_actual_execution_settings(selection, drift):
    document = selection[0]
    config = {'data': {'seq_len': 4096}, 'train': {'micro_batch_size': 8, 'grad_accum': 16,
        'activation_checkpointing': False, 'bf16': True, 'cuda_graphs': False, 'deterministic': True},
        'compile': {'enable': True}, 'loss': {'num_chunks': 4}}
    document['production_user_tradeoff_review'] = {}
    document['execution_contract'] = copy.deepcopy(training_execution_contract(config))
    require_execution_contract(document, config)
    if drift == 'deterministic': config['train']['deterministic'] = False
    elif drift == 'loss': config['loss']['num_chunks'] = 8
    elif drift == 'batch': config['train'].update(micro_batch_size=4, grad_accum=32)
    elif drift == 'compiler': config['compile']['enable'] = False
    else: document.pop('execution_contract')
    with pytest.raises(ValueError, match='execution'):
        require_execution_contract(document, config)


@pytest.mark.parametrize('drift', ['model', 'lr', 'backend', 'topology', 'parameters',
    'short_probe', 'tokens', 'missing_quality_review', 'nonfinite_tps', 'too_slow', 'expanded_budget'])
def test_unqualified_model_cannot_be_adopted(selection, drift):
    document, model, base, preference = selection
    if drift == 'model': document['model']['num_layers'] += 1
    elif drift == 'lr': document['optimizer']['lr'] = .0002
    elif drift == 'backend': document['attention']['sdpa_backend'] = 'cudnn'
    elif drift == 'topology': document['parallel']['shard_degree'] = 8
    elif drift == 'parameters': document['parameters'] += 1
    elif drift == 'short_probe': document['sustained_measurement']['steps'] = 24
    elif drift == 'tokens': document['sustained_measurement']['trained_tokens'] -= 1
    elif drift == 'missing_quality_review': document['sustained_measurement']['quality_review_completed'] = False
    elif drift == 'nonfinite_tps': document['sustained_measurement']['p10_tps_per_device'] = float('nan')
    elif drift == 'too_slow': document['sustained_measurement']['p10_tps_per_device'] = 40000
    else: preference['launch_runtime_ceiling_days'] = 45
    with pytest.raises(ValueError):
        require_selection(document, model, base, preference)


@pytest.fixture
def measured_run(tmp_path, selection):
    document, model, base, _ = selection
    source = Path(__file__).resolve().parents[3]
    benchmark = tmp_path/'experiments/e2_architecture/scripts/benchmark_model_sizes.py'
    benchmark.parent.mkdir(parents=True)
    shutil.copyfile(source/'experiments/e2_architecture/scripts/benchmark_model_sizes.py', benchmark)
    run = tmp_path/'run'
    run.mkdir()
    model_file = tmp_path/'model.yaml'
    model_file.write_text(yaml.safe_dump({'presets': {'measured': model}}))
    identity = {'identity_sha256': 'identity', 'source_sha256': source_sha256(tmp_path),
        'model_configs': str(model_file), 'model_configs_sha256': hashlib.sha256(model_file.read_bytes()).hexdigest()}
    config = {'model': {'preset': 'measured'}, 'optim': base['optim'],
              'attention': base['attention'], 'parallel': base['parallel']}
    domains = ('web', 'code', 'math', 'chinese', 'academic', 'books', 'encyclopedia', 'qa', 'multilingual')
    result = {'status': 'PASS', 'identity_sha256': 'identity', 'issues': [],
        'hyperparameters': {'world_size': 8, 'tokens_per_step': 4194304, 'trained_steps': 200,
                           'trained_tokens': 838860800},
        'series': {'throughput(tps)': [{'step': i, 'value': 53000} for i in range(1, 201)]},
        'metrics': {'validation_metrics/loss': {'last': 6.4}, **{
            f'validation_metrics/domain/{d}/loss': {'last': 6.4} for d in domains}}}
    values = {'result.json': result, 'run_identity.json': identity, 'resolved_config.yaml': config}
    for name, value in values.items():
        (run/name).write_text(yaml.safe_dump(value) if name.endswith('.yaml') else json.dumps(value))
    (run/'training.log').write_text(''.join(f'step: {step} loss: 6.4\n' for step in range(1,201)))
    inputs = {str(p.relative_to(tmp_path)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (model_file, *run.iterdir())}
    document['sustained_measurement']['run_dir'] = 'run'
    document['sustained_measurement']['memory_log'] = 'run/training.log'
    return tmp_path, document, inputs, values


def test_actual_sustained_result_closes_model_identity_and_throughput(measured_run):
    root, document, inputs, _ = measured_run
    verify_sustained_result(root, document, inputs)


@pytest.mark.parametrize('fault', ['wrong_actual_model', 'wrong_actual_lr', 'short_actual_run',
                                  'fabricated_throughput', 'missing_domain', 'stale_source'])
def test_adoption_rejects_mismatched_actual_training_despite_updated_evidence_hashes(measured_run, fault):
    root, document, inputs, values = measured_run
    if fault == 'wrong_actual_model':
        path = root/'model.yaml'
        model = yaml.safe_load(path.read_bytes())
        model['presets']['measured']['num_layers'] = 8
        path.write_text(yaml.safe_dump(model))
        values['run_identity.json']['model_configs_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        inputs['model.yaml'] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif fault == 'wrong_actual_lr': values['resolved_config.yaml']['optim']['lr'] = .0003
    elif fault == 'short_actual_run': values['result.json']['hyperparameters']['trained_steps'] = 24
    elif fault == 'fabricated_throughput': document['sustained_measurement']['p10_tps_per_device'] = 54000
    elif fault == 'missing_domain': values['result.json']['metrics'].pop('validation_metrics/domain/web/loss')
    else: values['run_identity.json']['source_sha256'] = '0'*64
    for name, value in values.items():
        path = root/'run'/name
        path.write_text(yaml.safe_dump(value) if name.endswith('.yaml') else json.dumps(value))
        inputs[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        verify_sustained_result(root, document, inputs)


def test_late_allocator_retries_cannot_hide_behind_a_24_step_memory_review(measured_run):
    root, document, inputs, _ = measured_run
    path = root/'run/training.log'
    path.write_text(path.read_text().replace('step: 120 loss: 6.4\n',
                    'step: 120 loss: 6.4\nmemory allocation failed with OOM\n'))
    inputs['run/training.log'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='late allocator retries'):
        verify_sustained_result(root, document, inputs)


def test_titan_retry_counter_alone_blocks_adoption_without_a_cpp_oom_message(measured_run):
    root, document, inputs, _ = measured_run
    path = root/'run/training.log'
    path.write_text(path.read_text().replace('step: 120 loss: 6.4\n',
                    'step: 120 loss: 6.4\nWARNING - 2 CUDA memory allocation retries.\n'))
    inputs['run/training.log'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='counter summaries'):
        verify_sustained_result(root, document, inputs)
