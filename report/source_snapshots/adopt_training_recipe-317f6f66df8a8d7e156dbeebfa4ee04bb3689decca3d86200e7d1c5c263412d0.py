#!/usr/bin/env python3
"""Freeze a newly qualified training graph while preserving exact selected data."""
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'pretrain/scripts'))
from qualification_identity import qualification_identity, report_matches
from user_pretrain_authorization import load_authorization, identity_exception_is_bound
from run_production_recovery import validate_report as validate_recovery
from training_tokenizer_calibration import load_training_ratios, REPORT as CALIBRATION
from freeze_1t_recipe import non_embedding_params, source_sha256, verify_identity_audit_snapshot
from training_data_origin import resolve_data_origin, read_bound, FROZEN, BASE
from publish_training_block_manifest import require_admission
from frozen_training_evidence import production_output_dir, production_mutation_lease, publish_checked
from validation_source_proof import require_validation_source_proof
from training_source_windows import require_window_admission, POLICY as WINDOW_POLICY
from temporal_order_admission import require_temporal_admission
from qualify_native_merge import require_native_qualification
from sustained_memory_review import review_memory_log
from allocator_retry_review import require_no_allocator_retries
from adoption_research_evidence import freeze_adoption_research_evidence
from development_selection_evidence import require_development_review
from checkpoint_durability_evidence import require_checkpoint_directory_durability
from production_selection_evidence import require_production_quality_review
from shape_selection_evidence import require_shape_quality_review
from global_training_budget import forecast_global_training
from runtime_limited_plan import resolve_effective_budget

PREFERENCE = 'data/health/model-scale-user-preference.json'
RELEASE = 'data/processed/releases/pretrain-1t-frozen.release.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind_consumption_producers(root, mixtures):
    inputs = {}
    for report in mixtures.values():
        preparation = report.get('consumption_preparation_tool')
        if preparation is not None:
            name = preparation.get('file')
            if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('invalid rebudget consumption producer path')
            path = root/name
            expected = preparation.get('sha256')
            if not path.is_file() or sha(path) != expected:
                path = root/'report/source_snapshots'/f'{Path(name).stem}-{expected}.py'
            read_bound(path, expected)
            inputs[str(path.relative_to(root))] = expected
        for filename, expected in (
                ('audit_1t_overlap.py', report.get('audit_tool_sha256')),
                ('tokenize_1t_orders.py', report.get('tokenization_tool_sha256'))):
            current = root/'experiments/e4_curriculum/scripts'/filename
            if current.is_file() and sha(current) == expected:
                path = current
            else:
                path = root/'report/source_snapshots'/f'{Path(filename).stem}-{expected}.py'
            read_bound(path, expected)
            inputs[str(path.relative_to(root))] = expected
    return inputs


def bind_preparation_producer(root, selection):
    expected = selection.get('preparation_tool_sha256')
    path = root/'experiments/e4_curriculum/scripts/prepare_production_model.py'
    if not path.is_file() or sha(path) != expected:
        path = root/'report/source_snapshots'/f'prepare_production_model-{expected}.py'
    read_bound(path, expected)
    return {str(path.relative_to(root)):expected}


def require_compiled_module_cases(row):
    expected = {(0, 2, 129), (5, 2, 129), (0, 8, 4096), (5, 8, 4096)}
    cases = row.get('cases', [])
    actual = {(case.get('layer'), case.get('batch_size'), case.get('seq_len')) for case in cases}
    if (row.get('status') != 'PASS' or len(cases) != 4 or actual != expected
            or any(case.get('status') != 'PASS' for case in cases)):
        raise ValueError('compiled hybrid attention must cover all four actual shapes including global 4K B8')


def training_execution_contract(config):
    train = config['train']
    return {'seq_len': config['data']['seq_len'],
            'train': {key: train[key] for key in ('micro_batch_size', 'grad_accum',
                'activation_checkpointing', 'bf16', 'cuda_graphs')},
            'deterministic': train.get('deterministic', False),
            'compile': config['compile'],
            'loss': config.get('loss', {'num_chunks': 8})}


def require_execution_contract(selection, config):
    contract = selection.get('execution_contract')
    required = ('production_user_tradeoff_review' in selection
        or str(selection.get('sustained_measurement', {}).get('run_dir', '')).startswith(
            'data/experiments/final_deterministic_production_controls_20261115/'))
    if contract is None:
        if required:
            raise ValueError('selected deterministic production measurement needs an explicit execution contract')
    elif contract != training_execution_contract(config):
        raise ValueError('selected deterministic/loss/compiler/batch execution differs from actual configuration')


def require_global_runtime(selection, base):
    bundle = selection.get('global_training_budget')
    if bundle is None:
        if 'production_user_tradeoff_review' in selection:
            raise ValueError('current user-approved model needs the entire 4K/16K/64K runtime budget')
        return None
    budget, measurements = bundle.get('approved_budget', {}), bundle.get('measurements', {})
    result = forecast_global_training(budget, measurements)
    if bundle.get('runtime_execution_plan') is not None:
        budget, result = resolve_effective_budget(budget, bundle['runtime_execution_plan'])
        if bundle['runtime_execution_plan']['measurements'] != measurements:
            raise ValueError('runtime allocation was derived from another measured stage throughput')
    stage = budget['stages'][0]
    if (stage['updates'] != base['train']['expected_iters']
            or sum(base['data']['stage_sample_limits'])*base['data']['seq_len'] != stage['tokens']
            or measurements['4k-pretrain']['p10_tps_per_device'] != selection['sustained_measurement']['p10_tps_per_device']
            or result != bundle.get('forecast')
            or result['status'] != 'WITHIN_APPROVED_TOTAL_RUNTIME'):
        raise ValueError('prepared 4K consumption or full-chain runtime differs from the approved total budget')
    return result


def verify_global_runtime_evidence(root, selection, inputs):
    bundle = selection.get('global_training_budget')
    if bundle is None:
        if 'production_user_tradeoff_review' in selection:
            raise ValueError('missing full-chain measured runtime evidence')
        return
    budget_name = 'data/health/user-total-pretrain-and-length-budget.json'
    if budget_name not in inputs or inputs[budget_name] != bundle.get('approved_budget_sha256'):
        raise ValueError('global runtime must bind the actual user-approved total-budget document')
    raw, _ = read_bound(root/budget_name, inputs[budget_name])
    if json.loads(raw) != bundle.get('approved_budget'):
        raise ValueError('declared global budget differs from the user-approved bytes')
    if bundle.get('runtime_execution_plan') is not None:
        name = bundle.get('runtime_execution_plan_file')
        if (not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts
                or name not in inputs or inputs[name] != bundle.get('runtime_execution_plan_sha256')):
            raise ValueError('runtime token allocation must bind its actual immutable derivation report')
        raw, _ = read_bound(root/name, inputs[name])
        document = json.loads(raw)
        if (document.get('nominal_budget_sha256') != bundle['approved_budget_sha256']
                or document.get('proposal') != bundle['runtime_execution_plan']):
            raise ValueError('runtime allocation report differs from the measured nominal budget derivation')
    spec = importlib.util.spec_from_file_location('_global_runtime_actual_series',
        root/'experiments/e2_architecture/scripts/benchmark_model_sizes.py')
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    for stage in bundle['approved_budget']['stages']:
        measured = bundle['measurements'][stage['name']]
        name = measured.get('run_dir')
        if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('global runtime needs a bound actual repository-relative stage run')
        values = {}
        for filename in ('result.json', 'run_identity.json', 'resolved_config.yaml'):
            relative = str(Path(name)/filename)
            if relative not in inputs:
                raise ValueError('global runtime stage lacks immutable result/identity/config evidence')
            payload, _ = read_bound(root/relative, inputs[relative])
            values[filename] = yaml.safe_load(payload) if filename.endswith('.yaml') else json.loads(payload)
        result, identity, config = (values[k] for k in ('result.json', 'run_identity.json', 'resolved_config.yaml'))
        if (result.get('status') != 'PASS' or result.get('identity_sha256') != identity.get('identity_sha256')
                or identity.get('source_sha256') != source_sha256(root)
                or config['data']['seq_len'] != stage['seq_len']
                or config['parallel']['world_size'] != 8
                or config['attention'] != selection['attention']
                or config['train'].get('deterministic', False) is not selection['execution_contract']['deterministic']
                or training_execution_contract(config) != measured.get('execution_contract')
                or config['parallel'] != measured.get('parallel')):
            raise ValueError('global runtime measurement has another source or stage execution policy')
        model_path = Path(identity['model_configs'])
        if not model_path.is_absolute():
            model_path = root/model_path
        if not model_path.is_relative_to(root) or str(model_path.relative_to(root)) not in inputs:
            raise ValueError('global runtime must bind each actually measured model configuration')
        payload, _ = read_bound(model_path, inputs[str(model_path.relative_to(root))])
        if sha(model_path) != identity['model_configs_sha256']:
            raise ValueError('global runtime model differs from actual training identity')
        model = yaml.safe_load(payload)['presets'][config['model']['preset']]
        if measured.get('model') != model:
            raise ValueError('global runtime stage must bind its exact context/recomputation model configuration')
        # Context capacity/recomputation may change at stage boundaries. The
        # learned architecture must be exactly the selected model throughout.
        architectural = lambda x: {k:v for k,v in x.items() if k not in
            {'max_seq_len', 'rope_theta', 'selective_recompute'}}
        if architectural(model) != architectural(selection['model']):
            raise ValueError('long-stage speed cannot be borrowed from a different architecture')
        steps, skip = (200, 50) if stage['seq_len'] == 4096 else (24, 8)
        evaluation = benchmark.evaluate(result, steps=steps, skip_steps=skip)
        if (evaluation['status'] != 'PASS_SCREENING'
                or evaluation['p10_tps_per_device'] != measured['p10_tps_per_device']):
            raise ValueError('global runtime throughput does not match the actual stage series')
        log = measured.get('memory_log')
        if not isinstance(log, str) or Path(log).is_absolute() or '..' in Path(log).parts or log not in inputs:
            raise ValueError('global runtime must bind full-stage memory and allocator evidence')
        payload, _ = read_bound(root/log, inputs[log])
        review_memory_log(payload, steps=steps, skip_steps=skip)
        require_no_allocator_retries(payload)


def require_selection(selection, model, base, preference):
    if (preference.get('status') != 'ACTIVE_USER_PREFERENCE'
            or preference.get('world_size') != 8 or preference.get('target_training_tokens') != 10**12
            or preference.get('launch_runtime_ceiling_days') != 33):
        raise ValueError('current larger-model preference must preserve the authorized eight-GPU 1T/33-day ceiling')
    if selection.get('status') != 'SELECTED_FOR_PRODUCTION_QUALIFICATION':
        raise ValueError('model selection has not completed its sustained quality review')
    if selection.get('model') != model or selection.get('optimizer') != base['optim']:
        raise ValueError('selected architecture/optimizer differs from the prepared production bytes')
    if selection.get('attention') != base['attention'] or selection.get('parallel') != base['parallel']:
        raise ValueError('selected attention backend/topology differs from production')
    require_execution_contract(selection, base)
    count = non_embedding_params(model)+int(model['vocab_size'])*int(model['hidden_size'])*(1 if model.get('tie_word_embeddings') else 2)
    if count != selection.get('parameters') or not 10**9 <= count <= 5*10**9:
        raise ValueError('selected parameter count is not the actual authorized 1B-5B model')
    measured = selection.get('sustained_measurement') or {}
    if (measured.get('steps') != 200 or measured.get('trained_tokens') != 838860800
            or measured.get('steady_points') != 150 or measured.get('world_size') != 8
            or measured.get('parameters') != count
            or measured.get('quality_review_completed') is not True):
        raise ValueError('selection lacks the matched 200-update/all-domain quality review')
    p10 = measured.get('p10_tps_per_device')
    if not isinstance(p10, (int, float)) or not math.isfinite(p10) or p10 <= 0:
        raise ValueError('selected sustained throughput is invalid')
    global_runtime = require_global_runtime(selection, base)
    days = global_runtime['total_days'] if global_runtime is not None else 10**12/(8*p10*.9*86400)
    if days > 33:
        raise ValueError('selected model exceeds the authorized 33-day launch ceiling')
    evidence = selection.get('evidence_sha256')
    if not isinstance(evidence, dict) or not evidence:
        raise ValueError('selection has no immutable measured evidence')
    return count, days


def verify_sustained_result(root, selection, inputs):
    measured = selection['sustained_measurement']
    name = measured.get('run_dir')
    if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('selected sustained measurement has no bound repository-relative run directory')
    directory = root/name
    documents = {}
    for filename in ('result.json', 'run_identity.json', 'resolved_config.yaml'):
        path = directory/filename
        relative = str(path.relative_to(root))
        if relative not in inputs:
            raise ValueError('selected sustained result/identity/config is missing from immutable evidence')
        raw, _ = read_bound(path, inputs[relative])
        documents[filename] = yaml.safe_load(raw) if filename.endswith('.yaml') else json.loads(raw)
    result, identity, config = (documents[k] for k in ('result.json', 'run_identity.json', 'resolved_config.yaml'))
    require_execution_contract(selection, config)
    if (result.get('status') != 'PASS' or result.get('identity_sha256') != identity.get('identity_sha256')
            or identity.get('source_sha256') != source_sha256(root)
            or config.get('parallel') != selection['parallel'] or config.get('attention') != selection['attention']):
        raise ValueError('selected sustained result belongs to another source/backend/topology')
    path = Path(identity['model_configs'])
    if not path.is_absolute():
        path = root/path
    if not path.is_relative_to(root) or str(path.relative_to(root)) not in inputs:
        raise ValueError('actual sustained model bytes are not bound by selection evidence')
    raw, _ = read_bound(path, identity['model_configs_sha256'])
    actual_model = yaml.safe_load(raw)['presets'][config['model']['preset']]
    if actual_model != selection['model']:
        raise ValueError('actual trained model differs from the declared selected architecture')
    for key in set(config['optim']) | set(selection['optimizer']):
        if key not in {'warmup_steps', 'decay_steps', 'lr_schedule', 'min_lr_ratio'}:
            if config['optim'].get(key) != selection['optimizer'].get(key):
                raise ValueError('actual sustained optimizer differs from selection: '+key)
    spec = importlib.util.spec_from_file_location('_adoption_sustained_accounting',
        root/'experiments/e2_architecture/scripts/benchmark_model_sizes.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    evaluated = module.evaluate(result, steps=200, skip_steps=50)
    if (evaluated['status'] != 'PASS_SCREENING' or evaluated['steady_points'] != 150
            or not math.isclose(evaluated['p10_tps_per_device'], measured['p10_tps_per_device'], abs_tol=1e-8)):
        raise ValueError('selected sustained throughput does not match the actual measured series')
    tags = [tag for tag in result['metrics'] if tag == 'validation_metrics/loss' or
            tag.startswith('validation_metrics/domain/') and tag.endswith('/loss')]
    expected_tags = {'validation_metrics/loss'} | {f'validation_metrics/domain/{d}/loss' for d in
        ('web', 'code', 'math', 'chinese', 'academic', 'books', 'encyclopedia', 'qa', 'multilingual')}
    if set(tags) != expected_tags or any(not math.isfinite(result['metrics'][tag]['last']) for tag in tags):
        raise ValueError('selected sustained validation must cover all nine domains with finite losses')
    log_name = measured.get('memory_log')
    if (not isinstance(log_name, str) or Path(log_name).is_absolute() or '..' in Path(log_name).parts
            or log_name not in inputs):
        raise ValueError('selected sustained measurement must bind its entire actual memory log')
    raw, _ = read_bound(root/log_name, inputs[log_name])
    review_memory_log(raw, steps=200, skip_steps=50)
    require_no_allocator_retries(raw)
    verify_global_runtime_evidence(root, selection, inputs)


def archive_parent(root, release):
    origin_sha = release['frozen_recipe_sha256']
    relative = f'data/processed/recipe_origins/{origin_sha}/manifest.json'
    path = root/relative
    raw, digest = read_bound(path)
    archive = json.loads(raw)
    if (archive.get('status') != 'PASS_IMMUTABLE_ORIGIN_ARCHIVE'
            or archive.get('frozen_recipe_sha256') != origin_sha):
        raise ValueError('original data recipe archive is absent or changed')
    row = archive['files'][FROZEN]
    name = row['archive']
    if Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('invalid archived data recipe path')
    parent_raw, _ = read_bound(root/name, origin_sha)
    return json.loads(parent_raw), archive, relative, digest


def freeze_adoption(root, selection_path):
    root, selection_path = Path(root).resolve(), Path(selection_path).resolve()
    if not selection_path.is_relative_to(root):
        raise ValueError('selection evidence must be a repository-relative immutable artifact')
    release = json.loads((root/RELEASE).read_bytes())
    parent, archive, archive_name, archive_sha = archive_parent(root, release)
    base = yaml.safe_load((root/BASE).read_bytes())
    model_path = root/'model/configs/pretrain_final_model_configs.yaml'
    model = yaml.safe_load(model_path.read_bytes())['presets'][base['model']['preset']]
    preference = json.loads((root/PREFERENCE).read_bytes())
    selection = json.loads(selection_path.read_bytes())
    preparation_inputs = bind_preparation_producer(root, selection)
    count, days = require_selection(selection, model, base, preference)
    inputs = {str(selection_path.relative_to(root)): sha(selection_path), PREFERENCE: sha(root/PREFERENCE),
        **preparation_inputs,
        RELEASE: sha(root/RELEASE), archive_name: archive_sha,
        'experiments/e4_curriculum/scripts/adopt_training_recipe.py': sha(Path(__file__)),
        'experiments/e4_curriculum/scripts/sustained_memory_review.py': sha(
            root/'experiments/e4_curriculum/scripts/sustained_memory_review.py'),
        'experiments/e4_curriculum/scripts/allocator_retry_review.py': sha(
            root/'experiments/e4_curriculum/scripts/allocator_retry_review.py'),
        'experiments/e4_curriculum/scripts/adoption_research_evidence.py': sha(
            root/'experiments/e4_curriculum/scripts/adoption_research_evidence.py'),
        'experiments/e4_curriculum/scripts/training_data_origin.py': sha(root/'experiments/e4_curriculum/scripts/training_data_origin.py')}
    inputs['experiments/e4_curriculum/scripts/global_training_budget.py'] = sha(
        root/'experiments/e4_curriculum/scripts/global_training_budget.py')
    inputs['experiments/e4_curriculum/scripts/runtime_limited_plan.py'] = sha(
        root/'experiments/e4_curriculum/scripts/runtime_limited_plan.py')
    for name, expected in selection['evidence_sha256'].items():
        read_bound(root/name, expected)
        inputs[name] = expected
    verify_sustained_result(root, selection, inputs)
    inputs.update(require_development_review(root, selection))
    inputs.update(require_production_quality_review(root, selection))
    inputs.update(require_shape_quality_review(root, selection))
    inputs.update(require_checkpoint_directory_durability(root))
    native_inputs, native_qualification = require_native_qualification(root)
    inputs.update(native_inputs)
    authorization = load_authorization(root)
    if (authorization['architecture'].get('config') != str(model_path.relative_to(root))
            or authorization['architecture'].get('preset') != base['model']['preset']
            or float(authorization['training']['peak_learning_rate']) != float(base['optim']['lr'])):
        raise ValueError('authorization record has not been updated from the user-approved model selection')
    for relative in ('data/health/pretrain-user-full-pretrain-authorization.json',
            'pretrain/configs/research_scope.json', 'pretrain/configs/research_budget.yaml',
            'data/health/torchtitan-qualification.json', 'data/health/production-recovery-qualification.json',
            'data/health/torchtitan-env-freeze.txt', 'report/experiment_identity_audit_latest.json',
            'data/processed/audits/pretrain_1t_token_streams.json'):
        inputs[relative] = sha(root/relative)
    current = qualification_identity(root)
    numerical = json.loads((root/'data/health/torchtitan-qualification.json').read_bytes())
    if numerical.get('status') != 'PASS' or not report_matches(numerical, current):
        raise ValueError('actual production graph needs current full GPU qualification')
    validate_recovery(root)
    if base['attention'].get('sdpa_backend') in {'cudnn_flex_swa', 'flash_flex_swa'}:
        parity_name = 'data/health/flex-module-parity.json'
        parity = json.loads((root/parity_name).read_bytes())
        rows = parity.get('results', [])
        if parity.get('status') != 'PASS' or len(rows) != 8:
            raise ValueError('hybrid SWA adoption needs all eight compiled actual attention-module parity reports')
        selected_heads = [row for row in rows if row.get('heads') == model['num_heads']]
        if len(selected_heads) != 2 or {row.get('master_fp32') for row in selected_heads} != {False, True}:
            raise ValueError('compiled attention parity does not cover the actual selected GQA head shape')
        for row in rows:
            require_compiled_module_cases(row)
            if (row.get('candidate_backend') != base['attention']['sdpa_backend']
                    or row.get('deterministic') is not base['train'].get('deterministic', False)):
                raise ValueError('compiled hybrid attention report must match the selected backend and deterministic policy')
            for name, digest in row.get('sources', {}).items():
                read_bound(root/name, digest)
                inputs[name] = digest
        inputs[parity_name] = sha(root/parity_name)
    for name in ('pretrain/scripts/audit_training_progress.py',
                 'pretrain/scripts/run_authorized_pretrain.py'):
        inputs[name] = sha(root/name)
    calibration = json.loads((root/CALIBRATION).read_bytes())
    if (calibration.get('schema_version') != 2
            or calibration.get('sampling') != 'independent-blake2b-domain-rank-permutation-v1'):
        raise ValueError('biased prefix calibration cannot be used for training adoption')
    ratios = load_training_ratios(root, inputs)
    order_path = root/'pretrain/backends/torchtitan/yuformer_ext/block_permutation.py'
    spec = importlib.util.spec_from_file_location('_adoption_training_order', order_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    env = {}
    for line in (root/'pretrain/configs/pretrain_base.env').read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            key, value = line.split('=', 1); env[key] = value
    rendered = (root/BASE).read_text()
    for key, value in env.items():
        rendered = rendered.replace('${'+key+'}', value)
    if '${' in rendered:
        raise ValueError('unresolved adoption data/model contract')
    order = module.configuration_order_identity(yaml.safe_load(rendered), require_production=True)
    master_path = Path(yaml.safe_load(rendered)['data']['block_permutation_manifest'])
    master = json.loads(master_path.read_bytes())
    publisher = root/'experiments/e4_curriculum/scripts/publish_training_block_manifest.py'
    if master.get('publication_tool_sha256') != sha(publisher):
        raise ValueError('production order was admitted by stale publication code')
    if master.get('evidence_publication_tool_sha256') != sha(
            root/'experiments/e4_curriculum/scripts/frozen_training_evidence.py'):
        raise ValueError('production order evidence requires current checked durable publication')
    inputs[str(publisher.relative_to(root))] = sha(publisher)
    admission = {}
    for row in master.get('admission_evidence', []):
        name = row.get('file')
        if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('invalid production order admission evidence path')
        path = master_path.parent/name
        raw, digest = read_bound(path, row.get('sha256'))
        admission[row['source']] = json.loads(raw)
        inputs[str(path.relative_to(root))] = digest
    full_name = 'data/processed/audits/pretrain_1t_token_streams.json'
    mixtures = {phase: admission.get(f'data/health/pretrain-1t-{phase}-mixture-preflight.json')
                for phase in ('warmup', 'stable', 'anneal')}
    require_admission(admission.get(full_name) or {}, mixtures, base['data']['stage_sample_limits'],
                      base['data']['seq_len'],
                      {phase: parent['stages'][phase]['target_final_domain_weights'] for phase in mixtures})
    if admission[full_name] != json.loads((root/full_name).read_bytes()):
        raise ValueError('current physical audit differs from immutable production order admission')
    inputs.update(bind_consumption_producers(root, mixtures))
    if (master.get('schema_version') != 3 or master.get('source_window_policy') != WINDOW_POLICY
            or master.get('window_selection_tool_sha256') != sha(
                root/'experiments/e4_curriculum/scripts/training_source_windows.py')):
        raise ValueError('training adoption requires explicit admitted source windows and current selection code')
    source_windows = []
    for index, phase in enumerate(('warmup','stable','anneal')):
        row = master['stages'][index]
        window = require_window_admission(root, mixtures[phase], phase=phase,
            stage_samples=base['data']['stage_sample_limits'][index], seq_len=base['data']['seq_len'],
            release_sha256=sha(root/RELEASE), stream_meta_sha256=row['stream_meta_sha256'],
            data_frozen_recipe_sha256=release['frozen_recipe_sha256'])
        if (row.get('source_sample_offset') != window['source_sample_offset']
                or row.get('source_window_selection') != mixtures[phase]['source_window']['selection_report']
                or window['block_samples'] != master['block_samples']):
            raise ValueError('training order differs from its exact consumption window admission: '+phase)
        inputs.update(window['inputs'])
        source_windows.append({'phase':phase, **mixtures[phase]['source_window']})
    validation_inputs, validation_archive = require_validation_source_proof(
        root, admission[full_name], master['source_roster_fingerprint'])
    inputs.update(validation_inputs)
    temporal_inputs, temporal_admission = require_temporal_admission(root, master_path)
    inputs.update(temporal_inputs)
    inputs[str(master_path.relative_to(root))] = sha(master_path)
    for row in order['artifacts']:
        inputs[str(Path(row['file']).relative_to(root))] = row['sha256']
    inputs['experiments/e4_curriculum/scripts/frozen_training_evidence.py'] = sha(
        root/'experiments/e4_curriculum/scripts/frozen_training_evidence.py')
    original_inputs = dict(inputs)
    inputs, proof_archive = freeze_adoption_research_evidence(root, inputs)
    frozen = copy.deepcopy(parent)
    frozen.update(frozen_at=datetime.now(timezone.utc).isoformat(), inputs=inputs,
        outputs={name: sha(root/name) for name in parent['outputs']},
        freeze_tool_sha256=sha(root/'experiments/e4_curriculum/scripts/freeze_1t_recipe.py'),
        learning_rate=float(base['optim']['lr']), tokenizer_domain_ratios_by_stage=ratios,
        user_authorization=authorization,
        architecture_evidence={'selection_status': 'USER_REQUESTED_MEASURED_MODEL_SCALE_SELECTION',
            'comparative_evidence': True, 'production_model_configs': str(model_path.relative_to(root)),
            'final_preset': base['model']['preset'], 'actual_params': count,
            'non_embedding_params': non_embedding_params(model)},
        training_adoption={'schema_version': 1, 'tool': str(Path(__file__).relative_to(ROOT)),
            'tool_sha256': sha(Path(__file__)), 'qualification_identity': current['sha256'],
            'selected_sustained_projected_days': days, 'final_production_canary_still_required': True},
        data_origin={'schema_version': 1, 'kind': 'TRAINING_ONLY_WITH_EXACT_DATA_REUSE',
            'parent_frozen_recipe_sha256': release['frozen_recipe_sha256'],
            'release_manifest_sha256': sha(root/RELEASE), 'archive_manifest': archive_name,
            'archive_manifest_sha256': archive_sha})
    frozen['training_adoption']['evidence_archive'] = proof_archive['manifest']
    frozen['training_adoption']['evidence_archive_sha256'] = proof_archive['manifest_sha256']
    frozen['training_adoption']['validation_source_proof'] = validation_archive
    frozen['training_adoption']['temporal_order_admission'] = temporal_admission
    frozen['training_adoption']['native_writeback_qualification'] = native_qualification
    frozen['training_adoption']['source_window_policy'] = WINDOW_POLICY
    frozen['training_adoption']['source_windows'] = source_windows
    if selection.get('global_training_budget') is not None:
        budget_name = 'data/health/user-total-pretrain-and-length-budget.json'
        budget = selection['global_training_budget']['approved_budget']
        runtime_plan = selection['global_training_budget'].get('runtime_execution_plan')
        if runtime_plan is not None:
            budget, _ = resolve_effective_budget(budget, runtime_plan)
        frozen['optimizer_stage_steps'] = dict(zip(('warmup', 'stable', 'anneal'),
            (phase['updates'] for phase in budget['stages'][0]['phases'])))
        frozen['data_origin']['approved_budget_repartition'] = {
            'file': budget_name, 'sha256': sha(root/budget_name),
            'validator_sha256': sha(root/'experiments/e4_curriculum/scripts/global_training_budget.py')}
        if runtime_plan is not None:
            frozen['data_origin']['approved_budget_repartition'].update(
                runtime_execution_plan={'file': selection['global_training_budget']['runtime_execution_plan_file'],
                    'sha256': selection['global_training_budget']['runtime_execution_plan_sha256']},
                runtime_validator_sha256=sha(root/'experiments/e4_curriculum/scripts/runtime_limited_plan.py'))
        frozen['training_adoption']['global_training_budget'] = selection['global_training_budget']
    frozen['data_origin']['reuse_scope'] = 'EXACT_PHYSICAL_POPULATION_AND_TOKEN_BYTES'
    frozen['data_origin']['training_consumption'] = 'EXPLICIT_BLOCK_ALIGNED_SOURCE_WINDOWS'
    frozen['data_origin']['supervised_prefixes_unchanged'] = all(
        row['source_sample_offset'] == 0 for row in source_windows)
    identity = json.loads((root/'report/experiment_identity_audit_latest.json').read_bytes())
    if not identity_exception_is_bound(root, frozen, identity):
        raise ValueError('identity audit exceeds the explicitly authorized historical source-drift waiver')
    original_identity = archive['files']['report/experiment_identity_audit_latest.json']
    old_identity = json.loads(read_bound(root/original_identity['archive'], original_identity['sha256'])[0])
    roster = lambda report: {(r['group'], r['name']) for r in report['rows']}
    if len(identity['rows']) != 113 or roster(identity) != roster(old_identity):
        raise ValueError('the historical source-drift waiver may not expand beyond its original 113 rows')
    verify_identity_audit_snapshot(root, identity)
    candidate = root/FROZEN
    staging = candidate.with_name(candidate.stem+'.adoption-preview.json')
    staging.write_text(json.dumps(frozen, sort_keys=True, indent=2)+'\n')
    resolve_data_origin(root, root/RELEASE, staging)
    for name, expected in {**inputs, **frozen['outputs']}.items():
        read_bound(root/name, expected)
    for name, expected in original_inputs.items():
        read_bound(root/name, expected)
    if qualification_identity(root) != current:
        raise ValueError('production qualification inputs changed during adoption freeze')
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    lease = production_mutation_lease(root, 'adopt-qualified-training-recipe') if args.publish else nullcontext()
    with lease:
        if args.publish:
            jobs = json.loads((root/'data/health/pretrain-1t-tokenization.json').read_bytes())
            if jobs.get('status') != 'PASS':
                raise ValueError('original tokenization controller must complete before changing its frozen data recipe')
            diagnostic = json.loads((root/'data/health/uniform-training-tokenizer-pilot.json').read_bytes())
            if diagnostic.get('status') == 'RUNNING':
                raise ValueError('independent diagnostic still protects the original frozen recipe')
            if any(production_output_dir(root).glob('checkpoint/step-*/.metadata')):
                raise ValueError('formal checkpoints already exist; prelaunch training adoption is forbidden')
        result = freeze_adoption(root, args.selection)
        if args.publish:
            target = root/FROZEN
            publish_checked(target, (json.dumps(result, sort_keys=True, indent=2)+'\n').encode())
    print(json.dumps({'status': 'PUBLISHED' if args.publish else 'VALIDATED_PREVIEW',
                      'model': result['architecture_evidence'], 'data_origin': result['data_origin']}, sort_keys=True))


if __name__ == '__main__':
    main()
