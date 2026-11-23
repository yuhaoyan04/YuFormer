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
from frozen_training_evidence import freeze_report_bindings, production_output_dir, production_mutation_lease, publish_checked
from validation_source_proof import require_validation_source_proof
from training_source_windows import require_window_admission, POLICY as WINDOW_POLICY
from temporal_order_admission import require_temporal_admission
from qualify_native_merge import require_native_qualification
from sustained_memory_review import review_memory_log
from allocator_retry_review import require_no_allocator_retries

PREFERENCE = 'data/health/model-scale-user-preference.json'
RELEASE = 'data/processed/releases/pretrain-1t-frozen.release.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind_consumption_producers(root, mixtures):
    inputs = {}
    for report in mixtures.values():
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
    days = 10**12/(8*p10*.9*86400)
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
        'experiments/e4_curriculum/scripts/training_data_origin.py': sha(root/'experiments/e4_curriculum/scripts/training_data_origin.py')}
    for name, expected in selection['evidence_sha256'].items():
        read_bound(root/name, expected)
        inputs[name] = expected
    verify_sustained_result(root, selection, inputs)
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
    if base['attention'].get('sdpa_backend') == 'cudnn_flex_swa':
        parity_name = 'data/health/flex-module-parity.json'
        parity = json.loads((root/parity_name).read_bytes())
        rows = parity.get('results', [])
        if parity.get('status') != 'PASS' or len(rows) != 8:
            raise ValueError('hybrid SWA adoption needs all eight compiled actual attention-module parity reports')
        selected_heads = [row for row in rows if row.get('heads') == model['num_heads']]
        if len(selected_heads) != 2 or {row.get('master_fp32') for row in selected_heads} != {False, True}:
            raise ValueError('compiled attention parity does not cover the actual selected GQA head shape')
        for row in rows:
            if (row.get('status') != 'PASS' or len(row.get('cases', [])) != 3
                    or any(case.get('status') != 'PASS' for case in row['cases'])):
                raise ValueError('compiled hybrid attention output/gradient parity is incomplete')
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
    inputs, proof_archive = freeze_report_bindings(root, inputs)
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
