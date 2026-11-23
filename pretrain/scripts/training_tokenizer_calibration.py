"""Fail-closed bindings for phase-specific training tokenizer measurements."""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import yaml

from source_document_binding import verified_source_document_binding

REPORT = 'data/processed/audits/training_tokenizer_calibration.json'
TOOL = 'experiments/e4_curriculum/scripts/calibrate_training_tokens.py'
SAMPLER = 'experiments/e4_curriculum/scripts/uniform_selected_sample.py'
SAMPLING = 'independent-blake2b-domain-rank-permutation-v1'


def bind_materialization_producer(root, parent, relative, recorded, inputs, digest):
    """Bind the actual historical producer through the parent data-origin archive."""
    if recorded == digest(root/relative):
        return
    origin_sha = parent.get('frozen_recipe_sha256')
    if (not isinstance(origin_sha, str) or len(origin_sha) != 64
            or any(c not in '0123456789abcdef' for c in origin_sha)):
        raise ValueError('historical calibration producer needs the bound parent data origin')
    name = f'data/processed/recipe_origins/{origin_sha}/manifest.json'
    archive = json.loads((root/name).read_bytes())
    if (archive.get('status') != 'PASS_IMMUTABLE_ORIGIN_ARCHIVE'
            or archive.get('frozen_recipe_sha256') != origin_sha):
        raise ValueError('historical calibration data-origin archive is invalid')
    frozen_name = 'experiments/e4_curriculum/results/frozen_1t_recipe.json'
    frozen = archive.get('files', {}).get(frozen_name, {})
    row = archive.get('files', {}).get(relative, {})
    for binding, expected in ((frozen, origin_sha), (row, recorded)):
        asset = binding.get('archive')
        if (not isinstance(expected, str) or len(expected) != 64
                or any(c not in '0123456789abcdef' for c in expected)
                or binding.get('sha256') != expected or not isinstance(asset, str)
                or not asset.startswith(f'data/processed/recipe_origins/{origin_sha}/objects/')
                or Path(asset).is_absolute() or '..' in Path(asset).parts
                or digest(root/asset) != expected):
            raise ValueError('historical calibration producer bytes differ from the parent data origin')
        inputs[asset] = expected
    inputs[name] = digest(root/name)


def load_training_ratios(root: Path, captured_inputs: dict[str, str]) -> dict[str, dict[str, float]]:
    spec = importlib.util.spec_from_file_location('bound_training_calibrator', root / TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = root / REPORT
    raw = path.read_bytes()
    report = json.loads(raw)
    if (report.get('schema_version') != 2 or report.get('status') != 'PASS'
            or report.get('purpose') != 'TRAINING_TOKENIZER_CALIBRATION'
            or report.get('includes_appended_eod') is not True
            or report.get('sampling') != SAMPLING
            or not isinstance(report.get('seed'), str) or not report['seed']
            or report.get('sampler_tool_sha256') != module.digest(root / SAMPLER)
            or report.get('tool_sha256') != module.digest(root / TOOL)
            or set(report.get('stages', {})) != {'warmup', 'stable', 'anneal'}):
        raise ValueError('invalid, biased-prefix, or stale training tokenizer calibration report')
    target = report.get('minimum_reference_tokens_per_domain')
    if type(target) is not int or target < module.MIN_REFERENCE_TOKENS:
        raise ValueError('training tokenizer calibration sample is too small')
    gate = json.loads((root / 'data/TOKENIZER_READY.json').read_text())
    if gate.get('status') != 'PASS' or report.get('tokenizer_sha256') != gate.get('tokenizer_sha256'):
        raise ValueError('training tokenizer calibration tokenizer differs from release')
    captured_inputs[REPORT] = module.hashlib.sha256(raw).hexdigest()
    captured_inputs[TOOL] = module.digest(root / TOOL)
    captured_inputs[SAMPLER] = module.digest(root / SAMPLER)
    captured_inputs['pretrain/scripts/training_tokenizer_calibration.py'] = module.digest(
        root / 'pretrain/scripts/training_tokenizer_calibration.py')
    captured_inputs['data/TOKENIZER_READY.json'] = module.digest(root / 'data/TOKENIZER_READY.json')
    for relative in ('pretrain/scripts/materialize_tokens.py', 'pretrain/scripts/source_document_binding.py',
                     'pretrain/scripts/merge_token_parts.c'):
        captured_inputs[relative] = module.digest(root / relative)
    parent_path = str(report['parent_release'])
    parent = json.loads((root / parent_path).read_text())
    if parent.get('status') != 'PASS' or module.digest(root / parent_path) != report['parent_release_sha256']:
        raise ValueError('training tokenizer calibration parent release changed')
    captured_inputs[parent_path] = report['parent_release_sha256']
    fingerprints = set()
    result = {}
    for stage, row in report['stages'].items():
        if row.get('status') != 'PASS' or set(row.get('domains', {})) != set(module.DOMAINS):
            raise ValueError('training tokenizer calibration must cover all nine domains in every phase')
        config = yaml.safe_load((root / parent['stages'][stage]['config']).read_text())
        if row.get('quality_policy') != module.quality_policy(config):
            raise ValueError(f'training tokenizer calibration quality policy changed: {stage}')
        evidence = row.get('evidence') or {}
        for relative, expected in evidence.items():
            if module.digest(root / relative) != expected:
                raise ValueError(f'training tokenizer calibration evidence changed: {relative}')
            captured_inputs[relative] = expected
        def one(suffix):
            matches = [root / name for name in evidence if Path(name).name == suffix]
            if len(matches) != 1:
                raise ValueError(f'calibration evidence must uniquely bind {suffix}')
            return matches[0]
        meta = json.loads(one('sample.meta.json').read_text())
        tokenizer = Path(meta['tokenizer'])
        if not tokenizer.is_absolute():
            tokenizer = root / tokenizer
        if module.digest(tokenizer) != report['tokenizer_sha256']:
            raise ValueError('training tokenizer calibration tokenizer bytes changed')
        captured_inputs[str(tokenizer)] = report['tokenizer_sha256']
        selection = json.loads(one('selection.json').read_text())
        parent_phase = parent['stages'][stage]
        population = module.selected_population(root, parent, parent_phase)
        manifest_name = parent_phase['manifest']
        captured_inputs[manifest_name] = module.digest(root/manifest_name)
        if (selection.get('sampling') != SAMPLING or selection.get('seed') != report['seed']
                or selection.get('phase') != stage
                or selection.get('sampler_tool_sha256') != report['sampler_tool_sha256']
                or selection.get('population_docs_by_domain') != {d: population[d]['selected_docs'] for d in module.DOMAINS}
                or selection.get('population_reference_tokens_by_domain') != {d: population[d]['selected_tokens'] for d in module.DOMAINS}):
            raise ValueError('training calibration lacks independent full-population sampling provenance')
        order, ledger = one('sample.parquet'), one('sample.doc_lengths.u32')
        parent_record = next(item for item in parent['artifacts']
                             if item['path'] == parent['stages'][stage]['order'])
        for producer, field in (('pretrain/scripts/materialize_tokens.py', 'materializer_tool_sha256'),
                                ('pretrain/scripts/merge_token_parts.c', 'native_merge_tool_sha256')):
            bind_materialization_producer(root, parent, producer, meta.get(field),
                                          captured_inputs, module.digest)
        if (not verified_source_document_binding(meta)
                or meta.get('tokenizer_sha256') != report['tokenizer_sha256']
                or meta.get('order_sha256') != module.digest(order)
                or selection.get('sample_sha256') != meta['order_sha256']
                or selection.get('parent_order_sha256') != parent_record['sha256']
                or selection.get('minimum_reference_tokens_per_domain') != target
                or meta['doc_lengths']['sha256'] != module.digest(ledger)
                or row.get('source_roster_fingerprint') != meta['source_document_binding']['roster_fingerprint']):
            raise ValueError(f'training tokenizer calibration stream provenance mismatch: {stage}')
        measured = module.measure(order, ledger, expected_order_sha256=meta['order_sha256'],
                                  expected_lengths_sha256=meta['doc_lengths']['sha256'])
        if (measured != row['domains'] or sum(v['actual_tokens_with_eod'] for v in measured.values()) != meta['total_tokens']
                or sum(v['docs'] for v in measured.values()) != selection.get('docs')):
            raise ValueError(f'training tokenizer calibration measured counts differ: {stage}')
        ratios = {}
        for domain, measurement in measured.items():
            ratio = measurement['ratio']
            if (measurement['reference_tokens'] < target or measurement['docs'] < 100
                    or not math.isfinite(ratio) or ratio <= 0):
                raise ValueError(f'invalid training tokenizer measurement: {stage}/{domain}')
            ratios[domain] = ratio
        result[stage] = ratios
        fingerprints.add(row['source_roster_fingerprint'])
    current_fingerprint, _ = module.materializer.source_roster_fingerprint(
        root / 'data/processed/s3', sorted((root / 'data/processed/s3').rglob('*.parquet')))
    if fingerprints != {current_fingerprint}:
        raise ValueError('training tokenizer calibration source roster changed')
    return result
