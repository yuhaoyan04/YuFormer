#!/usr/bin/env python3
"""Publish prepared indices only after complete physical and consumption audits."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import yaml
import numpy as np

import calibrate_training_tokens as calibration
import training_block_permutation as blocks
import block_permutation_reader as reader
import training_source_windows as windows
import frozen_training_evidence as training_evidence

ROOT = Path(__file__).resolve().parents[3]
STAGES = ('warmup', 'stable', 'anneal')


def shift_prepared_permutation(directory, prepared, *, source_sample_offset, block_samples):
    artifact = prepared['permutation']
    name = artifact.get('file')
    if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('invalid prepared block artifact path')
    samples = prepared['permutation_population_samples']
    if (type(source_sample_offset) is not int or source_sample_offset < 0
            or source_sample_offset % block_samples or samples % block_samples):
        raise ValueError('source window shift must preserve whole causal blocks')
    path = directory/name
    with path.open('rb') as handle:
        before = calibration.materializer._stat_identity(os.fstat(handle.fileno()))
        raw = handle.read()
        if (len(raw) != samples//block_samples*4 or artifact.get('bytes') != len(raw)
                or artifact.get('dtype') != '<u4' or hashlib.sha256(raw).hexdigest() != artifact.get('sha256')
                or calibration.materializer._stat_identity(os.fstat(handle.fileno())) != before
                or calibration.materializer.source_file_identity(path) != before):
            raise ValueError('prepared block artifact differs from its bound base permutation')
    values = np.frombuffer(raw, dtype='<u4')
    if not np.array_equal(np.sort(values), np.arange(len(values), dtype='<u4')):
        raise ValueError('prepared base permutation is not a complete logical window bijection')
    if source_sample_offset == 0:
        return dict(artifact)
    shifted = values.astype(np.uint64)+source_sample_offset//block_samples
    if int(shifted.max()) >= 2**32:
        raise ValueError('source window exceeds uint32 physical block representation')
    content = shifted.astype('<u4').tobytes()
    destination = directory/f"{prepared['phase']}.window-{source_sample_offset}.blocks.u32"
    if destination.exists():
        if destination.read_bytes() != content:
            raise ValueError('refusing to overwrite a different immutable source-window permutation')
    else:
        temporary = destination.with_name(destination.name+f'.tmp-{os.getpid()}')
        with temporary.open('wb') as handle:
            handle.write(content); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, destination)
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return {'file': destination.name, 'bytes': len(content), 'dtype': '<u4',
            'sha256': hashlib.sha256(content).hexdigest()}


def read_json(path):
    with path.open('rb') as handle:
        before = calibration.materializer._stat_identity(os.fstat(handle.fileno()))
        raw = handle.read()
        if (calibration.materializer._stat_identity(os.fstat(handle.fileno())) != before or
                calibration.materializer.source_file_identity(path) != before):
            raise ValueError('block publication input changed: '+str(path))
    return json.loads(raw), raw, hashlib.sha256(raw).hexdigest()


def require_admission(audit, mixtures, limits, seq_len, target_weights):
    if (audit.get('status') != 'PASS' or audit.get('roster_exact') is not True
            or audit.get('train_doc_lengths_audited') is not True
            or audit.get('train_source_rows_verified') is not True):
        raise ValueError('production block publication requires complete full-byte/source/ledger audits')
    rows = audit.get('streams') or []
    expected = set(STAGES) | {'val_'+d for d in calibration.DOMAINS}
    if {r.get('name') for r in rows} != expected or len(rows) != len(expected) or any(r.get('status') != 'PASS' for r in rows):
        raise ValueError('production block publication audit stream roster is not exact')
    for index, phase in enumerate(STAGES):
        value = mixtures.get(phase) or {}
        mixture = value.get('mixture') or {}
        domains = mixture.get('domains') or {}
        required = limits[index]*seq_len
        if (value.get('status') != 'PASS' or mixture.get('status') != 'PASS'
                or value.get('stage') != phase or value.get('configured_target_tokens') != required
                or mixture.get('consumed_tokens') != required or set(domains) != set(calibration.DOMAINS)
                or mixture.get('absolute_tolerance') != .002 or mixture.get('relative_tolerance') != .1
                or any(r.get('pass') is not True for r in domains.values())
                or sum(r.get('tokens', -1) for r in domains.values()) != required):
            raise ValueError('production block publication requires exact configured consumption mixture: '+phase)
        for domain in calibration.DOMAINS:
            target = target_weights[phase][domain]
            count = domains[domain].get('tokens')
            if (type(count) is not int or count < 0 or not math.isfinite(target) or target <= 0
                    or not math.isclose(domains[domain].get('target_weight', -1), target, abs_tol=1e-12)
                    or abs(count/required-target) > .002+1e-12
                    or abs(count/required-target)/target > .1+1e-12):
                raise ValueError('configured mixture does not meet the original frozen domain gate: '+phase+'/'+domain)


def publish(root, directory):
    plan_path = directory/'mapping_plan.json'
    plan, plan_raw, plan_sha = read_json(plan_path)
    geometry = plan['geometry']
    release_path = root/'data/processed/releases/pretrain-1t-frozen.release.json'
    release, release_raw, release_sha = read_json(release_path)
    base = yaml.safe_load((root/'pretrain/configs/pretrain_base.yaml').read_bytes())
    if (plan.get('status') != 'PREPARED_INDEX_ARRAYS_AWAITING_FULL_STREAM_GATES'
            or release.get('status') != 'PASS' or release_sha != geometry['release_manifest_sha256']
            or base['data']['seq_len'] != geometry['seq_len']
            or base['data']['stage_sample_limits'] != geometry['stage_sample_limits']
            or calibration.digest(blocks.__file__) != geometry['generator_tool_sha256']
            or calibration.digest(Path(__file__).with_name('prepare_training_blocks.py')) != geometry['preparation_tool_sha256']):
        raise ValueError('prepared mappings no longer match the exact data/geometry/generator contract')
    audit_path = root/'data/processed/audits/pretrain_1t_token_streams.json'
    audit, audit_raw, audit_sha = read_json(audit_path)
    if (audit.get('dataset_id') != release['dataset_id'] or audit.get('snapshot_id') != release['snapshot_id']
            or audit.get('audit_tool_sha256') != calibration.digest(root/'experiments/e4_curriculum/scripts/audit_1t_token_streams.py')
            or audit.get('shared_auditor_sha256') != calibration.digest(root/'pretrain/scripts/audit_token_stream_release.py')
            or audit.get('source_binding_tool_sha256') != calibration.digest(root/'pretrain/scripts/source_document_binding.py')):
        raise ValueError('block publication requires current data/auditor/source-binding provenance')
    evidence = [(plan_path, plan_raw, plan_sha), (release_path, release_raw, release_sha),
                (audit_path, audit_raw, audit_sha)]
    frozen_sha = release['frozen_recipe_sha256']
    if not isinstance(frozen_sha, str) or len(frozen_sha) != 64 or any(c not in '0123456789abcdef' for c in frozen_sha):
        raise ValueError('selected data release has an invalid original recipe identity')
    frozen_path = root/'data/processed/recipe_origins'/frozen_sha/'objects'/frozen_sha
    frozen, frozen_raw, actual_frozen_sha = read_json(frozen_path)
    if actual_frozen_sha != frozen_sha or frozen.get('status') != 'PASS':
        raise ValueError('original frozen data recipe archive is incomplete/corrupt')
    evidence.append((frozen_path, frozen_raw, frozen_sha))
    mixtures = {}
    for phase in STAGES:
        path = root/f'data/health/pretrain-1t-{phase}-mixture-preflight.json'
        mixtures[phase], raw, digest = read_json(path)
        evidence.append((path, raw, digest))
    require_admission(audit, mixtures, geometry['stage_sample_limits'], geometry['seq_len'],
                      {phase: frozen['stages'][phase]['target_final_domain_weights'] for phase in STAGES})
    artifacts = {row['path']: row for row in release['artifacts']}
    audited = {row['name']: row for row in audit['streams']}
    rows, capacities, metadata_hashes, fingerprints = [], [], [], set()
    for index, phase in enumerate(STAGES):
        prepared = plan['stages'][index]
        if prepared['phase'] != phase or prepared['stage_index'] != index:
            raise ValueError('prepared mapping phase order changed')
        meta_path = root/f'data/token_streams/pretrain-1t/{phase}.meta.json'
        meta, raw, digest = read_json(meta_path)
        source_order = release['stages'][phase]['order']
        proof = audited[phase]
        binary = meta_path.parent/f'{phase}.tokens.bin'
        stat = calibration.materializer.source_file_identity(binary)
        ledger_stat = calibration.materializer.source_file_identity(meta_path.parent/meta['doc_lengths']['file'])
        if (not calibration.verified_source_document_binding(meta)
                or prepared['source_order'] != source_order
                or meta.get('order_sha256') != artifacts[source_order]['sha256']
                or meta.get('data_snapshot_id') != release['snapshot_id']
                or proof.get('actual_sha256') != meta.get('sha256')
                or proof.get('actual_total_tokens_with_eod') != meta.get('total_tokens')
                or proof.get('actual_doc_lengths_sha256') != meta['doc_lengths']['sha256']
                or ledger_stat != {'bytes': proof.get('actual_doc_lengths_bytes'),
                    'device': proof.get('actual_doc_lengths_device'), 'inode': proof.get('actual_doc_lengths_inode'),
                    'mtime_ns': proof.get('actual_doc_lengths_mtime_ns')}
                or stat != meta.get('stream_stat_identity')
                or stat != {'bytes': proof.get('actual_bytes'), 'device': proof.get('actual_device'),
                            'inode': proof.get('actual_inode'), 'mtime_ns': proof.get('actual_mtime_ns')}
                or mixtures[phase].get('evidence_sha256', {}).get(str(meta_path.relative_to(root))) != digest):
            raise ValueError('current stream differs from admitted physical/consumed data: '+phase)
        fingerprints.add(meta['source_document_binding']['roster_fingerprint'])
        window = windows.require_window_admission(root, mixtures[phase], phase=phase,
            stage_samples=prepared['stage_sample_limit'], seq_len=geometry['seq_len'],
            release_sha256=release_sha, stream_meta_sha256=digest,
            data_frozen_recipe_sha256=release['frozen_recipe_sha256'])
        if window['block_samples'] != geometry['block_samples']:
            raise ValueError('source window admission uses a different causal block size')
        for window_name, expected_sha in window['inputs'].items():
            if Path(window_name).suffix == '.json':
                window_path = root/window_name
                _, window_raw, window_sha = read_json(window_path)
                if window_sha != expected_sha:
                    raise ValueError('source window evidence changed before block publication')
                evidence.append((window_path, window_raw, window_sha))
        shifted = shift_prepared_permutation(directory, prepared,
            source_sample_offset=window['source_sample_offset'], block_samples=geometry['block_samples'])
        evidence.append((meta_path, raw, digest))
        capacities.append((meta['total_tokens']-1)//geometry['seq_len'])
        metadata_hashes.append(digest)
        rows.append({'stage_index': index, 'stage_sample_limit': prepared['stage_sample_limit'],
            'permutation_population_samples': prepared['permutation_population_samples'],
            'stream_meta_sha256': digest, 'permutation': shifted,
            'base_permutation_sha256': prepared['permutation']['sha256'],
            'source_sample_offset': window['source_sample_offset'],
            'source_window_selection': mixtures[phase]['source_window']['selection_report']})
    if len(fingerprints) != 1:
        raise ValueError('production streams have different source roster generations')
    saved = []
    folder = directory/'evidence'
    folder.mkdir(exist_ok=True)
    for path, raw, digest in evidence:
        output = folder/(digest+'.json')
        if output.exists() and output.read_bytes() != raw:
            raise ValueError('immutable block admission evidence is corrupt')
        training_evidence.publish_checked(output, raw, immutable=True)
        saved.append({'source': str(path.relative_to(root)), 'file': 'evidence/'+output.name, 'sha256': digest})
    manifest = {'schema_version': 3, 'scope': 'PRODUCTION', 'status': 'PASS',
        'purpose': 'TRAINING_CAUSAL_BLOCK_PERMUTATION', 'algorithm': geometry['algorithm'],
        'seed': geometry['seed'], 'seq_len': geometry['seq_len'], 'block_samples': geometry['block_samples'],
        'dataset_id': release['dataset_id'], 'snapshot_id': release['snapshot_id'],
        'release_manifest_sha256': release_sha, 'data_frozen_recipe_sha256': release['frozen_recipe_sha256'],
        'source_roster_fingerprint': next(iter(fingerprints)), 'stages': rows,
        'source_window_policy': windows.POLICY,
        'window_selection_tool_sha256': calibration.digest(windows.__file__),
        'evidence_publication_tool_sha256': calibration.digest(training_evidence.__file__),
        'admission_evidence': saved, 'prepared_mapping_plan_sha256': plan_sha,
        'preparation_tool_sha256': geometry['preparation_tool_sha256'],
        'generator_tool_sha256': geometry['generator_tool_sha256'],
        'publication_tool_sha256': calibration.digest(__file__)}
    path = directory/'manifest.json'
    content = (json.dumps(manifest, sort_keys=True, indent=2)+'\n').encode()
    temporary = path.with_name(path.name+f'.tmp-{os.getpid()}')
    with temporary.open('wb') as handle:
        handle.write(content); handle.flush(); os.fsync(handle.fileno())
    try:
        reader.load_permutations(temporary, seq_len=geometry['seq_len'],
            stage_samples=geometry['stage_sample_limits'], stage_capacities=capacities,
            stream_meta_sha256=metadata_hashes, release_sha256=release_sha,
            dataset_id=release['dataset_id'], snapshot_id=release['snapshot_id'])
        if path.exists():
            if path.read_bytes() != content:
                raise ValueError('refusing to replace a different immutable production block manifest')
            temporary.unlink()
        else:
            os.replace(temporary, path)
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    value = publish(args.root.resolve(), args.directory.resolve())
    print(json.dumps({'status': value['status'], 'scope': value['scope'], 'manifest': str(args.directory/'manifest.json')}))
