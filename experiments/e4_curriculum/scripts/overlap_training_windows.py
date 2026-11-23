"""Resolve the actual production consumption intervals for the final overlap audit."""
import hashlib
import json
from pathlib import Path

import yaml

import block_permutation_reader as reader
from training_data_origin import read_bound
from training_source_windows import require_window_admission

PHASES = ('warmup', 'stable', 'anneal')


def consumption_cache(cache, phase, identity, source_sample_offset):
    if phase not in PHASES or type(source_sample_offset) is not int or source_sample_offset < 0:
        raise ValueError('overlap consumption cache requires an explicit valid source window')
    if source_sample_offset:
        return (Path(cache)/f'{phase}.window-{source_sample_offset}.consumed.records',
                f'{identity}:source_sample_offset={source_sample_offset}')
    return Path(cache)/f'{phase}.consumed.records', identity


def resolve_windows(root, config, release, release_sha256):
    root = Path(root).resolve()
    if config['data'].get('sampling_policy', 'sequential') == 'sequential':
        if config['data'].get('block_permutation_manifest'):
            raise ValueError('sequential overlap cannot hide a block-permutation manifest')
        return {'source_sample_offsets':[0]*3, 'windows':{}, 'inputs':{}}
    if config['data'].get('sampling_policy') != 'block_permutation':
        raise ValueError('overlap audit cannot infer an unsupported training consumption policy')
    env_name = 'pretrain/configs/pretrain_base.env'
    env_raw, env_sha = read_bound(root/env_name)
    variables = dict(line.split('=',1) for line in env_raw.decode().splitlines()
                     if line.strip() and not line.lstrip().startswith('#'))
    rendered = yaml.safe_dump(config)
    for _ in range(len(variables)+1):
        before = rendered
        for key, value in variables.items():
            rendered = rendered.replace('${'+key+'}',value)
        if rendered == before:
            break
    if '${' in rendered:
        raise ValueError('unresolved final-overlap production environment')
    actual_config = yaml.safe_load(rendered)
    order = reader.configuration_order_identity(actual_config, require_production=True)
    manifest_path = Path(actual_config['data']['block_permutation_manifest'])
    if not manifest_path.is_absolute():
        manifest_path = root/manifest_path
    master_raw, master_sha = read_bound(manifest_path, order['manifest_sha256'])
    master = json.loads(master_raw)
    if (master.get('schema_version') != 3 or master.get('scope') != 'PRODUCTION'
            or master.get('release_manifest_sha256') != release_sha256
            or master.get('data_frozen_recipe_sha256') != release['frozen_recipe_sha256']
            or len(master.get('stages', [])) != 3):
        raise ValueError('final overlap requires the actual three admitted production source windows')
    inputs = {env_name:env_sha, str(manifest_path.relative_to(root)):master_sha}
    for tool in (Path(__file__), Path(reader.__file__)):
        inputs[str(tool.relative_to(root))] = hashlib.sha256(tool.read_bytes()).hexdigest()
    windows, offsets = {}, []
    for index, phase in enumerate(PHASES):
        row = master['stages'][index]
        phase_path = root/f'data/health/pretrain-1t-{phase}-mixture-preflight.json'
        raw, sha = read_bound(phase_path)
        phase_report = json.loads(raw)
        window = require_window_admission(root, phase_report, phase=phase,
            stage_samples=config['data']['stage_sample_limits'][index], seq_len=config['data']['seq_len'],
            release_sha256=release_sha256, stream_meta_sha256=row['stream_meta_sha256'],
            data_frozen_recipe_sha256=release['frozen_recipe_sha256'])
        if (window['source_sample_offset'] != row['source_sample_offset']
                or window['block_samples'] != master['block_samples']
                or row['source_window_selection'] != phase_report['source_window']['selection_report']
                or order['source_sample_offsets'][index] != row['source_sample_offset']):
            raise ValueError('final-overlap interval differs from the runtime-admitted production order')
        offsets.append(row['source_sample_offset'])
        windows[phase] = {'binding':phase_report['source_window'],
                          'domain_tokens':phase_report['consumed']['domain_tokens']}
        inputs.update(window['inputs'])
        inputs[str(phase_path.relative_to(root))] = sha
    for name, sha in inputs.items():
        read_bound(root/name, sha)
    return {'source_sample_offsets':offsets, 'windows':windows, 'inputs':inputs}
