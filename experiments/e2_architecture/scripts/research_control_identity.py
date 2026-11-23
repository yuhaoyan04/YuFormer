"""Bind matched 100B research controls to their actual inputs and training code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'pretrain/scripts'))
from qualification_identity import qualification_identity
from production_dependency_identity import production_asset_paths


def research_control_identity(root, config_directory):
    root, config_directory = Path(root).resolve(), Path(config_directory).resolve()
    if not config_directory.is_relative_to(root/'experiments/e2_architecture/configs'):
        raise ValueError('research contracts must be repository experiment configurations')
    retained = {root/'pretrain/configs/pretrain_base.yaml', root/'pretrain/configs/pretrain_base.env',
                root/'model/configs/pretrain_final_model_configs.yaml'}
    production = production_asset_paths(root)-retained
    rows = {}
    for row in qualification_identity(root)['files']:
        path = (root/row['path']).resolve()
        if path not in production:
            rows[row['path']] = row

    def bind(path):
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError('actual research dependency is missing: '+str(path))
        label = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        raw = path.read_bytes()
        rows[label] = dict(path=label, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

    bind(__file__)
    for env_path in sorted(config_directory.glob('*.env')):
        bind(env_path)
        variables = dict(line.split('=', 1) for line in env_path.read_text().splitlines()
                         if line.strip() and not line.lstrip().startswith('#'))

        def expand(value):
            for key, replacement in variables.items():
                value = value.replace('${'+key+'}', replacement)
            if '${' in value:
                raise ValueError('unresolved actual research dependency: '+value)
            path = Path(value)
            return (path if path.is_absolute() else root/path).resolve()

        config_path = expand(variables['YUFORMER_CONFIG'])
        bind(config_path)
        config = yaml.safe_load(config_path.read_text())
        data = config['data']
        if data.get('sampling_policy', 'sequential') != 'sequential' or data.get('block_permutation_manifest'):
            raise ValueError('research identity only covers the declared flat 100B control corpus')
        bind(expand(config['model']['configs']))
        for key in ('release_manifest', 'token_stream_audit', 'tokenizer_gate'):
            bind(expand(data[key]))
        gate = json.loads(expand(data['tokenizer_gate']).read_bytes())
        bind(expand(gate['tokenizer_path']))
        for name in data['meta']:
            path = expand(name)
            if path.is_relative_to(root/'data/token_streams/pretrain-1t'):
                raise ValueError('a production stream cannot enter a research-scoped measurement')
            bind(path)
        validation = expand(data['val_dir'])
        for domain in ('web','code','math','chinese','academic','books','encyclopedia','qa','multilingual'):
            bind(validation/f'val_{domain}.meta.json')
    if not list(config_directory.glob('*.env')):
        raise ValueError('research measurement needs actual execution contracts')
    files = [rows[name] for name in sorted(rows)]
    return dict(schema_version=1, purpose='MATCHED_100B_RESEARCH_ONLY_NOT_PRODUCTION_QUALIFICATION',
                files=files, sha256=hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
