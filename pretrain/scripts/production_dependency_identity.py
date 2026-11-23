"""Resolve the actual production model and data-order dependencies for hashing."""
from __future__ import annotations

import json
from pathlib import Path

import yaml


def production_asset_paths(root):
    root = Path(root).resolve()
    base, environment = root/'pretrain/configs/pretrain_base.yaml', root/'pretrain/configs/pretrain_base.env'
    if not base.is_file() or not environment.is_file():
        return set()
    variables = {}
    for line in environment.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            key, value = line.split('=', 1)
            variables[key] = value

    def expand(value):
        for key, replacement in variables.items():
            value = value.replace('${'+key+'}', replacement)
        if '${' in value:
            raise ValueError('unresolved production dependency: '+value)
        path = Path(value)
        return (path if path.is_absolute() else root/path).resolve()

    config = yaml.safe_load(base.read_text())
    paths = {base.resolve(), environment.resolve(), expand(config['model']['configs'])}
    data = config.get('data') or {}
    for key in ('release_manifest', 'tokenizer_gate', 'token_stream_audit'):
        if data.get(key):
            paths.add(expand(data[key]))
    for name in data.get('meta', []):
        paths.add(expand(name))
    if data.get('sampling_policy', 'sequential') == 'block_permutation':
        if not data.get('block_permutation_manifest'):
            raise ValueError('production block order must bind its manifest')
        manifest_path = expand(data['block_permutation_manifest'])
        paths.add(manifest_path)
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_bytes())
            parent = manifest.get('parent_manifest')
            if parent:
                name = parent.get('file')
                if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
                    raise ValueError('invalid parent order manifest dependency')
                paths.add((manifest_path.parent/name).resolve())
            for row in manifest.get('stages', []):
                name = (row.get('permutation') or {}).get('file')
                if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
                    raise ValueError('invalid production block-order artifact dependency')
                paths.add((manifest_path.parent/name).resolve())
    return paths
