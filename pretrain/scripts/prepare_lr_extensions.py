"""Materialize matched-budget LR extensions without changing any existing run."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import yaml
from lr_pilot_contract import extensions, EXTENSION_GROUP


def write_once(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"refusing to change an existing supplemental contract: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'.{path.name}.{os.getpid()}')
    tmp.write_text(content)
    os.replace(tmp, path)


def prepare(root: Path) -> list[str]:
    names = []
    for index, row in enumerate(extensions(root)):
        name = row['run_name']
        base = root / f"pretrain/configs/e1_pilot/e1_lr_{row['size']}_vhigh"
        cfg = copy.deepcopy(yaml.safe_load(base.with_suffix('.yaml').read_text()))
        cfg['run_name'] = name
        cfg['optim']['lr'] = row['learning_rate']
        directory = root / 'pretrain/configs' / EXTENSION_GROUP
        env = dict(line.split('=', 1) for line in base.with_suffix('.env').read_text().splitlines()
                   if line and not line.startswith('#'))
        env.update(YUFORMER_CONFIG=f'configs/{EXTENSION_GROUP}/{name}.yaml',
                   YUFORMER_PRETRAIN_OUT=str(root / 'data/experiments' / EXTENSION_GROUP / name),
                   MASTER_PORT=str(29740 + index))
        write_once(directory / f'{name}.yaml', yaml.safe_dump(cfg, sort_keys=False))
        write_once(directory / f'{name}.env', ''.join(f'{key}={value}\n' for key, value in env.items()))
        names.append(name)
    return names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps({'prepared': prepare(args.root.resolve())}))
