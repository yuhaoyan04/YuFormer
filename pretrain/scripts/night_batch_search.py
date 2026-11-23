"""Materialize and bound adaptive LR extensions of the batch pilot grid."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import yaml
from batch_pilot_contract import CONTRACT, GROUP, LEVELS, MATCHED_TRAINED_TOKENS, extensions
from night_research import POLICY
from prepare_lr_extensions import write_once
from select_batch_pilots import atomic_json
from research_scope import load_scope


def extended_config(base: dict, row: dict) -> dict:
    cfg = copy.deepcopy(base)
    cfg['run_name'], cfg['optim']['lr'] = row['run_name'], row['learning_rate']
    if 'batch_pilot_resolution' in cfg['optim']:
        cfg['optim']['batch_pilot_resolution']['level_multiplier'] = row['learning_rate'] / base['optim']['lr']
    return cfg


def prepare(root: Path) -> list[str]:
    names = []
    for index, row in enumerate(extensions(root)):
        base_path = root / f"pretrain/configs/batch_pilot/batch_{row['batch']}_mid"
        base = yaml.safe_load(base_path.with_suffix('.yaml').read_text())
        if base['train']['budget_tokens'] != MATCHED_TRAINED_TOKENS:
            raise ValueError('batch extension must retain the matched ~2B budget')
        cfg = extended_config(base, row)
        name = row['run_name']; path = root / f'pretrain/configs/{GROUP}/{name}.yaml'
        values = dict(line.split('=', 1) for line in base_path.with_suffix('.env').read_text().splitlines()
                      if line and not line.startswith('#'))
        values.update(YUFORMER_CONFIG=f'configs/{GROUP}/{name}.yaml',
                      YUFORMER_PRETRAIN_OUT=str(root / f'data/experiments/{GROUP}/{name}'),
                      MASTER_PORT=str(29860 + index))
        write_once(path, yaml.safe_dump(cfg, sort_keys=False))
        write_once(path.with_suffix('.env'), ''.join(f'{k}={v}\n' for k, v in values.items()))
        names.append(name)
    return names


def plan(root: Path, selection: dict, audit: dict, policy: dict) -> dict:
    baseline = load_scope(root).get('batch_selection_policy')
    if baseline and not any(edge['batch'] == baseline['batch']
                            for edge in selection.get('boundary_selections', [])):
        return {'status': 'FIXED_BASELINE_POLICY', 'new_arms': [], 'limits': [], 'additional_tokens': 0}
    rules = policy.get('automatic_batch_extensions', {})
    if policy.get('enabled') is not True or rules.get('enabled') is not True:
        return {'status': 'DISABLED', 'new_arms': []}
    existing = extensions(root)
    limit = min(int(rules['max_new_arms']), int(rules['maximum_additional_tokens']) // MATCHED_TRAINED_TOKENS)
    proposed, blocked = [], []
    for edge in selection.get('boundary_selections', []):
        if baseline and edge['batch'] != baseline['batch']:
            continue
        batch = edge['batch']
        names = ({f'batch_{batch}_{level}' for level in LEVELS}
                 | {row['run_name'] for row in existing if row['batch'] == batch})
        curve = selection.get('curves', {}).get(batch, [])
        passed = {row['name'] for row in audit.get('rows', []) if row.get('status') == 'PASS' and row.get('name') in names}
        if (passed != names or {row['run_name'] for row in curve} != names or len(curve) != len(names)
                or selection.get('rejected') or any(row['batch'] == batch for row in selection.get('duplicates', []))):
            continue
        lr = float(edge['learning_rate']); rates = [float(row['learning_rate']) for row in curve]
        if lr not in {min(rates), max(rates)}:
            continue
        count = sum(row['batch'] == batch for row in existing)
        for factor in rules['upper_multipliers' if lr == max(rates) else 'lower_multipliers']:
            value = float(f'{lr * factor:.12g}')
            if any(math.isclose(value, rate, rel_tol=1e-10) for rate in rates):
                continue
            if (len(existing) + len(proposed) >= limit or count >= int(rules['max_extensions_per_batch'])
                    or not float(rules['minimum_learning_rate']) <= value <= float(rules['maximum_learning_rate'])):
                blocked.append({'batch': batch, 'learning_rate': value, 'reason': 'automatic_batch_budget_or_lr_limit'})
                continue
            level = f'ext{round(value * 1_000_000_000)}'
            proposed.append({'batch': batch, 'level': level, 'learning_rate': value,
                             'run_name': f'batch_{batch}_{level}', 'group': GROUP,
                             'origin': 'automatic_night_search'})
            count += 1
    return {'status': 'EXPAND' if proposed else 'LIMIT_REACHED' if blocked else 'WAIT_OR_BRACKETED',
            'new_arms': proposed, 'limits': blocked,
            'additional_tokens': len(proposed) * MATCHED_TRAINED_TOKENS}


def expand(root: Path, selection: dict, audit: dict, runner_live: bool) -> dict:
    policy = json.loads((root / POLICY).read_text())
    result = plan(root, selection, audit, policy)
    if runner_live and result['new_arms']:
        return {**result, 'status': 'WAIT_FOR_EXTENSION_QUEUE_COMPLETION'}
    if result['new_arms']:
        manifest = json.loads((root / CONTRACT).read_text())
        manifest['extensions'].extend(result['new_arms'])
        atomic_json(root / CONTRACT, manifest)
        prepare(root)
    return result
