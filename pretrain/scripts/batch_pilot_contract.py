"""Exact roster and search identity for matched-budget batch pilots."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from research_scope import load_scope, scope_sha256

BATCHES = ('1m', '2m', '4m')
LEVELS = ('low', 'mid', 'high')
GROUP = 'batch_pilot_extension'
CONTRACT = 'pretrain/configs/batch_pilot_search.json'
MATCHED_TRAINED_TOKENS = 2_000_000_000 // 4194304 * 4194304


def extensions(root: Path) -> list[dict]:
    path = root / CONTRACT
    if not path.is_file():
        return []
    doc = json.loads(path.read_text())
    if doc.get('schema_version') != 1 or not isinstance(doc.get('extensions'), list):
        raise ValueError('invalid batch search contract')
    names, rates = set(), set()
    for row in doc['extensions']:
        batch, level, lr, name = (row.get(k) for k in ['batch', 'level', 'learning_rate', 'run_name'])
        if (batch not in BATCHES or not isinstance(level, str) or not re.fullmatch(r'ext[0-9]+', level)
                or name != f'batch_{batch}_{level}' or row.get('group') != GROUP
                or type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0
                or name in names or (batch, lr) in rates):
            raise ValueError('invalid or duplicate batch extension')
        names.add(name); rates.add((batch, lr))
    return doc['extensions']


def expected_names(root: Path) -> set[str]:
    return ({f'batch_{batch}_{level}' for batch in BATCHES for level in LEVELS}
            | {row['run_name'] for row in extensions(root)})


def contract_sha256(root: Path) -> str | None:
    path = root / CONTRACT
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def selection_is_current(root: Path, value: dict) -> bool:
    expected = expected_names(root)
    policy = load_scope(root).get('batch_selection_policy')
    if (value.get('status') not in {'PASS', 'PASS_PROVISIONAL'}
            or value.get('expected_results') != len(expected)
            or value.get('observed_results') != len(expected)
            or not isinstance(value.get('selection'), dict)
            or value.get('research_scope_sha256') != scope_sha256(root)
            or value.get('selection_policy') != policy
            or any(value.get(key) for key in ('missing','duplicates','rejected','invalid'))):
        return False
    boundaries = value.get('boundary_selections') or []
    if policy:
        selected = value['selection']
        curve = value.get('curves', {}).get(policy['batch'], [])
        if (selected.get('batch') != policy['batch']
                or selected.get('tokens_per_step') != policy['tokens_per_step']
                or any(edge.get('batch') == policy['batch'] for edge in boundaries)
                or selected not in curve or any(row['validation_loss'] < selected['validation_loss'] for row in curve)
                or not any(row['learning_rate'] < selected['learning_rate'] for row in curve)
                or not any(row['learning_rate'] > selected['learning_rate'] for row in curve)):
            return False
    elif boundaries:
        return False
    if (root / CONTRACT).is_file():
        return (value.get('search_contract_sha256') == contract_sha256(root)
                and set(value.get('expected_run_names', [])) == expected)
    return True
