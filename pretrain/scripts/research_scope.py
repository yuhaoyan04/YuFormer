"""Declare active research families separately from experimental evidence."""
from __future__ import annotations

import json
import math
import hashlib
from pathlib import Path

POLICY = 'pretrain/configs/research_scope.json'
DEFERRED = {'e2', 'e3', 'e2_confirm', 'e3_confirm'}
DOMAINS = {'web', 'code', 'math', 'chinese', 'academic', 'books', 'encyclopedia', 'qa', 'multilingual'}


def load_scope(root: Path) -> dict:
    path = root / POLICY
    if not path.exists():
        return {'profile': 'full'}
    value = json.loads(path.read_text())
    if value.get('profile') not in {'full', 'scaling_only'}:
        raise ValueError('Unknown research scope')
    if value['profile'] == 'scaling_only':
        baseline = value.get('baseline', {})
        weights = baseline.get('mixture_weights', {})
        if (not baseline.get('architecture_preset') or set(weights) != DOMAINS
                or any(not math.isfinite(float(v)) or float(v) <= 0 for v in weights.values())
                or not math.isclose(sum(map(float, weights.values())), 1., abs_tol=1e-9)):
            raise ValueError('Scaling-only scope requires a valid fixed architecture and mixture')
    batch_policy = value.get('batch_selection_policy')
    if batch_policy is not None and (batch_policy.get('mode') != 'fixed_baseline'
            or batch_policy.get('batch') != '1m' or batch_policy.get('tokens_per_step') != 1048576):
        raise ValueError('Unsupported fixed batch baseline policy')
    return value


def scaling_only(root: Path) -> bool:
    return load_scope(root)['profile'] == 'scaling_only'


def group_enabled(root: Path, group: str) -> bool:
    return not (group in DEFERRED and scaling_only(root))


def ablation_requirements_satisfied(root: Path, *evidence: dict) -> bool:
    return scaling_only(root) or all(value.get('status') == 'PASS' for value in evidence)


def scope_binding_current(root: Path, frozen: dict) -> bool:
    path = root / POLICY
    if not path.is_file():
        return True
    return (frozen.get('inputs', {}).get(POLICY) == hashlib.sha256(path.read_bytes()).hexdigest()
            and frozen.get('research_scope', {}).get('profile') == load_scope(root)['profile'])


def scope_sha256(root: Path) -> str | None:
    path = root / POLICY
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
