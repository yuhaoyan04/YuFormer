"""Bounded LR search expansion and an independently qualified batch anchor."""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from lr_pilot_contract import CONTRACT, EXTENSION_GROUP, LEVELS, SIZES, extensions

POLICY = 'pretrain/configs/night_research_policy.json'
ANCHOR = 'experiments/e1_scaling/results/lr_200m_batch_anchor.json'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def anchor_names(root: Path, size: str = '200m') -> set[str]:
    return ({f'e1_lr_{size}_{level}' for level in LEVELS}
            | {row['run_name'] for row in extensions(root) if row['size'] == size})


def anchor_contract_sha(root: Path, size: str = '200m') -> str:
    rows = [row for row in extensions(root) if row['size'] == size]
    value = {'base_names': sorted(f'e1_lr_{size}_{level}' for level in LEVELS), 'extensions': rows}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build_batch_anchor(root: Path, selection: dict, audit: dict) -> dict:
    names = anchor_names(root)
    blocked = {'schema_version': 1, 'scope': '200m_anchor_for_batch_only', 'status': 'BLOCKED'}
    curves = selection.get('curves', {}).get('200m', [])
    rows = [row for row in audit.get('rows', []) if str(row.get('name', '')).startswith('e1_lr_200m_')]
    winner = selection.get('selections', {}).get('200m') or {}
    if ({row.get('run_name') for row in curves} != names or len(curves) != len(names)
            or {row.get('name') for row in rows} != names or len(rows) != len(names)
            or any(row.get('status') != 'PASS' for row in rows)
            or any(row.get('size') == '200m' for row in selection.get('duplicates', []))
            or selection.get('rejected')):
        return {**blocked, 'reason': 'complete_current_200m_results_and_identity_audit_required'}
    lr = winner.get('learning_rate')
    rates = [row['learning_rate'] for row in curves]
    selected_curve = next((row for row in curves if row['run_name'] == winner.get('run_name')), {})
    if (winner.get('run_name') not in names or not isinstance(lr, (int, float))
            or not any(rate < lr for rate in rates) or not any(rate > lr for rate in rates)
            or selected_curve.get('learning_rate') != lr
            or selected_curve.get('validation_loss') != winner.get('validation_loss')
            or winner.get('validation_loss') != min(row['validation_loss'] for row in curves)):
        return {**blocked, 'reason': '200m_optimum_is_not_bracketed'}
    files: set[Path] = set()
    for row in rows:
        env_path, config_path = Path(row['env']), Path(row['config'])
        files.update({env_path if env_path.is_absolute() else root / env_path,
                      config_path if config_path.is_absolute() else root / config_path})
        run = Path(row['run_dir'])
        if not run.is_absolute():
            run = root / run
        files.update(run / name for name in ('run_identity.json', 'current_attempt.json',
                                             'training_complete.json', 'result.json'))
        for path in (env_path if env_path.is_absolute() else root / env_path,
                     config_path if config_path.is_absolute() else root / config_path,
                     *(run / name for name in ('run_identity.json', 'current_attempt.json',
                                               'training_complete.json', 'result.json'))):
            key = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
            if audit.get('evidence_snapshot', {}).get(key) != sha256(path):
                return {**blocked, 'reason': 'anchor_evidence_changed_since_identity_audit'}
        identity = json.loads((run / 'run_identity.json').read_text())
        for key in ('model_configs', 'release_manifest', 'tokenizer_gate', 'token_stream_audit', 'tokenizer_path'):
            if identity.get(key):
                p = Path(identity[key]); files.add(p if p.is_absolute() else root / p)
        for stream in [*identity.get('streams', []), *identity.get('validation_streams', [])]:
            p = Path(stream['meta']); p = p if p.is_absolute() else root / p
            if sha256(p) != stream['meta_sha256']:
                return {**blocked, 'reason': 'anchor_stream_meta_changed_since_training'}
            files.add(p)
    snapshot = {}
    for path in sorted(files):
        if not path.is_file():
            return {**blocked, 'reason': f'anchor_evidence_missing:{path}'}
        key = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        snapshot[key] = sha256(path)
    return {'schema_version': 1, 'scope': '200m_anchor_for_batch_only', 'status': 'PASS',
            'observed_at': datetime.now(timezone.utc).isoformat(),
            'anchor_tool_sha256': sha256(Path(__file__)),
            'anchor_contract_sha256': anchor_contract_sha(root),
            'training_source_sha256': audit.get('source_sha256'),
            'expected_results': len(names), 'observed_results': len(names),
            'expected_run_names': sorted(names), 'boundary_selections': [],
            'selections': {'200m': winner}, 'curves': {'200m': curves},
            'evidence_snapshot': snapshot,
            'scope_note': 'Allows batch/LR calibration only; global LR resolution and E1/E2 require all anchors.'}


def batch_anchor_is_current(root: Path, report: dict) -> bool:
    from audit_experiment_identities import source_sha256
    names = anchor_names(root)
    if (report.get('scope') != '200m_anchor_for_batch_only' or report.get('status') != 'PASS'
            or report.get('expected_results') != len(names) or report.get('observed_results') != len(names)
            or set(report.get('expected_run_names', [])) != names
            or report.get('anchor_contract_sha256') != anchor_contract_sha(root)
            or report.get('anchor_tool_sha256') != sha256(Path(__file__))
            or report.get('training_source_sha256') != source_sha256(root)
            or not report.get('evidence_snapshot')):
        return False
    try:
        return all((root / name).is_file() and sha256(root / name) == digest
                   for name, digest in report['evidence_snapshot'].items())
    except (OSError, ValueError, TypeError):
        return False


def plan_lr_extensions(root: Path, selection: dict, policy: dict) -> dict:
    rules = policy.get('automatic_lr_extensions', {})
    if policy.get('enabled') is not True or rules.get('enabled') is not True:
        return {'status': 'DISABLED', 'new_arms': []}
    existing = extensions(root)
    automatic = sum(row.get('origin') == 'automatic_night_search' for row in existing)
    limit = min(int(rules['max_new_arms']), int(rules['maximum_additional_tokens']) // int(rules['budget_tokens_per_arm']))
    if int(rules['budget_tokens_per_arm']) != 500_000_000:
        raise ValueError('automatic LR extensions must retain the matched 500M budget')
    planned, blocked = [], []
    for boundary in selection.get('boundary_selections', []):
        size = boundary.get('size')
        if size not in SIZES:
            continue
        names = anchor_names(root, size)
        curve = selection.get('curves', {}).get(size, [])
        if {row['run_name'] for row in curve} != names or len(curve) != len(names):
            continue
        if any(row.get('size') == size for row in selection.get('duplicates', [])) or selection.get('rejected'):
            continue
        winner = (selection.get('selections', {}).get(size) or {})
        lr = float(winner['learning_rate'])
        rates = [float(row['learning_rate']) for row in curve]
        upper = lr == max(rates)
        if not upper and lr != min(rates):
            continue
        per_size = sum(row['size'] == size for row in existing)
        multipliers = rules['upper_multipliers' if upper else 'lower_multipliers']
        for multiplier in multipliers:
            value = float(f'{lr * multiplier:.12g}')
            if any(math.isclose(value, rate, rel_tol=1e-10) for rate in rates):
                continue
            if (automatic + len(planned) >= limit or per_size >= int(rules['max_extensions_per_size'])
                    or not float(rules['minimum_learning_rate']) <= value <= float(rules['maximum_learning_rate'])):
                blocked.append({'size': size, 'learning_rate': value, 'reason': 'automatic_search_budget_or_lr_limit'})
                continue
            # Numeric labels identify contracts; the LR itself is always explicit.
            level = f'ext{round(value * 1_000_000_000)}'
            name = f'e1_lr_{size}_{level}'
            if any(row['run_name'] == name for row in existing + planned):
                continue
            planned.append({'size': size, 'level': level, 'learning_rate': value,
                            'run_name': name, 'group': EXTENSION_GROUP,
                            'origin': 'automatic_night_search', 'boundary_parent': winner['run_name']})
            per_size += 1
    return {'status': 'EXPAND' if planned else 'LIMIT_REACHED' if blocked else 'WAIT_OR_BRACKETED',
            'new_arms': planned, 'limits': blocked,
            'automatic_arms_already_declared': automatic,
            'additional_tokens': len(planned) * int(rules['budget_tokens_per_arm'])}


def expand_lr_search(root: Path, selection: dict, extension_runner_live: bool, audit: dict) -> dict:
    from prepare_lr_extensions import prepare
    from select_lr_pilots import atomic_json
    path = root / POLICY
    policy = json.loads(path.read_text()) if path.is_file() else {}
    qualified = dict(selection)
    qualified['boundary_selections'] = [row for row in selection.get('boundary_selections', [])
        if anchor_names(root, row['size']) ==
        {item['name'] for item in audit.get('rows', []) if item.get('status') == 'PASS'
         and item.get('name') in anchor_names(root, row['size'])}]
    plan = plan_lr_extensions(root, qualified, policy)
    if extension_runner_live and plan['new_arms']:
        return {**plan, 'status': 'WAIT_FOR_EXTENSION_QUEUE_COMPLETION'}
    if plan['new_arms']:
        manifest = json.loads((root / CONTRACT).read_text())
        manifest['extensions'].extend(plan['new_arms'])
        atomic_json(root / CONTRACT, manifest)
        prepare(root)
    return plan
