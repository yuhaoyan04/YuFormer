"""Prepare and audit matched three-seed confirmations before recipe freeze."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import yaml

from audit_experiment_identities import audit, parse_env, sha256
from qualification_identity import qualification_identity, report_matches
from analyze_e2_results import non_embedding_params, scaling_loss
from research_scope import group_enabled

ROOT = Path(__file__).resolve().parents[2]
DOMAINS = ('academic', 'books', 'chinese', 'code', 'encyclopedia', 'math', 'multilingual', 'qa', 'web')


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True) + '\n'
    if path.is_file() and path.read_text() == text:
        return
    temporary = path.with_name(f'.{path.name}.{os.getpid()}')
    temporary.write_text(text)
    os.replace(temporary, path)


def result_path(root: Path, stage: str) -> Path:
    return root / f'experiments/{"e2_architecture" if stage == "e2" else "e3_data_mix"}/results/confirmation.json'


def prepare(root: Path, stage: str) -> dict:
    if not group_enabled(root, f'{stage}_confirm'):
        return {'status': 'DEFERRED', 'stage': stage, 'reason': 'research_scope'}
    policy_path = root / 'pretrain/configs/research_budget.yaml'
    policy = yaml.safe_load(policy_path.read_text())['confirmation']
    selection_path = result_path(root, stage).with_name('selection.json')
    selection = json.loads(selection_path.read_text())
    if selection.get('status') != 'PASS':
        raise ValueError('screen selection must PASS before confirmation')
    selected = selection['selection']['run_name']
    baseline = 'e2_l32' if stage == 'e2' else 'e3_m2_base'
    names = [('baseline', baseline)] + ([('selected', selected)] if selected != baseline else [])
    inputs = {str(policy_path.relative_to(root)): sha256(policy_path),
              str(selection_path.relative_to(root)): sha256(selection_path)}
    group = f'{stage}_confirm'
    destination = root / f'pretrain/configs/{group}'
    rows = []
    for role, source in names:
        source_config = root / f'pretrain/configs/{stage}/{source}.yaml'
        source_env = source_config.with_suffix('.env')
        inputs[str(source_config.relative_to(root))] = sha256(source_config)
        inputs[str(source_env.relative_to(root))] = sha256(source_env)
        for seed_index, seed in enumerate(policy['seeds'], 1):
            cfg = yaml.safe_load(source_config.read_text())
            name = f'{group}_{role}_s{seed_index}'
            cfg.update(run_name=name, seed=int(seed))
            train = cfg['train']
            tpu = (int(cfg['data']['seq_len']) * int(train['micro_batch_size'])
                   * int(train['grad_accum']) * int(cfg['parallel']['world_size']))
            iterations = int(policy['budget_tokens']) // tpu
            train.update(budget_tokens=int(policy['budget_tokens']), expected_iters=iterations,
                         eval_every=max(20, min(500, iterations // 4)), log_every=10,
                         save_every=max(20, min(1000, iterations // 4)))
            cfg['optim']['warmup_steps'] = max(5, min(2000, iterations // 100))
            target = destination / f'{name}.yaml'
            env = parse_env(source_env)
            env.update(YUFORMER_CONFIG=f'configs/{group}/{name}.yaml',
                       YUFORMER_PRETRAIN_OUT=str(root / f'data/experiments/{group}/{name}'),
                       MASTER_PORT=str(29800 + (0 if stage == 'e2' else 20) + len(rows)))
            destination.mkdir(parents=True, exist_ok=True)
            for path, content in ((target, yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)),
                                  (target.with_suffix('.env'), ''.join(f'{k}={v}\n' for k, v in env.items()))):
                if not path.is_file() or path.read_text() != content:
                    if (root / f'data/experiments/{group}/{name}/current_attempt.json').exists():
                        raise ValueError(f'confirmation contract drift after launch: {name}')
                    path.write_text(content)
            rows.append({'name': name, 'role': role, 'source': source, 'seed': int(seed),
                         'config_sha256': sha256(target), 'env_sha256': sha256(target.with_suffix('.env'))})
    result = {'status': 'PASS', 'stage': stage, 'group': group,
              'selected': selected, 'baseline': baseline, 'inputs': inputs, 'runs': rows,
              'policy': policy, 'generator_sha256': sha256(Path(__file__))}
    manifest = root / f'data/health/{group}-contracts.json'
    atomic(manifest, result)
    return result


def paired_interval(values: list[float]) -> dict:
    if len(values) != 3 or any(not math.isfinite(value) for value in values):
        raise ValueError('confirmation requires three finite paired seeds')
    mean = sum(values) / 3
    std = math.sqrt(sum((value - mean) ** 2 for value in values) / 2)
    half = 4.302652729911275 * std / math.sqrt(3)
    return {'mean': mean, 'ci95': [mean - half, mean + half], 'seed_deltas': values}


def analyze(root: Path, stage: str) -> dict:
    if not group_enabled(root, f'{stage}_confirm'):
        return {'status': 'DEFERRED', 'stage': stage, 'reason': 'research_scope'}
    manifest_path = root / f'data/health/{stage}_confirm-contracts.json'
    if not manifest_path.is_file():
        return {'status': 'BLOCKED', 'reason': 'confirmation_contracts_pending'}
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest['inputs'].items():
        if sha256(root / relative) != expected:
            return {'status': 'BLOCKED', 'reason': f'stale_confirmation_input:{relative}'}
    group = manifest['group']
    identity = audit(root, {group: {row['name'] for row in manifest['runs']}})
    if identity['status'] != 'PASS':
        return {'status': 'BLOCKED', 'reason': 'confirmation_identity_audit', 'identity_audit': identity}
    qualification_path = root / 'data/health/torchtitan-qualification.json'
    qualification = json.loads(qualification_path.read_text())
    if qualification.get('status') != 'PASS' or not report_matches(qualification, qualification_identity(root)):
        return {'status': 'BLOCKED', 'reason': 'qualification_not_current'}
    presets = yaml.safe_load((root / 'model/configs/model_configs.yaml').read_text())['presets']
    scaling_path = root / 'experiments/e1_scaling/results/scaling_fit.json'
    scaling = json.loads(scaling_path.read_text()) if stage == 'e2' else None
    metrics = {}
    for row in manifest['runs']:
        config_path = root / f'pretrain/configs/{group}/{row["name"]}.yaml'
        if sha256(config_path) != row['config_sha256'] or sha256(config_path.with_suffix('.env')) != row['env_sha256']:
            return {'status': 'BLOCKED', 'reason': 'confirmation_config_drift'}
        cfg = yaml.safe_load(config_path.read_text())
        result = json.loads((root / f'data/experiments/{group}/{row["name"]}/result.json').read_text())
        hyper = result.get('hyperparameters', {})
        tpu = cfg['data']['seq_len'] * cfg['train']['micro_batch_size'] * cfg['train']['grad_accum']
        tokens = cfg['train']['expected_iters'] * tpu
        if (hyper.get('budget_tokens') != cfg['train']['budget_tokens']
                or hyper.get('trained_tokens') != tokens or hyper.get('tokens_per_step') != tpu
                or hyper.get('trained_steps') != cfg['train']['expected_iters']):
            return {'status': 'BLOCKED', 'reason': 'confirmation_training_accounting_mismatch'}
        scalar = result['metrics']
        losses = {d: float(scalar[f'validation_metrics/domain/{d}/loss']['last']) for d in DOMAINS}
        loss = float(scalar['validation_metrics/loss']['last'])
        if not all(math.isfinite(value) and value > 0 for value in [loss, *losses.values()]):
            return {'status': 'FAIL', 'reason': 'nonfinite_confirmation_loss'}
        adjustment = 0.
        if stage == 'e2':
            if scaling.get('status') != 'PASS':
                return {'status': 'BLOCKED', 'reason': 'confirmation_scaling_pending'}
            current_n = non_embedding_params(presets[cfg['model']['preset']])
            reference_n = non_embedding_params(presets['pretrain-0.5b'])
            adjustment = scaling_loss(scaling['parameters'], current_n, tokens) - scaling_loss(scaling['parameters'], reference_n, tokens)
        metrics[row['role'], row['seed']] = (loss - adjustment, losses)
    policy = manifest['policy']
    seeds = policy['seeds']
    selected_role = 'selected' if any(row['role'] == 'selected' for row in manifest['runs']) else 'baseline'
    delta = paired_interval([metrics[selected_role, s][0] - metrics['baseline', s][0] for s in seeds])
    domains = {d: paired_interval([(metrics[selected_role, s][1][d] - metrics['baseline', s][1][d])
                                   / metrics['baseline', s][1][d] for s in seeds]) for d in DOMAINS}
    passed = (delta['ci95'][1] <= policy['practical_equivalence_absolute_loss']
              and all(v['ci95'][1] <= policy['maximum_domain_relative_regression'] for v in domains.values()))
    snapshot = {**identity['evidence_snapshot'], **manifest['inputs'],
                str(manifest_path.relative_to(root)): sha256(manifest_path),
                str(qualification_path.relative_to(root)): sha256(qualification_path),
                str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__))}
    if stage == 'e2':
        snapshot[str(scaling_path.relative_to(root))] = sha256(scaling_path)
    return {'status': 'PASS' if passed else 'FAIL', 'stage': stage,
            'reason': None if passed else 'longer_budget_noninferiority_failed',
            'selected': manifest['selected'], 'baseline': manifest['baseline'],
            'aggregate_normalized_loss_delta': delta, 'relative_domain_deltas': domains,
            'seed_count': 3, 'budget_tokens_per_arm': policy['budget_tokens'],
            'evidence_snapshot': snapshot, 'evidence_snapshot_count': len(snapshot),
            'scope': 'longer-budget noninferiority check; no claim of statistically significant superiority'}


def current(root: Path, stage: str, value: dict) -> bool:
    if value.get('status') != 'PASS' or not value.get('evidence_snapshot'):
        return False
    try:
        return all(sha256(root / relative) == expected for relative, expected in value['evidence_snapshot'].items())
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--stage', required=True, choices=('e2', 'e3'))
    parser.add_argument('--action', required=True, choices=('prepare', 'analyze'))
    args = parser.parse_args()
    value = prepare(args.root, args.stage) if args.action == 'prepare' else analyze(args.root, args.stage)
    if args.action == 'analyze':
        atomic(result_path(args.root, args.stage), value)
    print(json.dumps({'status': value['status'], 'stage': args.stage}))
    return 0 if value['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())
