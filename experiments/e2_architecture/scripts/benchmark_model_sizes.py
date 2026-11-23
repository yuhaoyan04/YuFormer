#!/usr/bin/env python3
"""Measure model-size throughput on all eight GPUs without changing production."""
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'pretrain/scripts'))
from qualification_identity import qualification_identity
from gpu_resource_policy import allowed_gpu_indices

PYTHON = Path('/opt/yuformer-torchtitan/bin/python')
CONFIGS = ROOT / 'experiments/e2_architecture/configs/gpu_size_sweep_20261114'
RESULTS = ROOT / 'experiments/e2_architecture/results/gpu_size_sweep_20261114'
RUNS = ROOT / 'data/experiments/gpu_size_sweep_20261114'
HEALTH = ROOT / 'data/health/model-size-sweep.json'
STEPS = 24
SKIP_STEPS = 8
TOKENS_PER_STEP = 4_194_304
SHAPES = [
    ('baseline_053b', None),
    ('size_1b_wide', (18, 2048, 5504)),
    ('size_1b_deep', (24, 1792, 4864)),
    ('size_15b', (18, 2560, 6912)),
    ('size_2b', (24, 2560, 6912)),
    ('size_3b', (26, 3072, 8192)),
    ('size_4b', (30, 3328, 8960)),
    ('size_5b', (32, 3584, 9728)),
]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def evaluate(result, *, steps=STEPS, skip_steps=SKIP_STEPS):
    issues = list(result.get('issues', []))
    hyper = result.get('hyperparameters') or {}
    if result.get('status') != 'PASS':
        issues.append('collected_result_not_pass')
    for key, expected in [('world_size', 8), ('tokens_per_step', TOKENS_PER_STEP),
                          ('trained_steps', steps), ('trained_tokens', steps * TOKENS_PER_STEP)]:
        if hyper.get(key) != expected:
            issues.append('accounting_mismatch:' + key)
    series = (result.get('series') or {}).get('throughput(tps)', [])
    steady = [p for p in series if skip_steps < int(p['step']) <= steps]
    values = [float(p['value']) for p in steady]
    if (len(values) != steps - skip_steps
            or {int(p['step']) for p in steady} != set(range(skip_steps + 1, steps + 1))
            or any(not math.isfinite(v) or v <= 0 for v in values)):
        issues.append('incomplete_or_nonfinite_steady_throughput')
    p10 = median = days = None
    if not issues:
        p10 = sorted(values)[max(0, math.ceil(len(values) * .1) - 1)]
        median = statistics.median(values)
        days = 1e12 / (8 * p10 * .9 * 86400)
    mfu = [float(p['value']) for p in (result.get('series') or {}).get('mfu(%)', [])
           if skip_steps < int(p['step']) <= steps and math.isfinite(float(p['value']))]
    return {'status': 'PASS_SCREENING' if not issues else 'FAIL', 'issues': sorted(set(issues)),
            'steady_points': len(values), 'p10_tps_per_device': p10,
            'median_tps_per_device': median, 'median_mfu_pct': statistics.median(mfu) if mfu else None,
            'projected_1t_days_duty_09': days,
            'within_30_days': days is not None and days <= 30,
            'within_33_days': days is not None and days <= 33,
            'production_qualified': False}


def prepare():
    sys.path.insert(0, str(ROOT / 'model/src'))
    import torch
    from yuformer_model.config import ModelConfig
    from yuformer_model.model import YuFormerModel
    base_model_path = ROOT / 'model/configs/pretrain_final_model_configs.yaml'
    base_model = yaml.safe_load(base_model_path.read_text())['presets']['pretrain-final']
    base_config = yaml.safe_load((ROOT / 'pretrain/configs/production_canary/production_8gpu.yaml').read_text())
    base_env = {}
    for line in (ROOT / 'pretrain/configs/production_canary/production_8gpu.env').read_text().splitlines():
        if line and not line.startswith('#'):
            key, value = line.split('=', 1)
            base_env[key] = value
    CONFIGS.mkdir(parents=True, exist_ok=True)
    candidates = []
    for name, shape in SHAPES:
        model = copy.deepcopy(base_model)
        model.pop('notes', None)
        if shape:
            layers, hidden, intermediate = shape
            model.update(num_layers=layers, hidden_size=hidden, intermediate_size=intermediate,
                         num_heads=hidden // 64, num_kv_heads=hidden // 256)
        model_path = CONFIGS / f'{name}.model.yaml'
        model_text = yaml.safe_dump({'presets': {'size-sweep': model}}, sort_keys=True)
        if model_path.exists() and model_path.read_text() != model_text:
            raise ValueError('refusing to overwrite a different candidate: ' + name)
        model_path.write_text(model_text)
        with torch.device('meta'):
            instance = YuFormerModel(ModelConfig(**model))
        params = instance.num_parameters()
        if name == 'baseline_053b' and params != 530107904:
            raise ValueError('baseline_053b requires the original 530107904-parameter model; resolve its archived bytes explicitly after production adoption')
        del instance
        variants = []
        for micro in (8, 4, 2):
            variant = f'{name}_b{micro}'
            config = copy.deepcopy(base_config)
            config['run_name'] = variant
            config['model'] = {'preset': 'size-sweep', 'configs': str(model_path)}
            config['optim'].update(warmup_steps=4)
            config['train'].update(expected_iters=STEPS, budget_tokens=STEPS * TOKENS_PER_STEP,
                                   micro_batch_size=micro, grad_accum=128 // micro,
                                   log_every=1, eval_every=STEPS, save_every=STEPS)
            assert config['parallel']['world_size'] == 8
            assert config['data']['seq_len'] * micro * config['train']['grad_accum'] * 8 == TOKENS_PER_STEP
            config_path = CONFIGS / f'{variant}.yaml'
            env_path = CONFIGS / f'{variant}.env'
            config_text = yaml.safe_dump(config, sort_keys=False)
            if config_path.exists() and config_path.read_text() != config_text:
                raise ValueError('candidate contract drift: ' + variant)
            config_path.write_text(config_text)
            env = dict(base_env, YUFORMER_CONFIG=str(config_path), YUFORMER_MODEL_CONFIGS=str(model_path),
                       YUFORMER_PRETRAIN_OUT=str(RUNS / variant), MASTER_PORT='29831',
                       OMP_NUM_THREADS='1', TORCHINDUCTOR_COMPILE_THREADS='2',
                       TORCHINDUCTOR_CACHE_DIR='/tmp/yuformer-inductor-cudnn-qualified-v3')
            env_path.write_text(''.join(f'{key}={value}\n' for key, value in env.items()))
            variants.append({'name': variant, 'microbatch': micro, 'grad_accum': 128 // micro,
                             'config': str(config_path), 'env': str(env_path),
                             'config_sha256': digest(config_path), 'env_sha256': digest(env_path)})
        candidates.append({'name': name, 'parameters': params, 'shape': model,
                           'model_path': str(model_path), 'model_sha256': digest(model_path),
                           'variants': variants})
    return {'schema_version': 1, 'prepared_at': now(), 'purpose': 'RESEARCH_THROUGHPUT_SCREENING_ONLY',
            'runner_tool_sha256': digest(__file__),
            'source_qualification_identity': qualification_identity(ROOT),
            'baseline_model_sha256': digest(base_model_path), 'steps': STEPS, 'skip_steps': SKIP_STEPS,
            'tokens_per_step': TOKENS_PER_STEP, 'world_size': 8, 'candidates': candidates,
            'data_preprocessing_must_continue': True, 'production_qualified': False}


def gpu_snapshot():
    p = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=15, check=True)
    keys = ('index', 'utilization_pct', 'memory_used_mib', 'memory_total_mib', 'temperature_c', 'power_w')
    return [dict(zip(keys, [float(x.strip()) for x in line.split(',')]))
            for line in p.stdout.splitlines() if line.strip()]


def require_idle():
    if allowed_gpu_indices(ROOT / 'data/health') != set(range(8)):
        raise ValueError('all eight GPUs must be authorized')
    p = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                       capture_output=True, text=True, timeout=15, check=True)
    if p.stdout.strip():
        raise ValueError('GPU compute processes are still active: ' + p.stdout.strip())
    pipeline = json.loads((ROOT / 'data/health/authorized-pretrain-pipeline.json').read_text())
    if pipeline.get('stage') == 'full_pretrain':
        raise ValueError('formal pretrain already active')


def run_variant(candidate, variant, state):
    require_idle()
    if digest(candidate['model_path']) != candidate['model_sha256']:
        raise ValueError('candidate model bytes changed')
    for label in ('config', 'env'):
        if digest(variant[label]) != variant[label + '_sha256']:
            raise ValueError('candidate contract bytes changed: ' + label)
    name = variant['name']
    run = RUNS / name
    log = RESULTS / f'{name}.log'
    result_path = RESULTS / f'{name}.json'
    if run.exists():
        raise ValueError('run output already exists without a reusable completed result: ' + name)
    environment = dict(os.environ)
    for key in ('YUFORMER_RUN_LEASE_HELD', 'YUFORMER_RUN_LEASE_TOKEN', 'YUFORMER_RUN_LEASE_GENERATION',
                'YUFORMER_ATTEMPT_ID', 'YUFORMER_ALLOW_TOPOLOGY_RESHARD', 'YUFORMER_OVERRIDE_GPUS_PER_NODE', 'EXTRA_ARGS'):
        environment.pop(key, None)
    started = time.monotonic()
    snapshot_max = []
    with log.open('w') as handle:
        process = subprocess.Popen(['bash', str(ROOT / 'pretrain/backends/torchtitan/launch.sh'), variant['env']],
                                   cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        state.update(active_candidate=candidate['name'], active_variant=name, child_pid=process.pid,
                     current_log=str(log), active_started_at=now())
        publish(HEALTH, state)
        try:
            while process.poll() is None:
                observed = gpu_snapshot()
                if not snapshot_max:
                    snapshot_max = observed
                else:
                    for peak, current in zip(snapshot_max, observed):
                        for key in ('memory_used_mib', 'temperature_c', 'power_w', 'utilization_pct'):
                            peak[key] = max(peak[key], current[key])
                state.update(updated_at=now(), current_gpu=observed, peak_gpu=snapshot_max,
                             elapsed_seconds=round(time.monotonic() - started, 1))
                publish(HEALTH, state)
                if time.monotonic() - started > 5400:
                    raise TimeoutError('screening variant exceeded 90 minutes')
                time.sleep(15)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    completed = {'candidate': candidate['name'], 'parameters': candidate['parameters'],
                 'variant': variant, 'returncode': process.returncode, 'finished_at': now(),
                 'wall_seconds': round(time.monotonic() - started, 1), 'log': str(log),
                 'peak_gpu': snapshot_max, 'production_qualified': False}
    if process.returncode == 0:
        collect = subprocess.run([str(PYTHON), str(ROOT / 'pretrain/scripts/collect_torchtitan_result.py'), str(run)],
                                 cwd=ROOT, capture_output=True, text=True)
        collected = json.loads((run / 'result.json').read_text())
        completed.update(collect_returncode=collect.returncode, evaluation=evaluate(collected),
                         result_sha256=digest(run / 'result.json'), identity_sha256=collected['identity_sha256'])
    else:
        with log.open('rb') as f:
            f.seek(max(0, log.stat().st_size - (1 << 20)))
            tail = f.read().decode(errors='replace')
        oom = 'out of memory' in tail.lower() or 'OutOfMemoryError' in tail
        completed.update(evaluation={'status': 'OOM' if oom else 'FAIL', 'issues': ['launch_failed']},
                         error_tail=tail[-6000:])
    if qualification_identity(ROOT) != state['source_qualification_identity']:
        completed['evaluation'] = {'status': 'FAIL', 'issues': ['training_source_changed_during_measurement']}
    publish(result_path, completed)
    return completed


def run():
    plan_path = RESULTS / 'plan.json'
    plan = json.loads(plan_path.read_text())
    if digest(__file__) != plan.get('runner_tool_sha256'):
        raise ValueError('screening runner differs from prepared plan')
    if qualification_identity(ROOT) != plan['source_qualification_identity']:
        raise ValueError('source identity differs from screening plan')
    state = {'status': 'RUNNING', 'pid': os.getpid(), 'started_at': now(),
             'purpose': plan['purpose'], 'source_qualification_identity': plan['source_qualification_identity'],
             'plan_sha256': digest(plan_path), 'cases': {}, 'production_qualified': False,
             'data_preprocessing_must_continue': True}
    publish(HEALTH, state)
    for candidate in plan['candidates']:
        for variant in candidate['variants']:
            path = RESULTS / f"{variant['name']}.json"
            if path.exists():
                completed = json.loads(path.read_text())
                if completed.get('variant') != variant or completed.get('parameters') != candidate['parameters']:
                    raise ValueError('prior result contract mismatch')
            else:
                completed = run_variant(candidate, variant, state)
            state['cases'][variant['name']] = completed
            state['updated_at'] = now()
            publish(HEALTH, state)
            status = completed['evaluation']['status']
            if status == 'PASS_SCREENING':
                break
            if status != 'OOM':
                raise RuntimeError('screening variant failed: ' + variant['name'])
            for _ in range(12):
                try:
                    require_idle()
                    break
                except ValueError:
                    time.sleep(5)
            else:
                raise RuntimeError('GPU processes did not drain after OOM')
    successful = [v for v in state['cases'].values() if v['evaluation']['status'] == 'PASS_SCREENING']
    baseline = next(v for v in successful if v['candidate'] == 'baseline_053b')
    baseline_tps = baseline['evaluation']['p10_tps_per_device']
    for row in successful:
        row['evaluation']['throughput_ratio_vs_concurrent_baseline'] = row['evaluation']['p10_tps_per_device'] / baseline_tps
    eligible = [v for v in successful if v['evaluation']['within_30_days']]
    state.update(status='COMPLETE_SCREENING', updated_at=now(),
                 largest_within_30_days=max(eligible, key=lambda v: v['parameters'])['candidate'] if eligible else None,
                 no_more_than_10pct_slowdown=[v['candidate'] for v in successful if v['evaluation']['throughput_ratio_vs_concurrent_baseline'] >= .9],
                 production_qualified=False, sustained_canary_required=True,
                 note='Throughput screening does not establish model quality or qualify a production architecture.')
    publish(HEALTH, state)
    publish(RESULTS / 'summary.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    lock = (RESULTS / '.orchestrator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.prepare_only:
        plan = prepare()
        publish(RESULTS / 'plan.json', plan)
        print(json.dumps([{'name': c['name'], 'parameters': c['parameters']} for c in plan['candidates']]))
        return
    try:
        run()
    except BaseException as exc:
        try:
            state = json.loads(HEALTH.read_text())
        except (OSError, ValueError):
            state = {}
        state.update(status='BLOCKED', error=f'{type(exc).__name__}: {exc}', updated_at=now())
        publish(HEALTH, state)
        raise


if __name__ == '__main__':
    main()
