#!/usr/bin/env python3
"""Queue a width/depth follow-up after the user-requested 1B--5B sweep."""
from pathlib import Path
import fcntl
import json
import os
import time

import benchmark_model_sizes as bench


def main():
    main_health = bench.HEALTH
    main_plan_path = bench.RESULTS / 'plan.json'
    bench.CONFIGS = bench.ROOT / 'experiments/e2_architecture/configs/gpu_width_followup_20261114'
    bench.RESULTS = bench.ROOT / 'experiments/e2_architecture/results/gpu_width_followup_20261114'
    bench.RUNS = bench.ROOT / 'data/experiments/gpu_width_followup_20261114'
    bench.HEALTH = bench.ROOT / 'data/health/model-width-followup.json'
    bench.SHAPES = [
        ('baseline_053b', None),
        ('size_1b_h2304_l14', (14, 2304, 6144)),
        ('size_1b_h2560_l12', (12, 2560, 6912)),
        ('size_1b_h2816_l10', (10, 2816, 7552)),
        ('size_1b_h3072_l8', (8, 3072, 8192)),
        ('size_4b_lowmem_control', (30, 3328, 8960)),
        ('size_5b_lowmem_control', (32, 3584, 9728)),
    ]
    bench.RESULTS.mkdir(parents=True, exist_ok=True)
    lock = (bench.RESULTS / '.orchestrator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        plan = bench.prepare()
        for candidate in plan['candidates']:
            if candidate['name'].endswith('_lowmem_control'):
                candidate['variants'] = [v for v in candidate['variants'] if v['microbatch'] == 2]
        wrapper_sha = bench.digest(__file__)
        plan.update(wrapper_tool_sha256=wrapper_sha,
                    quality_caveat='Fewer layers/full-attention blocks may change model quality; throughput alone is insufficient.')
        bench.publish(bench.RESULTS / 'plan.json', plan)
        while True:
            previous = json.loads(main_health.read_text())
            if previous['status'] == 'COMPLETE_SCREENING':
                break
            if previous['status'] != 'RUNNING':
                raise RuntimeError('main scale sweep needs repair before follow-up: ' + previous['status'])
            bench.publish(bench.HEALTH, {'status': 'WAITING_MAIN_SWEEP', 'pid': os.getpid(),
                                       'updated_at': bench.now(), 'plan_sha256': bench.digest(bench.RESULTS / 'plan.json'),
                                       'candidates': [{'name': c['name'], 'parameters': c['parameters']} for c in plan['candidates']],
                                       'data_preprocessing_must_continue': True, 'production_qualified': False})
            time.sleep(30)
        if bench.digest(__file__) != wrapper_sha:
            raise ValueError('follow-up runner changed while queued')
        main_plan = json.loads(main_plan_path.read_text())
        main_models = {c['name']: c for c in main_plan['candidates']}
        omitted = []
        selected = []
        for candidate in plan['candidates']:
            original = candidate['name'].removesuffix('_lowmem_control')
            existing = previous.get('cases', {}).get(original + '_b2')
            if (candidate['name'].endswith('_lowmem_control') and existing
                    and existing['evaluation']['status'] == 'PASS_SCREENING'
                    and main_models[original]['model_sha256'] == candidate['model_sha256']
                    and main_plan['source_qualification_identity'] == plan['source_qualification_identity']
                    and existing['variant']['microbatch'] == 2
                    and existing['variant']['grad_accum'] == 64):
                omitted.append({'candidate': candidate['name'], 'existing_main_variant': original + '_b2',
                                'existing_identity_sha256': existing['identity_sha256'],
                                'reason': 'same model bytes, source, microbatch and accumulation already measured'})
            else:
                selected.append(candidate)
        plan['candidates'] = selected
        plan['omitted_duplicate_controls'] = omitted
        bench.publish(bench.RESULTS / 'plan.json', plan)
        bench.run()
    except BaseException as exc:
        try:
            state = json.loads(bench.HEALTH.read_text())
        except (OSError, ValueError):
            state = {}
        state.update(status='BLOCKED', error=f'{type(exc).__name__}: {exc}', updated_at=bench.now())
        bench.publish(bench.HEALTH, state)
        raise


if __name__ == '__main__':
    main()
