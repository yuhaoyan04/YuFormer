#!/usr/bin/env python3
"""Measure adjacent width/depth points after the queued eight-GPU sweep."""
from pathlib import Path
import fcntl
import json
import os
import time

import benchmark_model_sizes as bench


def main():
    previous_health = bench.ROOT / 'data/health/model-width-followup.json'
    bench.CONFIGS = bench.ROOT / 'experiments/e2_architecture/configs/gpu_width_frontier_20261114'
    bench.RESULTS = bench.ROOT / 'experiments/e2_architecture/results/gpu_width_frontier_20261114'
    bench.RUNS = bench.ROOT / 'data/experiments/gpu_width_frontier_20261114'
    bench.HEALTH = bench.ROOT / 'data/health/model-width-frontier.json'
    bench.SHAPES = [('size_117b_h3072_l9', (9, 3072, 8192)),
                    ('size_123b_h3328_l8', (8, 3328, 8960))]
    bench.RESULTS.mkdir(parents=True, exist_ok=True)
    lock = (bench.RESULTS / '.orchestrator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'status': 'WAITING_WIDTH_FOLLOWUP', 'pid': os.getpid(), 'updated_at': bench.now(),
             'production_qualified': False, 'data_preprocessing_must_continue': True}
    bench.publish(bench.HEALTH, state)
    try:
        while True:
            previous = json.loads(previous_health.read_text())
            if previous['status'] == 'COMPLETE_SCREENING':
                break
            if previous['status'] not in {'RUNNING', 'WAITING_MAIN_SWEEP'}:
                raise RuntimeError('previous width sweep needs repair: ' + previous['status'])
            state['updated_at'] = bench.now()
            bench.publish(bench.HEALTH, state)
            time.sleep(30)
        plan = bench.prepare()
        plan['wrapper_tool_sha256'] = bench.digest(__file__)
        if plan['source_qualification_identity'] != previous['source_qualification_identity']:
            raise ValueError('source changed after preceding width measurements')
        baseline = previous['cases']['baseline_053b_b8']
        state.update(status='RUNNING', source_qualification_identity=plan['source_qualification_identity'],
                     baseline_reference={'variant': baseline['variant']['name'],
                         'identity_sha256': baseline['identity_sha256'],
                         'p10_tps_per_device': baseline['evaluation']['p10_tps_per_device'],
                         'result_sha256': baseline['result_sha256'],
                         'source': str(previous_health)}, cases={})
        plan['baseline_reference'] = state['baseline_reference']
        bench.publish(bench.RESULTS / 'plan.json', plan)
        state['plan_sha256'] = bench.digest(bench.RESULTS / 'plan.json')
        bench.publish(bench.HEALTH, state)
        for candidate in plan['candidates']:
            for variant in candidate['variants']:
                path = bench.RESULTS / (variant['name'] + '.json')
                if path.exists():
                    result = json.loads(path.read_text())
                    if result['variant'] != variant or result['parameters'] != candidate['parameters']:
                        raise ValueError('prior refinement result contract mismatch')
                else:
                    result = bench.run_variant(candidate, variant, state)
                state['cases'][variant['name']] = result
                state['updated_at'] = bench.now()
                bench.publish(bench.HEALTH, state)
                if result['evaluation']['status'] == 'PASS_SCREENING':
                    break
                if result['evaluation']['status'] != 'OOM':
                    raise RuntimeError('refinement failed: ' + variant['name'])
        if bench.digest(__file__) != plan['wrapper_tool_sha256']:
            raise ValueError('refinement runner changed while measuring')
        state.update(status='COMPLETE_SCREENING', updated_at=bench.now(), production_qualified=False)
        bench.publish(bench.HEALTH, state)
        bench.publish(bench.RESULTS / 'summary.json', state)
    except BaseException as exc:
        state.update(status='BLOCKED', updated_at=bench.now(), error=f'{type(exc).__name__}: {exc}')
        bench.publish(bench.HEALTH, state)
        raise


if __name__ == '__main__':
    main()
