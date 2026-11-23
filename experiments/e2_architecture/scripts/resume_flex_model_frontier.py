#!/usr/bin/env python3
"""Resume an immutable frontier plan after GPU process draining, preserving results."""
import json
import os
from pathlib import Path
import time

import benchmark_model_sizes as bench
import analyze_model_size_sweep as analysis


def main():
    bench.RESULTS = bench.ROOT/'experiments/e2_architecture/results/flex_frontier_20261114'
    bench.RUNS = bench.ROOT/'data/experiments/flex_frontier_20261114'
    bench.HEALTH = bench.ROOT/'data/health/model-flex-frontier.json'
    import sys
    sys.path.insert(0, str(bench.ROOT/'pretrain/src'))
    from yuformer_pretrain.lease import RunLease
    plan_path = bench.RESULTS/'plan.json'
    plan = json.loads(plan_path.read_bytes())
    state = json.loads(bench.HEALTH.read_bytes())
    if state['status'] != 'FAIL' or 'GPU compute processes are still active' not in state.get('error', ''):
        raise ValueError('only an explicit GPU-idle admission failure can use this resume')
    for path, expected in ((Path(bench.__file__), plan['runner_tool_sha256']),
            (Path(analysis.__file__), plan['analyzer_tool_sha256']),
            (Path(__file__).with_name('benchmark_flex_model_frontier.py'), plan['wrapper_tool_sha256'])):
        if bench.digest(path) != expected:
            raise ValueError('original immutable frontier source changed: '+str(path))
    if bench.qualification_identity(bench.ROOT) != plan['source_qualification_identity']:
        raise ValueError('training source changed before frontier resume')
    with RunLease(bench.RESULTS, 'resume-eight-gpu-hybrid-swa-frontier'):
        before = bench.RESULTS/'pre-resume-supervisor.json'
        if before.exists():
            raise ValueError('frontier resume already attempted')
        before.write_bytes(bench.HEALTH.read_bytes())
        state.update(status='RUNNING', pid=os.getpid(), resume_tool_sha256=bench.digest(__file__),
            resume_reason='Preserve successful measurements; wait for empty GPU process inventory before admission.',
            updated_at=bench.now())
        state.pop('error', None)
        try:
            for candidate in plan['candidates']:
                if bench.digest(candidate['model_path']) != candidate['model_sha256']:
                    raise ValueError('candidate model changed')
                for variant in candidate['variants']:
                    for key in ('config', 'env'):
                        if bench.digest(variant[key]) != variant[key+'_sha256']:
                            raise ValueError('candidate contract changed')
                    path = bench.RESULTS/f"{variant['name']}.json"
                    if path.exists():
                        result = json.loads(path.read_bytes())
                        if result.get('variant') != variant or result.get('parameters') != candidate['parameters']:
                            raise ValueError('existing completed measurement contract changed')
                        actual = bench.RUNS/variant['name']/'result.json'
                        if bench.digest(actual) != result.get('result_sha256'):
                            raise ValueError('completed measured result changed')
                        if bench.evaluate(json.loads(actual.read_bytes())) != result['evaluation']:
                            raise ValueError('completed measurement accounting no longer passes')
                    else:
                        deadline = time.monotonic()+600
                        while True:
                            try:
                                bench.require_idle()
                                break
                            except ValueError as exc:
                                if time.monotonic() >= deadline:
                                    raise
                                state.update(admission_wait=str(exc), updated_at=bench.now())
                                bench.publish(bench.HEALTH, state)
                                time.sleep(5)
                        state.pop('admission_wait', None)
                        result = bench.run_variant(candidate, variant, state)
                    result.update(analysis.memory_pressure(result['log']), log_sha256=bench.digest(result['log']))
                    state['cases'][variant['name']] = result
                    bench.publish(bench.HEALTH, state)
                    status = result['evaluation']['status']
                    if status == 'PASS_SCREENING' and not result['requires_smaller_batch_comparison']:
                        break
                    if status not in {'PASS_SCREENING', 'OOM'}:
                        raise RuntimeError('resumed frontier failed: '+variant['name'])
            eligible = [r for r in state['cases'].values() if r['evaluation']['status'] == 'PASS_SCREENING'
                and not r['requires_smaller_batch_comparison'] and r['evaluation']['within_33_days']]
            state.update(status='COMPLETE_SCREENING', largest_within_33_days=max(eligible,
                key=lambda r: r['parameters'], default=None), sustained_quality_and_actual_data_recovery_required=True)
        except Exception as exc:
            state.update(status='FAIL', error=repr(exc))
            raise
        finally:
            state['updated_at'] = bench.now()
            bench.publish(bench.HEALTH, state)


if __name__ == '__main__':
    main()
