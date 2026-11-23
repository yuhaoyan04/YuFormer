#!/usr/bin/env python3
"""Measure the hybrid backend on a bounded depth-preserving eight-GPU grid."""
import json
import os
from pathlib import Path

import yaml

import benchmark_model_sizes as bench
import analyze_model_size_sweep as analysis


def main():
    bench.CONFIGS = bench.ROOT/'experiments/e2_architecture/configs/flex_frontier_20261114'
    bench.RESULTS = bench.ROOT/'experiments/e2_architecture/results/flex_frontier_20261114'
    bench.RUNS = bench.ROOT/'data/experiments/flex_frontier_20261114'
    bench.HEALTH = bench.ROOT/'data/health/model-flex-frontier.json'
    bench.RESULTS.mkdir(parents=True, exist_ok=True)
    import sys
    sys.path.insert(0, str(bench.ROOT/'pretrain/src'))
    from yuformer_pretrain.lease import RunLease
    with RunLease(bench.RESULTS, 'eight-gpu-hybrid-swa-frontier'):
        parity_path = bench.ROOT/'data/health/flex-module-parity.json'
        parity = json.loads(parity_path.read_bytes())
        if (parity.get('status') != 'PASS' or len(parity.get('results', [])) != 8
                or any(row.get('status') != 'PASS' for row in parity['results'])):
            raise ValueError('all eight actual compiled attention-module parity cases must PASS')
        for row in parity['results']:
            for name, sha in row['sources'].items():
                if bench.digest(bench.ROOT/name) != sha:
                    raise ValueError('compiled module parity source is stale: '+name)
        bench.require_idle()
        if (bench.RESULTS/'plan.json').exists():
            raise ValueError('refusing to overwrite a frontier plan')
        bench.SHAPES = [('baseline_053b', None),
            ('flex_18l_h2048_g6', (18, 2048, 5504)),
            ('flex_14l_h2560_g6', (14, 2560, 6912)),
            ('flex_18l_h2560_g6', (18, 2560, 6912)),
            ('flex_20l_h2560_g6', (20, 2560, 6912)),
            ('flex_24l_h2560_g6', (24, 2560, 6912))]
        plan = bench.prepare()
        for candidate in plan['candidates']:
            if candidate['shape']['attention_every'] != 6:
                raise ValueError('this frontier must preserve the original six-layer global-attention cadence')
            for variant in candidate['variants']:
                path = Path(variant['config'])
                config = yaml.safe_load(path.read_bytes())
                config['attention']['sdpa_backend'] = 'cudnn_flex_swa'
                path.write_text(yaml.safe_dump(config, sort_keys=False))
                variant['config_sha256'] = bench.digest(path)
        plan.update(purpose='RESEARCH_HYBRID_BACKEND_DEPTH_FRONTIER_NOT_PRODUCTION_QUALIFICATION',
            wrapper_tool_sha256=bench.digest(__file__), module_parity_sha256=bench.digest(parity_path),
            analyzer_tool_sha256=bench.digest(analysis.__file__), authorized_launch_ceiling_days=33,
            target_days=30, selection_preference='LARGER_MODEL_NEAR_30_DAYS_SUBJECT_TO_QUALITY_AND_RECOVERY')
        bench.publish(bench.RESULTS/'plan.json', plan)
        state = dict(status='RUNNING', pid=os.getpid(), updated_at=bench.now(), cases={},
            purpose=plan['purpose'], source_qualification_identity=plan['source_qualification_identity'],
            plan_sha256=bench.digest(bench.RESULTS/'plan.json'), production_qualified=False,
            data_preprocessing_must_continue=True)
        bench.publish(bench.HEALTH, state)
        try:
            for candidate in plan['candidates']:
                for variant in candidate['variants']:
                    result = bench.run_variant(candidate, variant, state)
                    result.update(analysis.memory_pressure(result['log']))
                    state['cases'][variant['name']] = result
                    bench.publish(bench.HEALTH, state)
                    status = result['evaluation']['status']
                    if status == 'PASS_SCREENING' and not result['requires_smaller_batch_comparison']:
                        break
                    if status not in {'PASS_SCREENING', 'OOM'}:
                        raise RuntimeError('hybrid model frontier failed: '+variant['name'])
            if (bench.digest(__file__) != plan['wrapper_tool_sha256']
                    or bench.digest(bench.__file__) != plan['runner_tool_sha256']
                    or bench.digest(analysis.__file__) != plan['analyzer_tool_sha256']):
                raise ValueError('frontier source changed while running')
            eligible = [r for r in state['cases'].values() if r['evaluation']['status'] == 'PASS_SCREENING'
                and not r['requires_smaller_batch_comparison'] and r['evaluation']['within_33_days']]
            state.update(status='COMPLETE_SCREENING', updated_at=bench.now(),
                largest_within_33_days=max(eligible, key=lambda r: r['parameters'], default=None),
                sustained_quality_and_actual_data_recovery_required=True)
        except Exception as exc:
            state.update(status='FAIL', error=repr(exc), updated_at=bench.now())
            raise
        finally:
            bench.publish(bench.HEALTH, state)


if __name__ == '__main__':
    main()
