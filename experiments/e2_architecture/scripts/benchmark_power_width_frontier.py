#!/usr/bin/env python3
"""Check the two next hardware-friendly widths after matched LR controls finish."""
from pathlib import Path
import fcntl
import json
import os
import time

import yaml
import benchmark_model_sizes as bench
import analyze_model_size_sweep as analysis


def main():
    bench.CONFIGS = bench.ROOT/'experiments/e2_architecture/configs/power_width_frontier_20261114'
    bench.RESULTS = bench.ROOT/'experiments/e2_architecture/results/power_width_frontier_20261114'
    bench.RUNS = bench.ROOT/'data/experiments/power_width_frontier_20261114'
    bench.HEALTH = bench.ROOT/'data/health/model-power-width-frontier.json'
    bench.RESULTS.mkdir(parents=True, exist_ok=True)
    lock = (bench.RESULTS/'.orchestrator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'status': 'WAITING_LR_CONTROLS', 'pid': os.getpid(), 'updated_at': bench.now(),
        'purpose': 'BOUNDED_TWO_WIDTH_SCREENING_NOT_PRODUCTION_QUALIFICATION',
        'production_qualified': False, 'data_preprocessing_must_continue': True,
        'wrapper_tool_sha256': bench.digest(__file__), 'loaded_benchmark_sha256': bench.digest(bench.__file__),
        'loaded_analyzer_sha256': bench.digest(analysis.__file__),
        'queued_base_inputs': {relative: bench.digest(bench.ROOT/relative) for relative in
            ('model/configs/pretrain_final_model_configs.yaml',
             'pretrain/configs/production_canary/production_8gpu.yaml',
             'pretrain/configs/production_canary/production_8gpu.env')}}
    bench.publish(bench.HEALTH, state)
    try:
        prior_path = bench.ROOT/'data/health/model-budget-lr-controls.json'
        while True:
            prior = json.loads(prior_path.read_text())
            if prior['status'] in {'COMPLETE_LR_QUALITY_CONTROLS', 'NOT_NEEDED_QUALITY_REVIEW_CLEAR'}:
                break
            if prior['status'] not in {'RUNNING', 'WAITING_BUDGET_SUSTAINED'}:
                raise RuntimeError('LR controls need repair: '+prior['status'])
            state['updated_at'] = bench.now()
            bench.publish(bench.HEALTH, state)
            time.sleep(30)
        if bench.digest(bench.__file__) != state['loaded_benchmark_sha256']:
            raise ValueError('benchmark module changed while width frontier was queued')
        if (bench.digest(analysis.__file__) != state['loaded_analyzer_sha256'] or
                any(bench.digest(bench.ROOT/name) != value for name, value in state['queued_base_inputs'].items())):
            raise ValueError('queued analysis/model/data contracts changed before width measurement')
        bench.SHAPES = [('hybrid_8l_h3584_g2', (8, 3584, 9728)),
                        ('hybrid_8l_h4096_g2', (8, 4096, 11008))]
        plan = bench.prepare()
        current = json.loads((bench.ROOT/'data/health/model-budget-sustained-canary.json').read_text())
        if plan['source_qualification_identity'] != current['source_qualification_identity']:
            raise ValueError('training source changed before bounded width comparison')
        import sys
        import torch
        sys.path.insert(0, str(bench.ROOT/'model/src'))
        from yuformer_model.config import ModelConfig
        from yuformer_model.model import YuFormerModel
        for candidate in plan['candidates']:
            path = Path(candidate['model_path'])
            document = yaml.safe_load(path.read_text())
            shape = document['presets']['size-sweep']
            shape['attention_every'] = 2
            path.write_text(yaml.safe_dump(document, sort_keys=True))
            with torch.device('meta'):
                model = YuFormerModel(ModelConfig(**shape))
            candidate.update(shape=shape, parameters=model.num_parameters(), model_sha256=bench.digest(path))
            del model
        plan.update(purpose=state['purpose'], wrapper_tool_sha256=state['wrapper_tool_sha256'],
                    reference_model=1199628544, authorized_launch_ceiling_days=33,
                    reason='Bracket the next valid GQA width and the next power-of-two width with actual eight-GPU measurements.')
        bench.publish(bench.RESULTS/'plan.json', plan)
        state.update(status='RUNNING', source_qualification_identity=plan['source_qualification_identity'],
                     plan_sha256=bench.digest(bench.RESULTS/'plan.json'), cases={})
        bench.publish(bench.HEALTH, state)
        for candidate in plan['candidates']:
            for variant in candidate['variants']:
                result = bench.run_variant(candidate, variant, state)
                state['cases'][variant['name']] = result
                pressure = analysis.memory_pressure(result['log'])
                result.update(pressure)
                status = result['evaluation']['status']
                result['allocator_safe_within_33_days'] = bool(status == 'PASS_SCREENING' and
                    not pressure['requires_smaller_batch_comparison'] and
                    result['evaluation']['projected_1t_days_duty_09'] <= 33)
                state['updated_at'] = bench.now()
                bench.publish(bench.HEALTH, state)
                if status == 'PASS_SCREENING' and not pressure['requires_smaller_batch_comparison']:
                    break
                if status not in {'PASS_SCREENING', 'OOM'}:
                    raise RuntimeError('bounded width test failed: '+variant['name'])
        choices = [case for case in state['cases'].values() if case['allocator_safe_within_33_days']]
        best = max(choices, key=lambda case: case['parameters'], default=None)
        if (bench.digest(__file__) != state['wrapper_tool_sha256'] or
                bench.digest(bench.__file__) != state['loaded_benchmark_sha256'] or
                bench.digest(analysis.__file__) != state['loaded_analyzer_sha256'] or
                any(bench.digest(bench.ROOT/name) != value for name, value in state['queued_base_inputs'].items())):
            raise ValueError('bounded width orchestration code changed')
        state.update(status='COMPLETE_SCREENING', updated_at=bench.now(),
            largest_feasible_larger_candidate=best,
            further_qualification_required=bool(best),
            note='Only two adjacent width controls; larger candidates still require matched sustained/LR/quality/recovery gates.')
        bench.publish(bench.HEALTH, state)
        bench.publish(bench.RESULTS/'summary.json', state)
    except BaseException as exc:
        state.update(status='BLOCKED', updated_at=bench.now(), error=f'{type(exc).__name__}: {exc}')
        bench.publish(bench.HEALTH, state)
        raise


if __name__ == '__main__':
    main()
