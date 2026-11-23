#!/usr/bin/env python3
"""Compare declared hybrid-attention layouts on the qualified cuDNN path."""
import fcntl
import json
import os
from pathlib import Path
import time

import yaml
import benchmark_model_sizes as bench


def main():
    previous_health = bench.ROOT / 'data/health/model-width-frontier.json'
    bench.CONFIGS = bench.ROOT / 'experiments/e2_architecture/configs/attention_layouts_20261114'
    bench.RESULTS = bench.ROOT / 'experiments/e2_architecture/results/attention_layouts_20261114'
    bench.RUNS = bench.ROOT / 'data/experiments/attention_layouts_20261114'
    bench.HEALTH = bench.ROOT / 'data/health/model-attention-layouts.json'
    settings = [('hybrid_8l_h3072_g4', (8, 3072, 8192), 4),
                ('hybrid_9l_h3072_g2', (9, 3072, 8192), 2),
                ('hybrid_8l_h3328_g2', (8, 3328, 8960), 2)]
    bench.SHAPES = [(name, shape) for name, shape, _ in settings]
    bench.RESULTS.mkdir(parents=True, exist_ok=True)
    lock = (bench.RESULTS / '.orchestrator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'status': 'WAITING_FRONTIER', 'pid': os.getpid(), 'updated_at': bench.now(),
             'production_qualified': False, 'data_preprocessing_must_continue': True}
    bench.publish(bench.HEALTH, state)
    try:
        while True:
            previous = json.loads(previous_health.read_text())
            if previous['status'] == 'COMPLETE_SCREENING':
                break
            if previous['status'] not in {'WAITING_WIDTH_FOLLOWUP', 'RUNNING'}:
                raise RuntimeError('frontier measurements need repair: ' + previous['status'])
            state['updated_at'] = bench.now()
            bench.publish(bench.HEALTH, state)
            time.sleep(30)
        plan = bench.prepare()
        if plan['source_qualification_identity'] != previous['source_qualification_identity']:
            raise ValueError('source changed before attention-layout comparison')
        import torch
        import sys
        sys.path.insert(0, str(bench.ROOT / 'model/src'))
        from yuformer_model.config import ModelConfig
        from yuformer_model.model import YuFormerModel
        frequency = {name: every for name, _, every in settings}
        for candidate in plan['candidates']:
            path = Path(candidate['model_path'])
            model = candidate['shape']
            model['attention_every'] = frequency[candidate['name']]
            path.write_text(yaml.safe_dump({'presets': {'size-sweep': model}}, sort_keys=True))
            with torch.device('meta'):
                instance = YuFormerModel(ModelConfig(**model))
            candidate.update(parameters=instance.num_parameters(), model_sha256=bench.digest(path),
                             full_attention_layers=instance.config.num_full_attn_layers)
            del instance
        plan.update(wrapper_tool_sha256=bench.digest(__file__),
                    baseline_reference=previous['baseline_reference'],
                    purpose='HYBRID_ATTENTION_LAYOUT_THROUGHPUT_AND_QUALITY_SCREENING')
        bench.publish(bench.RESULTS / 'plan.json', plan)
        state.update(status='RUNNING', source_qualification_identity=plan['source_qualification_identity'],
                     plan_sha256=bench.digest(bench.RESULTS / 'plan.json'),
                     baseline_reference=plan['baseline_reference'], cases={})
        bench.publish(bench.HEALTH, state)
        for candidate in plan['candidates']:
            for variant in candidate['variants']:
                path = bench.RESULTS / (variant['name'] + '.json')
                if path.exists():
                    result = json.loads(path.read_text())
                    if result['variant'] != variant or result['parameters'] != candidate['parameters']:
                        raise ValueError('existing attention-layout result drift')
                else:
                    result = bench.run_variant(candidate, variant, state)
                state['cases'][variant['name']] = result
                state['updated_at'] = bench.now()
                bench.publish(bench.HEALTH, state)
                if result['evaluation']['status'] == 'PASS_SCREENING':
                    break
                if result['evaluation']['status'] != 'OOM':
                    raise RuntimeError('attention-layout comparison failed: ' + variant['name'])
        if bench.digest(__file__) != plan['wrapper_tool_sha256']:
            raise ValueError('attention-layout runner changed during measurements')
        state.update(status='COMPLETE_SCREENING', updated_at=bench.now())
        bench.publish(bench.HEALTH, state)
        bench.publish(bench.RESULTS / 'summary.json', state)
    except BaseException as exc:
        state.update(status='BLOCKED', updated_at=bench.now(), error=f'{type(exc).__name__}: {exc}')
        bench.publish(bench.HEALTH, state)
        raise


if __name__ == '__main__':
    main()
