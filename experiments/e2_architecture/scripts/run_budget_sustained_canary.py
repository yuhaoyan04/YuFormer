#!/usr/bin/env python3
"""Sustain the fresh four-group frontier choice after the earlier backup finishes."""
from pathlib import Path
import fcntl
import json
import math
import os
import time

import yaml
import benchmark_model_sizes as bench
import analyze_model_size_sweep as analysis


def main():
    analyzer_sha = bench.digest(analysis.__file__)
    previous_health = bench.ROOT / 'data/health/model-selected-sustained-canary.json'
    frontier = bench.ROOT / 'data/health/model-attention-layouts.json'
    health_files = {'size': 'model-size-sweep.json', 'width_and_batch': 'model-width-followup.json',
                    'width_frontier': 'model-width-frontier.json',
                    'attention_layout': 'model-attention-layouts.json'}
    bench.CONFIGS = bench.ROOT / 'experiments/e2_architecture/configs/budget_sustained_20261114'
    bench.RESULTS = bench.ROOT / 'experiments/e2_architecture/results/budget_sustained_20261114'
    bench.RUNS = bench.ROOT / 'data/experiments/budget_sustained_20261114'
    bench.HEALTH = bench.ROOT / 'data/health/model-budget-sustained-canary.json'
    bench.RESULTS.mkdir(parents=True, exist_ok=True)
    lock = (bench.RESULTS / '.orchestrator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'status': 'WAITING_BACKUP_SUSTAINED', 'pid': os.getpid(), 'updated_at': bench.now(),
             'purpose': 'SUSTAINED_MODEL_SELECTION_NOT_PRODUCTION_QUALIFICATION',
             'production_qualified': False, 'data_preprocessing_must_continue': True,
             'wrapper_tool_sha256': bench.digest(__file__),
             'loaded_analyzer_sha256': analyzer_sha, 'expected_candidate': 'hybrid_8l_h3328_g2',
             'expected_parameters': 1199628544, 'target_completion_days': 30,
             'authorized_launch_ceiling_days': 33,
             'selection_reason': 'User explicitly prefers larger near 30 days; retain authorized hard ceiling of 33 days'}
    bench.publish(bench.HEALTH, state)
    try:
        while True:
            previous = json.loads(previous_health.read_text())
            if previous['status'] in {'PASS_SUSTAINED_SCREENING', 'FAIL_RUNTIME_CEILING'}:
                state['backup_sustained_status'] = previous['status']
                state['backup_sustained_sha256'] = bench.digest(previous_health)
                break
            if previous['status'] != 'RUNNING':
                raise RuntimeError('backup sustained test requires repair: ' + previous['status'])
            state['updated_at'] = bench.now()
            bench.publish(bench.HEALTH, state)
            time.sleep(30)
        state['status'] = 'WAITING_ATTENTION_LAYOUTS'
        bench.publish(bench.HEALTH, state)
        while True:
            prior = json.loads(frontier.read_text())
            if prior['status'] == 'COMPLETE_SCREENING':
                break
            if prior['status'] not in {'WAITING_FRONTIER', 'RUNNING'}:
                raise RuntimeError('frontier measurement needs repair: ' + prior['status'])
            state['updated_at'] = bench.now()
            bench.publish(bench.HEALTH, state)
            time.sleep(30)
        if bench.digest(analysis.__file__) != analyzer_sha:
            raise ValueError('analyzer source changed while this job was queued')
        comparison = analysis.analyze()
        if set(comparison['groups']) != {'size', 'width_and_batch', 'width_frontier', 'attention_layout'}:
            raise ValueError('all four model measurement groups must be loaded')
        if comparison['status'] != 'SCREENING_COMPLETE':
            raise ValueError('all model measurement groups must complete first')
        feasible = [row for row in comparison['measurements']
                    if row['eligible_throughput_screening']
                    and row['projected_1t_days_duty_09'] <= 33]
        selected = max(feasible, key=lambda row: row['parameters'], default=None)
        if selected is None or selected['candidate'] == 'baseline_053b':
            state.update(status='NO_LARGER_30_DAY_CANDIDATE', updated_at=bench.now())
            bench.publish(bench.HEALTH, state)
            return
        if selected['candidate'] != state['expected_candidate'] or selected['parameters'] != state['expected_parameters']:
            raise ValueError('fresh selection differs from the explicit four-group frontier')
        source_health = bench.ROOT / 'data/health' / health_files[selected['group']]
        source = json.loads(source_health.read_text())
        case = source['cases'][selected['variant']]
        if bench.digest(case['variant']['config']) != case['variant']['config_sha256']:
            raise ValueError('selected screening config changed')
        source_plan = Path(case['log']).parent / 'plan.json'
        if bench.digest(source_plan) != source['plan_sha256']:
            raise ValueError('selected screening plan changed')
        source_model = next(c for c in json.loads(source_plan.read_text())['candidates']
                            if c['name'] == case['candidate'])
        source_cfg = yaml.safe_load(Path(case['variant']['config']).read_text())
        model_file = Path(source_cfg['model']['configs'])
        model = yaml.safe_load(model_file.read_text())['presets'][source_cfg['model']['preset']]
        if bench.digest(model_file) != source_model['model_sha256']:
            raise ValueError('selected screening model changed')
        bench.STEPS = 200
        original_evaluate = bench.evaluate
        bench.evaluate = lambda result: original_evaluate(result, steps=200, skip_steps=50)
        bench.SHAPES = [('budget_sustained',
                         (model['num_layers'], model['hidden_size'], model['intermediate_size']))]
        plan = bench.prepare()
        if plan['source_qualification_identity'] != source['source_qualification_identity']:
            raise ValueError('training source changed after selected screening')
        candidate = plan['candidates'][0]
        Path(candidate['model_path']).write_bytes(model_file.read_bytes())
        candidate.update(shape=model, model_sha256=bench.digest(candidate['model_path']))
        import torch
        import sys
        sys.path.insert(0, str(bench.ROOT / 'model/src'))
        from yuformer_model.config import ModelConfig
        from yuformer_model.model import YuFormerModel
        with torch.device('meta'):
            counted_model = YuFormerModel(ModelConfig(**model))
        candidate['parameters'] = counted_model.num_parameters()
        del counted_model
        if candidate['parameters'] != selected['parameters']:
            raise ValueError('sustained model parameter count differs from selection')
        if candidate['model_sha256'] != bench.digest(model_file):
            raise ValueError('sustained model bytes differ from the selected architecture')
        variant = next(v for v in candidate['variants'] if v['microbatch'] == selected['microbatch'])
        config = yaml.safe_load(Path(variant['config']).read_text())
        config['optim']['warmup_steps'] = 10
        config['train'].update(save_every=100, eval_every=100)
        Path(variant['config']).write_text(yaml.safe_dump(config, sort_keys=False))
        variant['config_sha256'] = bench.digest(variant['config'])
        plan.update(selected_screening=selected, wrapper_tool_sha256=state['wrapper_tool_sha256'],
                    skip_steps=50, baseline_comparison='existing identity-bound 200-update production canary',
                    purpose=state['purpose'])
        bench.publish(bench.RESULTS / 'plan.json', plan)
        state.update(status='RUNNING', source_qualification_identity=plan['source_qualification_identity'],
                     plan_sha256=bench.digest(bench.RESULTS / 'plan.json'),
                     selected_screening=selected, cases={})
        bench.publish(bench.HEALTH, state)
        measured = bench.run_variant(candidate, variant, state)
        state['cases'][variant['name']] = measured
        if measured['evaluation']['status'] != 'PASS_SCREENING':
            raise RuntimeError('sustained selected-model training failed')
        baseline_path = bench.ROOT / 'data/experiments/production_canary/production_8gpu/result.json'
        baseline_report = json.loads((bench.ROOT / 'data/health/production-throughput-canary.json').read_text())
        if (baseline_report.get('status') != 'PASS'
                or baseline_report['result_sha256'] != bench.digest(baseline_path)
                or baseline_report['qualification_identity'] != plan['source_qualification_identity']):
            raise ValueError('baseline quality reference is stale or unbound')
        baseline = json.loads(baseline_path.read_text())
        result = json.loads((bench.RUNS / variant['name'] / 'result.json').read_text())
        quality = {}
        for tag, value in result['metrics'].items():
            if tag == 'validation_metrics/loss' or (tag.startswith('validation_metrics/domain/') and tag.endswith('/loss')):
                before, after = baseline['metrics'][tag]['last'], value['last']
                if not math.isfinite(before) or not math.isfinite(after):
                    raise ValueError('nonfinite validation comparison')
                quality[tag] = {'baseline': before, 'candidate': after, 'delta': after-before,
                                'relative_delta': (after-before)/before}
        if len(quality) != 10:
            raise ValueError('quality comparison requires aggregate and all nine domains')
        days = measured['evaluation']['projected_1t_days_duty_09']
        state.update(status='PASS_SUSTAINED_SCREENING' if days <= 33 else 'FAIL_RUNTIME_CEILING',
                     updated_at=bench.now(), evaluation=measured['evaluation'],
                     within_30_day_target=days <= 30, authorized_33_day_ceiling_met=days <= 33,
                     baseline_result_sha256=bench.digest(baseline_path), quality_comparison=quality,
                     aggregate_validation_improved=quality['validation_metrics/loss']['delta'] <= 0,
                     quality_review_required=(quality['validation_metrics/loss']['delta'] > 0
                         or any(v['relative_delta'] > .02 for v in quality.values())),
                     production_qualified=False,
                     note='Learning-curve comparison is diagnostic; formal model adoption/LR/recovery/identity gates still required.')
        if bench.digest(analysis.__file__) != analyzer_sha:
            raise ValueError('analyzer source changed during frontier measurement')
        if bench.digest(__file__) != state['wrapper_tool_sha256']:
            raise ValueError('sustained runner changed while measuring')
        bench.publish(bench.HEALTH, state)
        bench.publish(bench.RESULTS / 'summary.json', state)
    except BaseException as exc:
        state.update(status='BLOCKED', updated_at=bench.now(), error=f'{type(exc).__name__}: {exc}')
        bench.publish(bench.HEALTH, state)
        raise


if __name__ == '__main__':
    main()
