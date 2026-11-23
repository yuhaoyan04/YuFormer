#!/usr/bin/env python3
"""Matched 1.2B learning-rate controls, only if its sustained quality needs review."""
from pathlib import Path
import fcntl
import json
import math
import os
import time

import yaml
import benchmark_model_sizes as bench


def require_matched_contract(candidate, control):
    normalized = []
    for document in (candidate, control):
        value = json.loads(json.dumps(document))
        value['run_name'] = '<LR-control-run>'
        value['model']['configs'] = '<same-verified-model-bytes>'
        value['optim']['lr'] = '<only-independent-variable>'
        normalized.append(value)
    if normalized[0] != normalized[1]:
        raise ValueError('LR controls differ in fields other than peak learning rate/run/model-copy path')


def quality(result, control):
    tags = [tag for tag in control['metrics'] if tag == 'validation_metrics/loss' or
            tag.startswith('validation_metrics/domain/') and tag.endswith('/loss')]
    if len(tags) != 10:
        raise ValueError('LR controls require aggregate and all nine validation domains')
    comparison = {}
    for tag in tags:
        before, after = control['metrics'][tag]['last'], result['metrics'][tag]['last']
        if not math.isfinite(before) or not math.isfinite(after):
            raise ValueError('LR control has a nonfinite validation loss')
        comparison[tag] = {'control': before, 'candidate': after, 'delta': after-before}
    return comparison


def main():
    bench.CONFIGS = bench.ROOT/'experiments/e2_architecture/configs/budget_lr_controls_20261114'
    bench.RESULTS = bench.ROOT/'experiments/e2_architecture/results/budget_lr_controls_20261114'
    bench.RUNS = bench.ROOT/'data/experiments/budget_lr_controls_20261114'
    bench.HEALTH = bench.ROOT/'data/health/model-budget-lr-controls.json'
    bench.RESULTS.mkdir(parents=True, exist_ok=True)
    lock = (bench.RESULTS/'.orchestrator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'status': 'WAITING_BUDGET_SUSTAINED', 'pid': os.getpid(), 'updated_at': bench.now(),
             'purpose': 'MATCHED_LR_QUALITY_CONTROLS_NOT_PRODUCTION_QUALIFICATION',
             'production_qualified': False, 'data_preprocessing_must_continue': True,
             'wrapper_tool_sha256': bench.digest(__file__),
             'loaded_benchmark_tool_sha256': bench.digest(bench.__file__),
             'peak_learning_rates': [0.0001, 0.0003]}
    bench.publish(bench.HEALTH, state)
    try:
        parent_path = bench.ROOT/'data/health/model-budget-sustained-canary.json'
        while True:
            parent = json.loads(parent_path.read_text())
            if parent['status'] == 'PASS_SUSTAINED_SCREENING':
                break
            if parent['status'] != 'RUNNING':
                raise ValueError('parent model canary needs repair: '+parent['status'])
            state['updated_at'] = bench.now()
            bench.publish(bench.HEALTH, state)
            time.sleep(30)
        if not parent.get('quality_review_required'):
            state.update(status='NOT_NEEDED_QUALITY_REVIEW_CLEAR', updated_at=bench.now())
            bench.publish(bench.HEALTH, state)
            return
        if bench.digest(bench.__file__) != state['loaded_benchmark_tool_sha256']:
            raise ValueError('loaded benchmark source changed while LR controls were queued')
        case = parent['cases']['budget_sustained_b8']
        if case['parameters'] != 1199628544 or case['evaluation']['status'] != 'PASS_SCREENING':
            raise ValueError('LR control model is not the verified 1.2B candidate')
        control_path = bench.ROOT/'data/experiments/budget_sustained_20261114/budget_sustained_b8/result.json'
        if bench.digest(control_path) != case['result_sha256']:
            raise ValueError('2e-4 learning-rate control result changed')
        control = json.loads(control_path.read_text())
        source_config = yaml.safe_load(Path(case['variant']['config']).read_text())
        if bench.digest(case['variant']['config']) != case['variant']['config_sha256']:
            raise ValueError('2e-4 control config changed')
        source_model = Path(source_config['model']['configs'])
        parent_plan_path = bench.ROOT/'experiments/e2_architecture/results/budget_sustained_20261114/plan.json'
        if bench.digest(parent_plan_path) != parent['plan_sha256']:
            raise ValueError('2e-4 control plan changed')
        source_candidate = json.loads(parent_plan_path.read_text())['candidates'][0]
        if bench.digest(source_model) != source_candidate['model_sha256']:
            raise ValueError('2e-4 control model bytes changed')
        model_bytes = source_model.read_bytes()
        shape = yaml.safe_load(model_bytes)['presets'][source_config['model']['preset']]
        if source_config['optim']['lr'] != .0002:
            raise ValueError('LR control does not have the expected peak 2e-4')
        bench.STEPS = 200
        evaluator = bench.evaluate
        bench.evaluate = lambda result: evaluator(result, steps=200, skip_steps=50)
        bench.SHAPES = [(name, (shape['num_layers'], shape['hidden_size'], shape['intermediate_size']))
                        for name in ('lr_1e4', 'lr_3e4')]
        plan = bench.prepare()
        if plan['source_qualification_identity'] != parent['source_qualification_identity']:
            raise ValueError('source identity changed after the 2e-4 control')
        import sys
        import torch
        sys.path.insert(0, str(bench.ROOT/'model/src'))
        from yuformer_model.config import ModelConfig
        from yuformer_model.model import YuFormerModel
        with torch.device('meta'):
            instance = YuFormerModel(ModelConfig(**shape))
        parameters = instance.num_parameters()
        del instance
        if parameters != 1199628544:
            raise ValueError('actual LR-control architecture parameter count mismatch')
        for candidate, lr in zip(plan['candidates'], state['peak_learning_rates']):
            Path(candidate['model_path']).write_bytes(model_bytes)
            candidate.update(parameters=parameters, shape=shape, model_sha256=bench.digest(candidate['model_path']))
            variant = candidate['variants'][0]
            config = yaml.safe_load(Path(variant['config']).read_text())
            config['optim'].update(lr=lr, warmup_steps=10)
            config['train'].update(save_every=100, eval_every=100)
            require_matched_contract(config, source_config)
            Path(variant['config']).write_text(yaml.safe_dump(config, sort_keys=False))
            variant['config_sha256'] = bench.digest(variant['config'])
            candidate['selected_variant'] = variant
            candidate['peak_lr'] = lr
        plan.update(purpose=state['purpose'], control_result_sha256=case['result_sha256'],
                    control_health_sha256=bench.digest(parent_path),
                    control_model_sha256=source_candidate['model_sha256'],
                    wrapper_tool_sha256=state['wrapper_tool_sha256'])
        bench.publish(bench.RESULTS/'plan.json', plan)
        state.update(status='RUNNING', cases={}, source_qualification_identity=plan['source_qualification_identity'],
                     plan_sha256=bench.digest(bench.RESULTS/'plan.json'),
                     control_result_sha256=case['result_sha256'])
        bench.publish(bench.HEALTH, state)
        for candidate in plan['candidates']:
            result = bench.run_variant(candidate, candidate['selected_variant'], state)
            state['cases'][candidate['name']] = result
            if result['evaluation']['status'] != 'PASS_SCREENING':
                raise ValueError('LR control needs repair: '+candidate['name'])
            path = bench.RUNS/candidate['selected_variant']['name']/'result.json'
            compared = quality(json.loads(path.read_text()), control)
            result.update(peak_lr=candidate['peak_lr'], quality_vs_2e4=compared,
                          aggregate_validation_improved=compared['validation_metrics/loss']['delta'] <= 0)
            state['updated_at'] = bench.now()
            bench.publish(bench.HEALTH, state)
        if (bench.digest(__file__) != state['wrapper_tool_sha256'] or
                bench.digest(bench.__file__) != state['loaded_benchmark_tool_sha256'] or
                bench.digest(source_model) != source_candidate['model_sha256']):
            raise ValueError('LR control orchestration source changed')
        state.update(status='COMPLETE_LR_QUALITY_CONTROLS', updated_at=bench.now(),
                     note='Learning-rate diagnostics only; production model/source/freeze/recovery qualification remains required.')
        bench.publish(bench.HEALTH, state)
        bench.publish(bench.RESULTS/'summary.json', state)
    except BaseException as exc:
        state.update(status='BLOCKED', updated_at=bench.now(), error=f'{type(exc).__name__}: {exc}')
        bench.publish(bench.HEALTH, state)
        raise


if __name__ == '__main__':
    main()
