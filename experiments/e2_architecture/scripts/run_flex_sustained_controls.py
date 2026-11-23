#!/usr/bin/env python3
"""Matched 200-update baseline and learning-rate controls for the hybrid frontier."""
import json
import math
import os
from pathlib import Path
import time

import yaml

import benchmark_model_sizes as bench
import analyze_model_size_sweep as analysis


def main():
    bench.CONFIGS = bench.ROOT/'experiments/e2_architecture/configs/flex_sustained_20261114'
    bench.RESULTS = bench.ROOT/'experiments/e2_architecture/results/flex_sustained_20261114'
    bench.RUNS = bench.ROOT/'data/experiments/flex_sustained_20261114'
    bench.HEALTH = bench.ROOT/'data/health/model-flex-sustained-controls.json'
    bench.RESULTS.mkdir(parents=True, exist_ok=True)
    import sys
    sys.path.insert(0, str(bench.ROOT/'pretrain/src'))
    from yuformer_pretrain.lease import RunLease
    with RunLease(bench.RESULTS, 'matched-hybrid-baseline-and-lr-controls'):
        pinned = {str(p): bench.digest(p) for p in
            (Path(__file__), Path(bench.__file__), Path(analysis.__file__),
             Path(__file__).with_name('benchmark_budget_lr_controls.py'),
             bench.ROOT/'model/configs/pretrain_final_model_configs.yaml',
             bench.ROOT/'pretrain/configs/production_canary/production_8gpu.yaml',
             bench.ROOT/'pretrain/configs/production_canary/production_8gpu.env')}
        queued_identity = bench.qualification_identity(bench.ROOT)
        state = dict(status='WAITING_HYBRID_FRONTIER', pid=os.getpid(), updated_at=bench.now(),
            purpose='MATCHED_HYBRID_SUSTAINED_QUALITY_CONTROLS_NOT_PRODUCTION_QUALIFICATION',
            production_qualified=False, data_preprocessing_must_continue=True,
            queued_source_identity=queued_identity['sha256'], pinned_inputs=pinned)
        bench.publish(bench.HEALTH, state)
        parent_path = bench.ROOT/'data/health/model-flex-frontier.json'
        while True:
            parent = json.loads(parent_path.read_bytes())
            if parent['status'] == 'COMPLETE_SCREENING':
                break
            if parent['status'] != 'RUNNING':
                raise ValueError('hybrid frontier needs repair: '+parent['status'])
            if any(bench.digest(p) != sha for p, sha in pinned.items()):
                raise ValueError('queued sustained-control source/config changed')
            state['updated_at'] = bench.now()
            bench.publish(bench.HEALTH, state)
            time.sleep(15)
        if bench.qualification_identity(bench.ROOT) != queued_identity:
            raise ValueError('training graph changed while sustained controls were queued')
        eligible = [r for r in parent['cases'].values() if r['evaluation']['status'] == 'PASS_SCREENING'
            and r['evaluation']['within_33_days'] and not r['requires_smaller_batch_comparison']
            and r['variant']['microbatch'] == 8 and 10**9 <= r['parameters'] <= 5*10**9]
        selected = max(eligible, key=lambda r: r['parameters'], default=None)
        if selected is None:
            raise ValueError('no measured larger model meets the preserved B8/A16 budget')
        parent_plan = Path(selected['log']).parent/'plan.json'
        if bench.digest(parent_plan) != parent['plan_sha256']:
            raise ValueError('hybrid parent plan changed')
        source = next(r for r in json.loads(parent_plan.read_bytes())['candidates'] if r['name'] == selected['candidate'])
        source_model = Path(source['model_path'])
        if bench.digest(source_model) != source['model_sha256']:
            raise ValueError('selected larger model bytes changed')
        model = source['shape']
        bench.STEPS = 200
        evaluator = bench.evaluate
        bench.evaluate = lambda result: evaluator(result, steps=200, skip_steps=50)
        shape = (model['num_layers'], model['hidden_size'], model['intermediate_size'])
        bench.SHAPES = [('matched_baseline', None), ('larger_lr1e4', shape), ('larger_lr2e4', shape)]
        plan = bench.prepare()
        if plan['source_qualification_identity'] != queued_identity:
            raise ValueError('training graph changed before sustained preparation')
        for index, candidate in enumerate(plan['candidates']):
            if index == 0:
                if candidate['parameters'] != 530107904:
                    raise ValueError('matched baseline is not the original measured 0.53B architecture')
            else:
                if candidate['parameters'] != source['parameters']:
                    raise ValueError('actual constructed model parameter count differs from the selected frontier')
                Path(candidate['model_path']).write_bytes(source_model.read_bytes())
                candidate.update(shape=model, parameters=source['parameters'], model_sha256=bench.digest(candidate['model_path']))
            variant = candidate['variants'][0]
            config = yaml.safe_load(Path(variant['config']).read_bytes())
            config['attention']['sdpa_backend'] = 'cudnn_flex_swa'
            config['optim'].update(lr=.0001 if index == 1 else .0002, warmup_steps=10)
            config['train'].update(eval_every=100, save_every=100)
            Path(variant['config']).write_text(yaml.safe_dump(config, sort_keys=False))
            variant['config_sha256'] = bench.digest(variant['config'])
            candidate['selected_variant'] = variant
        first = yaml.safe_load(Path(plan['candidates'][1]['selected_variant']['config']).read_bytes())
        second = yaml.safe_load(Path(plan['candidates'][2]['selected_variant']['config']).read_bytes())
        from benchmark_budget_lr_controls import require_matched_contract
        require_matched_contract(first, second)
        plan.update(purpose=state['purpose'], parent_plan_sha256=bench.digest(parent_plan),
            wrapper_tool_sha256=bench.digest(__file__), selected_candidate=selected, skip_steps=50,
            control_contract_checker_sha256=bench.digest(Path(__file__).with_name('benchmark_budget_lr_controls.py')))
        bench.publish(bench.RESULTS/'plan.json', plan)
        state.update(status='RUNNING', source_qualification_identity=queued_identity, cases={},
            plan_sha256=bench.digest(bench.RESULTS/'plan.json'), selected_screening=selected)
        bench.publish(bench.HEALTH, state)
        try:
            for candidate in plan['candidates']:
                deadline = time.monotonic()+600
                while True:
                    try:
                        bench.require_idle()
                        break
                    except ValueError:
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(5)
                variant = candidate['selected_variant']
                result = bench.run_variant(candidate, variant, state)
                result.update(analysis.memory_pressure(result['log']))
                state['cases'][variant['name']] = result
                bench.publish(bench.HEALTH, state)
                if result['evaluation']['status'] != 'PASS_SCREENING' or result['requires_smaller_batch_comparison']:
                    raise ValueError('sustained model training requires repair')
            baseline = json.loads((bench.RUNS/'matched_baseline_b8/result.json').read_bytes())
            comparisons = {}
            for name in ('larger_lr1e4_b8', 'larger_lr2e4_b8'):
                result = json.loads((bench.RUNS/name/'result.json').read_bytes())
                comparison = {}
                for tag, value in result['metrics'].items():
                    if tag == 'validation_metrics/loss' or tag.startswith('validation_metrics/domain/') and tag.endswith('/loss'):
                        before, after = baseline['metrics'][tag]['last'], value['last']
                        if not math.isfinite(before) or not math.isfinite(after):
                            raise ValueError('nonfinite sustained validation')
                        comparison[tag] = dict(baseline=before, candidate=after, delta=after-before,
                                               relative_delta=(after-before)/before)
                if len(comparison) != 10:
                    raise ValueError('sustained comparison must cover all nine validation domains')
                comparisons[name] = comparison
            best = min(comparisons, key=lambda n: comparisons[n]['validation_metrics/loss']['candidate'])
            if any(bench.digest(p) != sha for p, sha in pinned.items()):
                raise ValueError('sustained-control inputs changed during measurement')
            state.update(status='COMPLETE_SUSTAINED_CONTROLS', quality=comparisons,
                best_early_validation_variant=best, quality_review_required=True,
                production_qualified=False, updated_at=bench.now())
        except Exception as exc:
            state.update(status='FAIL', error=repr(exc), updated_at=bench.now())
            raise
        finally:
            bench.publish(bench.HEALTH, state)


if __name__ == '__main__':
    main()
