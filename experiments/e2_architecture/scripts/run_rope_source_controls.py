#!/usr/bin/env python3
"""Remeasure the exact three selected controls after the compiled RoPE backward fix."""
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

import yaml

import benchmark_model_sizes as bench
from research_control_identity import research_control_identity
import analyze_model_size_sweep as analysis
from benchmark_budget_lr_controls import require_matched_contract


def main():
    root = bench.ROOT
    original = root/'experiments/e2_architecture/results/flex_sustained_20261114/plan.json'
    previous = json.loads(original.read_bytes())
    bench.CONFIGS = root/'experiments/e2_architecture/configs/rope_source_controls_20261114'
    bench.RESULTS = root/'experiments/e2_architecture/results/rope_source_controls_20261114'
    bench.RUNS = root/'data/experiments/rope_source_controls_20261114'
    bench.HEALTH = root/'data/health/model-rope-source-controls.json'
    bench.CONFIGS.mkdir(parents=True, exist_ok=True)
    bench.RESULTS.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root/'pretrain/src'))
    sys.path.insert(0, str(root/'model/src'))
    from yuformer_pretrain.lease import RunLease
    from yuformer_model.config import ModelConfig
    from yuformer_model.model import YuFormerModel
    import torch
    deadline = time.monotonic()+600
    print('Waiting for eight actual current-source 4K attention-module parity reports', flush=True)
    while True:
        paths = [root/f'data/health/rope-fixed-4k-module-gpu{rank}.json' for rank in range(8)]
        if all(path.is_file() for path in paths):
            for rank, path in enumerate(paths):
                proof = json.loads(path.read_bytes())
                if (proof.get('status') != 'PASS' or proof.get('physical_gpu') != rank
                        or len(proof.get('cases', [])) != 4
                        or any(case.get('status') != 'PASS' for case in proof['cases'])
                        or any(bench.digest(root/name) != sha for name,sha in proof.get('sources', {}).items())
                        or set(proof.get('sources', {})) != {
                            'pretrain/backends/torchtitan/flex_module_parity.py',
                            'model/src/yuformer_model/attention.py',
                            'pretrain/backends/torchtitan/attention_parity.py'}):
                    raise ValueError('actual current-source 4K module parity is incomplete or failed: '+str(rank))
            break
        if time.monotonic() >= deadline:
            raise ValueError('actual 4K module parity did not complete; no research training launched')
        time.sleep(5)
    with RunLease(bench.RESULTS, 'matched-rope-fixed-source-model-controls'):
        candidates = []
        for old in previous['candidates']:
            candidate = copy.deepcopy(old)
            source_model = Path(old['model_path'])
            if bench.digest(source_model) != old['model_sha256']:
                raise ValueError('actual historically measured model bytes changed')
            model_path = bench.CONFIGS/source_model.name
            model_path.write_bytes(source_model.read_bytes())
            variant = copy.deepcopy(old['selected_variant'])
            for kind in ('config','env'):
                if bench.digest(variant[kind]) != variant[kind+'_sha256']:
                    raise ValueError('original matched-control execution contract changed')
            config = yaml.safe_load(Path(variant['config']).read_bytes())
            model = yaml.safe_load(model_path.read_bytes())['presets'][config['model']['preset']]
            with torch.device('meta'):
                instance = YuFormerModel(ModelConfig(**model))
            if instance.num_parameters() != candidate['parameters']:
                raise ValueError('fresh-control actual model parameter count changed')
            del instance
            if (config['data'].get('sampling_policy','sequential') != 'sequential'
                    or config['data'].get('block_permutation_manifest')
                    or config['train']['expected_iters'] != 200
                    or config['train']['budget_tokens'] != 838860800
                    or config['train']['micro_batch_size'] != 8 or config['train']['grad_accum'] != 16
                    or config['parallel']['world_size'] != 8
                    or config['parallel']['replicate_degree'] != 2 or config['parallel']['shard_degree'] != 4):
                raise ValueError('fresh control must preserve the actual matched research data/shape/budget')
            config['model']['configs'] = str(model_path)
            config_path = bench.CONFIGS/Path(variant['config']).name
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            values = dict(line.split('=',1) for line in Path(variant['env']).read_text().splitlines()
                          if line.strip() and not line.lstrip().startswith('#'))
            values.update(YUFORMER_CONFIG=str(config_path), YUFORMER_MODEL_CONFIGS=str(model_path),
                          YUFORMER_PRETRAIN_OUT=str(bench.RUNS/variant['name']), MASTER_PORT='29833')
            env_path = bench.CONFIGS/Path(variant['env']).name
            env_path.write_text(''.join(f'{key}={value}\n' for key,value in values.items()))
            variant.update(config=str(config_path),env=str(env_path),
                           config_sha256=bench.digest(config_path),env_sha256=bench.digest(env_path))
            candidate.update(model_path=str(model_path),model_sha256=bench.digest(model_path),
                             selected_variant=variant,variants=[variant])
            candidates.append(candidate)
        if [c['name'] for c in candidates] != ['matched_baseline','larger_lr1e4','larger_lr2e4']:
            raise ValueError('fresh matched-control roster differs from original scientific comparison')
        first, second = [yaml.safe_load(Path(c['selected_variant']['config']).read_bytes()) for c in candidates[1:]]
        require_matched_contract(first,second)
        bench.qualification_identity = lambda location: research_control_identity(location, bench.CONFIGS)
        current = bench.qualification_identity(root)
        pinned = {str(p):bench.digest(p) for p in (Path(__file__),Path(bench.__file__),Path(analysis.__file__),
                  Path(__file__).with_name('benchmark_budget_lr_controls.py'),
                  Path(__file__).with_name('research_control_identity.py'),original, *paths)}
        state = {'schema_version':1,'pid':os.getpid(),'status':'RUNNING','updated_at':bench.now(),
            'purpose':'FRESH_ROPE_FIXED_MATCHED_RESEARCH_CONTROLS_NOT_PRODUCTION_QUALIFICATION',
            'identity_scope':'matched_100b_research',
            'config_directory':str(bench.CONFIGS.relative_to(root)),
            'run_directory':str(bench.RUNS.relative_to(root)),
            'production_qualified':False,'data_preprocessing_must_continue':True,
            'source_qualification_identity':current,'pinned_inputs':pinned,'cases':{}}
        plan = {'schema_version':1,'purpose':state['purpose'],'candidates':candidates,
            'source_qualification_identity':current,'steps':200,'skip_steps':50,'world_size':8,
            'tokens_per_step':bench.TOKENS_PER_STEP,'original_plan_sha256':bench.digest(original),
            'wrapper_tool_sha256':bench.digest(__file__),'runner_tool_sha256':bench.digest(bench.__file__)}
        bench.publish(bench.RESULTS/'plan.json',plan)
        state['plan_sha256']=bench.digest(bench.RESULTS/'plan.json')
        evaluator=bench.evaluate
        bench.evaluate=lambda result:evaluator(result,steps=200,skip_steps=50)
        bench.publish(bench.HEALTH,state)
        try:
            for candidate in candidates:
                if any(bench.digest(name)!=sha for name,sha in pinned.items()):
                    raise ValueError('fresh matched-control source changed')
                deadline=time.monotonic()+600
                while True:
                    try:
                        bench.require_idle();break
                    except ValueError:
                        if time.monotonic()>=deadline:raise
                        time.sleep(5)
                variant=candidate['selected_variant']
                result=bench.run_variant(candidate,variant,state)
                result.update(analysis.memory_pressure(result['log'], skip_steps=50, final_step=201))
                state['cases'][variant['name']]=result
                bench.publish(bench.HEALTH,state)
                if result['evaluation']['status']!='PASS_SCREENING' or result['requires_smaller_batch_comparison']:
                    raise ValueError('actual fresh matched-control run failed; no historical result substituted')
            baseline=json.loads((bench.RUNS/'matched_baseline_b8/result.json').read_bytes())
            comparisons={}
            for name in ('larger_lr1e4_b8','larger_lr2e4_b8'):
                result=json.loads((bench.RUNS/name/'result.json').read_bytes());comparison={}
                for tag,value in result['metrics'].items():
                    if tag=='validation_metrics/loss' or tag.startswith('validation_metrics/domain/') and tag.endswith('/loss'):
                        before,after=baseline['metrics'][tag]['last'],value['last']
                        if not math.isfinite(before) or not math.isfinite(after):raise ValueError('nonfinite nine-domain validation')
                        comparison[tag]={'baseline':before,'candidate':after,'delta':after-before,'relative_delta':(after-before)/before}
                if len(comparison)!=10:raise ValueError('fresh comparison must cover all nine validation domains')
                comparisons[name]=comparison
            state.update(status='COMPLETE_SUSTAINED_CONTROLS',quality=comparisons,
                best_early_validation_variant=min(comparisons,key=lambda n:comparisons[n]['validation_metrics/loss']['candidate']),
                quality_review_required=True,updated_at=bench.now())
        except Exception as error:
            state.update(status='FAIL',error=repr(error),updated_at=bench.now());raise
        finally:
            bench.publish(bench.HEALTH,state)


if __name__=='__main__':
    main()
