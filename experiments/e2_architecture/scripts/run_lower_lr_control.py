"""Test a predeclared lower LR after the larger disjoint review fails Chinese."""
from pathlib import Path
import copy
import json
import os
import sys
import time

import yaml

import benchmark_model_sizes as bench
from research_control_identity import research_control_identity
from analyze_model_size_sweep import memory_pressure

ROOT=bench.ROOT
sys.path[:0]=[str(ROOT/'pretrain/src'),str(ROOT/'model/src'),
              str(ROOT/'experiments/e4_curriculum/scripts')]
from yuformer_pretrain.lease import RunLease
from sustained_memory_review import review_memory_log
from allocator_retry_review import require_no_allocator_retries

bench.CONFIGS=ROOT/'experiments/e2_architecture/configs/rope_source_controls_20261114/lower_lr_controls'
bench.RESULTS=ROOT/'experiments/e2_architecture/results/rope_source_controls_20261114/lower_lr_controls'
bench.RUNS=ROOT/'data/experiments/rope_source_controls_20261114/lower_lr_controls'
bench.HEALTH=ROOT/'data/health/lower-lr-control.json'


def main():
    bench.CONFIGS.mkdir(parents=True,exist_ok=True)
    bench.RESULTS.mkdir(parents=True,exist_ok=True)
    state=dict(pid=os.getpid(),status='WAITING_COMPLETE_DISJOINT_REVIEW',cases={},
               purpose='LOWER_LR_CONTROL_AFTER_CONFIRMED_CHINESE_REGRESSION',production_qualified=False)
    pinned={str(p):bench.digest(p) for p in (Path(__file__),Path(bench.__file__),
            Path(__file__).with_name('research_control_identity.py'),
            bench.RESULTS/'lower_lr_protocol.json',
            ROOT/'experiments/e4_curriculum/scripts/sustained_memory_review.py',
            ROOT/'experiments/e4_curriculum/scripts/allocator_retry_review.py')}
    state['loaded_source_inputs']=pinned
    with RunLease(bench.RESULTS,'predeclared-lower-lr-eight-gpu-control'):
        try:
            deadline=time.monotonic()+1800
            bench.publish(bench.HEALTH,state)
            while True:
                if any(bench.digest(path)!=expected for path,expected in pinned.items()):
                    raise ValueError('loaded lower-LR producer changed while waiting')
                review=json.loads((ROOT/'data/health/disjoint-validation-review.json').read_bytes())
                if review['status']=='PASS_DISJOINT_ALL_NINE_MEAN_IMPROVEMENT':
                    state.update(status='NOT_NEEDED_EXISTING_CANDIDATE_PASSED');return 0
                if review['status']=='FAIL_DISJOINT_ALL_NINE_MEAN_IMPROVEMENT':break
                if review['status'].startswith('FAIL_') or time.monotonic()>deadline:
                    raise ValueError('actual larger held-out review did not complete')
                time.sleep(5)
            source=ROOT/'data/experiments/rope_source_controls_20261114/larger_lr1e4_b8'
            config=yaml.safe_load((source/'resolved_config.yaml').read_bytes())
            identity=json.loads((source/'run_identity.json').read_bytes())
            source_model=Path(identity['model_configs'])
            if bench.digest(source_model)!=identity['model_configs_sha256']:
                raise ValueError('actual selected architecture changed')
            name='larger_lr8e5_b8'
            mp=bench.CONFIGS/(name+'.model.yaml')
            cp=bench.CONFIGS/(name+'.yaml')
            ep=bench.CONFIGS/(name+'.env')
            if any(p.exists() for p in (mp,cp,ep)) or (bench.RUNS/name).exists():
                raise ValueError('lower-LR control must use new immutable outputs')
            mp.write_bytes(source_model.read_bytes())
            config['run_name']=name
            config['out_dir']='${YUFORMER_PRETRAIN_OUT}'
            config['model']['configs']=str(mp)
            config['optim']['lr']=.00008
            if (config['train']['expected_iters']!=200 or config['train']['budget_tokens']!=838860800
                    or config['train']['micro_batch_size']!=8 or config['train']['grad_accum']!=16
                    or config['parallel']['world_size']!=8):
                raise ValueError('lower-LR control must preserve the matched architecture/data/update budget')
            cp.write_text(yaml.safe_dump(config,sort_keys=False))
            original=ROOT/'experiments/e2_architecture/configs/rope_source_controls_20261114/larger_lr1e4_b8.env'
            env=dict(line.split('=',1) for line in original.read_text().splitlines()
                     if line.strip() and not line.lstrip().startswith('#'))
            env.update(YUFORMER_CONFIG=str(cp),YUFORMER_MODEL_CONFIGS=str(mp),
                       YUFORMER_PRETRAIN_OUT=str(bench.RUNS/name),MASTER_PORT='29893')
            ep.write_text(''.join(f'{key}={value}\n' for key,value in env.items()))
            bench.qualification_identity=lambda root:research_control_identity(root,bench.CONFIGS)
            state.update(status='RUNNING',source_qualification_identity=bench.qualification_identity(ROOT))
            candidate=dict(name='larger_lr8e5',parameters=1516600064,
                           model_path=str(mp),model_sha256=bench.digest(mp))
            variant=dict(name=name,config=str(cp),env=str(ep),config_sha256=bench.digest(cp),
                         env_sha256=bench.digest(ep),microbatch=8,grad_accum=16)
            bench.publish(bench.RESULTS/'plan.json',dict(candidate=candidate,variant=variant,
                          source_inputs=pinned,protocol='lower_lr_protocol.json'))
            evaluate=bench.evaluate
            bench.evaluate=lambda result:evaluate(result,steps=200,skip_steps=50)
            deadline=time.monotonic()+600
            while True:
                try:bench.require_idle();break
                except ValueError:
                    if time.monotonic()>deadline:raise
                    time.sleep(5)
            result=bench.run_variant(candidate,variant,state)
            if any(bench.digest(path)!=expected for path,expected in pinned.items()):
                raise ValueError('actual lower-LR producer or policy changed during training')
            result.update(memory_pressure(result['log'],skip_steps=50,final_step=201))
            raw=Path(result['log']).read_bytes()
            review_memory_log(raw,steps=200,skip_steps=50)
            require_no_allocator_retries(raw)
            state['cases'][name]=result
            if result['evaluation']['status']!='PASS_SCREENING' or result['requires_smaller_batch_comparison']:
                raise ValueError('actual lower-LR control failed; no scientific admission')
            state.update(status='COMPLETE_REQUIRES_PRIMARY_AND_LARGER_HELDOUT_REVIEW')
            return 0
        except Exception as error:
            state.update(status='FAIL_CLOSED_LOWER_LR_CONTROL',error=repr(error));raise
        finally:
            state['updated_at']=bench.now();bench.publish(bench.HEALTH,state)


if __name__=='__main__':raise SystemExit(main())
