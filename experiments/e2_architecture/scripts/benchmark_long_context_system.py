#!/usr/bin/env python3
"""Eight-GPU long-shape system screening; no long-capability admission."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

import yaml

import benchmark_model_sizes as bench
from analyze_model_size_sweep import memory_pressure


def write_once(path, content):
    if path.exists():
        if path.read_text() != content:
            raise ValueError('refusing to overwrite a different long-system contract: '+str(path))
    else:
        path.write_text(content)


def prepare():
    root = bench.ROOT
    bench.CONFIGS = root/'experiments/e2_architecture/configs/long_context_system_20261114'
    bench.RESULTS = root/'experiments/e2_architecture/results/long_context_system_20261114'
    bench.RUNS = root/'data/experiments/long_context_system_20261114'
    bench.HEALTH = root/'data/health/long-context-system-screening.json'
    bench.CONFIGS.mkdir(parents=True, exist_ok=True)
    bench.RESULTS.mkdir(parents=True, exist_ok=True)
    source = root/'experiments/e2_architecture/configs/window_source_controls_20261114/larger_lr1e4_b8.yaml'
    base = yaml.safe_load(source.read_bytes())
    model = yaml.safe_load(Path(base['model']['configs']).read_bytes())['presets'][base['model']['preset']]
    values = dict(line.split('=',1) for line in source.with_suffix('.env').read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith('#'))
    sys.path.insert(0,str(root/'model/src'))
    import torch
    from yuformer_model.config import ModelConfig
    from yuformer_model.model import YuFormerModel
    candidates = []
    for length, micro, accumulation in ((16384,2,16),(65536,1,8)):
        name = f'context_{length}'
        shape = copy.deepcopy(model)
        shape['max_seq_len'] = length
        model_path = bench.CONFIGS/(name+'.model.yaml')
        write_once(model_path,yaml.safe_dump({'presets':{'long-system':shape}},sort_keys=True))
        with torch.device('meta'):
            instance = YuFormerModel(ModelConfig(**shape))
        parameters = instance.num_parameters()
        if parameters != 1516600064 or length*micro*accumulation*8 != bench.TOKENS_PER_STEP:
            raise ValueError('long system probe changed model parameters or global token update')
        del instance
        variants = []
        for checkpointing in (False,True):
            variant = f'{name}_b{micro}_ac{int(checkpointing)}'
            config = copy.deepcopy(base)
            config.update(run_name=variant,experiment_status='qualification')
            config['model'] = {'preset':'long-system','configs':str(model_path)}
            config['data'].update(seq_len=length, val_samples_per_domain=8*micro)
            config['optim'].update(lr=1e-5,warmup_steps=4)
            config['train'].update(expected_iters=24,budget_tokens=24*bench.TOKENS_PER_STEP,
                micro_batch_size=micro,grad_accum=accumulation,activation_checkpointing=checkpointing,
                save_every=24,eval_every=24,log_every=1,keep_last_n=2)
            config_path = bench.CONFIGS/(variant+'.yaml')
            write_once(config_path,yaml.safe_dump(config,sort_keys=False))
            env = dict(values,YUFORMER_CONFIG=str(config_path),YUFORMER_MODEL_CONFIGS=str(model_path),
                       YUFORMER_PRETRAIN_OUT=str(bench.RUNS/variant),MASTER_PORT='29843')
            env_path = config_path.with_suffix('.env')
            write_once(env_path,''.join(f'{key}={value}\n' for key,value in env.items()))
            variants.append(dict(name=variant,microbatch=micro,grad_accum=accumulation,
                config=str(config_path),env=str(env_path),config_sha256=bench.digest(config_path),
                env_sha256=bench.digest(env_path),activation_checkpointing=checkpointing))
        candidates.append(dict(name=name,seq_len=length,parameters=parameters,model_path=str(model_path),
                               model_sha256=bench.digest(model_path),variants=variants))
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    args = parser.parse_args()
    candidates = prepare()
    if not args.run:
        print(json.dumps({'status':'PREPARED_NOT_RUN','lengths':[c['seq_len'] for c in candidates],
                          'production_qualified':False}),flush=True)
        return
    root = bench.ROOT
    controls = json.loads((root/'data/health/model-window-source-controls.json').read_bytes())
    selection = json.loads((root/'experiments/e2_architecture/results/model-scale-production-selection.json').read_bytes())
    if (controls.get('status') != 'COMPLETE_SUSTAINED_CONTROLS'
            or not selection['sustained_measurement']['run_dir'].startswith('data/experiments/window_source_controls_')):
        raise ValueError('fresh controls and actual selection refresh must finish before using GPUs for long shapes')
    sys.path.insert(0,str(root/'pretrain/src'))
    from yuformer_pretrain.lease import RunLease
    with RunLease(bench.RESULTS,'long-context-research-system-screening'):
        state=dict(schema_version=1,pid=os.getpid(),status='RUNNING',production_qualified=False,
            long_context_capability_qualified=False,data_preprocessing_must_continue=True,
            purpose='RESEARCH_100B_LONG_SHAPE_NUMERICS_MEMORY_AND_IO_ONLY',
            source_qualification_identity=bench.qualification_identity(root),
            wrapper_tool_sha256=bench.digest(__file__),cases={})
        bench.publish(bench.HEALTH,state)
        try:
            for candidate in candidates:
                passed = False
                for variant in candidate['variants']:
                    if bench.digest(__file__) != state['wrapper_tool_sha256']:
                        raise ValueError('long system screening source changed during run')
                    result = bench.run_variant(candidate,variant,state)
                    result.update(memory_pressure(result['log'],skip_steps=8,final_step=25))
                    state['cases'][variant['name']] = result
                    bench.publish(bench.HEALTH,state)
                    if (result['evaluation']['status']=='PASS_SCREENING'
                            and not result['requires_smaller_batch_comparison']):
                        passed = True
                        break
                    if result['evaluation']['status'] not in ('OOM','PASS_SCREENING'):
                        raise ValueError('long system runtime failed; inspect actual log before changing kernels')
                if not passed:
                    raise ValueError('long system shape did not fit even with checkpointing')
            state.update(status='PASS_LONG_SHAPE_SYSTEM_SCREENING',updated_at=bench.now())
        except Exception as error:
            state.update(status='BLOCKED_LONG_SHAPE_SYSTEM',error=repr(error),updated_at=bench.now())
            raise
        finally:
            bench.publish(bench.HEALTH,state)


if __name__=='__main__':
    main()
