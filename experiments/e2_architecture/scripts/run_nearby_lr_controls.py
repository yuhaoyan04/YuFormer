"""Run the two predeclared nearby learning rates without changing production."""
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

ROOT = bench.ROOT
sys.path[:0] = [str(ROOT/'pretrain/src'), str(ROOT/'model/src'),
                str(ROOT/'experiments/e4_curriculum/scripts')]
from yuformer_pretrain.lease import RunLease
from sustained_memory_review import review_memory_log
from allocator_retry_review import require_no_allocator_retries

BASE = ROOT/'experiments/e2_architecture/results/rope_source_controls_20261114'
bench.CONFIGS = ROOT/'experiments/e2_architecture/configs/rope_source_controls_20261114/nearby_lr_controls'
bench.RESULTS = BASE/'nearby_lr_controls'
bench.RUNS = ROOT/'data/experiments/rope_source_controls_20261114/nearby_lr_controls'
bench.HEALTH = ROOT/'data/health/nearby-lr-controls.json'
PROTOCOL = bench.RESULTS/'protocol.json'


def prepare(protocol):
    source = ROOT/'data/experiments/rope_source_controls_20261114/larger_lr1e4_b8'
    original = yaml.safe_load((source/'resolved_config.yaml').read_bytes())
    identity = json.loads((source/'run_identity.json').read_bytes())
    model = Path(identity['model_configs'])
    if (bench.digest(model) != identity['model_configs_sha256']
            or identity['source_sha256'] != protocol['training_source_sha256']):
        raise ValueError('actual model/code differs from the predeclared nearby-LR study')
    original_env = ROOT/'experiments/e2_architecture/configs/rope_source_controls_20261114/larger_lr1e4_b8.env'
    variables = dict(line.split('=', 1) for line in original_env.read_text().splitlines()
                     if line.strip() and not line.lstrip().startswith('#'))
    variants = []
    for index, row in enumerate(protocol['controls']):
        name = row['case']
        mp, cp, ep = [bench.CONFIGS/(name+suffix) for suffix in ('.model.yaml', '.yaml', '.env')]
        if any(path.exists() for path in (mp, cp, ep)) or (bench.RUNS/name).exists():
            raise ValueError('nearby-LR outputs must be new: '+name)
        config = copy.deepcopy(original)
        config.update(run_name=name, out_dir='${YUFORMER_PRETRAIN_OUT}')
        config['model']['configs'] = str(mp)
        config['optim']['lr'] = row['peak_lr']
        if (config['train']['expected_iters'] != 200 or config['train']['budget_tokens'] != 838860800
                or config['train']['micro_batch_size'] != 8 or config['train']['grad_accum'] != 16
                or config['parallel']['world_size'] != 8):
            raise ValueError('nearby-LR study changed the matched training budget')
        mp.write_bytes(model.read_bytes())
        cp.write_text(yaml.safe_dump(config, sort_keys=False))
        env = dict(variables, YUFORMER_CONFIG=str(cp), YUFORMER_MODEL_CONFIGS=str(mp),
                   YUFORMER_PRETRAIN_OUT=str(bench.RUNS/name), MASTER_PORT=str(29901+index))
        ep.write_text(''.join(f'{key}={value}\n' for key, value in env.items()))
        candidate = dict(name=name, parameters=1516600064, model_path=str(mp), model_sha256=bench.digest(mp))
        variant = dict(name=name, microbatch=8, grad_accum=16, config=str(cp), env=str(ep),
                       config_sha256=bench.digest(cp), env_sha256=bench.digest(ep))
        variants.append((candidate, variant))
    return variants


def main():
    bench.CONFIGS.mkdir(parents=True, exist_ok=True)
    bench.RESULTS.mkdir(parents=True, exist_ok=True)
    state = dict(pid=os.getpid(), status='WAITING_READONLY_LOWER_LR_VALIDATION', cases={},
                 production_qualified=False, formal_pretrain_started=False)
    pinned = {str(path):bench.digest(path) for path in (Path(__file__), Path(bench.__file__),
        Path(__file__).with_name('research_control_identity.py'),
        Path(__file__).with_name('analyze_model_size_sweep.py'), PROTOCOL,
        ROOT/'experiments/e4_curriculum/scripts/sustained_memory_review.py',
        ROOT/'experiments/e4_curriculum/scripts/allocator_retry_review.py')}
    state['loaded_source_inputs'] = pinned
    with RunLease(bench.RESULTS, 'predeclared-nearby-learning-rate-controls'):
        try:
            bench.publish(bench.HEALTH, state)
            deadline = time.monotonic()+1800
            while True:
                if any(bench.digest(path) != value for path, value in pinned.items()):
                    raise ValueError('nearby-LR producer changed while waiting')
                previous = json.loads((ROOT/'data/health/lower-lr-validation.json').read_bytes())
                if previous['status'] == 'COMPLETE_REQUIRES_QUALITY_COMPARISON':
                    break
                if previous['status'].startswith('FAIL') or time.monotonic() > deadline:
                    raise ValueError('the earlier actual validation did not complete')
                time.sleep(5)
            protocol = json.loads(PROTOCOL.read_bytes())
            variants = prepare(protocol)
            bench.qualification_identity = lambda root:research_control_identity(root, bench.CONFIGS)
            state.update(status='RUNNING', source_qualification_identity=bench.qualification_identity(ROOT))
            bench.publish(bench.RESULTS/'plan.json', dict(protocol_sha256=bench.digest(PROTOCOL),
                          controls=variants, source_inputs=pinned,
                          source_qualification_identity=state['source_qualification_identity']))
            evaluate = bench.evaluate
            bench.evaluate = lambda result:evaluate(result, steps=200, skip_steps=50)
            for candidate, variant in variants:
                if any(bench.digest(path) != value for path, value in pinned.items()):
                    raise ValueError('nearby-LR producer changed before training')
                deadline = time.monotonic()+600
                while True:
                    try:
                        bench.require_idle()
                        break
                    except ValueError:
                        if time.monotonic() > deadline:
                            raise
                        time.sleep(5)
                result = bench.run_variant(candidate, variant, state)
                result.update(memory_pressure(result['log'], skip_steps=50, final_step=201))
                raw = Path(result['log']).read_bytes()
                review_memory_log(raw, steps=200, skip_steps=50)
                require_no_allocator_retries(raw)
                state['cases'][variant['name']] = result
                if (result['evaluation']['status'] != 'PASS_SCREENING'
                        or not result['evaluation']['within_33_days']
                        or result['requires_smaller_batch_comparison']):
                    raise ValueError('nearby-LR actual system/throughput gate failed')
                bench.publish(bench.HEALTH, state)
            if any(bench.digest(path) != value for path, value in pinned.items()):
                raise ValueError('nearby-LR producer changed during training')
            state['status'] = 'COMPLETE_REQUIRES_PRIMARY_AND_LARGER_DEVELOPMENT_REVIEW'
        except Exception as error:
            state.update(status='FAIL_CLOSED_NEARBY_LR_CONTROLS', error=repr(error))
            raise
        finally:
            state['updated_at'] = bench.now()
            bench.publish(bench.HEALTH, state)


if __name__ == '__main__':
    main()
