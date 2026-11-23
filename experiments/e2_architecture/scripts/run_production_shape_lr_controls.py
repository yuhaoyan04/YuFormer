"""Run a finite, predeclared architecture/LR grid on actual production samples."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml

import benchmark_model_sizes as bench
from benchmark_budget_lr_controls import require_matched_contract
from development_validation_review import compare, load_evaluation
from evaluate_disjoint_validation import checkpoint_identity

ROOT = bench.ROOT
sys.path[:0] = [str(ROOT/'pretrain/scripts'), str(ROOT/'pretrain/src'),
                str(ROOT/'model/src'), str(ROOT/'experiments/e4_curriculum/scripts')]
from audit_training_progress import audit, verify_reference_schedule
from checkpoint_durability_evidence import require_checkpoint_directory_durability
from freeze_1t_recipe import source_sha256
from qualification_identity import qualification_identity
from sustained_memory_review import review_memory_log
from allocator_retry_review import require_no_allocator_retries
from yuformer_pretrain.lease import RunLease

COHORT = 'production_shape_lr_controls_20261115_authorized'
RESULTS = ROOT/'experiments/e2_architecture/results'/COHORT
CONFIGS = ROOT/'experiments/e2_architecture/configs'/COHORT
RUNS = ROOT/'data/experiments'/COHORT
HEALTH = ROOT/'data/health/production-shape-lr-controls.json'
PROTOCOL = RESULTS/'protocol.json'
REFERENCE = ROOT/'experiments/e2_architecture/results/production_matched_controls_20261115'
REFERENCE_RUNS = ROOT/'data/experiments/production_matched_controls_20261115'
PYTHON = str(bench.PYTHON)


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2)+'\n').encode()


def declare():
    if RESULTS.exists() or CONFIGS.exists() or RUNS.exists():
        raise ValueError('preserve every earlier attempt; this declaration is write-once')
    baseline_protocol, baseline, bound = load_evaluation(
        ROOT, REFERENCE/'validation/plan.json', 'matched_baseline_b8')
    failed = json.loads((REFERENCE/'validation/review.json').read_bytes())
    if failed['status'] != 'FAIL_PRODUCTION_PRIMARY_OR_EXPANDED_ALL_NINE':
        raise ValueError('this finite followup requires the actual prior quality failure')
    current = qualification_identity(ROOT)
    if source_sha256(ROOT) != baseline_protocol['training_source_sha256']:
        raise ValueError('actual comparison source changed')
    bound.update(require_checkpoint_directory_durability(ROOT))
    for path in (Path(__file__), Path(bench.__file__),
                 REFERENCE/'protocol.json', REFERENCE/'validation/review.json',
                 ROOT/'data/health/production-matched-controls.json',
                 Path(__file__).with_name('benchmark_budget_lr_controls.py'),
                 Path(__file__).with_name('development_validation_review.py'),
                 Path(__file__).with_name('evaluate_disjoint_validation.py'),
                 ROOT/'pretrain/scripts/audit_training_progress.py',
                 ROOT/'experiments/e4_curriculum/scripts/sustained_memory_review.py',
                 ROOT/'experiments/e4_curriculum/scripts/allocator_retry_review.py'):
        bound[str(path.relative_to(ROOT))] = bench.digest(path)
    RESULTS.mkdir(parents=True)
    CONFIGS.mkdir(parents=True)
    template = yaml.safe_load((REFERENCE_RUNS/'matched_baseline_b8/resolved_config.yaml').read_bytes())
    env_path = ROOT/'experiments/e2_architecture/configs/production_matched_controls_20261115/matched_baseline_b8.env'
    env = dict(line.split('=', 1) for line in env_path.read_text().splitlines()
               if line.strip() and not line.lstrip().startswith('#'))
    bound[str(env_path.relative_to(ROOT))] = bench.digest(env_path)
    cases = []
    for family, shape_name, parameters in [('flex_frontier_20261114', 'flex_14l_h2560_g6', 1219045632),
                                          ('gpu_size_sweep_20261114', 'size_1b_deep', 1002005248)]:
        if not 10**9 <= parameters <= 5*10**9:
            raise ValueError('every production candidate must satisfy the exact authorized1B-5B range')
        shape_path = ROOT/'experiments/e2_architecture/configs'/family/(shape_name+'.model.yaml')
        shape = yaml.safe_load(shape_path.read_bytes())['presets']['size-sweep']
        bound[str(shape_path.relative_to(ROOT))] = bench.digest(shape_path)
        import torch
        from yuformer_model.config import ModelConfig
        from yuformer_model.model import YuFormerModel
        with torch.device('meta'):
            instance = YuFormerModel(ModelConfig(**shape))
        if instance.num_parameters() != parameters:
            raise ValueError('actual candidate parameter count differs')
        optimizer_groups = len(list(instance.parameters()))
        del instance
        for lr_index, peak_lr in enumerate((.0001, .00015)):
            case = shape_name+('_lr1e4_b8' if lr_index == 0 else '_lr15e5_b8')
            mp, cp, ep = [CONFIGS/(case+suffix) for suffix in ('.model.yaml', '.yaml', '.env')]
            mp.write_text(yaml.safe_dump({'presets': {'production-control': shape}}, sort_keys=True))
            config = copy.deepcopy(template)
            config.update(run_name=case, out_dir='${YUFORMER_PRETRAIN_OUT}')
            config['model'] = dict(preset='production-control', configs=str(mp))
            config['optim']['lr'] = peak_lr
            normalized = copy.deepcopy(config)
            normalized['out_dir'] = template['out_dir']
            require_matched_contract(normalized, template)
            cp.write_text(yaml.safe_dump(config, sort_keys=False))
            variables = dict(env, YUFORMER_CONFIG=str(cp), YUFORMER_MODEL_CONFIGS=str(mp),
                             YUFORMER_PRETRAIN_OUT=str(RUNS/case), MASTER_PORT=str(29941+len(cases)))
            ep.write_text(''.join(f'{key}={value}\n' for key, value in variables.items()))
            for path in (mp, cp, ep):
                bound[str(path.relative_to(ROOT))] = bench.digest(path)
            cases.append(dict(name=case, parameters=parameters, peak_lr=peak_lr,
                optimizer_groups=optimizer_groups,
                model_path=str(mp), model_sha256=bench.digest(mp),
                variant=dict(name=case, microbatch=8, grad_accum=16, config=str(cp), env=str(ep),
                             config_sha256=bench.digest(cp), env_sha256=bench.digest(ep))))
    declaration = dict(schema_version=1, status='PREDECLARED_BEFORE_ANY_NEW_GRID_TRAINING',
        declared_at=bench.now(), cases=cases, run_order_fixed=True,
        purpose='BOUNDED_ARCHITECTURE_LR_DEVELOPMENT_FOLLOWUP_NOT_PRODUCTION_QUALIFICATION',
        rationale=('Prior1.5166B LR1e-4 improves eight domains but fails Chinese. Test the two existing '
                   'smaller dense architectures with a fixed two-LR grid on identical actual production '
                   'samples. Keep all earlier failures; do not modify data, core math, margins or budget.'),
        training_source_sha256=source_sha256(ROOT), qualification_identity=current,
        source_inputs=bound, reference_plan=str((REFERENCE/'validation/plan.json').relative_to(ROOT)),
        reference_case='matched_baseline_b8', updates=200, target_tokens=838860800,
        expanded_window={k:baseline_protocol[k] for k in (
            'domains','seq_len','world_size','local_batch_size','global_batch_size',
            'source_sample_offset','samples_per_domain','global_batches_per_domain',
            'target_interval','validation_metadata_sha256','uncertainty')},
        gate=dict(primary_all_nine_strict=True, expanded_all_nine_strict=True,
                  aggregate_strict=True, noninferiority_margin=0, forecast_days_ceiling=33,
                  finite_all_updates=True, allocator_retries=0),
        expansion_policy='Only candidates passing all nine original means and macro are expanded; failed primary is terminal FAIL, never PASS.',
        selection_rule='Largest parameter count among all strict passes; within same architecture choose smallest expanded macro, then case name.',
        no_global_optimum_or_unseen_test_claim=True, old_failures_preserved=True,
        production_qualified=False, formal_pretrain_started=False)
    PROTOCOL.write_bytes(encoded(declaration))
    return declaration


def require_inputs(protocol):
    if qualification_identity(ROOT) != protocol['qualification_identity']:
        raise ValueError('full live production qualification inputs changed')
    for name, expected in protocol['source_inputs'].items():
        if bench.digest(ROOT/name) != expected:
            raise ValueError('predeclared input changed: '+name)


def prepare_evaluation(protocol, cases):
    reference_plan = json.loads((ROOT/protocol['reference_plan']).read_bytes())
    folder = RESULTS/'validation'
    folder.mkdir()
    inputs = dict(protocol['source_inputs'])
    inputs[str(PROTOCOL.relative_to(ROOT))] = bench.digest(PROTOCOL)
    evaluation_protocol = dict(protocol['expanded_window'], cases=cases,
                              training_source_sha256=protocol['training_source_sha256'])
    pp = folder/'protocol.json'
    pp.write_bytes(encoded(evaluation_protocol))
    inputs[str(pp.relative_to(ROOT))] = bench.digest(pp)
    rows = {}
    runtime_keys = ('TORCHINDUCTOR_CACHE_DIR','TORCHINDUCTOR_COMPILE_THREADS','OMP_NUM_THREADS',
                    'PYTORCH_CUDA_ALLOC_CONF','CUBLAS_WORKSPACE_CONFIG','NVIDIA_TF32_OVERRIDE',
                    'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')
    for case in cases:
        run = RUNS/case
        config = yaml.safe_load((run/'resolved_config.yaml').read_bytes())
        config.update(out_dir=str(RUNS/'validation'/case), run_name=case+'_readonly')
        cp = folder/(case+'.yaml')
        cp.write_text(yaml.safe_dump(config, sort_keys=False))
        env = dict(line.split('=',1) for line in (CONFIGS/(case+'.env')).read_text().splitlines()
                   if line.strip() and not line.lstrip().startswith('#'))
        runtime = {k:env[k] for k in runtime_keys if k in env}
        if runtime.get('TORCHINDUCTOR_CACHE_DIR') != '/tmp/yuformer-inductor-cudnn-qualified-v3':
            raise ValueError('do not change the actual numeric cache')
        checkpoint = run/'checkpoint/step-200'
        for path in (cp, run/'result.json', run/'run_identity.json', run/'resolved_config.yaml', checkpoint/'.metadata'):
            inputs[str(path.relative_to(ROOT))] = bench.digest(path)
        rows[case] = dict(config=str(cp.relative_to(ROOT)), runtime_environment=runtime,
            checkpoint=str(checkpoint.relative_to(ROOT)), checkpoint_stat=checkpoint_identity(checkpoint),
            checkpoint_metadata_sha256=bench.digest(checkpoint/'.metadata'),
            primary_result=str((run/'result.json').relative_to(ROOT)),
            report=str((RUNS/'validation'/case/'scores.json').relative_to(ROOT)))
    plan = dict(protocol=str(pp.relative_to(ROOT)), cases=rows,
                validation_streams=reference_plan['validation_streams'], source_inputs=inputs)
    path = folder/'plan.json'
    path.write_bytes(encoded(plan))
    return path, plan


def run():
    protocol = json.loads(PROTOCOL.read_bytes())
    state = dict(pid=os.getpid(), status='WAITING_CURRENT_SOURCE_SYSTEM_PIPELINE', cases={},
                 protocol_sha256=bench.digest(PROTOCOL), production_qualified=False,
                 formal_pretrain_started=False, source_qualification_identity=protocol['qualification_identity'])
    with RunLease(RESULTS, 'production-shape-lr-bounded-followup'):
        try:
            bench.publish(HEALTH, state)
            while True:
                require_inputs(protocol)
                upstream = json.loads((ROOT/'data/health/current-a138-gpu-recovery-pipeline.json').read_bytes())
                if upstream['status'] == 'PASS_SYSTEM_GATES_QUALITY_REMAINS_BLOCKED':
                    from run_production_recovery import validate_report
                    validate_report(ROOT)
                    break
                if upstream['status'] not in ('STARTING', 'RUNNING'):
                    raise ValueError('current system pipeline did not pass: '+upstream['status'])
                bench.publish(HEALTH, state)
                time.sleep(30)
            bench.CONFIGS, bench.RESULTS, bench.RUNS, bench.HEALTH = CONFIGS, RESULTS, RUNS, HEALTH
            original_evaluator = bench.evaluate
            bench.evaluate = lambda result: original_evaluator(result, steps=200, skip_steps=50)
            reference_protocol, reference, _ = load_evaluation(ROOT, ROOT/protocol['reference_plan'], protocol['reference_case'])
            eligible = []
            for row in protocol['cases']:
                require_inputs(protocol)
                state['status'] = 'RUNNING_MATCHED_GRID'
                measured = bench.run_variant(row, row['variant'], state)
                state['cases'][row['name']] = measured
                if measured['evaluation']['status'] != 'PASS_SCREENING':
                    raise ValueError('new candidate system test failed: '+row['name'])
                if not measured['evaluation']['within_33_days']:
                    raise ValueError('new candidate exceeds declared forecast ceiling')
                log = Path(measured['log']).read_bytes()
                review_memory_log(log, steps=200, skip_steps=50)
                require_no_allocator_retries(log)
                runtime = audit(RUNS/row['name'], check_checkpoint=True)
                runtime['reference_scheduler'] = verify_reference_schedule(
                    yaml.safe_load((RUNS/row['name']/'resolved_config.yaml').read_bytes()))
                if (runtime['checked_lr_points'] != 400 or runtime['checkpoint']['status'] != 'PASS'
                        or runtime['checkpoint']['step'] != 200 or runtime['checkpoint']['samples_consumed'] != 204800
                        or runtime['checkpoint']['global_tokens_seen'] != 838860800
                        or runtime['checkpoint']['optimizer_groups_checked'] != row['optimizer_groups']
                        or runtime['reference_scheduler']['status'] != 'PASS'
                        or runtime['reference_scheduler']['scheduler_epochs_checked'] != 201):
                    raise ValueError('new candidate final runtime evidence does not close')
                bench.publish(RESULTS/(row['name']+'-runtime.json'), runtime)
                primary = json.loads((RUNS/row['name']/'result.json').read_bytes())
                comparison = {domain:dict(candidate=primary['metrics'][f'validation_metrics/domain/{domain}/loss']['last'],
                    baseline=reference['domains'][domain]['original_primary_loss']) for domain in reference_protocol['domains']}
                for value in comparison.values():
                    value.update(delta=value['candidate']-value['baseline'])
                primary_pass = all(value['delta'] < 0 for value in comparison.values())
                primary_pass &= primary['metrics']['validation_metrics/loss']['last'] < float(np.mean(
                    [reference['domains'][domain]['original_primary_loss'] for domain in reference_protocol['domains']]))
                measured.update(primary_comparison=comparison, primary_passed=bool(primary_pass),
                                quality_status='PENDING_EXPANDED' if primary_pass else 'FAIL_PRIMARY_ALL_NINE')
                if primary_pass:
                    eligible.append(row['name'])
                bench.publish(HEALTH, state)
            if eligible:
                path, plan = prepare_evaluation(protocol, eligible)
                for case in eligible:
                    require_inputs(protocol)
                    bench.require_idle()
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', YUFORMER_REPO_ROOT=str(ROOT),
                               PYTHONPATH=f"{ROOT/'pretrain/backends/torchtitan'}:{ROOT/'model/src'}:{ROOT/'pretrain/src'}")
                    env.update(plan['cases'][case]['runtime_environment'])
                    for key in ('RESUME_ITER','YUFORMER_ALLOW_TOPOLOGY_RESHARD','YUFORMER_OVERRIDE_GPUS_PER_NODE',
                                'YUFORMER_RUN_LEASE_HELD','YUFORMER_RUN_LEASE_TOKEN','YUFORMER_RUN_LEASE_GENERATION',
                                'YUFORMER_ATTEMPT_ID','EXTRA_ARGS'):
                        env.pop(key, None)
                    with (RESULTS/'validation'/(case+'.log')).open('x') as output:
                        child = subprocess.Popen([PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',
                            str(Path(__file__).with_name('evaluate_disjoint_validation.py')),'--case',case,
                            '--plan',str(path),'--plan-sha256',bench.digest(path)],
                            cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
                        state.update(status='RUNNING_READONLY_EXPANDED', active_candidate=case, child_pid=child.pid)
                        bench.publish(HEALTH,state)
                        if child.wait():
                            raise ValueError('new candidate readonly expanded evaluation failed')
                    p, candidate, _ = load_evaluation(ROOT,path,case)
                    comparisons = compare(p,candidate,reference_protocol,reference,
                                          np.random.default_rng(p['uncertainty']['seed']))
                    passed = (all(v['mean_strictly_improves'] for v in comparisons.values())
                              and candidate['aggregate_mean_loss'] < reference['aggregate_mean_loss'])
                    state['cases'][case].update(quality_status='PASS_PRIMARY_AND_EXPANDED_ALL_NINE' if passed else
                        'FAIL_EXPANDED_ALL_NINE', expanded_comparison=comparisons,
                        expanded_macro_loss=candidate['aggregate_mean_loss'])
                    bench.publish(HEALTH,state)
            require_inputs(protocol)
            passing = [row for row in state['cases'].values() if row['quality_status']=='PASS_PRIMARY_AND_EXPANDED_ALL_NINE']
            passing.sort(key=lambda row:(-row['parameters'],row['expanded_macro_loss'],row['candidate']))
            state.update(status='COMPLETE_BOUNDED_GRID_REQUIRES_SELECTION_REVIEW' if passing else
                         'FAIL_ALL_PREDECLARED_CANDIDATES_QUALITY',
                         preferred_case=passing[0]['candidate'] if passing else None)
            bench.publish(RESULTS/'review.json',state)
            bench.publish(HEALTH,state)
        except Exception as exc:
            state.update(status='FAIL_CLOSED',error=repr(exc))
            bench.publish(HEALTH,state)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declare', action='store_true')
    args = parser.parse_args()
    if args.declare:
        declare()
        print('Declared fixed four-case architecture/LR development grid; no GPU training launched.')
    else:
        run()
