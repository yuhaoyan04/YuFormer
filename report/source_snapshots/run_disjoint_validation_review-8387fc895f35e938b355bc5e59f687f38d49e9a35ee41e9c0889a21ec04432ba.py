"""Run the predeclared larger held-out review after all training controls finish."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import yaml

import benchmark_model_sizes as bench
from evaluate_disjoint_validation import checkpoint_identity, file_identity
from research_control_identity import research_control_identity

ROOT = bench.ROOT
sys.path.insert(0, str(ROOT/'pretrain/src'))
from yuformer_pretrain.lease import RunLease

FOLDER = ROOT/'experiments/e2_architecture/results/rope_source_controls_20261114/disjoint_validation'
PROTOCOL = ROOT/'experiments/e2_architecture/results/rope_source_controls_20261114/disjoint_validation_protocol.json'
CONFIGS = ROOT/'experiments/e2_architecture/configs/rope_source_controls_20261114/disjoint_validation'
RUNS = ROOT/'data/experiments/rope_source_controls_20261114/disjoint_validation'
HEALTH = ROOT/'data/health/disjoint-validation-review.json'


def prepare(protocol):
    physical_path = ROOT/'data/processed/audits/pretrain_1t_token_streams.json'
    physical = json.loads(physical_path.read_bytes())
    if physical.get('status') != 'PASS' or physical.get('roster_exact') is not True:
        raise ValueError('actual complete physical audit is required before additional validation')
    rows = {row['name']:row for row in physical['streams']}
    inputs = {str(path.relative_to(ROOT)):bench.digest(path) for path in
              (Path(__file__), Path(bench.__file__), Path(__file__).with_name('evaluate_disjoint_validation.py'),
               PROTOCOL, physical_path)}
    streams = {}
    for domain in protocol['domains']:
        name = f'data/token_streams/m3_20261020_yf64k/val_{domain}.meta.json'
        meta_path = ROOT/name
        if bench.digest(meta_path) != protocol['validation_metadata_sha256'][name]:
            raise ValueError('predeclared validation metadata changed')
        meta = json.loads(meta_path.read_bytes())
        row = rows['val_'+domain]
        binary = meta_path.with_name('val_'+domain+'.tokens.bin')
        actual = {key:row['actual_'+key] for key in ('bytes','device','inode','mtime_ns')}
        if (row['status'] != 'PASS' or row['actual_sha256'] != meta['sha256']
                or file_identity(binary) != actual or meta['dtype'] != '<u2'):
            raise ValueError('validation bytes differ from their actual full audit: '+domain)
        inputs[name] = bench.digest(meta_path)
        streams[domain] = dict(file=str(binary.relative_to(ROOT)), stat=actual,
                               full_audit_sha256=row['actual_sha256'])
    cases = {}
    for case in protocol['cases']:
        source = ROOT/'data/experiments/rope_source_controls_20261114'/case
        primary = json.loads((source/'result.json').read_bytes())
        identity = json.loads((source/'run_identity.json').read_bytes())
        config = yaml.safe_load((source/'resolved_config.yaml').read_bytes())
        if (primary['status'] != 'PASS' or identity['source_sha256'] != protocol['training_source_sha256']
                or primary['hyperparameters']['trained_steps'] != 200):
            raise ValueError('additional review requires the exact completed current-source checkpoints')
        config['out_dir'] = str(RUNS/case)
        config['run_name'] = case+'_disjoint_validation'
        path = CONFIGS/(case+'.yaml')
        raw = yaml.safe_dump(config, sort_keys=False)
        if path.exists() or (RUNS/case).exists():
            raise ValueError('held-out review outputs must be new; previous attempts are preserved')
        path.write_text(raw)
        checkpoint = source/'checkpoint/step-200'
        for dependency in (source/'result.json', source/'run_identity.json', source/'resolved_config.yaml',
                           checkpoint/'.metadata', Path(identity['model_configs']), path):
            inputs[str(dependency.relative_to(ROOT))] = bench.digest(dependency)
        cases[case] = dict(config=str(path.relative_to(ROOT)), checkpoint=str(checkpoint.relative_to(ROOT)),
                          checkpoint_metadata_sha256=bench.digest(checkpoint/'.metadata'),
                          checkpoint_stat=checkpoint_identity(checkpoint),
                          primary_result=str((source/'result.json').relative_to(ROOT)),
                          report=str((RUNS/case/'scores.json').relative_to(ROOT)))
    return dict(protocol=str(PROTOCOL.relative_to(ROOT)), cases=cases,
                validation_streams=streams, source_inputs=inputs)


def review(protocol, plan):
    reports = {case:json.loads((ROOT/row['report']).read_bytes()) for case,row in plan['cases'].items()}
    for case, report in reports.items():
        if (report['status'] != 'PASS_READONLY_DISJOINT_VALIDATION'
                or report['protocol_sha256'] != bench.digest(PROTOCOL)
                or report['training_source_sha256'] != protocol['training_source_sha256']
                or report['checkpoint_metadata_sha256'] != plan['cases'][case]['checkpoint_metadata_sha256']
                or report['source_inputs'] != plan['source_inputs']
                or report['training_steps_executed'] != 0 or report['checkpoint_modified'] is not False):
            raise ValueError('actual disjoint evaluation is incomplete or belongs to another checkpoint')
    rng = np.random.default_rng(protocol['uncertainty']['seed'])
    comparisons = {}
    candidate = reports['larger_lr1e4_b8']
    for other in ('matched_baseline_b8','larger_lr2e4_b8'):
        domains = {}
        for domain in protocol['domains']:
            left = np.asarray(candidate['domains'][domain]['batch_mean_losses'])
            right = np.asarray(reports[other]['domains'][domain]['batch_mean_losses'])
            if left.shape != (44,) or right.shape != (44,) or not np.isfinite([left,right]).all():
                raise ValueError('paired review needs all forty-four finite matching batches')
            delta = left-right
            samples = rng.integers(0,len(delta),size=(10000,len(delta)))
            low, high = np.quantile(delta[samples].mean(axis=1), [.025,.975])
            domains[domain] = dict(candidate_mean=float(left.mean()), control_mean=float(right.mean()),
                                   delta=float(delta.mean()), paired_bootstrap_95_ci=[float(low),float(high)],
                                   mean_strictly_improves=bool(delta.mean()<0))
        comparisons[other] = domains
    passed = all(row['mean_strictly_improves'] for comparison in comparisons.values() for row in comparison.values())
    passed &= all(candidate['aggregate_mean_loss'] < reports[case]['aggregate_mean_loss']
                  for case in ('matched_baseline_b8','larger_lr2e4_b8'))
    return dict(status='PASS_DISJOINT_ALL_NINE_MEAN_IMPROVEMENT' if passed else 'FAIL_DISJOINT_ALL_NINE_MEAN_IMPROVEMENT',
                protocol=str(PROTOCOL.relative_to(ROOT)), protocol_sha256=bench.digest(PROTOCOL),
                primary_review_preserved_as_failed=True, comparisons=comparisons,
                report_sha256={str((ROOT/plan['cases'][case]['report']).relative_to(ROOT)):
                               bench.digest(ROOT/plan['cases'][case]['report']) for case in reports},
                production_qualified=False, formal_pretrain_started=False)


def main():
    FOLDER.mkdir(parents=True,exist_ok=True)
    CONFIGS.mkdir(parents=True,exist_ok=True)
    state = dict(pid=os.getpid(), status='WAITING_COMPLETED_SOURCE_CONTROLS', cases={},
                 protocol_sha256=bench.digest(PROTOCOL), production_qualified=False,
                 formal_pretrain_started=False)
    loaded = {str(path):bench.digest(path) for path in
              (Path(__file__), Path(bench.__file__), Path(__file__).with_name('evaluate_disjoint_validation.py'))}
    state['loaded_source_inputs'] = loaded
    with RunLease(FOLDER, 'readonly-larger-disjoint-heldout-review'):
        try:
            bench.publish(HEALTH,state)
            deadline=time.monotonic()+10800
            while True:
                if any(bench.digest(path) != expected for path,expected in loaded.items()):
                    raise ValueError('loaded review producer changed while waiting')
                controls=json.loads((ROOT/'data/health/model-rope-source-controls.json').read_bytes())
                if controls['status']=='COMPLETE_SUSTAINED_CONTROLS':break
                if controls['status']!='RUNNING' or time.monotonic()>deadline:
                    raise ValueError('actual current-source controls did not complete successfully')
                time.sleep(15)
            if research_control_identity(ROOT, ROOT/controls['config_directory']) != controls['source_qualification_identity']:
                raise ValueError('actual matched-control research identity changed before independent review')
            protocol=json.loads(PROTOCOL.read_bytes())
            plan=prepare(protocol)
            plan_path=FOLDER/'plan.json'
            bench.publish(plan_path,plan)
            for case in protocol['cases']:
                bench.require_idle()
                state.update(status='RUNNING_READONLY_DISJOINT_VALIDATION',active_case=case)
                bench.publish(HEALTH,state)
                log=FOLDER/(case+'.log')
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='1',
                         PYTHONPATH=f"{ROOT/'pretrain/backends/torchtitan'}:{ROOT/'model/src'}:{ROOT/'pretrain/src'}",
                         TORCHINDUCTOR_COMPILE_THREADS='2',TORCHINDUCTOR_CACHE_DIR='/tmp/yuformer-disjoint-eval-f127',
                         YUFORMER_REPO_ROOT=str(ROOT))
                for key in ('RESUME_ITER','YUFORMER_ALLOW_TOPOLOGY_RESHARD','YUFORMER_OVERRIDE_GPUS_PER_NODE',
                            'YUFORMER_RUN_LEASE_HELD','YUFORMER_RUN_LEASE_TOKEN','YUFORMER_ATTEMPT_ID'):
                    env.pop(key,None)
                with log.open('w') as output:
                    child=subprocess.Popen([str(bench.PYTHON),'-m','torch.distributed.run','--standalone',
                        '--nproc_per_node=8',str(Path(__file__).with_name('evaluate_disjoint_validation.py')),
                        '--case',case,'--plan',str(plan_path),'--plan-sha256',bench.digest(plan_path)],
                        cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
                    state['child_pid']=child.pid;bench.publish(HEALTH,state)
                    code=child.wait()
                state['cases'][case]=dict(returncode=code,log=str(log))
                if code:raise ValueError('actual read-only evaluation failed: '+case)
            result=review(protocol,plan)
            bench.publish(FOLDER/'review.json',result)
            state.update(status=result['status'],review=str((FOLDER/'review.json').relative_to(ROOT)))
            return 0 if result['status'].startswith('PASS_') else 2
        except Exception as error:
            state.update(status='FAIL_CLOSED_DISJOINT_REVIEW',error=repr(error));raise
        finally:
            state['updated_at']=bench.now();bench.publish(HEALTH,state)


if __name__=='__main__':
    raise SystemExit(main())
