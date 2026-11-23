"""Evaluate the predeclared actual-production checkpoint pair on the same window."""
from pathlib import Path
import copy
import json
import os
import subprocess
import sys

import numpy as np
import yaml

import benchmark_model_sizes as bench
from evaluate_disjoint_validation import checkpoint_identity, file_identity
from development_validation_review import load_evaluation, compare

ROOT = bench.ROOT
sys.path.insert(0,str(ROOT/'pretrain/src'))
from yuformer_pretrain.lease import RunLease

BASE = ROOT/'experiments/e2_architecture/results/final_deterministic_production_controls_20261115'
CONFIGS = ROOT/'experiments/e2_architecture/configs/final_deterministic_production_controls_20261115/validation'
RUNS = ROOT/'data/experiments/final_deterministic_production_controls_20261115/validation'
HEALTH = ROOT/'data/health/final-deterministic-production-review.json'


def prepare():
    health_path = ROOT/'data/health/final-deterministic-production-controls.json'
    health = json.loads(health_path.read_bytes())
    if health['status'] != 'COMPLETE_REQUIRES_PAIRED_PRIMARY_AND_EXPANDED_QUALITY_REVIEW':
        raise ValueError('actual production checkpoint pair must finish before evaluation')
    declaration_path = BASE/'protocol.json'
    declaration = json.loads(declaration_path.read_bytes())
    protocol = copy.deepcopy(declaration['expanded_window'])
    protocol.update(cases=[row['case'] for row in declaration['cases']],
        training_source_sha256=declaration['training_source_sha256'],
        target_candidate=declaration['candidate'], reference_cases=declaration['reference_cases'],
        training_protocol_sha256=bench.digest(declaration_path),
        purpose='PREDECLARED_ACTUAL_PRODUCTION_PAIR_DEVELOPMENT_VALIDATION', production_qualified=False)
    folder = BASE/'validation'
    if folder.exists() or CONFIGS.exists() or RUNS.exists():
        raise ValueError('preserve earlier production validation attempts')
    folder.mkdir()
    CONFIGS.mkdir(parents=True)
    protocol_path = folder/'protocol.json'
    bench.publish(protocol_path,protocol)
    physical_path = ROOT/'data/processed/audits/pretrain_1t_token_streams.json'
    physical = json.loads(physical_path.read_bytes())
    if physical['status'] != 'PASS' or not physical['roster_exact']:
        raise ValueError('actual complete production physical audit is required')
    audit_rows = {row['name']:row for row in physical['streams']}
    inputs = {str(p.relative_to(ROOT)):bench.digest(p) for p in (Path(__file__), Path(bench.__file__),
        Path(__file__).with_name('evaluate_disjoint_validation.py'),
        Path(__file__).with_name('development_validation_review.py'), declaration_path, protocol_path,
        health_path, physical_path)}
    streams = {}
    for domain in protocol['domains']:
        meta_path = ROOT/f'data/token_streams/m3_20261020_yf64k/val_{domain}.meta.json'
        if bench.digest(meta_path) != protocol['validation_metadata_sha256'][str(meta_path.relative_to(ROOT))]:
            raise ValueError('predeclared actual validation metadata changed')
        metadata = json.loads(meta_path.read_bytes())
        row = audit_rows['val_'+domain]
        binary = meta_path.with_name('val_'+domain+'.tokens.bin')
        actual = {key:row['actual_'+key] for key in ('bytes','device','inode','mtime_ns')}
        if row['status'] != 'PASS' or row['actual_sha256'] != metadata['sha256'] or file_identity(binary) != actual:
            raise ValueError('fully audited actual validation stream changed')
        inputs[str(meta_path.relative_to(ROOT))] = bench.digest(meta_path)
        streams[domain] = dict(file=str(binary.relative_to(ROOT)),stat=actual,full_audit_sha256=row['actual_sha256'])
    cases = {}
    for case in protocol['cases']:
        run = ROOT/'data/experiments/final_deterministic_production_controls_20261115'/case
        result = json.loads((run/'result.json').read_bytes())
        identity = json.loads((run/'run_identity.json').read_bytes())
        config = yaml.safe_load((run/'resolved_config.yaml').read_bytes())
        if (result['status'] != 'PASS' or result['hyperparameters']['trained_steps'] != 200
                or identity['source_sha256'] != protocol['training_source_sha256']
                or bench.digest(run/'result.json') != health['cases'][case]['result_sha256']):
            raise ValueError('actual production checkpoint identity/budget differs')
        config.update(out_dir=str(RUNS/case),run_name=case+'_production_development_validation')
        cp = CONFIGS/(case+'.yaml')
        cp.write_text(yaml.safe_dump(config,sort_keys=False))
        ep = ROOT/'experiments/e2_architecture/configs/final_deterministic_production_controls_20261115'/(case+'.env')
        values = dict(line.split('=',1) for line in ep.read_text().splitlines()
                      if line.strip() and not line.lstrip().startswith('#'))
        runtime = {k:values[k] for k in ('TORCHINDUCTOR_CACHE_DIR','TORCHINDUCTOR_COMPILE_THREADS','OMP_NUM_THREADS',
            'PYTORCH_CUDA_ALLOC_CONF','CUBLAS_WORKSPACE_CONFIG','NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE') if k in values}
        if runtime.get('TORCHINDUCTOR_CACHE_DIR') != '/tmp/yuformer-inductor-cudnn-qualified-v3':
            raise ValueError('actual training compiler cache must be preserved')
        checkpoint = run/'checkpoint/step-200'
        for p in (run/'result.json',run/'run_identity.json',run/'resolved_config.yaml',
                  Path(identity['model_configs']),cp,ep,checkpoint/'.metadata'):
            inputs[str(p.relative_to(ROOT))] = bench.digest(p)
        cases[case] = dict(config=str(cp.relative_to(ROOT)),runtime_environment=runtime,
            checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_stat=checkpoint_identity(checkpoint),
            checkpoint_metadata_sha256=bench.digest(checkpoint/'.metadata'),
            primary_result=str((run/'result.json').relative_to(ROOT)), report=str((RUNS/case/'scores.json').relative_to(ROOT)))
    plan = dict(protocol=str(protocol_path.relative_to(ROOT)),cases=cases,validation_streams=streams,source_inputs=inputs)
    path = folder/'plan.json'
    bench.publish(path,plan)
    return path,plan


from review_production_matched_controls import review


def main():
    state = dict(pid=os.getpid(),status='PREPARING',production_qualified=False,formal_pretrain_started=False,cases={})
    with RunLease(BASE/'validation_lease','readonly-actual-production-paired-validation'):
        try:
            plan_path,plan = prepare()
            for case in ('matched_baseline_b8','larger_lr1e4_b8'):
                bench.require_idle()
                env = dict(os.environ,CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',YUFORMER_REPO_ROOT=str(ROOT),
                    PYTHONPATH=f"{ROOT/'pretrain/backends/torchtitan'}:{ROOT/'model/src'}:{ROOT/'pretrain/src'}")
                env.update(plan['cases'][case]['runtime_environment'])
                for key in ('RESUME_ITER','YUFORMER_ALLOW_TOPOLOGY_RESHARD','YUFORMER_OVERRIDE_GPUS_PER_NODE',
                            'YUFORMER_RUN_LEASE_HELD','YUFORMER_RUN_LEASE_TOKEN','YUFORMER_ATTEMPT_ID'):
                    env.pop(key,None)
                log = plan_path.parent/(case+'.log')
                with log.open('x') as output:
                    child = subprocess.Popen([str(bench.PYTHON),'-m','torch.distributed.run','--standalone',
                        '--nproc_per_node=8',str(Path(__file__).with_name('evaluate_disjoint_validation.py')),
                        '--case',case,'--plan',str(plan_path),'--plan-sha256',bench.digest(plan_path)],
                        cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
                    state.update(status='RUNNING_READONLY_PRODUCTION_VALIDATION',active_case=case,child_pid=child.pid)
                    bench.publish(HEALTH,state)
                    code = child.wait()
                state['cases'][case] = dict(returncode=code,log=str(log))
                if code:
                    raise ValueError('actual production-pair read-only validation failed: '+case)
            result = review(plan_path)
            bench.publish(plan_path.parent/'review.json',result)
            state.update(status=result['status'],report=str(plan_path.parent/'review.json'),
                         report_sha256=bench.digest(plan_path.parent/'review.json'))
        except Exception as error:
            state.update(status='FAIL_CLOSED_PRODUCTION_PAIR_VALIDATION',error=repr(error))
            raise
        finally:
            state['updated_at'] = bench.now()
            bench.publish(HEALTH,state)


if __name__ == '__main__':
    main()
