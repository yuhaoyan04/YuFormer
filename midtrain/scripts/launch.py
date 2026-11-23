#!/usr/bin/env python3
"""Launch one sealed long-context stage under the shared distributed run lease."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import secrets
import subprocess
import sys
import time


def atomic(path, payload):
    temporary = path.with_name(path.name+f'.tmp-{os.getpid()}')
    with temporary.open('w') as handle:
        handle.write(json.dumps(payload, sort_keys=True, indent=2)+'\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary,path)


def environment(path):
    env = dict(os.environ)
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            key,value = line.split('=',1)
            if not key.replace('_','').isalnum() or key[0].isdigit():
                raise ValueError('invalid explicit stage environment key')
            env[key] = value
    for key in ('YUFORMER_PYTHON','YUFORMER_TORCHTITAN_ROOT','YUFORMER_SHARED_BACKEND_ROOT',
                'YUFORMER_REPO_ROOT','YUFORMER_BUDGET_CONTRACT_ROOT','YUFORMER_CONFIG'):
        if not Path(env[key]).is_absolute():
            raise ValueError('stage dependency must be an absolute declared path: '+key)
    root = Path(env['YUFORMER_REPO_ROOT'])
    env['PYTHONPATH'] = os.pathsep.join([env['YUFORMER_SHARED_BACKEND_ROOT'],str(root/'model/src'),
        str(root/'midtrain/src'),str(root/'pretrain/src'),env.get('PYTHONPATH','')])
    if env.get('EXTRA_ARGS') or int(env['GPUS_PER_NODE']) != 8:
        raise ValueError('long stages require eight GPUs and forbid runtime overrides')
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    args = parser.parse_args()
    env = environment(args.env.resolve())
    os.environ.update(env)
    for path in reversed(env['PYTHONPATH'].split(os.pathsep)):
        if path: sys.path.insert(0,path)
    from yuformer_midtrain.preflight import preflight
    contract = preflight(env['YUFORMER_CONFIG'],publish=False)
    if args.preflight_only:
        print(json.dumps({'status':contract['status'],'out_dir':contract['out_dir']}))
        return 0
    root,out = Path(env['YUFORMER_REPO_ROOT']),Path(contract['out_dir'])
    sys.path.insert(0,str(root/'pretrain/scripts'))
    if env.get('YUFORMER_RUN_LEASE_HELD') != '1':
        from gpu_resource_policy import allowed_gpu_indices
        devices = [int(d) for d in env['CUDA_VISIBLE_DEVICES'].split(',')]
        if len(devices)!=8 or len(set(devices))!=8 or not set(devices)<=allowed_gpu_indices(root/'data/health'):
            raise ValueError('long-stage GPU roster is not the authorized eight-card allocation')
        from run_with_lease import run
        return run(out,'midtrain:'+out.name,[env['YUFORMER_PYTHON'],str(Path(__file__).resolve()),'--env',str(args.env.resolve())])
    # The shared fence is checked by its CLI because its public API is a CLI.
    subprocess.run([env['YUFORMER_PYTHON'],str(root/'pretrain/scripts/assert_run_lease.py'),
                    '--out-dir',str(out)],check=True,env=env)
    contract = preflight(env['YUFORMER_CONFIG'],publish=True)
    attempt_id = secrets.token_hex(16)
    env['YUFORMER_ATTEMPT_ID'] = attempt_id
    env['YUFORMER_CURSOR_AUDIT_DIR'] = str(out/'cursor_audit')
    (out/'attempts').mkdir(exist_ok=True)
    previous_path = out/'current_attempt.json'
    previous = json.loads(previous_path.read_bytes()) if previous_path.exists() else {}
    archive = out/'attempts'/('superseded-before-'+attempt_id)
    from artifact_archive import move_artifact
    for artifact in [out/'training_complete.json',out/'result.json',*out.glob('**/events.out.tfevents.*')]:
        if artifact.exists() and out/'attempts' not in artifact.parents:
            move_artifact(artifact,archive/artifact.relative_to(out))
    attempt = {'schema_version':1,'status':'RUNNING','attempt_id':attempt_id,
        'started_at':datetime.now(timezone.utc).isoformat(),'started_at_ns':time.time_ns(),
        'identity_sha256':contract['identity']['identity_sha256'],'hostname':platform.node(),
        'launcher_pid':os.getpid(),'python_executable':env['YUFORMER_PYTHON'],
        'python_version':platform.python_version(),'gpus_per_node':8,
        'parent_attempt_id':previous.get('attempt_id')}
    for path in (out/'current_attempt.json',out/'attempts'/f'{attempt_id}.json'): atomic(path,attempt)
    command = [env['YUFORMER_PYTHON'],'-m','torch.distributed.run','--nproc_per_node','8',
        '--nnodes','1','--master_addr',env.get('MASTER_ADDR','127.0.0.1'),
        '--master_port',env.get('MASTER_PORT','29527'),'-m','torchtitan.train',
        '--module','yuformer_midtrain','--config','midtrain_from_yaml']
    completed = subprocess.run(command,env=env,cwd=env['YUFORMER_TORCHTITAN_ROOT'])
    subprocess.run([env['YUFORMER_PYTHON'],str(root/'pretrain/scripts/assert_run_lease.py'),
                    '--out-dir',str(out)],check=True,env=env)
    preflight(env['YUFORMER_CONFIG'],publish=True)
    completion = {'schema_version':1,'status':'PASS' if completed.returncode==0 else 'FAIL',
        'returncode':completed.returncode,'finished_at':datetime.now(timezone.utc).isoformat(),
        'run_name':contract['identity']['run_name'],'identity_sha256':contract['identity']['identity_sha256'],
        'attempt_id':attempt_id}
    atomic(out/'training_complete.json',completion)
    # Preserve the resume step recorded by the qualified shared Trainer.
    current = json.loads((out/'current_attempt.json').read_bytes())
    current.update(completion)
    for path in (out/'current_attempt.json',out/'attempts'/f'{attempt_id}.json'): atomic(path,current)
    return completed.returncode


if __name__ == '__main__':
    raise SystemExit(main())
