"""Recover dead outer supervisors while preserving the live training attempt.

This is an explicit operator handoff of the existing generation, never a new
training launch. Original owner bytes and all changed status files are archived.
The existing bash launcher remains the sole producer of terminal train markers.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path('/opt/huawei/explorer-env/dataset/yyh_arcagi/YUFormer/YuFormer')
HEALTH = ROOT/'data/health'
RUN = ROOT/'data/runs/pretrain-final-1t-8gpu'
PYTHON = '/opt/yuformer-torchtitan/bin/python'
ATTEMPT = 'c5bc01ad619700958f8fdac0967605f2'
IDENTITY = '8eeae4ac539490c01ed9987b027434fafc82d8f24df473efb3350a19cff05b5d'
GENERATION = RUN/'.run.lease/generation-00000000000000000006'
TOKEN = '<REDACTED_RUNTIME_LEASE_TOKEN>'
LIVE_LAUNCHER = 430884
LIVE_TORCHRUN = 430905
WORKERS = tuple(range(430909, 430917))
STATE = HEALTH/'live-pretrain-supervisor-recovery-20261118.json'
sys.path.insert(0, str(ROOT/'pretrain/src'))
from yuformer_pretrain.lease import RunLease


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def atomic(path, data):
    temp = path.with_name(path.name+f'.tmp-{os.getpid()}')
    with temp.open('w') as handle:
        handle.write(json.dumps(data, sort_keys=True, indent=2)+'\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def proc(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return None
    fields = raw[raw.rfind(')')+2:].split()
    return {'pid': pid, 'state': fields[0], 'ppid': int(fields[1]),
            'start_ticks': fields[19], 'exit_code_raw': int(fields[49])}


def alive(record):
    return bool(record and record['state'] not in ('Z', 'X'))


def owner_dead(owner):
    if owner.get('hostname') != socket.gethostname():
        return False
    return not alive(proc(int(owner['pid'])))


def validate_attempt(attempt, identity_sha):
    if (attempt.get('attempt_id') != ATTEMPT or identity_sha != IDENTITY
            or attempt.get('status') != 'RUNNING'):
        raise ValueError('actual live attempt/identity changed')


def self_test():
    assert not alive(None) and not alive({'state': 'Z'})
    assert alive({'state': 'R'})
    assert not owner_dead({'hostname': 'foreign-host', 'pid': os.getpid()})
    assert not owner_dead({'hostname': socket.gethostname(), 'pid': os.getpid()})
    for attempt, digest in [({'attempt_id': 'wrong', 'status': 'RUNNING'}, IDENTITY),
                            ({'attempt_id': ATTEMPT, 'status': 'PASS'}, IDENTITY),
                            ({'attempt_id': ATTEMPT, 'status': 'RUNNING'}, 'wrong')]:
        try:
            validate_attempt(attempt, digest)
        except ValueError:
            pass
        else:
            raise AssertionError('unsafe attempt accepted')
    validate_attempt({'attempt_id': ATTEMPT, 'status': 'RUNNING'}, IDENTITY)
    print('PASS_RECOVERY_REJECTS_LIVE_OWNER_FOREIGN_OWNER_WRONG_ATTEMPT_IDENTITY_TERMINAL')


def recover():
    lock = (HEALTH/'live-pretrain-supervisor-recovery-20261118.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pointer = read(HEALTH/'outer-supervisor-recovery-20261118-incident-pointer.json')
    if sha(pointer['incident']) != pointer['sha256']:
        raise ValueError('incident evidence changed')
    incident = read(pointer['incident'])
    archive = Path(incident['archive'])
    for path, digest in incident['source_inputs'].items():
        if sha(path) != digest:
            raise ValueError('frozen source changed: '+path)
    validate_attempt(read(RUN/'current_attempt.json'), sha(RUN/'run_identity.json'))
    original_owner = read(GENERATION/'owner.json')
    if original_owner.get('token') != TOKEN or not owner_dead(original_owner):
        raise ValueError('only a conclusively dead local generation owner may be replaced')
    original_processes = {pid: proc(pid) for pid in (LIVE_LAUNCHER, LIVE_TORCHRUN, *WORKERS)}
    if not all(alive(record) for record in original_processes.values()):
        raise ValueError('the original complete training process roster is not live')
    if original_processes[LIVE_TORCHRUN]['ppid'] != LIVE_LAUNCHER:
        raise ValueError('torchrun parent mismatch')
    if any(original_processes[pid]['ppid'] != LIVE_TORCHRUN for pid in WORKERS):
        raise ValueError('training worker parent mismatch')
    env = dict(item.split(b'=', 1) for item in Path(f'/proc/{LIVE_LAUNCHER}/environ').read_bytes().split(b'\0') if b'=' in item)
    if (env.get(b'YUFORMER_ATTEMPT_ID', b'').decode() != ATTEMPT
            or env.get(b'YUFORMER_RUN_LEASE_TOKEN', b'').decode() != TOKEN
            or env.get(b'YUFORMER_RUN_LEASE_GENERATION', b'').decode() != str(GENERATION)):
        raise ValueError('live launcher does not hold the original exact generation')
    run_lease = RunLease(RUN, 'operator-handoff-existing-live-training-generation', heartbeat_seconds=3600)
    run_lease.path, run_lease.token = GENERATION, TOKEN
    run_lease.assert_owner()
    owner = dict(original_owner, pid=os.getpid(),
                 operator_handoff_at=now(), operator_handoff_script=str(Path(__file__)),
                 operator_handoff_script_sha256=sha(__file__),
                 previous_owner_pid=original_owner['pid'],
                 previous_owner_evidence=str(archive/'owner.json.snapshot'),
                 handoff_scope='CONTINUE_SAME_LIVE_ATTEMPT_SAME_GENERATION_NO_TRAINING_RELAUNCH')
    # Retain the exact generation/token used by the eight live workers. Minting a
    # new generation would fence those workers and invalidate their completion.
    atomic(GENERATION/'owner.json', owner)
    atomic(RUN/'run.lock', owner)
    run_lease.renew()
    pipeline_lease = RunLease(HEALTH/'authorized-pretrain-pipeline',
                              'operator-recovery-existing-live-pretrain', stale_seconds=600).acquire()
    pipeline_path = HEALTH/'authorized-pretrain-pipeline.json'
    pipeline = read(pipeline_path)
    if pipeline.get('pid') != 430805 or pipeline.get('stage') != 'full_pretrain':
        raise ValueError('unexpected pipeline transition before recovery')
    pipeline.update(pid=os.getpid(), updated_at=now(),
                    recovery_of_pipeline_pid=430805,
                    recovery_evidence=str(archive/'incident.json'),
                    observation_scope='ADOPT_EXISTING_LIVE_TRAINING_NO_RELAUNCH')
    pipeline['stages']['full_pretrain'].update(pid=LIVE_LAUNCHER,
                    previous_wrapper_pid=430858,
                    process_scope='ORIGINAL_STILL_RUNNING_BASH_LAUNCHER')
    atomic(pipeline_path, pipeline)
    state = {'status': 'RECOVERED_SAME_LIVE_ATTEMPT_SUPERVISING_BACKGROUND_GUARDS',
             'pid': os.getpid(), 'started_at': now(), 'attempt_id': ATTEMPT,
             'training_relaunched': False, 'run_generation': str(GENERATION),
             'owner_handoff': owner, 'original_processes': original_processes,
             'incident': pointer, 'source_inputs': incident['source_inputs'],
             'script_sha256': sha(__file__), 'helpers': {}, 'alerts': []}
    specs = [
        ('eta_lr', '/opt/yuformer-build/observe_total_runtime_and_actual_lr_20261115_v3.py', [], 443143),
        ('native', '/opt/yuformer-build/watch_formal_native_checkpoint_states_20261115.py', [], 434802),
        ('nine_domain', '/opt/yuformer-build/watch_all_production_nine_domain_validation_20261115_v2.py', [], 448423),
        ('long_lr', '/opt/yuformer-build/observe_frozen_long_stage_lr_and_native_counters_20261115.py', [], 441248),
        ('prefetch', '/opt/yuformer-build/prefetch_frozen_future_blocks_20261115.py', [], 434050),
        ('context_chain', str(ROOT/'eval/scripts/run_context_extension_pipeline.py'),
            ['--config', str(ROOT/'eval/configs/authorized_context_chain_20261115_v1.json')], 440338),
    ]
    children = {}
    for name, script, args, old_pid in specs:
        if alive(proc(old_pid)):
            raise ValueError('original helper unexpectedly live: '+name)
        log = archive/(name+'-recovered.log')
        with log.open('ab') as handle:
            child = subprocess.Popen([PYTHON, script, *args], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                start_new_session=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        children[name] = child
        state['helpers'][name] = {'pid': child.pid, 'previous_pid': old_pid,
                                  'script': script, 'log': str(log)}
    # ContextPipeline supplies the frozen physical UUID list explicitly for each
    # GPU worker, overriding this supervisor's CPU-only CUDA environment.
    while True:
        run_lease.renew()
        pipeline_lease.assert_owner()
        state['updated_at'] = now()
        state['alerts'] = []
        if sha(__file__) != state['script_sha256']:
            raise ValueError('recovery execution source changed')
        if any(sha(path) != digest for path, digest in incident['source_inputs'].items()):
            raise ValueError('frozen execution sources changed during recovery supervision')
        for name, child in children.items():
            rc = child.poll()
            state['helpers'][name]['returncode'] = rc
            if rc is not None:
                # Completion of the context controller is independently verified;
                # no failed guard is silently restarted or its alert overwritten.
                if not (name == 'context_chain' and rc == 0 and read(HEALTH/'authorized-total-pretrain-context-chain-20261115-v1.json').get('status') == 'PASS_COMPLETE_4K_WSD_16K_64K_TRAINING_AND_ACTUAL_CAPABILITY_GATES'):
                    state['alerts'].append('BACKGROUND_HELPER_EXITED_'+name)
        current = proc(LIVE_LAUNCHER)
        same_launcher = alive(current) and current['start_ticks'] == original_processes[LIVE_LAUNCHER]['start_ticks']
        if not same_launcher:
            complete = read(RUN/'training_complete.json') if (RUN/'training_complete.json').exists() else {}
            attempt = read(RUN/'current_attempt.json')
            valid_terminal = (complete.get('attempt_id') == ATTEMPT and attempt.get('attempt_id') == ATTEMPT
                and complete.get('identity_sha256') == attempt.get('identity_sha256')
                and complete.get('status') == attempt.get('status') == 'PASS'
                and complete.get('returncode') == 0)
            pipeline['status'] = 'COMPLETE' if valid_terminal else 'BLOCKED'
            pipeline['stages']['full_pretrain']['status'] = 'PASS' if valid_terminal else 'FAIL'
            pipeline['updated_at'] = now()
            if valid_terminal:
                pipeline['stages']['full_pretrain']['returncode'] = 0
                state['status'] = 'ACTUAL_4K_TERMINAL_VERIFIED_CONTEXT_CHAIN_CONTINUES'
            else:
                pipeline['error'] = 'original launcher exited without a matching successful terminal artifact'
                state['alerts'].append('ORIGINAL_TRAINING_TERMINAL_REQUIRES_DIAGNOSIS')
            atomic(pipeline_path, pipeline)
            # Never manufacture training_complete/current_attempt or run result.
            state['actual_terminal'] = complete
        atomic(STATE, state)
        if state['alerts']:
            print(json.dumps(state, sort_keys=True), flush=True)
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        try:
            recover()
        except BaseException as exc:
            atomic(STATE, {'status': 'RECOVERY_SUPERVISOR_ERROR_REQUIRES_REVIEW',
                'pid': os.getpid(), 'updated_at': now(), 'error': repr(exc),
                'traceback': traceback.format_exc(),
                'training_relaunched': False})
            raise
