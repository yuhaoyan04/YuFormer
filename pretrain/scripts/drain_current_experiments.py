"""Observe a user-requested drain, then stop only the captured idle coordinators."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

from freeze_code_candidate import atomic_json

ROOT = Path(__file__).resolve().parents[2]
HEALTH = ROOT / 'data/health'


def exact_process(pid: int, required: list[str]) -> bool:
    try:
        path = Path('/proc') / str(pid)
        stat = path.joinpath('stat').read_text()
        if stat[stat.rfind(')') + 2:].split()[0] == 'Z':
            return False
        argv = [os.fsdecode(arg) for arg in path.joinpath('cmdline').read_bytes().split(b'\0') if arg]
        return all(arg in argv for arg in required)
    except OSError:
        return False


def unfinished(request: dict, queues: dict, coordinator_alive: dict) -> list[str]:
    remaining = []
    for group, jobs in request['running_at_request'].items():
        for name, captured in jobs.items():
            live = exact_process(int(captured['pid']),
                                 [str(ROOT / 'pretrain/backends/torchtitan/launch.sh'), captured['env']])
            current = queues.get(group, {}).get('jobs', {}).get(name, {})
            # Let the existing queue finish result collection and process cleanup.
            collecting = (coordinator_alive.get(group, False)
                          and current.get('status') == 'RUNNING'
                          and current.get('attempt_id') == captured['attempt_id'])
            if live or collecting:
                remaining.append(name)
    return remaining


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    pause = HEALTH / 'pretrain-user-pause.json'
    if not pause.exists():
        raise SystemExit('A persistent user pause must exist before observing drain.')
    lock = (HEALTH / 'pretrain-user-drain.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    coordinators = request['queue_coordinators']
    while True:
        queues = {group: json.loads((HEALTH / f'{group}-queue.json').read_text())
                  for group in request['running_at_request']}
        alive = {group: exact_process(int(pid), [str(ROOT / 'pretrain/scripts/run_experiment_queue.py'),
                                                '--group', group])
                 for group, pid in coordinators.items()}
        # Keep the admission barrier closed even if another tool edits allocations.
        for path in (HEALTH / 'gpu-allocations').glob('*.json'):
            allocation = json.loads(path.read_text())
            if allocation.get('max_parallel') != 0:
                allocation.update(max_parallel=0, mode='USER_DRAIN_NO_NEW_LAUNCHES')
                atomic_json(path, allocation)
        remaining = unfinished(request, queues, alive)
        state = {'status': 'DRAINING' if remaining else 'STOPPING_IDLE_COORDINATORS',
                 'updated_at': datetime.now(timezone.utc).isoformat(),
                 'observer_pid': os.getpid(), 'remaining_jobs': remaining,
                 'request': str(args.request), 'new_launches_allowed': False,
                 'retry_allowed': False, 'automatic_resume_allowed': False}
        atomic_json(pause, state)
        if remaining:
            time.sleep(60)
            continue
        # A queue may have collected the last result between the snapshot and here;
        # never stop any coordinator whose current state still has active training.
        for group, pid in coordinators.items():
            current = json.loads((HEALTH / f'{group}-queue.json').read_text())
            if any(job.get('status') == 'RUNNING' for job in current.get('jobs', {}).values()):
                raise SystemExit(f'Refusing to terminate {group}: active job remains.')
            if alive.get(group):
                os.kill(int(pid), signal.SIGTERM)
        for _ in range(30):
            if not any(exact_process(int(pid), [str(ROOT / 'pretrain/scripts/run_experiment_queue.py')])
                       for pid in coordinators.values()):
                break
            time.sleep(1)
        else:
            raise SystemExit('An idle coordinator did not exit; pause latch remains active.')
        state.update(status='PAUSED', completed_at=datetime.now(timezone.utc).isoformat(),
                     remaining_jobs=[], observer_will_exit=True)
        atomic_json(pause, state)
        request.update(status='DRAINED_AND_PAUSED', completed_at=state['completed_at'])
        atomic_json(args.request, request)
        print(json.dumps(state, sort_keys=True), flush=True)
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
