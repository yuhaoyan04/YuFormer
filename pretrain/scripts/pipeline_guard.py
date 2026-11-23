"""Low-cost supervisor of the readiness watcher and queue coordinators."""
from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from process_watchdog import enable_subreaper, reap_orphan_children
from gpu_resource_policy import allowed_gpu_indices

ROOT = Path(__file__).resolve().parents[2]
HEALTH = ROOT / 'data/health'
WATCHER = ROOT / 'pretrain/scripts/pretrain_readiness_watch.py'
QUEUE = ROOT / 'pretrain/scripts/run_experiment_queue.py'


def read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def exact_process(pid: int, script: Path) -> bool:
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        if raw[raw.rfind(')') + 2:].split()[0] == 'Z':
            return False
        argv = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        return any(Path(os.fsdecode(arg)).resolve() == script for arg in argv if arg)
    except (OSError, ValueError):
        return False


def restart_allowed(record: dict, now: float) -> bool:
    history = [stamp for stamp in record.get('restart_epochs', []) if now - stamp < 86400]
    record['restart_epochs'] = history
    return len(history) < 3 and (not history or now - history[-1] >= 300)


def process_finished(pid: int) -> bool:
    if pid <= 0:
        return True
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        return raw[raw.rfind(')') + 2:].split()[0] == 'Z'
    except OSError:
        return True


def wake_registered_watcher(pid: int) -> bool:
    registration = read(HEALTH / 'pretrain-watch-wakeup.json')
    if (registration.get('pid') != pid or registration.get('signal') != 'SIGUSR1'
            or not exact_process(pid, WATCHER)):
        return False
    try:
        os.kill(pid, signal.SIGUSR1)
        return True
    except ProcessLookupError:
        return False


def live_training_gpus(actions: dict) -> set[int]:
    owned = set()
    for name in actions:
        if not name.endswith('_queue'):
            continue
        jobs = read(HEALTH / f'{name[:-6]}-queue.json').get('jobs', {})
        owned.update(int(job['gpu']) for job in jobs.values()
                     if job.get('status') == 'RUNNING' and 'gpu' in job and exact_process(
                         int(job.get('pid', 0)), ROOT / 'pretrain/backends/torchtitan/launch.sh'))
    return owned


def queues_waiting_for_admission(actions: dict) -> list[str]:
    if allowed_gpu_indices(HEALTH).issubset(live_training_gpus(actions)):
        return []
    waiting = []
    for name, action in actions.items():
        if not name.endswith('_queue') or action.get('status') not in {'RUNNING', 'WAITING_EXTERNAL'}:
            continue
        jobs = read(HEALTH / f'{name[:-6]}-queue.json').get('jobs', {})
        eligible = (exact_process(int(action.get('pid', 0)), QUEUE)
                    or action.get('blocker') == 'experiment_gpu_allocation')
        pending = any(job.get('status') in {'PENDING', 'RETRY_WAIT'} or (
            job.get('status') == 'RETRY_EXHAUSTED' and job.get('failure_kind') != 'nonfinite_gradient'
            and int(action.get('attempts', 0)) < 3) for job in jobs.values())
        if eligible and pending:
            waiting.append(name)
    return waiting


def admitted_gpu_is_idle(excluded_gpus: set[int] | None = None) -> bool:
    try:
        allowed = allowed_gpu_indices(HEALTH) - (excluded_gpus or set())
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used',
                                 '--format=csv,noheader,nounits'], capture_output=True,
                                text=True, timeout=10)
        if result.returncode:
            return False
        return any(int(index.strip()) in allowed and int(memory.strip()) <= 1024
                   for line in result.stdout.splitlines() if line.strip()
                   for index, memory in [line.split(',')])
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return False


def wake_starved_queues(state: dict, actions: dict, watcher_pid: int, now: float) -> bool:
    waiting = queues_waiting_for_admission(actions)
    if not waiting or not admitted_gpu_is_idle(live_training_gpus(actions)):
        state.pop('idle_queue_since_epoch', None)
        state['queue_admission_health'] = 'NO_IDLE_RUNNABLE_QUEUE'
        return False
    since = state.setdefault('idle_queue_since_epoch', now)
    state['queue_admission_health'] = 'IDLE_WITH_PENDING_JOBS'
    if now - since < 300 or now - state.get('last_idle_queue_wakeup_epoch', 0) < 300:
        return False
    if not wake_registered_watcher(watcher_pid):
        return False
    state['last_idle_queue_wakeup_epoch'] = now
    state['events'].append({'epoch': now, 'action': 'wake_idle_runnable_queues', 'queues': waiting})
    return True


def save(value: dict) -> None:
    path = HEALTH / 'pipeline-guard.json'
    tmp = path.with_name(f'.{path.name}.{os.getpid()}')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(tmp, path)


def main() -> int:
    if (HEALTH / 'pretrain-user-pause.json').exists():
        print('User pause is active; automatic supervision is disabled.')
        return 0
    HEALTH.mkdir(parents=True, exist_ok=True)
    lock = (HEALTH / 'pipeline-guard.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('another pipeline guard already owns the lock')
    enable_subreaper()
    signal.signal(signal.SIGHUP, lambda *_: os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve())]))
    state = read(HEALTH / 'pipeline-guard.json') or {'restart_records': {}, 'events': []}
    state.update(pid=os.getpid(), status='RUNNING')
    child = None
    while True:
        now = time.time()
        if child is not None:
            child.poll()  # Reap the watcher only; trainers remain owned by queues.
        record = read(HEALTH / 'pretrain-watch-wakeup.json')
        pid = int(record.get('pid', 0))
        healthy = exact_process(pid, WATCHER)
        latest = HEALTH / 'pretrain-readiness-latest.json'
        allowance = max(3600, int(read(latest).get('monitoring', {}).get('next_poll_seconds', 900)) + 2700)
        age = now - max(latest.stat().st_mtime if latest.exists() else 0,
                        float(state.get('watcher_started_epoch', 0)))
        reason = 'watcher_missing' if not healthy else 'watcher_snapshot_timeout' if age > allowance else None
        reload_request = read(HEALTH / 'pretrain-watch-reload-request.json')
        intentional_reload = (bool(reload_request.get('request_id'))
                              and reload_request.get('request_id') != state.get('applied_reload_request'))
        if intentional_reload:
            reason = 'requested_code_reload'
        budget = state['restart_records'].setdefault('watcher', {})
        if reason and (intentional_reload or restart_allowed(budget, now)):
            if healthy:
                # Kill only a stuck coordinator. Existing queues/trainers are
                # deliberately retained and adopted by its replacement.
                os.kill(pid, signal.SIGKILL)
                if child is not None and child.pid == pid:
                    child.wait(timeout=30)
            log = (ROOT / 'report/logs/pretrain_readiness_watch.log').open('a')
            try:
                child = subprocess.Popen([sys.executable, str(WATCHER), '--auto-bootstrap',
                                          '--interval', '900', '--long-interval', '7200'],
                                         cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
            finally:
                log.close()
            pid = child.pid
            if intentional_reload:
                state['applied_reload_request'] = reload_request['request_id']
            else:
                budget['restart_epochs'].append(now)
            state['watcher_started_epoch'] = now
            state['events'].append({'epoch': now, 'action': 'restart_watcher', 'reason': reason, 'pid': pid})
        elif reason:
            state['status'] = 'RESTART_LIMIT_REACHED'
        else:
            state['status'] = 'RUNNING'
        actions = read(HEALTH / 'pretrain-automation-actions.json').get('actions', {})
        for name, action in actions.items():
            action_pid = int(action.get('pid', 0))
            notified = state.setdefault('completion_notifications', {})
            if (action.get('status') == 'RUNNING' and action_pid > 0
                    and process_finished(action_pid) and notified.get(name) != action_pid):
                if wake_registered_watcher(pid):
                    notified[name] = action_pid
                    state['events'].append({'epoch': now, 'action': 'wake_on_completion', 'name': name, 'pid': action_pid})
            if not name.endswith('_queue') or action.get('status') != 'RUNNING':
                continue
            queue_pid = int(action.get('pid', 0))
            queue_state = HEALTH / f'{name[:-6]}-queue.json'
            if not queue_state.exists() or now - queue_state.stat().st_mtime <= 600:
                continue
            if not exact_process(queue_pid, QUEUE):
                # Existing watcher handles bounded retries for exited queues.
                wake_registered_watcher(pid)
                continue
            budget = state['restart_records'].setdefault(name, {})
            if restart_allowed(budget, now):
                os.kill(queue_pid, signal.SIGKILL)
                budget['restart_epochs'].append(now)
                state['events'].append({'epoch': now, 'action': 'fence_stuck_queue', 'name': name, 'pid': queue_pid})
                wake_registered_watcher(pid)
        wake_starved_queues(state, actions, pid, now)
        # Adopted coordinator zombies are our children after watcher death.
        # Reap only explicit coordinator PIDs, never an active Popen trainer.
        reaped = reap_orphan_children({child.pid} if child is not None else {pid})
        state['orphan_children_reaped'] = int(state.get('orphan_children_reaped', 0)) + len(reaped)
        state.update(updated_epoch=now, watcher_pid=pid, watcher_deadline_seconds=allowance)
        state['events'] = state['events'][-100:]
        save(state)
        time.sleep(60)


if __name__ == '__main__':
    raise SystemExit(main())
