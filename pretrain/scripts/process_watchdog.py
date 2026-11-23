"""Monitor semantic training progress without trusting launcher heartbeats."""
from __future__ import annotations

import ctypes
import os
import re
import signal
import time
from pathlib import Path


ANSI = re.compile(r'\x1b\[[0-9;]*m')
STEP = re.compile(r'\bstep:\s*(\d+)')
TPS = re.compile(r'\btps:\s*([\d,]+)')


def cleanup_attempt_processes(job: dict, run_dir: Path) -> list[int]:
    attempt = str(job.get('attempt_id', ''))
    if not re.fullmatch(r'[0-9a-f]{32}', attempt):
        return []
    required = {os.fsencode(f'YUFORMER_ATTEMPT_ID={attempt}'),
                os.fsencode(f'YUFORMER_PRETRAIN_OUT={run_dir}')}
    killed = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            raw = (entry / 'stat').read_text()
            fields = raw[raw.rfind(')') + 2:].split()
            if fields[0] == 'Z' or not required.issubset(set((entry / 'environ').read_bytes().split(b'\0'))):
                continue
            again = (entry / 'stat').read_text()
            if again[again.rfind(')') + 2:].split()[19] == fields[19]:
                os.kill(int(entry.name), signal.SIGKILL)
                killed.append(int(entry.name))
        except (OSError, ValueError, IndexError):
            pass
    return killed


def enable_subreaper() -> bool:
    return ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0


def reap_descendants(pids: list[int]) -> list[int]:
    pending = []
    for pid in pids:
        try:
            if os.waitpid(pid, os.WNOHANG)[0] == 0:
                pending.append(pid)
        except ChildProcessError:
            # A child can still belong to its live intermediate parent.
            if Path(f'/proc/{pid}').exists():
                pending.append(pid)
        except ProcessLookupError:
            pass
    return pending


def reap_orphan_children(exclude_pids: set[int]) -> list[int]:
    """Reap inherited zombies after a coordinator dies, without stealing Popen children."""
    try:
        children = Path(f'/proc/self/task/{os.getpid()}/children').read_text().split()
    except OSError:
        return []
    reaped = []
    for value in children:
        pid = int(value)
        if pid in exclude_pids:
            continue
        try:
            if os.waitpid(pid, os.WNOHANG)[0] == pid:
                reaped.append(pid)
        except ChildProcessError:
            pass
    return reaped


def observe_progress(job: dict, run_dir: Path, *, now: float | None = None,
                     stall_seconds: float = 1800, startup_seconds: float = 3600,
                     poll_seconds: float = 60) -> str | None:
    now = time.time() if now is None else now
    state = job.setdefault('watchdog', {})
    if now - float(state.get('last_checked_epoch', 0)) < poll_seconds:
        return None
    state['last_checked_epoch'] = now
    if state.get('attempt_id') != job.get('attempt_id'):
        state.clear()
        state.update(attempt_id=job.get('attempt_id'), last_checked_epoch=now,
                     last_progress_epoch=now, highest_step=-1)
    try:
        with (run_dir / 'queue.log').open('rb') as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 131072))
            tail = ANSI.sub('', handle.read().decode('utf-8', errors='replace'))
        # queue.log is append-only across attempts. Limit observations to the
        # current attempt's bytes, including after a queue supervisor restart.
        marker = f"attempt={job.get('attempt_id', '')}"
        tail = tail.rsplit(marker, 1)[-1] if marker in tail else tail
        steps = [int(value) for value in STEP.findall(tail)]
        current = steps[-1] if steps else -1
    except OSError:
        tail, current = '', -1
    if 'Checkpoint background process failed during initialization:' in tail:
        state['checkpoint_failure_detected_epoch'] = now
        return 'checkpoint_background_initialization_failure'
    if current > int(state.get('highest_step', -1)):
        state.update(highest_step=current, last_progress_epoch=now)
    tps = TPS.findall(tail)
    limit = startup_seconds if int(state['highest_step']) < 0 else stall_seconds
    if tps:
        try:
            import yaml
            cfg = yaml.safe_load((run_dir / 'resolved_config.yaml').read_text())
            train = cfg['train']
            tokens = (int(cfg['data']['seq_len']) * int(train['micro_batch_size'])
                      * int(train['grad_accum']) * int(cfg['parallel']['world_size']))
            expected_log_seconds = tokens * int(train['log_every']) / max(1, int(tps[-1].replace(',', '')))
            limit = max(limit, 3 * expected_log_seconds)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    state.update(timeout_seconds=limit, silent_seconds=max(0, now - float(state['last_progress_epoch'])))
    if state['silent_seconds'] >= limit:
        return 'startup_timeout' if int(state['highest_step']) < 0 else 'training_progress_timeout'
    return None
