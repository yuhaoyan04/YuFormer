#!/usr/bin/env python3
"""Bounded read-only cache warming ahead of an existing serial NPZ unpacker."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--producer-pid', type=int, required=True)
    p.add_argument('--phase', choices=('warmup', 'stable', 'anneal'), required=True)
    p.add_argument('--expected-parts', type=int, required=True)
    p.add_argument('--seconds', type=float, default=300)
    p.add_argument('--ahead', type=int, default=8)
    args = p.parse_args()
    root = args.root.resolve()
    if not 2 <= args.ahead <= 8 or not 0 < args.seconds <= 7200 or args.expected_parts < 1:
        raise ValueError('prefetch requires 2..8 ahead parts, positive <=7200 seconds, and known encoded count')
    proc = Path(f'/proc/{args.producer_pid}')
    command = [os.fsdecode(v) for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
    script = str(root/'pretrain/scripts/materialize_tokens.py')
    if (script not in command or '--name' not in command
            or command[command.index('--name')+1] != args.phase
            or '--out-dir' not in command
            or Path(command[command.index('--out-dir')+1]).resolve() != root/'data/token_streams/pretrain-1t'):
        raise ValueError('producer command does not match the requested actual phase/output')
    producer_start = (proc/'stat').read_text().split(') ', 1)[1].split()[19]
    raw = root/f'data/token_streams/pretrain-1t/..{args.phase}.tokens.bin.tmp-{args.producer_pid}.merge-parts'
    health = root/f'data/health/{args.phase}-merge-readonly-prefetch-{args.producer_pid}.json'
    first = json.loads((raw/'part000000.json').read_bytes())
    source = Path(first['source']).parent
    source = source.resolve()
    if not source.is_relative_to(root/'data/token_streams/pretrain-1t'):
        raise ValueError('encoded source part directory is outside the actual token output')
    parts = sorted(source.glob('part*.npz'))
    total = len(parts)
    if total != args.expected_parts or [v.name for v in parts] != [f'part{i:06d}.npz' for i in range(total)]:
        raise ValueError('expected complete contiguous encoded population is not ready')
    frontier = 0
    while frontier < total and (raw/f'part{frontier:06d}.json').is_file():
        frontier += 1
    initial = frontier
    initial_time = time.monotonic()
    state = {'pid': os.getpid(), 'producer_pid': args.producer_pid, 'phase': args.phase, 'producer_starttime_ticks': producer_start, 'status': 'READONLY_PREFETCH_RUNNING',
             'initial_prepared_parts': initial, 'source_bytes_modified': False,
             'source_code_modified': False, 'producer_interrupted': False,
             'workers': 2, 'ahead_parts': args.ahead, 'total_parts': total,
             'prefetched_files': 0, 'bytes_read': 0, 'maximum_pending_files': 0,
             'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    completed, futures = set(), {}

    def identity(stat):
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def read_only(index):
        path = source/f'part{index:06d}.npz'
        with path.open('rb') as handle:
            before = identity(os.fstat(handle.fileno()))
            count = 0
            while True:
                block = handle.read(8 << 20)
                if not block:
                    break
                count += len(block)
            if identity(os.fstat(handle.fileno())) != before or identity(path.stat()) != before:
                raise ValueError('encoded source part changed while prefetching: '+str(path))
        return count

    def save():
        elapsed = time.monotonic()-initial_time
        state.update(updated_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=elapsed,
            prepared_parts=frontier,
            observed_parts_per_minute=(frontier-initial)*60/elapsed if elapsed else None)
        temporary = health.with_name(health.name+f'.tmp-{os.getpid()}')
        temporary.write_text(json.dumps(state, sort_keys=True, indent=2)+'\n')
        os.replace(temporary, health)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            while time.monotonic()-initial_time < args.seconds:
                proc = Path(f'/proc/{args.producer_pid}/stat')
                if (not proc.exists() or proc.read_text().split(') ', 1)[1].split()[0] == 'Z'
                        or proc.read_text().split(') ', 1)[1].split()[19] != producer_start):
                    state['status'] = 'PRODUCER_ENDED'
                    break
                while frontier < total and (raw/f'part{frontier:06d}.json').is_file():
                    frontier += 1
                if frontier >= total:
                    state['status'] = 'ALL_RAW_PARTS_PREPARED'
                    break
                for index, future in list(futures.items()):
                    if index < frontier and future.cancel():
                        del futures[index]
                    elif future.done():
                        state['bytes_read'] += future.result()
                        state['prefetched_files'] += 1
                        completed.add(index)
                        del futures[index]
                for index in range(frontier+1, min(frontier+args.ahead, total)):
                    if len(futures) >= args.ahead:
                        break
                    if index not in completed and index not in futures:
                        futures[index] = pool.submit(read_only, index)
                state['maximum_pending_files'] = max(state['maximum_pending_files'], len(futures))
                save()
                time.sleep(2)
            for future in futures.values():
                state['bytes_read'] += future.result()
                state['prefetched_files'] += 1
        if state['status'] == 'READONLY_PREFETCH_RUNNING':
            state['status'] = 'BOUNDED_PREFETCH_INTERVAL_COMPLETE'
    except Exception as error:
        state.update(status='PREFETCH_STOPPED', error=repr(error))
    save()
    print(json.dumps(state, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
