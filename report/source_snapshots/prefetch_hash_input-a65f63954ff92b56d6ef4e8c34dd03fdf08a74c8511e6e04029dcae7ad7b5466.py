#!/usr/bin/env python3
"""Warm a bounded read-only window ahead of an existing full-file hash reader."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time


def identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--producer-pid', type=int, required=True)
    parser.add_argument('--phase', choices=('warmup', 'stable', 'anneal'), required=True)
    parser.add_argument('--seconds', type=float, default=180)
    args = parser.parse_args()
    if not 0 < args.seconds <= 7200:
        raise ValueError('bounded interval must be positive and <=7200 seconds')
    root = args.root.resolve()
    path = root / f'data/token_streams/pretrain-1t/.{args.phase}.tokens.bin.tmp-{args.producer_pid}'
    proc = Path(f'/proc/{args.producer_pid}')
    command = [os.fsdecode(v) for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
    if (str(root/'pretrain/scripts/materialize_tokens.py') not in command
            or '--name' not in command or command[command.index('--name')+1] != args.phase):
        raise ValueError('actual hashing producer phase does not match')
    start_ticks = (proc/'stat').read_text().split(') ', 1)[1].split()[19]
    expected = identity(path.stat())
    size = expected[2]

    def cursor():
        current = (proc/'stat').read_text().split(') ', 1)[1].split()
        if current[0] == 'Z' or current[19] != start_ticks:
            raise ValueError('original hashing producer ended')
        candidates = []
        for fd in (proc/'fd').iterdir():
            try:
                if os.readlink(fd) == str(path):
                    info = dict(line.split(':', 1) for line in (proc/'fdinfo'/fd.name).read_text().splitlines())
                    if int(info['flags'].strip(), 8) & os.O_ACCMODE == os.O_RDONLY:
                        candidates.append(int(info['pos']))
            except FileNotFoundError:
                pass
        if len(candidates) != 1:
            raise ValueError('requires exactly one observed full-file hash reader')
        return candidates[0]

    def read_only(offset, length):
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            if identity(os.fstat(descriptor)) != expected:
                raise ValueError('hash input identity changed before read')
            remaining, count = length, 0
            while remaining:
                block = os.pread(descriptor, min(8 << 20, remaining), offset+count)
                if not block:
                    raise ValueError('short immutable hash input')
                count += len(block)
                remaining -= len(block)
            if identity(os.fstat(descriptor)) != expected:
                raise ValueError('hash input changed during read')
            return count
        finally:
            os.close(descriptor)

    initial = cursor()
    if not 0 < initial < size * .9:
        raise ValueError('hash reader must be active and before 90% of the file')
    begun = time.monotonic()
    state = {'pid': os.getpid(), 'producer_pid': args.producer_pid, 'phase': args.phase,
             'status': 'BOUNDED_READONLY_HASH_PREFETCH_RUNNING',
             'source_bytes_modified': False, 'source_code_modified': False,
             'producer_interrupted': False, 'workers': 4, 'ahead_bytes': 2 << 30,
             'block_bytes': 64 << 20, 'read_buffer_bytes': 8 << 20,
             'initial_hash_cursor': initial, 'producer_starttime_ticks': start_ticks,
             'prefetched_bytes': 0, 'maximum_pending_reads': 0,
             'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    health = root/f'data/health/{args.phase}-sha-readonly-prefetch.json'
    futures, completed = {}, set()
    position = initial

    def save():
        elapsed = time.monotonic()-begun
        state.update(updated_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=elapsed,
                     hash_cursor=position, observed_hash_mib_s=(position-initial)/(1 << 20)/elapsed)
        temporary = health.with_name(health.name+f'.tmp-{os.getpid()}')
        temporary.write_text(json.dumps(state, sort_keys=True, indent=2)+'\n')
        os.replace(temporary, health)

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            while time.monotonic()-begun < args.seconds:
                new_position = cursor()
                if new_position < position or new_position >= size*.9:
                    state['status'] = 'READER_REWOUND_OR_NEAR_COMPLETION'
                    break
                position = new_position
                for index, future in list(futures.items()):
                    if future.done():
                        state['prefetched_bytes'] += future.result()
                        completed.add(index)
                        del futures[index]
                unit = state['block_bytes']
                completed = {index for index in completed if (index+1)*unit > position}
                first = position//unit+1
                for index in range(first, min((position+state['ahead_bytes'])//unit, size//unit)):
                    if len(futures) >= 4:
                        break
                    if index not in completed and index not in futures:
                        futures[index] = pool.submit(read_only, index*unit, min(unit, size-index*unit))
                state['maximum_pending_reads'] = max(state['maximum_pending_reads'], len(futures))
                save()
                time.sleep(.2)
            for future in futures.values():
                state['prefetched_bytes'] += future.result()
        if state['status'] == 'BOUNDED_READONLY_HASH_PREFETCH_RUNNING':
            state['status'] = 'BOUNDED_INTERVAL_COMPLETE'
    except Exception as error:
        state.update(status='PREFETCH_STOPPED', error=repr(error))
    if identity(path.stat()) != expected:
        raise ValueError('hash input identity changed before final observation')
    save()
    print(json.dumps(state, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
