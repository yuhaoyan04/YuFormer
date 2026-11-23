"""CPU-only qualification of concurrent process DCP saves on existing rendezvous stores."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker(output: Path) -> int:
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    import torch.distributed.checkpoint._async_process_executor as executor
    from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType
    dist.init_process_group('gloo')
    rank = dist.get_rank()
    # Simulate the original race: a purportedly free port is already the
    # live rendezvous listener. Prefix-store mode must never bind it again.
    executor.get_free_port = lambda: int(os.environ['MASTER_PORT'])
    info = executor._ProcessGroupInitInfo()
    assert info.use_prefix_store and info.tcp_store_master_port == int(os.environ['MASTER_PORT'])
    for step in (1, 2):
        expected = torch.arange(64, dtype=torch.float32) + step
        state = {'weight': expected.clone(), 'step': torch.tensor([step])}
        directory = output / f'step-{step}'
        future = dcp.async_save(state, checkpoint_id=directory,
                                async_checkpointer_type=AsyncCheckpointerType.PROCESS)
        future.result(timeout=120)
        loaded = {'weight': torch.zeros_like(expected), 'step': torch.zeros(1, dtype=torch.int64)}
        dcp.load(loaded, checkpoint_id=directory)
        assert torch.equal(loaded['weight'], expected) and loaded['step'].item() == step
    daemon = executor._CHECKPOINT_PROCESS
    if daemon is not None:
        daemon.__del__()
        executor._CHECKPOINT_PROCESS = None
    output.mkdir(parents=True, exist_ok=True)
    (output / f'rank-{rank}.json').write_text(json.dumps({
        'status': 'PASS', 'rank': rank, 'world_size': dist.get_world_size(),
        'saves_and_loads': 2, 'prefix_store': True, 'forced_free_port_collision': True,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}, sort_keys=True) + '\n')
    dist.destroy_process_group()
    return 0


def qualify(output: Path, jobs: int, world: int) -> int:
    from process_watchdog import enable_subreaper, reap_descendants
    from process_control import terminate_process_tree
    enable_subreaper()
    output.mkdir(parents=True, exist_ok=True)
    children, pending = [], []
    try:
        for index in range(jobs):
            run = output / f'job-{index}'
            log = (output / f'job-{index}.log').open('w')
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'DCP_USE_PREFIX_STORE': '1', 'OMP_NUM_THREADS': '1'}
            command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
                       f'--nproc-per-node={world}', str(Path(__file__).resolve()), '--worker', '--output', str(run)]
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            log.close(); children.append(process)
        deadline = time.monotonic() + 240
        while any(p.poll() is None for p in children) and time.monotonic() < deadline:
            time.sleep(1)
        rows = []
        for index, process in enumerate(children):
            if process.poll() is None:
                victims = terminate_process_tree(process.pid)
                process.wait(timeout=30); pending.extend(pid for pid in victims if pid != process.pid)
            ranks = [json.loads(p.read_text()) for p in sorted((output / f'job-{index}').glob('rank-*.json'))]
            rows.append({'job': index, 'returncode': process.returncode, 'ranks': ranks,
                         'status': 'PASS' if process.returncode == 0 and len(ranks) == world
                         and all(row['status'] == 'PASS' for row in ranks) else 'FAIL'})
        report = {'status': 'PASS' if all(row['status'] == 'PASS' for row in rows) else 'FAIL',
                  'jobs': jobs, 'world_size_each': world, 'rows': rows,
                  'scope': 'CPU Gloo checkpoint transport, repeated async process saves and exact reloads; no GPU compute'}
        (output / 'report.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
        print(json.dumps({'status': report['status'], 'jobs': jobs, 'world_size_each': world,
                          'report': str(output / 'report.json')}), flush=True)
        return 0 if report['status'] == 'PASS' else 1
    finally:
        for process in children:
            if process.poll() is None:
                pending.extend(pid for pid in terminate_process_tree(process.pid) if pid != process.pid)
                process.wait(timeout=30)
        reap_descendants(pending)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--world-size', type=int, default=1)
    args = parser.parse_args()
    raise SystemExit(worker(args.output.resolve()) if args.worker else qualify(args.output.resolve(), args.jobs, args.world_size))
