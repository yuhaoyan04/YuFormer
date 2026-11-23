#!/usr/bin/env python3
"""Independent SWA operator qualification; never changes the training backend."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'experiments/e2_architecture/results/flex_swa_20261114'
HEALTH = ROOT/'data/health/flex-swa-pilot.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2)+'\n')
    os.replace(temporary, path)


def worker(rank):
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    heads = (24, 32, 40, 52)[rank//2]
    layout = ('rope_qk_contiguous', 'all_transposed')[rank % 2]
    seed = int.from_bytes(hashlib.blake2b(f'flex-swa-20261114:{rank}'.encode(), digest_size=8).digest(), 'little')
    torch.manual_seed(seed)
    report = dict(status='RUNNING', purpose='OPERATOR_RESEARCH_NOT_PRODUCTION_QUALIFICATION',
        production_qualified=False, pid=os.getpid(), physical_gpu=rank, heads=heads,
        kv_heads=heads//4, head_dim=64, window=128, dtype='bfloat16', layout=layout,
        seed=seed, torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
        device=torch.cuda.get_device_name(), source_sha256=digest(__file__),
        environment={k: os.environ[k] for k in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS',
            'TORCHINDUCTOR_COMPILE_THREADS', 'TORCHINDUCTOR_CACHE_DIR')}, cases=[])
    path = OUT/f'gpu-{rank}.json'
    publish(path, report)
    try:
        compiled = torch.compile(flex_attention, fullgraph=True, dynamic=False)

        def mask_mod(b, h, q, kv):
            return (q >= kv) & (q-kv < 128)

        def inputs(batch, length):
            values = [torch.randn(batch, length, h, 64, device='cuda', dtype=torch.bfloat16).transpose(1, 2)
                      for h in (heads, heads//4, heads//4)]
            if layout == 'rope_qk_contiguous':
                values[:2] = [value.contiguous() for value in values[:2]]
            return [value.detach().requires_grad_() for value in values]

        def errors(actual, expected):
            delta, reference = actual.float()-expected.float(), expected.float()
            return dict(finite=bool(actual.isfinite().all() and expected.isfinite().all()),
                relative_l2=float(delta.norm()/reference.norm().clamp_min(1e-12)),
                max_abs_over_reference_max=float(delta.abs().max()/reference.abs().max().clamp_min(1e-12)))

        for length in (65, 128, 129, 257):
            values = inputs(2, length)
            gradient = torch.randn_like(values[0])
            position = torch.arange(length, device='cuda')
            distance = position[:, None]-position[None, :]
            dense = (distance >= 0) & (distance < 128)
            blocks = create_block_mask(mask_mod, None, None, length, length, device='cuda')
            outputs, gradients = {}, {}
            for backend in ('math_fp32', 'cudnn_bf16', 'flex_bf16'):
                tensors = [v.detach().float().requires_grad_() if backend == 'math_fp32'
                           else v.detach().requires_grad_() for v in values]
                if backend == 'flex_bf16':
                    output = compiled(*tensors, block_mask=blocks, enable_gqa=True)
                else:
                    with sdpa_kernel(SDPBackend.MATH if backend == 'math_fp32' else SDPBackend.CUDNN_ATTENTION):
                        output = F.scaled_dot_product_attention(*[v.contiguous() for v in tensors],
                            attn_mask=dense, enable_gqa=True, dropout_p=0.)
                output.backward(gradient.to(output.dtype))
                outputs[backend] = output.detach()
                gradients[backend] = [v.grad.detach() for v in tensors]
            metrics = {}
            for actual, reference in (('cudnn_bf16', 'math_fp32'), ('flex_bf16', 'math_fp32'), ('flex_bf16', 'cudnn_bf16')):
                comparison = {'output': errors(outputs[actual], outputs[reference])}
                comparison.update({name: errors(a, b) for name, a, b in
                    zip(('dq', 'dk', 'dv'), gradients[actual], gradients[reference], strict=True)})
                metrics[actual+'_vs_'+reference] = comparison
            passed = all(v['finite'] and v['relative_l2'] <= .02 and v['max_abs_over_reference_max'] <= .05
                         for comparison in metrics.values() for v in comparison.values())
            report['cases'].append(dict(seq_len=length, batch_size=2, status='PASS' if passed else 'FAIL', metrics=metrics))
            publish(path, report)
            if not passed:
                raise ValueError('forward/backward parity failed without widening existing tolerances')
            del values, gradient, outputs, gradients, tensors, output, blocks

        values = inputs(8, 4096)
        gradient = torch.randn_like(values[0])
        position = torch.arange(4096, device='cuda')
        distance = position[:, None]-position[None, :]
        dense = (distance >= 0) & (distance < 128)
        blocks = create_block_mask(mask_mod, None, None, 4096, 4096, device='cuda')
        timings = {}
        for backend in ('cudnn_bf16', 'flex_bf16'):
            def step():
                for value in values:
                    value.grad = None
                if backend == 'flex_bf16':
                    output = compiled(*values, block_mask=blocks, enable_gqa=True)
                else:
                    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                        output = F.scaled_dot_product_attention(*[v.contiguous() for v in values],
                            attn_mask=dense, enable_gqa=True, dropout_p=0.)
                output.backward(gradient)
                return output
            for _ in range(5):
                step()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            milliseconds = []
            for _ in range(20):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                output = step()
                end.record()
                end.synchronize()
                milliseconds.append(start.elapsed_time(end))
            finite = bool(output.isfinite().all()) and all(bool(v.grad.isfinite().all()) for v in values)
            if not finite:
                raise ValueError('nonfinite measured output or gradient')
            timings[backend] = dict(median_ms=statistics.median(milliseconds), samples_ms=milliseconds,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(), finite=True)
            report['timings'] = timings
            publish(path, report)
        report.update(status='PASS_OPERATOR_ONLY', batch_size=8, seq_len=4096,
            speedup=timings['cudnn_bf16']['median_ms']/timings['flex_bf16']['median_ms'],
            relative_l2_tolerance=.02, max_abs_over_reference_max_tolerance=.05)
    except Exception:
        report.update(status='FAIL', error=traceback.format_exc())
    publish(path, report)
    return 0 if report['status'] == 'PASS_OPERATOR_ONLY' else 1


def parent():
    import benchmark_model_sizes as bench
    sys.path.insert(0, str(ROOT/'pretrain/src'))
    from yuformer_pretrain.lease import RunLease
    OUT.mkdir(parents=True, exist_ok=True)
    source = digest(__file__)
    with RunLease(OUT, 'eight-gpu-flex-swa-operator-pilot'):
        bench.require_idle()
        if any(OUT.glob('gpu-*.json')):
            raise ValueError('refusing to overwrite operator pilot results')
        state = dict(status='RUNNING', pid=os.getpid(), purpose='OPERATOR_RESEARCH_NOT_PRODUCTION_QUALIFICATION',
            production_qualified=False, source_sha256=source, children=[], updated_at=bench.now())
        processes, handles = [], []
        started = time.monotonic()
        try:
            for rank in range(8):
                environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), OMP_NUM_THREADS='1',
                    TORCHINDUCTOR_COMPILE_THREADS='2', TORCHINDUCTOR_CACHE_DIR='/tmp/yuformer-flex-swa-pilot-v1')
                handle = (OUT/f'gpu-{rank}.log').open('w')
                handles.append(handle)
                process = subprocess.Popen([str(bench.PYTHON), __file__, '--rank', str(rank)], cwd=ROOT,
                    env=environment, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                processes.append(process)
                state['children'].append(dict(physical_gpu=rank, pid=process.pid))
            while any(p.poll() is None for p in processes):
                if digest(__file__) != source:
                    raise ValueError('operator pilot source changed while running')
                if time.monotonic()-started > 1200:
                    raise TimeoutError('operator pilot exceeded 20 minutes')
                state.update(updated_at=bench.now(), elapsed_seconds=time.monotonic()-started,
                    completed=sum(p.poll() is not None for p in processes))
                publish(HEALTH, state)
                time.sleep(15)
            results = [json.loads((OUT/f'gpu-{rank}.json').read_text()) for rank in range(8)]
            state.update(status='PASS_OPERATOR_ONLY' if all(p.returncode == 0 for p in processes) else 'FAIL',
                updated_at=bench.now(), elapsed_seconds=time.monotonic()-started, results=results)
            publish(HEALTH, state)
        finally:
            for process in processes:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
            for handle in handles:
                handle.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', type=int, choices=range(8))
    args = parser.parse_args()
    if args.rank is None:
        parent()
    else:
        raise SystemExit(worker(args.rank))
