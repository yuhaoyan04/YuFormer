#!/usr/bin/env python3
"""Compiled actual attention-module parity for the optional hybrid backend."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'model/src'))
from yuformer_model.attention import GQAAttention, precompute_rope_freqs
from yuformer_model.config import ModelConfig
from attention_parity import error_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rank', type=int, choices=range(8), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    heads = (24, 32, 40, 52)[args.rank//2]
    master_fp32 = bool(args.rank % 2)
    seed = int.from_bytes(hashlib.blake2b(f'flex-module-20261114:{args.rank}'.encode(), digest_size=8).digest(), 'little')
    torch.manual_seed(seed)
    paths = [Path(__file__), ROOT/'model/src/yuformer_model/attention.py',
             ROOT/'pretrain/backends/torchtitan/attention_parity.py']
    pinned = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    report = dict(status='RUNNING', purpose='ACTUAL_COMPILED_ATTENTION_MODULE_PARITY',
        physical_gpu=args.rank, heads=heads, master_fp32=master_fp32, seed=seed,
        torch_version=str(torch.__version__), sources=pinned,
        relative_l2_tolerance=.02, max_abs_over_reference_max_tolerance=.05, cases=[])
    try:
        config = ModelConfig(hidden_size=heads*64, intermediate_size=4*heads*64,
            num_layers=6, num_heads=heads, num_kv_heads=heads//4,
            qk_norm=True, attention_output_gate=True, window_size=128, attention_every=6)
        for layer, batch, length in ((0, 2, 129), (5, 2, 129), (0, 8, 4096)):
            reference = GQAAttention(config, layer).cuda()
            if not master_fp32:
                reference.bfloat16()
            reference.sdpa_backend = 'cudnn'
            candidate = copy.deepcopy(reference)
            candidate.sdpa_backend = 'cudnn_flex_swa'
            candidate.prepare_flex_swa(length, torch.device('cuda', 0))
            compiled = torch.compile(candidate, fullgraph=True, dynamic=False)
            cos, sin = precompute_rope_freqs(config.head_dim, length, config.rope_base, config.rope_percent)
            cos, sin = cos.cuda(), sin.cuda()
            source = torch.randn(batch, length, config.hidden_size, device='cuda', dtype=torch.bfloat16)
            gradient = torch.randn_like(source)
            outputs, input_grads, parameter_grads = [], [], []
            for module in (reference, compiled):
                value = source.detach().requires_grad_()
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    output = module(value, cos, sin)
                output.backward(gradient)
                outputs.append(output.detach())
                input_grads.append(value.grad.detach())
                parameter_grads.append({name.removeprefix('_orig_mod.'): p.grad.detach()
                                        for name, p in module.named_parameters()})
            metrics = {'output': error_metrics(outputs[1], outputs[0]),
                       'input_gradient': error_metrics(input_grads[1], input_grads[0])}
            metrics.update({name: error_metrics(parameter_grads[1][name], value)
                            for name, value in parameter_grads[0].items()})
            passed = all(row['finite'] and row['relative_l2'] <= .02
                         and row['max_abs_over_reference_max'] <= .05 for row in metrics.values())
            report['cases'].append(dict(layer=layer, is_full=config.is_full_attn(layer), batch_size=batch,
                seq_len=length, status='PASS' if passed else 'FAIL', metrics=metrics))
            if not passed:
                raise ValueError('compiled actual attention module parity failed')
            del reference, candidate, compiled, outputs, input_grads, parameter_grads, source, gradient, value, output
            torch.cuda.empty_cache()
        if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != sha for name, sha in pinned.items()):
            raise ValueError('qualified module source changed during parity run')
        report['status'] = 'PASS'
    except Exception:
        report.update(status='FAIL', error=traceback.format_exc())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name+f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
    os.replace(temporary, args.output)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
