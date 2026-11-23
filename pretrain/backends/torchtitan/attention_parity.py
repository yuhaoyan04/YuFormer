#!/usr/bin/env python3
"""Qualify BF16 cuDNN GQA against the mathematical SDPA forward/backward."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


def compiled_module_check():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "model/src"))
    from yuformer_model.attention import GQAAttention, precompute_rope_freqs
    from yuformer_model.config import ModelConfig
    config = ModelConfig(vocab_size=65536, hidden_size=1536, intermediate_size=4096,
                         num_layers=1, num_heads=24, num_kv_heads=6,
                         max_seq_len=4096, window_size=128, attention_every=6,
                         qk_norm=True, attention_output_gate=True)
    module = GQAAttention(config, 0).to(device="cuda", dtype=torch.bfloat16)
    module.sdpa_backend = "cudnn"
    compiled = torch.compile(module, backend="inductor", fullgraph=True)
    cos, sin = precompute_rope_freqs(config.head_dim, 4096, config.rope_base, config.rope_percent)
    cos, sin = cos.cuda(), sin.cuda()
    inputs = torch.randn(8, 4096, 1536, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    torch.cuda.reset_peak_memory_stats()
    compiled(inputs, cos, sin).float().square().mean().backward()
    module.zero_grad(set_to_none=True)
    inputs.grad = None
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        output = compiled(inputs, cos, sin)
        output.float().square().mean().backward()
        torch.cuda.synchronize()
    operators = sorted(event.key for event in profile.key_averages()
                       if "scaled_dot_product" in event.key)
    observed = (any("cudnn_attention" in key and "backward" not in key for key in operators)
                and any("cudnn_attention_backward" in key for key in operators))
    finite = (bool(output.isfinite().all() and inputs.grad.isfinite().all())
              and all(parameter.grad is not None and bool(parameter.grad.isfinite().all())
                      for parameter in module.parameters()))
    result = {"status": "PASS" if observed and finite else "FAIL", "batch_size": 8,
              "seq_len": 4096, "fullgraph": True, "finite_output_and_gradients": finite,
              "observed_operators": operators,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
    del module, compiled, inputs, output
    torch.cuda.empty_cache()
    return result


def error_metrics(actual, expected):
    delta = actual.float() - expected.float()
    scale = expected.float()
    return {
        "finite": bool(actual.isfinite().all() and expected.isfinite().all()),
        "relative_l2": float(delta.norm() / scale.norm().clamp_min(1e-12)),
        "max_abs_over_reference_max": float(delta.abs().max() / scale.abs().max().clamp_min(1e-12)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    torch.manual_seed(20261113)
    cases = []
    for length, window in ((127, 128), (128, 128), (129, 128), (4096, 128), (4096, None)):
        inputs = [torch.randn(shape, device="cuda", dtype=torch.bfloat16)
                  for shape in ((1, 24, length, 64), (1, 6, length, 64), (1, 6, length, 64))]
        gradient = torch.randn_like(inputs[0])
        mask = None
        if window is not None:
            position = torch.arange(length, device="cuda")
            distance = position[:, None] - position[None, :]
            mask = (distance >= 0) & (distance < window)
        outputs, gradients = [], []
        for backend in (SDPBackend.MATH, SDPBackend.CUDNN_ATTENTION):
            tensors = [value.detach().clone().requires_grad_() for value in inputs]
            with sdpa_kernel(backend):
                output = F.scaled_dot_product_attention(
                    *tensors, attn_mask=mask, is_causal=window is None, enable_gqa=True)
                output.backward(gradient)
            outputs.append(output.detach())
            gradients.append([value.grad.detach() for value in tensors])
            del tensors, output
        metrics = {"output": error_metrics(outputs[1], outputs[0])}
        metrics.update({name: error_metrics(actual, expected) for name, actual, expected
                        in zip(("dq", "dk", "dv"), gradients[1], gradients[0], strict=True)})
        passed = all(row["finite"] and row["relative_l2"] <= 0.02
                     and row["max_abs_over_reference_max"] <= 0.05 for row in metrics.values())
        cases.append({"seq_len": length, "window": window, "metrics": metrics,
                      "status": "PASS" if passed else "FAIL"})
        del inputs, gradient, outputs, gradients, mask, metrics
        torch.cuda.empty_cache()
    try:
        compiled = compiled_module_check()
    except Exception as exc:
        compiled = {"status": "FAIL", "error": repr(exc)}
    report = {"schema_version": 1, "status": "PASS" if all(
        row["status"] == "PASS" for row in cases) and compiled["status"] == "PASS" else "FAIL",
        "dtype": "bfloat16", "query_heads": 24, "kv_heads": 6, "head_dim": 64,
        "reference_backend": "math", "candidate_backend": "cudnn",
        "relative_l2_tolerance": 0.02, "max_abs_over_reference_max_tolerance": 0.05,
        "torch_version": torch.__version__, "cases": cases, "compiled_module": compiled}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
