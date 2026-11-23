#!/usr/bin/env python3
"""Compare the TorchTitan wrapper with the research model on one GPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from yuformer_ext.model import YuFormerTitanModel
from yuformer_model.config import ModelConfig
from yuformer_model.model import YuFormerModel
from yuformer_model.layers import RMSNorm
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-configs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", default="e1-50m")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal numerical parity gate")
    torch.manual_seed(20261021)
    torch.cuda.manual_seed_all(20261021)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")
    cfg = ModelConfig.from_preset(args.preset, args.model_configs)
    reference = YuFormerModel(cfg)
    wrapper_cfg = YuFormerTitanModel.Config(preset=args.preset, config_path=str(args.model_configs))
    with torch.device("meta"):
        candidate = YuFormerTitanModel(wrapper_cfg)
    candidate.to_empty(device="cpu")
    with torch.no_grad():
        for module in candidate.modules():
            if isinstance(module, RMSNorm):
                module.weight.fill_(float("nan"))
        torch.manual_seed(20261021)
        reference._init_parameters()
        torch.manual_seed(20261021)
        candidate.init_states(buffer_device=torch.device("cpu"))
    init_delta = max(float((left - right).abs().max())
                     for left, right in zip(reference.parameters(), candidate.parameters(), strict=True))
    norms_initialized = all(torch.equal(module.weight, torch.ones_like(module.weight))
                            for module in candidate.modules() if isinstance(module, RMSNorm))
    dtype = getattr(torch, args.dtype)
    reference = reference.to(device=device, dtype=dtype)
    candidate = candidate.to(device=device, dtype=dtype)
    tokens = torch.randint(0, cfg.vocab_size, (2, 128), device=device)
    labels = torch.roll(tokens, shifts=-1, dims=1)
    reference.zero_grad(set_to_none=True)
    candidate.zero_grad(set_to_none=True)
    ref_logits = reference(tokens)
    cand_logits = candidate(tokens).float()
    ref_loss = F.cross_entropy(ref_logits.view(-1, cfg.vocab_size), labels.view(-1))
    cand_loss = F.cross_entropy(cand_logits.view(-1, cfg.vocab_size), labels.view(-1))
    ref_loss.backward()
    cand_loss.backward()
    logits_delta = float((ref_logits - cand_logits).abs().max())
    loss_delta = float((ref_loss - cand_loss).abs())
    grad_delta = max(
        float((left.grad - right.grad).abs().max())
        for left, right in zip(reference.parameters(), candidate.core.parameters(), strict=True)
        if left.grad is not None and right.grad is not None
    )
    expected_grads = {name: parameter.grad.clone()
                      for name, parameter in candidate.core.named_parameters()
                      if parameter.grad is not None}
    candidate.zero_grad(set_to_none=True)
    candidate._skip_lm_head = True
    chunked = ChunkedLossWrapper.Config(
        num_chunks=8, loss_fn=CrossEntropyLoss.Config(global_vocab_size=cfg.vocab_size)
    ).build()
    chunked.set_lm_head(candidate.lm_head)
    chunk_loss, _ = chunked(candidate(tokens), labels,
                          global_valid_tokens=torch.tensor(labels.numel(),device=device))
    chunk_loss.backward()
    chunk_grad_delta = max(float((parameter.grad - expected_grads[name]).abs().max())
                           for name, parameter in candidate.core.named_parameters()
                           if name in expected_grads)
    grad_scale = max(float(value.abs().max()) for value in expected_grads.values())
    squared_delta = sum(float((parameter.grad.float() - expected_grads[name].float()).square().sum())
                        for name, parameter in candidate.core.named_parameters() if name in expected_grads)
    squared_reference = sum(float(value.float().square().sum()) for value in expected_grads.values())
    relative_l2 = (squared_delta / max(squared_reference, 1e-30)) ** 0.5
    tolerances = {"logits_max_abs": 1e-5, "loss_abs": 1e-6, "grad_max_abs": 1e-5}
    metrics = {"logits_max_abs": logits_delta, "loss_abs": loss_delta, "grad_max_abs": grad_delta}
    # BF16 chunks change gradient accumulation order. Bound rounding against
    # gradient scale and require a small full-vector relative error as well.
    epsilon = torch.finfo(dtype).eps
    tolerances.update(initialization_max_abs=0.0, chunk_loss_abs=1e-5,
                      chunk_grad_max_abs=max(1e-6, 4 * epsilon * grad_scale),
                      chunk_grad_relative_l2=(2 * epsilon if dtype == torch.bfloat16 else 1e-5))
    metrics.update(initialization_max_abs=init_delta,chunk_loss_abs=float((chunk_loss-cand_loss).abs()),
                   chunk_grad_max_abs=chunk_grad_delta, chunk_grad_relative_l2=relative_l2)
    status = ("PASS" if norms_initialized and all(metrics[key] <= limit for key, limit in tolerances.items())
              else "FAIL")
    result = {"schema_version": 1, "status": status, "preset": args.preset,
              "dtype": args.dtype, "seed": 20261021, "metrics": metrics,
              "tolerances": tolerances}
    result["normalization_weights_initialized"] = norms_initialized
    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
