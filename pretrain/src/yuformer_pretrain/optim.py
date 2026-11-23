"""优化器构建：--optimizer adamw|muon 路由。

muon 模式（ZGCM 配方）：2D 矩阵（含 QKV/MLP 投影）走 Muon（NS5 正交化，
等效 lr = lr × 0.2 × sqrt(max(fan))）；scalar / embedding / lm_head 走 AdamW。
返回优化器列表（muon = [AdamW, Muon]），trainer/checkpoint 逐一处理，
与 FSDP.optim_state_dict 的 per-optimizer 语义对齐。
"""
from __future__ import annotations

import math

import torch

from .config import OptimConfig
from .muon import Muon


def is_muon_param(name: str, p: torch.Tensor) -> bool:
    if p.ndim != 2:
        return False
    if "embed" in name or "lm_head" in name:
        return False
    return True


def build_optimizers(model: torch.nn.Module, cfg: OptimConfig) -> list[torch.optim.Optimizer]:
    fused = torch.cuda.is_available()
    if cfg.optimizer == "adamw":
        return [torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, betas=cfg.adam_betas,
            weight_decay=cfg.weight_decay, fused=fused,
        )]
    muon_params = [p for n, p in model.named_parameters() if is_muon_param(n, p)]
    adamw_params = [p for n, p in model.named_parameters() if not is_muon_param(n, p)]
    adamw = torch.optim.AdamW(
        adamw_params, lr=cfg.lr, betas=cfg.adam_betas,
        weight_decay=cfg.weight_decay, fused=fused,
    )
    muon = Muon(
        muon_params, lr=cfg.lr, momentum=cfg.muon_momentum,
        nesterov=cfg.muon_nesterov, ns_steps=cfg.muon_ns_steps,
        weight_decay=cfg.weight_decay,
    )
    return [adamw, muon]


def set_lrs(optimizers: list[torch.optim.Optimizer], lr: float) -> None:
    for o in optimizers:
        for g in o.param_groups:
            g["lr"] = lr


def zero_grad(optimizers: list[torch.optim.Optimizer]) -> None:
    for o in optimizers:
        o.zero_grad(set_to_none=True)


def step(optimizers: list[torch.optim.Optimizer]) -> None:
    for o in optimizers:
        o.step()


def lr_at(step: int, total: int, cfg: OptimConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)
    if cfg.lr_schedule == "constant":
        return cfg.lr
    if cfg.lr_schedule == "wsd":
        stable_end = total - cfg.decay_steps
        if step < stable_end:
            return cfg.lr
        progress = (step - stable_end) / max(1, cfg.decay_steps)
    else:
        progress = (step - cfg.warmup_steps) / max(1, total - cfg.warmup_steps)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cos)
