"""Muon 优化器（Newton-Schulz 5 步正交化，ZGCM/Keller Jordan 配方）。

2D 矩阵参数走 Muon；scalar / embedding / lm_head 走 AdamW（路由在 optim.py）。
等效学习率按 Moonshot 口径调整：lr_adj = lr × 0.2 × sqrt(max(fan_out, fan_in))。
"""
from __future__ import annotations

import math
from typing import Any

import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params: Any,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            mom = group["momentum"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None or p.ndim != 2:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1.0 - mom)
                d = g.lerp_(buf, mom) if group["nesterov"] else buf
                u = zeropower_via_newtonschulz5(d, steps=group["ns_steps"])
                u = u * max(1.0, p.size(0) / p.size(1)) ** 0.5
                lr_eff = lr * 0.2 * math.sqrt(max(p.size(0), p.size(1)))
                if wd:
                    p.mul_(1.0 - lr_eff * wd)
                p.add_(u.to(p.dtype), alpha=-lr_eff)
        return loss
