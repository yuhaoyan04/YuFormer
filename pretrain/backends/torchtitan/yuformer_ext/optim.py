"""AdamW parameter partition and a fail-fast finite-gradient update guard."""
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig


class GuardedOptimizers(OptimizersContainer):
    @dataclass(kw_only=True, slots=True)
    class Config(OptimizersContainer.Config):
        pass

    def step(self, *args, **kwargs):
        gradients = [
            parameter.grad.to_local() if isinstance(parameter.grad, DTensor) else parameter.grad
            for part in self.model_parts for parameter in part.parameters()
            if parameter.grad is not None
        ]
        if not gradients:
            raise RuntimeError("optimizer update has no gradients")
        norms = torch._foreach_norm(gradients, float("inf"))
        finite = torch.isfinite(torch.stack(norms)).all().to(torch.int32)
        if dist.is_initialized():
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite.item()):
            raise RuntimeError("non-finite gradients; optimizer update refused")
        return super().step(*args, **kwargs)


def adamw_config(optim: dict) -> GuardedOptimizers.Config:
    kwargs = {
        "lr": float(optim["lr"]), "betas": tuple(optim.get("adam_betas", (0.9, 0.95))),
        "eps": float(optim.get("adam_eps", 1e-8)),
    }
    return GuardedOptimizers.Config(param_groups=[
        ParamGroupConfig(pattern=r"norm.*\.weight$", optimizer_name="AdamW",
                         optimizer_kwargs={**kwargs, "weight_decay": 0.0}),
        ParamGroupConfig(pattern=r".*", optimizer_name="AdamW", optimizer_kwargs={
            **kwargs, "weight_decay": float(optim.get("weight_decay", 0.1)),
        }),
    ])
