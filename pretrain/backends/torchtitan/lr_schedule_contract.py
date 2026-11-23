#!/usr/bin/env python3
"""Verify the production WSD boundaries against TorchTitan's real scheduler."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import torch
import yaml

from torchtitan.components.optimizer import LRSchedulersContainer, OptimizersContainer, default_adamw


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def verify(config_path: Path) -> dict:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data, train, optim, parallel = cfg["data"], cfg["train"], cfg["optim"], cfg["parallel"]
    if optim.get("lr_schedule") != "wsd":
        raise ValueError("production config must use WSD")
    total = int(train["expected_iters"])
    warmup = int(optim["warmup_steps"])
    decay = int(optim["decay_steps"])
    samples_per_step = (int(train["micro_batch_size"]) * int(train["grad_accum"])
                        * int(parallel["world_size"]))
    limits = [int(value) for value in data["stage_sample_limits"]]
    stage_steps = [value // samples_per_step for value in limits]
    arithmetic = {
        "three_stages": len(stage_steps) == 3,
        "optimizer_step_aligned": all(value % samples_per_step == 0 for value in limits),
        "stage_steps_close": sum(stage_steps) == total,
        "warmup_matches_first_stage": stage_steps[0] == warmup,
        "decay_matches_last_stage": stage_steps[-1] == decay,
    }

    model = torch.nn.Linear(1, 1, bias=False)
    optimizer_config = default_adamw(lr=1.0)
    optimizer_config.implementation = "for-loop"
    optimizers = OptimizersContainer(optimizer_config, model_parts=[model])
    scheduler_config = LRSchedulersContainer.Config(
        warmup_steps=warmup,
        total_steps=total,
        decay_ratio=decay / total,
        decay_type="cosine",
        min_lr_factor=float(optim["min_lr_ratio"]),
    )
    schedulers = scheduler_config.build(optimizers=optimizers, training_steps=total)
    scheduler = next(iter(schedulers))
    lr_lambda = scheduler.lr_lambdas[0]
    stable_end = warmup + stage_steps[1] - 1
    first_decay = stable_end + 1
    factors = {
        "warmup_first": float(lr_lambda(0)),
        "warmup_last": float(lr_lambda(warmup - 1)),
        "stable_first": float(lr_lambda(warmup)),
        "stable_last": float(lr_lambda(stable_end)),
        "anneal_first": float(lr_lambda(first_decay)),
        "anneal_second": float(lr_lambda(first_decay + 1)),
        "training_last": float(lr_lambda(total - 1)),
    }
    minimum = float(optim["min_lr_ratio"])
    runtime = {
        "warmup_starts_positive": math.isclose(factors["warmup_first"], 1 / warmup,
                                                rel_tol=0, abs_tol=1e-15),
        "warmup_reaches_peak": math.isclose(factors["warmup_last"], 1.0,
                                             rel_tol=0, abs_tol=1e-15),
        "stable_is_peak": factors["stable_first"] == factors["stable_last"] == 1.0,
        # TorchTitan uses a virtual final step: the first anneal step remains at
        # peak and the final training step stays infinitesimally above min LR.
        "anneal_boundary_is_continuous": factors["anneal_first"] == 1.0,
        "anneal_decreases": minimum < factors["anneal_second"] < 1.0,
        "last_step_near_minimum": (
            minimum < factors["training_last"]
            and factors["training_last"] - minimum <= 2.0 / decay
        ),
    }
    checks = {**arithmetic, **runtime}
    return {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "config": str(config_path),
        "checks": checks,
        "total_steps": total,
        "stage_steps": stage_steps,
        "samples_per_step": samples_per_step,
        "factors": factors,
        "torch_version": torch.__version__,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.config.resolve())
    atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
