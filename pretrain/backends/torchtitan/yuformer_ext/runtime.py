"""Explicit trainer state and per-rank RNG recovery for the pinned backend."""
from __future__ import annotations

import json
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from torchtitan.trainer import Trainer


def capture_rng() -> dict:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (torch.cuda.get_rng_state() if torch.cuda.is_available() else None),
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"] is not None:
        torch.cuda.set_rng_state(state["torch_cuda"].cpu())


class YuFormerTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        pass

    def state_dict(self) -> dict:
        local = capture_rng()
        world = dist.get_world_size()
        rank_states = [None] * world
        dist.all_gather_object(rank_states, local)
        # All ranks publish the same bytes payload; DCP may deduplicate it
        # without discarding any rank's RNG, including during resharding.
        return {
            "step": self.step,
            "global_tokens_seen": self.ntokens_seen * world,
            "samples_consumed": self.dataloader.state_dict()["consumed_samples"],
            "rng_payload": pickle.dumps({"world_size": world, "states": rank_states}),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        world, rank = dist.get_world_size(), dist.get_rank()
        self.step = int(state_dict["step"])
        tokens = int(state_dict["global_tokens_seen"])
        if tokens % world:
            raise ValueError("checkpoint global token count is not divisible by runtime world size")
        self.ntokens_seen = tokens // world
        self._checkpoint_consumed_samples = int(state_dict["samples_consumed"])
        payload = pickle.loads(state_dict["rng_payload"])
        source_world = int(payload["world_size"])
        if source_world != world and os.environ.get("YUFORMER_ALLOW_TOPOLOGY_RESHARD") != "1":
            raise ValueError("RNG topology changed without an explicit reshard contract")
        restore_rng(payload["states"][rank % source_world])

    def train(self) -> None:
        # Let upstream own model/optimizer/scheduler/DCP mechanics. This hook
        # runs after its checkpointer.load and before the first optimizer step.
        super().train()
        if self.step != self.config.training.steps:
            raise RuntimeError("training ended before the configured optimizer-step budget")
        expected = self.config.training.steps * self.config.training.global_batch_size
        actual = self.dataloader.state_dict()["consumed_samples"]
        if os.environ.get("YUFORMER_ALLOW_TOPOLOGY_RESHARD") != "1" and actual != expected:
            raise RuntimeError(f"final data cursor mismatch: {actual} != {expected}")
        # Upstream saves the final DCP before validation. A restart from that
        # checkpoint skips its loop entirely; rerun final validation explicitly.
        if self._resumed_at_final_step and self.config.validator.enable:
            self.validator.validate(self.model_parts, self.step)
        if dist.get_rank() == 0:
            self._write_receipt("training_accounting.json", {
                "status": "PASS", "step": self.step,
                "consumed_samples": actual,
                "global_tokens_seen": self.ntokens_seen * dist.get_world_size(),
                "tokens_per_update": self.config.training.global_batch_size * self.config.training.seq_len,
                "topology_reshard": os.environ.get("YUFORMER_ALLOW_TOPOLOGY_RESHARD") == "1",
                "attempt_id": os.environ.get("YUFORMER_ATTEMPT_ID"),
            })

    def batch_generator(self, dataloader):
        # The upstream generator is first entered only after DCP load.
        consumed = dataloader.state_dict()["consumed_samples"]
        expected_resume = os.environ.get("RESUME_ITER")
        if expected_resume is not None and int(expected_resume) != self.step:
            raise ValueError(f"RESUME_ITER={expected_resume} differs from restored checkpoint step {self.step}")
        checkpoint_cursor = getattr(self, "_checkpoint_consumed_samples", consumed)
        if consumed != checkpoint_cursor:
            raise RuntimeError("trainer and dataloader checkpoint cursors disagree")
        scheduler_steps = self.lr_schedulers.state_dict()["last_epoch"]
        if scheduler_steps != self.step:
            raise RuntimeError("scheduler and trainer checkpoint steps disagree")
        self._resumed_at_final_step = self.step == self.config.training.steps
        self._write_receipt(f"resume_receipt-rank{dist.get_rank()}.json", {
            "step": self.step, "consumed_samples": consumed,
            "scheduler_step": scheduler_steps,
            "attempt_id": os.environ.get("YUFORMER_ATTEMPT_ID"),
            "world_size": dist.get_world_size(), "rank": dist.get_rank(),
        })
        if dist.get_rank() == 0:
            out = Path(self.config.dump_folder)
            attempt_id = os.environ.get("YUFORMER_ATTEMPT_ID")
            for path in (out / "current_attempt.json", out / "attempts" / f"{attempt_id}.json"):
                if path.is_file():
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if payload.get("attempt_id") != attempt_id:
                        raise RuntimeError("attempt identity changed during checkpoint restore")
                    payload["resumed_from_step"] = self.step
                    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
                    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
                    os.replace(temporary, path)
        return super().batch_generator(dataloader)

    def _write_receipt(self, name: str, payload: dict) -> None:
        path = Path(self.config.dump_folder) / name
        temporary = path.with_name(f".{name}.tmp-{os.getpid()}")
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
