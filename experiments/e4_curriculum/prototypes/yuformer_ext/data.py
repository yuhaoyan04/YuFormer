"""Deterministic, DCP-stateful reader for immutable pretokenized streams."""
from __future__ import annotations

import bisect
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

from torchtitan.components.dataloader import BaseDataLoader, DataloaderExhaustedError

from .block_permutation import load_permutations, map_sample

from .stream_math import (
    domain_round_robin_location,
    resume_step,
    round_robin_location,
    sample_count,
    sample_offset,
)


def _edge_sha256(path: Path, edge_bytes: int = 1 << 20) -> str:
    with path.open("rb") as fh:
        return _open_file_edge_sha256(fh, edge_bytes)


def _open_file_edge_sha256(fh, edge_bytes: int = 1 << 20) -> str:
    size = os.fstat(fh.fileno()).st_size
    fh.seek(0)
    head = fh.read(edge_bytes)
    fh.seek(max(0, size - edge_bytes))
    tail = fh.read(edge_bytes)
    h = hashlib.sha256()
    h.update(str(size).encode("ascii"))
    h.update(head)
    if size > edge_bytes:
        h.update(tail)
    return h.hexdigest()


def _file_identity(stat) -> dict[str, int]:
    return {"bytes": stat.st_size, "device": stat.st_dev,
            "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns}


class PretokenizedDataLoader(BaseDataLoader):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseDataLoader.Config):
        stages: list[str] = field(default_factory=list)
        meta: list[str] = field(default_factory=list)
        stage_sample_limits: list[int] = field(default_factory=list)
        tokenizer_gate: str = ""
        release_manifest: str = ""
        epochs: int = 1
        infinite: bool = True
        sampling_policy: str = "sequential"
        emit_stage_id: bool = False
        block_permutation_manifest: str = ""

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: Any,
        seq_len: int,
        local_batch_size: int,
        snapshot_every_n_steps: int | None = None,
    ):
        del tokenizer, snapshot_every_n_steps
        if len(config.stages) != len(config.meta) or not config.stages:
            raise ValueError("stages/meta must be non-empty and one-to-one")
        if config.stage_sample_limits and len(config.stage_sample_limits) != len(config.stages):
            raise ValueError("stage_sample_limits must be empty or one-to-one with stages")
        self.config = config
        self.dp_world_size = int(dp_world_size)
        self.dp_rank = int(dp_rank)
        self.seq_len = int(seq_len)
        self.local_batch_size = int(local_batch_size)
        self._step = 0

        gate = json.loads(Path(config.tokenizer_gate).read_text(encoding="utf-8"))
        if gate.get("status") != "PASS":
            raise ValueError("tokenizer gate is not PASS")
        release_raw = Path(config.release_manifest).read_bytes()
        release = json.loads(release_raw)
        stream_meta_sha256 = []
        self._block_permutation = None
        if release.get("status") != "PASS" or not release.get("snapshot_id"):
            raise ValueError("data release is not PASS")
        if gate.get("dataset_id") != release.get("dataset_id"):
            raise ValueError("tokenizer gate and data release dataset_id differ")
        if gate.get("snapshot_id") != release.get("snapshot_id"):
            raise ValueError("tokenizer gate and data release snapshot_id differ")
        self._arrays: list[np.memmap] = []
        self._ends: list[int] = []
        self._stage_samples: list[int] = []
        self._stage_capacities: list[int] = []
        total = 0
        for stage_index, (stream_name, meta_name) in enumerate(
                zip(config.stages, config.meta, strict=True)):
            stream, meta_path = Path(stream_name), Path(meta_name)
            meta_raw = meta_path.read_bytes()
            meta = json.loads(meta_raw)
            stream_meta_sha256.append(hashlib.sha256(meta_raw).hexdigest())
            if meta.get("tokenizer_sha256") != gate.get("tokenizer_sha256"):
                raise ValueError(f"tokenizer identity mismatch: {meta_path}")
            if int(meta.get("vocab_size", -1)) != int(gate.get("vocab_size", -2)):
                raise ValueError(f"tokenizer vocabulary mismatch: {meta_path}")
            if meta.get("data_snapshot_id") != release.get("snapshot_id"):
                raise ValueError(f"data snapshot identity mismatch: {meta_path}")
            boundary = meta.get("document_boundary") or {}
            if boundary.get("policy") != "append_eod" or int(boundary.get("count", -1)) != int(meta.get("n_docs", -2)):
                raise ValueError(f"invalid document boundary contract: {meta_path}")
            dtype = np.dtype(meta["dtype"])
            with stream.open("rb") as stream_handle:
                identity = _file_identity(os.fstat(stream_handle.fileno()))
                published = meta.get("stream_stat_identity")
                if published is not None and published != identity:
                    raise ValueError(f"stream published identity mismatch: {stream}")
                if (meta.get("edge_sha256")
                        and _open_file_edge_sha256(stream_handle) != meta["edge_sha256"]):
                    raise ValueError(f"stream edge hash mismatch: {stream}")
                array = np.memmap(stream_handle, dtype=dtype, mode="r")
                if (_file_identity(os.fstat(stream_handle.fileno())) != identity
                        or _file_identity(stream.stat()) != identity):
                    raise ValueError(f"stream changed while creating mapping: {stream}")
            if int(meta["total_tokens"]) != int(array.size):
                raise ValueError(f"stream length mismatch: {stream}")
            # Consecutive causal windows share their boundary token: it is the
            # previous window's final target and the next window's first input.
            # This covers every stream token once as a target (except token 0)
            # instead of dropping one token per sample.
            samples = sample_count(int(array.size), self.seq_len)
            self._stage_capacities.append(samples)
            if config.stage_sample_limits:
                requested = int(config.stage_sample_limits[stage_index])
                if requested < 1 or requested > samples:
                    raise ValueError(
                        f"invalid stage sample limit {requested} for capacity {samples}: {stream}"
                    )
                samples = requested
            if samples < 1:
                raise ValueError(f"stream has no complete causal sample: {stream}")
            total += samples
            self._ends.append(total)
            self._stage_samples.append(samples)
            self._arrays.append(array)
        if config.sampling_policy in {"sequential", "block_permutation"}:
            self._samples_per_epoch = total
        elif config.sampling_policy in {"round_robin", "domain_round_robin"}:
            self._samples_per_epoch = min(self._stage_samples) * len(self._stage_samples)
        else:
            raise ValueError(f"unsupported sampling policy: {config.sampling_policy}")
        self._capacity = total * int(config.epochs)
        if config.sampling_policy in {"round_robin", "domain_round_robin"}:
            self._capacity = self._samples_per_epoch * int(config.epochs)
        if config.sampling_policy == "block_permutation":
            if not config.stage_sample_limits or not config.block_permutation_manifest:
                raise ValueError("block permutation requires explicit stage_sample_limits and manifest")
            self._block_permutation = load_permutations(
                config.block_permutation_manifest, seq_len=self.seq_len,
                stage_samples=self._stage_samples, stage_capacities=self._stage_capacities,
                stream_meta_sha256=stream_meta_sha256,
                release_sha256=hashlib.sha256(release_raw).hexdigest(),
                dataset_id=release["dataset_id"], snapshot_id=release["snapshot_id"],
            )
        elif config.block_permutation_manifest:
            raise ValueError("block permutation manifest requires its explicit sampling policy")

    def _sample(self, sample_idx: int) -> tuple[np.ndarray, np.ndarray, int]:
        if sample_idx >= self._capacity:
            raise DataloaderExhaustedError("pretokenized stream capacity exhausted")
        idx = sample_idx % self._samples_per_epoch
        if self.config.sampling_policy == "round_robin":
            stage, local = round_robin_location(idx, self._stage_samples)
        elif self.config.sampling_policy == "domain_round_robin":
            global_batch = self.dp_world_size * self.local_batch_size
            try:
                stage, local = domain_round_robin_location(
                    idx, self._stage_samples, global_batch
                )
            except IndexError as exc:
                raise DataloaderExhaustedError(str(exc)) from exc
        else:
            stage = bisect.bisect_right(self._ends, idx)
            base = 0 if stage == 0 else self._ends[stage - 1]
            local = idx - base
        if self._block_permutation is not None:
            local = map_sample(local, self._block_permutation["arrays"][stage],
                               self._block_permutation["block_samples"])
        offset = sample_offset(local, self.seq_len)
        window = np.asarray(self._arrays[stage][offset : offset + self.seq_len + 1], dtype=np.int64)
        return window[:-1], window[1:], stage

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        while True:
            global_start = (
                self._step * self.dp_world_size * self.local_batch_size
                + self.dp_rank * self.local_batch_size
            )
            inputs, labels, stage_ids = [], [], []
            for item in range(self.local_batch_size):
                x, y, stage = self._sample(global_start + item)
                inputs.append(x)
                labels.append(y)
                stage_ids.append(stage)
            self._step += 1
            input_dict = {"input": torch.from_numpy(np.stack(inputs))}
            if self.config.emit_stage_id:
                input_dict["_yuformer_stage_id"] = torch.tensor(stage_ids, dtype=torch.int64)
            yield input_dict, torch.from_numpy(np.stack(labels))

    def state_dict(self) -> dict[str, Any]:
        state = {
            "consumed_samples": self._step * self.dp_world_size * self.local_batch_size,
            "seq_len": self.seq_len,
        }
        if self.config.sampling_policy != "sequential":
            state["sampling_policy"] = self.config.sampling_policy
        if self._block_permutation is not None:
            state["block_permutation_manifest_sha256"] = self._block_permutation["manifest_sha256"]
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if not state_dict:
            return
        if state_dict.get("sampling_policy", "sequential") != self.config.sampling_policy:
            raise ValueError("cannot resume a dataloader with a different sampling policy")
        expected_manifest = (self._block_permutation["manifest_sha256"]
                             if self._block_permutation is not None else None)
        if state_dict.get("block_permutation_manifest_sha256") != expected_manifest:
            raise ValueError("cannot resume a dataloader with a different block permutation manifest")
        if int(state_dict["seq_len"]) != self.seq_len:
            raise ValueError("cannot resume a dataloader with a different sequence length")
        consumed = int(state_dict["consumed_samples"])
        if consumed < 0 or consumed > self._capacity:
            raise ValueError("checkpoint data cursor exceeds the configured capacity")
        self._step = resume_step(consumed, self.dp_world_size, self.local_batch_size)
        audit_root = os.environ.get("YUFORMER_CURSOR_AUDIT_DIR")
        if audit_root:
            root = Path(audit_root)
            root.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema_version": 1,
                "consumed_samples": consumed,
                "dp_world_size": self.dp_world_size,
                "dp_rank": self.dp_rank,
                "local_batch_size": self.local_batch_size,
                "resumed_dataloader_step": self._step,
                "seq_len": self.seq_len,
            }
            target = root / (
                f"load-{consumed}-world{self.dp_world_size}-rank{self.dp_rank}.json"
            )
            temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
            temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(temporary, target)
