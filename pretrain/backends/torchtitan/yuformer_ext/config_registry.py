"""Translate the repository YAML contract into TorchTitan v0.3 typed config."""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path

import yaml

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import LRSchedulersContainer, default_adamw
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import CompileConfig, DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.trainer import Trainer

from . import model_registry
from .data import PretokenizedDataLoader
from .stream_math import validation_steps as exact_validation_steps
from .runtime import YuFormerTrainer
from .checkpoint import YuFormerCheckpointManager
from .optim import adamw_config
from .metrics import YuFormerMetricsProcessor


EXPECTED_VALIDATION_DOMAINS = {
    "web", "code", "math", "chinese", "academic", "books",
    "encyclopedia", "qa", "multilingual",
}
from .validation import DomainValidator


def _load_contract() -> dict:
    path = os.environ.get("YUFORMER_CONFIG")
    if not path:
        raise ValueError("YUFORMER_CONFIG is required")
    text = os.path.expandvars(Path(path).read_text(encoding="utf-8"))
    if "${" in text:
        raise ValueError("unresolved environment variable in YUFORMER_CONFIG")
    cfg = yaml.safe_load(text)
    if cfg.get("backend") != "torchtitan-v0.3.0":
        raise ValueError("formal TorchTitan launcher requires backend=torchtitan-v0.3.0")
    if cfg.get("experiment_status") in {"superseded", "invalidated", "probe_only"}:
        raise ValueError("non-formal experiment status cannot use the formal launcher")
    return cfg


def yuformer_from_yaml() -> Trainer.Config:
    cfg = _load_contract()
    sdpa_backend = str((cfg.get("attention") or {}).get("sdpa_backend", "auto"))
    if sdpa_backend not in {"auto", "cudnn", "cudnn_flex_swa", "flash_flex_swa"}:
        raise ValueError("attention.sdpa_backend must be auto, cudnn, cudnn_flex_swa or flash_flex_swa")
    hardware = json.loads((Path(__file__).parents[1] / "backend.lock.json").read_text())["hardware"]
    data, train, optim, parallel = cfg["data"], cfg["train"], cfg["optim"], cfg["parallel"]
    gate_path = Path(data["tokenizer_gate"])
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS":
        raise ValueError("tokenizer gate is not PASS")
    tokenizer_json = Path(gate["tokenizer_path"])
    if not tokenizer_json.is_absolute():
        repo_root = os.environ.get("YUFORMER_REPO_ROOT")
        if not repo_root:
            raise ValueError("YUFORMER_REPO_ROOT is required for a relative tokenizer_path")
        tokenizer_json = Path(repo_root) / tokenizer_json
    if not tokenizer_json.is_file():
        raise FileNotFoundError(tokenizer_json)
    tokenizer_hash = hashlib.sha256()
    with tokenizer_json.open("rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            tokenizer_hash.update(block)
    actual_tokenizer_sha256 = tokenizer_hash.hexdigest()
    if actual_tokenizer_sha256 != gate.get("tokenizer_sha256"):
        raise ValueError(
            f"tokenizer artifact hash mismatch: {tokenizer_json}: "
            f"{actual_tokenizer_sha256} != {gate.get('tokenizer_sha256')}"
        )

    world = int(parallel["world_size"])
    actual_world = int(os.environ.get("WORLD_SIZE", world))
    if actual_world != world and os.environ.get("YUFORMER_ALLOW_TOPOLOGY_RESHARD") != "1":
        raise ValueError(
            f"runtime WORLD_SIZE={actual_world} differs from contract world_size={world}; "
            "set YUFORMER_ALLOW_TOPOLOGY_RESHARD=1 only for a recorded reshard canary"
        )
    local_batch = int(train["micro_batch_size"])
    # A recorded reshard canary may restore the same DCP state on another
    # topology. TorchTitan's batch contract must follow the runtime DP world;
    # the YAML world remains the normal-run identity and launch guard.
    global_batch = local_batch * actual_world * int(train["grad_accum"])
    val_dir = Path(data["val_dir"])
    val_streams = sorted(val_dir.glob("val_*.tokens.bin"))
    val_metas = [path.with_suffix("").with_suffix(".meta.json") for path in val_streams]
    val_domains = [path.stem.removeprefix("val_").removesuffix(".tokens") for path in val_streams]
    if not val_streams or any(not path.is_file() for path in val_metas):
        raise ValueError(f"validation stream roster is incomplete: {val_dir}")
    if set(val_domains) != EXPECTED_VALIDATION_DOMAINS or len(val_domains) != len(EXPECTED_VALIDATION_DOMAINS):
        raise ValueError(f"validation stream domain roster is not exact: {val_domains}")
    val_samples_per_domain = int(data["val_samples_per_domain"])
    # One validator step consumes ``local_batch`` samples on every DP rank.
    # Dividing only by world size silently evaluated local_batch times the
    # declared per-domain sample budget (4x in the production contract).
    validation_steps = exact_validation_steps(
        val_samples_per_domain, len(val_streams), actual_world, local_batch
    )
    decay_ratio = None
    if optim.get("lr_schedule") == "wsd":
        decay_ratio = int(optim["decay_steps"]) / int(train["expected_iters"])
    elif optim.get("lr_schedule") != "cosine":
        raise ValueError("formal TorchTitan path supports cosine or WSD schedules")

    return YuFormerTrainer.Config(
        model_spec=model_registry(cfg["model"]["preset"], cfg["model"]["configs"], sdpa_backend=sdpa_backend),
        hf_assets_path=str(tokenizer_json.parent),
        dump_folder=cfg["out_dir"],
        tokenizer=HuggingFaceTokenizer.Config(),
        dataloader=PretokenizedDataLoader.Config(
            stages=list(data["stages"]),
            meta=list(data["meta"]),
            stage_sample_limits=[int(value) for value in data.get("stage_sample_limits", [])],
            sampling_policy=data.get("sampling_policy", "sequential"),
            block_permutation_manifest=data.get("block_permutation_manifest", ""),
            tokenizer_gate=str(gate_path),
            release_manifest=data["release_manifest"],
            epochs=int(data.get("epochs", 1)),
        ),
        loss=ChunkedLossWrapper.Config(
            num_chunks=int(cfg.get("loss", {}).get("num_chunks", 8)),
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=int(gate["vocab_size"])),
        ),
        optimizer=adamw_config(optim),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=int(optim["warmup_steps"]),
            total_steps=int(train["expected_iters"]),
            decay_ratio=decay_ratio,
            decay_type="cosine",
            min_lr_factor=float(optim.get("min_lr_ratio", 0.0)),
        ),
        training=TrainingConfig(
            local_batch_size=local_batch,
            global_batch_size=global_batch,
            seq_len=int(data["seq_len"]),
            steps=int(train["expected_iters"]),
            max_norm=float(optim.get("clip_grad", 1.0)),
            dtype="float32",
            mixed_precision_param="bfloat16" if train.get("bf16", True) else "float32",
            mixed_precision_reduce="float32",
            disable_cuda_graphs=not bool(train.get("cuda_graphs", False)),
        ),
        debug=DebugConfig(
            seed=int(cfg["seed"]),
            deterministic=bool(train.get("deterministic", False)),
            deterministic_warn_only=False,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=int(parallel.get("replicate_degree", 1)),
            data_parallel_shard_degree=int(parallel.get("shard_degree", -1)),
            fsdp_reshard_after_forward=str(parallel.get("fsdp_reshard_after_forward", "default")),
            spmd_backend="partial_dtensor",
        ),
        compile=CompileConfig(
            enable=bool(cfg.get("compile", {}).get("enable", False)),
            components=list(cfg.get("compile", {}).get("components", ["model", "loss"])),
            backend=str(cfg.get("compile", {}).get("backend", "inductor")),
        ),
        checkpoint=YuFormerCheckpointManager.Config(
            enable=int(train.get("save_every", 0)) > 0,
            interval=max(1, int(train.get("save_every", 1))),
            last_save_model_only=False,
            async_mode="async_with_pinned_mem",
            keep_latest_k=int(train.get("keep_last_n", 2)),
            keep_every=int(train.get("keep_every", 0)),
        ),
        metrics=YuFormerMetricsProcessor.Config(
            log_freq=int(train.get("log_every", 20)),
            enable_tensorboard=True,
            bf16_peak_flops=float(hardware["bf16_dense_peak_flops_per_device"]),
            expected_gpu_name=hardware["gpu_name"],
        ),
        validator=DomainValidator.Config(
            enable=True,
            freq=int(train["eval_every"]),
            final_step=int(train["expected_iters"]),
            steps=validation_steps,
            dataloader=PretokenizedDataLoader.Config(
                stages=[str(path) for path in val_streams],
                meta=[str(path) for path in val_metas],
                tokenizer_gate=str(gate_path),
                release_manifest=data["release_manifest"],
                epochs=1,
                infinite=True,
                sampling_policy="domain_round_robin",
                emit_stage_id=True,
            ),
            domain_names=val_domains,
        ),
        activation_checkpoint=(
            FullAC.Config() if train.get("activation_checkpointing", False) else None
        ),
    )
