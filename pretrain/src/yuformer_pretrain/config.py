"""Pretrain 配置（宪法 2：自包含 = 单一真源；宪法 3：外部资产经环境变量）。

YAML 中 `${ENV_NAME}` 占位符在加载时由环境解析，缺失即 fail-fast。
宪法 6：expected_iters 只做身份守卫（实际迭代数由 budget_tokens / tokens_per_step 推导）。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


class ConfigError(ValueError):
    pass


def _resolve_env(value: Any, env_file: str | None = None) -> Any:
    if isinstance(value, str):

        def sub(m: re.Match[str]) -> str:
            name = m.group(1)
            if name not in os.environ:
                raise ConfigError(f"环境变量 {name} 未设置（配置引用 {m.group(0)}，{env_file or 'unknown config'}）")
            return os.environ[name]

        return _ENV_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: _resolve_env(v, env_file) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v, env_file) for v in value]
    return value


@dataclass
class DataConfig:
    stages: list[str]
    meta: list[str]
    val_dir: str | None
    seq_len: int
    epochs: int
    val_samples_per_domain: int = 0
    val_max_tokens_per_domain: int = 2_000_000
    release_manifest: str | None = None
    require_release: bool = False
    require_eod: bool = False


@dataclass
class OptimConfig:
    optimizer: str
    lr: float
    weight_decay: float
    warmup_steps: int
    lr_schedule: str
    min_lr_ratio: float
    adam_betas: tuple[float, float]
    muon_momentum: float
    muon_nesterov: bool
    muon_ns_steps: int
    clip_grad: float
    decay_steps: int = 0


@dataclass
class TrainConfig:
    budget_tokens: int
    micro_batch_size: int
    grad_accum: int
    expected_iters: int
    activation_checkpointing: bool
    bf16: bool
    eval_every: int
    log_every: int
    save_every: int
    seq_len: int
    # E-48：分块 CE 词表块大小（0=关闭走全量 logits 旧路径）。默认 8192——每块 logits
    # 显存 = tokens×8192×4B（mbs4/4K = 0.5GB），解除 fp32 logits 对 mbs 的锁死。
    ce_chunk: int = 8192
    # E-49：选择性重计算（"none"/"mlp"/"attn"）；与 activation_checkpointing（全层）互斥。
    selective_recompute: str = "none"
    # E-51：torch.compile 热路径（FSDP 包装前编译底层 module；默认关，数值低位与同 build
    # 续训一致，跨 build 不保证逐位一致——断点协议以同 build 续训为准）。
    compile: bool = False
    compile_mode: str = "default"
    # E-51：bf16 卫生（默认关，见 model/config.py 的同名开关说明）。
    bf16_hygiene: bool = False
    # E-52：注意力后端 "sdpa"（默认）| "flash"（需 flash_attn 包，见 model 侧说明）。
    attn_impl: str = "sdpa"
    # checkpoint 保留策略：0=禁用自动删除；启用时保留最近 N 个，另可保留每 K iter 里程碑。
    keep_last_n: int = 0
    keep_every: int = 0


@dataclass
class ParallelConfig:
    world_size: int
    # no_shard：DDP 语义（梯度 all-reduce，参数每卡全量）——0.5B 单机显存充裕，
    #   Muon 对完整 2D 矩阵做 NS5，语义精确；full_shard：参数/梯度分片（更大模型），
    #   Muon 在分片视图上仅作近似（见 README 限制）
    sharding: str = "no_shard"


@dataclass
class PretrainConfig:
    run_name: str
    seed: int
    model_preset: str
    model_configs: str
    out_dir: str
    resume: str | None
    data: DataConfig
    optim: OptimConfig
    train: TrainConfig
    parallel: ParallelConfig
    init_weights_from: str | None = None  # 2026-10-15 V6/WSD 退火段：载权重重置优化器（ZGCM --pretrained-checkpoint 语义）
    raw: dict[str, Any] = field(default_factory=dict)
    experiment_status: str = "active"
    backend: str = "pytorch-fsdp1-compat"

    @property
    def tokens_per_step(self) -> int:
        return self.train.seq_len * self.train.micro_batch_size * self.train.grad_accum * self.parallel.world_size

    @property
    def derived_iters(self) -> int:
        return self.train.budget_tokens // self.tokens_per_step

    def validate(self, allow_iter_override: bool = False) -> None:
        if self.backend not in ("pytorch-fsdp1-compat", "torchtitan-v0.3.0"):
            raise ConfigError(f"unsupported training backend: {self.backend}")
        if self.experiment_status != "active":
            raise ConfigError(
                f"配置状态为 {self.experiment_status}，禁止启动；请创建新的 active 配置和 out_dir"
            )
        if self.optim.optimizer not in ("adamw", "muon"):
            raise ConfigError(f"optimizer 必须是 adamw|muon，得到 {self.optim.optimizer}")
        if self.optim.lr_schedule not in ("constant", "cosine", "wsd"):
            raise ConfigError(
                f"lr_schedule 必须是 constant|cosine|wsd，得到 {self.optim.lr_schedule}"
            )
        if len(self.data.stages) != len(self.data.meta):
            raise ConfigError("data.stages 与 data.meta 长度必须一致")
        if self.data.require_release and not self.data.release_manifest:
            raise ConfigError("data.require_release=true 时必须声明 data.release_manifest")
        if self.data.require_release and not self.data.require_eod:
            raise ConfigError("正式 release 训练必须设置 data.require_eod=true")
        if self.train.micro_batch_size * self.train.grad_accum < 1:
            raise ConfigError("micro_batch_size × grad_accum 必须 ≥ 1")
        if self.parallel.sharding not in ("no_shard", "full_shard"):
            raise ConfigError(f"parallel.sharding 必须是 no_shard|full_shard，得到 {self.parallel.sharding}")
        if self.optim.optimizer == "muon" and self.parallel.sharding == "full_shard":
            raise ConfigError(
                "muon（多优化器路由）与 full_shard 不兼容：FSDP.optim_state_dict 不支持多优化器；"
                "0.5B 单机请用 no_shard（DDP 语义，参数每卡全量）"
            )
        if self.train.keep_last_n < 0 or self.train.keep_every < 0:
            raise ConfigError("train.keep_last_n 与 train.keep_every 必须 >= 0")
        if self.optim.lr_schedule == "wsd" and not (
            0 < self.optim.decay_steps < self.derived_iters - self.optim.warmup_steps
        ):
            raise ConfigError(
                "wsd 需要 0 < optim.decay_steps < derived_iters - warmup_steps"
            )
        if not allow_iter_override and self.derived_iters != self.train.expected_iters:
            raise ConfigError(
                f"身份守卫失败：budget_tokens/tokens_per_step = {self.derived_iters} "
                f"!= expected_iters = {self.train.expected_iters}（tokens_per_step={self.tokens_per_step}）"
            )


def load_config(path: str | Path, allow_iter_override: bool = False) -> PretrainConfig:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"配置文件不存在: {p}")
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    raw = _resolve_env(raw, str(p))

    data = raw["data"]
    optim = raw["optim"]
    train = raw["train"]
    parallel = raw.get("parallel", {})
    cfg = PretrainConfig(
        backend=str(raw.get("backend", "pytorch-fsdp1-compat")),
        run_name=str(raw["run_name"]),
        seed=int(raw["seed"]),
        model_preset=str(raw["model"]["preset"]),
        model_configs=str(raw["model"]["configs"]),
        out_dir=str(raw["out_dir"]),
        resume=raw.get("resume"),
        init_weights_from=raw.get("init_weights_from"),
        data=DataConfig(
            stages=[str(s) for s in data["stages"]],
            meta=[str(s) for s in data["meta"]],
            val_dir=data.get("val_dir"),
            seq_len=int(data["seq_len"]),
            epochs=int(data.get("epochs", 1)),
            val_samples_per_domain=int(data.get("val_samples_per_domain", 0)),
            val_max_tokens_per_domain=int(data.get("val_max_tokens_per_domain", 2_000_000)),
            release_manifest=data.get("release_manifest"),
            require_release=bool(data.get("require_release", False)),
            require_eod=bool(data.get("require_eod", False)),
        ),
        optim=OptimConfig(
            optimizer=str(optim.get("optimizer", "adamw")),
            lr=float(optim["lr"]),
            weight_decay=float(optim.get("weight_decay", 0.1)),
            warmup_steps=int(optim.get("warmup_steps", 0)),
            lr_schedule=str(optim.get("lr_schedule", "constant")),
            decay_steps=int(optim.get("decay_steps", 0)),
            min_lr_ratio=float(optim.get("min_lr_ratio", 0.1)),
            adam_betas=tuple(float(x) for x in optim.get("adam_betas", [0.9, 0.95])),
            muon_momentum=float(optim.get("muon_momentum", 0.95)),
            muon_nesterov=bool(optim.get("muon_nesterov", True)),
            muon_ns_steps=int(optim.get("muon_ns_steps", 5)),
            clip_grad=float(optim.get("clip_grad", 1.0)),
        ),
        train=TrainConfig(
            budget_tokens=int(train["budget_tokens"]),
            micro_batch_size=int(train["micro_batch_size"]),
            grad_accum=int(train["grad_accum"]),
            expected_iters=int(train["expected_iters"]),
            activation_checkpointing=bool(train.get("activation_checkpointing", False)),
            bf16=bool(train.get("bf16", True)),
            eval_every=int(train.get("eval_every", 1000)),
            log_every=int(train.get("log_every", 10)),
            save_every=int(train.get("save_every", 1000)),
            seq_len=int(data["seq_len"]),
            ce_chunk=int(train.get("ce_chunk", 8192)),
            selective_recompute=str(train.get("selective_recompute", "none")),
            compile=bool(train.get("compile", False)),
            compile_mode=str(train.get("compile_mode", "default")),
            bf16_hygiene=bool(train.get("bf16_hygiene", False)),
            attn_impl=str(train.get("attn_impl", "sdpa")),
            keep_last_n=int(train.get("keep_last_n", 0)),
            keep_every=int(train.get("keep_every", 0)),
        ),
        parallel=ParallelConfig(
            world_size=int(parallel.get("world_size", 1)),
            sharding=str(parallel.get("sharding", "no_shard")),
        ),
        raw=raw,
        experiment_status=str(raw.get("experiment_status", "active")),
    )
    cfg.validate(allow_iter_override=allow_iter_override)
    return cfg


def config_echo(cfg: PretrainConfig) -> str:
    return json.dumps(cfg.raw, sort_keys=True, ensure_ascii=False)
