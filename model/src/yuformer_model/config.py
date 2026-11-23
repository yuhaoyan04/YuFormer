from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    hidden_size: int = 512
    intermediate_size: int = 1024
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int = 4
    max_seq_len: int = 4096
    window_size: int = 128
    attention_every: int = 6
    rope_every: int = 1
    rope_base: float = 5_000_000.0
    rope_percent: float = 0.334
    qk_norm: bool = False
    attention_output_gate: bool = False
    # E-49：选择性重计算（冲刺项 1）——"none" 关 / "mlp" 只重算 MLP 块（省 ~56% 激活
    # 显存、重算代价 ~15-17% 层 FLOPs，对照 full ckpt 的 ~30% 全层税）/ "attn" 只重算注意力。
    # 与 trainer 的 activation_checkpointing（全层）互斥。
    selective_recompute: str = "none"
    # E-51：bf16 卫生（默认关）——RMSNorm 方差/gate sigmoid/RoPE cos-sin 热路径不再
    # 上溯 fp32（省显存带宽，数值低位变化；开前需对 loss 做 A/B）。
    bf16_hygiene: bool = False
    # E-52：注意力后端 "sdpa"（默认）| "flash"（flash-attn 包，GQA/SWA 原生免 mask）。
    attn_impl: str = "sdpa"

    norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    pad_token_id: int = 0
    bos_token_id: int | None = None
    eos_token_id: int = 0
    notes: str = ""

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def rotary_dim(self) -> int:
        rot_dim = int(self.head_dim * self.rope_percent)
        if rot_dim % 2 != 0:
            rot_dim -= 1
        return max(rot_dim, 2)

    @property
    def num_kv_groups(self) -> int:
        return self.num_heads // self.num_kv_heads

    @property
    def num_full_attn_layers(self) -> int:
        return sum(1 for i in range(self.num_layers) if self.is_full_attn(i))

    @property
    def num_swa_layers(self) -> int:
        return self.num_layers - self.num_full_attn_layers

    def layer_number(self, layer_idx: int) -> int:
        return layer_idx + 1

    def is_full_attn(self, layer_idx: int) -> bool:
        return self.layer_number(layer_idx) % self.attention_every == 0

    def is_swa(self, layer_idx: int) -> bool:
        return not self.is_full_attn(layer_idx)

    def use_rope(self, layer_idx: int) -> bool:
        return self.layer_number(layer_idx) % self.rope_every == 0

    def use_gate(self, layer_idx: int) -> bool:
        return self.is_swa(layer_idx) and self.attention_output_gate

    def __post_init__(self) -> None:
        for name in ("vocab_size", "hidden_size", "intermediate_size", "num_layers",
                     "num_heads", "num_kv_heads", "max_seq_len", "window_size",
                     "attention_every", "rope_every"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("qk_norm", "attention_output_gate", "tie_word_embeddings", "bf16_hygiene"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.selective_recompute not in ("none", "mlp", "attn"):
            raise ValueError("selective_recompute must be none|mlp|attn")
        for name in ("rope_base", "norm_eps", "rope_percent"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.vocab_size % 64 != 0:
            raise ValueError(f"vocab_size must be a multiple of 64, got {self.vocab_size}")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE rotate-half, got {self.head_dim}")
        if self.attention_every < 1:
            raise ValueError("attention_every must be >= 1")
        if self.rope_every < 1:
            raise ValueError("rope_every must be >= 1")
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if not 0.0 < self.rope_percent <= 1.0:
            raise ValueError("rope_percent must be in (0, 1]")
        if self.attn_impl not in ("sdpa", "flash"):
            raise ValueError(f"attn_impl must be sdpa|flash, got {self.attn_impl}")
        if self.attn_impl == "flash":
            # E-52：init 期 fail-fast——flash 后端要求 flash_attn>=2.3.0（window_size），
            # 不满足时给出可行动的报错，而不是首个 forward 才 TypeError。
            from .attention import flash_backend_error
            err = flash_backend_error()
            if err is not None:
                raise ValueError(err)

    @classmethod
    def from_preset(cls, name: str, config_path: str | Path | None = None) -> "ModelConfig":
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "model_configs.yaml"
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        presets = data.get("presets", {})
        if name not in presets:
            available = list(presets.keys())
            raise ValueError(f"preset '{name}' not found. Available: {available}")
        return cls(**presets[name])

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)
