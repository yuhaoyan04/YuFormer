from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 512
    intermediate_size: int = 1024
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int = 4
    max_seq_len: int = 4096
    window_size: int = 128
    attention_every: int = 4
    rope_every: int = 2
    rope_base: float = 500000.0
    rope_percent: float = 0.334
    norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    notes: str = ""

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def num_full_attn_layers(self) -> int:
        return sum(1 for i in range(self.num_layers) if self.is_full_attn(i))

    @property
    def num_swa_layers(self) -> int:
        return self.num_layers - self.num_full_attn_layers

    def is_full_attn(self, layer_idx: int) -> bool:
        if self.attention_every <= 1:
            return True
        return (layer_idx % self.attention_every) == 0

    def is_swa(self, layer_idx: int) -> bool:
        return not self.is_full_attn(layer_idx)

    def use_rope(self, layer_idx: int) -> bool:
        if self.rope_every <= 1:
            return True
        return (layer_idx % self.rope_every) == 0

    def __post_init__(self) -> None:
        assert self.hidden_size % self.num_heads == 0, "hidden_size must be divisible by num_heads"
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        assert self.rope_percent > 0 and self.rope_percent <= 1.0, "rope_percent must be in (0, 1]"

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
