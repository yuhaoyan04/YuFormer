"""TorchTitan BaseModel wrapper around the research YuFormer decoder."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from torchtitan.protocols.model import BaseModel
from yuformer_model.attention import GQAAttention, precompute_rope_freqs
from yuformer_model.config import ModelConfig
from yuformer_model.model import YuFormerModel


class YuFormerTitanModel(BaseModel):
    _skip_lm_head: bool = False
    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        preset: str
        config_path: str
        sdpa_backend: str = "auto"

        def update_from_config(self, *, config: Any, **kwargs: Any) -> None:
            # The research YAML/model preset is the single source of model shape.
            del config, kwargs

        def get_nparams_and_flops(self, model: BaseModel, seq_len: int) -> tuple[int, int]:
            nparams = sum(p.numel() for p in model.parameters())
            cfg = ModelConfig.from_preset(self.preset, Path(self.config_path))
            embedding_params = cfg.vocab_size * cfg.hidden_size
            # Input embedding lookup is not a dense projection. The output
            # projection is counted once even when its weights are tied.
            dense_params = nparams - embedding_params
            if cfg.tie_word_embeddings:
                dense_params += embedding_params
            context_sum = sum(seq_len if cfg.is_full_attn(index)
                              else min(cfg.window_size, seq_len)
                              for index in range(cfg.num_layers))
            flops_per_token = 6 * dense_params + 12 * cfg.hidden_size * context_sum
            return nparams, int(flops_per_token)

    def __init__(self, config: Config):
        super().__init__()
        self.titan_config = config
        model_config = ModelConfig.from_preset(config.preset, Path(config.config_path))
        if config.sdpa_backend not in {"auto", "cudnn", "cudnn_flex_swa", "flash_flex_swa"}:
            raise ValueError("sdpa_backend must be auto, cudnn, cudnn_flex_swa or flash_flex_swa")
        if config.sdpa_backend in {"cudnn", "cudnn_flex_swa", "flash_flex_swa"} and model_config.attn_impl != "sdpa":
            raise ValueError("explicit SDPA cannot override a non-SDPA model preset")
        self.core = YuFormerModel(model_config)
        for module in self.core.modules():
            if isinstance(module, GQAAttention):
                module.sdpa_backend = config.sdpa_backend
        self.model_config = model_config
        if model_config.tie_word_embeddings:
            # TorchTitan's chunked loss requires an lm_head callable.  Keep a
            # standard Linear module while sharing its weight with embeddings.
            head = nn.Linear(model_config.hidden_size, model_config.vocab_size, bias=False)
            head.weight = self.core.embed_tokens.weight
            self.core.lm_head = head

    def verify_module_protocol(self) -> None:
        # The existing research decoder uses standard nn.Module children.  Its
        # state initialization and FSDP2 policy are owned by this adapter.
        return

    def init_states(self, *, buffer_device: torch.device | None = None) -> None:
        if self.model_config.tie_word_embeddings:
            self.core.lm_head.weight = self.core.embed_tokens.weight
        self.core._init_parameters()
        if self.model_config.tie_word_embeddings:
            self.core.lm_head.weight = self.core.embed_tokens.weight
        device = buffer_device or next(self.parameters()).device
        cfg = self.model_config
        cos, sin = precompute_rope_freqs(
            cfg.head_dim, cfg.max_seq_len, cfg.rope_base, cfg.rope_percent
        )
        self.core.rope_cos = cos.to(device=device)
        self.core.rope_sin = sin.to(device=device)
        if self.titan_config.sdpa_backend in {'cudnn_flex_swa', 'flash_flex_swa'}:
            for module in self.core.modules():
                if isinstance(module, GQAAttention):
                    module.prepare_flex_swa(cfg.max_seq_len, device)

    @property
    def layers(self):
        return self.core.layers

    @property
    def lm_head(self):
        return self.core.lm_head

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        if input_ids.shape[1] > self.model_config.max_seq_len:
            raise ValueError("sequence length exceeds the model RoPE capacity")
        attention_mask = kwargs.get("attention_mask")
        position_ids = kwargs.get("position_ids")
        hidden = self.core.embed_tokens(input_ids)
        for layer in self.core.layers:
            hidden = layer(
                hidden,
                self.core.rope_cos,
                self.core.rope_sin,
                attention_mask,
                position_ids,
            )
        hidden = self.core.final_norm(hidden)
        if self._skip_lm_head:
            return hidden
        return self.core.lm_head(hidden)
