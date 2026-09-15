from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import ModelConfig
from .attention import precompute_rope_freqs
from .transformer import TransformerBlock


class YuFormerModel(nn.Module):
    """YuFormer Dense Transformer Decoder.

    Hybrid attention (SWA/Full alternating) + Hybrid position encoding (RoPE/NoPE alternating).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        self.layers = nn.ModuleList(
            [TransformerBlock(config, i) for i in range(config.num_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps) if False else _RMSNormFinal(config)

        if config.tie_word_embeddings:
            self.lm_head = self.embed_tokens
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        head_dim = config.head_dim
        self._rope_data = precompute_rope_freqs(
            head_dim,
            config.max_seq_len,
            config.rope_base,
            config.rope_percent,
        )

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

        hidden = self.embed_tokens(input_ids)

        rope = self._rope_data.to(hidden.device)
        rope_cos = rope[..., 0]
        rope_sin = rope[..., 1]

        for layer in self.layers:
            hidden = layer(hidden, rope_cos, rope_sin, attention_mask, position_ids)

        hidden = self.norm(hidden)
        if self.config.tie_word_embeddings:
            logits = F.linear(hidden, self.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden)
        return logits

    def num_parameters(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.embed_tokens.weight.numel()
        return total


class _RMSNormFinal(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.eps = config.norm_eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps).to(x.dtype)
        return self.weight * x


def create_model(config: ModelConfig) -> YuFormerModel:
    return YuFormerModel(config)
