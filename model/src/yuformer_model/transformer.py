from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from .attention import GQAAttention
from .layers import RMSNorm, SwiGLUMLP


class TransformerBlock(nn.Module):
    """One transformer layer: Pre-norm GQA Attention + Pre-norm SwiGLU MLP.

    Layer index determines attention type (SWA/Full) and position encoding (RoPE/NoPE).
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.self_attn = GQAAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(hidden, rope_cos, rope_sin, attention_mask, position_ids)
        hidden = residual + attn_out

        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        mlp_out = self.mlp(hidden)
        return residual + mlp_out
