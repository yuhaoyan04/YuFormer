from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from .attention import GQAAttention
from .layers import RMSNorm, SwiGLUMLP


class TransformerBlock(nn.Module):
    """One transformer layer: Pre-norm GQA Attention + Pre-norm SwiGLU MLP.

    Layer index determines attention type (full causal vs gated SWA) and
    position encoding (RoPE/NoPE) via the 1-indexed rules in ModelConfig.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        norm_fp32 = not config.bf16_hygiene
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.norm_eps, fp32_variance=norm_fp32)
        self.self_attn = GQAAttention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.norm_eps, fp32_variance=norm_fp32)
        self.mlp = SwiGLUMLP(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mode = self.config.selective_recompute
        from torch.utils.checkpoint import checkpoint
        use_ck = hidden_states.requires_grad
        residual = hidden_states
        hidden = self.input_layernorm(hidden_states)
        if mode == "attn" and use_ck:
            attn_out = checkpoint(self.self_attn, hidden, rope_cos, rope_sin,
                                  attention_mask, position_ids, use_reentrant=False)
        else:
            attn_out = self.self_attn(hidden, rope_cos, rope_sin, attention_mask, position_ids)
        hidden = residual + attn_out

        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        if mode == "mlp" and use_ck:
            mlp_out = checkpoint(self.mlp, hidden, use_reentrant=False)
        else:
            mlp_out = self.mlp(hidden)
        return residual + mlp_out
