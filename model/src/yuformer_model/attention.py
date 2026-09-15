from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    rope_base: float = 500000.0,
    rope_percent: float = 0.334,
) -> torch.Tensor:
    rot_dim = max(1, int(head_dim * rope_percent))
    if rot_dim % 2 != 0:
        rot_dim += 1
    if rot_dim > head_dim:
        rot_dim = head_dim if head_dim % 2 == 0 else head_dim - 1
    num_pairs = rot_dim // 2
    inv_freq = 1.0 / (rope_base ** (torch.arange(0, num_pairs).float() / num_pairs))
    positions = torch.arange(max_seq_len).float()
    freqs = torch.outer(positions, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return torch.stack([cos, sin], dim=-1)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    seq_len = x.size(-2)
    cos = cos[:seq_len].to(x.dtype)
    sin = sin[:seq_len].to(x.dtype)

    rot_dim = cos.size(-1)
    x_rot = x[..., : 2 * rot_dim]
    x_pass = x[..., 2 * rot_dim :]

    x1 = x_rot[..., :rot_dim]
    x2 = x_rot[..., rot_dim:]

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    if x_pass.numel() > 0:
        return torch.cat([rotated, x_pass], dim=-1)
    return rotated


class GQAAttention(nn.Module):
    """Grouped Query Attention with SWA/Full + RoPE/NoPE.

    Args:
        config: ModelConfig
        layer_idx: layer index (determines SWA/Full and RoPE/NoPE)
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.is_full = config.is_full_attn(layer_idx)
        self.use_rope = config.use_rope(layer_idx)
        self.window_size = config.window_size if not self.is_full else 0

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.num_kv_groups = self.num_heads // self.num_kv_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_rope and rope_cos is not None and rope_sin is not None:
            cos = rope_cos[:seq_len]
            sin = rope_sin[:seq_len]
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        if self.window_size > 0 and not self.is_full:
            attn_mask = self._build_swa_mask(seq_len, hidden_states.device, hidden_states.dtype)
            if attention_mask is not None:
                attn_mask = attn_mask * attention_mask
            is_causal = False
        else:
            attn_mask = attention_mask
            is_causal = attn_mask is None

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out)

    def _build_swa_mask(self, seq_len: int, device, dtype) -> torch.Tensor:
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        for i in range(seq_len):
            left = max(0, i - self.window_size)
            mask[i, :left] = float("-inf")
        mask = mask.tril()
        mask[mask == 0] = float("-inf")
        for i in range(seq_len):
            left = max(0, i - self.window_size)
            right = i + 1
            mask[i, left:right] = 0.0
        return mask.unsqueeze(0).unsqueeze(0)
