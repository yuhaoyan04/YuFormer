from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import RMSNorm

import re as _re

# E-42：版本串可能是 "2.2.0a0"/"2.5.1+cu121"——int("0a0") 直接崩；正则取主版本号前两位。
_ver_m = _re.match(r"(\d+)\.(\d+)", torch.__version__)
_SDPA_SUPPORTS_GQA = (int(_ver_m.group(1)), int(_ver_m.group(2))) >= (2, 5) if _ver_m else False

_MASK_CACHE: dict[tuple[str, int, int, str], torch.Tensor] = {}
_FLEX_MASK_CACHE: dict = {}

def _parse_major_minor(version_str: str) -> Optional[tuple[int, int]]:
    m = _re.match(r"(\d+)\.(\d+)", version_str or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


# E-52：flash-attn 后端（可选，默认 sdpa）。flash_attn_func 直接吃 GQA（Hq 整除 Hkv，
# 免 repeat_interleave 的显存/带宽税），SWA 层走 window_size 免 S×S mask 物化。
# 能力考据（上游 Dao-AILab/flash-attention 各 tag 源码实测）：window_size kwarg 自
# v2.3.0 起存在（2.1/2.2 无）；MQA/GQA 原生（KV 头数整除 Q 头数）自 2.1.0 起声明。
# 本后端全/滑窗层都传 window_size，故整体要求 flash_attn>=2.3.0——2.0.x（如本机
# 2.0.4）装了包也会在首个调用处 TypeError，必须在此 fail-fast。
_FLASH_MIN_VERSION = (2, 3)
try:
    import flash_attn as _fa_pkg
    from flash_attn import flash_attn_func as _flash_attn_func
    _HAS_FLASH_ATTN = True
except Exception:
    _fa_pkg = None
    _flash_attn_func = None
    _HAS_FLASH_ATTN = False

_FLASH_ATTN_VERSION = _parse_major_minor(getattr(_fa_pkg, "__version__", "")) if _fa_pkg else None
_FLASH_SUPPORTS_WINDOW = _FLASH_ATTN_VERSION is not None and _FLASH_ATTN_VERSION >= _FLASH_MIN_VERSION
# 后端真可用 = 包在 + 版本满足能力（装包即激活的 guard 会把 2.0.x 误放进来）。
_FLASH_ATTN_USABLE = _HAS_FLASH_ATTN and _FLASH_SUPPORTS_WINDOW

if _HAS_FLASH_ATTN:
    @torch._dynamo.disable
    def _fa_call(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window: tuple) -> torch.Tensor:
        # torch 2.2 dynamo 无法穿透 flash_attn 的 autograd.Function（tuple 入参 +
        # higher-order op 报错，陷入重编译）；显式 graph break，由 compile 融合其余部分。
        return _flash_attn_func(q, k, v, dropout_p=0.0, causal=True, window_size=window)


def flash_backend_error() -> Optional[str]:
    """flash 后端不可用的原因字符串；可用返回 None。"""
    if not _HAS_FLASH_ATTN:
        return (
            'attn_impl=flash 但 flash-attn 未安装；请改用 attn_impl="sdpa"'
            f" 或安装 flash-attn>={_FLASH_MIN_VERSION[0]}.{_FLASH_MIN_VERSION[1]}"
        )
    if not _FLASH_SUPPORTS_WINDOW:
        ver = getattr(_fa_pkg, "__version__", "?")
        return (
            f"attn_impl=flash 但 flash-attn {ver} 不支持 window_size"
            f"（需要 >={_FLASH_MIN_VERSION[0]}.{_FLASH_MIN_VERSION[1]}）；"
            '请改用 attn_impl="sdpa" 或升级 flash-attn'
        )
    return None


def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    rope_base: float = 5_000_000.0,
    rope_percent: float = 0.334,
) -> tuple[torch.Tensor, torch.Tensor]:
    rot_dim = int(head_dim * rope_percent)
    if rot_dim % 2 != 0:
        rot_dim -= 1
    rot_dim = max(min(rot_dim, head_dim), 2)
    inv_freq = rope_base ** (-torch.arange(0, rot_dim, 2, dtype=torch.float32) / rot_dim)
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    pairs = cos.size(-1)
    if position_ids is None:
        cos = cos[: x.size(-2)].unsqueeze(0).unsqueeze(0)
        sin = sin[: x.size(-2)].unsqueeze(0).unsqueeze(0)
    else:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    cos = cos.to(x.dtype)
    sin = sin.to(x.dtype)

    x_rot = x[..., : 2 * pairs]
    x_pass = x[..., 2 * pairs :]
    x1 = x_rot[..., :pairs]
    x2 = x_rot[..., pairs:]

    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    if x_pass.size(-1) > 0:
        return torch.cat([rotated, x_pass], dim=-1)
    return rotated


def _cached_bool_mask(kind: str, seq_len: int, device: torch.device, window_size: int = 0) -> torch.Tensor:
    key = (kind, seq_len, window_size, str(device))
    mask = _MASK_CACHE.get(key)
    if mask is None:
        pos = torch.arange(seq_len, device=device)
        if kind == "causal":
            mask = pos[:, None] >= pos[None, :]
        else:
            diff = pos[:, None] - pos[None, :]
            mask = (diff >= 0) & (diff < window_size)
        _MASK_CACHE[key] = mask
    return mask


class GQAAttention(nn.Module):
    """GQA with fused QKV projection, optional output gate (SWA layers only), QK-RMSNorm and RoPE.

    Layer type (full causal vs sliding window) and gating are decided by
    ``config.is_full_attn(layer_idx)`` / ``config.use_gate(layer_idx)`` with
    1-indexed layer numbers. Fused projection layout is [q | g | k | v]; the
    gate slice is present only on gated SWA layers and is applied after head
    concatenation, before o_proj: ``out = out * sigmoid(gate.float())``.
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
        self.has_gate = config.use_gate(layer_idx)
        self.window_size = config.window_size
        self.sdpa_backend = "auto"

        self.q_width = self.num_heads * self.head_dim
        self.kv_width = self.num_kv_heads * self.head_dim
        gate_width = self.q_width if self.has_gate else 0
        self.qkv_proj = nn.Linear(
            self.hidden_size, self.q_width + gate_width + 2 * self.kv_width, bias=False
        )
        self.o_proj = nn.Linear(self.q_width, self.hidden_size, bias=False)

        if config.qk_norm:
            qk_fp32 = not config.bf16_hygiene
            self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps, fp32_variance=qk_fp32)
            self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps, fp32_variance=qk_fp32)

    @staticmethod
    def _key_keep(attention_mask: torch.Tensor) -> torch.Tensor:
        if attention_mask.dim() == 2:
            return attention_mask[:, None, None, :]
        if attention_mask.dim() == 4:
            return attention_mask
        raise ValueError(f"attention_mask must be [batch, seq] or [batch, 1, seq, seq], got {attention_mask.dim()}D")

    def prepare_flex_swa(self, seq_len: int, device: torch.device) -> None:
        from torch.nn.attention.flex_attention import create_block_mask
        if self.is_full:
            return
        key = (seq_len, self.window_size, str(device))
        if key not in _FLEX_MASK_CACHE:
            window = self.window_size

            def mask_mod(batch, head, query, kv):
                return (query >= kv) & (query-kv < window)

            _FLEX_MASK_CACHE[key] = create_block_mask(
                mask_mod, None, None, seq_len, seq_len, device=str(device), BLOCK_SIZE=128)
        self._flex_block_mask = _FLEX_MASK_CACHE[key]
        self._flex_seq_len = seq_len

    def _attn_mask(
        self,
        seq_len: int,
        attention_mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> tuple[Optional[torch.Tensor], bool]:
        if self.is_full:
            if attention_mask is None:
                return None, True
            causal = _cached_bool_mask("causal", seq_len, device)
            mask = causal[None, None] & self._key_keep(attention_mask)
        else:
            window = _cached_bool_mask("swa", seq_len, device, self.window_size)
            if attention_mask is None:
                return window[None, None], False
            mask = window[None, None] & self._key_keep(attention_mask)
        # E-42：全 False 的 query 行（padding 深区 query 的 window 全落在被 pad 的 key
        # 上）在 torch<2.4 的 SDPA 下 softmax(-inf) 出 NaN 并污染整行；2.5+ 返回零。
        # 给空行补对角线自键 → 数值有界、语义仅影响被丢弃的 padding 位，跨版本行为一致。
        empty = ~mask.any(-1, keepdim=True)
        if bool(empty.any()):
            eye = torch.eye(seq_len, dtype=torch.bool, device=device)[None, None]
            mask = mask | (empty & eye)
        return mask, False

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)
        if self.has_gate:
            q, g, k, v = torch.split(qkv, [self.q_width, self.q_width, self.kv_width, self.kv_width], dim=-1)
        else:
            q, k, v = torch.split(qkv, [self.q_width, self.kv_width, self.kv_width], dim=-1)
            g = None

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.config.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rope and rope_cos is not None and rope_sin is not None:
            q = apply_rope(q, rope_cos, rope_sin, position_ids)
            k = apply_rope(k, rope_cos, rope_sin, position_ids)

        if self.config.attn_impl == "flash":
            # E-52：flash-attn 直连（[B,S,H,D]，GQA 原生、SWA window 原生，
            # 免 repeat_interleave 与 bool mask）。仅支持无 padding 的定长样本。
            if attention_mask is not None:
                raise ValueError("attn_impl=flash 不支持 attention_mask（pretrain 定长恒为 None）")
            err = flash_backend_error()
            if err is not None:
                raise RuntimeError(err)
            if q.dtype not in (torch.float16, torch.bfloat16):
                # qk_norm 的 fp32 权重会把 q/k 抬成 fp32（SDPA 在 autocast 表内会被
                # 自动转 bf16，flash_attn_func 是裸算子不吃这套）——对齐 SDPA 语义
                # 显式下转（训练恒在 bf16 autocast 下，数值口径一致）。
                dt = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() \
                    else torch.bfloat16
                q, k, v = q.to(dt), k.to(dt), v.to(dt)
            window = (-1, -1) if self.is_full else (self.window_size - 1, 0)
            out = _fa_call(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), window)
            out = out.reshape(batch_size, seq_len, self.q_width)
        elif self.sdpa_backend == 'cudnn_flex_swa' and not self.is_full:
            from torch.nn.attention.flex_attention import flex_attention
            if attention_mask is not None:
                raise ValueError('cudnn_flex_swa requires unpadded fixed-length pretraining samples')
            if not hasattr(self, '_flex_block_mask') or self._flex_seq_len != seq_len:
                raise ValueError('prepare_flex_swa must bind the exact sequence length and device before compilation')
            if torch.is_autocast_enabled():
                dtype = torch.get_autocast_dtype('cuda')
                q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
            out = flex_attention(q, k, v, block_mask=self._flex_block_mask,
                                 enable_gqa=self.num_heads != self.num_kv_heads,
                                 kernel_options={'BACKEND': 'TRITON'})
            out = out.transpose(1, 2).reshape(batch_size, seq_len, self.q_width)
        else:
            attn_mask, is_causal = self._attn_mask(seq_len, attention_mask, hidden_states.device)

            sdpa_kwargs: dict = {}
            if self.num_kv_heads != self.num_heads:
                if _SDPA_SUPPORTS_GQA:
                    sdpa_kwargs["enable_gqa"] = True
                else:
                    k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
                    v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)

            if self.sdpa_backend in {"cudnn", "cudnn_flex_swa"}:
                from torch.nn.attention import SDPBackend, sdpa_kernel
                with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                    out = F.scaled_dot_product_attention(
                        q.contiguous(), k.contiguous(), v.contiguous(),
                        attn_mask=attn_mask, dropout_p=0.0,
                        is_causal=is_causal, **sdpa_kwargs,
                    )
            else:
                out = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=is_causal,
                    **sdpa_kwargs,
                )
            out = out.transpose(1, 2).reshape(batch_size, seq_len, self.q_width)
        if g is not None:
            gate = torch.sigmoid(g.float()) if not self.config.bf16_hygiene else torch.sigmoid(g)
            out = out * gate.to(out.dtype)
        return self.o_proj(out)
