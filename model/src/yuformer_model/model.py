from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import precompute_rope_freqs
from .config import ModelConfig
from .layers import RMSNorm
from .transformer import TransformerBlock


class YuFormerModel(nn.Module):
    """YuFormer dense transformer decoder, ZGCM-1 style hybrid attention.

    1-indexed layer numbers: layer n uses full causal attention iff
    ``n % attention_every == 0``; the remaining layers use sliding-window
    attention (window = window_size tokens including self) with an optional
    output gate. ``rope_every`` is a YuFormer research switch (NoPE
    ablation); ZGCM-1 uses RoPE on every layer (rope_every=1).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # Packed language modelling consumes EOD as a real, trainable token.
        # Tokenizers may alias PAD to EOD; padding is handled by attention/loss
        # masks and must not zero or freeze that shared embedding row.
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList(
            [TransformerBlock(config, i) for i in range(config.num_layers)]
        )
        self.final_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

        if config.tie_word_embeddings:
            self.lm_head = self.embed_tokens
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        cos, sin = precompute_rope_freqs(
            config.head_dim,
            config.max_seq_len,
            config.rope_base,
            config.rope_percent,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self._init_parameters()

    def _init_parameters(self) -> None:
        std = 0.02
        depth_std = 0.02 / math.sqrt(2 * self.config.num_layers)
        initialized: set[int] = set()
        for name, module in self.named_modules():
            if not isinstance(module, (nn.Linear, nn.Embedding, RMSNorm)):
                continue
            if id(module.weight) in initialized:
                continue
            initialized.add(id(module.weight))
            if isinstance(module, nn.Linear):
                w_std = depth_std if name.endswith(("o_proj", "down_proj")) else std
                nn.init.normal_(module.weight, mean=0.0, std=w_std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
        if self.embed_tokens.padding_idx is not None:
            with torch.no_grad():
                self.embed_tokens.weight[self.embed_tokens.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        ce_chunk: int = 0,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """labels 给定则返回训练 loss（ce_chunk>0 走 E-48 分块 CE）；否则返回 logits。"""
        if labels is not None:
            return self._loss(
                input_ids, labels, ce_chunk, attention_mask, position_ids, ignore_index
            )
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}"
            )
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            raise TypeError("attention_mask must be a bool tensor (True = keep)")
        if position_ids is not None and position_ids.shape != (batch_size, seq_len):
            raise ValueError(
                f"position_ids must have shape {(batch_size, seq_len)}, got {tuple(position_ids.shape)}"
            )

        hidden = self.embed_tokens(input_ids)

        for layer in self.layers:
            hidden = layer(hidden, self.rope_cos, self.rope_sin, attention_mask, position_ids)

        hidden = self.final_norm(hidden)
        if self.config.tie_word_embeddings:
            logits = F.linear(hidden, self.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden)
        return logits.float()

    def _loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        ce_chunk: int,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        ignore_index: int,
    ) -> torch.Tensor:
        """训练 loss 实现（E-48）：ce_chunk>0 分块 CE 不物化全量 fp32 logits；
        ce_chunk=0 回退旧路径用于对拍。必须由 forward 转入（FSDP 根内执行）。"""
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(hidden, self.rope_cos, self.rope_sin, attention_mask, position_ids)
        hidden = self.final_norm(hidden)
        weight = self.embed_tokens.weight if self.config.tie_word_embeddings else self.lm_head.weight
        if ce_chunk <= 0:
            logits = F.linear(hidden, weight)
            return F.cross_entropy(
                logits.float().view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=ignore_index,
            )
        return chunked_cross_entropy(hidden, weight, labels, ce_chunk, ignore_index=ignore_index)

    def num_parameters(self, non_embedding: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if non_embedding:
            total -= self.embed_tokens.weight.numel()
            if self.lm_head is not self.embed_tokens:
                total -= self.lm_head.weight.numel()
        return total


def chunked_cross_entropy(
    hidden: torch.Tensor,
    lm_weight: torch.Tensor,
    labels: torch.Tensor,
    chunk: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """分块融合 CE（E-48）：按词表块计算 lm_head 并即时消费，fp32 在线 logsumexp。

    数值契约（用户裁定验收标准）：
      - logsumexp/目标 logit 累积全程 fp32——bf16 累积在大词表下会溢出出 NaN，
        且禁止"logits 转 bf16 存"的捷径；
      - 每个样本的 label 只落在一个块内，acc_tgt 直接加法合并（无需 rescale）；
        acc_lse 用首块初始化后再 logaddexp，避免 logaddexp(0, ·) 引入 e^0 幻影项；
      - 与 F.cross_entropy(F.linear(hidden, w).float(), labels) 数值等价（GEMM 分块
        的 bf16 舍入差异除外，loss 级容差 1e-4）。
    """
    n = labels.numel()
    flat_h = hidden.reshape(n, hidden.size(-1))
    flat_y = labels.reshape(-1)
    valid = flat_y != ignore_index
    safe_y = torch.where(valid, flat_y, torch.zeros_like(flat_y))
    vocab = lm_weight.size(0)
    acc_lse = None
    acc_tgt = torch.zeros(n, dtype=torch.float32, device=hidden.device)
    for v0 in range(0, vocab, chunk):
        v1 = min(v0 + chunk, vocab)
        logits_c = F.linear(flat_h, lm_weight[v0:v1]).float()  # [n, vc] fp32
        lse_c = torch.logsumexp(logits_c, dim=-1)
        acc_lse = lse_c if acc_lse is None else torch.logaddexp(acc_lse, lse_c)
        # E-51：无条件 gather（与分支版逐位一致：where(in_c) 已清零块外贡献，
        # clamp 后 gather 值被丢弃）——去掉 bool(in_c.any()) 的 CPU↔GPU 同步，
        # eager 每 micro-batch 省 chunk 次 sync，compile 消除 graph break。
        in_c = valid & (safe_y >= v0) & (safe_y < v1)
        idx = (safe_y - v0).clamp_(0, v1 - v0 - 1).unsqueeze(1)
        tgt = logits_c.gather(1, idx).squeeze(1)
        acc_tgt = acc_tgt + torch.where(in_c, tgt, torch.zeros_like(tgt))
        del logits_c
    return (acc_lse - acc_tgt)[valid].mean()


def create_model(config: ModelConfig) -> YuFormerModel:
    return YuFormerModel(config)
