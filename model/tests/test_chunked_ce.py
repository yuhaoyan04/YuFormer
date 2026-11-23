"""E-48 分块融合 CE 验收测试（用户裁定标准）：

1. Loss 对拍：chunked 与 F.cross_entropy(F.linear(...).float()) 差 < 1e-4（bf16 容差）；
2. 数值安全：logsumexp/目标 logit 全程 fp32 累积——bf16 累积在大词表溢出即 NaN；
3. 边界：label 恰在块边界 / chunk=1 / chunk>vocab / tied & untied / 梯度一致性。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from yuformer_model.model import chunked_cross_entropy


def _ref(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = F.linear(hidden, weight).float()
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))


def _case(vocab: int, n: int, chunk: int, dtype: torch.dtype, seed: int, device: str = "cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    h = torch.randn(n, 64, generator=g).to(device=device, dtype=dtype)
    w = torch.randn(vocab, 64, generator=g).mul_(0.05).to(device=device, dtype=dtype)
    y = torch.randint(0, vocab, (n,), generator=g).to(device)
    got = chunked_cross_entropy(h, w, y, chunk)
    ref = _ref(h, w, y)
    return got, ref


def test_fp32_exact_parity():
    got, ref = _case(vocab=65536, n=4096, chunk=8192, dtype=torch.float32, seed=0)
    assert abs(got.item() - ref.item()) < 1e-5, (got.item(), ref.item())


def test_bf16_parity_within_1e4():
    # 验收标准 1：bf16 容差 < 1e-4（GEMM 分块舍入差异经 N 平均后应远小于此）
    if torch.cuda.is_available():
        got, ref = _case(vocab=65536, n=16384, chunk=8192, dtype=torch.bfloat16, seed=1, device="cuda")
        diff = abs(got.item() - ref.item())
        assert diff < 1e-4, f"bf16 loss diff {diff}"


def test_label_at_chunk_boundary():
    g = torch.Generator().manual_seed(2)
    vocab, chunk, n = 100, 16, 64
    h = torch.randn(n, 32, generator=g)
    w = torch.randn(vocab, 32, generator=g).mul_(0.05)
    y = torch.randint(0, vocab, (n,), generator=g)
    y[0], y[1], y[2] = chunk - 1, chunk, vocab - 1  # 块尾 / 下一块首 / 词表尾
    got = chunked_cross_entropy(h, w, y, chunk)
    ref = _ref(h, w, y)
    assert abs(got.item() - ref.item()) < 1e-5


def test_degenerate_chunk_sizes():
    for chunk in (1, 3, 65536):  # chunk=1 / 非整除 / chunk>vocab
        got, ref = _case(vocab=1000, n=512, chunk=chunk, dtype=torch.float32, seed=3)
        assert abs(got.item() - ref.item()) < 1e-5, chunk


def test_fp32_no_accumulation_overflow_large_vocab():
    # 验收标准 2：大词表 + 大相异分数下 fp32 累积不溢出（bf16 累积这里会 NaN）
    g = torch.Generator().manual_seed(4)
    vocab, n = 1 << 17, 2048
    h = torch.randn(n, 64, generator=g).mul_(8.0)   # 放大 logits 量级
    w = torch.randn(vocab, 64, generator=g)
    y = torch.randint(0, vocab, (n,), generator=g)
    got = chunked_cross_entropy(h, w, y, 4096)
    assert torch.isfinite(got), got
    bad = chunked_cross_entropy(h.bfloat16().float(), w, y, 4096)  # 对照：输入已含 bf16 量化
    assert torch.isfinite(bad)


def test_gradient_consistency():
    g = torch.Generator().manual_seed(5)
    vocab, n = 4096, 256
    h0 = torch.randn(n, 32, generator=g, requires_grad=True)
    w0 = torch.randn(vocab, 32, generator=g).mul_(0.05)
    y = torch.randint(0, vocab, (n,), generator=g)
    h1 = h0.detach().clone().requires_grad_(True)
    loss_a = chunked_cross_entropy(h0, w0, y, 512)
    loss_b = _ref(h1, w0, y)
    loss_a.backward()
    loss_b.backward()
    rel = (h0.grad - h1.grad).norm() / h1.grad.norm()
    assert rel < 1e-4, rel


def test_ignore_index_loss_and_gradient_parity():
    g = torch.Generator().manual_seed(6)
    vocab, n = 2048, 257
    h0 = torch.randn(n, 32, generator=g, requires_grad=True)
    w = torch.randn(vocab, 32, generator=g).mul_(0.05)
    y = torch.randint(0, vocab, (n,), generator=g)
    y[::3] = -100
    h1 = h0.detach().clone().requires_grad_(True)

    got = chunked_cross_entropy(h0, w, y, 257, ignore_index=-100)
    ref = F.cross_entropy(F.linear(h1, w).float(), y, ignore_index=-100)
    assert abs(got.item() - ref.item()) < 1e-5
    got.backward()
    ref.backward()
    assert torch.all(h0.grad[::3] == 0)
    rel = (h0.grad - h1.grad).norm() / h1.grad.norm()
    assert rel < 1e-4, rel


def test_ignore_index_all_masked_matches_cross_entropy_nan():
    h = torch.randn(8, 16)
    w = torch.randn(32, 16)
    y = torch.full((8,), -100)
    got = chunked_cross_entropy(h, w, y, 7)
    ref = F.cross_entropy(F.linear(h, w), y, ignore_index=-100)
    assert torch.isnan(got) and torch.isnan(ref)
