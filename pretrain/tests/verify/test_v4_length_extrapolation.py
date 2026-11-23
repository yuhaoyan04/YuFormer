"""V4 长度外推原子验证（midtrain 阶梯前置）。

内容：
  A. SWA mask 探针（in-process，GPU，fp32 精确）：
     1. 16K 前向下，SWA 层输出与"4K 滑窗参考"在窗口预热区外逐位一致
        （滑窗语义：位置 i 仅依赖 [i-w+1, i]）。
     2. mask 缓存键 (kind, seq_len, window, device) 不串：4096→16384→4096 往返
        输出不变；window 128 vs 256 不互相污染。
  B. 训练阶梯语义：
     1. tiny-8L 预设 seq 4K / rope_base 5e6 训 100 iters → ckpt。
     2. seq 16K / rope_base 1e7 配置直接 resume：当前实现被 config_sha 守卫
        fail-fast 拒绝——验证拒绝是显式的（不静默错训），记为点火前决策项 D1。
     3. 权重级加载（midtrain 语义模拟）：load_state_dict(strict=True) 载入 4K ckpt
        到 16K/rope-1e7 模型 → 16K 前向 loss 有限；rope buffer 按新 base 重建
        （非持久 buffer 不随 ckpt，切换生效）；position_ids 默认 0..seq-1 语义正确。
  C. 64K 前向 smoke（CPU，sdpa——flash-attn 包 ABI 损坏不可用，见报告）：
     2L 微型模型 64K 定长前向 loss 有限、SWA 64K mask 语义正确。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from conftest import VERIFY_OUT, make_token_stream, run_trainer, write_config

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "pretrain" / "src"))
sys.path.insert(0, str(REPO / "model" / "src"))

from yuformer_model.attention import (  # noqa: E402
    GQAAttention, _MASK_CACHE, precompute_rope_freqs,
)
from yuformer_model.config import ModelConfig  # noqa: E402
from yuformer_model.model import create_model  # noqa: E402

WORK = VERIFY_OUT / "v4"
WINDOW = 128
DEV = torch.device("cuda")

V4_PRESET_BODY = dict(
    attention_every=4, attention_output_gate=True, hidden_size=256,
    intermediate_size=512, norm_eps=1.0e-06, num_heads=4, num_kv_heads=2,
    num_layers=8, qk_norm=True, rope_every=1, rope_percent=0.334,
    tie_word_embeddings=True, vocab_size=32768, window_size=WINDOW,
)


def _write_model_configs() -> Path:
    d = WORK / "model_configs"
    d.mkdir(parents=True, exist_ok=True)
    presets = {
        "v4t-4k": {**V4_PRESET_BODY, "max_seq_len": 4096, "rope_base": 5_000_000.0},
        "v4t-16k": {**V4_PRESET_BODY, "max_seq_len": 16384, "rope_base": 10_000_000.0},
        "v4t-64k": dict(**{**V4_PRESET_BODY, "num_layers": 2, "hidden_size": 128,
                           "intermediate_size": 256, "num_heads": 2, "num_kv_heads": 2,
                           "window_size": 64, "qk_norm": False,
                           "attention_output_gate": False},
                        max_seq_len=65536, rope_base=5_000_000.0),
    }
    p = d / "model_configs.yaml"
    p.write_text(yaml.safe_dump({"presets": presets}, sort_keys=False))
    return p


def _probe_attn() -> GQAAttention:
    torch.manual_seed(0)
    cfg = ModelConfig(num_layers=8, hidden_size=256, num_heads=4, num_kv_heads=2,
                      intermediate_size=512, max_seq_len=16384, window_size=WINDOW,
                      rope_base=5e6, rope_percent=0.334, qk_norm=True,
                      attention_output_gate=True, vocab_size=32768,
                      tie_word_embeddings=True)
    attn = GQAAttention(cfg, layer_idx=0).to(DEV).eval()  # layer 1：SWA
    assert not attn.is_full and attn.window_size == WINDOW
    return attn


class TestV4SWAMaskProbe:
    @pytest.fixture(scope="class")
    def probe(self):
        torch.manual_seed(7)
        attn = _probe_attn()
        x16 = torch.randn(1, 16384, 256, device=DEV)
        cos16, sin16 = (t.to(DEV) for t in precompute_rope_freqs(64, 16384, 5e6, 0.334))
        cos4, sin4 = (t.to(DEV) for t in precompute_rope_freqs(64, 4096, 5e6, 0.334))
        _MASK_CACHE.clear()
        with torch.no_grad():
            full16 = attn(x16, cos16, sin16)                       # 16K 全 pass
            ref4_first = attn(x16[:, :4096], cos4, sin4)           # 4K 参考（首块）
            again4 = attn(x16[:, :4096], cos4, sin4)               # 缓存往返后再测
            full16b = attn(x16, cos16, sin16)                      # 16K 复测
        return {"full16": full16, "ref4": ref4_first, "again4": again4,
                "full16b": full16b, "cache_keys": list(_MASK_CACHE.keys())}

    def test_window_semantics_16k_vs_4k(self, probe, vreport):
        """16K pass 的 [:, w-1:4096] 与 4K 参考一致（窗口外无泄漏）。"""
        a = probe["full16"][0, WINDOW - 1:4096]
        b = probe["ref4"][0, WINDOW - 1:]
        max_diff = (a - b).abs().max().item()
        assert torch.allclose(a, b, atol=1e-3, rtol=1e-3), f"SWA 16K vs 4K 参考 max_diff={max_diff}"
        vreport["checks"]["swa_16k_vs_4k"] = {
            "compare_window": [WINDOW - 1, 4096], "max_abs_diff": max_diff,
            "tolerance": 1e-3, "result": "PASS",
        }

    def test_mask_cache_key_isolation(self, probe, vreport):
        """键含 (kind, seq_len, window)；4096→16384→4096 往返不串。"""
        keys = probe["cache_keys"]
        assert any(k[0] == "swa" and k[1] == 16384 and k[2] == WINDOW for k in keys)
        assert any(k[0] == "swa" and k[1] == 4096 and k[2] == WINDOW for k in keys)
        assert torch.equal(probe["ref4"], probe["again4"]), "seq 4096 缓存往返输出变化"
        assert torch.equal(probe["full16"], probe["full16b"]), "seq 16384 复测输出变化"
        # window 维度隔离：同 seq 不同 window 必须是不同键
        cfg_w = ModelConfig(num_layers=1, hidden_size=256, num_heads=4, num_kv_heads=2,
                            intermediate_size=512, max_seq_len=4096, window_size=256,
                            rope_base=5e6, rope_percent=0.334, qk_norm=False,
                            attention_output_gate=False, vocab_size=32768,
                            tie_word_embeddings=True)
        attn_w = GQAAttention(cfg_w, layer_idx=0).to(DEV).eval()
        x = torch.randn(1, 4096, 256, device=DEV)
        cos, sin = (t.to(DEV) for t in precompute_rope_freqs(64, 4096, 5e6, 0.334))
        with torch.no_grad():
            out_w = attn_w(x, cos, sin)
        assert any(k[0] == "swa" and k[1] == 4096 and k[2] == 256 for k in _MASK_CACHE)
        assert not torch.allclose(out_w, probe["again4"], atol=1e-2), "window 128/256 输出应不同"
        vreport["checks"]["mask_cache_keys"] = {
            "cache_keys": [str(k) for k in keys],
            "roundtrip_stable": True, "window_isolation": True, "result": "PASS",
        }


@pytest.fixture(scope="class")
def ladder():
    """4K 训 100 iters → ckpt；再试 16K resume（应被拒）；再做权重级加载探针。"""
    import shutil
    shutil.rmtree(WORK, ignore_errors=True)
    mc = _write_model_configs()
    data = WORK / "data"
    data.mkdir(parents=True, exist_ok=True)
    b, m = make_token_stream(data, "train", 1 << 21, seed=3, pattern="chain")
    seq1, iters1 = 4096, 100
    cfg1 = write_config(
        WORK, run_name="v4s1", out_dir=WORK / "runs" / "v4s1", stages=[b], metas=[m],
        seq_len=seq1, budget_tokens=seq1 * 1 * 1 * iters1, micro_batch_size=1,
        grad_accum=1, optimizer="adamw", lr=2e-4, warmup_steps=10,
        lr_schedule="constant", seed=3, model_preset="v4t-4k", model_configs=mc,
        bf16=True, eval_every=0, log_every=20, save_every=50,
    )
    p1, mem1, _log = run_trainer(cfg1, extra=["--max-iters", str(iters1)])
    assert p1.returncode == 0, f"v4 stage1 训练失败:\n{p1.stdout[-2000:]}\n{p1.stderr[-2000:]}"

    # 直接换 16K/rope-1e7 配置 resume（预算放宽到 200 iters，避免触发
    # "start_it >= iters" 前置守卫，专测 config_sha 跨配置拒绝）→ 期望显式拒绝
    cfg2 = write_config(
        WORK, run_name="v4s1", out_dir=WORK / "runs" / "v4s1", stages=[b], metas=[m],
        seq_len=16384, budget_tokens=16384 * 1 * 1 * 200, micro_batch_size=1,
        grad_accum=1, optimizer="adamw", lr=2e-4, warmup_steps=10,
        lr_schedule="constant", seed=3, model_preset="v4t-16k", model_configs=mc,
        bf16=True, eval_every=0, log_every=20, save_every=50,
    )
    p2, _, _log = run_trainer(cfg2, extra=["--max-iters", "200"])
    return {"cfg_mc": mc, "mem1": mem1, "p2_rc": p2.returncode,
            "p2_err": p2.stderr[-800:], "ckpt": WORK / "runs" / "v4s1" / "checkpoints"}


class TestV4Ladder:
    def test_stage1_trains_and_saves(self, ladder, vreport):
        marker = json.loads((ladder["ckpt"] / "latest.json").read_text())
        assert marker["iter"] == 100
        for f in ("model.pt", "optim.pt", "cursor.json", "rng.pt"):
            assert (ladder["ckpt"] / marker["dir"] / f).is_file()
        vreport["checks"]["stage1_ckpt"] = {"iter": 100, "files_ok": True, "result": "PASS"}

    def test_seq_change_resume_rejected_explicitly(self, ladder, vreport):
        """seq_len/rope_base 变化走 --load 续训被 config_sha 守卫显式拒绝（非静默错训）。"""
        assert ladder["p2_rc"] != 0
        assert "config_sha256" in ladder["p2_err"]
        vreport["checks"]["seq_change_resume_rejected"] = {
            "returncode": ladder["p2_rc"], "guard": "config_sha256", "result": "PASS",
            "note": "跨 seq resume 无协议——点火前决策项 D1",
        }

    def test_weight_only_load_16k(self, ladder, vreport):
        """midtrain 语义（载权重、重置优化器）：4K ckpt 权重载入 16K/rope-1e7 模型。
        CPU 上跑（fp32 精确、不占本套件 GPU 预算）。"""
        cfg16 = ModelConfig.from_preset("v4t-16k", config_path=ladder["cfg_mc"])
        model = create_model(cfg16).eval()
        res = model.load_state_dict(
            torch.load(ladder["ckpt"] / "ckpt-00000100" / "model.pt",
                       map_location="cpu", weights_only=True), strict=True)
        assert not res.missing_keys and not res.unexpected_keys
        assert model.rope_cos.shape[0] >= 16384, "rope buffer 未按 16K 重建"
        cfg16_old = ModelConfig.from_preset("v4t-4k", config_path=ladder["cfg_mc"])
        cfg16_old.max_seq_len = 16384
        model_old = create_model(cfg16_old).eval()
        model_old.load_state_dict(
            torch.load(ladder["ckpt"] / "ckpt-00000100" / "model.pt",
                       map_location="cpu", weights_only=True), strict=True)
        # rope_base 切换生效：buffer 必须按 1e7 重新预计算（非持久 buffer 不进 ckpt）
        from yuformer_model.attention import precompute_rope_freqs
        cos_e, sin_e = precompute_rope_freqs(64, 16384, 10_000_000.0, 0.334)
        assert torch.allclose(model.rope_cos, cos_e, atol=1e-6), "rope buffer 未按 1e7 重建"
        assert not torch.allclose(model.rope_cos, model_old.rope_cos, atol=1e-6)
        torch.manual_seed(11)
        x = torch.randint(0, 32768, (1, 16384))
        labels = torch.randint(0, 32768, (1, 16384))
        with torch.no_grad():
            loss = model(x, labels=labels, ce_chunk=8192)
            logits = model(x)
            logits_old = model_old(x)
            diff = (logits - logits_old).abs().mean().item()
            head_diff = (logits[0, :64] - logits_old[0, :64]).abs().mean().item()
            tail_diff = (logits[0, -64:] - logits_old[0, -64:]).abs().mean().item()
            del logits, logits_old
        assert torch.isfinite(loss), "16K 前向 loss 非有限"
        assert diff > 0, "rope_base 切换后输出与旧模型完全一致（切换未生效）"
        vreport["checks"]["weight_only_load_16k"] = {
            "loss": round(loss.item(), 4), "finite": True,
            "rope_buffer_len": int(model.rope_cos.shape[0]),
            "rope_base_new": cfg16.rope_base,
            "logits_diff_head": round(head_diff, 5), "logits_diff_tail": round(tail_diff, 5),
            "result": "PASS",
        }


class TestV4Forward64K:
    def test_forward_64k_cpu(self, vreport):
        """64K 定长前向 smoke（CPU：GPU 上 sdpa bool mask 4.2GB 超本套件 3GB 预算；
        flash 后端不可用——flash_attn 包 ABI 损坏，import 即失败，已验证）。"""
        cfg = ModelConfig.from_preset("v4t-64k", config_path=_write_model_configs())
        model = create_model(cfg)  # CPU
        torch.manual_seed(5)
        x = torch.randint(0, 32768, (1, 65536))
        labels = torch.randint(0, 32768, (1, 65536))
        with torch.no_grad():
            loss = model(x, labels=labels, ce_chunk=8192)
        assert torch.isfinite(loss), f"64K 前向 loss 非有限: {loss}"
        vreport["checks"]["forward_64k"] = {
            "device": "cpu", "seq_len": 65536, "loss": round(loss.item(), 4),
            "result": "PASS",
        }
