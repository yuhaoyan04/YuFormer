"""lr_at 学习率调度单测：warmup 线性段、constant/cosine 形态、边界条件，
以及 Muon 双优化器路由的配置解析层（lr multiplier 硬编码现状钉住，不改实现）。"""
import math

import pytest
import torch

from yuformer_pretrain.config import OptimConfig
from yuformer_pretrain.muon import Muon, zeropower_via_newtonschulz5
from yuformer_pretrain.optim import build_optimizers, is_muon_param, lr_at


def _cfg(**kw):
    base = dict(optimizer="adamw", lr=1.0, weight_decay=0.0, warmup_steps=10,
                lr_schedule="constant", min_lr_ratio=0.1, adam_betas=(0.9, 0.95),
                muon_momentum=0.95, muon_nesterov=True, muon_ns_steps=5,
                clip_grad=1.0)
    base.update(kw)
    return OptimConfig(**base)


def test_warmup_linear_ramp():
    cfg = _cfg(lr=1.0, warmup_steps=10, lr_schedule="constant")
    assert lr_at(0, 100, cfg) == pytest.approx(0.1)
    assert lr_at(4, 100, cfg) == pytest.approx(0.5)
    assert lr_at(9, 100, cfg) == pytest.approx(1.0)  # warmup 终点 = base
    assert lr_at(10, 100, cfg) == pytest.approx(1.0)
    seq = [lr_at(s, 100, cfg) for s in range(10)]
    assert all(b >= a for a, b in zip(seq, seq[1:]))  # 线性单调上升


def test_warmup_with_realistic_lr():
    cfg = _cfg(lr=2e-4, warmup_steps=10)
    assert lr_at(0, 100, cfg) == pytest.approx(2e-5)
    assert lr_at(9, 100, cfg) == pytest.approx(2e-4)


def test_constant_holds_after_warmup_and_past_total():
    cfg = _cfg(lr=1.0, warmup_steps=10, lr_schedule="constant")
    assert lr_at(10, 50, cfg) == 1.0
    assert lr_at(49, 50, cfg) == 1.0
    assert lr_at(10**6, 50, cfg) == 1.0  # constant 与 total 无关，越界不衰减


def test_cosine_hand_computed_points():
    cfg = _cfg(lr=1.0, warmup_steps=0, lr_schedule="cosine", min_lr_ratio=0.1)
    total = 100
    # progress = step/total；lr = base × (0.1 + 0.9 × 0.5 × (1+cos(π·progress)))
    assert lr_at(0, total, cfg) == pytest.approx(1.0, abs=1e-12)
    assert lr_at(25, total, cfg) == pytest.approx(0.8681980515339465, abs=1e-12)
    assert lr_at(50, total, cfg) == pytest.approx(0.55, abs=1e-12)
    assert lr_at(75, total, cfg) == pytest.approx(0.2318019484660536, abs=1e-12)
    assert lr_at(100, total, cfg) == pytest.approx(0.1, abs=1e-12)


def test_cosine_monotonic_decay_to_min():
    cfg = _cfg(lr=1.0, warmup_steps=0, lr_schedule="cosine", min_lr_ratio=0.1)
    total = 100
    seq = [lr_at(s, total, cfg) for s in range(total + 1)]
    assert all(b <= a for a, b in zip(seq, seq[1:]))
    assert seq[0] == pytest.approx(1.0)      # decay 起点 = base
    assert seq[-1] == pytest.approx(0.1)     # decay 终点 = base × min_lr_ratio


def test_cosine_warmup_then_decay_shape():
    cfg = _cfg(lr=1.0, warmup_steps=10, lr_schedule="cosine", min_lr_ratio=0.1)
    total = 110
    assert lr_at(9, total, cfg) == pytest.approx(1.0)    # warmup 终点
    assert lr_at(10, total, cfg) == pytest.approx(1.0)   # decay 起点（progress=0）
    assert lr_at(60, total, cfg) == pytest.approx(0.55)  # progress=0.5
    assert lr_at(110, total, cfg) == pytest.approx(0.1)  # progress=1


def test_wsd_holds_then_decays_over_declared_tail():
    cfg = _cfg(
        lr=1.0,
        warmup_steps=10,
        lr_schedule="wsd",
        decay_steps=20,
        min_lr_ratio=0.1,
    )
    total = 100
    assert lr_at(9, total, cfg) == pytest.approx(1.0)
    assert lr_at(10, total, cfg) == pytest.approx(1.0)
    assert lr_at(79, total, cfg) == pytest.approx(1.0)
    assert lr_at(80, total, cfg) == pytest.approx(1.0)
    assert lr_at(90, total, cfg) == pytest.approx(0.55)
    assert lr_at(100, total, cfg) == pytest.approx(0.1)
    assert lr_at(101, total, cfg) == pytest.approx(0.1)


def test_step_beyond_total_clamps_to_min():
    cfg = _cfg(lr=1.0, warmup_steps=0, lr_schedule="cosine", min_lr_ratio=0.1)
    assert lr_at(101, 100, cfg) == pytest.approx(0.1)
    assert lr_at(10**6, 100, cfg) == pytest.approx(0.1)


def test_zero_warmup_no_division_error():
    cfg_cos = _cfg(lr=1.0, warmup_steps=0, lr_schedule="cosine", min_lr_ratio=0.1)
    assert lr_at(0, 100, cfg_cos) == pytest.approx(1.0)
    cfg_const = _cfg(lr=1.0, warmup_steps=0, lr_schedule="constant")
    assert lr_at(0, 100, cfg_const) == 1.0


def test_min_lr_ratio_zero_decays_to_zero():
    cfg = _cfg(lr=1.0, warmup_steps=0, lr_schedule="cosine", min_lr_ratio=0.0)
    assert lr_at(100, 100, cfg) == 0.0
    assert lr_at(50, 100, cfg) == pytest.approx(0.5)


def test_min_lr_ratio_one_is_flat():
    cfg = _cfg(lr=1.0, warmup_steps=0, lr_schedule="cosine", min_lr_ratio=1.0)
    for s in (0, 25, 50, 75, 100):
        assert lr_at(s, 100, cfg) == 1.0


def test_total_equals_warmup_no_decay_room():
    # total == warmup：max(1, total-warmup) 守卫防 0 除；首个 post-warmup step
    # progress=0 仍拿 base，再往后 clamp 到 min
    cfg = _cfg(lr=1.0, warmup_steps=10, lr_schedule="cosine", min_lr_ratio=0.1)
    assert lr_at(9, 10, cfg) == pytest.approx(1.0)
    assert lr_at(10, 10, cfg) == pytest.approx(1.0)
    assert lr_at(11, 10, cfg) == pytest.approx(0.1)


def test_is_muon_param_routing():
    p2d = torch.nn.Parameter(torch.randn(4, 8))
    p1d = torch.nn.Parameter(torch.randn(8))
    assert is_muon_param("proj.weight", p2d)
    assert not is_muon_param("norm.weight", p1d)          # 非 2D
    assert not is_muon_param("proj.bias", p1d)            # 非 2D
    assert not is_muon_param("embed_tokens.weight", p2d)  # 名字含 embed
    assert not is_muon_param("lm_head.weight", p2d)       # 名字含 lm_head


def _mixed_model():
    torch.manual_seed(0)
    m = torch.nn.Module()
    m.embed_tokens = torch.nn.Embedding(32, 8)
    m.norm = torch.nn.LayerNorm(8)
    m.proj = torch.nn.Linear(8, 16)
    return m


def _group_fqns(model, opt):
    id2n = {id(p): n for n, p in model.named_parameters()}
    return {id2n[id(p)] for g in opt.param_groups for p in g["params"]}


def test_build_optimizers_adamw_single(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)  # CPU: fused=False
    model = _mixed_model()
    opts = build_optimizers(model, _cfg(optimizer="adamw", lr=2e-4, weight_decay=0.1))
    assert len(opts) == 1
    (adamw,) = opts
    assert _group_fqns(model, adamw) == {n for n, _ in model.named_parameters()}
    g0 = adamw.param_groups[0]
    assert g0["lr"] == 2e-4
    assert g0["betas"] == (0.9, 0.95)
    assert g0["weight_decay"] == 0.1


def test_build_optimizers_muon_partition(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = _mixed_model()
    cfg = _cfg(optimizer="muon")
    opts = build_optimizers(model, cfg)
    assert [type(o) for o in opts] == [torch.optim.AdamW, Muon]
    adamw, muon = opts
    all_fqns = {n for n, _ in model.named_parameters()}
    a, m = _group_fqns(model, adamw), _group_fqns(model, muon)
    assert a | m == all_fqns          # 全覆盖
    assert not (a & m)                # 不相交
    assert m == {"proj.weight"}       # 唯二矩阵走 Muon
    assert "embed_tokens.weight" in a  # embedding 走 AdamW
    mg = muon.param_groups[0]
    assert mg["lr"] == cfg.lr
    assert mg["momentum"] == cfg.muon_momentum
    assert mg["nesterov"] == cfg.muon_nesterov
    assert mg["ns_steps"] == cfg.muon_ns_steps


def test_muon_lr_multiplier_current_behavior():
    """lr_eff = lr × 0.2 × sqrt(max(fan)) 是硬编码配方（muon.py），测现状不改实现：
    若配方调整，本测试会失败并提示同步更新。"""
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(6, 8))
    g = torch.randn(6, 8)
    orig = w.detach().clone()
    opt = Muon([w], lr=0.05, momentum=0.0, nesterov=False,
               weight_decay=0.0, ns_steps=5)
    w.grad = g.clone()
    opt.step()
    u = zeropower_via_newtonschulz5(g, steps=5) * max(1.0, 6 / 8) ** 0.5
    lr_eff = 0.05 * 0.2 * math.sqrt(8)
    assert torch.equal(w.detach(), orig - lr_eff * u)
