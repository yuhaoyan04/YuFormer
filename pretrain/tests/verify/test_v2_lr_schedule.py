"""V2 LR schedule / warmup 原子验证。

三层断言：
  1. 数值形状（单测级）：warmup 线性 0→peak 逐步对拍（非仅单调）；constant 在
     warmup 后严格等于 peak（一丝不衰减——ZGCM 配方）；cosine 端点
     （step=warmup → peak；step=total → min_lr_ratio×peak）与 warmump 边界
     （最后一步 warmup = peak，下一步 = peak）。
  2. wiring 级：build_optimizers + set_lrs 后，优化器 param_groups 里的真实 lr
     等于 lr_at 给定值（覆盖 muon 双优化器两条路由）。
  3. 集成级：子进程 tiny 训练 log_every=1，metrics.jsonl 记录的逐步 lr 与独立
     重算的期望序列逐步一致（抓 trainer 用错 step 索引/传错优化器等接线 bug）。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

from conftest import VERIFY_OUT, make_token_stream, read_metrics, run_trainer, write_config

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "pretrain" / "src"))

from yuformer_pretrain.config import OptimConfig  # noqa: E402
from yuformer_pretrain.optim import build_optimizers, lr_at, set_lrs  # noqa: E402

PEAK = 2e-4


def _cfg(**kw) -> OptimConfig:
    base = dict(optimizer="adamw", lr=PEAK, weight_decay=0.1, warmup_steps=10,
                lr_schedule="constant", min_lr_ratio=0.1, adam_betas=(0.9, 0.95),
                muon_momentum=0.95, muon_nesterov=True, muon_ns_steps=5, clip_grad=1.0)
    base.update(kw)
    return OptimConfig(**base)


class TestV2LRShape:
    def test_warmup_linear_exact(self, vreport):
        cfg = _cfg(warmup_steps=10, lr_schedule="constant")
        expected = [PEAK * (s + 1) / 10 for s in range(10)]
        actual = [lr_at(s, 100, cfg) for s in range(10)]
        assert actual == pytest.approx(expected, rel=1e-12)  # 逐步对拍，非仅单调
        vreport["checks"]["warmup_linear_exact"] = {
            "lr[0]": actual[0], "lr[9]": actual[9], "expected[0]": expected[0],
            "result": "PASS",
        }

    def test_warmup_boundary_then_constant_exact(self, vreport):
        cfg = _cfg(warmup_steps=10, lr_schedule="constant")
        assert lr_at(9, 100, cfg) == pytest.approx(PEAK, rel=1e-15)   # warmup 末步 = peak
        for s in (10, 11, 50, 99, 100, 10**6):                        # 下一步起 = peak，越界不衰减
            assert lr_at(s, 100, cfg) == pytest.approx(PEAK, rel=1e-15)
        vreport["checks"]["constant_exact"] = {"result": "PASS"}

    def test_cosine_endpoints(self, vreport):
        cfg = _cfg(warmup_steps=10, lr_schedule="cosine", min_lr_ratio=0.1)
        total = 100
        assert lr_at(10, total, cfg) == pytest.approx(PEAK, rel=1e-15)          # warmup 后 = peak
        assert lr_at(total, total, cfg) == pytest.approx(PEAK * 0.1, rel=1e-15)  # 末端 = min_ratio×peak
        mid = lr_at((10 + total) // 2, total, cfg)
        assert mid == pytest.approx(PEAK * (0.1 + 0.9 * 0.5), rel=1e-15)         # 中点
        # 端点外侧不衰破 min_lr_ratio
        assert lr_at(total + 1, total, cfg) == pytest.approx(PEAK * 0.1, rel=1e-15)
        vreport["checks"]["cosine_endpoints"] = {
            "lr@warmup": PEAK, "lr@end": PEAK * 0.1, "lr@mid": mid, "result": "PASS",
        }


class TestV2OptimizerWiring:
    @pytest.mark.parametrize("optimizer", ["adamw", "muon"])
    def test_param_groups_receive_lr(self, optimizer, vreport):
        import torch

        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Embedding(64, 32), torch.nn.Linear(32, 64), torch.nn.Tanh(),
            torch.nn.Linear(64, 32),
        ).to("cuda")
        opts = build_optimizers(model, _cfg(optimizer=optimizer))
        for step in (0, 5, 9, 10, 50):
            want = lr_at(step, 100, _cfg(optimizer=optimizer))
            set_lrs(opts, want)
            for o in opts:
                for g in o.param_groups:
                    assert g["lr"] == pytest.approx(want, rel=1e-15)
        vreport["checks"][f"wiring_{optimizer}"] = {"optimizers": len(opts), "result": "PASS"}


class TestV2Integration:
    @pytest.fixture(scope="class")
    def run(self):
        import shutil
        d = VERIFY_OUT / "v2"
        shutil.rmtree(d, ignore_errors=True)
        data = d / "data"
        data.mkdir(parents=True, exist_ok=True)
        b, m = make_token_stream(data, "train", 1 << 18, seed=7, pattern="chain")
        seq, mbs, accum, iters, warmup = 128, 2, 1, 60, 10
        cfg = write_config(
            d, run_name="v2run", out_dir=d / "runs" / "v2run", stages=[b], metas=[m],
            seq_len=seq, budget_tokens=seq * mbs * accum * iters,
            micro_batch_size=mbs, grad_accum=accum, optimizer="muon", lr=PEAK,
            warmup_steps=warmup, lr_schedule="cosine", min_lr_ratio=0.1,
            seed=7, model_preset="tiny", bf16=True, eval_every=0, log_every=1,
            save_every=0,
        )
        p, mem, _log = run_trainer(cfg, extra=["--max-iters", str(iters)])
        assert p.returncode == 0, f"V2 集成 run 失败:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}"
        return {"metrics": read_metrics(d / "runs" / "v2run", "v2run"), "warmup": warmup,
                "iters": iters, "mem": mem}

    def test_logged_lr_matches_schedule(self, run, vreport):
        cfg = _cfg(optimizer="muon", warmup_steps=run["warmup"],
                   lr_schedule="cosine", min_lr_ratio=0.1)
        mismatches = []
        for step in range(1, run["iters"] + 1):  # metrics step = it+1
            want = lr_at(step - 1, run["iters"], cfg)
            got = run["metrics"][step]["lr"]
            if abs(got - want) > 1e-15 * max(1.0, want):
                mismatches.append((step, got, want))
        assert not mismatches, f"lr 序列不一致（前 5 处）: {mismatches[:5]}"
        vreport["checks"]["integration_logged_lr"] = {
            "steps_checked": run["iters"], "mismatches": 0,
            "lr@1": run["metrics"][1]["lr"], "lr@warmup_end": run["metrics"][run["warmup"]]["lr"],
            "lr@final": run["metrics"][run["iters"]]["lr"],
            "result": "PASS",
        }

    def test_vram_budget(self, run, vreport):
        assert run["mem"] < 3200
        vreport["checks"]["vram"] = {"peak_delta_mb": run["mem"], "budget_mb": 3200,
                                     "result": "PASS"}
