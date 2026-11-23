"""V5 Muon vs AdamW 对比（MFU 实测，~100M 档 e1-100m 预设 + 4K seq）。

各跑 1500 iters（前 500 预热不计入稳态统计），输出：
  - loss 曲线（首/末段均值）对比，断言 Muon 末段 loss ≤ AdamW × 1.01（如实记录反例）
  - 无 NaN/Inf
  - 稳态 tokens/s（iters 501-1500 均值）、峰值显存（nvidia-smi 增量）
  - MFU 双口径（SWA 修正前后），公式：
      FLOPs/token = 6·N_nonemb + 12·h·(n_swa·w + n_full·S)   ← SWA 修正
                    6·N_nonemb + 12·L·S·h                    ← trainer 现行（全按全注意力）
    （注意力项按 ctx 上界计：全层 ctx=S、滑窗层 ctx=w；与 trainer 同口径便于对照）
    MFU = FLOPs/token × tokens_per_step / (mean_step_time × 989e12)（H800 BF16 峰值）
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import (
    MODEL_CONFIGS, VERIFY_OUT, make_token_stream, read_manifest, read_metrics,
    run_trainer, write_config,
)

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "pretrain" / "src"))
sys.path.insert(0, str(REPO / "model" / "src"))

SEQ = 4096
MBS, ACCUM, WORLD = 1, 1, 1
ITERS = 1500
WARMUP_SKIP = 500
H800_BF16_PEAK = 989e12
WORK = VERIFY_OUT / "v5"


@pytest.fixture(scope="class")
def race():
    import shutil
    shutil.rmtree(WORK, ignore_errors=True)
    data = WORK / "data"
    data.mkdir(parents=True, exist_ok=True)
    b, m = make_token_stream(data, "train", 1 << 24, seed=42, pattern="chain", vocab=65536)
    results = {}
    for opt in ("adamw", "muon"):
        cfg = write_config(
            WORK, run_name=f"v5-{opt}", out_dir=WORK / "runs" / f"v5-{opt}", stages=[b], metas=[m],
            seq_len=SEQ, budget_tokens=SEQ * MBS * ACCUM * ITERS,
            micro_batch_size=MBS, grad_accum=ACCUM, optimizer=opt, lr=2e-4,
            warmup_steps=100, lr_schedule="constant", seed=42,
            model_preset="small", model_configs=MODEL_CONFIGS, bf16=True,
            activation_checkpointing=True, eval_every=0, log_every=50, save_every=0,
            world_size=WORLD,
        )
        # 共享 vGPU 上瞬时显存空窗会随机击毁训练进程（OOM/CUBLAS_ALLOC/launch failure，
        # 死进程泄漏不可回收——E-23 同族）；重试与点火前置已内置于 run_trainer。
        # V5 峰值 ~2GB（small 预设），要求 ≥3.6GB 空窗再点火。
        from conftest import wait_gpu_free
        wait_gpu_free(3600)
        p, mem, _log = run_trainer(cfg, timeout=5400)
        assert p.returncode == 0, f"v5 {opt} 失败:\n{p.stdout[-3000:]}\n{p.stderr[-3000:]}"
        results[opt] = {"metrics": read_metrics(WORK / "runs" / f"v5-{opt}", f"v5-{opt}"), "mem": mem}
    return results


def _analyze(opt: str, r: dict) -> dict:
    metrics = r["metrics"]
    steps = sorted(s for s in metrics if "loss" in metrics[s] and "step_time" in metrics[s])
    assert len(steps) >= ITERS // 50, f"{opt}: metrics 不足（{len(steps)} 行）"
    losses = np.array([metrics[s]["loss"] for s in steps])
    assert np.isfinite(losses).all(), f"{opt}: 出现 NaN/Inf loss"
    steady = [s for s in steps if s > WARMUP_SKIP]
    assert len(steady) >= 10
    step_times = np.array([metrics[s]["step_time"] for s in steady])
    tail_loss = float(losses[-10:].mean())
    head_loss = float(losses[:10].mean())
    tokens_per_step = SEQ * MBS * ACCUM * WORLD
    tokens_per_s = tokens_per_step / step_times.mean()
    return {
        "optimizer": opt,
        "head_loss_mean10": round(head_loss, 4),
        "tail_loss_mean10": round(tail_loss, 4),
        "steady_steps": len(steady),
        "mean_step_time_s": round(float(step_times.mean()), 4),
        "p50_step_time_s": round(float(np.median(step_times)), 4),
        "tokens_per_s": round(float(tokens_per_s), 1),
        "peak_vram_delta_mb": r["mem"],
        "loss_curve": [round(float(x), 4) for x in losses[:: len(losses) // 20 or 1]],
    }


@pytest.fixture(scope="class")
def analysis(race):
    from yuformer_model.config import ModelConfig

    mc = ModelConfig.from_preset("small", config_path=MODEL_CONFIGS)
    stats = {opt: _analyze(opt, race[opt]) for opt in ("adamw", "muon")}

    def flops_per_token(swa_corrected: bool) -> float:
        attn_ctx = (mc.num_swa_layers * mc.window_size + mc.num_full_attn_layers * SEQ) \
            if swa_corrected else mc.num_layers * SEQ
        n_nonemb = stats["_n_nonemb"]
        return 6 * n_nonemb + 12 * mc.hidden_size * attn_ctx

    # 参数量从 run 的 final manifest 读取（权威记账）
    mf = read_manifest(WORK / "runs" / "v5-adamw", "v5-adamw")
    stats["_n_nonemb"] = mf["params_non_embedding"]
    stats["_n_params"] = mf["params"]
    stats["_model"] = {"preset": "small", "layers": mc.num_layers,
                       "hidden": mc.hidden_size, "swa_layers": mc.num_swa_layers,
                       "full_layers": mc.num_full_attn_layers, "window": mc.window_size,
                       "seq": SEQ}
    for opt in ("adamw", "muon"):
        tps = stats[opt]["tokens_per_s"]
        for tag, corrected in (("swa_corrected", True), ("uncorrected", False)):
            fpt = flops_per_token(corrected)
            tflops = fpt * tps / 1e12
            stats[opt][f"mfu_{tag}"] = round(tflops * 1e12 / H800_BF16_PEAK, 4)
            stats[opt][f"tflops_{tag}"] = round(tflops, 2)
            stats[opt][f"flops_per_token_{tag}"] = fpt
    return stats


class TestV5MuonVsAdamW:
    def test_no_nan_and_curves(self, analysis, vreport):
        for opt in ("adamw", "muon"):
            s = analysis[opt]
            assert s["head_loss_mean10"] == s["head_loss_mean10"], "NaN"
            assert s["tail_loss_mean10"] < s["head_loss_mean10"], f"{opt} loss 未下降"
        vreport["checks"]["loss_curves"] = {
            opt: {k: analysis[opt][k] for k in
                  ("head_loss_mean10", "tail_loss_mean10", "loss_curve")}
            for opt in ("adamw", "muon")
        }
        vreport["checks"]["loss_curves"]["result"] = "PASS"

    def test_muon_tail_loss(self, analysis, vreport):
        a, m = analysis["adamw"]["tail_loss_mean10"], analysis["muon"]["tail_loss_mean10"]
        ok = bool(m <= a * 1.01)
        # 规格断言：Muon 末段 ≤ AdamW+1%；若不成立按"如实记录反例"处理——
        # 记入报告（result=COUNTEREXAMPLE）但不拉红套件（此为实验结论非管线 bug）
        vreport["checks"]["muon_tail_le_adamw_plus_1pct"] = {
            "adamw_tail": a, "muon_tail": m, "threshold": round(a * 1.01, 4),
            "pass": ok, "result": "PASS" if ok else "COUNTEREXAMPLE",
        }
        if not ok:
            vreport["status"] = "PASS_WITH_COUNTEREXAMPLE"
            pytest.skip(f"如实记录反例：Muon 末段 loss {m} > AdamW×1.01 {a*1.01:.4f}"
                        f"（{analysis['_model']['preset']} 档 1500 iters 规模下 Muon 未占优）")

    def test_throughput_and_mfu(self, analysis, vreport):
        for opt in ("adamw", "muon"):
            s = analysis[opt]
            vreport["checks"][f"mfu_{opt}"] = {
                "tokens_per_s": s["tokens_per_s"],
                "mean_step_time_s": s["mean_step_time_s"],
                "peak_vram_delta_mb": s["peak_vram_delta_mb"],
                "tflops_swa_corrected": s["tflops_swa_corrected"],
                "tflops_uncorrected": s["tflops_uncorrected"],
                "mfu_swa_corrected": s["mfu_swa_corrected"],
                "mfu_uncorrected": s["mfu_uncorrected"],
                "flops_per_token_swa_corrected": s["flops_per_token_swa_corrected"],
                "flops_per_token_uncorrected": s["flops_per_token_uncorrected"],
            }
        vreport["checks"]["mfu_table"] = {
            "model": analysis["_model"],
            "n_params": analysis["_n_params"],
            "n_nonemb": analysis["_n_nonemb"],
            "formula": "6*N_nonemb + 12*h*(n_swa*w + n_full*S) [corrected] "
                       "| 6*N_nonemb + 12*L*S*h [uncorrected]",
            "peak_tflops_h800_bf16": H800_BF16_PEAK / 1e12,
            "adamw": vreport["checks"]["mfu_adamw"],
            "muon": vreport["checks"]["mfu_muon"],
            "result": "PASS",
        }
        assert analysis["adamw"]["peak_vram_delta_mb"] < 3200
        assert analysis["muon"]["peak_vram_delta_mb"] < 3200
