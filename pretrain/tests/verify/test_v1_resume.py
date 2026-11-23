"""V1 断点续训原子性（宪法 #4/#6 + E-1/E-2 事故模式）。

设计：
  路线 A：tiny 预设 + 合成 chain 数据，独立目录不间断跑 300 iters（bf16，贴近生产）。
  路线 B：同配置内容、独立目录/ run_name，目标 300 iters；ckpt-100 落盘即 SIGKILL
          强杀（模拟断电/OOM kill），再用 RESUME_ITER=100 守卫续训到 300。
断言：
  1. 两路线 iter 200/300 的 loss 差 < 2e-3（CUDA 非确定容差）。
  2. checkpoint cursor 恢复后数据序与全量 replay 对应窗口逐样本 checksum 一致
     （无重复无遗漏）；cursor 记账精确（100 步 = 100×spr，300 步 = 300×spr）。
  3. checkpoint 文件完整（model/optim/cursor/rng + latest.json 原子生效）。
  4. 负向用例：RESUME_ITER 与 marker 不符时 fail-fast 拒绝启动（exit 4）。
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess as sp
import sys
import time
from pathlib import Path

import pytest

from conftest import (
    MODEL_SRC, PRETRAIN_SRC, TRACKER_ROOT, VERIFY_OUT, make_token_stream,
    read_metrics, run_trainer, write_config,
)

SEQ = 256
MICRO_BS = 2
GRAD_ACCUM = 2
SPR = MICRO_BS * GRAD_ACCUM  # samples per step, world=1
ITERS = 300
CKPT_ITER = 100
SEED = 1234
LOSS_TOL = 2e-3
WORK = VERIFY_OUT / "v1"


def _replay_checksums(stages, metas, start_sample: int, n_samples: int) -> list[str]:
    """从全局样本偏移 start_sample 起取 n_samples 个样本的逐样本 sha1（数据模块级 replay）。"""
    from yuformer_pretrain.data import DataCursor, PrefetchLoader, TokenStream, concat_streams

    streams = [TokenStream.open(b, m, SEQ) for b, m in zip(stages, metas)]
    total, _ = concat_streams(streams)
    cursor = DataCursor(total, epochs=1 << 20)
    cursor.consumed = start_sample
    out = []
    with PrefetchLoader(streams, cursor, num_workers=2) as src:
        for _ in range(n_samples):
            arr = src.next_sample()
            out.append(hashlib.sha1(arr.tobytes()).hexdigest())
    return out


def _trainer_env() -> dict:
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": "3",
        "PYTHONPATH": f"{PRETRAIN_SRC}:{MODEL_SRC}:{env.get('PYTHONPATH', '')}",
        "YUFORMER_TRACKER_ROOT": str(TRACKER_ROOT),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(29517 + (os.getpid() % 500)),
        "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0",
    })
    return env


@pytest.fixture(scope="class")
def world():
    import shutil
    shutil.rmtree(WORK, ignore_errors=True)
    data = WORK / "data"
    data.mkdir(parents=True, exist_ok=True)
    b, m = make_token_stream(data, "train", 1 << 20, seed=SEED, pattern="chain")
    budget = SEQ * MICRO_BS * GRAD_ACCUM * ITERS

    def cfg_for(run_name: str) -> Path:
        return write_config(
            WORK, run_name=run_name, out_dir=WORK / "runs" / run_name, stages=[b], metas=[m],
            seq_len=SEQ, budget_tokens=budget, micro_batch_size=MICRO_BS,
            grad_accum=GRAD_ACCUM, optimizer="adamw", lr=2e-4, warmup_steps=20,
            lr_schedule="constant", seed=SEED, model_preset="tiny", bf16=True,
            eval_every=0, log_every=10, save_every=CKPT_ITER,
        )

    cfg_a = cfg_for("v1runA")
    pa, mem_a, _log = run_trainer(cfg_a, extra=["--max-iters", str(ITERS)])
    assert pa.returncode == 0, f"路线 A 失败:\n{pa.stdout[-2000:]}\n{pa.stderr[-2000:]}"

    cfg_b = cfg_for("v1runB")
    # start_new_session：kill 时按进程组 SIGKILL，连 PrefetchLoader fork worker 一起清
    # （E-2 家族：daemon worker 在父被 SIGKILL 时收不到 atexit，会孤儿化并继承 PIPE 写端）
    leg1_log = open(cfg_b.with_suffix(".leg1.log"), "w", encoding="utf-8")
    proc = sp.Popen(
        [sys.executable, "-m", "yuformer_pretrain.trainer", "--config", str(cfg_b)],
        env=_trainer_env(), stdout=leg1_log, stderr=sp.STDOUT,
        start_new_session=True,
    )
    killed = False
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        marker = WORK / "runs" / "v1runB" / "checkpoints" / "latest.json"
        if marker.is_file() and json.loads(marker.read_text())["iter"] == CKPT_ITER:
            time.sleep(1.0)  # 等 os.replace 完全落盘
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            killed = True
            break
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    proc.wait()
    leg1_log.close()
    out_b1 = cfg_b.with_suffix(".leg1.log").read_text(errors="replace")
    assert killed, f"路线 B 未在限期内 kill（ckpt-100 未落盘？）:\n{out_b1[-2000:]}"

    # 负向用例：RESUME_ITER 错误值 → exit 4
    p_bad, _, _log = run_trainer(cfg_b, env_extra={"RESUME_ITER": str(CKPT_ITER - 1)})

    # 路线 B 续训：RESUME_ITER=100 守卫 → 300
    p_b2, mem_b, _log = run_trainer(cfg_b, extra=["--max-iters", str(ITERS)],
                                  env_extra={"RESUME_ITER": str(CKPT_ITER)})
    assert p_b2.returncode == 0, f"路线 B 续训失败:\n{p_b2.stdout[-2000:]}\n{p_b2.stderr[-2000:]}"

    return {
        "stages": [b], "metas": [m],
        "losses_a": read_metrics(WORK / "runs" / "v1runA", "v1runA"),
        "losses_b": read_metrics(WORK / "runs" / "v1runB", "v1runB"),
        "mem_a": mem_a, "mem_b": mem_b,
        "p_bad_rc": p_bad.returncode, "p_bad_err": p_bad.stderr[-500:],
    }


class TestV1ResumeAtomicity:
    def test_loss_parity_across_crash(self, world, vreport):
        la, lb = world["losses_a"], world["losses_b"]
        # log_every=10 → 已记录 step = 1,11,21,…,291,300；取断点后的全部公共 step 比对
        common = sorted(s for s in set(la) & set(lb)
                        if s > CKPT_ITER and "loss" in la[s] and "loss" in lb[s])
        assert len(common) >= 10, f"公共 log 点不足: {common}"
        diffs = {s: abs(la[s]["loss"] - lb[s]["loss"]) for s in common}
        worst = max(diffs, key=diffs.get)
        assert diffs[worst] < LOSS_TOL, f"iter {worst} loss 差 {diffs[worst]} ≥ {LOSS_TOL}"
        vreport["checks"]["loss_parity"] = {
            "steps_compared": len(common),
            "max_abs_diff": diffs[worst], "worst_step": worst,
            "loss_A@300": la[ITERS]["loss"], "loss_B@300": lb[ITERS]["loss"],
            "bit_identical": all(d == 0.0 for d in diffs.values()),
            "tolerance": LOSS_TOL, "result": "PASS",
        }

    def test_marker_negative(self, world, vreport):
        assert world["p_bad_rc"] == 4, f"RESUME_ITER 守卫未拒绝：rc={world['p_bad_rc']}"
        assert "RESUME_ITER" in world["p_bad_err"]
        vreport["checks"]["marker_rejects_wrong_resume_iter"] = {
            "exit_code": world["p_bad_rc"], "expected": 4, "result": "PASS",
        }

    def test_checkpoint_integrity_and_cursor(self, world, vreport):
        ck = WORK / "runs" / "v1runB" / "checkpoints"
        marker = json.loads((ck / "latest.json").read_text())
        assert marker["iter"] == ITERS
        d100 = ck / "ckpt-00000100"
        d300 = ck / f"ckpt-{ITERS:08d}"
        for d in (d100, d300):
            for f in ("model.pt", "optim.pt", "cursor.json", "rng.pt"):
                assert (d / f).is_file(), f"checkpoint 缺 {f} @ {d}"
        cur100 = json.loads((d100 / "cursor.json").read_text())
        cur300 = json.loads((d300 / "cursor.json").read_text())
        assert cur100["cursor"]["consumed_samples"] == CKPT_ITER * SPR
        assert cur300["cursor"]["consumed_samples"] == ITERS * SPR
        vreport["checks"]["checkpoint_integrity"] = {
            "consumed@100": cur100["cursor"]["consumed_samples"],
            "consumed@300": cur300["cursor"]["consumed_samples"],
            "expected": [CKPT_ITER * SPR, ITERS * SPR],
            "files": ["model.pt", "optim.pt", "cursor.json", "rng.pt"],
            "result": "PASS",
        }

    def test_data_order_replay(self, world, vreport):
        ck = WORK / "runs" / "v1runB" / "checkpoints"
        cur100 = json.loads((ck / "ckpt-00000100" / "cursor.json").read_text())
        start = cur100["cursor"]["consumed_samples"]
        n = (ITERS - CKPT_ITER) * SPR
        ck_from_ckpt = _replay_checksums(world["stages"], world["metas"], start, n)
        ck_from_zero = _replay_checksums(world["stages"], world["metas"], 0, start + n)[start:]
        assert ck_from_ckpt == ck_from_zero
        vreport["checks"]["data_order_replay"] = {
            "samples_compared": n, "mismatches": 0, "result": "PASS",
        }

    def test_vram_budget(self, world, vreport):
        for k in ("mem_a", "mem_b"):
            assert world[k] < 3200, f"{k} 峰值显存 {world[k]}MB 超预算"
        vreport["checks"]["vram"] = {
            "peak_delta_mb": {"A": world["mem_a"], "B": world["mem_b"]},
            "budget_mb": 3200, "result": "PASS",
        }
