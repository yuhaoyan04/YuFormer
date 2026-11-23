"""V3 课程学习两段衔接原子验证。

构造两个常量 token 流（stage1 全 0x0001 / stage2 全 0x0002，序可检测）：
  1. cursor 按 stage 顺序消费：stage 内样本全 1、stage2 全 2，边界精确。
  2. manifest 记录段切换点（data_stage_samples / data_stage_switch_at_samples）。
  3. 全程 token checksum 总和守恒（无重复无遗漏）：
     Σ = 1×n1×span + 2×n2_consumed×span。
  4. stage2 中途断点续训（ckpt 在段内）后数据序仍正确，loss 与不间断参照一致。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import VERIFY_OUT, make_token_stream, read_manifest, read_metrics, run_trainer, write_config

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "pretrain" / "src"))

SEQ = 32
SPAN = SEQ + 1
MICRO_BS = 1
GRAD_ACCUM = 2
SPR = MICRO_BS * GRAD_ACCUM
S1_SAMPLES = 40          # stage1：40 样本（边界落在 iter 20 中间）
S2_SAMPLES = 160
ITERS = 60               # 全程 120 样本 → 深入 stage2
CKPT_ITER = 40           # consumed=80，stage2 内 40 样本处断点
LOSS_TOL = 2e-3
WORK = VERIFY_OUT / "v3"


def _open_streams():
    from yuformer_pretrain.data import TokenStream

    return [TokenStream.open(WORK / "data" / f"{n}.tokens.bin",
                             WORK / "data" / f"{n}.meta.json", SEQ)
            for n in ("stage1", "stage2")]


@pytest.fixture(scope="class")
def world():
    import shutil
    shutil.rmtree(WORK, ignore_errors=True)
    data = WORK / "data"
    data.mkdir(parents=True, exist_ok=True)
    b1, m1 = make_token_stream(data, "stage1", S1_SAMPLES * SPAN, seed=0, pattern=1)
    b2, m2 = make_token_stream(data, "stage2", S2_SAMPLES * SPAN, seed=0, pattern=2)
    budget = SEQ * SPR * ITERS

    def cfg_for(run_name):
        return write_config(
            WORK, run_name=run_name, out_dir=WORK / "runs" / run_name,
            stages=[b1, b2], metas=[m1, m2], seq_len=SEQ,
            budget_tokens=budget, micro_batch_size=MICRO_BS, grad_accum=GRAD_ACCUM,
            optimizer="adamw", lr=2e-4, warmup_steps=5, lr_schedule="constant",
            seed=11, model_preset="tiny", bf16=True, eval_every=0, log_every=10,
            save_every=CKPT_ITER,
        )

    # 参照：不间断 60 iters
    cfg_ref = cfg_for("v3ref")
    p_ref, mem_ref, _log = run_trainer(cfg_ref, extra=["--max-iters", str(ITERS)])
    assert p_ref.returncode == 0, f"v3ref 失败:\n{p_ref.stdout[-2000:]}\n{p_ref.stderr[-2000:]}"
    # 断点路线：40 iters（ckpt-40 在 stage2 内）→ 续训到 60
    cfg_br = cfg_for("v3brk")
    p1, _, _log = run_trainer(cfg_br, extra=["--max-iters", str(CKPT_ITER)])
    assert p1.returncode == 0, f"v3brk leg1 失败:\n{p1.stderr[-2000:]}"
    p2, mem_br, _log = run_trainer(cfg_br, extra=["--max-iters", str(ITERS)],
                                 env_extra={"RESUME_ITER": str(CKPT_ITER)})
    assert p2.returncode == 0, f"v3brk 续训失败:\n{p2.stdout[-2000:]}\n{p2.stderr[-2000:]}"

    return {
        "mem": max(mem_ref, mem_br),
        "losses_ref": read_metrics(WORK / "runs" / "v3ref", "v3ref"),
        "losses_br": read_metrics(WORK / "runs" / "v3brk", "v3brk"),
        "manifest_ref": read_manifest(WORK / "runs" / "v3ref", "v3ref"),
    }


class TestV3Curriculum:
    def test_stage_order_and_boundary(self, world, vreport):
        streams = _open_streams()
        s1, s2 = streams
        assert s1.samples == S1_SAMPLES and s2.samples == S2_SAMPLES
        # 全局序号 0..39 全 1，40.. 全 2（抽查 + 边界两侧）
        probes = [0, 20, 39, 40, 41, 100, 199]
        vals = {}
        from yuformer_pretrain.data import concat_streams
        total, lut = concat_streams(streams)
        for g in probes:
            stage, local = (int(lut[g, 0]), int(lut[g, 1]))
            arr = streams[stage].sample(local)
            vals[g] = (stage, int(np.unique(arr)[0]))
        for g in (0, 20, 39):
            assert vals[g] == (0, 1), f"全局样本 {g} 应在 stage1 且全 1，实际 {vals[g]}"
        for g in (40, 41, 100, 199):
            assert vals[g] == (1, 2), f"全局样本 {g} 应在 stage2 且全 2，实际 {vals[g]}"
        vreport["checks"]["stage_order"] = {"probes": vals, "result": "PASS"}

    def test_manifest_records_switch(self, world, vreport):
        mf = world["manifest_ref"]
        assert mf["data_stage_samples"] == [S1_SAMPLES, S2_SAMPLES]
        assert mf["data_stage_switch_at_samples"] == [S1_SAMPLES]
        assert len(mf["data_sha256"]) == 2 and len(mf["data_source"]) == 2
        vreport["checks"]["manifest_switch"] = {
            "stage_samples": mf["data_stage_samples"],
            "switch_at": mf["data_stage_switch_at_samples"],
            "result": "PASS",
        }

    def test_token_checksum_conservation(self, world, vreport):
        """全程消费 120 样本：token 总和守恒；常量流下"无重复无遗漏"的等价判据是
        序列只含 2 个相异样本指纹，且在 sample 40 处恰好切换一次。"""
        from yuformer_pretrain.data import DataCursor, PrefetchLoader, concat_streams

        streams = _open_streams()
        total, _ = concat_streams(streams)
        cursor = DataCursor(total, epochs=1)
        seen = []
        with PrefetchLoader(streams, cursor, num_workers=2) as src:
            for _ in range(ITERS * SPR):
                seen.append(src.next_sample())
        assert cursor.consumed == ITERS * SPR
        checksums = [hashlib.sha1(a.tobytes()).hexdigest() for a in seen]
        distinct = sorted(set(checksums))
        assert len(distinct) == 2, f"常量流应恰有 2 个相异指纹，实际 {len(distinct)}"
        first_switch = next(i for i, c in enumerate(checksums) if c == distinct[1])
        assert first_switch == S1_SAMPLES, f"段切换点 {first_switch} != {S1_SAMPLES}"
        assert all(c == distinct[0] for c in checksums[:S1_SAMPLES])
        assert all(c == distinct[1] for c in checksums[S1_SAMPLES:])
        token_sum = int(sum(int(a.sum()) for a in seen))
        expect = 1 * S1_SAMPLES * SPAN + 2 * (ITERS * SPR - S1_SAMPLES) * SPAN
        assert token_sum == expect, f"token 总和不守恒: {token_sum} != {expect}"
        vreport["checks"]["conservation"] = {
            "samples": len(seen), "distinct": len(distinct),
            "switch_at": first_switch, "token_sum": token_sum, "expected": expect,
            "result": "PASS",
        }

    def test_midstage2_resume(self, world, vreport):
        lr_, lb = world["losses_ref"], world["losses_br"]
        common = sorted(s for s in set(lr_) & set(lb)
                        if s > CKPT_ITER and "loss" in lr_[s] and "loss" in lb[s])
        assert len(common) >= 2, f"公共 log 点不足: {common}"
        diffs = {s: abs(lr_[s]["loss"] - lb[s]["loss"]) for s in common}
        worst = max(diffs, key=diffs.get)
        assert diffs[worst] < LOSS_TOL, f"iter {worst} loss 差 {diffs[worst]}"
        ck = WORK / "runs" / "v3brk" / "checkpoints"
        cur40 = json.loads((ck / f"ckpt-{CKPT_ITER:08d}" / "cursor.json").read_text())
        assert cur40["cursor"]["consumed_samples"] == CKPT_ITER * SPR
        assert S1_SAMPLES < CKPT_ITER * SPR < S1_SAMPLES + S2_SAMPLES  # 断点确在 stage2 内
        vreport["checks"]["midstage2_resume"] = {
            "ckpt_consumed": cur40["cursor"]["consumed_samples"],
            "stage1_end": S1_SAMPLES,
            "steps_compared": len(common),
            "max_abs_diff": diffs[worst], "worst_step": worst,
            "bit_identical": all(d == 0.0 for d in diffs.values()),
            "tolerance": LOSS_TOL, "result": "PASS",
        }

    def test_vram_budget(self, world, vreport):
        assert world["mem"] < 3200
        vreport["checks"]["vram"] = {"peak_delta_mb": world["mem"], "budget_mb": 3200,
                                     "result": "PASS"}
