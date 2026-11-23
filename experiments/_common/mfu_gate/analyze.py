#!/usr/bin/env python3
"""MFU 门禁稳态分析：读 trainer 的 metrics.jsonl，出稳态 step_time / TFLOPS / MFU /
邻居基线统计（E-50 口径）。用法：
  python3 analyze.py /tmp/mfu_gate_20261014/results/metrics.jsonl \
      --flops-per-token 4045800000 --tokens-per-step 786432 [--gpu-self 7] [--skip-log-points 1]
flops_per_token 用 model 精确算（trainer 同公式：6*nonemb + 12*L*S*H）：
  PYTHONPATH=model/src python3 -c "from yuformer_model import *; ..."
"""
import argparse
import json
import sys

H800_BF16_PEAK = 989e12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics")
    ap.add_argument("--flops-per-token", type=float, required=True)
    ap.add_argument("--tokens-per-step", type=int, required=True)
    ap.add_argument("--gpu-self", type=int, default=7)
    ap.add_argument("--skip-log-points", type=int, default=1,
                    help="丢弃前 N 个 log 点（compile/热身），默认 1（仅 compile 点）")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.metrics) if l.strip()]
    pts = [(r["step"], r["step_time"]) for r in rows if "step_time" in r]
    pts.sort()
    print(f"metrics 点总数={len(pts)}  step范围={pts[0][0]}..{pts[-1][0]}")
    print(f"compile/首点: step={pts[0][0]} t={pts[0][1]:.2f}s（剔除）")
    steady = pts[args.skip_log_points:]
    ts = [t for _, t in steady]
    n = len(ts)
    mean_t = sum(ts) / n
    mn, mx = min(ts), max(ts)
    wall = sum(ts)
    flops_step = args.flops_per_token * args.tokens_per_step
    achieved = flops_step / mean_t
    print(f"\n稳态窗口: {n} 个 log 点, 覆盖训练 {wall/60:.1f} min")
    print(f"step_time mean={mean_t:.3f}s min={mn:.3f}s max={mx:.3f}s (p-p {(mx-mn)/mean_t*100:.1f}%)")
    print(f"吞吐        ={args.tokens_per_step/mean_t:,.0f} tokens/s")
    print(f" achieved   ={achieved/1e12:.1f} TFLOPS")
    print(f" MFU        ={100*achieved/H800_BF16_PEAK:.2f}%  (gate 18% -> 需 step<={flops_step/(0.18*H800_BF16_PEAK):.2f}s)")

    snap = [r for r in rows if "gpu_util_all" in r]
    if snap:
        others_u, others_m = [], []
        self_u, self_m = [], []
        for r in snap:
            ua, ma = r["gpu_util_all"], r["gpu_mem_mb_all"]
            for i, (u, m) in enumerate(zip(ua, ma)):
                (self_u if i == args.gpu_self else others_u).append(u)
                (self_m if i == args.gpu_self else others_m).append(m)
        nn = len(others_u)
        print(f"\n邻居基线(E-50, 除GPU{args.gpu_self}外全卡, n={nn}快照):")
        print(f" util%  mean={sum(others_u)/nn:.0f} min={min(others_u)} max={max(others_u)}")
        print(f" memMB  mean={sum(others_m)/nn:.0f} min={min(others_m)} max={max(others_m)}")
        ns = len(self_u)
        print(f"本卡(GPU{args.gpu_self}) util mean={sum(self_u)/ns:.0f} memMB mean={sum(self_m)/ns:.0f} max={max(self_m)}")


if __name__ == "__main__":
    main()
