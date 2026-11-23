"""GPU 空窗检测：等待外部容器任务结束的真空闲窗口，就绪即落盘标志。

判定（防误触发）：单卡空闲 ≥60GB 且连续 3 个采样周期（20min）→ 该卡 READY；
全部 4 卡 READY → 写 data/health/gpu_window_ready.json（实验启动的可信信号）。
外部负载是动态轮转的（新组不断起）——单次空闲可能是批间隙，必须持续窗口。
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

READY_GB = 60
HOLD_CYCLES = 3
INTERVAL_SEC = 600
OUT = Path("../data/health")


def free_gb_by_gpu() -> list[float]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30,
        ).stdout
        res = []
        for line in out.strip().splitlines():
            idx, used, total = [int(x) for x in line.split(",")]
            res.append((total - used) / 1024.0)
        return res
    except Exception:
        return []


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    streak = [0, 0, 0, 0]
    while True:
        free = free_gb_by_gpu()
        ts = time.strftime("%F %H:%M:%S")
        if len(free) == 4:
            ready_now = [f >= READY_GB for f in free]
            streak = [s + 1 if r else 0 for s, r in zip(streak, ready_now)]
            all_ready = all(s >= HOLD_CYCLES for s in streak)
            line = f"{ts} free={[f'{f:.0f}G' for f in free]} streak={streak} all_ready={all_ready}"
            print(line, flush=True)
            with open(OUT / "gpu_window.jsonl", "a") as fh:
                fh.write(json.dumps({"ts": ts, "free_gb": [round(f, 1) for f in free], "streak": streak, "all_ready": all_ready}) + "\n")
            if all_ready:
                (OUT / "gpu_window_ready.json").write_text(
                    json.dumps({"ts": ts, "free_gb": [round(f, 1) for f in free]}, sort_keys=True)
                )
                print("GPU WINDOW READY — 可开跑 E0-Q2/E1/E2", flush=True)
        else:
            print(f"{ts} nvidia-smi unreadable", flush=True)
        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main()
