"""全链健康监控（轻量版）：日志尾解析进度 + summary mtime 窗口检测 error 增量。

IO 饱和环境下数 summary 文件不可行（glob 全树分钟级）——进度主体取自 driver
progress 日志（append-only 单文件，读尾 O(1)）；error 增量用 mtime > 窗口起点
的 summary 计数（stat only）。--deep 时才做全量计数核对。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import time
from pathlib import Path

PROGRESS_RE = re.compile(r"progress\[(\w+)\]: (\d+)/(\d+) shards?, ([\d.]+) shards/s")
ETA_RE = re.compile(r"(\d+) remaining")

LOGS = {
    "decontam": "../data/decontam3.log",
    "ndv2": "../data/ndv2_full.log",
    "s4": "../data/s4_full.log",
    "s5": "../data/s5_full.log",
    "s6": "../data/s6_full.log",
}
PROC_PATTERNS = {
    "decontam_driver": "scale[.]driver.*stage decontam",
    "ndv2_driver": "scale[.]driver.*stage ndv2",
    "s4_driver": "scale[.]driver.*stage s4",
    "scorer_train": "scorer_trai[n]",
}
ERROR_DIRS = {
    "decontam": "../data/processed/decontam",
    "s5": "../data/processed/s5",
    "s4_scored": "../data/processed/s4_scored",
}


def tail_lines(path: str, nbytes: int = 8192) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - nbytes))
            return f.read().decode(errors="replace").splitlines()
    except OSError:
        return []


def parse_progress(log: str) -> dict | None:
    best = None
    for line in tail_lines(log):
        m = PROGRESS_RE.search(line)
        if m:
            best = {
                "done": int(m.group(2)),
                "total": int(m.group(3)),
                "shards_per_s": float(m.group(4)),
                "line": line.strip()[:150],
            }
    if best:
        remain = best["total"] - best["done"]
        if best["shards_per_s"] > 0:
            best["eta_h"] = round(remain / best["shards_per_s"] / 3600, 1)
        best["pct"] = round(best["done"] / max(best["total"], 1) * 100, 1)
    return best


def mtime_window_errors(directory: str, since_ts: float) -> tuple[int, int]:
    """窗口内 mtime 变化的 summary 数（touch = 重写：成功覆盖或新 error）。"""
    touched = 0
    err_like = 0
    for p in glob.glob(f"{directory}/*/**/*.summary.json", recursive=True):
        try:
            if os.path.getmtime(p) < since_ts:
                continue
        except OSError:
            continue
        touched += 1
        try:
            with open(p, "rb") as f:
                head = f.read(1024)
            if b"shard_error" in head or b'"error"' in head:
                err_like += 1
        except OSError:
            pass
    return touched, err_like


def proc_alive(pattern: str) -> int:
    try:
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=10)
        return len([l for l in out.stdout.splitlines() if l.strip()])
    except Exception:
        return -1


def resources() -> dict:
    out: dict = {}
    try:
        out["load1"] = round(os.getloadavg()[0], 1)
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable"):
                    out["mem_avail_gb"] = round(int(line.split()[1]) / 1e6, 1)
    except OSError:
        pass
    try:
        st = os.statvfs("..")
        out["disk_free_gb"] = round(st.f_bavail * st.f_frsize / 1e9, 1)
    except OSError:
        pass
    return out


def snapshot(health_dir: Path) -> dict:
    ts = time.time()
    snap: dict = {"ts": ts, "time_str": time.strftime("%F %H:%M:%S"), "progress": {}, "procs": {}, "errors_window": {}, "resources": resources(), "flags": []}
    for name, log in LOGS.items():
        p = parse_progress(log)
        if p:
            snap["progress"][name] = p
    for name, pat in PROC_PATTERNS.items():
        snap["procs"][name] = proc_alive(pat)
    state_path = health_dir / "monitor_state.json"
    prev = None
    if state_path.is_file():
        try:
            prev = json.loads(state_path.read_text())
        except Exception:
            prev = None
    if prev:
        dt_min = (ts - prev["ts"]) / 60
        for name, cur in snap["progress"].items():
            p = (prev.get("progress") or {}).get(name)
            if p and dt_min >= 30 and cur["done"] == p["done"] and cur["pct"] < 99.9:
                snap["flags"].append(f"STALL: {name} {dt_min:.0f}min 无进度（{cur['done']}/{cur['total']}，E-1 规则核对停滞守卫）")
        for name in ("decontam", "ndv2"):
            procs = snap["procs"].get(f"{name}_driver", 0)
            if procs == 0 and name == "ndv2" and (snap["progress"].get(name) or {}).get("pct", 100) < 99:
                snap["flags"].append(f"DEAD?: ndv2 driver 进程 0 个但任务未完成")
    health_dir.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"ts": ts, "progress": snap["progress"]}, sort_keys=True))
    (health_dir / "latest.json").write_text(json.dumps(snap, ensure_ascii=False, sort_keys=True, indent=1))
    with open(health_dir / "health_history.jsonl", "a") as f:
        f.write(json.dumps({"ts": ts, "progress": snap["progress"], "procs": snap["procs"], "flags": snap["flags"]}, sort_keys=True) + "\n")
    return snap


def render(snap: dict) -> str:
    lines = [f"=== health @ {snap['time_str']} ==="]
    for name, p in snap["progress"].items():
        lines.append(f"  {name:10s} {p['done']:>6}/{p['total']:<6} {p['pct']:>5}%  {p['shards_per_s']:.2f}/s  eta={p.get('eta_h', '-'):>5}h")
    lines.append(f"  procs: {json.dumps(snap['procs'])}")
    r = snap["resources"]
    lines.append(f"  res: load={r.get('load1')} mem={r.get('mem_avail_gb')}GB disk={r.get('disk_free_gb')}GB")
    if snap["flags"]:
        lines.append("  FLAGS:")
        for fl in snap["flags"]:
            lines.append(f"    ! {fl}")
    else:
        lines.append("  flags: (none)")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--health-dir", default="../data/health")
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args(argv)
    if args.once:
        print(render(snapshot(Path(args.health_dir))))
        return 0
    while True:
        try:
            print(render(snapshot(Path(args.health_dir))), flush=True)
        except Exception as e:
            print(f"monitor error: {type(e).__name__}: {e}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
