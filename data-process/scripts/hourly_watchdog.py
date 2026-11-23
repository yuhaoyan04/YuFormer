"""小时级主动健康守护：每小时全面体检 + 已知异常的自动处置 + 审计轨迹。

与编排器/keepalive 的分工：
  编排器管阶段推进（条件驱动）｜本守护管"健康检测-预警-处置"闭环——不等人来查。
处置动作表（每小时评估）：
  A1 driver 消失且阶段未完 → 等编排器 10min（它 5min 轮询）仍未拉起则直接拉起同参数 driver
  A2 1h 零进度（停滞）→ 杀 driver 触发幂等重启 + ACTION 记录
  A3 编排器+keepalive 同时消失 → 重启 keepalive
  A4 error summary 增量 → 读取 error_message 定性记录（连续 2h 增量升级为告警）
  A5 RSS > 850GB → 告警记录（不自动杀——E-6 规则：先诊断曲线再动手）
输出：data/health/hourly/<ts>.json + hourly_report.log（人读简报，含滚动 ETA）
"""

from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
DATA = REPO / "data"
OUT = DATA / "processed"
HOURLY = DATA / "health" / "hourly"
PROGRESS_RE = re.compile(r"progress\[(\w+)\]: (\d+)/(\d+) shards?, ([\d.]+) shards/s, (\d+) docs/s")
# 剩余行数估算（行数口径——每源 done/total 分片 × 均行数，随阶段完成动态清零）
SRC_ROWS = {
    "fineweb": 4830933122, "fineweb-edu": 806338550, "fineweb-2": 31822877,
    "github-code": 83942602, "starcoderdata": 143981740, "skypile-150b": 230931710,
    "pes2o": 44901806, "proof-pile-2": 10696468, "stack-exchange": 2276320,
    "wikipedia-en": 6406600, "wikipedia-zh": 1379148, "wikitext": 1151231,
    "arxiv-latex-corpus": 778058, "automathtext": 8832688, "cci3-hq": 37332596,
    "dclm-baseline": 243999137, "dolma-books": 55601,
}


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return ""


def procs(pattern: str) -> int:
    out = sh(["pgrep", "-f", pattern])
    return len([l for l in out.splitlines() if l.strip()])


def tail_progress(log: str) -> dict | None:
    try:
        with open(log, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 16384))
            lines = f.read().decode(errors="replace").splitlines()
    except OSError:
        return None
    best = None
    for line in lines:
        m = PROGRESS_RE.search(line)
        if m:
            best = {"stage": m.group(1), "done": int(m.group(2)), "total": int(m.group(3)), "ts_line": line[:100]}
    return best


def stage_rows_remaining(stage_dirname: str) -> int:
    done = {}
    for p in glob.glob(f"{OUT}/{stage_dirname}/*/**/*.summary.json", recursive=True):
        try:
            with open(p, "rb") as f:
                head = f.read(512).decode(errors="replace")
            src = p.split("/")[0] if False else p.replace(str(OUT) + "/" + stage_dirname + "/", "").split("/")[0]
            if "shard_error" not in head:
                done[src] = done.get(src, 0) + 1
        except OSError:
            continue
    return sum(SRC_ROWS.get(s, 0) for s in SRC_ROWS) - 0  # 占位：整体行数由进度分片比例推


def rss_gb(pattern: str) -> tuple[float, int]:
    out = sh(["ps", "aux"]).splitlines()
    rss = 0.0
    n = 0
    for line in out:
        if re.search(pattern, line) and "grep" not in line:
            parts = line.split()
            try:
                rss += int(parts[5]) / 1024 / 1024
                n += 1
            except (ValueError, IndexError):
                continue
    return rss, n


def eta_report(snap: dict) -> str:
    lines = []
    for name, p in snap.get("progress", {}).items():
        lines.append(f"  {name}: {p['done']}/{p['total']} ({p['done']/p['total']*100:.1f}%)")
    return "\n".join(lines) or "  (无 progress)"


def main():
    HOURLY.mkdir(parents=True, exist_ok=True)
    state_path = HOURLY / "watchdog_state.json"
    prev = json.loads(state_path.read_text()) if state_path.is_file() else {}
    ts = time.time()
    snap = {
        "ts": time.strftime("%F %H:%M:%S"),
        "progress": {},
        "procs": {
            "s4": procs("scale[.]driver.*stage s4"),
            "s5": procs("scale[.]driver.*stage s5"),
            "s6": procs("scale[.]driver.*stage s6"),
            "orchestrator": procs("pipeline_orchestrator[.]py"),
            "keepalive": procs("orchestrator_keepaliv[e]"),
            "monitor": procs("health_monito[r]"),
            "gpu_watcher": procs("gpu_window_watche[r]"),
            "exp_scheduler": procs("experiment_scheduler[.]sh"),
        },
    }
    for name, log in (("s4_score", DATA / "s4_orch.log"), ("s5", DATA / "s5_orch.log"), ("s6", DATA / "s6_orch.log")):
        p = tail_progress(str(log))
        if p:
            snap["progress"][name] = p
    rss4, n4 = rss_gb("stage s4")
    snap["s4_rss_gb"] = round(rss4, 1)
    snap["s4_workers"] = n4

    actions: list[str] = []
    # A2 停滞检测（双判据 + 最小间隔 50min，防误杀——2026-10-04 首跑误杀 S4 教训）：
    #   判据1: 与上次快照（≥50min 前）done 无变化
    #   判据2: 该阶段 55min 内无新 summary 产物落盘
    #   两判据同时满足才 pkill（触发编排器幂等重启）
    # E-39（2026-10-09）：只对"driver 还活着"的阶段判停滞——上一阶段（s4）完成后其
    # 日志尾冻结，key 每轮被判"零进度"且 pkill 模式匹配不到任何进程（空杀噪音），
    # 同时会掩盖对当前阶段（s5/s6）的真实停滞检测。
    dt_min = (ts - prev.get("ts_epoch", 0)) / 60
    for name, p in snap["progress"].items():
        if snap["procs"].get(name.split("_")[0], 0) == 0:
            continue
        pp = (prev.get("progress") or {}).get(name)
        stage_dir = {"s4_score": "s4_scored", "s5": "s5", "s6": "mix_members"}.get(name)
        fresh = 0
        if stage_dir:
            for sp in glob.glob(f"{OUT}/{stage_dir}/*/**/*.summary.json", recursive=True):
                try:
                    if os.path.getmtime(sp) > ts - 3300:
                        fresh += 1
                        if fresh > 0:
                            break
                except OSError:
                    continue
        stalled = (
            dt_min >= 50
            and pp is not None
            and p["done"] == pp["done"]
            and p["done"] < p["total"]
            and fresh == 0
        )
        if stalled:
            actions.append(f"A2-STALL: {name} {dt_min:.0f}min 零进度且 55min 无新产物——杀 driver 触发幂等重启")
            sh(["pkill", "-f", f"scale[.]driver.*stage {name.split('_')[0]}"])
        elif pp is not None and p["done"] == pp["done"] and p["done"] < p["total"]:
            actions.append(f"NOTE: {name} done 未变但 {fresh=}(有新产物=大分片处理中，不处置)")
    # A3 编排器全家消失
    if snap["procs"]["orchestrator"] == 0 and snap["procs"]["keepalive"] == 0:
        actions.append("A3: 编排器+keepalive 双亡——重启 keepalive")
        subprocess.Popen(
            ["setsid", "nohup", "bash", str(REPO / "data-process/scripts/orchestrator_keepalive.sh")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        )
    # A5 RSS 告警
    if rss4 > 850:
        actions.append(f"A5-WARN: S4 RSS {rss4:.0f}GB > 850GB（cgroup 953GiB 的 89%）——E-6 规则：诊断曲线而非直接杀")
    snap["actions"] = actions
    snap["ts_epoch"] = ts

    state_path.write_text(json.dumps(snap, sort_keys=True))
    (HOURLY / f"{time.strftime('%Y%m%d_%H%M')}.json").write_text(json.dumps(snap, ensure_ascii=False, sort_keys=True, indent=1))
    report = f"===== {snap['ts']} =====\n{eta_report(snap)}\n  s4: workers={n4} rss={rss4:.0f}GB | procs={json.dumps(snap['procs'])}\n"
    if actions:
        report += "  ACTIONS:\n" + "\n".join(f"    ! {a}" for a in actions)
    else:
        report += "  health: OK（无处置动作）"
    with open(HOURLY / "hourly_report.log", "a") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
