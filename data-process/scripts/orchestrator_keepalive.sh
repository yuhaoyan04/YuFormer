#!/usr/bin/env bash
# 编排器 keepalive（2026-10-08 重建——hourly_watchdog A3 依赖本脚本）：
#   1) pipeline_orchestrator.py 死亡即拉起（5s 轮询）
#   2) 每小时驱动 hourly_watchdog.py（停滞双判据杀 driver）+ health_monitor.py（体检上报）
# 启动：setsid nohup bash scripts/orchestrator_keepalive.sh >/dev/null 2>&1 &
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # data-process/scripts
DIR="$(cd "$SCRIPT_DIR/.." && pwd)"                          # data-process（工作目录）
DATA="$(cd "$DIR/../data" && pwd)"
cd "$DIR"
last_watchdog=0
echo "[keepalive] $(date '+%F %T') 启动 pid=$$" >> "$DATA/keepalive.log"
while true; do
  if ! pgrep -f "pipeline_orchestrator[.]py --mode fast" >/dev/null; then
    echo "[keepalive] $(date '+%F %T') orchestrator 不在——拉起" >> "$DATA/keepalive.log"
    setsid nohup python3 scripts/pipeline_orchestrator.py --mode fast >> "$DATA/orchestrator_stdout.log" 2>&1 &
  fi
  now=$(date +%s)
  if [ $((now - last_watchdog)) -ge 3600 ]; then
    last_watchdog=$now
    # E-38：health_monitor.py 是常驻循环 daemon（非单次执行）——顺序调用会永久阻塞
    # 本 keepalive，watchdog 从此停摆。监控职责由 hourly_watchdog 承担，不再拉起
    # monitor（其 LOGS 配置指向 s4_full.log 等旧文件，产出的是误导性过期数据）。
    python3 scripts/hourly_watchdog.py >> "$DATA/health/hourly/watchdog_stdout.log" 2>&1 \
      || echo "[keepalive] $(date '+%F %T') watchdog 运行失败" >> "$DATA/keepalive.log"
  fi
  sleep 60
done
