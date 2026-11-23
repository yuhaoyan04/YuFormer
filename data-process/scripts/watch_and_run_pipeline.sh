#!/usr/bin/env bash
# 下载完成看门狗：主下载进程退出 → 校验审计报告 → 自动启动数据处理管线
#
# 用法: bash scripts/watch_and_run_pipeline.sh
# 环境变量:
#   YUFORMER_PYTHON   运行时 python（默认 /usr/bin/python3）
#   PIPELINE_STAGE    自动运行的阶段（默认 all = s0+s1，后续 S2+ 就绪后扩链）

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${STAGE_DIR}/.." && pwd)"
DATA_ROOT="${REPO_ROOT}/data"
RAW_DIR="${DATA_ROOT}/pretrain-raw"
REPORT="${RAW_DIR}/download_report.json"
MARKER="${DATA_ROOT}/.pipeline_auto_started"
LOG="${DATA_ROOT}/pipeline.log"
PY="${YUFORMER_PYTHON:-/usr/bin/python3}"
STAGE="${PIPELINE_STAGE:-all}"
CHECK_INTERVAL=120
# fasttext 语言检测模型（S1 用；缺失则 S1 自动降级 lang='unk' 并在 summary 记录）
export YUFORMER_LID_MODEL="${YUFORMER_LID_MODEL:-${DATA_ROOT}/models/lid.176.ftz}"

echo "[watchdog] $(date '+%F %T') 启动 | 监视下载进程与 ${REPORT}"

while true; do
  # 1. 下载进程还在跑 → 继续等
  if pgrep -f 'download_pretrain_data.py --all' > /dev/null; then
    sleep "${CHECK_INTERVAL}"
    continue
  fi

  # 2. 进程已退出但报告不存在 → 异常终止（人为 kill/崩溃），报告后退出
  if [ ! -f "${REPORT}" ]; then
    echo "[watchdog] $(date '+%F %T') 下载进程已退出但无审计报告——判定为异常终止，不自动启动管线。请人工检查 download.log 后手动启动。"
    exit 1
  fi

  # 3. 校验审计报告
  failed=$(python3 -c "
import json
r = json.load(open('${REPORT}'))
fails = [s['name'] for s in r['sources'] if s['status'] == 'failed']
print(' '.join(fails))
")
  n_ok=$(python3 -c "
import json
r = json.load(open('${REPORT}'))
print(sum(1 for s in r['sources'] if s['status'] in ('ok', 'partial')))
")
  echo "[watchdog] $(date '+%F %T') 下载完成：${n_ok} 源 ok/partial${failed:+，失败源: ${failed}}（失败源不阻塞管线，其数据缺失会在 S0 自检时暴露）"

  # 4. 防重复触发
  if [ -f "${MARKER}" ]; then
    echo "[watchdog] 检测到已启动标记 $(cat "${MARKER}" 2>/dev/null)，退出。"
    exit 0
  fi

  # 5. 等待管线代码就绪（coder 可能仍在实现中），最多等 6 小时
  export PYTHONPATH="${STAGE_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"
  for i in $(seq 1 72); do
    if "${PY}" -c "import yuformer_data.scale.driver" > /dev/null 2>&1; then
      break
    fi
    if [ "${i}" -eq 72 ]; then
      echo "[watchdog] 管线代码 6 小时未就绪，放弃。请检查实现进度后手动启动。"
      exit 1
    fi
    sleep 300
  done

  # 6. 启动管线
  date '+%F %T' > "${MARKER}"
  echo "[watchdog] $(date '+%F %T') 自动启动数据处理管线: stage=${STAGE}"
  cd "${STAGE_DIR}"
  "${PY}" -m yuformer_data.scale.driver \
    --stage "${STAGE}" \
    --workers "${YUFORMER_PIPELINE_WORKERS:-160}" \
    --data-root "${RAW_DIR}" \
    --out-root "${DATA_ROOT}/processed" \
    >> "${LOG}" 2>&1
  rc=$?
  echo "[watchdog] $(date '+%F %T') 管线退出码 ${rc}，日志: ${LOG}"
  exit "${rc}"
done
