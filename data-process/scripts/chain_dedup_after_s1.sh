#!/usr/bin/env bash
# S1→S2→S3 自动接力链：S1 全量 driver 一退出，自动续跑精确去重 + MinHash 近重去重
#（2026-09-17 时序锁定：去重完成后才进入 tokenizer 训练与 S4 打分——见 PIPELINE_DESIGN S4）
#
# 用法: bash scripts/chain_dedup_after_s1.sh
# 前提: S1 driver（--stage all）已在运行或已完成；本脚本等其退出后接续，自身幂等。

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${STAGE_DIR}/.." && pwd)"
DATA_ROOT="${REPO_ROOT}/data"
LOG="${DATA_ROOT}/pipeline.log"
PY="/usr/bin/python3"
export PYTHONPATH="${STAGE_DIR}/src"
export YUFORMER_LID_MODEL="${DATA_ROOT}/models/lid.176.ftz"
MARKER="${DATA_ROOT}/.dedup_chain_done"

echo "[chain] $(date '+%F %T') 启动：等待 S1 全量 driver 退出"

while pgrep -f 'scale[.]driver --stage all' > /dev/null; do
  sleep 180
done

if [ -f "${MARKER}" ]; then
  echo "[chain] 检测到完成标记（$(cat "${MARKER}" 2>/dev/null)），退出"
  exit 0
fi

cd "${STAGE_DIR}"
echo "[chain] $(date '+%F %T') S1 已退出，开始 S2+S3 去重全量"
"${PY}" -m yuformer_data.scale.driver --stage dedup \
  --workers 160 --data-root "${DATA_ROOT}/pretrain-raw" \
  --out-root "${DATA_ROOT}/processed" >> "${LOG}" 2>&1
rc=$?

if [ "${rc}" -eq 0 ]; then
  date '+%F %T' > "${MARKER}"
  echo "[chain] $(date '+%F %T') S2+S3 去重完成。后续（tokenizer 训练 + S4 打分）经去重后池子启动。"
else
  echo "[chain] $(date '+%F %T') 去重退出码 ${rc}，未写完成标记，可安全重跑本脚本"
fi
exit "${rc}"
