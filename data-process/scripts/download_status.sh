#!/usr/bin/env bash
# YuFormer 预训练数据下载进度查看
#
# 用法: bash scripts/download_status.sh [数据根目录]
#   默认数据根目录: 仓库内 data/pretrain-raw（可用 YUFORMER_DATA_ROOT 覆盖）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${1:-${YUFORMER_DATA_ROOT:-${REPO_ROOT}/data/pretrain-raw}}"
LOG="${REPO_ROOT}/data/download.log"

echo "=== 已下载体量 (${DATA_ROOT}) ==="
du -sh "${DATA_ROOT}" 2>/dev/null || echo "(目录尚不存在)"
echo
echo "=== 逐源体量 ==="
du -sh "${DATA_ROOT}"/*/ 2>/dev/null | sort -rh || true
echo
if [ -f "${DATA_ROOT}/download_report.json" ]; then
  echo "=== 最近完成的下载报告 ==="
  python3 -c "
import json
r = json.load(open('${DATA_ROOT}/download_report.json'))
for s in r['sources']:
    print(f\"  {s['status']:>7}  {s['name']:<22} {s['cumulative_gb']:>10.2f} GB  (dl={s['downloaded']}, skip={s['skipped']}, fail={s['failed']})\")
"
fi
if [ -f "${LOG}" ]; then
  echo
  echo "=== 下载日志末尾 ==="
  tail -5 "${LOG}"
fi
