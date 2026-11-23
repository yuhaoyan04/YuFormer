#!/usr/bin/env bash
# YuFormer 预训练数据一键下载（hf-mirror 镜像版）
#
# 用法:
#   # 全量下载（按 configs/data_sources.yaml 配额，断点续传）
#   bash scripts/download_pretrain_data.sh
#
#   # 仅列出文件与大小，不下载
#   bash scripts/download_pretrain_data.sh --dry-run
#
#   # 每源只下载 1 个文件（验证可访问性）
#   bash scripts/download_pretrain_data.sh --test
#
#   # 只下载某个源
#   bash scripts/download_pretrain_data.sh --source proof-pile-2
#
#   # 后台 nohup 运行（大任务推荐）
#   nohup bash scripts/download_pretrain_data.sh > ../data/download.log 2>&1 &
#
# 可用环境变量覆盖:
#   YUFORMER_DATA_ROOT   数据根目录（默认: 仓库内 data/pretrain-raw）
#   YUFORMER_DL_WORKERS  并发下载线程数（默认: 12）
#   HF_ENDPOINT          镜像地址（默认: https://hf-mirror.com）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"          # data-process/
REPO_ROOT="$(cd "${STAGE_DIR}/.." && pwd)"           # YuFormer/

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export YUFORMER_DL_WORKERS="${YUFORMER_DL_WORKERS:-12}"
DATA_ROOT="${YUFORMER_DATA_ROOT:-${REPO_ROOT}/data/pretrain-raw}"
CONFIG="${STAGE_DIR}/configs/data_sources.yaml"

mkdir -p "${DATA_ROOT}"

echo "============================================================"
echo "  YuFormer Pretrain Data Download (hf-mirror)"
echo "  endpoint : ${HF_ENDPOINT}"
echo "  data root: ${DATA_ROOT}"
echo "  config   : ${CONFIG}"
echo "  workers  : ${YUFORMER_DL_WORKERS}"
echo "  args     : $*"
echo "============================================================"

# --dry-run / --test 之外的正式下载，打印恢复提示
case " $* " in
  *" --dry-run "*|*" --test "*) ;;
  *) echo "断点续传: 已存在且大小匹配的文件自动跳过；中断后重跑本脚本即可。"
     echo "进度查看: bash ${SCRIPT_DIR}/download_status.sh" ;;
esac

exec python3 "${SCRIPT_DIR}/download_pretrain_data.py" \
  --config "${CONFIG}" \
  --data-root "${DATA_ROOT}" \
  --all "$@"
