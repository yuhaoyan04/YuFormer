#!/usr/bin/env bash
# pretrain launcher（宪法 1/2：env 文件 = 实验契约，本脚本只做机械翻译 env→CLI）。
#
# 用法: scripts/launch.sh configs/pretrain_base.env
# 必需变量（env 文件内声明）:
#   YUFORMER_PYTHON      python 可执行文件
#   YUFORMER_CONFIG      训练配置 yaml（自包含）
#   GPUS_PER_NODE        单机卡数（= parallel.world_size）
#   MASTER_PORT          分布式端口
# 可选:
#   RESUME_ITER          断点续训身份守卫（必须与 latest.json iter 相等）
#   EXTRA_ARGS           附加 CLI（如 --max-iters，smoke 用）
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <env file>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PRETRAIN_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${PRETRAIN_ROOT}/.." && pwd)"
export PYTHONPATH="${PRETRAIN_ROOT}/src:${REPO_ROOT}/model/src:${PYTHONPATH:+:${PYTHONPATH}}"
export YUFORMER_TRACKER_ROOT="${REPO_ROOT}/experiments/_common"

ENV_FILE="$(realpath "$1")"
# E-64：set -a 使 source 的变量全部导出——trainer 子进程需要 YUFORMER_TOKEN_STREAMS
# 等展开 yaml 占位符；此前无导出，所有臂启动即 ConfigError（live-fire 抓到）。
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

: "${YUFORMER_PYTHON:?}"
: "${YUFORMER_CONFIG:?}"
: "${GPUS_PER_NODE:?}"
: "${MASTER_PORT:?}"

cd "${PRETRAIN_ROOT}"
[[ -f "${YUFORMER_CONFIG}" ]] || { echo "config not found: ${YUFORMER_CONFIG} (cwd=$(pwd))" >&2; exit 3; }

CONFIG_BACKEND="$(python3 -c \
  'import sys, yaml; print((yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {}).get("backend", "pytorch-fsdp1-compat"))' \
  "${YUFORMER_CONFIG}")"
if [[ "${CONFIG_BACKEND}" != "pytorch-fsdp1-compat" ]]; then
  echo "config backend=${CONFIG_BACKEND} is not supported by scripts/launch.sh; run the pinned TorchTitan path after its preflight and adapter gates pass" >&2
  exit 4
fi

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export NODE_RANK="${NODE_RANK:-0}"
export NUM_NODES="${NUM_NODES:-1}"
export PYTHONUNBUFFERED=1
# E-49：expandable_segments 回收碎片显存——0.5B/mbs8+selective 组合在共享卡（邻居占
# 13GB）上靠它从 OOM 变为可跑（65.6GB 中 5.7GB reserved-碎片被利用）。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

exec "${YUFORMER_PYTHON}" -m torch.distributed.run \
  --nproc_per_node "${GPUS_PER_NODE}" \
  --nnodes "${NUM_NODES}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  -m yuformer_pretrain.trainer \
  --config "${YUFORMER_CONFIG}" \
  ${EXTRA_ARGS:-}
