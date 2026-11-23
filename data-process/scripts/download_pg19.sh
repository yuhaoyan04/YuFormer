#!/usr/bin/env bash
# PG19 书籍语料下载器（GCS 直采版）
#
# deepmind/pg19 的 HF 仓库只含 28,602 个文件名清单，正文托管在
# GCS bucket: gs://deepmind-gutenberg（已验证本机可达，无需翻墙）。
# 本脚本: 清单从 HF 拉取 → 多线程 curl 下载正文 txt → 断点可续。
#
# 用法:
#   bash scripts/download_pg19.sh                     # 下载到 <数据根>/pg19
#   YUFORMER_DATA_ROOT=/path/to/data bash scripts/download_pg19.sh
#   PG19_WORKERS=32 bash scripts/download_pg19.sh     # 调整并发（默认 24）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${STAGE_DIR}/.." && pwd)"
DATA_ROOT="${YUFORMER_DATA_ROOT:-${REPO_ROOT}/data/pretrain-raw}"
OUT_DIR="${DATA_ROOT}/pg19"
WORKERS="${PG19_WORKERS:-24}"
BUCKET="https://storage.googleapis.com/deepmind-gutenberg"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "${OUT_DIR}/lists"
echo "=== PG19 download -> ${OUT_DIR} (workers=${WORKERS}) ==="

# 1. 拉取三个 split 的文件清单（train 28602 / validation 50 / test 100）
for split in train validation test; do
  list_local="${OUT_DIR}/lists/${split}_files.txt"
  if [ ! -s "${list_local}" ]; then
    python3 - "$split" "$list_local" <<'PYEOF'
import sys
from huggingface_hub import hf_hub_download
split, out = sys.argv[1], sys.argv[2]
p = hf_hub_download('deepmind/pg19', repo_type='dataset',
                    filename=f'data/{split}_files.txt')
files = open(p).read().split()
open(out, 'w').write('\n'.join(files) + '\n')
print(f'  list {split}: {len(files)} files')
PYEOF
  fi
done

# 2. 多线程下载正文；已存在且非空的文件跳过（断点续传）
fetch_one() {
  local rel="$1"
  local out="${PG19_OUT}/${rel}"
  if [ -s "${out}" ]; then return 0; fi
  mkdir -p "$(dirname "${out}")"
  curl -sf --retry 3 --retry-delay 2 -o "${out}" "${PG19_BUCKET}/${rel}" \
    || { rm -f "${out}"; echo "FAIL ${rel}" >> "${PG19_OUT}/failed.txt"; }
}
export PG19_OUT="${OUT_DIR}" PG19_BUCKET="${BUCKET}"
export -f fetch_one

for split in train validation test; do
  echo "--- ${split} ---"
  xargs -a "${OUT_DIR}/lists/${split}_files.txt" -d '\n' -n 1 -P "${WORKERS}" \
    bash -c 'fetch_one "$0"'
  n_done=$(find "${OUT_DIR}/${split}" -name '*.txt' -size +0c 2>/dev/null | wc -l)
  n_list=$(wc -l < "${OUT_DIR}/lists/${split}_files.txt")
  echo "  ${split}: ${n_done}/${n_list} files"
done

echo "=== PG19 done: $(du -sh "${OUT_DIR}" | cut -f1) ==="
if [ -s "${OUT_DIR}/failed.txt" ]; then
  echo "注意: $(wc -l < "${OUT_DIR}/failed.txt") 个文件失败，见 ${OUT_DIR}/failed.txt（重跑本脚本即可续传）"
else
  rm -f "${OUT_DIR}/failed.txt"
  echo "全部文件下载成功。"
fi
exit 0
