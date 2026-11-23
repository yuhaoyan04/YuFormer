#!/usr/bin/env bash
# 全量验证套件（串行；每项可单独跑：python -m pytest pretrain/tests/verify/test_v1_resume.py -q）
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"
export YUFORMER_RUN_GPU_VERIFY=1
export YUFORMER_VERIFY_GPU="${YUFORMER_VERIFY_GPU:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-${YUFORMER_VERIFY_GPU}}"
export YUFORMER_VERIFY_RESET=1
export YUFORMER_VERIFY_SCOPE="V1-V5"
exec /usr/bin/python3 -m pytest pretrain/tests/verify/test_v1_resume.py pretrain/tests/verify/test_v2_lr_schedule.py pretrain/tests/verify/test_v3_curriculum.py pretrain/tests/verify/test_v4_length_extrapolation.py pretrain/tests/verify/test_v5_muon.py -q "$@"
