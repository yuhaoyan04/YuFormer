#!/usr/bin/env bash
# Build the exact TorchTitan v0.3.0 CUDA 12.6 environment after the host driver gate passes.
set -euo pipefail

ENV_DIR="${YUFORMER_TORCHTITAN_ENV:-/opt/yuformer-torchtitan}"
SOURCE_DIR="${YUFORMER_TORCHTITAN_ROOT:-/opt/torchtitan-v0.3.0}"
EXPECTED_DRIVER="525.60.13"

version_ge() {
  [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1)" = "$2" ]
}

actual_driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1p' | tr -d '[:space:]')"
if ! version_ge "$actual_driver" "$EXPECTED_DRIVER"; then
  echo "YuFormer cu126 requires NVIDIA driver >=${EXPECTED_DRIVER}; found ${actual_driver}." >&2
  echo "Use a CUDA 12 compatible host or upgrade the driver before bootstrapping." >&2
  exit 2
fi

bash "$(dirname "$0")/prepare_env.sh"

YUFORMER_TORCHTITAN_ROOT="$SOURCE_DIR" \
  "$ENV_DIR/bin/python" "$(dirname "$0")/preflight.py"
