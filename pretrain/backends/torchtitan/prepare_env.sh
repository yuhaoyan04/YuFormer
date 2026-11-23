#!/usr/bin/env bash
# Prepare the pinned TorchTitan userspace without requiring a usable GPU driver.
set -eEuo pipefail

# Python 3.12 removed stdlib distutils. Some host images still export this
# legacy override globally, which makes setuptools.build_meta unimportable.
unset SETUPTOOLS_USE_DISTUTILS
# The host pip configuration adds NVIDIA's package index to every resolution.
# Torch packages use an explicit PyTorch index below; all remaining packages
# come from PyPI.  Ignoring the host config keeps retries against an unrelated
# flaky extra index from turning a recoverable install into an hours-long job.
export PIP_CONFIG_FILE=/dev/null
export PIP_DISABLE_PIP_VERSION_CHECK=1

ENV_DIR="${YUFORMER_TORCHTITAN_ENV:-/opt/yuformer-torchtitan}"
SOURCE_DIR="${YUFORMER_TORCHTITAN_ROOT:-/opt/torchtitan-v0.3.0}"
MANAGED_PYTHON="/opt/yuformer-python/cpython-3.12-linux-x86_64-gnu/bin/python3.12"
PYTHON_BIN="${YUFORMER_PYTHON312:-${MANAGED_PYTHON}}"
EXPECTED_COMMIT="086bf6c166ec85c1298eb5596fa9bf95f6a2d840"
REPOSITORY="https://github.com/pytorch/torchtitan.git"
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
STATE_PATH="${YUFORMER_TORCHTITAN_PREPARE_STATE:-${REPO_ROOT}/data/health/torchtitan-env-prepare.json}"
FREEZE_PATH="${YUFORMER_TORCHTITAN_FREEZE_PATH:-${REPO_ROOT}/data/health/torchtitan-env-freeze.txt}"
USERSPACE_REPORT="${YUFORMER_TORCHTITAN_USERSPACE_REPORT:-${REPO_ROOT}/data/health/torchtitan-userspace.json}"
BUILD_DIR="${ENV_DIR}.building"
TARGET_DIR="${BUILD_DIR}"
COMPLETED=0

write_state() {
  local status="$1"
  local detail="$2"
  "$PYTHON_BIN" - "$STATE_PATH" "$status" "$detail" "$ENV_DIR" "$SOURCE_DIR" "$$" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "status": sys.argv[2],
    "detail": sys.argv[3],
    "environment": sys.argv[4],
    "source": sys.argv[5],
    "pid": int(sys.argv[6]),
    "updated_at": datetime.now(timezone.utc).isoformat(),
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
finally:
    try:
        os.unlink(temporary)
    except FileNotFoundError:
        pass
PY
}

on_exit() {
  local rc=$?
  if [ "$COMPLETED" -eq 0 ]; then
    write_state "FAIL" "environment preparation exited with rc=${rc}" || true
  fi
}
trap on_exit EXIT

mkdir -p "$(dirname "$STATE_PATH")" "$(dirname "$FREEZE_PATH")" "$(dirname "$ENV_DIR")"
exec 9>"${ENV_DIR}.prepare.lock"
if ! flock -n 9; then
  write_state "RUNNING" "another environment preparation process holds the lock"
  COMPLETED=1
  exit 0
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python 3.12 executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi
actual_python="$($PYTHON_BIN -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')"
if [ "$actual_python" != "3.12" ]; then
  echo "Expected Python 3.12, found ${actual_python}: ${PYTHON_BIN}" >&2
  exit 2
fi

if [ ! -d "$SOURCE_DIR/.git" ]; then
  git clone --filter=blob:none --no-checkout "$REPOSITORY" "$SOURCE_DIR"
  git -C "$SOURCE_DIR" fetch --depth 1 origin "$EXPECTED_COMMIT"
  git -C "$SOURCE_DIR" checkout --detach "$EXPECTED_COMMIT"
fi
actual_commit="$(git -C "$SOURCE_DIR" rev-parse HEAD)"
if [ "$actual_commit" != "$EXPECTED_COMMIT" ]; then
  echo "TorchTitan checkout mismatch: ${actual_commit}" >&2
  exit 2
fi
if [ -n "$(git -C "$SOURCE_DIR" status --porcelain --untracked-files=no)" ]; then
  echo "TorchTitan tracked source is dirty: ${SOURCE_DIR}" >&2
  exit 2
fi

if [ -x "$ENV_DIR/bin/python" ]; then
  TARGET_DIR="$ENV_DIR"
  if [ -f "$FREEZE_PATH" ] && "$ENV_DIR/bin/python" "$(dirname "$0")/validate_userspace.py" \
      --environment "$ENV_DIR" --source "$SOURCE_DIR" --freeze "$FREEZE_PATH" \
      --out "$USERSPACE_REPORT" >/dev/null; then
    write_state "PASS" "userspace already ready; reused without dependency resolution"
    COMPLETED=1
    exit 0
  fi
fi
write_state "RUNNING" "installing pinned userspace dependencies"

if [ ! -x "$TARGET_DIR/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$TARGET_DIR"
fi
"$TARGET_DIR/bin/python" -m pip install --upgrade pip
"$TARGET_DIR/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu126 \
  'torch==2.14.0' 'torchvision==0.29.0' 'torchao==0.18.0' 'triton==3.8.0'
"$TARGET_DIR/bin/python" -m pip install -r "$SOURCE_DIR/requirements.txt"
"$TARGET_DIR/bin/python" -m pip install 'torch-checkpointing==0.1.0'
"$TARGET_DIR/bin/python" -m pip install -e "$SOURCE_DIR" --no-deps --no-build-isolation
"$TARGET_DIR/bin/python" -m pip check

if [ "$TARGET_DIR" = "$BUILD_DIR" ]; then
  if [ -e "$ENV_DIR" ]; then
    echo "Refusing to replace environment created concurrently: ${ENV_DIR}" >&2
    exit 2
  fi
  mv "$BUILD_DIR" "$ENV_DIR"
fi

# venv console and activation scripts contain the absolute temporary prefix.
# Repair it after atomic publication so both direct entry points and sourced
# activation scripts refer to the published environment.
"$PYTHON_BIN" - "$BUILD_DIR" "$ENV_DIR" <<'PY'
import os
import sys
from pathlib import Path

old = os.fsencode(str(Path(sys.argv[1])))
new = os.fsencode(str(Path(sys.argv[2])))
for path in sorted((Path(sys.argv[2]) / "bin").iterdir()):
    if not path.is_file() or path.is_symlink():
        continue
    content = path.read_bytes()
    if old in content:
        temporary = path.with_name(f".{path.name}.shebang-{os.getpid()}")
        temporary.write_bytes(content.replace(old, new))
        temporary.chmod(path.stat().st_mode)
        os.replace(temporary, path)
PY

freeze_tmp="${FREEZE_PATH}.tmp-$$"
"$ENV_DIR/bin/python" -m pip freeze --all > "$freeze_tmp"
mv "$freeze_tmp" "$FREEZE_PATH"
"$ENV_DIR/bin/python" "$(dirname "$0")/validate_userspace.py" \
  --environment "$ENV_DIR" --source "$SOURCE_DIR" --freeze "$FREEZE_PATH" \
  --out "$USERSPACE_REPORT"
write_state "PASS" "userspace ready; GPU/driver preflight remains a separate gate"
COMPLETED=1

