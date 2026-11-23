#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <env file>" >&2
  exit 2
fi
ENV_FILE="$(realpath "$1")"
BACKEND_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${BACKEND_DIR}/../../.." && pwd)"
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a
if [[ -n "${YUFORMER_OVERRIDE_GPUS_PER_NODE:-}" ]]; then
  export GPUS_PER_NODE="${YUFORMER_OVERRIDE_GPUS_PER_NODE}"
fi
: "${YUFORMER_PYTHON:?}"
: "${YUFORMER_TORCHTITAN_ROOT:?}"
: "${YUFORMER_CONFIG:?}"
: "${GPUS_PER_NODE:?}"
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  echo "formal runtime overrides are forbidden; declare changes in a separate YAML contract" >&2
  exit 3
fi
if [[ -z "${YUFORMER_ATTEMPT_ID:-}" ]]; then
  export YUFORMER_ATTEMPT_ID="$("${YUFORMER_PYTHON}" -c 'import secrets; print(secrets.token_hex(16))')"
fi
[[ -x "${YUFORMER_PYTHON}" ]] || { echo "TorchTitan python is not executable: ${YUFORMER_PYTHON}" >&2; exit 3; }
[[ -d "${YUFORMER_TORCHTITAN_ROOT}" ]] || { echo "TorchTitan source root is missing: ${YUFORMER_TORCHTITAN_ROOT}" >&2; exit 3; }

if [[ "${YUFORMER_CONFIG}" != /* ]]; then
  export YUFORMER_CONFIG="$(realpath "${REPO_ROOT}/pretrain/${YUFORMER_CONFIG}")"
fi

export PYTHONPATH="${BACKEND_DIR}:${REPO_ROOT}/model/src:${PYTHONPATH:+:${PYTHONPATH}}"
export YUFORMER_REPO_ROOT="${REPO_ROOT}"
export YUFORMER_TORCHTITAN_ROOT="${YUFORMER_TORCHTITAN_ROOT:?}"
if [[ "${YUFORMER_CONFIG}" == "${REPO_ROOT}/pretrain/configs/pretrain_base.yaml" ]]; then
  "${YUFORMER_PYTHON}" "${REPO_ROOT}/pretrain/scripts/validate_production_contract.py" \
    --root "${REPO_ROOT}" --env "${ENV_FILE}"
fi
if [[ "${YUFORMER_RUN_LEASE_HELD:-0}" != "1" ]]; then
  "${YUFORMER_PYTHON}" "${BACKEND_DIR}/preflight.py" --out "${REPO_ROOT}/data/health/torchtitan-preflight.json"
fi
OUT_DIR="$("${YUFORMER_PYTHON}" "${BACKEND_DIR}/validate_contract.py" \
  --repo "${REPO_ROOT}" --config "${YUFORMER_CONFIG}" --print-out-dir)"
if [[ "${YUFORMER_RUN_LEASE_HELD:-0}" != "1" ]]; then
  exec "${YUFORMER_PYTHON}" "${REPO_ROOT}/pretrain/scripts/run_with_lease.py" \
    --out-dir "${OUT_DIR}" --run-name "torchtitan:$(basename "${OUT_DIR}")" -- \
    bash "$0" "${ENV_FILE}"
fi
"${YUFORMER_PYTHON}" "${REPO_ROOT}/pretrain/scripts/assert_run_lease.py" \
  --out-dir "${OUT_DIR}"
export YUFORMER_CURSOR_AUDIT_DIR="${OUT_DIR}/cursor_audit"
mkdir -p "${OUT_DIR}"
"${YUFORMER_PYTHON}" "${BACKEND_DIR}/validate_contract.py" \
  --repo "${REPO_ROOT}" --config "${YUFORMER_CONFIG}"
export YUFORMER_OUT_DIR="${OUT_DIR}"
"${YUFORMER_PYTHON}" - <<'PY'
import json
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

out = Path(os.environ["YUFORMER_OUT_DIR"])
attempt_id = os.environ["YUFORMER_ATTEMPT_ID"]
if not re.fullmatch(r"[A-Za-z0-9._-]{16,128}", attempt_id):
    raise ValueError("invalid YUFORMER_ATTEMPT_ID")
identity = json.loads((out / "run_identity.json").read_text(encoding="utf-8"))
try:
    previous_attempt = json.loads((out / "current_attempt.json").read_text(encoding="utf-8"))
except (OSError, ValueError):
    previous_attempt = {}
attempts = out / "attempts"
attempts.mkdir(parents=True, exist_ok=True)
started_at = datetime.now(timezone.utc).isoformat()
payload = {
    "schema_version": 1,
    "status": "RUNNING",
    "attempt_id": attempt_id,
    "started_at": started_at,
    "started_at_ns": time.time_ns(),
    "identity_sha256": identity["identity_sha256"],
    "hostname": platform.node(),
    "launcher_pid": os.getppid(),
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "gpus_per_node": int(os.environ["GPUS_PER_NODE"]),
    "parent_attempt_id": previous_attempt.get("attempt_id"),
}
for target in (attempts / f"{attempt_id}.json", out / "current_attempt.json"):
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)

# Preserve previous evidence for audit, but make it impossible for this attempt
# to inherit a terminal marker or TensorBoard final-step metrics.
archive = attempts / f"superseded-before-{attempt_id}"
for artifact in (out / "training_complete.json", out / "result.json"):
    if artifact.is_file():
        archive.mkdir(parents=True, exist_ok=True)
        os.replace(artifact, archive / artifact.name)
for event in sorted(out.glob("**/events.out.tfevents.*")):
    if attempts in event.parents:
        continue
    destination = archive / "events" / event.relative_to(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(event, destination)
PY

cd "${YUFORMER_TORCHTITAN_ROOT}"
set +e
"${YUFORMER_PYTHON}" -m torch.distributed.run \
  --nproc_per_node "${GPUS_PER_NODE}" \
  --nnodes "${NUM_NODES:-1}" \
  --node_rank "${NODE_RANK:-0}" \
  --master_addr "${MASTER_ADDR:-127.0.0.1}" \
  --master_port "${MASTER_PORT:-29517}" \
  -m torchtitan.train --module yuformer_ext --config yuformer_from_yaml
train_rc=$?
set -e
export YUFORMER_TRAIN_RC="${train_rc}"
export YUFORMER_FINISHED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
"${YUFORMER_PYTHON}" "${REPO_ROOT}/pretrain/scripts/assert_run_lease.py" \
  --out-dir "${OUT_DIR}"
"${YUFORMER_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

out = Path(os.environ["YUFORMER_OUT_DIR"])
identity = json.loads((out / "run_identity.json").read_text(encoding="utf-8"))
rc = int(os.environ["YUFORMER_TRAIN_RC"])
payload = {
    "schema_version": 1,
    "status": "PASS" if rc == 0 else "FAIL",
    "returncode": rc,
    "finished_at": os.environ["YUFORMER_FINISHED_AT"],
    "run_name": identity["run_name"],
    "identity_sha256": identity["identity_sha256"],
    "attempt_id": os.environ["YUFORMER_ATTEMPT_ID"],
}
target = out / "training_complete.json"
temporary = out / f".training_complete.tmp-{os.getpid()}"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary, target)
attempt_target = out / "attempts" / f"{payload['attempt_id']}.json"
try:
    attempt_payload = json.loads(attempt_target.read_text(encoding="utf-8"))
except (OSError, ValueError):
    attempt_payload = {}
attempt_payload.update(payload)
for attempt_output in (attempt_target, out / "current_attempt.json"):
    attempt_temporary = attempt_output.with_name(
        f".{attempt_output.name}.tmp-{os.getpid()}"
    )
    attempt_temporary.write_text(
        json.dumps(attempt_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(attempt_temporary, attempt_output)
PY
exit "${train_rc}"
