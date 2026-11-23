#!/usr/bin/env python3
"""Persistently audit pretrain readiness and retry safe setup actions."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from user_pretrain_authorization import load_authorization
from qualification_identity import qualification_identity, report_matches  # noqa: E402
from artifact_archive import move_artifact  # noqa: E402
from performance_evidence import promote_bound_loss_analysis
from gpu_resource_policy import allowed_gpu_indices, resource_policy
from research_scope import (group_enabled, scaling_only, load_scope, scope_binding_current,
                            ablation_requirements_satisfied, scope_sha256, POLICY as SCOPE_POLICY)
import research_confirmation
import night_research
import night_batch_search
import batch_pilot_contract as batch_contract
from lr_pilot_contract import extensions as lr_extensions, selection_is_current, EXTENSION_GROUP


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pretrain/src"))
from yuformer_pretrain.lease import RunLease
HEALTH = ROOT / "data/health"
PREFLIGHT = ROOT / "pretrain/backends/torchtitan/preflight.py"
BOOTSTRAP = ROOT / "pretrain/backends/torchtitan/bootstrap_env.sh"
STACK_AUDIT = ROOT / "pretrain/scripts/audit_pretrain_stack.py"
CONTRACT_AUDIT = ROOT / "pretrain/scripts/validate_formal_contracts.py"
RESULT_COLLECTOR = ROOT / "pretrain/scripts/collect_torchtitan_result.py"
LR_SELECTOR = ROOT / "pretrain/scripts/select_lr_pilots.py"
LR_RESOLVER = ROOT / "pretrain/scripts/resolve_experiment_lrs.py"
BATCH_PREPARER = ROOT / "pretrain/scripts/prepare_batch_pilots.py"
BATCH_SELECTOR = ROOT / "pretrain/scripts/select_batch_pilots.py"
E2_ANALYZER = ROOT / "pretrain/scripts/analyze_e2_results.py"
E1_ANALYZER = ROOT / "pretrain/scripts/analyze_e1_scaling.py"
E1_CURVE_RENDERER = ROOT / "pretrain/scripts/render_e1_scaling.py"
E3_ANALYZER = ROOT / "experiments/e3_data_mix/scripts/analyze_results.py"
E3_CONFIG_GENERATOR = ROOT / "experiments/e3_data_mix/scripts/generate_configs.py"
QUALIFICATION_RUNNER = ROOT / "pretrain/scripts/run_torchtitan_qualification.py"
QUALIFICATION_REPORT = HEALTH / "torchtitan-qualification.json"
E3_AUDIT = ROOT / "experiments/e3_data_mix/scripts/audit_matrix.py"
E3_AUDIT_REPORT = ROOT / "experiments/e3_data_mix/results/matrix_audit.json"
E3_ORDER_MATERIALIZER = ROOT / "experiments/e3_data_mix/scripts/materialize_orders.py"
E3_ORDER_MANIFEST = ROOT / "data/experiments/e3_data_mix/orders/manifest.json"
E3_ORDER_PID = HEALTH / "e3-order-materialize.pid"
E3_TOKENIZER = ROOT / "experiments/e3_data_mix/scripts/tokenize_orders.py"
E3_TOKENIZATION_STATE = HEALTH / "e3-tokenization.json"
E3_TOKENIZATION_PID = HEALTH / "e3-tokenization.pid"
E3_STREAM_AUDITOR = ROOT / "experiments/e3_data_mix/scripts/audit_token_streams.py"
E3_STREAM_AUDIT = ROOT / "data/processed/audits/e3_token_streams_20261021.json"
E3_STREAM_AUDIT_PID = HEALTH / "e3-stream-audit.pid"
QUEUE_RUNNER = ROOT / "pretrain/scripts/run_experiment_queue.py"
PILOT_QUEUE_REPORT = HEALTH / "e1_pilot-queue.json"
BATCH_QUEUE_REPORT = HEALTH / "batch_pilot-queue.json"
E2_QUEUE_REPORT = HEALTH / "e2-queue.json"
E1_QUEUE_REPORT = HEALTH / "e1-queue.json"
E3_QUEUE_REPORT = HEALTH / "e3-queue.json"
STREAM_AUDIT = ROOT / "data/processed/audits/formal_token_stream_release_mix_m3_e1_100b_m3_20261020.json"
SOURCE_ROOT = Path("/opt/torchtitan-v0.3.0")
ENV_PYTHON = Path("/opt/yuformer-torchtitan/bin/python")
MANAGED_PYTHON312 = Path("/opt/yuformer-python/cpython-3.12-linux-x86_64-gnu/bin/python3.12")
ACTION_STATE = HEALTH / "pretrain-automation-actions.json"
SHARED_GPU_POLICY = HEALTH / "shared-gpu-scaling-policy.json"
PLAN_PATH = HEALTH / "pretrain-automation-plan.json"
PILOT_ROOT = ROOT / "data/experiments/e1_pilot"
BATCH_ROOT = ROOT / "data/experiments/batch_pilot"
E2_ROOT = ROOT / "data/experiments/e2"
E1_ROOT = ROOT / "data/experiments/e1"
E3_ROOT = ROOT / "data/experiments/e3"
LR_SELECTION = ROOT / "experiments/e1_scaling/results/lr_selection.json"
BATCH_ANCHOR = ROOT / night_research.ANCHOR
LR_RESOLUTION = ROOT / "experiments/e1_scaling/results/lr_resolution.json"
BATCH_PILOT_RESOLUTION = ROOT / "experiments/e1_scaling/results/batch_pilot_resolution.json"
BATCH_SELECTION = ROOT / "experiments/e1_scaling/results/batch_selection.json"
E2_SELECTION = ROOT / "experiments/e2_architecture/results/selection.json"
E1_SCALING_FIT = ROOT / "experiments/e1_scaling/results/scaling_fit.json"
PERFORMANCE_PREPARER = ROOT / "pretrain/scripts/prepare_performance_canaries.py"
PERFORMANCE_CONTRACTS = ROOT / "experiments/e1_scaling/results/performance_contracts.json"
PERFORMANCE_QUEUE_REPORT = HEALTH / "e1_performance-queue.json"
E3_CONFIG_REPORT = ROOT / "experiments/e3_data_mix/results/config_generation.json"
E3_SELECTION = ROOT / "experiments/e3_data_mix/results/selection.json"
E3_CONFIG_PID = HEALTH / "e3-config-generation.pid"
EXPERIMENT_IDENTITY_AUDITOR = ROOT / "pretrain/scripts/audit_experiment_identities.py"
EXPERIMENT_IDENTITY_AUDIT = ROOT / "report/experiment_identity_audit_latest.json"
RECIPE_FREEZER = ROOT / "experiments/e4_curriculum/scripts/freeze_1t_recipe.py"
FROZEN_RECIPE = ROOT / "experiments/e4_curriculum/results/frozen_1t_recipe.json"
RECIPE_FREEZE_PID = HEALTH / "pretrain-recipe-freeze.pid"
PRODUCTION_CANARY_RUNNER = ROOT / "pretrain/scripts/run_production_canary.py"
PRODUCTION_CANARY_REPORT = HEALTH / "production-throughput-canary.json"
PRODUCTION_CANARY_PID = HEALTH / "production-throughput-canary.pid"
PRODUCTION_CANARY_CONFIG = ROOT / "pretrain/configs/production_canary/production_8gpu.yaml"
PRODUCTION_CANARY_ENV = ROOT / "pretrain/configs/production_canary/production_8gpu.env"
FINAL_PLAN_AUDITOR = ROOT / "experiments/e4_curriculum/scripts/audit_1t_plan.py"
FINAL_PLAN_AUDIT = ROOT / "experiments/e4_curriculum/results/e4_1t_plan_static_audit.json"
FINAL_PLAN_AUDIT_PID = HEALTH / "pretrain-1t-plan-audit.pid"
FINAL_ORDER_RUNNER = ROOT / "experiments/e4_curriculum/scripts/materialize_1t_orders.py"
FINAL_ORDER_STATE = HEALTH / "pretrain-1t-order-materialization.json"
FINAL_ORDER_PID = HEALTH / "pretrain-1t-order-materialization.pid"
FINAL_RELEASE_BUILDER = ROOT / "experiments/e4_curriculum/scripts/build_1t_order_release.py"
FINAL_RELEASE = ROOT / "data/processed/releases/pretrain-1t-frozen.release.json"
FINAL_RELEASE_PID = HEALTH / "pretrain-1t-release.pid"
FINAL_TOKENIZER = ROOT / "experiments/e4_curriculum/scripts/tokenize_1t_orders.py"
FINAL_TOKENIZATION_STATE = HEALTH / "pretrain-1t-tokenization.json"
FINAL_TOKENIZATION_PID = HEALTH / "pretrain-1t-tokenization.pid"
FINAL_STREAM_AUDITOR = ROOT / "experiments/e4_curriculum/scripts/audit_1t_token_streams.py"
FINAL_STREAM_AUDIT = ROOT / "data/processed/audits/pretrain_1t_token_streams.json"
FINAL_STREAM_AUDIT_PID = HEALTH / "pretrain-1t-stream-audit.pid"
FINAL_OVERLAP_AUDITOR = ROOT / "experiments/e4_curriculum/scripts/audit_1t_overlap.py"
FINAL_OVERLAP_AUDIT = ROOT / "data/processed/audits/pretrain_1t_overlap.json"
FINAL_OVERLAP_PID = HEALTH / "pretrain-1t-overlap.pid"
PRODUCTION_CONTRACT_VALIDATOR = ROOT / "pretrain/scripts/validate_production_contract.py"
PRODUCTION_CONTRACT_REPORT = HEALTH / "pretrain-production-contract.json"
PRODUCTION_CONTRACT_PID = HEALTH / "pretrain-production-contract.pid"
BACKOFF_SECONDS = (300, 1800, 7200)
MAX_ATTEMPTS = len(BACKOFF_SECONDS)
MIN_EXPERIMENT_GPUS = 4
MAX_EXPERIMENT_GPUS = 8
QUALIFICATION_PARTIAL_STAGES = (
    "lr_schedule_contract", "numerical_parity", "distributed_gradient_parity", "single_gpu", "fsdp4_dcp",
    "recovery_reference", "train_interrupt_resume", "save_interrupt_resume",
    "load_interrupt_resume", "reshard_4to2",
)
LONG_RUNNING_ACTIONS = {
    "e3_order_materialization",
    "e3_tokenization",
    "pretrain_1t_order_materialization",
    "pretrain_1t_tokenization",
    "e1_pilot_queue",
    "e1_pilot_extension_queue",
    "batch_pilot_queue",
    'batch_pilot_extension_queue',
    "e1_queue",
    "e1_performance_queue",
    "e2_queue",
    'e2_confirm_queue',
    'e3_confirm_queue',
    "e3_queue",
    "production_throughput_canary",
}


def select_experiment_gpus(gpus: list[dict[str, Any]],
                           excluded_gpus: set[int] | None = None) -> list[int]:
    """Use idle single-arm slots within the current user GPU ownership limit."""
    excluded_gpus = excluded_gpus or set()
    allowed = allowed_gpu_indices(HEALTH)
    return [
        int(row["index"]) for row in gpus
        if int(row["memory_used_mb"]) <= 1024 and int(row["index"]) not in excluded_gpus
        and int(row['index']) in allowed
    ][:MAX_EXPERIMENT_GPUS]


def active_shared_gpu_policy() -> dict[str, Any] | None:
    """Return the fail-closed policy for provisional loss-only experiments."""
    if not resource_policy(HEALTH)['allow_busy_gpu_sharing']:
        return None
    policy = load_json(SHARED_GPU_POLICY)
    if not policy or policy.get("status") != "ACTIVE_PROVISIONAL":
        return None
    controls = policy.get("controls") or {}
    gpu_ids = controls.get("gpu_indices", policy.get("gpu_indices"))
    required = {
        "formal_acceptance_requires_later_torchtitan_qualification_pass": True,
        "single_gpu_contracts_only": True,
        "tokens_per_update_preserved": 262144,
    }
    if (not isinstance(gpu_ids, list) or not gpu_ids
            or any(controls.get(key) != value for key, value in required.items())
            or controls.get("selection_metrics_allowed") != ["validation_loss"]
            or set(controls.get("performance_metrics_excluded", []))
            != {"throughput(tps)", "mfu(%)", "wall_time"}
            or int(controls.get("max_concurrent_jobs", 0)) > len(gpu_ids)):
        return None
    normalized = sorted({int(value) for value in gpu_ids} & allowed_gpu_indices(HEALTH))
    if not normalized:
        return None
    if any(value < 0 or value >= MAX_EXPERIMENT_GPUS for value in normalized):
        return None
    return {**policy, "gpu_indices": normalized}


def discover_group_runner(group: str, policy: dict[str, Any] | None = None) -> int:
    """Find an exact queue runner, including one started before watcher adoption."""
    candidates: list[int] = []
    if policy and group == "e1_pilot":
        try:
            candidates.append(int(policy.get("queue_pid", 0)))
        except (TypeError, ValueError):
            pass
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = entry.joinpath("cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if (b"--group" in argv and argv.index(b"--group") + 1 < len(argv)
                and argv[argv.index(b"--group") + 1] == os.fsencode(group)
                and runner_matches(int(entry.name), QUEUE_RUNNER)):
            candidates.append(int(entry.name))
    return next((pid for pid in candidates if runner_matches(pid, QUEUE_RUNNER)), 0)


def active_gpu_reservations(exclude_action: str | None = None) -> set[int]:
    state = load_json(ACTION_STATE) or {}
    reserved = set()
    for name, record in (state.get("actions") or {}).items():
        if name == exclude_action or record.get("status") != "RUNNING":
            continue
        script = (QUALIFICATION_RUNNER if name == "torchtitan_qualification"
                  else QUEUE_RUNNER if name.endswith("_queue") else None)
        if script and runner_matches(int(record.get("pid", 0)), script):
            if script == QUEUE_RUNNER:
                group = name[:-6]
                allocation = load_json(HEALTH / f'gpu-allocations/{group}.json') or {}
                pool = allocation.get('gpus', record.get('gpus', []))
                width = int(allocation.get('max_parallel', len(pool)))
                reserved.update(int(value) for value in pool[:width])
                queue = load_json(HEALTH / f'{group}-queue.json') or {}
                reserved.update(int(job['gpu']) for job in queue.get('jobs', {}).values()
                                if job.get('status') == 'RUNNING' and 'gpu' in job
                                and runner_matches(int(job.get('pid', 0)), ROOT / 'pretrain/backends/torchtitan/launch.sh'))
            else:
                reserved.update(int(value) for value in record.get('gpus', []))
    return reserved


def queue_gpu_allocation(gpus: list[dict[str, Any]], qualification_passed: bool,
                         shared_policy: dict[str, Any] | None,
                         excluded_gpus: set[int] | None = None) -> tuple[list[int], str]:
    """Choose idle formal slots or the explicitly controlled provisional slots."""
    excluded_gpus = excluded_gpus or set()
    idle = select_experiment_gpus(gpus, excluded_gpus)
    if qualification_passed and idle:
        return idle, "formal_idle"
    if shared_policy:
        shared = [value for value in shared_policy["gpu_indices"]
                  if value not in excluded_gpus and value in allowed_gpu_indices(HEALTH)]
        return shared, "provisional_shared"
    return [], "unavailable"


def analysis_ready(value: dict[str, Any]) -> bool:
    """A provisional loss analysis may unlock research, never production freeze."""
    return value.get("status") in {"PASS", "PASS_PROVISIONAL"}


def publish_queue_allocation(group: str, record: dict[str, Any], allocated: list[int],
                             mode: str, shared_policy: dict[str, Any] | None) -> list[int]:
    queue = load_json(HEALTH / f"{group}-queue.json") or {}
    jobs = queue.get("jobs", {})
    owned = sorted({int(job["gpu"]) for job in jobs.values()
                    if job.get("status") == "RUNNING" and "gpu" in job
                    and runner_matches(int(job.get("pid", 0)), ROOT / "pretrain/backends/torchtitan/launch.sh")})
    allowed = allowed_gpu_indices(HEALTH)
    if not set(owned).issubset(allowed):
        raise ValueError('A live trainer is outside the current user GPU resource policy')
    allocated = [gpu for gpu in allocated if gpu in allowed]
    names = {path.stem for path in (ROOT / f"pretrain/configs/{group}").glob("*.env")}
    def retry_generation_available(job: dict[str, Any]) -> bool:
        return (job.get('status') == 'RETRY_EXHAUSTED'
                and job.get('failure_kind') != 'nonfinite_gradient'
                and int(record.get('attempts', 0)) < MAX_ATTEMPTS)

    remaining = sum(jobs.get(name, {}).get('status') != 'PASS' and (
        jobs.get(name, {}).get('status') != 'RETRY_EXHAUSTED'
        or retry_generation_available(jobs.get(name, {}))) for name in names)
    limit = min(8, remaining) if names else 8
    if group == "e1_performance":
        limit = min(limit, 5)
    if group in {"e1", "e2"}:
        other = "e2" if group == "e1" else "e1"
        if group_enabled(ROOT, other) and not experiment_result_roster_complete(other):
            actions = (load_json(ACTION_STATE) or {}).get("actions", {})
            other_record = actions.get(f"{other}_queue", {})
            other_slots = (other_record.get("gpus", []) if other_record.get("status") == "RUNNING"
                           and runner_matches(int(other_record.get("pid", 0)), QUEUE_RUNNER) else [])
            capacity = len(set(owned) | set(allocated) | set(other_slots))
            share = (capacity + 1) // 2 if group == "e1" else max(1, capacity // 2)
            limit = min(limit, 4, share)
    pool = list(dict.fromkeys([*owned, *allocated]))[:max(limit, len(owned))]
    shared = (sorted(set(pool) & set(shared_policy["gpu_indices"]))
              if mode == "provisional_shared" and shared_policy else [])
    launch_width = len(pool)
    priority_gpus = set(shared_policy['gpu_indices']) if shared_policy else set()
    if resource_policy(HEALTH)['status'] == 'ACTIVE' and not shared_policy:
        priority_gpus = allowed
    if group in {'batch_pilot', batch_contract.GROUP} and priority_gpus:
        pilot_jobs = (load_json(HEALTH / f'{EXTENSION_GROUP}-queue.json') or {}).get('jobs', {})
        lr_action = (load_json(ACTION_STATE) or {}).get('actions', {}).get(f'{EXTENSION_GROUP}_queue', {})
        waiting_lr = sum(job.get('status') in {'PENDING', 'RETRY_WAIT'} or (
            job.get('status') == 'RETRY_EXHAUSTED' and job.get('failure_kind') != 'nonfinite_gradient'
            and int(lr_action.get('attempts', 0)) < MAX_ATTEMPTS)
            for job in pilot_jobs.values())
        lr_allocation = load_json(HEALTH / f'gpu-allocations/{EXTENSION_GROUP}.json') or {}
        lr_running = sum(job.get('status') == 'RUNNING' and runner_matches(
            int(job.get('pid', 0)), ROOT / 'pretrain/backends/torchtitan/launch.sh')
            for job in pilot_jobs.values())
        lr_free_slots = (max(0, int(lr_allocation.get('max_parallel', 0)) - lr_running)
                         if lr_action.get('status') == 'RUNNING'
                         and runner_matches(int(lr_action.get('pid', 0)), QUEUE_RUNNER) else 0)
        needs_lr = waiting_lr > lr_free_slots
        if needs_lr:
            controlled = priority_gpus
            pool = [gpu for gpu in pool if gpu not in controlled] + [gpu for gpu in pool if gpu in controlled]
            launch_width = sum(gpu not in controlled for gpu in pool)
    payload = {"schema_version": 1, "group": group, "gpus": pool,
               "shared_gpus": shared, "max_parallel": launch_width, "mode": mode}
    path = HEALTH / "gpu-allocations" / f"{group}.json"
    if load_json(path) != payload:
        atomic_json(path, payload)
    record.update(gpus=pool, allocation_file=str(path), allocation_mode=mode)
    return pool


def qualification_needs_eight(report: dict[str, Any]) -> bool:
    return all(report.get("stages", {}).get(stage, {}).get("status") == "PASS"
               for stage in QUALIFICATION_PARTIAL_STAGES)


def completed_checkpoint_exists(run_dir: Path) -> bool:
    for step_dir in (run_dir / "checkpoint").glob("step-*"):
        if ((step_dir / ".metadata").is_file()
                and any(step_dir.glob("*.distcp"))):
            return True
    return False


def checkpointed_shared_handoff(qualification: dict[str, Any],
                                gpus: list[dict[str, Any]],
                                policy: dict[str, Any] | None,
                                enabled: bool) -> dict[str, Any]:
    """Yield shared jobs only when the final eight-GPU gate is ready to run."""
    event: dict[str, Any] = {"name": "shared_queue_qualification_handoff"}
    if (not enabled or not policy or qualification.get("status") == "PASS"
            or not qualification_needs_eight(qualification)):
        event["decision"] = "not_required"
        return event
    idle = {int(row["index"]) for row in gpus if int(row["memory_used_mb"]) <= 1024}
    shared = set(policy["gpu_indices"])
    if not (set(range(MAX_EXPERIMENT_GPUS)) - shared).issubset(idle):
        event["decision"] = "wait_for_nonshared_idle_gpus"
        return event
    for group in ("e1_pilot", EXTENSION_GROUP, "batch_pilot", "e1", "e2", "e3"):
        pid = discover_group_runner(group, policy)
        if not pid:
            continue
        report = load_json(HEALTH / f"{group}-queue.json") or {}
        uncheckpointed = []
        for name, job in (report.get("jobs") or {}).items():
            if job.get("status") != "RUNNING":
                continue
            env_path = Path(str(job.get("env", "")))
            try:
                values = {}
                for raw in env_path.read_text(encoding="utf-8").splitlines():
                    line = raw.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        values[key] = value
                run_dir = Path(values["YUFORMER_PRETRAIN_OUT"])
            except (OSError, KeyError, ValueError):
                uncheckpointed.append(name)
                continue
            if not completed_checkpoint_exists(run_dir):
                uncheckpointed.append(name)
        if uncheckpointed:
            event.update({"decision": "wait_for_checkpoints", "group": group,
                          "pid": pid, "uncheckpointed": sorted(uncheckpointed)})
            return event
        os.kill(pid, signal.SIGTERM)
        event.update({"decision": "started", "group": group, "pid": pid,
                      "reason": "checkpointed_handoff_to_final_8gpu_qualification"})
        return event
    event["decision"] = "queue_already_stopped"
    return event


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def command(args: list[str], *, env: dict[str, str] | None = None, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)


def monitor_interval_seconds(result: dict[str, Any], short_interval: int,
                             long_interval: int) -> int:
    """Use a short cadence around transitions and a sparse cadence for stable long work."""
    actions = result.get("actions", [])
    if not isinstance(actions, list):
        return short_interval
    decisions = {row.get("decision") for row in actions if isinstance(row, dict)}
    if decisions & {"started", "failed", "retry_exhausted", "stalled_suspect"}:
        return short_interval
    if any(
        isinstance(row, dict)
        and row.get("name") in LONG_RUNNING_ACTIONS
        and row.get("decision") == "running"
        for row in actions
    ):
        return long_interval
    return short_interval


def gpu_snapshot() -> list[dict[str, Any]]:
    result = command([
        "nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    rows = []
    if result.returncode:
        return rows
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 6:
            rows.append({
                "index": int(fields[0]), "name": fields[1], "driver": fields[2],
                "memory_total_mb": int(fields[3]), "memory_used_mb": int(fields[4]),
                "utilization_pct": int(fields[5]),
            })
    return rows


def run_bootstrap_if_eligible(preflight: dict[str, Any], enabled: bool) -> list[dict[str, Any]]:
    state = load_json(ACTION_STATE) or {"schema_version": 1, "actions": {}}
    actions = state.setdefault("actions", {})
    record = actions.setdefault("bootstrap_torchtitan_env", {"attempts": 0, "status": "WAITING"})
    checks = preflight.get("checks") or {}
    python312 = shutil.which("python3.12")
    if not python312 and MANAGED_PYTHON312.is_file():
        python312 = str(MANAGED_PYTHON312)
    event: dict[str, Any] = {"name": "bootstrap_torchtitan_env"}
    if preflight.get("status") == "ready":
        record.update({"status": "PASS", "last_observed_at": now_iso()})
        event["decision"] = "already_ready"
    elif not enabled:
        event["decision"] = "disabled"
    elif checks.get("nvidia_driver") is not True:
        record.update({"status": "WAITING_EXTERNAL", "blocker": "nvidia_driver", "last_observed_at": now_iso()})
        event["decision"] = "wait_for_driver"
    elif not python312:
        record.update({"status": "WAITING_EXTERNAL", "blocker": "python3.12", "last_observed_at": now_iso()})
        event["decision"] = "wait_for_python3.12"
    elif int(record.get("attempts", 0)) >= MAX_ATTEMPTS:
        record["status"] = "RETRY_EXHAUSTED"
        event["decision"] = "retry_exhausted"
    elif float(record.get("next_retry_epoch", 0)) > time.time():
        event.update({"decision": "backoff", "next_retry_at": record.get("next_retry_at")})
    else:
        attempt = int(record.get("attempts", 0)) + 1
        bootstrap_env = dict(os.environ)
        bootstrap_env.update({"YUFORMER_PYTHON312": python312, "YUFORMER_TORCHTITAN_ROOT": str(SOURCE_ROOT)})
        try:
            result = command(["bash", str(BOOTSTRAP)], env=bootstrap_env, timeout=7200)
            rc, detail = result.returncode, (result.stderr or result.stdout)[-4000:]
        except subprocess.TimeoutExpired as exc:
            rc, detail = 124, f"TimeoutExpired after {exc.timeout}s"
        retry_epoch = time.time() + BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)]
        record.update({
            "attempts": attempt, "last_attempt_at": now_iso(), "last_rc": rc,
            "last_output_tail": detail, "status": "PASS" if rc == 0 else "RETRY_WAIT",
            "next_retry_epoch": retry_epoch,
            "next_retry_at": datetime.fromtimestamp(retry_epoch, timezone.utc).isoformat(),
        })
        event.update({"decision": "attempted", "attempt": attempt, "rc": rc})
    state["updated_at"] = now_iso()
    atomic_json(ACTION_STATE, state)
    return [event]


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.waitpid(pid, os.WNOHANG)
    except (ChildProcessError, ProcessLookupError):
        pass
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return len(stat) > 2 and stat[2] != "Z"
    except (OSError, ValueError):
        return False


def process_matches(pid: int, argv: list[str]) -> bool:
    """Reject stale/reused PID files before treating an action as running."""
    if not process_alive(pid):
        return False
    try:
        observed = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    expected = os.fsencode(argv[1] if len(argv) > 1 else argv[0])
    return expected in observed


def runner_matches(pid: int, script: Path) -> bool:
    """Prove an action PID still belongs to its declared Python runner."""
    if not process_alive(pid):
        return False
    try:
        observed = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        cwd = Path(f"/proc/{pid}/cwd").resolve()
    except OSError:
        return False
    expected = script.resolve()
    for raw in observed:
        if not raw:
            continue
        candidate = Path(os.fsdecode(raw))
        if candidate.suffix not in {".py", ".sh"}:
            continue
        try:
            resolved = candidate.resolve() if candidate.is_absolute() else (cwd / candidate).resolve()
        except OSError:
            continue
        if resolved == expected:
            return True
    return False


def process_tree_telemetry(root_pid: int) -> dict[str, Any]:
    """Collect a cheap progress heartbeat for a background process tree."""
    parents: dict[int, int] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text(encoding="utf-8")
            ppid = int(next(line.split()[1] for line in status.splitlines()
                            if line.startswith("PPid:")))
            parents[int(entry.name)] = ppid
        except (OSError, StopIteration, ValueError):
            continue
    pids, frontier = {root_pid}, {root_pid}
    while frontier:
        children = {pid for pid, parent in parents.items() if parent in frontier and pid not in pids}
        pids.update(children)
        frontier = children
    result: dict[str, Any] = {"pids": sorted(pids), "read_bytes": 0, "write_bytes": 0,
                              "cpu_ticks": 0, "rss_kib": 0, "states": {}}
    for pid in pids:
        try:
            io_values = {}
            for line in Path(f"/proc/{pid}/io").read_text(encoding="utf-8").splitlines():
                key, value = line.split(":", 1)
                io_values[key] = int(value.strip())
            stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
            fields = stat_text[stat_text.rfind(")") + 2:].split()
            status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines()
            rss = next((int(line.split()[1]) for line in status if line.startswith("VmRSS:")), 0)
            state = fields[0]
            result["read_bytes"] += io_values.get("read_bytes", 0)
            result["write_bytes"] += io_values.get("write_bytes", 0)
            result["cpu_ticks"] += int(fields[11]) + int(fields[12])
            result["rss_kib"] += rss
            result["states"][state] = int(result["states"].get(state, 0)) + 1
        except (OSError, ValueError, IndexError):
            continue
    result["signature"] = ":".join(str(result[key]) for key in ("read_bytes", "write_bytes", "cpu_ticks"))
    return result


def observe_running(record: dict[str, Any], event: dict[str, Any], pid: int) -> None:
    telemetry = process_tree_telemetry(pid)
    previous_signature = record.get("progress_signature")
    stale = int(record.get("stale_observations", 0)) + 1 if previous_signature == telemetry["signature"] else 0
    if previous_signature != telemetry['signature']:
        record['last_progress_epoch'] = time.time()
    elif time.time() - float(record.setdefault('last_progress_epoch', time.time())) >= 7200:
        from process_control import terminate_process_tree
        victims = terminate_process_tree(pid)
        record.update(status='RETRY_WAIT', recovery_reason='background_process_tree_no_progress',
                      last_stall_at=now_iso(), terminated_pids=victims)
        record.pop('pid', None)
        event.update(decision='stalled_retry', terminated_pids=victims)
        return
    record.update({"status": "RUNNING", "pid": pid, "last_observed_at": now_iso(),
                   "progress_signature": telemetry["signature"], "stale_observations": stale,
                   "telemetry": telemetry})
    record.pop("blocker", None)
    event.update({"decision": "stalled_suspect" if stale >= 6 else "running", "pid": pid,
                  "stale_observations": stale, "telemetry": telemetry})


def start_qualification_if_eligible(preflight: dict[str, Any], contracts: dict[str, Any],
                                    gpus: list[dict[str, Any]], enabled: bool,
                                    reserved_gpus: set[int] | None = None
                                    ) -> tuple[dict[str, Any], dict[str, Any]]:
    state = load_json(ACTION_STATE) or {"schema_version": 1, "actions": {}}
    actions = state.setdefault("actions", {})
    record = actions.setdefault("torchtitan_qualification", {"attempts": 0, "status": "WAITING"})
    report = load_json(QUALIFICATION_REPORT) or {"status": "PENDING", "stages": {}}
    current_identity = qualification_identity(ROOT)
    production_world = int((load_json(HEALTH / "pretrain-user-full-pretrain-authorization.json") or {})
                           .get("training", {}).get("world_size", 4))
    qualification_stale = False
    if report.get("status") == "PASS" and (
            not report_matches(report, current_identity)
            or (production_world >= 8 and report.get("stages", {}).get(
                "reshard_4to8", {}).get("status") != "PASS")):
        qualification_stale = True
        report = {"status": "STALE", "stages": {},
                  "current_qualification_identity": current_identity,
                  "reason": "qualification inputs changed since PASS evidence was recorded"}
    reserved_gpus = reserved_gpus or set()
    idle = [int(row["index"]) for row in gpus if int(row["memory_used_mb"]) <= 1024]
    # Preserve the explicitly shared single-GPU experiment lane whenever four
    # other idle devices can satisfy the current qualification stage.
    idle.sort(key=lambda value: (value in reserved_gpus, value))
    # Every gate through 4->2 reshard needs at most four devices.  Run that
    # work as soon as four cards are free; reserve the eight-card requirement
    # for the final 4->8 resume instead of idling the available hardware.
    required_gpus = (
        8 if production_world >= 8 or qualification_needs_eight(report)
        else 4
    )
    event: dict[str, Any] = {"name": "torchtitan_qualification"}
    pid = int(record.get("pid", 0))
    if pid and not runner_matches(pid, QUALIFICATION_RUNNER):
        pid = 0
        record.pop("pid", None)
    if qualification_stale and (not pid or not runner_matches(pid, QUALIFICATION_RUNNER)):
        record.update({"attempts": 0, "status": "STALE"})
        for key in ("pid", "progress_signature", "stale_observations",
                    "next_retry_epoch", "next_retry_at"):
            record.pop(key, None)
        pid = 0
    if report.get("status") == "PASS":
        record.update({"status": "PASS", "last_observed_at": now_iso()})
        record.pop("pid", None)
        event["decision"] = "already_passed"
    elif pid and runner_matches(pid, QUALIFICATION_RUNNER):
        observe_running(record, event, pid)
    elif not enabled:
        event["decision"] = "disabled"
    elif preflight.get("status") != "ready":
        record.update({"status": "WAITING_PREREQUISITE", "blocker": "preflight", "last_observed_at": now_iso()})
        event["decision"] = "wait_for_preflight"
    elif contracts.get("status") != "PASS":
        record.update({"status": "WAITING_PREREQUISITE", "blocker": "formal_contracts", "last_observed_at": now_iso()})
        event["decision"] = "wait_for_contracts"
    elif len(idle) < required_gpus:
        blocker = "eight_idle_gpus" if required_gpus == 8 else "four_idle_gpus"
        record.update({"status": "WAITING_EXTERNAL", "blocker": blocker,
                       "required_gpus": required_gpus, "last_observed_at": now_iso()})
        event["decision"] = "wait_for_idle_gpus"
    elif int(record.get("attempts", 0)) >= MAX_ATTEMPTS:
        record["status"] = "RETRY_EXHAUSTED"
        event["decision"] = "retry_exhausted"
    elif float(record.get("next_retry_epoch", 0)) > time.time():
        event.update({"decision": "backoff", "next_retry_at": record.get("next_retry_at")})
    else:
        attempt = int(record.get("attempts", 0)) + 1
        # If all eight are already free, hand them to the runner so it can
        # continue directly into 4->8 without waiting for another watch cycle.
        selected_gpus = idle[:8] if len(idle) >= 8 else idle[:required_gpus]
        gpu_arg = ",".join(map(str, selected_gpus))
        log_path = ROOT / "report/logs/torchtitan_qualification.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                [str(ENV_PYTHON), str(QUALIFICATION_RUNNER), "--gpus", gpu_arg],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        retry_epoch = time.time() + BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)]
        record.update({
            "attempts": attempt, "status": "RUNNING", "pid": process.pid,
            "gpus": selected_gpus, "required_gpus": required_gpus,
            "last_attempt_at": now_iso(), "log": str(log_path),
            "next_retry_epoch": retry_epoch,
            "next_retry_at": datetime.fromtimestamp(retry_epoch, timezone.utc).isoformat(),
        })
        event.update({"decision": "started", "attempt": attempt, "pid": process.pid,
                      "gpus": selected_gpus, "required_gpus": required_gpus})
    state["updated_at"] = now_iso()
    atomic_json(ACTION_STATE, state)
    if record.get("status") == "RUNNING" and report.get("status") == "PENDING":
        report = {**report, "status": "IN_PROGRESS"}
    return report, event


def start_pilots_if_eligible(qualification: dict[str, Any], selection: dict[str, Any],
                             gpus: list[dict[str, Any]], enabled: bool,
                             shared_policy: dict[str, Any] | None = None
                             ) -> tuple[dict[str, Any], dict[str, Any]]:
    state = load_json(ACTION_STATE) or {"schema_version": 1, "actions": {}}
    actions = state.setdefault("actions", {})
    record = actions.setdefault("e1_pilot_queue", {"attempts": 0, "status": "WAITING"})
    queue = load_json(PILOT_QUEUE_REPORT) or {"status": "PENDING", "jobs": {}}
    allocated, allocation_mode = queue_gpu_allocation(
        gpus, qualification.get("status") == "PASS", shared_policy,
        active_gpu_reservations("e1_pilot_queue"),
    )
    if enabled:
        allocated = publish_queue_allocation("e1_pilot", record, allocated, allocation_mode, shared_policy)
    event: dict[str, Any] = {"name": "e1_pilot_queue"}
    pid = int(record.get("pid", 0))
    if pid and not runner_matches(pid, QUEUE_RUNNER):
        pid = 0
        record.pop("pid", None)
    if not pid:
        pid = discover_group_runner("e1_pilot", shared_policy)
        if pid:
            record.update({"pid": pid, "status": "RUNNING", "adopted": True,
                           "allocation_mode": allocation_mode,
                           "gpus": allocated, "last_observed_at": now_iso()})
    result_roster_complete = experiment_result_roster_complete("e1_pilot")
    if (queue.get("status") == "PASS" and not result_roster_complete
            and (not pid or not runner_matches(pid, QUEUE_RUNNER))):
        queue = invalidate_stale_queue_pass(PILOT_QUEUE_REPORT, queue, record)
        pid = 0
    if selection.get("status") == "PASS":
        record.update({"status": "PASS", "last_observed_at": now_iso()})
        event["decision"] = "already_passed"
    elif queue.get("status") == "PASS" and result_roster_complete:
        record.update({"status": "WAITING_ANALYSIS", "last_observed_at": now_iso(),
                       "blocker": "lr_selection_gate"})
        event["decision"] = "wait_for_lr_selection_gate"
    elif pid and runner_matches(pid, QUEUE_RUNNER):
        observe_running(record, event, pid)
    elif not enabled:
        event["decision"] = "disabled"
    elif qualification.get("status") != "PASS" and not shared_policy:
        record.update({"status": "WAITING_PREREQUISITE", "blocker": "qualification", "last_observed_at": now_iso()})
        event["decision"] = "wait_for_qualification"
    elif not allocated:
        record.update({"status": "WAITING_EXTERNAL", "blocker": "experiment_gpu_allocation", "last_observed_at": now_iso()})
        event["decision"] = "wait_for_idle_gpus"
    elif int(record.get("attempts", 0)) >= MAX_ATTEMPTS:
        record["status"] = "RETRY_EXHAUSTED"
        event["decision"] = "retry_exhausted"
    elif float(record.get("next_retry_epoch", 0)) > time.time():
        event.update({"decision": "backoff", "next_retry_at": record.get("next_retry_at")})
    else:
        attempt = int(record.get("attempts", 0)) + 1
        gpu_arg = ",".join(map(str, allocated))
        log_path = ROOT / "report/logs/e1_pilot_queue.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            argv = [str(ENV_PYTHON), str(QUEUE_RUNNER), "--group", "e1_pilot",
                    "--gpus", gpu_arg, "--max-parallel", "8",
                    "--allocation-file", str(HEALTH / "gpu-allocations/e1_pilot.json")]
            if allocation_mode == "provisional_shared":
                argv.append("--provisional-loss-only")
            process = subprocess.Popen(
                argv,
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        retry_epoch = time.time() + BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)]
        record.update({"attempts": attempt, "status": "RUNNING", "pid": process.pid,
                       "gpus": allocated, "allocation_mode": allocation_mode,
                       "provisional": allocation_mode == "provisional_shared",
                       "last_attempt_at": now_iso(), "log": str(log_path),
                       "next_retry_epoch": retry_epoch,
                       "next_retry_at": datetime.fromtimestamp(retry_epoch, timezone.utc).isoformat()})
        event.update({"decision": "started", "attempt": attempt, "pid": process.pid,
                      "gpus": allocated, "max_parallel": len(allocated)})
        queue = {**queue, "status": "IN_PROGRESS"}
    state["updated_at"] = now_iso()
    atomic_json(ACTION_STATE, state)
    return queue, event


def refresh_pilot_results() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    python = shutil.which("python3") or "python3"
    completions = list(PILOT_ROOT.glob("*/training_complete.json"))
    completions.extend((ROOT / 'data/experiments' / EXTENSION_GROUP).glob('*/training_complete.json'))
    for completion in sorted(completions):
        run_dir = completion.parent
        output = run_dir / "result.json"
        if output.is_file() and output.stat().st_mtime >= completion.stat().st_mtime:
            continue
        result = command([python, str(RESULT_COLLECTOR), str(run_dir), "--output", str(output)])
        events.append({"name": "collect_pilot_result", "run_dir": str(run_dir), "rc": result.returncode})
    selection_run = command([
        python, str(LR_SELECTOR), "--results-root", str(PILOT_ROOT), "--output", str(LR_SELECTION),
        '--root', str(ROOT),
    ])
    selection = load_json(LR_SELECTION) or {"status": "BLOCKED", "observed_results": 0}
    if selection_run.returncode not in (0, 2):
        events.append({"name": "select_lr_pilots", "rc": selection_run.returncode})
    return selection, events


def refresh_batch_anchor(selection: dict, identity_audit: dict, enabled: bool) -> tuple[dict, list]:
    events = []
    anchor = night_research.build_batch_anchor(ROOT, selection, identity_audit)
    # Keep the anchor byte-stable as unrelated 800M results arrive.
    from select_lr_pilots import atomic_json as stable_json
    stable_json(BATCH_ANCHOR, anchor)
    policy = load_json(ROOT / night_research.POLICY) or {}
    if (enabled and policy.get('enabled') is True
            and policy.get('batch_can_start_from_qualified_200m_anchor') is True
            and night_research.batch_anchor_is_current(ROOT, anchor)):
        selection_sha = sha256(BATCH_ANCHOR)
        resolution = load_json(BATCH_PILOT_RESOLUTION) or {}
        if (resolution.get("lr_selection_sha256") != selection_sha
                or resolution.get("preparer_tool_sha256") != sha256(BATCH_PREPARER)):
            resolved = command([
                shutil.which('python3') or 'python3', str(BATCH_PREPARER),
                '--root', str(ROOT), '--selection', str(BATCH_ANCHOR),
                '--output', str(BATCH_PILOT_RESOLUTION),
            ])
            events.append({'name': 'prepare_batch_pilots_from_200m_anchor', 'rc': resolved.returncode,
                           'output_tail': (resolved.stderr or resolved.stdout)[-1000:] if resolved.returncode else ''})
    return anchor, events


def batch_pilot_resolution_is_current(selection: dict[str, Any], contract_path: Path) -> bool:
    resolution = load_json(BATCH_PILOT_RESOLUTION)
    if (not night_research.batch_anchor_is_current(ROOT, selection) or not resolution
            or resolution.get("status") != "PASS" or not BATCH_ANCHOR.is_file()
            or resolution.get("lr_selection_sha256") != sha256(BATCH_ANCHOR)
            or resolution.get("preparer_tool_sha256") != sha256(BATCH_PREPARER)):
        return False
    expected = {str(path.relative_to(ROOT)): sha256(path)
                for path in (ROOT / "pretrain/configs/batch_pilot").glob("*.yaml")}
    recorded = {row.get("config"): row.get("config_sha256")
                for row in resolution.get("configs", []) if isinstance(row, dict)}
    return (len(expected) == 9 and recorded == expected
            and contracts_cover_current_configs(contract_path, expected))


def contracts_cover_current_configs(contract_path: Path, expected: dict[str, str]) -> bool:
    report = load_json(contract_path) or {}
    if report.get('status') != 'PASS' or not report.get('roster_exact'):
        return False
    rows = {row.get('config'): row for row in report.get('rows', [])}
    for name, digest in expected.items():
        row = rows.get(name, {})
        if row.get('status') != 'PASS' or row.get('config_sha256') != digest:
            return False
        path = ROOT / name
        env = ROOT / str(row.get('env', ''))
        if (not path.is_file() or sha256(path) != digest or not env.is_file()
                or sha256(env) != row.get('env_sha256')):
            return False
    return bool(expected)


def finalize_experiment_resolution(lr_selection: dict[str, Any],
                                   batch_selection: dict[str, Any]) -> dict[str, Any]:
    if lr_selection.get("status") != "PASS" or not analysis_ready(batch_selection):
        return load_json(LR_RESOLUTION) or {"status": "PENDING"}
    python = shutil.which("python3") or "python3"
    current = load_json(LR_RESOLUTION) or {}
    lr_sha = sha256(LR_SELECTION)
    batch_sha = sha256(BATCH_SELECTION)
    if (current.get("selection_sha256") != lr_sha
            or current.get("batch_selection_sha256") != batch_sha
            or current.get('research_scope_sha256') != scope_sha256(ROOT)
            or current.get("resolver_tool_sha256") != sha256(LR_RESOLVER)):
        command([python, str(LR_RESOLVER), "--selection", str(LR_SELECTION),
                 "--batch-selection", str(BATCH_SELECTION), "--output", str(LR_RESOLUTION)])
    return load_json(LR_RESOLUTION) or {"status": "FAIL"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact_trigger(*paths: Path, salt: str = "") -> str:
    """Create a retry generation from small prerequisite/evidence artifacts."""
    digest = hashlib.sha256(salt.encode("utf-8"))
    for path in paths:
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"MISSING")
        digest.update(b"\0")
    return digest.hexdigest()


def post_freeze_plan_audit_current(frozen: dict[str, Any],
                                   audit: dict[str, Any]) -> bool:
    """Accept the shared static-audit path only when it binds frozen outputs."""
    if (frozen.get("status") != "PASS" or audit.get("status") != "PASS_STATIC"
            or audit.get("errors") or not FROZEN_RECIPE.is_file()
            or not FINAL_PLAN_AUDIT.is_file()
            or FINAL_PLAN_AUDIT.stat().st_mtime_ns < FROZEN_RECIPE.stat().st_mtime_ns):
        return False
    inputs = audit.get("inputs") or {}
    if (inputs.get("frozen_recipe") != str(FROZEN_RECIPE.relative_to(ROOT))
            or inputs.get("frozen_recipe_sha256") != sha256(FROZEN_RECIPE)):
        return False
    outputs = frozen.get("outputs") or {}
    boundary = (audit.get("checks") or {}).get("optimizer_aligned_stage_boundaries") or {}
    bound = [(boundary.get("config"), boundary.get("config_sha256"))]
    bound.extend((row.get("mix_config"), row.get("mix_config_sha256"))
                 for row in audit.get("stages", []))
    if len(bound) != 4:
        return False
    for relative, observed_sha in bound:
        if (not relative or not observed_sha or outputs.get(relative) != observed_sha
                or not (ROOT / relative).is_file() or sha256(ROOT / relative) != observed_sha):
            return False
    return True


def production_canary_current(frozen: dict[str, Any], report: dict[str, Any]) -> bool:
    if (frozen.get("status") != "PASS" or report.get("status") != "PASS"
            or not FROZEN_RECIPE.is_file() or not PRODUCTION_CANARY_CONFIG.is_file()
            or not PRODUCTION_CANARY_ENV.is_file()):
        return False
    evaluation = report.get("evaluation") or {}
    if (frozen.get('training_adoption', {}).get('global_training_budget') is not None
            and evaluation.get('runtime_scope') != 'TOTAL_4K_16K_64K'):
        return False
    return (
        report.get("frozen_recipe_sha256") == sha256(FROZEN_RECIPE)
        and report.get("config_sha256") == sha256(PRODUCTION_CANARY_CONFIG)
        and report.get("env_sha256") == sha256(PRODUCTION_CANARY_ENV)
        and evaluation.get("status") == "PASS"
        and isinstance(evaluation.get("projected_total_training_days", evaluation.get("projected_1t_days_from_p10")), (int, float))
        and int(evaluation.get("world_size", -1)) == 8
        and report_matches(report, qualification_identity(ROOT))
        and float(evaluation.get("projected_total_training_days", evaluation.get("projected_1t_days_from_p10"))) <= float(
            load_authorization(ROOT)["training"]["launch_runtime_ceiling_days"])
    )


def lr_resolution_is_current(selection: dict[str, Any], contract_path: Path) -> bool:
    """Require a PASS resolution bound to selection and every current E1/E2 config."""
    resolution = load_json(LR_RESOLUTION)
    if (not selection_is_current(ROOT, selection)
            or not resolution or not analysis_ready(resolution)):
        return False
    if resolution.get("resolver_tool_sha256") != sha256(LR_RESOLVER):
        return False
    if resolution.get('research_scope_sha256') != scope_sha256(ROOT):
        return False
    budget_path = ROOT / 'pretrain/configs/research_budget.yaml'
    if budget_path.is_file() and resolution.get('budget_policy_sha256') != sha256(budget_path):
        return False
    if not LR_SELECTION.is_file() or resolution.get("selection_sha256") != sha256(LR_SELECTION):
        return False
    if (not BATCH_SELECTION.is_file()
            or resolution.get("batch_selection_sha256") != sha256(BATCH_SELECTION)):
        return False
    rows = resolution.get("configs")
    expected_count = sum(1 for group in ("e1", "e2")
                         if group_enabled(ROOT, group)
                         for _ in (ROOT / "pretrain/configs" / group).glob("*.yaml"))
    if not isinstance(rows, list) or len(rows) != expected_count:
        return False
    expected = {
        str(path.relative_to(ROOT)): sha256(path)
        for group in ("e1", "e2")
        if group_enabled(ROOT, group)
        for path in (ROOT / "pretrain/configs" / group).glob("*.yaml")
    }
    recorded = {row.get("config"): row.get("config_sha256") for row in rows if isinstance(row, dict)}
    if recorded != expected:
        return False
    return contracts_cover_current_configs(contract_path, expected)


IDENTITY_DRIFT_ERRORS = {
    "run_name_mismatch", "resolved_config_drift", "training_source_drift", "model_configs_drift",
    "completion_identity_mismatch", "result_identity_mismatch",
    "dataset_id_drift", "snapshot_id_drift", "tokenizer_id_drift",
    "tokenizer_sha256_drift", "token_stream_audit_drift", "data_gate_not_pass",
}


def quarantine_identity_drift(audit: dict[str, Any], enabled: bool) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if not enabled:
        return events
    experiments_root = (ROOT / "data/experiments").resolve()
    for row in audit.get("rows", []):
        errors = set(row.get("errors") or [])
        if not errors.intersection(IDENTITY_DRIFT_ERRORS):
            continue
        run_dir = Path(str(row.get("run_dir", ""))).resolve()
        try:
            run_dir.relative_to(experiments_root)
        except ValueError:
            events.append({"name": "quarantine_identity_drift", "decision": "refused_outside_root",
                           "run_dir": str(run_dir)})
            continue
        if not run_dir.is_dir():
            continue
        lease = RunLease(run_dir, "read-only-audit")
        generations = lease._generations()
        if generations:
            _, current_generation = max(generations)
            owner, age, released = lease._owner(current_generation)
            if not released and not lease._same_host_owner_dead(owner) and age < lease.stale_seconds:
                events.append({"name": "quarantine_identity_drift", "decision": "deferred_active_lease",
                               "run_dir": str(run_dir), "errors": sorted(errors)})
                continue
        archive_root = run_dir.parent / "_stale_identity"
        archive_root.mkdir(parents=True, exist_ok=True)
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target = archive_root / f"{run_dir.name}-{suffix}"
        move_artifact(run_dir, target)
        events.append({"name": "quarantine_identity_drift", "decision": "archived",
                       "group": row.get("group"), "run": row.get("name"),
                       "errors": sorted(errors), "archive": str(target)})
    return events


def invalidate_unbound_analyses(identity_audit: dict[str, Any],
                                lr_selection: dict[str, Any],
                                analyses: dict[str, dict[str, Any]]) -> None:
    failed_groups = {row.get("group") for row in identity_audit.get("rows", [])
                     if row.get("status") != "PASS"}
    if failed_groups.intersection({'e1_pilot', EXTENSION_GROUP}) and lr_selection.get("status") == "PASS":
        lr_selection["status"] = "BLOCKED"
        lr_selection["blocker"] = "experiment_identity_audit"
    if batch_contract.GROUP in failed_groups:
        failed_groups.add('batch_pilot')
    for group, analysis in analyses.items():
        if group in failed_groups and analysis_ready(analysis):
            analysis["status"] = "BLOCKED"
            analysis["blocker"] = "experiment_identity_audit"


def refresh_group_results(group: str, results_root: Path, analyzer: Path,
                          analysis_output: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    python = shutil.which("python3") or "python3"
    completions = list(results_root.glob('*/training_complete.json'))
    if group == 'batch_pilot':
        completions.extend((ROOT / 'data/experiments' / batch_contract.GROUP).glob('*/training_complete.json'))
    for completion in sorted(completions):
        run_dir = completion.parent
        output = run_dir / "result.json"
        if output.is_file() and output.stat().st_mtime_ns >= completion.stat().st_mtime_ns:
            continue
        result = command([python, str(RESULT_COLLECTOR), str(run_dir), "--output", str(output)])
        events.append({"name": f"collect_{group}_result", "run_dir": str(run_dir),
                       "rc": result.returncode})
    args = [python, str(analyzer), "--results-root", str(results_root), "--output", str(analysis_output)]
    if group in {"e1", 'batch_pilot'}:
        args.extend(["--root", str(ROOT)])
    analyzed = command(args, timeout=1800)
    analysis = load_json(analysis_output) or {"status": "BLOCKED", "observed_results": 0}
    if analyzed.returncode not in (0, 2):
        events.append({"name": f"analyze_{group}_results", "rc": analyzed.returncode,
                       "output_tail": (analyzed.stderr or analyzed.stdout)[-1000:]})
    return analysis, events


def experiment_result_roster_complete(group: str) -> bool:
    """Verify that every declared arm still has a committed PASS result."""
    env_paths = sorted((ROOT / "pretrain/configs" / group).glob("*.env"))
    if not env_paths:
        return False
    for env_path in env_paths:
        values: dict[str, str] = {}
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
        run_dir = values.get("YUFORMER_PRETRAIN_OUT")
        result = load_json(Path(run_dir) / "result.json") if run_dir else None
        if not result or result.get("status") != "PASS":
            return False
    return True


def advance_confirmation(stage: str, screen: dict, gpus: list,
                         qualification: dict, enabled: bool, shared_policy: dict | None) -> tuple[dict, dict]:
    if not group_enabled(ROOT, f'{stage}_confirm'):
        return {'status': 'DEFERRED', 'reason': 'research_scope'}, {
            'name': f'{stage}_confirmation', 'decision': 'deferred_by_research_scope'}
    policy_path = ROOT / 'pretrain/configs/research_budget.yaml'
    if not policy_path.is_file() or not yaml_policy_confirmation_required(policy_path):
        return {'status': 'PASS', 'scope': 'confirmation_not_required_by_policy'}, {'name': f'{stage}_confirmation', 'decision': 'not_required'}
    path = research_confirmation.result_path(ROOT, stage)
    analysis = load_json(path) or {'status': 'BLOCKED', 'reason': 'screen_pending'}
    contracts_ready = False
    if screen.get('status') == 'PASS' and enabled:
        try:
            research_confirmation.prepare(ROOT, stage)
            from validate_formal_contracts import validate_all, atomic_write
            confirmation_contracts = validate_all(ROOT)
            atomic_write(ROOT / 'report/formal_contract_validation_latest.json',
                         json.dumps(confirmation_contracts, indent=2, sort_keys=True) + '\n')
            contracts_ready = confirmation_contracts.get('status') == 'PASS'
            analysis = research_confirmation.analyze(ROOT, stage)
            research_confirmation.atomic(path, analysis)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            analysis = {'status': 'BLOCKED', 'reason': f'confirmation_setup:{exc}'}
            research_confirmation.atomic(path, analysis)
    queue, event = start_group_if_eligible(
        f'{stage}_confirm', screen.get('status') == 'PASS' and contracts_ready, f'{stage}_screen_selection',
        analysis, gpus, enabled, shared_policy=shared_policy,
        qualification_passed=qualification.get('status') == 'PASS')
    return {**analysis, 'queue_status': queue.get('status')}, event


def yaml_policy_confirmation_required(path: Path) -> bool:
    import yaml
    return yaml.safe_load(path.read_text()).get('confirmation', {}).get('required') is True


def invalidate_stale_queue_pass(report_path: Path, queue: dict[str, Any],
                                record: dict[str, Any]) -> dict[str, Any]:
    """Re-open a queue whose PASS result artifacts were removed or invalidated."""
    queue = {**queue, "status": "STALE_RESULT_EVIDENCE", "updated_at": now_iso()}
    record.update({"attempts": 0, "status": "RETRY_TRIGGERED",
                   "recovery_reason": "pass_result_evidence_missing"})
    for key in ("pid", "next_retry_epoch", "next_retry_at", "progress_signature",
                "stale_observations"):
        record.pop(key, None)
    atomic_json(report_path, queue)
    return queue


def start_group_if_eligible(group: str, prerequisite_ready: bool, blocker: str,
                            analysis: dict[str, Any], gpus: list[dict[str, Any]],
                            enabled: bool, excluded_gpus: set[int] | None = None,
                            shared_policy: dict[str, Any] | None = None,
                            qualification_passed: bool = False,
                            ) -> tuple[dict[str, Any], dict[str, Any]]:
    if not group_enabled(ROOT, group):
        return {'status': 'DEFERRED', 'jobs': {}}, {
            'name': f'{group}_queue', 'decision': 'deferred_by_research_scope'}
    state = load_json(ACTION_STATE) or {"schema_version": 1, "actions": {}}
    actions = state.setdefault("actions", {})
    action_name = f"{group}_queue"
    record = actions.setdefault(action_name, {"attempts": 0, "status": "WAITING"})
    report_path = {"batch_pilot": BATCH_QUEUE_REPORT,
                   "e1": E1_QUEUE_REPORT, "e2": E2_QUEUE_REPORT,
                   "e3": E3_QUEUE_REPORT, "e1_performance": PERFORMANCE_QUEUE_REPORT,
                   'e2_confirm': HEALTH / 'e2_confirm-queue.json',
                   'e3_confirm': HEALTH / 'e3_confirm-queue.json',
                   EXTENSION_GROUP: HEALTH / f'{EXTENSION_GROUP}-queue.json',
                   batch_contract.GROUP: HEALTH / f'{batch_contract.GROUP}-queue.json'}[group]
    if group == "e1_performance":
        shared_policy = None
        prerequisite_ready = prerequisite_ready and qualification_passed
    queue = load_json(report_path) or {"status": "PENDING", "jobs": {}}
    allocated, allocation_mode = queue_gpu_allocation(
        gpus, qualification_passed, shared_policy,
        (excluded_gpus or set()) | active_gpu_reservations(action_name),
    )
    if group == "e1_performance":
        allocated = select_experiment_gpus(
            gpus, (excluded_gpus or set()) | active_gpu_reservations(action_name))[:5]
        allocation_mode = "formal_idle" if allocated else "unavailable"
    if enabled:
        allocated = publish_queue_allocation(group, record, allocated, allocation_mode, shared_policy)
    event: dict[str, Any] = {"name": action_name}
    pid = int(record.get("pid", 0))
    if pid and not runner_matches(pid, QUEUE_RUNNER):
        pid = 0
        record.pop("pid", None)
    if not pid:
        pid = discover_group_runner(group, shared_policy)
        if pid:
            record.update({"pid": pid, "status": "RUNNING", "adopted": True,
                           "allocation_mode": allocation_mode,
                           "gpus": allocated, "last_observed_at": now_iso()})
    result_roster_complete = experiment_result_roster_complete(group)
    if (queue.get("status") == "PASS" and not result_roster_complete
            and (not pid or not runner_matches(pid, QUEUE_RUNNER))):
        queue = invalidate_stale_queue_pass(report_path, queue, record)
        pid = 0
    if analysis_ready(analysis):
        record.update({"status": "PASS", "last_observed_at": now_iso()})
        event["decision"] = "already_passed"
    elif queue.get("status") == "PASS" and result_roster_complete:
        record.update({"status": "WAITING_ANALYSIS", "last_observed_at": now_iso(),
                       "blocker": "analysis_gate"})
        event["decision"] = "wait_for_analysis_gate"
    elif pid and runner_matches(pid, QUEUE_RUNNER):
        observe_running(record, event, pid)
    elif not enabled:
        event["decision"] = "disabled"
    elif not prerequisite_ready:
        record.update({"status": "WAITING_PREREQUISITE", "blocker": blocker,
                       "last_observed_at": now_iso()})
        event["decision"] = f"wait_for_{blocker}"
    elif not allocated:
        record.update({"status": "WAITING_EXTERNAL", "blocker": "experiment_gpu_allocation",
                       "last_observed_at": now_iso()})
        event["decision"] = "wait_for_idle_gpus"
    elif int(record.get("attempts", 0)) >= MAX_ATTEMPTS:
        record["status"] = "RETRY_EXHAUSTED"
        event["decision"] = "retry_exhausted"
    elif float(record.get("next_retry_epoch", 0)) > time.time():
        event.update({"decision": "backoff", "next_retry_at": record.get("next_retry_at")})
    else:
        attempt = int(record.get("attempts", 0)) + 1
        gpu_arg = ",".join(map(str, allocated))
        log_path = ROOT / f"report/logs/{group}_queue.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            argv = [str(ENV_PYTHON), str(QUEUE_RUNNER), "--group", group,
                    "--gpus", gpu_arg, "--max-parallel", "8",
                    "--allocation-file", str(HEALTH / "gpu-allocations" / f"{group}.json")]
            if allocation_mode == "provisional_shared":
                argv.append("--provisional-loss-only")
            process = subprocess.Popen(
                argv,
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        retry_epoch = time.time() + BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)]
        record.update({"attempts": attempt, "status": "RUNNING", "pid": process.pid,
                       "gpus": allocated, "allocation_mode": allocation_mode,
                       "provisional": allocation_mode == "provisional_shared",
                       "last_attempt_at": now_iso(), "log": str(log_path),
                       "next_retry_epoch": retry_epoch,
                       "next_retry_at": datetime.fromtimestamp(retry_epoch, timezone.utc).isoformat()})
        event.update({"decision": "started", "attempt": attempt, "pid": process.pid,
                      "gpus": allocated, "max_parallel": len(allocated)})
        queue = {**queue, "status": "IN_PROGRESS"}
    state["updated_at"] = now_iso()
    atomic_json(ACTION_STATE, state)
    return queue, event


def experiment_group_prerequisites(qualification: dict[str, Any], lr_ready: bool,
                                   e1_analysis: dict[str, Any],
                                   provisional_enabled: bool = False
                                   ) -> dict[str, tuple[bool, str]]:
    """Keep the GPU experiment DAG explicit and prevent E1/E2 dependency cycles."""
    environment_allowed = qualification.get("status") == "PASS" or provisional_enabled
    resolved = environment_allowed and lr_ready
    return {
        "e1": (resolved, "current_lr_resolution"),
        "e2": (resolved, "current_lr_resolution"),
    }


def start_e3_orders_if_eligible(stream: dict[str, Any], matrix: dict[str, Any],
                                enabled: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    state = load_json(ACTION_STATE) or {"schema_version": 1, "actions": {}}
    actions = state.setdefault("actions", {})
    record = actions.setdefault("e3_order_materialization", {"attempts": 0, "status": "WAITING"})
    manifest = load_json(E3_ORDER_MANIFEST) or {"status": "PENDING", "outputs": {}}
    event: dict[str, Any] = {"name": "e3_order_materialization"}
    pid = int(record.get("pid", 0))
    try:
        file_pid = int(E3_ORDER_PID.read_text(encoding="utf-8").strip())
        if runner_matches(file_pid, E3_ORDER_MATERIALIZER):
            pid = file_pid
    except (OSError, ValueError):
        pass
    if pid and not runner_matches(pid, E3_ORDER_MATERIALIZER):
        pid = 0
        record.pop("pid", None)
    if manifest.get("status") == "PASS" and len(manifest.get("outputs", {})) == 8:
        record.update({"status": "PASS", "last_observed_at": now_iso()})
        event["decision"] = "already_passed"
    elif pid and runner_matches(pid, E3_ORDER_MATERIALIZER):
        observe_running(record, event, pid)
    elif not enabled:
        event["decision"] = "disabled"
    elif stream.get("status") != "PASS" or matrix.get("status") != "PASS":
        record.update({"status": "WAITING_PREREQUISITE", "blocker": "source_or_matrix",
                       "last_observed_at": now_iso()})
        event["decision"] = "wait_for_source_or_matrix"
    elif int(record.get("attempts", 0)) >= MAX_ATTEMPTS:
        record["status"] = "RETRY_EXHAUSTED"
        event["decision"] = "retry_exhausted"
    elif float(record.get("next_retry_epoch", 0)) > time.time():
        event.update({"decision": "backoff", "next_retry_at": record.get("next_retry_at")})
    else:
        attempt = int(record.get("attempts", 0)) + 1
        log_path = ROOT / "report/logs/e3_order_materialize.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                [shutil.which("python3") or "python3", str(E3_ORDER_MATERIALIZER)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        E3_ORDER_PID.write_text(f"{process.pid}\n", encoding="utf-8")
        retry_epoch = time.time() + BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)]
        record.update({"attempts": attempt, "status": "RUNNING", "pid": process.pid,
                       "last_attempt_at": now_iso(), "log": str(log_path),
                       "next_retry_epoch": retry_epoch,
                       "next_retry_at": datetime.fromtimestamp(retry_epoch, timezone.utc).isoformat()})
        event.update({"decision": "started", "attempt": attempt, "pid": process.pid})
        manifest = {**manifest, "status": "IN_PROGRESS"}
    state["updated_at"] = now_iso()
    atomic_json(ACTION_STATE, state)
    return manifest, event


def start_background_action(*, action_name: str, prerequisite_ready: bool, blocker: str,
                            completion: dict[str, Any], completion_predicate,
                            pid_path: Path, argv: list[str], log_path: Path,
                            enabled: bool, retry_trigger: str | None = None
                            ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run a persistent non-GPU stage with bounded whole-process retries."""
    state = load_json(ACTION_STATE) or {"schema_version": 1, "actions": {}}
    actions = state.setdefault("actions", {})
    record = actions.setdefault(action_name, {"attempts": 0, "status": "WAITING"})
    event: dict[str, Any] = {"name": action_name}
    pid = int(record.get("pid", 0))
    try:
        file_pid = int(pid_path.read_text(encoding="utf-8").strip())
        if process_matches(file_pid, argv):
            pid = file_pid
        elif not process_alive(file_pid):
            pid_path.unlink(missing_ok=True)
    except (OSError, ValueError):
        pass
    if pid and not process_matches(pid, argv):
        pid = 0
        record.pop("pid", None)
    trigger_changed = (
        retry_trigger is not None and record.get("retry_trigger") != retry_trigger
    )
    if trigger_changed and not pid:
        record.update({"attempts": 0, "status": "RETRY_TRIGGERED",
                       "retry_trigger": retry_trigger})
        for key in ("progress_signature", "stale_observations",
                    "next_retry_epoch", "next_retry_at"):
            record.pop(key, None)
    if completion_predicate(completion) and not trigger_changed:
        record.update({"status": "PASS", "last_observed_at": now_iso()})
        for key in ("blocker", "pid", "progress_signature", "stale_observations"):
            record.pop(key, None)
        pid_path.unlink(missing_ok=True)
        event["decision"] = "already_passed"
    elif pid:
        observe_running(record, event, pid)
    elif not enabled:
        event["decision"] = "disabled"
    elif not prerequisite_ready:
        record.update({"status": "WAITING_PREREQUISITE", "blocker": blocker,
                       "last_observed_at": now_iso()})
        event["decision"] = f"wait_for_{blocker}"
    elif int(record.get("attempts", 0)) >= MAX_ATTEMPTS:
        record["status"] = "RETRY_EXHAUSTED"
        event["decision"] = "retry_exhausted"
    elif float(record.get("next_retry_epoch", 0)) > time.time():
        event.update({"decision": "backoff", "next_retry_at": record.get("next_retry_at")})
    else:
        attempt = int(record.get("attempts", 0)) + 1
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
        retry_epoch = time.time() + BACKOFF_SECONDS[min(attempt - 1, len(BACKOFF_SECONDS) - 1)]
        record.update({"attempts": attempt, "status": "RUNNING", "pid": process.pid,
                       "last_attempt_at": now_iso(), "log": str(log_path),
                       "stale_observations": 0, "progress_signature": None,
                       "next_retry_epoch": retry_epoch,
                       "next_retry_at": datetime.fromtimestamp(retry_epoch, timezone.utc).isoformat()})
        record.pop("blocker", None)
        event.update({"decision": "started", "attempt": attempt, "pid": process.pid})
        completion = {**completion, "status": "IN_PROGRESS"}
    state["updated_at"] = now_iso()
    atomic_json(ACTION_STATE, state)
    return completion, event


def tokenization_complete(value: dict[str, Any], expected_count: int,
                          repair_audit_sha256: str | None) -> bool:
    expected = value.get("expected_jobs")
    if (not isinstance(expected, list) or len(expected) != expected_count
            or len(set(expected)) != expected_count):
        return False
    return (
        value.get("status") == "PASS"
        and all((value.get("jobs", {}).get(name) or {}).get("status") == "PASS"
                for name in expected)
        and (repair_audit_sha256 is None
             or value.get("repaired_audit_sha256") == repair_audit_sha256)
    )


def final_tokenization_complete(value: dict[str, Any], repair_audit_sha256: str | None) -> bool:
    return tokenization_complete(value, 12, repair_audit_sha256)


def write_plan(stream: dict[str, Any], contracts: dict[str, Any], preflight: dict[str, Any],
               gpus: list[dict[str, Any]], lr_selection: dict[str, Any],
               qualification: dict[str, Any], pilot_queue: dict[str, Any],
               batch_analysis: dict[str, Any], batch_queue: dict[str, Any],
               mixture_matrix: dict[str, Any], e2_analysis: dict[str, Any],
               e2_queue: dict[str, Any], e1_analysis: dict[str, Any],
               e1_queue: dict[str, Any], e3_analysis: dict[str, Any],
               e3_queue: dict[str, Any], frozen_recipe: dict[str, Any],
               production_canary: dict[str, Any],
               final_plan_audit: dict[str, Any], final_orders: dict[str, Any],
               final_tokenization: dict[str, Any],
               final_stream_audit: dict[str, Any],
               final_overlap: dict[str, Any], production_contract: dict[str, Any]) -> None:
    p0 = "PASS" if stream.get("status") == "PASS" and stream.get("roster_exact") is True else "BLOCKED"
    p1 = "PASS_STATIC" if contracts.get("status") == "PASS" else "FAIL"
    i0 = "PASS" if preflight.get("status") == "ready" else "BLOCKED"
    idle = [row["index"] for row in gpus if row["memory_used_mb"] <= 1024]
    if qualification.get("status") == "PASS":
        p2 = "PASS"
    elif qualification.get("status") in {"IN_PROGRESS", "FAIL"}:
        p2 = qualification["status"]
    else:
        p2 = "READY_TO_RUN" if p1 == "PASS_STATIC" and i0 == "PASS" and len(idle) >= 8 else "PENDING"
    if (final_stream_audit.get("status") == "PASS"
            and final_overlap.get("status") == "PASS"
            and production_contract.get("status") == "PASS"):
        p8_p10 = "PASS"
    elif final_tokenization.get("status") in {
            "IN_PROGRESS", "FAIL", "BLOCKED_STORAGE", "RETRY_WAIT", "RETRY_EXHAUSTED"}:
        p8_p10 = final_tokenization["status"]
    elif final_tokenization.get("status") == "PASS":
        p8_p10 = "TOKENIZATION_PASS"
    elif final_orders.get("status") == "PASS":
        p8_p10 = "ORDERS_PASS"
    elif final_orders.get("status") in {"IN_PROGRESS", "FAIL"}:
        p8_p10 = final_orders["status"]
    else:
        p8_p10 = "PENDING"
    atomic_json(PLAN_PATH, {
        "schema_version": 2, "updated_at": now_iso(),
        "policy": {"poll_seconds": {"short": 900, "long": 7200, "adaptive": True},
                   "retryable_attempts": MAX_ATTEMPTS,
                   "retry_backoff_seconds": list(BACKOFF_SECONDS), "fail_closed": True,
                   "do_not_start_1t_before_scaling_and_mix_freeze": True},
        "stages": [
            {"id": "P0", "name": "100B token stream closure", "status": p0,
             "evidence": str(STREAM_AUDIT.relative_to(ROOT))},
            {"id": "P1", "name": "experiment static closure", "status": p1,
             "checks": {"formal_contracts_pass": contracts.get("passed", 0),
                        "formal_contracts_total": contracts.get("total", 0),
                        "roster_exact": contracts.get("roster_exact", False)}},
            {"id": "I0", "name": "TorchTitan environment", "status": i0,
             "failed_checks": preflight.get("failed_checks", [])},
            {"id": "P2", "name": "numerical/recovery/reshard qualification", "status": p2,
             "depends_on": ["P1", "I0"], "idle_gpu_indices": idle},
            {"id": "P3", "name": "LR and tokens/update pilot",
             "status": ("PASS" if (lr_selection.get("status") == "PASS"
                                    and batch_analysis.get("status") == "PASS") else
                        (batch_queue.get("status")
                         if batch_queue.get("status") in {"IN_PROGRESS", "FAIL"}
                         else ("LR_PASS" if lr_selection.get("status") == "PASS" else
                               (pilot_queue.get("status")
                                if pilot_queue.get("status") in {"IN_PROGRESS", "FAIL"}
                                else "PENDING")))),
             "depends_on": ["P2"], "lr_results_observed": lr_selection.get("observed_results", 0),
             "batch_results_observed": batch_analysis.get("observed_results", 0),
             'batch_depends_on': 'qualified_200m_anchor',
             "batch_queue_status": batch_queue.get("status")},
            {"id": "P4", "name": "E1 scaling",
             "status": ("PASS" if e1_analysis.get("status") == "PASS" else
                        (e1_queue.get("status") if e1_queue.get("status") in {"IN_PROGRESS", "FAIL"}
                         else "PENDING")),
             "depends_on": ["P3"], "results_observed": e1_analysis.get("observed_results", 0)},
            {"id": "P5", "name": "architecture and mixture response",
             "status": ('DEFERRED' if scaling_only(ROOT) else "PASS" if (e2_analysis.get("status") == "PASS"
                                    and e3_analysis.get("status") == "PASS") else
                        ("PASS_ARCHITECTURE" if e2_analysis.get("status") == "PASS" else
                        (e2_queue.get("status") if e2_queue.get("status") in {"IN_PROGRESS", "FAIL"}
                         else (e3_queue.get("status") if e3_queue.get("status") in {"IN_PROGRESS", "FAIL"}
                               else "PENDING")))),
             "training_depends_on": ["P3"], "analysis_depends_on": ["P4"],
             "mixture_matrix_static": mixture_matrix.get("status"),
             "mixture_arms": len(mixture_matrix.get("arms", [])),
             "architecture_results_observed": e2_analysis.get("observed_results", 0),
             "mixture_results_observed": e3_analysis.get("observed_results", 0)},
            {"id": "P6", "name": "freeze scale and mixture",
             "status": "PASS" if frozen_recipe.get("status") == "PASS" else "PENDING",
             "depends_on": ["P4"] if scaling_only(ROOT) else ["P4", "P5"]},
            {"id": "P7", "name": "post-freeze 1T capacity and WSD audit",
             "status": final_plan_audit.get("status", "PENDING"),
             "depends_on": ["P6"], "errors": final_plan_audit.get("errors", [])},
            {"id": "I1", "name": "frozen production eight-GPU throughput canary",
             "status": production_canary.get("status", "PENDING"),
             "depends_on": ["P2", "P6"],
             "projected_1t_days_from_p10": (
                 production_canary.get("evaluation") or {}
             ).get("projected_1t_days_from_p10")},
            {"id": "P8-P10", "name": "1T order, tokenization, independent audit",
             "status": p8_p10,
             "depends_on": ["P7", "I1"],
             "storage_preflight": final_tokenization.get("storage_preflight")},
        ],
    })


def snapshot(auto_bootstrap: bool) -> dict[str, Any]:
    HEALTH.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, sorted(allowed_gpu_indices(HEALTH))))
    env["YUFORMER_TORCHTITAN_ROOT"] = str(SOURCE_ROOT)
    preflight_path = HEALTH / "torchtitan-preflight.json"
    preflight_python = (
        str(ENV_PYTHON) if ENV_PYTHON.is_file()
        else (str(MANAGED_PYTHON312) if MANAGED_PYTHON312.is_file()
              else (shutil.which("python3") or "python3"))
    )
    preflight_run = command([preflight_python, str(PREFLIGHT), "--out", str(preflight_path)], env=env)
    preflight = load_json(preflight_path) or {"status": "blocked", "failed_checks": ["preflight_report_missing"]}
    actions = run_bootstrap_if_eligible(preflight, auto_bootstrap)
    if actions[0].get("decision") == "attempted" and ENV_PYTHON.is_file():
        preflight_run = command([str(ENV_PYTHON), str(PREFLIGHT), "--out", str(preflight_path)], env=env)
        preflight = load_json(preflight_path) or preflight

    contract_path = ROOT / "report/formal_contract_validation_latest.json"
    night_policy = load_json(ROOT / night_research.POLICY) or {}
    if auto_bootstrap and night_policy.get('enabled') is True:
        from prepare_lr_extensions import prepare as repair_extension_contracts
        repair_extension_contracts(ROOT)
        night_batch_search.prepare(ROOT)
    contract_run = command([shutil.which("python3") or "python3", str(CONTRACT_AUDIT), "--output", str(contract_path)])
    contracts = load_json(contract_path) or {"status": "FAIL", "passed": 0, "total": 0}
    e3_run = command([shutil.which("python3") or "python3", str(E3_AUDIT),
                      "--output", str(E3_AUDIT_REPORT)])
    mixture_matrix = load_json(E3_AUDIT_REPORT) or {"status": "FAIL", "arms": []}
    lr_selection, pilot_events = refresh_pilot_results()
    actions.extend(pilot_events)
    batch_analysis, batch_events = refresh_group_results(
        "batch_pilot", BATCH_ROOT, BATCH_SELECTOR, BATCH_SELECTION
    )
    actions.extend(batch_events)
    lr_ready = lr_resolution_is_current(lr_selection, contract_path)
    e1_analysis, e1_events = refresh_group_results("e1", E1_ROOT, E1_ANALYZER, E1_SCALING_FIT)
    e2_analysis, e2_events = (refresh_group_results("e2", E2_ROOT, E2_ANALYZER, E2_SELECTION)
                            if group_enabled(ROOT, 'e2') else ({'status': 'DEFERRED'}, []))
    e3_analysis, e3_events = (refresh_group_results("e3", E3_ROOT, E3_ANALYZER, E3_SELECTION)
                            if group_enabled(ROOT, 'e3') else ({'status': 'DEFERRED'}, []))
    actions.extend(e2_events)
    actions.extend(e1_events)
    actions.extend(e3_events)
    identity_run = command([shutil.which("python3") or "python3", str(EXPERIMENT_IDENTITY_AUDITOR),
                            "--output", str(EXPERIMENT_IDENTITY_AUDIT)], timeout=1800)
    identity_audit = load_json(EXPERIMENT_IDENTITY_AUDIT) or {
        "status": "BLOCKED", "passed": 0, "total": 91, "roster_exact": False,
    }
    actions.extend(quarantine_identity_drift(identity_audit, auto_bootstrap))
    invalidate_unbound_analyses(identity_audit, lr_selection,
                                {"batch_pilot": batch_analysis, "e1": e1_analysis,
                                 "e2": e2_analysis, "e3": e3_analysis})
    if auto_bootstrap:
        plotted = command([shutil.which('python3') or 'python3', str(E1_CURVE_RENDERER),
                           '--analysis', str(E1_SCALING_FIT),
                           '--identity-audit', str(EXPERIMENT_IDENTITY_AUDIT),
                           '--output-dir', str(E1_SCALING_FIT.parent / 'curves')], timeout=180)
        actions.append({'name': 'e1_scaling_curves', 'command_rc': plotted.returncode,
                        'decision': 'updated_or_current' if plotted.returncode == 0 else 'render_failed',
                        'output_tail': (plotted.stderr or plotted.stdout)[-500:]})
    batch_anchor, anchor_events = refresh_batch_anchor(lr_selection, identity_audit, auto_bootstrap)
    actions.extend(anchor_events)
    for group, analysis, path in (('batch_pilot', batch_analysis, BATCH_SELECTION),
                                  ("e2", e2_analysis, E2_SELECTION),
                                  ("e3", e3_analysis, E3_SELECTION)):
        promoted = promote_bound_loss_analysis(ROOT, group, analysis)
        if promoted is not analysis:
            analysis.clear()
            analysis.update(promoted)
            atomic_json(path, promoted)
    finalize_experiment_resolution(lr_selection, batch_analysis)
    # Resolution writes contracts; validate the resulting files before launch.
    contract_run = command([shutil.which('python3') or 'python3', str(CONTRACT_AUDIT), '--output', str(contract_path)])
    contracts = load_json(contract_path) or {'status': 'FAIL'}
    batch_resolution_ready = batch_pilot_resolution_is_current(batch_anchor, contract_path)
    lr_ready = lr_resolution_is_current(lr_selection, contract_path)
    lr_ready = (lr_ready and lr_selection.get("status") == "PASS"
                and analysis_ready(batch_analysis))
    stack_path = ROOT / "report/pretrain_stack_audit_20261021.json"
    stack_run = command([shutil.which("python3") or "python3", str(STACK_AUDIT), "--output", str(stack_path)])
    stack = load_json(stack_path) or {"status": "FAIL", "defects": ["audit output missing"]}
    stream = load_json(STREAM_AUDIT) or {"status": "MISSING"}
    e3_orders, e3_order_event = start_e3_orders_if_eligible(
        stream, mixture_matrix, auto_bootstrap
    )
    actions.append(e3_order_event)
    e3_stream_audit = load_json(E3_STREAM_AUDIT) or {"status": "PENDING", "streams": []}
    e3_repair_sha256 = (
        sha256(E3_STREAM_AUDIT)
        if e3_stream_audit.get("status") == "FAIL" and E3_STREAM_AUDIT.is_file()
        else None
    )
    e3_tokenization = load_json(E3_TOKENIZATION_STATE) or {"status": "PENDING", "jobs": {}}
    e3_tokenization, e3_token_event = start_background_action(
        action_name="e3_tokenization",
        prerequisite_ready=(e3_orders.get("status") == "PASS"
                            and len(e3_orders.get("outputs", {})) == 8),
        blocker="e3_orders", completion=e3_tokenization,
        completion_predicate=lambda value: tokenization_complete(
            value, 17, e3_repair_sha256
        ),
        pid_path=E3_TOKENIZATION_PID,
        argv=[shutil.which("python3") or "python3", str(E3_TOKENIZER), "--workers", "32"],
        log_path=ROOT / "report/logs/e3_tokenization.log", enabled=auto_bootstrap,
        retry_trigger=artifact_trigger(
            E3_ORDER_MANIFEST, salt=e3_repair_sha256 or "no-repair-audit"
        ),
    )
    actions.append(e3_token_event)
    e3_stream_audit, e3_audit_event = start_background_action(
        action_name="e3_stream_audit", prerequisite_ready=e3_tokenization.get("status") == "PASS",
        blocker="e3_tokenization", completion=e3_stream_audit,
        completion_predicate=lambda value: (value.get("status") == "PASS"
                                             and value.get("roster_exact") is True
                                             and len(value.get("streams", [])) == 17),
        pid_path=E3_STREAM_AUDIT_PID,
        argv=[shutil.which("python3") or "python3", str(E3_STREAM_AUDITOR)],
        log_path=ROOT / "report/logs/e3_stream_audit.log", enabled=auto_bootstrap,
        retry_trigger=artifact_trigger(E3_TOKENIZATION_STATE),
    )
    actions.append(e3_audit_event)
    e3_config = load_json(E3_CONFIG_REPORT) or {"status": "PENDING", "generated": []}
    e3_config, e3_config_event = start_background_action(
        action_name="e3_config_generation",
        prerequisite_ready=(e3_stream_audit.get("status") == "PASS"
                            and analysis_ready(e2_analysis) and group_enabled(ROOT, 'e3')),
        blocker="e2_selection_and_e3_stream_audit", completion=e3_config,
        completion_predicate=lambda value: (value.get("status") == "PASS"
                                             and len(value.get("generated", [])) == 8
                                             and all(row.get("contract_status") == "PASS"
                                                     for row in value.get("generated", []))),
        pid_path=E3_CONFIG_PID,
        argv=[shutil.which("python3") or "python3", str(E3_CONFIG_GENERATOR)],
        log_path=ROOT / "report/logs/e3_config_generation.log", enabled=auto_bootstrap,
        retry_trigger=artifact_trigger(
            E2_SELECTION, E3_STREAM_AUDIT,
            ROOT / "experiments/e3_data_mix/configs/e3_matrix.yaml",
            ROOT / 'pretrain/configs/research_budget.yaml',
        ),
    )
    actions.append(e3_config_event)
    policy = resource_policy(HEALTH)
    gpus = [row for row in gpu_snapshot() if int(row['index']) in policy['allowed_gpu_indices']]
    idle = [row["index"] for row in gpus if row["memory_used_mb"] <= 1024]
    shared_policy = active_shared_gpu_policy()
    shared_gpus = set(shared_policy["gpu_indices"]) if shared_policy else set()
    handoff_event = checkpointed_shared_handoff(
        load_json(QUALIFICATION_REPORT) or {"status": "PENDING", "stages": {}},
        gpus, shared_policy, auto_bootstrap,
    )
    actions.append(handoff_event)
    qualification, qualification_event = start_qualification_if_eligible(
        preflight, contracts, gpus, auto_bootstrap, reserved_gpus=shared_gpus
    )
    actions.append(qualification_event)
    expansion = {'status': 'WAIT_FOR_QUALIFICATION', 'new_arms': []}
    batch_expansion = {'status': 'WAIT_FOR_QUALIFICATION', 'new_arms': []}
    if auto_bootstrap and qualification.get('status') == 'PASS' and contracts.get('status') == 'PASS':
        expansion = night_research.expand_lr_search(
            ROOT, lr_selection, bool(discover_group_runner(EXTENSION_GROUP, None)), identity_audit)
        batch_expansion = night_batch_search.expand(
            ROOT, batch_analysis, identity_audit, bool(discover_group_runner(batch_contract.GROUP, None)))
        if expansion.get('status') == 'EXPAND' or batch_expansion.get('status') == 'EXPAND':
            lr_selection, expansion_events = refresh_pilot_results()
            actions.extend(expansion_events)
            contract_run = command([shutil.which('python3') or 'python3', str(CONTRACT_AUDIT), '--output', str(contract_path)])
            contracts = load_json(contract_path) or {'status': 'FAIL'}
            batch_resolution_ready = batch_pilot_resolution_is_current(batch_anchor, contract_path)
    atomic_json(HEALTH / 'night-research-status.json', {
        'observed_at': now_iso(), 'expansion': expansion,
        'batch_expansion': batch_expansion,
        'batch_anchor_status': batch_anchor.get('status'),
        'batch_resolution_current': batch_resolution_ready,
        'policy': night_research.POLICY,
    })
    pilot_queue, pilot_event = start_pilots_if_eligible(
        qualification, lr_selection, gpus, auto_bootstrap, shared_policy
    )
    actions.append(pilot_event)
    extension_complete = experiment_result_roster_complete(EXTENSION_GROUP)
    extension_queue, extension_event = start_group_if_eligible(
        EXTENSION_GROUP,
        bool(lr_extensions(ROOT)) and qualification.get('status') == 'PASS' and contracts.get('status') == 'PASS',
        'declared_lr_extensions_and_qualification',
        {'status': 'PASS' if extension_complete else 'PENDING'},
        gpus, auto_bootstrap, shared_policy=shared_policy,
        qualification_passed=qualification.get('status') == 'PASS',
    )
    actions.append(extension_event)
    performance_contracts = load_json(PERFORMANCE_CONTRACTS) or {"status": "PENDING"}
    if lr_ready and auto_bootstrap:
        prepared = command([shutil.which("python3") or "python3", str(PERFORMANCE_PREPARER),
                            "--root", str(ROOT)])
        performance_contracts = (load_json(PERFORMANCE_CONTRACTS) or {"status": "FAIL"})
        if prepared.returncode:
            performance_contracts = {"status": "FAIL"}
    performance_complete = experiment_result_roster_complete("e1_performance")
    performance_queue, performance_event = start_group_if_eligible(
        "e1_performance", lr_ready and performance_contracts.get("status") == "PASS"
        and qualification.get("status") == "PASS", "resolved_e1_and_qualification",
        {"status": "PASS" if performance_complete else "PENDING"}, gpus, auto_bootstrap,
        shared_policy=None, qualification_passed=qualification.get("status") == "PASS",
    )
    actions.append(performance_event)
    batch_queue, batch_event = start_group_if_eligible(
        "batch_pilot",
        contracts.get("status") == "PASS"
        and (qualification.get("status") == "PASS" or shared_policy is not None)
        and batch_resolution_ready,
        "qualified_200m_anchor_and_batch_pilot_resolution", batch_analysis, gpus, auto_bootstrap,
        shared_policy=shared_policy,
        qualification_passed=qualification.get("status") == "PASS",
    )
    actions.append(batch_event)
    batch_extension_complete = experiment_result_roster_complete(batch_contract.GROUP)
    batch_extension_queue, batch_extension_event = start_group_if_eligible(
        batch_contract.GROUP,
        bool(batch_contract.extensions(ROOT)) and batch_resolution_ready
        and contracts.get('status') == 'PASS' and qualification.get('status') == 'PASS',
        'qualified_batch_anchor_and_declared_extensions',
        {'status': 'PASS' if batch_extension_complete else 'PENDING'},
        gpus, auto_bootstrap, shared_policy=shared_policy,
        qualification_passed=qualification.get('status') == 'PASS')
    actions.append(batch_extension_event)
    group_prerequisites = experiment_group_prerequisites(
        qualification, lr_ready, e1_analysis, shared_policy is not None
    )
    e1_queue, e1_event = start_group_if_eligible(
        "e1", *group_prerequisites["e1"], e1_analysis, gpus, auto_bootstrap,
        shared_policy=shared_policy,
        qualification_passed=qualification.get("status") == "PASS",
    )
    actions.append(e1_event)
    e2_queue, e2_event = start_group_if_eligible(
        "e2", *group_prerequisites["e2"], e2_analysis, gpus, auto_bootstrap,
        shared_policy=shared_policy,
        qualification_passed=qualification.get("status") == "PASS",
    )
    actions.append(e2_event)
    action_state = load_json(ACTION_STATE) or {"actions": {}}
    e1_record = action_state.get("actions", {}).get("e1_queue", {})
    e1_reserved = (set(map(int, e1_record.get("gpus", [])))
                   if e1_record.get("status") == "RUNNING"
                   and runner_matches(int(e1_record.get("pid", 0)), QUEUE_RUNNER) else set())
    e3_queue, e3_event = start_group_if_eligible(
        "e3", e3_config.get("status") == "PASS", "e3_config_generation",
        e3_analysis, gpus, auto_bootstrap, excluded_gpus=e1_reserved,
        shared_policy=shared_policy,
        qualification_passed=qualification.get("status") == "PASS",
    )
    actions.append(e3_event)
    e2_confirmation, e2_confirmation_event = advance_confirmation(
        'e2', e2_analysis, gpus, qualification, auto_bootstrap, shared_policy)
    e3_confirmation, e3_confirmation_event = advance_confirmation(
        'e3', e3_analysis, gpus, qualification, auto_bootstrap, shared_policy)
    actions.extend([e2_confirmation_event, e3_confirmation_event])
    frozen_recipe = load_json(FROZEN_RECIPE) or {"status": "PENDING"}
    frozen_recipe, freeze_event = start_background_action(
        action_name="pretrain_recipe_freeze",
        prerequisite_ready=(contracts.get("status") == "PASS"
                            and qualification.get("status") == "PASS"
                            and e1_analysis.get("status") == "PASS"
                            and ablation_requirements_satisfied(ROOT, e2_analysis, e3_analysis,
                                                               e2_confirmation, e3_confirmation)
                            and e3_stream_audit.get("status") == "PASS"
                            and identity_audit.get("status") == "PASS"),
        blocker="qualification_active_research_scope_and_bound_evidence",
        completion=frozen_recipe,
        completion_predicate=lambda value: value.get("status") == "PASS" and scope_binding_current(ROOT, value),
        pid_path=RECIPE_FREEZE_PID,
        argv=[shutil.which("python3") or "python3", str(RECIPE_FREEZER)],
        log_path=ROOT / "report/logs/pretrain_recipe_freeze.log", enabled=auto_bootstrap,
        retry_trigger=artifact_trigger(
            contract_path, E1_SCALING_FIT, E2_SELECTION, E3_SELECTION, LR_RESOLUTION,
            E3_STREAM_AUDIT, EXPERIMENT_IDENTITY_AUDIT, RECIPE_FREEZER,
            research_confirmation.result_path(ROOT, 'e2'),
            research_confirmation.result_path(ROOT, 'e3'),
            ROOT / 'pretrain/configs/research_budget.yaml',
            ROOT / SCOPE_POLICY,
        ),
    )
    actions.append(freeze_event)
    production_canary = load_json(PRODUCTION_CANARY_REPORT) or {"status": "PENDING"}
    canary_prerequisite = (
        frozen_recipe.get("status") == "PASS"
        and qualification.get("status") == "PASS"
        and preflight.get("status") == "ready"
        and len(idle) >= 8
    )
    production_canary, canary_event = start_background_action(
        action_name="production_throughput_canary",
        prerequisite_ready=canary_prerequisite,
        blocker="frozen_recipe_qualification_environment_and_eight_idle_gpus",
        completion=production_canary,
        completion_predicate=lambda value: production_canary_current(frozen_recipe, value),
        pid_path=PRODUCTION_CANARY_PID,
        argv=[str(ENV_PYTHON), str(PRODUCTION_CANARY_RUNNER)],
        log_path=ROOT / "report/logs/production_throughput_canary.log",
        enabled=auto_bootstrap,
        retry_trigger=artifact_trigger(FROZEN_RECIPE, PRODUCTION_CANARY_RUNNER),
    )
    actions.append(canary_event)
    canary_ready = production_canary_current(frozen_recipe, production_canary)
    effective_production_canary = (
        production_canary if canary_ready
        else {**production_canary, "status": "PENDING_OR_STALE"}
    )
    final_plan_audit = load_json(FINAL_PLAN_AUDIT) or {"status": "PENDING", "errors": []}
    final_plan_audit, final_plan_event = start_background_action(
        action_name="pretrain_1t_plan_audit",
        prerequisite_ready=frozen_recipe.get("status") == "PASS", blocker="frozen_recipe",
        completion=final_plan_audit,
        completion_predicate=lambda value: post_freeze_plan_audit_current(
            frozen_recipe, value
        ),
        pid_path=FINAL_PLAN_AUDIT_PID,
        argv=[shutil.which("python3") or "python3", str(FINAL_PLAN_AUDITOR),
              "--output", str(FINAL_PLAN_AUDIT)],
        log_path=ROOT / "report/logs/pretrain_1t_plan_audit.log",
        enabled=auto_bootstrap,
        retry_trigger=artifact_trigger(FROZEN_RECIPE, FINAL_PLAN_AUDITOR),
    )
    actions.append(final_plan_event)
    final_plan_ready = post_freeze_plan_audit_current(frozen_recipe, final_plan_audit)
    effective_final_plan_audit = (
        final_plan_audit if final_plan_ready
        else {**final_plan_audit, "status": "PENDING_POST_FREEZE"}
    )
    final_orders = load_json(FINAL_ORDER_STATE) or {"status": "PENDING", "jobs": {}}
    final_orders, final_order_event = start_background_action(
        action_name="pretrain_1t_order_materialization",
        prerequisite_ready=final_plan_ready,
        blocker="post_freeze_1t_plan_audit",
        completion=final_orders,
        completion_predicate=lambda value: (value.get("status") == "PASS"
                                             and len(value.get("jobs", {})) == 3),
        pid_path=FINAL_ORDER_PID,
        argv=[shutil.which("python3") or "python3", str(FINAL_ORDER_RUNNER), "--workers", "64"],
        log_path=ROOT / "report/logs/pretrain_1t_order_materialization.log",
        enabled=auto_bootstrap,
        retry_trigger=artifact_trigger(FROZEN_RECIPE, FINAL_PLAN_AUDIT, FINAL_ORDER_RUNNER),
    )
    actions.append(final_order_event)
    final_release = load_json(FINAL_RELEASE) or {"status": "PENDING", "stages": {}}
    final_release, final_release_event = start_background_action(
        action_name="pretrain_1t_release", prerequisite_ready=final_orders.get("status") == "PASS",
        blocker="final_orders", completion=final_release,
        completion_predicate=lambda value: (value.get("status") == "PASS"
                                             and len(value.get("stages", {})) == 3),
        pid_path=FINAL_RELEASE_PID,
        argv=[shutil.which("python3") or "python3", str(FINAL_RELEASE_BUILDER)],
        log_path=ROOT / "report/logs/pretrain_1t_release.log", enabled=auto_bootstrap,
        retry_trigger=artifact_trigger(FINAL_ORDER_STATE, FROZEN_RECIPE,
                                       FINAL_RELEASE_BUILDER),
    )
    actions.append(final_release_event)
    final_stream_audit = load_json(FINAL_STREAM_AUDIT) or {"status": "PENDING", "streams": []}
    repair_audit_sha256 = (
        sha256(FINAL_STREAM_AUDIT)
        if final_stream_audit.get("status") == "FAIL" and FINAL_STREAM_AUDIT.is_file()
        else None
    )
    final_tokenization = load_json(FINAL_TOKENIZATION_STATE) or {"status": "PENDING", "jobs": {}}
    final_tokenization, final_token_event = start_background_action(
        action_name="pretrain_1t_tokenization", prerequisite_ready=final_release.get("status") == "PASS",
        blocker="final_release", completion=final_tokenization,
        completion_predicate=lambda value: final_tokenization_complete(
            value, repair_audit_sha256
        ),
        pid_path=FINAL_TOKENIZATION_PID,
        argv=[shutil.which("python3") or "python3", str(FINAL_TOKENIZER), "--workers", "128"],
        log_path=ROOT / "report/logs/pretrain_1t_tokenization.log", enabled=auto_bootstrap,
        retry_trigger=artifact_trigger(
            FINAL_RELEASE, FINAL_TOKENIZER,
            salt=repair_audit_sha256 or "no-repair-audit"
        ),
    )
    actions.append(final_token_event)
    final_stream_audit, final_audit_event = start_background_action(
        action_name="pretrain_1t_stream_audit",
        prerequisite_ready=final_tokenization.get("status") == "PASS",
        blocker="final_tokenization", completion=final_stream_audit,
        completion_predicate=lambda value: (value.get("status") == "PASS"
                                             and value.get("roster_exact") is True
                                             and len(value.get("streams", [])) == 12),
        pid_path=FINAL_STREAM_AUDIT_PID,
        argv=[shutil.which("python3") or "python3", str(FINAL_STREAM_AUDITOR)],
        log_path=ROOT / "report/logs/pretrain_1t_stream_audit.log", enabled=auto_bootstrap,
        retry_trigger=artifact_trigger(
            FINAL_TOKENIZATION_STATE, FINAL_STREAM_AUDITOR,
            ROOT / "pretrain/scripts/audit_token_stream_release.py",
        ),
    )
    actions.append(final_audit_event)
    final_overlap = load_json(FINAL_OVERLAP_AUDIT) or {"status": "PENDING", "checks": {}}
    final_overlap, final_overlap_event = start_background_action(
        action_name="pretrain_1t_overlap_audit",
        prerequisite_ready=final_stream_audit.get("status") == "PASS",
        blocker="final_stream_audit",
        completion=final_overlap,
        completion_predicate=lambda value: (
            value.get("status") == "PASS"
            and int(value.get("schema_version", 0)) >= 2
            and value.get("checks", {}).get("configured_consumption_exact") is True
        ),
        pid_path=FINAL_OVERLAP_PID,
        argv=[shutil.which("python3") or "python3", str(FINAL_OVERLAP_AUDITOR)],
        log_path=ROOT / "report/logs/pretrain_1t_overlap_audit.log", enabled=auto_bootstrap,
        retry_trigger=artifact_trigger(
            FINAL_STREAM_AUDIT, FINAL_OVERLAP_AUDITOR,
            ROOT / "experiments/e4_curriculum/scripts/overlap_merge.c",
            ROOT / "experiments/e4_curriculum/scripts/overlap_consumed_merge.c",
        ),
    )
    actions.append(final_overlap_event)
    production_contract = load_json(PRODUCTION_CONTRACT_REPORT) or {"status": "PENDING"}
    production_contract, production_contract_event = start_background_action(
        action_name="pretrain_production_contract",
        prerequisite_ready=(final_stream_audit.get("status") == "PASS"
                            and final_overlap.get("status") == "PASS"
                            and final_plan_ready and canary_ready),
        blocker="final_plan_stream_overlap_and_throughput_canary_audits",
        completion=production_contract,
        completion_predicate=lambda value: value.get("status") == "PASS",
        pid_path=PRODUCTION_CONTRACT_PID,
        argv=[shutil.which("python3") or "python3", str(PRODUCTION_CONTRACT_VALIDATOR)],
        log_path=ROOT / "report/logs/pretrain_production_contract.log", enabled=auto_bootstrap,
        retry_trigger=artifact_trigger(
            FROZEN_RECIPE, FINAL_PLAN_AUDIT, FINAL_STREAM_AUDIT, FINAL_OVERLAP_AUDIT,
            PRODUCTION_CANARY_REPORT, PRODUCTION_CONTRACT_VALIDATOR,
        ),
    )
    actions.append(production_contract_event)
    write_plan(stream, contracts, preflight, gpus, lr_selection, qualification,
               pilot_queue, batch_analysis, batch_queue, mixture_matrix,
               e2_analysis, e2_queue, e1_analysis, e1_queue,
               e3_analysis, e3_queue, frozen_recipe, effective_production_canary,
               effective_final_plan_audit,
               final_orders, final_tokenization, final_stream_audit,
               final_overlap, production_contract)

    if stream.get("status") != "PASS":
        phase = "token_stream_audit_blocked"
    elif contracts.get("status") != "PASS" or stack.get("defects"):
        phase = "static_contract_blocked"
    elif preflight.get("status") != "ready":
        phase = "environment_blocked"
    elif production_contract.get("status") == "PASS":
        phase = "pretrain_ready"
    elif any(row.get("name", "").endswith("_queue")
             and row.get("decision") in {"running", "started"} for row in actions):
        phase = "research_experiments_running"
    elif not idle:
        phase = "waiting_for_idle_gpu"
    else:
        phase = "qualification_ready"
    return {
        "schema_version": 2, "observed_at": now_iso(), "phase": phase, "actions": actions,
        "stream_audit": {"path": str(STREAM_AUDIT.relative_to(ROOT)), "status": stream.get("status"),
                         "roster_exact": stream.get("roster_exact")},
        "formal_contracts": {"status": contracts.get("status"), "passed": contracts.get("passed"),
                             "total": contracts.get("total"), "roster_exact": contracts.get("roster_exact"),
                             "command_rc": contract_run.returncode},
        "lr_pilots": {"status": lr_selection.get("status"),
                      'expected_results': lr_selection.get('expected_results'),
                      "observed_results": lr_selection.get("observed_results", 0),
                      "selections": lr_selection.get("selections", {}),
                      "resolution_current": lr_ready},
        'night_research': {'expansion': expansion, 'batch_expansion': batch_expansion,
                           'batch_anchor_status': batch_anchor.get('status'),
                           'batch_resolution_current': batch_resolution_ready},
        "qualification": {"status": qualification.get("status"),
                          "stages": qualification.get("stages", {})},
        "pilot_queue": {"status": pilot_queue.get("status"), "jobs": pilot_queue.get("jobs", {})},
        'pilot_extension_queue': {'status': extension_queue.get('status'),
                                  'jobs': extension_queue.get('jobs', {})},
        'batch_extension_queue': {'status': batch_extension_queue.get('status'),
                                  'jobs': batch_extension_queue.get('jobs', {})},
        "batch_pilot": {"analysis_status": batch_analysis.get("status"),
                        "observed_results": batch_analysis.get("observed_results", 0),
                        "selection": batch_analysis.get("selection"),
                        "queue_status": batch_queue.get("status"),
                        "jobs": batch_queue.get("jobs", {})},
        "e2": {"analysis_status": e2_analysis.get("status"),
               "observed_results": e2_analysis.get("observed_results", 0),
               "selection": e2_analysis.get("selection"),
               "queue_status": e2_queue.get("status"), "jobs": e2_queue.get("jobs", {})},
        "e1": {"analysis_status": e1_analysis.get("status"),
               "observed_results": e1_analysis.get("observed_results", 0),
               "rmse": e1_analysis.get("rmse"),
               "queue_status": e1_queue.get("status"), "jobs": e1_queue.get("jobs", {})},
        "e1_performance": {"config_status": performance_contracts.get("status"),
                           "queue_status": performance_queue.get("status"),
                           "jobs": performance_queue.get("jobs", {}),
                           "evidence_status": (e1_analysis.get("independent_performance") or {}).get("status"),
                           "admission": "idle GPUs only; never use the provisional shared lane"},
        "e3": {"config_status": e3_config.get("status"),
               "analysis_status": e3_analysis.get("status"),
               "observed_results": e3_analysis.get("observed_results", 0),
               "selection": e3_analysis.get("selection"),
               "queue_status": e3_queue.get("status"), "jobs": e3_queue.get("jobs", {})},
        "frozen_recipe": {"status": frozen_recipe.get("status"),
                          "architecture": frozen_recipe.get("architecture_evidence"),
                          "mixture": frozen_recipe.get("mixture_evidence")},
        "production_canary": {"status": effective_production_canary.get("status"),
                               "current": canary_ready,
                               "evaluation": production_canary.get("evaluation", {})},
        "final_plan_audit": {"status": effective_final_plan_audit.get("status"),
                             "post_freeze_current": final_plan_ready,
                             "errors": final_plan_audit.get("errors", []),
                             "warnings": final_plan_audit.get("warnings", [])},
        "experiment_identity_audit": {"status": identity_audit.get("status"),
                                      "passed": identity_audit.get("passed"),
                                      "total": identity_audit.get("total"),
                                      "roster_exact": identity_audit.get("roster_exact"),
                                      "command_rc": identity_run.returncode},
        "final_orders": {"status": final_orders.get("status"),
                         "jobs": final_orders.get("jobs", {})},
        "final_release": {"status": final_release.get("status"),
                          "stages": len(final_release.get("stages", {}))},
        "final_tokenization": {"status": final_tokenization.get("status"),
                               "jobs": final_tokenization.get("jobs", {}),
                               "storage_preflight": final_tokenization.get("storage_preflight")},
        "final_stream_audit": {"status": final_stream_audit.get("status"),
                               "roster_exact": final_stream_audit.get("roster_exact"),
                               "streams": len(final_stream_audit.get("streams", []))},
        "final_overlap_audit": {"status": final_overlap.get("status"),
                                "checks": final_overlap.get("checks", {}),
                                "replayed_fraction": final_overlap.get("replayed_consumption_fraction")},
        "production_contract": {"status": production_contract.get("status"),
                                "identity_sha256": production_contract.get("identity_sha256"),
                                "error": production_contract.get("error")},
        "mixture_matrix": {"status": mixture_matrix.get("status"),
                           "arms": len(mixture_matrix.get("arms", [])),
                           "command_rc": e3_run.returncode},
        "mixture_orders": {"status": e3_orders.get("status"),
                           "arms": len(e3_orders.get("outputs", {}))},
        "mixture_tokenization": {"status": e3_tokenization.get("status"),
                                 "jobs": e3_tokenization.get("jobs", {})},
        "mixture_stream_audit": {"status": e3_stream_audit.get("status"),
                                 "roster_exact": e3_stream_audit.get("roster_exact"),
                                 "streams": len(e3_stream_audit.get("streams", []))},
        "preflight": {"status": preflight.get("status"), "failed_checks": preflight.get("failed_checks", []),
                      "command_rc": preflight_run.returncode},
        "stack": {"status": stack.get("status"), "defects": stack.get("defects", []),
                  "blockers": stack.get("blockers", []), "command_rc": stack_run.returncode},
        'confirmations': {'e2': e2_confirmation, 'e3': e3_confirmation},
        "gpus": gpus, "idle_gpu_indices": idle, "gpu_resource_policy": policy,
        "research_scope": load_scope(ROOT),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=900,
                        help="short/transition poll interval (default: 15 minutes)")
    parser.add_argument("--long-interval", type=int, default=7200,
                        help="stable long-task poll interval (default: 2 hours)")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--auto-bootstrap", action="store_true")
    args = parser.parse_args()
    if (HEALTH / 'pretrain-user-pause.json').exists():
        print('User pause is active; automatic experiment admission is disabled.')
        return 0
    if args.interval < 60 or args.long_interval < args.interval:
        raise SystemExit("interval must be >=60 seconds and long-interval must be >= interval")
    HEALTH.mkdir(parents=True, exist_ok=True)
    lock_handle = (HEALTH / "pretrain-readiness-watch.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another readiness watcher already owns the lock")
    latest = HEALTH / "pretrain-readiness-latest.json"
    history = HEALTH / "pretrain-readiness-history.jsonl"
    wakeup = threading.Event()
    signal.signal(signal.SIGUSR1, lambda *_: wakeup.set())
    signal.signal(signal.SIGHUP, lambda *_: os.execv(sys.executable, [sys.executable, *sys.argv]))
    atomic_json(HEALTH / "pretrain-watch-wakeup.json", {"pid": os.getpid(), "signal": "SIGUSR1"})
    while True:
        next_interval = args.interval
        try:
            result = snapshot(args.auto_bootstrap)
            next_interval = monitor_interval_seconds(result, args.interval, args.long_interval)
            result["monitoring"] = {
                "mode": "long" if next_interval == args.long_interval else "short",
                "next_poll_seconds": next_interval,
            }
            atomic_json(latest, result)
            with history.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        except Exception as exc:
            error = {"observed_at": now_iso(), "type": type(exc).__name__, "message": str(exc)}
            atomic_json(HEALTH / "pretrain-readiness-error-latest.json", error)
            print(f"readiness watcher error: {type(exc).__name__}: {exc}", flush=True)
        if args.once:
            return 0
        wakeup.wait(next_interval)
        wakeup.clear()


if __name__ == "__main__":
    raise SystemExit(main())
