#!/usr/bin/env python3
"""Run a group of one-GPU TorchTitan experiments with bounded retries."""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pretrain/src"))
from yuformer_pretrain.lease import RunLease
from yuformer_pretrain.config import ConfigError
from artifact_archive import move_artifact
from process_control import terminate_process_tree
from process_watchdog import enable_subreaper, reap_descendants, observe_progress, cleanup_attempt_processes
from queue_control import read_allocation, notify_watcher
from gpu_resource_policy import allowed_gpu_indices, resource_policy
from research_scope import group_enabled
LAUNCH = ROOT / "pretrain/backends/torchtitan/launch.sh"
COLLECT = ROOT / "pretrain/scripts/collect_torchtitan_result.py"
PYTHON = Path("/opt/yuformer-torchtitan/bin/python")
BACKOFF = (60, 300, 1800)
_QUEUE_LEASE = None


def checkpoint_transport_environment() -> tuple[dict[str, str], dict[str, Any]]:
    path = ROOT / 'data/health/dcp-transport-qualification.json'
    try:
        value = json.loads(path.read_text())
        if (value.get('status') == 'PASS' and value.get('mode') == 'existing_rendezvous_prefix_store'
                and value.get('concurrent_single_rank_jobs') == 8
                and value.get('multi_rank_world_size') == 2):
            for item in value.get('evidence', []):
                source = ROOT / item['path']
                if hashlib.sha256(source.read_bytes()).hexdigest() != item['sha256']:
                    return {}, {'mode': 'default_free_port', 'reason': 'qualification_evidence_drift'}
            if len(value.get('evidence', [])) != 2:
                return {}, {'mode': 'default_free_port', 'reason': 'qualification_evidence_missing'}
            return {'DCP_USE_PREFIX_STORE': '1'}, {
                'mode': value['mode'], 'qualification_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return {}, {'mode': 'default_free_port', 'reason': 'prefix_store_not_qualified'}


def gpu_compute_pids(gpu: int) -> set[int] | None:
    try:
        probe = subprocess.run([
            "nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if probe.returncode:
        return None
    try:
        return {int(line.strip()) for line in probe.stdout.splitlines() if line.strip()}
    except ValueError:
        return None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if _QUEUE_LEASE is not None:
        _QUEUE_LEASE.assert_owner()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_env(path: Path) -> dict[str, str]:
    values = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            values[key] = value
    return values


def read_result(run_dir: Path) -> dict[str, Any] | None:
    try:
        value = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def collect_result(run_dir: Path) -> bool:
    try:
        result = subprocess.run([str(PYTHON), str(COLLECT), str(run_dir)], cwd=ROOT,
                                capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired) as exc:
        atomic_json(run_dir / 'result_collection_error.json',
                    {'status': 'FAIL', 'observed_at': now_iso(), 'error': str(exc)})
        return False


def current_attempt_matches(run_dir: Path, attempt_id: str) -> bool:
    try:
        marker = json.loads((run_dir / "current_attempt.json").read_text(encoding="utf-8"))
        completion = json.loads((run_dir / "training_complete.json").read_text(encoding="utf-8"))
        return (bool(attempt_id) and marker.get("attempt_id") == attempt_id
                and completion.get("attempt_id") == attempt_id)
    except (OSError, ValueError):
        return False


def result_is_current_pass(run_dir: Path, result: dict[str, Any] | None) -> bool:
    if not result or result.get("status") != "PASS":
        return False
    attempt_id = str(result.get("attempt_id", ""))
    try:
        context = json.loads((run_dir / "execution_context.json").read_text())
        if context.get("performance_probe") and (
                context.get("contention_detected") is not False
                or int(context.get("contention_checks", 0)) < 2):
            return False
    except FileNotFoundError:
        pass
    except (OSError, ValueError, TypeError):
        return False
    return current_attempt_matches(run_dir, attempt_id)


def process_alive(pid: int) -> bool:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
        return len(fields) > 2 and fields[2] != "Z"
    except (OSError, ValueError):
        return False


def launch_process_matches(pid: int, env_path: Path) -> bool:
    if not process_alive(pid):
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return os.fsencode(str(LAUNCH)) in command and os.fsencode(str(env_path)) in command


def renew_retry_generation(job: dict[str, Any]) -> bool:
    """Allow an outer supervisor restart to perform real bounded retries."""
    if job.get('failure_kind') == 'nonfinite_gradient':
        return False
    if job.get("status") not in {"RETRY_EXHAUSTED", "FAIL"} or int(job.get("attempts", 0)) < len(BACKOFF):
        return False
    job.update({"status": "PENDING", "attempts": 0,
                "retry_generation": int(job.get("retry_generation", 0)) + 1,
                "recovery_reason": "outer_supervisor_retry"})
    job.pop("next_retry_epoch", None)
    job.pop("next_retry_at", None)
    return True


def adopted_gpu_reservations(jobs: dict[str, dict[str, Any]],
                             locally_running: set[str]) -> set[int]:
    """Reserve GPUs used by exact live launchers inherited after queue restart."""
    reserved: set[int] = set()
    for name, job in jobs.items():
        if job.get("status") != "RUNNING" or name in locally_running:
            continue
        try:
            pid, gpu = int(job["pid"]), int(job["gpu"])
            env_path = Path(str(job["env"]))
        except (KeyError, TypeError, ValueError):
            continue
        if launch_process_matches(pid, env_path):
            reserved.add(gpu)
    return reserved


def adopted_process_groups(jobs: dict[str, dict[str, Any]],
                           locally_running: set[str]) -> set[int]:
    """Return exact live adopted launcher process groups for signal forwarding."""
    groups: set[int] = set()
    for name, job in jobs.items():
        if job.get("status") != "RUNNING" or name in locally_running:
            continue
        try:
            pid = int(job["pid"])
            env_path = Path(str(job["env"]))
        except (KeyError, TypeError, ValueError):
            continue
        if launch_process_matches(pid, env_path):
            groups.add(pid)
    return groups


def gpu_slot_root(gpu: int) -> Path:
    return ROOT / "data/health/gpu-slots" / socket.gethostname() / f"gpu-{gpu}"


def gpu_slot_busy(gpu: int) -> bool:
    lease = RunLease(gpu_slot_root(gpu), "read-only", stale_seconds=600)
    generations = lease._generations()
    if not generations:
        return False
    owner, age, released = lease._owner(max(generations)[1])
    return not released and not lease._same_host_owner_dead(owner) and age < 600


def training_command(gpu: int, name: str, env_path: Path) -> list[str]:
    return [str(PYTHON), str(ROOT / "pretrain/scripts/run_with_lease.py"),
            "--out-dir", str(gpu_slot_root(gpu)), "--run-name", f"gpu-{gpu}:{name}",
            "--", "env", "-u", "YUFORMER_RUN_LEASE_HELD", "-u", "YUFORMER_RUN_LEASE_TOKEN",
            "-u", "YUFORMER_RUN_LEASE_GENERATION", "bash", str(LAUNCH), str(env_path)]


def launch_admitted(gpu: int, group: str, allocation: dict | None) -> bool:
    if not group_enabled(ROOT, group):
        return False
    policy = resource_policy(ROOT / 'data/health')
    if gpu not in policy['allowed_gpu_indices']:
        return False
    if gpu_slot_busy(gpu):
        return False
    if not policy['allow_busy_gpu_sharing']:
        return gpu_compute_pids(gpu) == set()
    if group != "e1_performance" and (allocation is None or gpu in allocation.get("shared_gpus", [])):
        return True
    return gpu_compute_pids(gpu) == set()


def estimated_work(job: dict) -> float:
    """Start expensive pending arms first without changing their contracts."""
    try:
        import yaml
        from analyze_e2_results import flops_per_token
        values = parse_env(Path(job['env']))
        path = Path(values['YUFORMER_CONFIG'])
        if not path.is_absolute():
            path = ROOT / 'pretrain' / path
        cfg = yaml.safe_load(path.read_text())
        presets = yaml.safe_load(Path(values['YUFORMER_MODEL_CONFIGS']).read_text())['presets']
        cost = flops_per_token(presets[cfg['model']['preset']], cfg['data']['seq_len'])
        return float(cfg['train']['budget_tokens']) * cost * (4 / 3 if cfg['train'].get('activation_checkpointing') else 1)
    except (OSError, ValueError, KeyError, TypeError):
        return 0.


def main() -> int:
    global _QUEUE_LEASE
    if (ROOT / 'data/health/pretrain-user-pause.json').exists():
        print('User pause is active; no new queue or retry will be started.')
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", required=True,
                        choices=("e1_pilot", "batch_pilot", "e1", "e1_seedrep", "e2", "e3", "e1_performance",
                                 'e2_confirm', 'e3_confirm', 'e1_pilot_extension', 'batch_pilot_extension'))
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--allocation-file", type=Path,
                        help="watcher-owned live GPU allocation; permits growth without interrupting training")
    parser.add_argument(
        "--include",
        help="comma-separated arm names for a bounded partial queue run",
    )
    parser.add_argument(
        "--provisional-loss-only", action="store_true",
        help="mark results as loss evidence and exclude contended performance metrics",
    )
    args = parser.parse_args()
    if not group_enabled(ROOT, args.group):
        print(f'{args.group} is deferred by the current research scope.')
        return 0
    subreaper_enabled = enable_subreaper()
    if args.group == "e1_performance" and args.provisional_loss_only:
        raise SystemExit("performance canaries cannot run in shared loss-only mode")
    try:
        _QUEUE_LEASE = RunLease(
            ROOT / "data/health/queue-leases" / args.group,
            f"queue:{args.group}", stale_seconds=600,
        ).acquire()
    except ConfigError as exc:
        raise SystemExit(f"queue already has an active writer: {exc}") from exc
    atexit.register(_QUEUE_LEASE.release)
    gpu_ids = [int(value) for value in args.gpus.split(",")]
    if not set(gpu_ids).issubset(allowed_gpu_indices(ROOT / 'data/health')):
        raise SystemExit('Requested GPUs exceed the user resource policy')
    if len(set(gpu_ids)) != len(gpu_ids) or any(value < 0 for value in gpu_ids):
        raise SystemExit("GPU allocation must contain unique non-negative device indices")
    width = min(args.max_parallel, len(gpu_ids))
    if width < 1:
        raise SystemExit("at least one GPU is required")
    all_env_paths = sorted((ROOT / "pretrain/configs" / args.group).glob("*.env"))
    declared = {path.stem for path in all_env_paths}
    requested = ({name.strip() for name in args.include.split(",") if name.strip()}
                 if args.include else declared)
    unknown = sorted(requested - declared)
    if unknown:
        raise SystemExit(f"unknown {args.group} arm(s): {','.join(unknown)}")
    if not requested:
        raise SystemExit("--include must name at least one arm")
    env_paths = [path for path in all_env_paths if path.stem in requested]
    target_names = {path.stem for path in env_paths}
    state_path = ROOT / f"data/health/{args.group}-queue.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {"schema_version": 1, "group": args.group, "jobs": {}}
    jobs = state.setdefault("jobs", {})
    state['subreaper_enabled'] = subreaper_enabled
    state['watchdog_policy'] = {'poll_seconds': 60, 'stall_seconds': 1800,
                                'startup_seconds': 3600, 'adaptive_log_interval_multiplier': 3}
    retired = state.setdefault("retired_jobs", {})
    for name in sorted(set(jobs) - declared):
        retired[name] = {**jobs.pop(name), "retired_at": now_iso(),
                         "reason": "contract_removed_from_formal_roster"}
    for path in env_paths:
        job = jobs.setdefault(path.stem, {"status": "PENDING", "attempts": 0, "env": str(path)})
        run_dir = Path(parse_env(path)["YUFORMER_PRETRAIN_OUT"])
        if args.group == "e1_performance" and job.get("status") == "RUNNING":
            # A detached probe has an unobserved contention interval. Restart
            # this short measurement from scratch instead of inheriting it.
            pid = int(job.get("pid", 0))
            if launch_process_matches(pid, path):
                terminate_process_tree(pid)
            if run_dir.exists():
                move_artifact(run_dir, run_dir.parent / "_unmonitored" / f"{path.stem}-{time.time_ns()}")
            job.update(status="PENDING", recovery_reason="performance_monitor_interrupted")
        result = read_result(run_dir)
        if result_is_current_pass(run_dir, result):
            job["status"] = "PASS"
        elif job.get("status") == "PASS":
            # A supervisor may quarantine stale evidence after config/source/data
            # drift. PASS without its bound result must re-enter the queue.
            job.update({"status": "PENDING", "attempts": 0,
                        "retry_generation": int(job.get("retry_generation", 0)) + 1,
                        "recovery_reason": "pass_evidence_missing"})
        elif job.get("status") in {"RETRY_EXHAUSTED", "FAIL"} and renew_retry_generation(job):
            pass
        elif job.get("status") == "RUNNING" and not launch_process_matches(
                int(job.get("pid", 0)), Path(job["env"])):
            # The queue may itself be restarted after a crash or host reboot. A stale
            # RUNNING record must re-enter bounded retry rather than deadlock forever.
            if int(job.get("attempts", 0)) >= len(BACKOFF):
                job.update({"status": "RETRY_EXHAUSTED", "finished_at": now_iso(),
                            "recovery_reason": "stale_running_process"})
                renew_retry_generation(job)
            else:
                job.update({"status": "RETRY_WAIT", "next_retry_epoch": time.time(),
                            "next_retry_at": now_iso(),
                            "recovery_reason": "stale_running_process"})
    running: dict[int, tuple[str, subprocess.Popen, Any, Path]] = {}
    pending_reaps: list[int] = []

    def stop(_signum, _frame):
        locally_running = {name for name, _, _, _ in running.values()}
        for _, process, log_handle, _ in running.values():
            if process.poll() is None:
                pending_reaps.extend(terminate_process_tree(process.pid))
                process.wait(timeout=30)
            log_handle.close()
        # A restarted queue adopts launchers that are no longer represented by
        # a local Popen.  Forward termination to those exact, command-verified
        # process groups as well so a resource handoff cannot orphan training.
        for pid in adopted_process_groups(jobs, locally_running):
            pending_reaps.extend(terminate_process_tree(pid))
        reap_descendants([pid for pid in pending_reaps if pid not in
                          {process.pid for _, process, _, _ in running.values()}])
        sys.exit(128 + int(_signum))

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    while True:
        pending_reaps = reap_descendants(pending_reaps)
        allocation = None
        if args.allocation_file:
            try:
                allocation = read_allocation(args.allocation_file, args.group,
                                             allowed_gpu_indices(ROOT / 'data/health'))
                gpu_ids = allocation["gpus"]
                width = min(args.max_parallel, allocation["max_parallel"])
                state["allocation"] = allocation
                state.pop("allocation_error", None)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                width = 0
                state["allocation_error"] = str(exc)
        if (ROOT / 'data/health/pretrain-user-pause.json').exists():
            width = 0
        # A restarted queue cannot adopt the old Popen object, but it can
        # observe the exact launch command. Once that process exits, collect
        # its committed evidence before deciding whether to retry.
        locally_running = {name for name, _, _, _ in running.values()}
        for name, job in sorted(jobs.items()):
            if job.get("status") != "RUNNING" or name in locally_running:
                continue
            env_path = Path(job["env"])
            if launch_process_matches(int(job.get("pid", 0)), env_path):
                reason = observe_progress(job, Path(parse_env(env_path)['YUFORMER_PRETRAIN_OUT']))
                if reason is None:
                    continue
                pending_reaps.extend(terminate_process_tree(int(job['pid'])))
                job.update(recovery_reason=reason, last_watchdog_reason=reason, last_watchdog_failure=now_iso())
            run_dir = Path(parse_env(env_path)["YUFORMER_PRETRAIN_OUT"])
            pending_reaps.extend(cleanup_attempt_processes(job, run_dir))
            attempt_id = str(job.get("attempt_id", ""))
            if current_attempt_matches(run_dir, attempt_id):
                collect_result(run_dir)
            result = read_result(run_dir)
            if result_is_current_pass(run_dir, result):
                job.update({"status": "PASS", "returncode": 0,
                            "finished_at": now_iso(),
                            "recovery_reason": "adopted_process_completed"})
            elif int(job.get("attempts", 0)) >= len(BACKOFF):
                job.update({"status": "RETRY_EXHAUSTED", "finished_at": now_iso(),
                            "recovery_reason": "adopted_process_failed"})
            else:
                job.update({"status": "RETRY_WAIT", "next_retry_epoch": time.time(),
                            "next_retry_at": now_iso(),
                            "recovery_reason": "adopted_process_failed"})
            state["updated_at"] = now_iso()
            atomic_json(state_path, state)
            notify_watcher(ROOT)

        for gpu, (name, process, log_handle, run_dir) in list(running.items()):
            rc = process.poll()
            if rc is None:
                reason = observe_progress(jobs[name], run_dir)
                if reason is not None:
                    victims = terminate_process_tree(process.pid)
                    pending_reaps.extend(pid for pid in victims if pid != process.pid)
                    process.wait(timeout=30)
                    rc = process.returncode
                    jobs[name].update(recovery_reason=reason, last_watchdog_reason=reason, last_watchdog_failure=now_iso())
            if args.group == "e1_performance" and rc is None:
                context_path = run_dir / "execution_context.json"
                context = json.loads(context_path.read_text())
                processes = gpu_compute_pids(gpu)
                context["contention_checks"] = int(context.get("contention_checks", 0)) + 1
                if processes is None or len(processes) > 1:
                    context["contention_detected"] = True
                    jobs[name]["performance_contaminated"] = True
                    pending_reaps.extend(pid for pid in terminate_process_tree(process.pid)
                                         if pid != process.pid)
                    process.wait(timeout=60)
                    rc = process.returncode
                atomic_json(context_path, context)
            if rc is None:
                continue
            pending_reaps.extend(pid for pid in cleanup_attempt_processes(jobs[name], run_dir)
                                 if pid != process.pid)
            log_handle.close()
            if args.group == "e1_performance" and jobs[name].get("performance_contaminated"):
                target = run_dir.parent / "_contended" / f"{name}-{time.time_ns()}"
                move_artifact(run_dir, target)
                jobs[name]["contaminated_archive"] = str(target)
            result = None
            if rc == 0:
                if collect_result(run_dir):
                    result = read_result(run_dir)
            job = jobs[name]
            if rc == 4 and gpu_slot_busy(gpu):
                job.update(status="RETRY_WAIT", attempts=max(0, int(job["attempts"]) - 1),
                           next_retry_epoch=time.time() + 30,
                           recovery_reason="gpu_slot_owned_by_another_live_launcher")
                del running[gpu]
                continue
            if rc == 0 and result_is_current_pass(run_dir, result):
                job.update({"status": "PASS", "returncode": 0, "finished_at": now_iso()})
            else:
                attempt = int(job["attempts"])
                try:
                    with (run_dir / 'queue.log').open('rb') as failure_log:
                        failure_log.seek(0, 2)
                        failure_log.seek(max(0, failure_log.tell() - 32768))
                        failure_tail = failure_log.read()
                        marker = f"attempt={job.get('attempt_id', '')}".encode()
                        if marker in failure_tail:
                            failure_tail = failure_tail.rsplit(marker, 1)[-1]
                        nonfinite = b'non-finite gradients; optimizer update refused' in failure_tail
                except OSError:
                    nonfinite = False
                if nonfinite:
                    job['failure_kind'] = 'nonfinite_gradient'
                if attempt >= len(BACKOFF) or (nonfinite and attempt >= 2):
                    job.update({"status": "RETRY_EXHAUSTED", "returncode": rc,
                                "finished_at": now_iso()})
                else:
                    next_epoch = time.time() + BACKOFF[attempt - 1]
                    job.update({"status": "RETRY_WAIT", "returncode": rc,
                                "next_retry_epoch": next_epoch,
                                "next_retry_at": datetime.fromtimestamp(next_epoch, timezone.utc).isoformat()})
            del running[gpu]
            state["updated_at"] = now_iso()
            atomic_json(state_path, state)
            notify_watcher(ROOT)

        terminal = {"PASS", "RETRY_EXHAUSTED"}
        if all(jobs[name]["status"] in terminal for name in target_names) and not running:
            break

        adopted_gpus = adopted_gpu_reservations(jobs, locally_running)
        free_gpus = [gpu for gpu in gpu_ids[:width]
                     if gpu not in running and gpu not in adopted_gpus]
        free_gpus = [gpu for gpu in free_gpus if launch_admitted(gpu, args.group, allocation)]
        ready = [
            (name, job) for name, job in sorted(jobs.items()) if name in target_names
            if job["status"] in {"PENDING", "RETRY_WAIT"}
            and float(job.get("next_retry_epoch", 0)) <= time.time()
            and int(job.get("attempts", 0)) < len(BACKOFF)
        ]
        ready.sort(key=lambda item: -estimated_work(item[1]))
        for gpu, (name, job) in zip(free_gpus, ready):
            if gpu_slot_busy(gpu):
                continue
            env_path = Path(job["env"])
            values = parse_env(env_path)
            run_dir = Path(values["YUFORMER_PRETRAIN_OUT"])
            if args.group == "e1_performance" and (run_dir / "run_identity.json").exists():
                lease = RunLease(run_dir, "restart-short-performance-probe").acquire()
                lease.release()
                move_artifact(run_dir, run_dir.parent / "_restarted" / f"{name}-{time.time_ns()}")
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "queue.log"
            log_handle = log_path.open("a", encoding="utf-8")
            env = dict(os.environ)
            env.update(values)
            transport_env, checkpoint_transport = checkpoint_transport_environment()
            env.update(transport_env)
            attempt_id = secrets.token_hex(16)
            log_handle.write(f'\nYUFORMER_QUEUE_ATTEMPT attempt={attempt_id}\n')
            log_handle.flush()
            loss_only = (bool(args.provisional_loss_only) if allocation is None else
                         args.group != "e1_performance")
            atomic_json(run_dir / "execution_context.json", {
                "schema_version": 1, "attempt_id": attempt_id,
                'checkpoint_transport': checkpoint_transport,
                "loss_only": loss_only,
                "allocation_mode": allocation.get("mode") if allocation else "fixed_cli",
                "performance_probe": args.group == "e1_performance",
                "idle_admission": args.group == "e1_performance",
                "contention_detected": False,
                "contention_checks": 0,
                "isolation_policy": "empty GPU admission; at most one compute PID sampled every five seconds",
                "performance_metrics_excluded": (
                    ["throughput(tps)", "mfu(%)", "wall_time"]
                    if loss_only else []
                ),
                "recorded_at": now_iso(),
            })
            env.update({"CUDA_VISIBLE_DEVICES": str(gpu), "YUFORMER_OVERRIDE_GPUS_PER_NODE": "1",
                        "YUFORMER_ATTEMPT_ID": attempt_id})
            process = subprocess.Popen(
                training_command(gpu, name, env_path),
                cwd=ROOT, env=env,
                stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True,
            )
            job.update({"status": "RUNNING", "attempts": int(job["attempts"]) + 1,
                        "performance_contaminated": False,
                        "total_attempts": int(job.get("total_attempts", 0)) + 1,
                        "pid": process.pid, "gpu": gpu, "attempt_id": attempt_id,
                        "provisional_loss_only": loss_only,
                        "started_at": now_iso(), "log": str(log_path)})
            job.pop('watchdog', None)
            running[gpu] = (name, process, log_handle, run_dir)
        state["status"] = "IN_PROGRESS"
        state["updated_at"] = now_iso()
        atomic_json(state_path, state)
        time.sleep(5)

    target_pass = all(jobs[name]["status"] == "PASS" for name in target_names)
    for _ in range(10):
        pending_reaps = reap_descendants(pending_reaps)
        if not pending_reaps:
            break
        time.sleep(.02)
    state['pending_reap_pids'] = pending_reaps
    state["status"] = (
        "PASS" if target_pass and target_names == declared
        else "PARTIAL_PASS" if target_pass
        else "FAIL"
    )
    state["target_jobs"] = sorted(target_names)
    state["updated_at"] = now_iso()
    atomic_json(state_path, state)
    notify_watcher(ROOT)
    return 0 if state["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
