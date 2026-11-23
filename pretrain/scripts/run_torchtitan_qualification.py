#!/usr/bin/env python3
"""Run and resume the four fail-closed TorchTitan qualification gates."""
from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pretrain/src"))
sys.path.insert(0, str(ROOT / "pretrain/scripts"))

from yuformer_pretrain.config import ConfigError  # noqa: E402
from yuformer_pretrain.lease import RunLease  # noqa: E402
from qualification_identity import qualification_identity, report_matches  # noqa: E402
from process_control import terminate_process_tree  # noqa: E402
from artifact_archive import move_artifact  # noqa: E402

BACKEND = ROOT / "pretrain/backends/torchtitan"
CONFIGS = ROOT / "pretrain/configs/qualification"
RUNS = ROOT / "data/experiments/qualification"
REPORT = ROOT / "data/health/torchtitan-qualification.json"
PYTHON = Path("/opt/yuformer-torchtitan/bin/python")
_RUNNER_LEASE: RunLease | None = None
_QUALIFICATION_IDENTITY: dict[str, Any] | None = None
_REQUIRED_STAGES = (
    "lr_schedule_contract", "numerical_parity", "attention_backend_parity", "distributed_gradient_parity",
    "single_gpu", "fsdp4_dcp", "recovery_reference", "train_interrupt_resume",
    "save_interrupt_resume", "load_interrupt_resume", "reshard_4to2",
    "large_model_accumulation",
)


def required_stages_for_gpu_count(gpu_count: int) -> tuple[str, ...]:
    """Require only topology transfers that this host can actually validate."""
    if gpu_count >= 8:
        return (*_REQUIRED_STAGES, "reshard_4to8")
    return _REQUIRED_STAGES


def acquire_runner_lease() -> RunLease:
    """Fence qualification writes across hosts sharing the experiment root."""
    try:
        return RunLease(
            RUNS,
            "torchtitan-qualification",
            heartbeat_seconds=30,
            stale_seconds=10 * 60,
        ).acquire()
    except ConfigError as exc:
        raise SystemExit(f"qualification runner is already active: {exc}") from exc


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if _RUNNER_LEASE is not None:
        _RUNNER_LEASE.assert_owner()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_report() -> dict[str, Any]:
    try:
        value = json.loads(REPORT.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def save_stage(document: dict[str, Any], name: str, status: str, **evidence: Any) -> None:
    document.setdefault("stages", {})[name] = {
        "status": status, "observed_at": now_iso(), **evidence,
    }
    document.update({
        "schema_version": 1, "status": (
            "PASS" if all(document["stages"].get(item, {}).get("status") == "PASS"
                           for item in _REQUIRED_STAGES)
            else ("FAIL" if status == "FAIL" else "IN_PROGRESS")
        ), "updated_at": now_iso(), "required_stages": list(_REQUIRED_STAGES),
        "qualification_identity": _QUALIFICATION_IDENTITY,
    })
    atomic_json(REPORT, document)


def parse_env(path: Path) -> dict[str, str]:
    values = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip() and not raw.lstrip().startswith("#"):
            key, value = raw.split("=", 1)
            values[key] = value
    return values


def pending_save_step(checkpoint_steps: list[int], log_text: str,
                      writing_steps: tuple[int, ...] = ()) -> int | None:
    if not checkpoint_steps:
        return None
    latest = max(checkpoint_steps)
    requested = [int(value) for value in re.findall(
        r"Received async checkpoint request with id=[^\n]*/step-(\d+)(?:\s|$)", log_text
    )]
    # DCP logs id=None when an explicit storage_writer supplies the path.
    # Require actual newly written shards without published metadata as well.
    if re.search(r"Received async checkpoint request with id=None(?:\s|$)", log_text):
        requested.extend(writing_steps)
    return min((step for step in requested if step > latest), default=None)


def shard_write_snapshot(run_dir: Path) -> dict[str, tuple[int, int]]:
    snapshots = {}
    for path in run_dir.glob("checkpoint/step-*/*.distcp"):
        try:
            if path.parent.name.removeprefix("step-").isdigit() and path.is_file():
                info = path.stat()
                snapshots[str(path)] = (info.st_mtime_ns, info.st_size)
        except FileNotFoundError:
            continue
    return snapshots


def active_write_steps(run_dir: Path, initial: dict[str, tuple[int, int]]) -> tuple[int, ...]:
    current = shard_write_snapshot(run_dir)
    return tuple(sorted({int(Path(name).parent.name.removeprefix("step-"))
        for name, stat in current.items() if stat[1] > 0 and initial.get(name) != stat
        and not (Path(name).parent/".metadata").is_file()}))


def saving_interrupt_observed(evidence: dict[str, Any]) -> bool:
    baseline = evidence.get("killed_at_step")
    pending = evidence.get("pending_checkpoint_step")
    return (type(baseline) is int and type(pending) is int and pending > baseline
            and evidence.get("pending_checkpoint_complete_after_kill") is False)


def injection_decision(mode: str, checkpoint_steps: list[int], log_text: str,
                       writing_steps: tuple[int, ...] = ()) -> tuple[bool, int | None]:
    latest = max(checkpoint_steps) if checkpoint_steps else None
    step_numbers = [int(value) for value in re.findall(r"step:\s*(\d+)", log_text)]
    decision = (
        (mode == "checkpoint" and latest is not None)
        or (mode == "training" and latest is not None and step_numbers and max(step_numbers) > latest)
        or (mode == "saving" and pending_save_step(checkpoint_steps, log_text, writing_steps) is not None)
        or (mode == "loading" and latest is not None and "Loading the checkpoint from" in log_text)
    )
    return bool(decision), latest


def launch(name: str, gpu_ids: list[int], *, actual_world: int | None = None,
           timeout: int = 3600, inject_mode: str | None = None) -> tuple[int, dict[str, Any]]:
    env_path = CONFIGS / f"{name}.env"
    values = parse_env(env_path)
    world = actual_world or int(values["GPUS_PER_NODE"])
    values["YUFORMER_OVERRIDE_GPUS_PER_NODE"] = str(world)
    values["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids[:world]))
    if actual_world is not None and actual_world != int(values["GPUS_PER_NODE"]):
        values["YUFORMER_ALLOW_TOPOLOGY_RESHARD"] = "1"
    env = dict(os.environ)
    env.update(values)
    run_dir = Path(values["YUFORMER_PRETRAIN_OUT"])
    run_dir.mkdir(parents=True, exist_ok=True)
    initial_shards = shard_write_snapshot(run_dir) if inject_mode == "saving" else {}
    log_path = run_dir / "qualification.log"
    log_offset = log_path.stat().st_size if log_path.exists() else 0
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now_iso()}] launch world={world} gpus={values['CUDA_VISIBLE_DEVICES']}\n")
        log.flush()
        process = subprocess.Popen(
            ["bash", str(BACKEND / "launch.sh"), str(env_path)], cwd=ROOT, env=env,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        killed_at = None
        pending_checkpoint = None
        pending_complete_after_kill = None
        pending_shards_before_kill = {}
        started = time.monotonic()
        while process.poll() is None:
            if inject_mode:
                checkpoints = sorted(
                    run_dir.glob("checkpoint/step-*/.metadata"),
                    key=lambda path: int(path.parent.name.split("-")[-1]),
                )
                text = log_path.read_text(encoding="utf-8", errors="replace")[log_offset:]
                checkpoint_steps = [int(path.parent.name.split("-")[-1]) for path in checkpoints]
                writing_steps = active_write_steps(run_dir, initial_shards) if inject_mode == "saving" else ()
                should_kill, latest_checkpoint = injection_decision(
                    inject_mode, checkpoint_steps, text, writing_steps
                )
                if should_kill:
                    killed_at = latest_checkpoint
                    if inject_mode == "saving":
                        pending_checkpoint = pending_save_step(checkpoint_steps, text, writing_steps)
                        pending_shards_before_kill = {
                            name: stat for name, stat in shard_write_snapshot(run_dir).items()
                            if Path(name).parent.name == f"step-{pending_checkpoint}"}
                    terminate_process_tree(process.pid, signal.SIGKILL)
                    process.wait(timeout=60)
                    if pending_checkpoint is not None:
                        pending_complete_after_kill = (
                            run_dir / "checkpoint" / f"step-{pending_checkpoint}" / ".metadata"
                        ).is_file()
                    break
            if time.monotonic() - started > timeout:
                terminate_process_tree(process.pid, signal.SIGKILL)
                process.wait(timeout=60)
                return 124, {"log": str(log_path), "reason": "timeout"}
            time.sleep(0.05 if inject_mode else 2)
    return int(process.returncode or 0), {
        "log": str(log_path), "killed_at_step": killed_at, "injection_mode": inject_mode,
        "pending_checkpoint_step": pending_checkpoint,
        "pending_checkpoint_complete_after_kill": pending_complete_after_kill,
        "pending_checkpoint_shards_before_kill": pending_shards_before_kill,
    }


def collect(name: str) -> tuple[int, dict[str, Any]]:
    run_dir = RUNS / name
    result_path = run_dir / "result.json"
    completed = subprocess.run(
        [str(PYTHON), str(ROOT / "pretrain/scripts/collect_torchtitan_result.py"),
         str(run_dir), "--output", str(result_path)], cwd=ROOT, capture_output=True, text=True,
    )
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        result = {"status": "FAIL", "issues": [completed.stderr[-2000:]]}
    return completed.returncode, result


def verify_cursor(name: str, *, source_world: int, resumed_world: int,
                  checkpoint_step: int, local_batch_size: int = 1,
                  grad_accum: int = 1) -> dict[str, Any]:
    expected_consumed = checkpoint_step * source_world * local_batch_size * grad_accum
    paths = sorted((RUNS / name / "cursor_audit").glob(
        f"load-{expected_consumed}-world{resumed_world}-rank*.json"
    ))
    rows = []
    for path in paths:
        try:
            rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
    ranks = sorted(int(row.get("dp_rank", -1)) for row in rows)
    expected_step = expected_consumed // (resumed_world * local_batch_size)
    valid = (
        ranks == list(range(resumed_world))
        and all(int(row.get("consumed_samples", -1)) == expected_consumed for row in rows)
        and all(int(row.get("resumed_dataloader_step", -1)) == expected_step for row in rows)
        and all(int(row.get("local_batch_size", -1)) == local_batch_size for row in rows)
    )
    return {
        "status": "PASS" if valid else "FAIL", "source_world": source_world,
        "resumed_world": resumed_world, "checkpoint_step": checkpoint_step,
        "expected_consumed_samples": expected_consumed,
        "expected_resumed_dataloader_step": expected_step, "observed_ranks": ranks,
        "files": [str(path) for path in paths],
    }


def compare_loss(reference_name: str, resumed_name: str, *, after_step: int,
                 through_step: int = 12, tolerance: float = 0.02) -> dict[str, Any]:
    reference = read_json_result(RUNS / reference_name / "result.json")
    resumed = read_json_result(RUNS / resumed_name / "result.json")
    key = "loss_metrics/global_avg_loss"
    left = {int(point["step"]): float(point["value"])
            for point in reference.get("series", {}).get(key, [])}
    right = {int(point["step"]): float(point["value"])
             for point in resumed.get("series", {}).get(key, [])}
    expected_steps = list(range(after_step + 1, through_step + 1))
    steps = sorted(step for step in set(left) & set(right) if after_step < step <= through_step)
    missing_reference_steps = sorted(set(expected_steps) - set(left))
    missing_resumed_steps = sorted(set(expected_steps) - set(right))
    finite = all(math.isfinite(value) for step in steps for value in (left[step], right[step]))
    deltas = [abs(left[step] - right[step]) for step in steps]
    maximum = max(deltas) if deltas and finite else None
    complete = bool(expected_steps) and not missing_reference_steps and not missing_resumed_steps
    return {"status": "PASS" if complete and finite and maximum <= tolerance else "FAIL",
            "reference": reference_name, "resumed": resumed_name, "steps": steps,
            "expected_steps": expected_steps, "missing_reference_steps": missing_reference_steps,
            "missing_resumed_steps": missing_resumed_steps, "all_losses_finite": finite,
            "max_abs_loss_delta": maximum, "tolerance": tolerance}


def read_json_result(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def main() -> int:
    global _RUNNER_LEASE, _QUALIFICATION_IDENTITY, _REQUIRED_STAGES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default=os.environ.get("YUFORMER_QUALIFICATION_GPUS", "0,1,2,3,4,5,6,7"))
    args = parser.parse_args()
    gpu_ids = [int(value) for value in args.gpus.split(",")]
    if len(gpu_ids) < 4 or len(set(gpu_ids)) != len(gpu_ids) or any(gpu < 0 for gpu in gpu_ids):
        raise SystemExit("qualification requires at least four distinct non-negative GPU indices")
    _REQUIRED_STAGES = required_stages_for_gpu_count(len(gpu_ids))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lease = acquire_runner_lease()
    _RUNNER_LEASE = lease
    # Every ordinary return below releases the lease.  SIGKILL is recovered
    # immediately on the same host; cross-host takeover is heartbeat-gated and
    # fenced by RunLease generations.
    atexit.register(lease.release)
    report = load_report()
    _QUALIFICATION_IDENTITY = qualification_identity(ROOT)
    if report and not report_matches(report, _QUALIFICATION_IDENTITY):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        archive = RUNS / "_stale" / stamp
        archive.mkdir(parents=True, exist_ok=False)
        for path in RUNS.iterdir():
            if path.name not in {".leases", ".run.lease", "run.lock", "_stale"}:
                move_artifact(path, archive / path.name)
        os.replace(REPORT, archive / REPORT.name)
        report = {}

    if report.get("stages", {}).get("lr_schedule_contract", {}).get("status") != "PASS":
        schedule_report = REPORT.parent / "torchtitan-lr-schedule-contract.json"
        completed = subprocess.run([
            str(PYTHON), str(BACKEND / "lr_schedule_contract.py"),
            "--config", str(ROOT / "pretrain/configs/pretrain_base.yaml"),
            "--output", str(schedule_report),
        ], cwd=ROOT, capture_output=True, text=True, timeout=300)
        schedule = read_json_result(schedule_report)
        passed = completed.returncode == 0 and schedule.get("status") == "PASS"
        save_stage(report, "lr_schedule_contract", "PASS" if passed else "FAIL",
                   report=str(schedule_report), returncode=completed.returncode,
                   output_tail=(completed.stderr or completed.stdout)[-2000:])
        if not passed:
            return 1

    if report.get("stages", {}).get("numerical_parity", {}).get("status") != "PASS":
        parity = REPORT.parent / "torchtitan-numerical-parity.json"
        env = dict(os.environ)
        env.update({"CUDA_VISIBLE_DEVICES": str(gpu_ids[0]),
                    "PYTHONPATH": f"{BACKEND}:{ROOT / 'model/src'}"})
        completed = subprocess.run([
            str(PYTHON), str(BACKEND / "numerical_parity.py"),
            "--model-configs", str(ROOT / "model/configs/model_configs.yaml"),
            "--output", str(parity),
        ], cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
        save_stage(report, "numerical_parity", "PASS" if completed.returncode == 0 else "FAIL",
                   report=str(parity), returncode=completed.returncode,
                   output_tail=(completed.stderr or completed.stdout)[-2000:])
        if completed.returncode:
            return 1

    if report.get("stages", {}).get("distributed_gradient_parity", {}).get("status") != "PASS":
        parity = REPORT.parent / "torchtitan-distributed-gradient-parity.json"
        env = dict(os.environ)
        env.update({"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpu_ids[:2])),
                    "PYTHONPATH": f"{BACKEND}:{ROOT / 'model/src'}"})
        completed = subprocess.run([
            str(PYTHON), "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
            str(BACKEND / "distributed_parity.py"), "--output", str(parity),
        ], cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
        save_stage(report, "distributed_gradient_parity", "PASS" if completed.returncode == 0 else "FAIL",
                   report=str(parity), returncode=completed.returncode,
                   output_tail=(completed.stderr or completed.stdout)[-4000:])
        if completed.returncode:
            return 1

    if report.get("stages", {}).get("attention_backend_parity", {}).get("status") != "PASS":
        parity = REPORT.parent / "torchtitan-attention-backend-parity.json"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_ids[0]))
        completed = subprocess.run([
            str(PYTHON), str(BACKEND / "attention_parity.py"), "--output", str(parity),
        ], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
        result = read_json_result(parity)
        passed = completed.returncode == 0 and result.get("status") == "PASS"
        save_stage(report, "attention_backend_parity", "PASS" if passed else "FAIL",
                   report=str(parity), returncode=completed.returncode,
                   output_tail=(completed.stderr or completed.stdout)[-2000:])
        if not passed:
            return 1

    for stage, name, world in (
        ('large_model_accumulation', 'q6_large_accumulation', 1),
        ("single_gpu", "q1_single_gpu", 1),
        ("fsdp4_dcp", "q2_fsdp4_dcp", 4),
        ("recovery_reference", "q3_recovery_reference", 4),
    ):
        if report.get("stages", {}).get(stage, {}).get("status") == "PASS":
            continue
        rc, evidence = launch(name, gpu_ids, actual_world=world)
        collect_rc, result = collect(name) if rc == 0 else (1, {"status": "FAIL"})
        passed = rc == 0 and collect_rc == 0 and result.get("status") == "PASS"
        save_stage(report, stage, "PASS" if passed else "FAIL", launch_returncode=rc,
                   result_status=result.get("status"), **evidence)
        if not passed:
            return 1

    for stage, name, injection_mode in (
        ("train_interrupt_resume", "q3_train_interrupt", "training"),
        ("save_interrupt_resume", "q3_save_interrupt", "saving"),
    ):
        if report.get("stages", {}).get(stage, {}).get("status") == "PASS":
            continue
        marker = RUNS / name / "fault_injected.json"
        killed_at = None
        if not marker.is_file():
            _, injected = launch(name, gpu_ids, actual_world=4, inject_mode=injection_mode)
            killed_at = injected.get("killed_at_step")
            if killed_at is None:
                save_stage(report, stage, "FAIL", reason=f"{injection_mode} interrupt was not observed", **injected)
                return 1
            if injection_mode == "saving" and not saving_interrupt_observed(injected):
                save_stage(report, stage, "FAIL",
                           reason="checkpoint completed before save interruption; retry required", **injected)
                return 1
            atomic_json(marker, {"status": "PASS", "mode": injection_mode,
                                 "injected_at_step": killed_at, "observed_at": now_iso(),
                                 "injection": injected})
        else:
            killed_at = json.loads(marker.read_text(encoding="utf-8"))["injected_at_step"]
        rc, evidence = launch(name, gpu_ids, actual_world=4)
        collect_rc, result = collect(name) if rc == 0 else (1, {"status": "FAIL"})
        cursor = verify_cursor(name, source_world=4, resumed_world=4, checkpoint_step=killed_at)
        parity = compare_loss(
            "q3_recovery_reference", name, after_step=killed_at, through_step=24
        )
        passed = (rc == 0 and collect_rc == 0 and result.get("status") == "PASS"
                  and cursor["status"] == "PASS" and parity["status"] == "PASS")
        evidence["injection_mode"] = injection_mode
        save_stage(report, stage, "PASS" if passed else "FAIL", injected_at_step=killed_at,
                   resumed_world=4, launch_returncode=rc,
                   result_status=result.get("status"), cursor=cursor, loss_parity=parity,
                   fault_evidence=read_json_result(marker).get("injection"),
                   **evidence)
        if not passed:
            return 1

    stage, name = "load_interrupt_resume", "q3_load_interrupt"
    if report.get("stages", {}).get(stage, {}).get("status") != "PASS":
        marker = RUNS / name / "fault_injected.json"
        marker_data: dict[str, Any] = {}
        if marker.is_file():
            marker_data = json.loads(marker.read_text(encoding="utf-8"))
        if not marker_data.get("seed_checkpoint_step"):
            _, seeded = launch(name, gpu_ids, actual_world=4, inject_mode="checkpoint")
            if seeded.get("killed_at_step") is None:
                save_stage(report, stage, "FAIL", reason="could not seed checkpoint for load interrupt", **seeded)
                return 1
            marker_data.update({"seed_checkpoint_step": seeded["killed_at_step"], "observed_at": now_iso()})
            atomic_json(marker, marker_data)
        if not marker_data.get("load_interrupted"):
            _, interrupted = launch(name, gpu_ids, actual_world=4, inject_mode="loading")
            if interrupted.get("killed_at_step") is None:
                save_stage(report, stage, "FAIL", reason="load interrupt was not observed", **interrupted)
                return 1
            marker_data.update({"load_interrupted": True,
                                "load_interrupted_at_step": interrupted["killed_at_step"],
                                "observed_at": now_iso()})
            atomic_json(marker, marker_data)
        rc, evidence = launch(name, gpu_ids, actual_world=4)
        collect_rc, result = collect(name) if rc == 0 else (1, {"status": "FAIL"})
        cursor = verify_cursor(
            name, source_world=4, resumed_world=4,
            checkpoint_step=int(marker_data["seed_checkpoint_step"]),
        )
        parity = compare_loss(
            "q3_recovery_reference", name,
            after_step=int(marker_data["seed_checkpoint_step"]), through_step=24,
        )
        passed = (rc == 0 and collect_rc == 0 and result.get("status") == "PASS"
                  and cursor["status"] == "PASS" and parity["status"] == "PASS")
        save_stage(report, stage, "PASS" if passed else "FAIL", marker=marker_data,
                   resumed_world=4, launch_returncode=rc,
                   result_status=result.get("status"), cursor=cursor, loss_parity=parity,
                   **evidence)
        if not passed:
            return 1

    reshard_cases = [("reshard_4to2", "q4_reshard_4to2", 2)]
    if len(gpu_ids) >= 8:
        reshard_cases.append(("reshard_4to8", "q5_reshard_4to8", 8))
    else:
        save_stage(
            report, "reshard_4to8", "NOT_APPLICABLE_UNAVAILABLE",
            source_world=4, resumed_world=8, required_gpus=8,
            available_gpus=len(gpu_ids),
            reason="the production host has four GPUs; 4-to-8 reshard qualification is out of scope",
        )
    for stage, name, resumed_world in reshard_cases:
        if report.get("stages", {}).get(stage, {}).get("status") == "PASS":
            continue
        marker = RUNS / name / "fault_injected.json"
        killed_at = None
        if not marker.is_file():
            _, injected = launch(name, gpu_ids, actual_world=4, inject_mode="checkpoint")
            killed_at = injected.get("killed_at_step")
            if killed_at is None:
                save_stage(report, stage, "FAIL", reason="no complete checkpoint before reshard", **injected)
                return 1
            atomic_json(marker, {"status": "PASS", "source_world": 4,
                                 "injected_at_step": killed_at, "observed_at": now_iso()})
        else:
            killed_at = json.loads(marker.read_text(encoding="utf-8"))["injected_at_step"]
        if resumed_world > len(gpu_ids):
            save_stage(
                report, stage, "WAITING_RESOURCES", source_world=4,
                resumed_world=resumed_world, injected_at_step=killed_at,
                required_gpus=resumed_world, available_gpus=len(gpu_ids),
                reason=f"waiting for {resumed_world} simultaneously idle GPUs",
            )
            return 0
        rc, evidence = launch(name, gpu_ids, actual_world=resumed_world)
        collect_rc, result = collect(name) if rc == 0 else (1, {"status": "FAIL"})
        cursor = verify_cursor(
            name, source_world=4, resumed_world=resumed_world, checkpoint_step=killed_at
        )
        passed = (rc == 0 and collect_rc == 0 and result.get("status") == "PASS"
                  and cursor["status"] == "PASS")
        save_stage(report, stage, "PASS" if passed else "FAIL", source_world=4,
                   resumed_world=resumed_world, injected_at_step=killed_at,
                   launch_returncode=rc, result_status=result.get("status"), cursor=cursor,
                   **evidence)
        if not passed:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
