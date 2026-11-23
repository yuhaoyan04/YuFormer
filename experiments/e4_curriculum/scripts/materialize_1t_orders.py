#!/usr/bin/env python3
"""Run the frozen three-stage S6 materialization with resumable retries."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "data-process/src"))
from yuformer_data.scale.materialize import config_echo, load_mix_config  # noqa: E402
FREEZE = ROOT / "experiments/e4_curriculum/results/frozen_1t_recipe.json"
BASE_RELEASE = ROOT / "data/processed/releases/mix_m3_e1_100b-20261019.release.json"
BASE_MANIFEST = ROOT / "data/processed/mix_order/mix_m3_e1_100b.manifest.json"
STATE = ROOT / "data/health/pretrain-1t-order-materialization.json"
STAGES = ("warmup", "stable", "anneal")
BACKOFF = (300, 1800, 7200)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def upstream_snapshot_identity(manifest: dict[str, Any]) -> str | None:
    identity = manifest.get("upstream_snapshot_sha256")
    snapshot = manifest.get("upstream_snapshot") or {}
    return identity if isinstance(identity, str) and snapshot.get("snapshot_sha256") == identity else None


def base_upstream_snapshot(root: Path) -> str:
    release = json.loads((root / BASE_RELEASE.relative_to(ROOT)).read_text(encoding="utf-8"))
    manifest_path = root / BASE_MANIFEST.relative_to(ROOT)
    relative = str(manifest_path.relative_to(root))
    records = [row for row in release.get("artifacts", []) if row.get("path") == relative]
    if (release.get("status") != "PASS" or len(records) != 1 or not manifest_path.is_file()
            or int(records[0].get("bytes", -1)) != manifest_path.stat().st_size
            or records[0].get("sha256") != sha256(manifest_path)):
        raise ValueError("base release does not bind the current S6 manifest")
    identity = upstream_snapshot_identity(json.loads(manifest_path.read_text(encoding="utf-8")))
    if identity is None:
        raise ValueError("base S6 manifest has an unclosed upstream snapshot")
    for row in release.get("artifacts", []):
        relative = row.get("path", "")
        if not relative.startswith("data/processed/val_sets/"):
            continue
        path = root / relative
        if (not path.is_file() or path.stat().st_size != int(row.get("bytes", -1))
                or sha256(path) != row.get("sha256")):
            raise ValueError(f"base validation artifact drift before S6: {relative}")
    return identity


def output_valid(root: Path, config_path: Path) -> tuple[bool, dict[str, Any] | None]:
    cfg = load_mix_config(config_path)
    mix_name = cfg["mix_name"]
    manifest_path = root / f"data/processed/mix_order/{mix_name}.manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, None
    order_paths = [root / "data/processed/mix_order" / name
                   for name in manifest.get("order_tables", [])]
    artifacts = {
        str(row.get("path")): row for row in manifest.get("published_artifacts", [])
        if isinstance(row, dict)
    }
    members = root / "data/processed" / str(manifest.get("members", ""))

    def artifact_current(path: Path) -> bool:
        try:
            row = artifacts[str(path.relative_to(root / "data/processed").as_posix())]
            stat = path.stat()
            return (stat.st_size, stat.st_dev, stat.st_ino, stat.st_mtime_ns) == (
                int(row["bytes"]), int(row["device"]), int(row["inode"]),
                int(row["mtime_ns"]),
            )
        except (OSError, KeyError, TypeError, ValueError):
            return False

    valid = (manifest.get("mix_name") == mix_name
             and manifest.get("config_echo") == config_echo(cfg)
             and len(order_paths) == 1
             and artifact_current(members)
             and all(artifact_current(path) for path in order_paths))
    return valid, manifest


def run(root: Path, workers: int) -> dict[str, Any]:
    freeze_path = root / FREEZE.relative_to(ROOT)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "PASS":
        raise ValueError("frozen recipe is not PASS")
    freeze_tool = root / "experiments/e4_curriculum/scripts/freeze_1t_recipe.py"
    if freeze.get("freeze_tool_sha256") != sha256(freeze_tool):
        raise ValueError("frozen recipe uses a stale freeze tool")
    expected_upstream = base_upstream_snapshot(root)
    for relative, expected in freeze.get("outputs", {}).items():
        path = root / relative
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"frozen recipe output drift: {relative}")
    try:
        state = json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {"schema_version": 1, "status": "IN_PROGRESS", "jobs": {}}
    jobs = state.setdefault("jobs", {})
    retired = state.setdefault("retired_jobs", {})
    for name in sorted(set(jobs) - set(STAGES)):
        retired[name] = jobs.pop(name)
    state["expected_jobs"] = list(STAGES)
    manifests = {}
    for stage in STAGES:
        relative = freeze["stages"][stage]["config"]
        config = root / relative
        job = jobs.setdefault(stage, {"status": "PENDING", "attempts": 0,
                                      "config": relative})
        valid, manifest = output_valid(root, config)
        if valid and upstream_snapshot_identity(manifest or {}) == expected_upstream:
            job["status"] = "PASS"
            manifests[stage] = manifest
            continue
        if int(job.get("attempts", 0)) >= len(BACKOFF):
            job.update({"attempts": 0,
                        "retry_generation": int(job.get("retry_generation", 0)) + 1})
        while int(job.get("attempts", 0)) < len(BACKOFF):
            attempt = int(job.get("attempts", 0)) + 1
            log_path = root / f"report/logs/pretrain_1t_s6_{stage}.log"
            job.update({"status": "RUNNING", "attempts": attempt,
                        "total_attempts": int(job.get("total_attempts", 0)) + 1,
                        "started_at": now_iso(), "log": str(log_path)})
            atomic_json(STATE, {**state, "status": "IN_PROGRESS", "updated_at": now_iso()})
            env = dict(os.environ)
            env["PYTHONPATH"] = str(root / "data-process/src")
            # This dedicated launcher owns the frozen production 1T S6 jobs.
            # The repository-level pause blocks unattended legacy S6 writers;
            # keep the sentinel intact and use the documented owner override.
            env["YUFORMER_ALLOW_PAUSED_S6"] = "1"
            with log_path.open("a", encoding="utf-8") as log:
                result = subprocess.run([
                    sys.executable, "-m", "yuformer_data.scale.driver", "--stage", "s6",
                    "--mix-config", str(config), "--workers", str(workers),
                    "--out-root", str(root / "data/processed"),
                ], cwd=root / "data-process", env=env, stdout=log, stderr=subprocess.STDOUT)
            valid, manifest = output_valid(root, config)
            if result.returncode == 0 and valid:
                actual_upstream = upstream_snapshot_identity(manifest or {})
                if actual_upstream != expected_upstream:
                    raise ValueError(
                        f"{stage}: newly materialized upstream snapshot differs from base release: "
                        f"{actual_upstream} != {expected_upstream}"
                    )
                job.update({"status": "PASS", "returncode": 0, "finished_at": now_iso()})
                manifests[stage] = manifest
                atomic_json(STATE, {**state, "updated_at": now_iso()})
                break
            job.update({"status": "RETRY_WAIT", "returncode": result.returncode,
                        "finished_at": now_iso()})
            atomic_json(STATE, {**state, "updated_at": now_iso()})
            if attempt < len(BACKOFF):
                time.sleep(BACKOFF[attempt - 1])
        if job.get("status") != "PASS":
            job["status"] = "RETRY_EXHAUSTED"
            state.update({"status": "FAIL", "updated_at": now_iso()})
            atomic_json(STATE, state)
            return state
    state.update({"status": "PASS", "updated_at": now_iso(),
                  "manifests": {stage: {"mix_name": value["mix_name"],
                                        "order_tables": value["order_tables"]}
                                for stage, value in manifests.items()}})
    atomic_json(STATE, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1))
    args = parser.parse_args()
    result = run(args.root.resolve(), args.workers)
    print(json.dumps({"status": result["status"], "jobs": len(result["jobs"])}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
