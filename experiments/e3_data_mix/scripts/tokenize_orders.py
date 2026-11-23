#!/usr/bin/env python3
"""Materialize all E3 order tables with resumable per-arm retries."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]
MATERIALIZER = ROOT / "pretrain/scripts/materialize_tokens.py"
DEFAULT_ORDERS = ROOT / "data/experiments/e3_data_mix/orders"
DEFAULT_STREAMS = ROOT / "data/token_streams/e3_20261021_yf64k"
BASE_RELEASE = ROOT / "data/processed/releases/mix_m3_e1_100b-20261019.release.json"
AUDIT = ROOT / "data/processed/audits/e3_token_streams_20261021.json"
VAL_STREAMS = ROOT / "data/token_streams/m3_20261020_yf64k"
VAL_ORDERS = ROOT / "data/processed/val_sets"
STATE = ROOT / "data/health/e3-tokenization.json"
BACKOFF = (60, 300, 1800)
MATRIX = ROOT / "experiments/e3_data_mix/configs/e3_matrix.yaml"
DOMAINS = {"web", "code", "math", "chinese", "academic", "books",
           "encyclopedia", "qa", "multilingual"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def edge_sha256(path: Path, edge_bytes: int = 1 << 20) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256(str(size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(edge_bytes))
        if size > edge_bytes:
            handle.seek(max(0, size - edge_bytes))
            digest.update(handle.read(edge_bytes))
    return digest.hexdigest()


def valid_pair(out_dir: Path, name: str, *, order: dict[str, Any],
               snapshot: dict[str, Any], tokenizer_sha256: str,
               tokenizer_gate_sha256: str) -> bool:
    binary, meta_path = out_dir / f"{name}.tokens.bin", out_dir / f"{name}.meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        itemsize = 2 if meta.get("dtype") == "<u2" else 4 if meta.get("dtype") == "<u4" else 0
        return (itemsize > 0 and binary.is_file()
                and int(meta.get("schema_version", 0)) >= 2
                and meta.get("name") == name
                and binary.stat().st_size == int(meta["bytes"])
                and binary.stat().st_size == int(meta["total_tokens"]) * itemsize
                and edge_sha256(binary) == meta.get("edge_sha256")
                and meta.get("order_sha256") == order.get("sha256")
                and int(meta.get("order_bytes", -1)) == int(order.get("size", -2))
                and meta.get("data_snapshot_id") == snapshot.get("snapshot_id")
                and meta.get("data_release_name") == snapshot.get("release_name")
                and meta.get("tokenizer_sha256") == tokenizer_sha256
                and meta.get("tokenizer_gate_sha256") == tokenizer_gate_sha256)
    except (OSError, ValueError, KeyError):
        return False


def run(orders: Path, streams: Path, workers: int) -> dict[str, Any]:
    manifest_path = orders / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matrix = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    expected_arms = set(matrix.get("arms", {}))
    if (manifest.get("status") != "PASS"
            or set(manifest.get("outputs", {})) != expected_arms
            or len(expected_arms) != 8
            or manifest.get("matrix_sha256") != sha256(MATRIX)):
        raise ValueError("E3 order manifest is not bound to the exact eight-arm matrix")
    base_release = json.loads(BASE_RELEASE.read_text(encoding="utf-8"))
    if base_release.get("status") != "PASS":
        raise ValueError("base release for validation streams is not PASS")
    try:
        state = json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {"schema_version": 1, "status": "IN_PROGRESS", "jobs": {}}
    jobs = state.setdefault("jobs", {})
    streams.mkdir(parents=True, exist_ok=True)
    VAL_STREAMS.mkdir(parents=True, exist_ok=True)
    tokenizer_path = ROOT / "tokenizer/tokenizers/m3_20261020/yf-64k-bpe/tokenizer.json"
    gate_path = ROOT / "data/TOKENIZER_READY.json"
    tokenizer_sha = sha256(tokenizer_path)
    gate_sha = sha256(gate_path)
    repair_audit_sha256 = None
    force_names: set[str] = set()
    try:
        audit = json.loads(AUDIT.read_text(encoding="utf-8"))
        if audit.get("status") == "FAIL":
            repair_audit_sha256 = sha256(AUDIT)
            force_names = {
                str(row.get("name")) for row in audit.get("streams", [])
                if isinstance(row, dict) and row.get("status") == "FAIL" and row.get("name")
            }
    except (OSError, ValueError):
        pass

    specs: list[dict[str, Any]] = []
    for name, row in sorted(manifest["outputs"].items()):
        specs.append({"name": name, "order_identity": row, "snapshot": manifest,
                      "snapshot_path": manifest_path, "order": orders / row["file"],
                      "out_dir": streams, "max_tokens": 0})
    base_artifacts = {row["path"]: row for row in base_release.get("artifacts", [])
                      if isinstance(row, dict) and isinstance(row.get("path"), str)}
    val_orders = sorted(VAL_ORDERS.glob("*.parquet"))
    if {order.stem for order in val_orders} != DOMAINS:
        raise ValueError("validation order roster is not the canonical nine domains")
    for order in val_orders:
        relative = str(order.relative_to(ROOT))
        artifact = base_artifacts.get(relative)
        if artifact is None:
            raise ValueError(f"validation order is absent from base release: {relative}")
        specs.append({
            "name": f"val_{order.stem}",
            "order_identity": {"sha256": artifact["sha256"], "size": artifact["bytes"]},
            "snapshot": base_release, "snapshot_path": BASE_RELEASE,
            "order": order, "out_dir": VAL_STREAMS, "max_tokens": 20_000_000,
        })
    expected_names = {spec["name"] for spec in specs}
    state["expected_jobs"] = sorted(expected_names)

    quarantined: list[str] = []
    if repair_audit_sha256 is not None:
        expected_train = set(manifest["outputs"])
        expected_val = expected_names - expected_train
        published_train = {
            path.name.removesuffix(suffix)
            for suffix in (".meta.json", ".tokens.bin")
            for path in streams.glob(f"*{suffix}")
        }
        published_val = {
            path.name.removesuffix(suffix)
            for suffix in (".meta.json", ".tokens.bin")
            for path in VAL_STREAMS.glob(f"val_*{suffix}")
        }
        quarantine = (ROOT / "data/token_streams/_quarantine"
                      / f"e3-{repair_audit_sha256[:16]}")
        for label, out_dir, names in (
            ("train", streams, published_train - expected_train),
            ("validation", VAL_STREAMS, published_val - expected_val),
        ):
            for name in sorted(names):
                if not name or Path(name).name != name:
                    raise ValueError(f"unsafe unexpected stream name: {name!r}")
                for suffix in (".tokens.bin", ".meta.json"):
                    source = out_dir / f"{name}{suffix}"
                    if source.is_file():
                        target = quarantine / label / source.name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(source, target)
                        quarantined.append(str(target.relative_to(ROOT)))

    for spec in specs:
        name = spec["name"]
        job = jobs.setdefault(name, {"status": "PENDING", "attempts": 0})
        if name not in force_names and valid_pair(spec["out_dir"], name,
                      order=spec["order_identity"], snapshot=spec["snapshot"],
                      tokenizer_sha256=tokenizer_sha, tokenizer_gate_sha256=gate_sha):
            job["status"] = "PASS"
            continue
        if int(job.get("attempts", 0)) >= len(BACKOFF):
            job.update({"attempts": 0,
                        "retry_generation": int(job.get("retry_generation", 0)) + 1})
        while int(job.get("attempts", 0)) < len(BACKOFF):
            attempt = int(job.get("attempts", 0)) + 1
            job.update({"status": "RUNNING", "attempts": attempt,
                        "total_attempts": int(job.get("total_attempts", 0)) + 1,
                        "started_at": now_iso()})
            state.update({"status": "IN_PROGRESS", "updated_at": now_iso()})
            atomic_json(STATE, state)
            order = spec["order"]
            log_path = ROOT / f"report/logs/e3_tokenize_{name}.log"
            command = [
                os.environ.get("PYTHON", "python"), str(MATERIALIZER),
                "--order", str(order),
                "--s3-root", str(ROOT / "data/processed/s3"),
                "--tokenizer", str(tokenizer_path),
                "--out-dir", str(spec["out_dir"]), "--name", name,
                "--data-snapshot", str(spec["snapshot_path"]),
                "--tokenizer-gate", str(gate_path), "--workers", str(workers),
            ]
            if spec["max_tokens"]:
                command.extend(["--max-tokens", str(spec["max_tokens"])])
            with log_path.open("a", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode == 0 and valid_pair(
                    spec["out_dir"], name, order=spec["order_identity"], snapshot=spec["snapshot"],
                    tokenizer_sha256=tokenizer_sha, tokenizer_gate_sha256=gate_sha):
                job.update({"status": "PASS", "returncode": 0, "finished_at": now_iso(),
                            "log": str(log_path)})
                atomic_json(STATE, {**state, "updated_at": now_iso()})
                break
            job.update({"status": "RETRY_WAIT", "returncode": result.returncode,
                        "finished_at": now_iso(), "log": str(log_path)})
            atomic_json(STATE, {**state, "updated_at": now_iso()})
            if attempt < len(BACKOFF):
                time.sleep(BACKOFF[attempt - 1])
        if job.get("status") != "PASS":
            job["status"] = "RETRY_EXHAUSTED"
            state.update({"status": "FAIL", "updated_at": now_iso()})
            atomic_json(STATE, state)
            return state
    state.update({"status": "PASS", "updated_at": now_iso(), "streams": str(streams)})
    if repair_audit_sha256 is not None:
        state["repaired_audit_sha256"] = repair_audit_sha256
        state["repaired_streams"] = sorted(force_names)
        state["quarantined_unexpected_artifacts"] = quarantined
    atomic_json(STATE, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", type=Path, default=DEFAULT_ORDERS)
    parser.add_argument("--streams", type=Path, default=DEFAULT_STREAMS)
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("workers must be positive")
    result = run(args.orders.resolve(), args.streams.resolve(), args.workers)
    print(json.dumps({"status": result["status"], "jobs": len(result["jobs"])}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
