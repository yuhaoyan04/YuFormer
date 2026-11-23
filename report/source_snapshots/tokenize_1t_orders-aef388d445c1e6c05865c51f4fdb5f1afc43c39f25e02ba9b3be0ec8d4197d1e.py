#!/usr/bin/env python3
"""Tokenize the frozen 1T WSD stages with resumable per-stage retries."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "pretrain/scripts"))
from source_document_binding import verified_source_document_binding
RELEASE = ROOT / "data/processed/releases/pretrain-1t-frozen.release.json"
BASE_RELEASE = ROOT / "data/processed/releases/mix_m3_e1_100b-20261019.release.json"
AUDIT = ROOT / "data/processed/audits/pretrain_1t_token_streams.json"
STATE = ROOT / "data/health/pretrain-1t-tokenization.json"
STREAMS = ROOT / "data/token_streams/pretrain-1t"
VAL_STREAMS = ROOT / "data/token_streams/m3_20261020_yf64k"
VAL_ORDERS = ROOT / "data/processed/val_sets"
MATERIALIZER = ROOT / "pretrain/scripts/materialize_tokens.py"
STAGES = ("warmup", "stable", "anneal")
BACKOFF = (300, 1800, 7200)
STORAGE_SAFETY_NUMERATOR = 11
STORAGE_SAFETY_DENOMINATOR = 10
STORAGE_RESERVE_BYTES = 100 << 30


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


def audit_stage_mixture(root: Path, name: str, release: dict[str, Any]) -> dict[str, Any]:
    script = root / "experiments/e4_curriculum/scripts/audit_1t_overlap.py"
    spec = importlib.util.spec_from_file_location("early_stage_overlap", script)
    overlap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(overlap)
    frozen_path = root / "experiments/e4_curriculum/results/frozen_1t_recipe.json"
    config_path = root / "pretrain/configs/pretrain_base.yaml"
    plan_path = root / "experiments/e4_curriculum/configs/e4_1t_wsd_context.yaml"
    frozen = json.loads(frozen_path.read_text())
    release_path = root / RELEASE.relative_to(ROOT)
    if (frozen.get("status") != "PASS" or release.get("status") != "PASS"
            or json.loads(release_path.read_text()) != release
            or release.get("frozen_recipe_sha256") != sha256(frozen_path)):
        raise ValueError("early mixture audit requires the current frozen order release")
    config = yaml.safe_load(config_path.read_text())
    plan = yaml.safe_load(plan_path.read_text())
    limit = int(config["data"]["stage_sample_limits"][STAGES.index(name)])
    seq_len = int(config["data"]["seq_len"])
    order_relative = release["stages"][name]["order"]
    record = next(row for row in release["artifacts"] if row["path"] == order_relative)
    order = root / order_relative
    before = overlap.require_published_stat(order, record)
    meta_path = root / f"data/token_streams/pretrain-1t/{name}.meta.json"
    meta = json.loads(meta_path.read_text())
    if not verified_source_document_binding(meta):
        raise ValueError(f"early mixture audit requires verified source document rows: {name}")
    ledger = meta.get("doc_lengths") or {}
    lengths = meta_path.parent / str(ledger.get("file", ""))
    lengths_before = overlap.stat_identity(lengths)
    if (meta.get("order_sha256") != record["sha256"]
            or lengths.name != f"{name}.doc_lengths.u32"
            or ledger.get("dtype") != "<u4"
            or lengths.stat().st_size != int(ledger.get("bytes", -1))
            or sha256(lengths) != ledger.get("sha256")):
        raise ValueError(f"early mixture audit has an unbound document-length ledger: {name}")
    cache = root / "data/processed/overlap_1t_cache"
    cache.mkdir(parents=True, exist_ok=True)
    identity = f"{record['sha256']}:{ledger['sha256']}:{limit}:{seq_len}"
    extracted = overlap.extract_consumed(
        order, lengths, cache / f"{name}.consumed.records", identity=identity,
        trained_tokens=limit * seq_len,
    )
    if (overlap.stat_identity(order) != before
            or overlap.stat_identity(lengths) != lengths_before):
        raise ValueError(f"inputs changed during early mixture audit: {name}")
    result = overlap.audit_consumed_mixture(
        extracted["domain_tokens"], frozen["stages"][name]["target_final_domain_weights"],
        absolute_tolerance=float(plan["release_gate"]["max_domain_weight_absolute_error"]),
        relative_tolerance=float(plan["release_gate"]["max_domain_weight_relative_error"]),
    )
    return {"status": result["status"], "stage": name, "mixture": result,
            "configured_target_tokens": limit * seq_len, "consumed": extracted,
            "evidence_sha256": {str(path.relative_to(root)): sha256(path)
                                for path in (release_path, frozen_path, config_path, plan_path, meta_path)},
            "order_stat": before, "ledger_stat": lengths_before,
            "audit_tool_sha256": sha256(script),
            "tokenization_tool_sha256": sha256(Path(__file__))}


def storage_preflight(release: dict[str, Any], tokenizer_gate: dict[str, Any],
                      completed_stages: set[str], output_dir: Path) -> dict[str, Any]:
    """Conservatively bound free space needed by the sequential materializer.

    Final streams consume one token-width copy. For large streams, external
    merge keeps the original NPZ part cache, an unpacked raw part copy, and the
    staging binary at the same time. Reserve two extra copies of the largest
    remaining stage. Existing partial files are already reflected in filesystem
    free space; counting the whole remaining stage is intentionally conservative
    and fails before producing TB-scale output.
    """
    vocab_size = int(tokenizer_gate.get("vocab_size", 0))
    if vocab_size <= 0:
        raise ValueError("tokenizer gate does not declare a positive vocab_size")
    token_width = 2 if vocab_size <= (1 << 16) else 4
    remaining: dict[str, int] = {}
    for name in STAGES:
        stage = release.get("stages", {}).get(name, {})
        tokens = int(stage.get("target_final_tokens", 0))
        if tokens <= 0:
            raise ValueError(f"{name}: release does not declare target_final_tokens")
        if name not in completed_stages:
            remaining[name] = tokens * token_width
    final_bytes = sum(remaining.values())
    largest_stage_bytes = max(remaining.values(), default=0)
    peak_working_bytes = final_bytes + 2 * largest_stage_bytes
    required_free_bytes = (
        peak_working_bytes * STORAGE_SAFETY_NUMERATOR
        + STORAGE_SAFETY_DENOMINATOR - 1
    ) // STORAGE_SAFETY_DENOMINATOR + STORAGE_RESERVE_BYTES
    usage = shutil.disk_usage(output_dir)
    return {
        "status": "PASS" if usage.free >= required_free_bytes else "BLOCKED",
        "filesystem_total_bytes": usage.total,
        "filesystem_used_bytes": usage.used,
        "filesystem_free_bytes": usage.free,
        "token_width_bytes": token_width,
        "remaining_stage_output_bytes": remaining,
        "remaining_final_bytes": final_bytes,
        "largest_remaining_stage_bytes": largest_stage_bytes,
        "peak_working_bytes_before_margin": peak_working_bytes,
        "safety_factor": STORAGE_SAFETY_NUMERATOR / STORAGE_SAFETY_DENOMINATOR,
        "reserve_bytes": STORAGE_RESERVE_BYTES,
        "required_free_bytes": required_free_bytes,
    }


def valid_pair(out_dir: Path, name: str, *, stage: dict[str, Any], release: dict[str, Any],
               tokenizer_sha256: str, tokenizer_gate_sha256: str,
               require_doc_lengths: bool = False) -> bool:
    binary, meta_path = out_dir / f"{name}.tokens.bin", out_dir / f"{name}.meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if require_doc_lengths and not verified_source_document_binding(meta):
            return False
        if require_doc_lengths:
            stat = binary.stat()
            current = {"bytes": stat.st_size, "device": stat.st_dev,
                       "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns}
            if meta.get("stream_stat_identity") != current:
                return False
        width = {"<u2": 2, "<u4": 4}.get(meta.get("dtype"), 0)
        order_record = next(
            (row for row in release.get("artifacts", []) if row.get("path") == stage.get("order")), None
        )
        lengths_record = meta.get("doc_lengths") or {}
        lengths_path = out_dir / str(lengths_record.get("file", ""))
        lengths_valid = (
            not require_doc_lengths
            or (int(meta.get("schema_version", 0)) >= 3
                and lengths_path.is_file()
                and lengths_path.name == f"{name}.doc_lengths.u32"
                and lengths_record.get("dtype") == "<u4"
                and int(lengths_record.get("rows", -1)) == int(meta.get("n_docs", -2))
                and int(lengths_record.get("bytes", -1)) == lengths_path.stat().st_size
                and lengths_path.stat().st_size == int(meta.get("n_docs", -1)) * 4
                and edge_sha256(lengths_path) == lengths_record.get("edge_sha256"))
        )
        return (width > 0 and order_record is not None and binary.is_file() and lengths_valid
                and int(meta.get("schema_version", 0)) >= 2 and meta.get("name") == name
                and binary.stat().st_size == int(meta["bytes"])
                and int(meta["total_tokens"]) * width == binary.stat().st_size
                and edge_sha256(binary) == meta.get("edge_sha256")
                and meta.get("order_sha256") == order_record.get("sha256")
                and int(meta.get("order_bytes", -1)) == int(order_record.get("bytes", -2))
                and meta.get("data_snapshot_id") == release.get("snapshot_id")
                and meta.get("data_release_name") == release.get("release_name")
                and meta.get("tokenizer_sha256") == tokenizer_sha256
                and meta.get("tokenizer_gate_sha256") == tokenizer_gate_sha256)
    except (OSError, ValueError, KeyError):
        return False


def run(workers: int) -> dict[str, Any]:
    release = json.loads(RELEASE.read_text(encoding="utf-8"))
    base_release = json.loads(BASE_RELEASE.read_text(encoding="utf-8"))
    if release.get("status") != "PASS" or set(release.get("stages", {})) != set(STAGES):
        raise ValueError("frozen 1T order release is not PASS with three stages")
    if base_release.get("status") != "PASS":
        raise ValueError("base release for validation streams is not PASS")
    try:
        state = json.loads(STATE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        state = {"schema_version": 1, "status": "IN_PROGRESS", "jobs": {}}
    jobs = state.setdefault("jobs", {})
    state["status"] = "IN_PROGRESS"
    STREAMS.mkdir(parents=True, exist_ok=True)
    VAL_STREAMS.mkdir(parents=True, exist_ok=True)
    tokenizer_path = ROOT / "tokenizer/tokenizers/m3_20261020/yf-64k-bpe/tokenizer.json"
    gate_path = ROOT / "data/TOKENIZER_READY.json"
    tokenizer_gate = json.loads(gate_path.read_text(encoding="utf-8"))
    tokenizer_sha = sha256(tokenizer_path)
    gate_sha = sha256(gate_path)
    repair_audit_sha256 = None
    force_names: set[str] = set()
    audit: dict[str, Any] = {}
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
    for name in STAGES:
        job = jobs.setdefault(name, {"status": "PENDING", "attempts": 0})
        stage = release["stages"][name]
        specs.append({"name": name, "stage": stage, "release": release,
                      "release_path": RELEASE, "order": ROOT / stage["order"],
                      "out_dir": STREAMS, "max_tokens": 0, "doc_lengths": True})
    for order in sorted(VAL_ORDERS.glob("*.parquet")):
        name = f"val_{order.stem}"
        specs.append({"name": name, "stage": {"order": str(order.relative_to(ROOT))},
                      "release": base_release, "release_path": BASE_RELEASE,
                      "order": order, "out_dir": VAL_STREAMS, "max_tokens": 20_000_000,
                      "doc_lengths": False})

    expected_names = {spec["name"] for spec in specs}
    state["expected_jobs"] = sorted(expected_names)
    completed_stages = {
        spec["name"] for spec in specs if spec["name"] in STAGES and valid_pair(
            spec["out_dir"], spec["name"], stage=spec["stage"], release=spec["release"],
            tokenizer_sha256=tokenizer_sha, tokenizer_gate_sha256=gate_sha,
            require_doc_lengths=True,
        )
    }
    storage = storage_preflight(release, tokenizer_gate, completed_stages, STREAMS)
    state["storage_preflight"] = storage
    if storage["status"] != "PASS":
        state.update({"status": "BLOCKED_STORAGE", "updated_at": now_iso()})
        atomic_json(STATE, state)
        return state
    quarantined: list[str] = []
    if repair_audit_sha256 is not None:
        expected_train = set(STAGES)
        expected_val = expected_names - expected_train
        published_train = {
            path.name.removesuffix(suffix)
            for suffix in (".meta.json", ".tokens.bin", ".doc_lengths.u32")
            for path in STREAMS.glob(f"*{suffix}")
        }
        published_val = {
            path.name.removesuffix(suffix)
            for suffix in (".meta.json", ".tokens.bin", ".doc_lengths.u32")
            for path in VAL_STREAMS.glob(f"val_*{suffix}")
        }
        quarantine = (ROOT / "data/token_streams/_quarantine"
                      / f"pretrain-1t-{repair_audit_sha256[:16]}")
        for label, out_dir, names in (
            ("train", STREAMS, published_train - expected_train),
            ("validation", VAL_STREAMS, published_val - expected_val),
        ):
            for name in sorted(names):
                if not name or Path(name).name != name:
                    raise ValueError(f"unsafe unexpected stream name: {name!r}")
                for suffix in (".tokens.bin", ".meta.json", ".doc_lengths.u32"):
                    source = out_dir / f"{name}{suffix}"
                    if source.is_file():
                        target = quarantine / label / source.name
                        target.parent.mkdir(parents=True, exist_ok=True)
                        os.replace(source, target)
                        quarantined.append(str(target.relative_to(ROOT)))
    def mixture_ready(spec: dict[str, Any], job: dict[str, Any]) -> bool:
        if not spec["doc_lengths"]:
            return True
        try:
            result = audit_stage_mixture(ROOT, spec["name"], spec["release"])
        except (ValueError, OSError, KeyError) as error:
            result = {"status": "BLOCKED_PREFLIGHT", "stage": spec["name"],
                      "error": str(error), "error_type": type(error).__name__,
                      "tokenization_tool_sha256": sha256(Path(__file__))}
        path = ROOT / f"data/health/pretrain-1t-{spec['name']}-mixture-preflight.json"
        atomic_json(path, result)
        job["consumed_mixture_preflight"] = {"status": result["status"],
                                             "report": str(path.relative_to(ROOT)),
                                             "sha256": sha256(path)}
        if result["status"] != "PASS":
            blocked = ("BLOCKED_PREFLIGHT" if result["status"] == "BLOCKED_PREFLIGHT"
                       else "BLOCKED_MIXTURE")
            job["status"] = blocked
            state["status"] = blocked
        state["updated_at"] = now_iso()
        atomic_json(STATE, state)
        return result["status"] == "PASS"

    for spec in specs:
        name = spec["name"]
        job = jobs.setdefault(name, {"status": "PENDING", "attempts": 0})
        if name not in force_names and valid_pair(
                spec["out_dir"], name, stage=spec["stage"], release=spec["release"],
                tokenizer_sha256=tokenizer_sha, tokenizer_gate_sha256=gate_sha,
                require_doc_lengths=spec["doc_lengths"]):
            job["status"] = "PASS"
            if not mixture_ready(spec, job):
                return state
            continue
        if int(job.get("attempts", 0)) >= len(BACKOFF):
            job.update({"attempts": 0,
                        "retry_generation": int(job.get("retry_generation", 0)) + 1})
        while int(job.get("attempts", 0)) < len(BACKOFF):
            attempt = int(job.get("attempts", 0)) + 1
            log_path = ROOT / f"report/logs/pretrain_1t_tokenize_{name}.log"
            job.update({"status": "RUNNING", "attempts": attempt,
                        "total_attempts": int(job.get("total_attempts", 0)) + 1,
                        "started_at": now_iso(), "log": str(log_path)})
            atomic_json(STATE, {**state, "status": "IN_PROGRESS", "updated_at": now_iso()})
            order = spec["order"]
            command = [
                sys.executable, str(MATERIALIZER), "--order", str(order),
                "--s3-root", str(ROOT / "data/processed/s3"),
                "--tokenizer", str(tokenizer_path),
                "--out-dir", str(spec["out_dir"]), "--name", name,
                "--data-snapshot", str(spec["release_path"]),
                "--tokenizer-gate", str(gate_path),
                "--workers", str(workers),
            ]
            if spec["max_tokens"]:
                command.extend(["--max-tokens", str(spec["max_tokens"])])
            if spec["doc_lengths"]:
                command.append("--write-doc-lengths")
            with log_path.open("a", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode == 0 and valid_pair(
                    spec["out_dir"], name, stage=spec["stage"], release=spec["release"], tokenizer_sha256=tokenizer_sha,
                    tokenizer_gate_sha256=gate_sha,
                    require_doc_lengths=spec["doc_lengths"]):
                job.update({"status": "PASS", "returncode": 0, "finished_at": now_iso()})
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
        if not mixture_ready(spec, job):
            return state
    state.update({"status": "PASS", "updated_at": now_iso(), "streams": str(STREAMS)})
    if repair_audit_sha256 is not None:
        state["repaired_audit_sha256"] = repair_audit_sha256
        state["repaired_streams"] = sorted(force_names)
        state["quarantined_unexpected_artifacts"] = quarantined
    atomic_json(STATE, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=min(128, os.cpu_count() or 1))
    args = parser.parse_args()
    result = run(args.workers)
    print(json.dumps({"status": result["status"], "jobs": len(result["jobs"])}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
