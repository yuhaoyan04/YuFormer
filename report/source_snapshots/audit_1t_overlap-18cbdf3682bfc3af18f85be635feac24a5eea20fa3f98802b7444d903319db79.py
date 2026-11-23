#!/usr/bin/env python3
"""Build and audit the cross-stage doc occurrence ledger for frozen 1T WSD."""
from __future__ import annotations

import argparse
import math
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


ROOT = Path(__file__).resolve().parents[3]
STAGES = ("warmup", "stable", "anneal")
RECORD_DTYPE = np.dtype([("doc_id", "S32"), ("n_tokens", "<u4")], align=False)
CONSUMED_DTYPE = np.dtype([
    ("doc_id", "S32"), ("doc_tokens", "<u4"), ("begin", "<u4"), ("end", "<u4")
], align=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(64 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def stat_identity(path: Path) -> dict[str, int]:
    value = path.stat()
    return {"bytes": value.st_size, "device": value.st_dev,
            "inode": value.st_ino, "mtime_ns": value.st_mtime_ns}


def require_published_stat(path: Path, artifact: dict[str, Any]) -> dict[str, int]:
    current = stat_identity(path)
    expected = {key: int(artifact.get(key, -1)) for key in current}
    if current != expected:
        raise ValueError(f"order stat identity differs from frozen release: {path}")
    return current


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def extract_sorted(order: Path, output: Path, identity: str) -> dict[str, Any]:
    meta_path = output.with_suffix(".json")
    try:
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if (old.get("identity") == identity and output.is_file()
                and output.stat().st_size == int(old["rows"]) * RECORD_DTYPE.itemsize
                and old.get("sha256") == sha256(output)):
            return old
    except (OSError, ValueError, KeyError):
        pass
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    rows, tokens, previous = 0, 0, None
    with temporary.open("wb") as handle:
        for batch in pq.ParquetFile(order).iter_batches(
                batch_size=1_000_000, columns=["doc_id", "n_tokens"]):
            ids = batch.column(0)
            if ids.null_count:
                raise ValueError(f"null doc_id in {order}")
            if pa.types.is_large_string(ids.type):
                offset_dtype, offset_width = np.int64, 8
            elif pa.types.is_string(ids.type):
                offset_dtype, offset_width = np.int32, 4
            else:
                raise ValueError(f"unsupported doc_id type {ids.type} in {order}")
            offsets = np.frombuffer(ids.buffers()[1], dtype=offset_dtype,
                                    count=len(ids) + 1, offset=ids.offset * offset_width)
            if not np.all(np.diff(offsets) == 32):
                raise ValueError(f"non-32-byte doc_id in {order}")
            data = memoryview(ids.buffers()[2])[int(offsets[0]):int(offsets[-1])]
            id_values = np.frombuffer(data, dtype="S32", count=len(ids))
            if len(id_values) > 1 and np.any(id_values[1:] <= id_values[:-1]):
                raise ValueError(f"order is not strictly doc_id sorted: {order}")
            if previous is not None and id_values[0] <= previous:
                raise ValueError(f"order batches are not globally doc_id sorted: {order}")
            previous = id_values[-1]
            nt = batch.column(1).to_numpy(zero_copy_only=False).astype("<u4", copy=False)
            records = np.empty(len(ids), dtype=RECORD_DTYPE)
            records["doc_id"], records["n_tokens"] = id_values, nt
            handle.write(records.tobytes())
            rows += len(records); tokens += int(nt.sum(dtype=np.uint64))
    os.replace(temporary, output)
    meta = {"identity": identity, "rows": rows, "reference_tokens": tokens,
            "sha256": sha256(output),
            "record_bytes": RECORD_DTYPE.itemsize, "source": str(order)}
    atomic_json(meta_path, meta)
    return meta


def consumed_position_bounds(lengths: np.ndarray, trained_tokens: int) -> dict[str, int]:
    """Locate the document-local intervals covering target tokens [1, N+1)."""
    target_start, target_stop = 1, int(trained_tokens) + 1
    if trained_tokens < 1:
        raise ValueError("trained token count must be positive")
    cumulative = 0
    first = last = None
    first_start = last_start = 0
    chunk_rows = 8_000_000
    for base in range(0, len(lengths), chunk_rows):
        values = np.asarray(lengths[base:base + chunk_rows], dtype=np.uint64)
        ends = np.cumsum(values, dtype=np.uint64) + cumulative
        if first is None and len(ends) and int(ends[-1]) > target_start:
            local = int(np.searchsorted(ends, target_start, side="right"))
            first = base + local
            first_start = cumulative if local == 0 else int(ends[local - 1])
        if last is None and len(ends) and int(ends[-1]) >= target_stop:
            local = int(np.searchsorted(ends, target_stop, side="left"))
            last = base + local
            last_start = cumulative if local == 0 else int(ends[local - 1])
            break
        if len(ends):
            cumulative = int(ends[-1])
    if first is None or last is None:
        raise ValueError(f"stream has insufficient tokens for {trained_tokens} targets")
    return {"first_position": first, "last_position": last,
            "first_global_start": first_start, "last_global_start": last_start,
            "target_start": target_start, "target_stop": target_stop}


def extract_consumed(order: Path, lengths_path: Path, output: Path, *,
                     identity: str, trained_tokens: int) -> dict[str, Any]:
    meta_path = output.with_suffix(".json")
    try:
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if (old.get("schema_version") == 2 and old.get("identity") == identity and output.is_file()
                and sum(old.get("domain_tokens", {}).values()) == trained_tokens
                and output.stat().st_size == int(old["rows"]) * CONSUMED_DTYPE.itemsize
                and old.get("sha256") == sha256(output)):
            return old
    except (OSError, ValueError, KeyError):
        pass
    lengths = np.memmap(lengths_path, dtype="<u4", mode="r")
    bounds = consumed_position_bounds(lengths, trained_tokens)
    first, last = bounds["first_position"], bounds["last_position"]
    seen = np.zeros(last - first + 1, dtype=np.bool_)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    rows = consumed = 0
    domain_tokens: dict[str, int] = {}
    previous = None
    with temporary.open("wb") as handle:
        for batch in pq.ParquetFile(order).iter_batches(
                batch_size=1_000_000, columns=["doc_id", "position", "domain"]):
            ids = batch.column(0)
            if ids.null_count:
                raise ValueError(f"null doc_id in {order}")
            offset_dtype, offset_width = ((np.int64, 8) if pa.types.is_large_string(ids.type)
                                          else (np.int32, 4))
            offsets = np.frombuffer(ids.buffers()[1], dtype=offset_dtype,
                                    count=len(ids) + 1, offset=ids.offset * offset_width)
            if not np.all(np.diff(offsets) == 32):
                raise ValueError(f"non-32-byte doc_id in {order}")
            data = memoryview(ids.buffers()[2])[int(offsets[0]):int(offsets[-1])]
            id_values = np.frombuffer(data, dtype="S32", count=len(ids))
            if len(id_values) > 1 and np.any(id_values[1:] <= id_values[:-1]):
                raise ValueError(f"order is not strictly doc_id sorted: {order}")
            if previous is not None and len(id_values) and id_values[0] <= previous:
                raise ValueError(f"order batches are not globally doc_id sorted: {order}")
            if len(id_values):
                previous = id_values[-1]
            positions = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            mask = (positions >= first) & (positions <= last)
            if not bool(np.any(mask)):
                continue
            selected_positions = positions[mask]
            relative = selected_positions - first
            if bool(np.any(seen[relative])):
                raise ValueError(f"duplicate consumed position in {order}")
            seen[relative] = True
            doc_lengths = np.asarray(lengths[selected_positions], dtype="<u4")
            begins = np.zeros(len(selected_positions), dtype="<u4")
            ends = doc_lengths.copy()
            first_mask = selected_positions == first
            last_mask = selected_positions == last
            begins[first_mask] = bounds["target_start"] - bounds["first_global_start"]
            ends[last_mask] = bounds["target_stop"] - bounds["last_global_start"]
            if bool(np.any(begins >= ends)) or bool(np.any(ends > doc_lengths)):
                raise ValueError(f"invalid consumed document interval in {order}")
            records = np.empty(len(selected_positions), dtype=CONSUMED_DTYPE)
            records["doc_id"] = id_values[mask]
            records["doc_tokens"] = doc_lengths
            records["begin"], records["end"] = begins, ends
            domains = batch.column(2).filter(pa.array(mask)).to_pylist()
            counts = ends.astype(np.uint64) - begins.astype(np.uint64)
            if any(not isinstance(domain, str) or not domain for domain in domains):
                raise ValueError(f"invalid consumed domain in {order}")
            domains = np.asarray(domains)
            for domain in np.unique(domains):
                domain_tokens[str(domain)] = domain_tokens.get(str(domain), 0) + int(
                    counts[domains == domain].sum(dtype=np.uint64))
            handle.write(records.tobytes())
            rows += len(records)
            consumed += int((ends.astype(np.uint64) - begins).sum(dtype=np.uint64))
    if not bool(np.all(seen)):
        raise ValueError(f"consumed position range is incomplete in {order}")
    if consumed != trained_tokens:
        raise ValueError(f"consumed tokens {consumed} != configured {trained_tokens}")
    os.replace(temporary, output)
    meta = {"schema_version": 2, "identity": identity, "rows": rows, "consumed_tokens": consumed,
            "domain_tokens": domain_tokens,
            "sha256": sha256(output), "record_bytes": CONSUMED_DTYPE.itemsize,
            "source": str(order), "doc_lengths": str(lengths_path), **bounds}
    atomic_json(meta_path, meta)
    return meta


def audit_consumed_mixture(counts: dict[str, int], targets: dict[str, float], *,
                           absolute_tolerance: float, relative_tolerance: float) -> dict[str, Any]:
    if (not targets or not all(math.isfinite(float(w)) and 0 < float(w) <= 1
                               for w in targets.values())
            or not math.isclose(sum(targets.values()), 1.0, abs_tol=1e-9)):
        raise ValueError("invalid frozen domain mixture weights")
    if any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError("invalid consumed domain token counts")
    if not all(math.isfinite(t) and 0 <= t < 1
               for t in (absolute_tolerance, relative_tolerance)):
        raise ValueError("invalid consumed mixture tolerance")
    total = sum(counts.values())
    rows = {}
    for domain, target in targets.items():
        actual = counts.get(domain, 0) / total if total else 0.0
        error = abs(actual - target)
        rows[domain] = {"tokens": counts.get(domain, 0), "target_weight": target,
                        "actual_weight": actual, "absolute_error": error,
                        "relative_error": error / target,
                        "pass": error <= absolute_tolerance + 1e-12
                        and error / target <= relative_tolerance + 1e-12}
    passed = total > 0 and set(counts) == set(targets) and all(r["pass"] for r in rows.values())
    return {"status": "PASS" if passed else "FAIL", "consumed_tokens": total,
            "absolute_tolerance": absolute_tolerance, "relative_tolerance": relative_tolerance,
            "domains": rows, "unexpected_domains": sorted(set(counts) - set(targets))}


def validated_data_contract(root: Path, release_path: Path, frozen_path: Path):
    from training_data_origin import read_bound, resolve_data_origin
    frozen_raw, frozen_sha = read_bound(frozen_path)
    release_raw, release_sha = read_bound(release_path)
    binding = resolve_data_origin(root, release_path, frozen_path)
    if (binding['release_sha256'] != release_sha
            or binding.get('training_recipe_sha256', binding['data_recipe_sha256']) != frozen_sha):
        raise ValueError('overlap data contract changed during origin validation')
    read_bound(frozen_path, frozen_sha)
    read_bound(release_path, release_sha)
    return json.loads(frozen_raw), json.loads(release_raw), binding


def audit(root: Path, cache: Path) -> dict[str, Any]:
    release_path = root / "data/processed/releases/pretrain-1t-frozen.release.json"
    plan_path = root / "experiments/e4_curriculum/configs/e4_1t_wsd_context.yaml"
    config_path = root / "pretrain/configs/pretrain_base.yaml"
    frozen_path = root / "experiments/e4_curriculum/results/frozen_1t_recipe.json"
    frozen, release, data_origin = validated_data_contract(root, release_path, frozen_path)
    plan = yaml.safe_load(plan_path.read_text())
    if release.get("status") != "PASS":
        raise ValueError("frozen order release is not PASS")
    artifacts = {row["path"]: row for row in release["artifacts"]}
    cache.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text())
    limits = [int(value) for value in config["data"]["stage_sample_limits"]]
    seq_len = int(config["data"]["seq_len"])
    extracted, raw_paths, consumed_extracted, consumed_paths = {}, [], {}, []
    evidence_snapshot = {
        path.resolve().relative_to(root.resolve()).as_posix(): sha256(path)
        for path in (release_path, plan_path, config_path, frozen_path)
    }
    artifact_stats: dict[str, dict[str, int]] = {}
    for stage in STAGES:
        order_relative = release["stages"][stage]["order"]
        row = artifacts[order_relative]
        order_path = root / order_relative
        before = require_published_stat(order_path, row)
        raw = cache / f"{stage}.records"
        extracted[stage] = extract_sorted(order_path, raw, row["sha256"])
        if stat_identity(order_path) != before:
            raise ValueError(f"order changed while overlap audit read it: {order_path}")
        artifact_stats[order_relative] = before
        raw_paths.append(raw)
        meta_path = root / f"data/token_streams/pretrain-1t/{stage}.meta.json"
        stream_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        evidence_snapshot[meta_path.relative_to(root).as_posix()] = sha256(meta_path)
        lengths_record = stream_meta.get("doc_lengths") or {}
        lengths_path = meta_path.parent / str(lengths_record.get("file", ""))
        lengths_before = stat_identity(lengths_path) if lengths_path.is_file() else None
        lengths_sha = sha256(lengths_path) if lengths_before is not None else ""
        if lengths_before is not None and stat_identity(lengths_path) != lengths_before:
            raise ValueError(f"document-length ledger changed while hashing: {lengths_path}")
        if (stream_meta.get("order_sha256") != row["sha256"]
                or lengths_path.name != f"{stage}.doc_lengths.u32"
                or lengths_record.get("dtype") != "<u4"
                or not lengths_path.is_file()
                or lengths_path.stat().st_size != int(lengths_record.get("bytes", -1))
                or lengths_sha != lengths_record.get("sha256")):
            raise ValueError(f"unbound or corrupt document-length ledger for {stage}")
        artifact_stats[lengths_path.relative_to(root).as_posix()] = lengths_before
        length_identity = (f"{row['sha256']}:{lengths_record.get('sha256')}:"
                           f"{limits[len(consumed_paths)]}:{seq_len}")
        consumed_raw = cache / f"{stage}.consumed.records"
        consumed_extracted[stage] = extract_consumed(
            root / order_relative, lengths_path, consumed_raw, identity=length_identity,
            trained_tokens=limits[len(consumed_paths)] * seq_len,
        )
        consumed_paths.append(consumed_raw)
    binary = cache / "overlap_merge"
    source = root / "experiments/e4_curriculum/scripts/overlap_merge.c"
    compiled = subprocess.run(["cc", "-O3", "-std=c11", str(source), "-o", str(binary)],
                              capture_output=True, text=True)
    if compiled.returncode:
        raise RuntimeError((compiled.stderr or compiled.stdout)[-2000:])
    merged = subprocess.run([str(binary), *map(str, raw_paths)], capture_output=True, text=True)
    if merged.returncode:
        raise RuntimeError((merged.stderr or merged.stdout)[-2000:])
    metrics = json.loads(merged.stdout)
    consumed_binary = cache / "overlap_consumed_merge"
    consumed_source = root / "experiments/e4_curriculum/scripts/overlap_consumed_merge.c"
    compiled = subprocess.run(["cc", "-O3", "-std=c11", str(consumed_source),
                               "-o", str(consumed_binary)], capture_output=True, text=True)
    if compiled.returncode:
        raise RuntimeError((compiled.stderr or compiled.stdout)[-2000:])
    merged = subprocess.run([str(consumed_binary), *map(str, consumed_paths)],
                            capture_output=True, text=True)
    if merged.returncode:
        raise RuntimeError((merged.stderr or merged.stdout)[-2000:])
    consumed_metrics = json.loads(merged.stdout)
    overlap = plan["pretrain"]["overlap_policy"]
    replay = (consumed_metrics["occurrence_consumed_tokens"]
              - consumed_metrics["unique_consumed_tokens"])
    fraction = replay / consumed_metrics["occurrence_consumed_tokens"]
    mixture = {
        stage: audit_consumed_mixture(
            consumed_extracted[stage]["domain_tokens"],
            frozen["stages"][stage]["target_final_domain_weights"],
            absolute_tolerance=float(plan["release_gate"]["max_domain_weight_absolute_error"]),
            relative_tolerance=float(plan["release_gate"]["max_domain_weight_relative_error"]),
        ) for stage in STAGES
    }
    checks = {
        "consumed_domain_mixture_within_tolerance": all(
            value["status"] == "PASS" for value in mixture.values()),
        "token_identity_consistent": metrics["token_mismatches"] == 0,
        "minimum_unique_reference_tokens": (
            metrics["unique_reference_tokens"] >= int(overlap["minimum_unique_reference_tokens"])),
        "maximum_replayed_consumption_fraction": (
            fraction <= float(overlap["maximum_replayed_consumption_fraction"])),
        "actual_token_identity_consistent": consumed_metrics["token_mismatches"] == 0,
        "configured_consumption_exact": (
            consumed_metrics["occurrence_consumed_tokens"]
            == sum(limits) * seq_len
            == sum(row["consumed_tokens"] for row in consumed_extracted.values())),
        "occurrence_totals_match_extraction": (
            metrics["occurrence_docs"] == sum(row["rows"] for row in extracted.values())
            and metrics["occurrence_reference_tokens"] == sum(
                row["reference_tokens"] for row in extracted.values())),
    }
    return {"schema_version": 2, "status": "PASS" if all(checks.values()) else "FAIL",
            "audit_tool_sha256": sha256(Path(__file__)),
            "data_origin": data_origin,
            "merge_tool_sha256": {
                source.relative_to(root).as_posix(): sha256(source),
                consumed_source.relative_to(root).as_posix(): sha256(consumed_source),
            },
            "checks": checks, "stages": extracted, "metrics": metrics,
            "consumed_domain_mixture": mixture,
            "consumed_stages": consumed_extracted, "consumed_metrics": consumed_metrics,
            "replayed_consumed_tokens": replay, "replayed_consumption_fraction": fraction,
            "consumption_interval": "causal_targets_stream_indices_[1,N*seq_len+1)",
            "evidence_snapshot": evidence_snapshot,
            "evidence_snapshot_count": len(evidence_snapshot),
            "large_artifact_stat_identities": artifact_stats}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/processed/overlap_1t_cache")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/processed/audits/pretrain_1t_overlap.json")
    args = parser.parse_args()
    result = audit(args.root.resolve(), args.cache.resolve())
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "checks": result["checks"],
                      "replayed_fraction": result["replayed_consumption_fraction"]}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
