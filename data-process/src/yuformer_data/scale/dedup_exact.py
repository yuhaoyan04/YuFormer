from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .dedup_common import (
    S2_DROP,
    S2_KEEP,
    check_memory_cap,
    doc_bin_to_id,
    drops_for,
    empty_table,
    fatal_summary,
    fingerprint,
    load_drops_lookup,
    load_dedup_config,
    normalize_for_hash,
    rank_of,
    run_tasks,
    source_rank_map,
    stack_u8_to_void,
    text_sha256,
    void_to_stack_u8,
    write_drops_lookup,
    write_stage_summary,
)
from .ingest import BatchWriter, write_summary
from .schema import S1_SCHEMA

log = logging.getLogger("yuformer.scale.dedup")

S2_SCHEMA: pa.Schema = S1_SCHEMA.append(
    pa.field("s2_decision", pa.string(), nullable=False)
)

S2_PART_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("bucket", pa.int32(), nullable=False),
        pa.field("h0", pa.uint64(), nullable=False),
        pa.field("h1", pa.uint64(), nullable=False),
        pa.field("h2", pa.uint64(), nullable=False),
        pa.field("h3", pa.uint64(), nullable=False),
        pa.field("d0", pa.uint64(), nullable=False),
        pa.field("d1", pa.uint64(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
    ]
)

S2_DECISIONS_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("decision", pa.string(), nullable=False),
        pa.field("dup_of", pa.string(), nullable=False),
    ]
)


def _bucket4(idx: int) -> str:
    return f"{idx:04x}"


def s2_part_path(out_root: Path, domain: str, source: str, relpath: str) -> Path:
    return out_root / "s2_hash_parts" / domain / source / (relpath + ".parquet")


def s2_part_summary_path(out_root: Path, source: str, relpath: str) -> Path:
    base = out_root / "s2_hash_parts" / "_summaries" / source / relpath
    return base.with_name(base.name + ".summary.json")


def s2_decisions_dir(out_root: Path, domain: str) -> Path:
    return out_root / "s2_decisions" / domain


def s2_lookup_dir(out_root: Path, domain: str) -> Path:
    return out_root / "s2_lookup" / domain


def s2_output_paths(out_root: Path, source: str, relpath: str) -> tuple[Path, Path]:
    base = out_root / "s2" / source / relpath
    return base.with_name(base.name + ".parquet"), base.with_name(base.name + ".summary.json")


# ---------------------------------------------------------------- pass 1

def process_shard_s2_hash(
    s1_path: Path, out_root: Path, source: str, relpath: str, num_buckets: int
) -> dict[str, Any]:
    out_path = s2_part_path(out_root, _domain_of(s1_path), source, relpath)
    summary_path = s2_part_summary_path(out_root, source, relpath)
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text("utf-8"))
        if summary.get("input_fingerprint") == fingerprint(s1_path):
            summary["skipped"] = True
            return summary

    started = time.monotonic()
    in_rows = out_rows = error_rows = 0
    try:
        pf = pq.ParquetFile(s1_path)
        writers: dict[str, Any] = {}
        for batch in pf.iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                in_rows += 1
                try:
                    digest = text_sha256(row["text"])
                except Exception:
                    error_rows += 1
                    continue
                bucket = int.from_bytes(digest[:2], "big") % num_buckets
                h = int.from_bytes(digest, "big")
                d_raw = bytes.fromhex(row["doc_id"])
                d = int.from_bytes(d_raw[:8], "little")
                d1 = int.from_bytes(d_raw[8:], "little")
                writer = writers.get(row["domain"])
                if writer is None:
                    writer = writers[row["domain"]] = _PartWriter(
                        s2_part_path(out_root, row["domain"], source, relpath),
                        S2_PART_SCHEMA,
                    )
                writer.add(
                    {
                        "bucket": bucket,
                        "h0": (h >> 192) & 0xFFFFFFFFFFFFFFFF,
                        "h1": (h >> 128) & 0xFFFFFFFFFFFFFFFF,
                        "h2": (h >> 64) & 0xFFFFFFFFFFFFFFFF,
                        "h3": h & 0xFFFFFFFFFFFFFFFF,
                        "d0": d & 0xFFFFFFFFFFFFFFFF,
                        "d1": d1,
                        "source": source,
                    }
                )
                out_rows += 1
        for writer in writers.values():
            writer.finish()
    except Exception as exc:
        for writer in writers.values():
            writer.abort()
        summary = {
            "stage": "s2_hash",
            "source": source,
            "relpath": relpath,
            "domain": "?",
            "in_rows": in_rows,
            "out_rows": out_rows,
            "error_rows": error_rows,
            "elapsed_sec": round(time.monotonic() - started, 3),
            "errors": {f"shard_error:{type(exc).__name__}": 1},
            "skipped": False,
        }
        write_summary(summary_path, summary)
        return summary

    domain = next(iter(writers), "?") if writers else _domain_of(s1_path)
    summary = {
        "stage": "s2_hash",
        "source": source,
        "relpath": relpath,
        "input_fingerprint": fingerprint(s1_path),
        "domain": domain if writers else "?",
        "in_rows": in_rows,
        "out_rows": out_rows,
        "error_rows": error_rows,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "mb_per_sec": round(
            s1_path.stat().st_size / max(time.monotonic() - started, 1e-9) / 1e6, 2
        ),
        "errors": {},
        "skipped": False,
    }
    write_summary(summary_path, summary)
    return summary


def _domain_of(s1_path: Path) -> str:
    pf = pq.ParquetFile(s1_path)
    batch = next(pf.iter_batches(batch_size=1, columns=["domain"]), None)
    if batch is None:
        return "?"
    return batch.column("domain")[0].as_py()


class _PartWriter:
    def __init__(self, path: Path, schema: pa.Schema):
        self.path = path
        self.schema = schema
        self.cols: dict[str, list[Any]] = {name: [] for name in schema.names}
        self.rows = 0

    def add(self, row: dict[str, Any]) -> None:
        for name in self.schema.names:
            self.cols[name].append(row[name])
        self.rows += 1

    def finish(self) -> None:
        if self.rows == 0:
            return
        order = np.lexsort(
            (
                np.asarray(self.cols["d1"], dtype=np.uint64),
                np.asarray(self.cols["d0"], dtype=np.uint64),
                np.asarray(self.cols["h3"], dtype=np.uint64),
                np.asarray(self.cols["h2"], dtype=np.uint64),
                np.asarray(self.cols["h1"], dtype=np.uint64),
                np.asarray(self.cols["h0"], dtype=np.uint64),
                np.asarray(self.cols["bucket"], dtype=np.int32),
            )
        )
        arrays = {
            "bucket": np.asarray(self.cols["bucket"], dtype=np.int32)[order],
            "h0": np.asarray(self.cols["h0"], dtype=np.uint64)[order],
            "h1": np.asarray(self.cols["h1"], dtype=np.uint64)[order],
            "h2": np.asarray(self.cols["h2"], dtype=np.uint64)[order],
            "h3": np.asarray(self.cols["h3"], dtype=np.uint64)[order],
            "d0": np.asarray(self.cols["d0"], dtype=np.uint64)[order],
            "d1": np.asarray(self.cols["d1"], dtype=np.uint64)[order],
            "source": np.asarray(self.cols["source"], dtype=object)[order],
        }
        import os

        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + f".tmp-{os.getpid()}")
        pq.write_table(pa.table(arrays, schema=self.schema), tmp, compression="zstd")
        os.replace(tmp, self.path)

    def abort(self) -> None:
        self.cols = {name: [] for name in self.schema.names}
        self.rows = 0


# ---------------------------------------------------------------- pass 2

def s2_reduce_task_id(domain: str, lo: int, hi: int) -> str:
    return f"{domain}/{lo:04x}-{hi:04x}"


def process_s2_reduce(
    out_root: Path,
    domain: str,
    lo: int,
    hi: int,
    cfg: dict[str, Any],
    expected_sources: list[str],
) -> dict[str, Any]:
    num_buckets = int(cfg["s2"]["num_buckets"])
    cap_gb = float(cfg["s2"]["memory_cap_gb"])
    rng = cfg["s2"]["reduce_range"]
    rank_map = source_rank_map(list(cfg.get("source_priority") or []))
    task_id = s2_reduce_task_id(domain, lo, hi)
    out_dir = s2_decisions_dir(out_root, domain)
    summary_paths = [
        out_dir / f"{_bucket4(b)}.summary.json" for b in range(lo, min(hi, num_buckets))
    ]
    if all(p.is_file() for p in summary_paths):
        existing = json.loads(summary_paths[0].read_text("utf-8"))
        if existing.get("domain_sources") == expected_sources:
            summaries = [json.loads(p.read_text("utf-8")) for p in summary_paths]
            for s in summaries:
                s["skipped"] = True
            return _merge_reduce_summaries(domain, lo, summaries)

    started = time.monotonic()
    part_root = out_root / "s2_hash_parts" / domain
    files = sorted(part_root.rglob("*.parquet")) if part_root.is_dir() else []
    try:
        rec, sources = _load_bucket_range(files, lo, hi, cap_gb)
        decisions, groups, dropped = _group_decisions(rec, sources, rank_map)
        _write_reduce_outputs(out_dir, lo, hi, num_buckets, rng, decisions, expected_sources)
    except Exception as exc:
        summary = fatal_summary("s2_reduce", task_id, exc, started)
        summary["domain"] = domain
        write_summary(out_dir / f"reduce-{lo:04x}-{hi:04x}.error.json", summary)
        return summary

    elapsed = time.monotonic() - started
    summary = {
        "stage": "s2_reduce",
        "task": task_id,
        "domain": domain,
        "groups": groups,
        "dropped": dropped,
        "rows": int(len(rec)),
        "parts": len(files),
        "elapsed_sec": round(elapsed, 3),
        "errors": {},
        "skipped": False,
    }
    return summary


def _load_bucket_range(
    files: list[Path], lo: int, hi: int, cap_gb: float
) -> tuple[np.ndarray, list[str]]:
    import pyarrow.dataset as ds

    recs: list[np.ndarray] = []
    src_lists: list[list[str]] = []
    total = 0
    for f in files:
        dset = ds.dataset(f, format="parquet")
        expr = (ds.field("bucket") >= lo) & (ds.field("bucket") < hi)
        for batch in dset.to_batches(
            batch_size=65536,
            filter=expr,
            columns=["bucket", "h0", "h1", "h2", "h3", "d0", "d1", "source"],
        ):
            n = batch.num_rows
            if n == 0:
                continue
            cols = [batch.column(i).to_numpy(zero_copy_only=False) for i in range(7)]
            rec = np.empty(n, dtype=[("b", "<i4"), ("h", "V32"), ("d", "V16")])
            rec["b"] = cols[0]
            rec["h"] = stack_u8_to_void(cols[1:5], 32)
            rec["d"] = stack_u8_to_void(cols[5:7], 16)
            recs.append(rec)
            src_lists.append(batch.column(7).to_pylist())
            total += n * rec.dtype.itemsize
            check_memory_cap(total, cap_gb, f"s2 reduce load [{lo},{hi})")
    if not recs:
        empty = np.empty(0, dtype=[("b", "<i4"), ("h", "V32"), ("d", "V16")])
        return empty, []
    rec = np.concatenate(recs)
    order = np.argsort(rec, kind="stable")
    src_flat = np.array([s for lst in src_lists for s in lst], dtype=object)
    return rec[order], src_flat[order].tolist()


def _group_decisions(
    rec: np.ndarray, sources: list[str], rank_map: dict[str, int]
) -> tuple[list[tuple[int, bytes, str, str]], int, int]:
    decisions: list[tuple[int, bytes, str, str]] = []
    if len(rec) == 0:
        return decisions, 0, 0
    same = (rec["b"][1:] == rec["b"][:-1]) & (rec["h"][1:] == rec["h"][:-1])
    ends = np.append(np.flatnonzero(~same) + 1, len(rec))
    groups = dropped = 0
    prev = 0
    for nxt in ends:
        if nxt - prev >= 2:
            groups += 1
            bucket = int(rec["b"][prev])
            members = rec["d"][prev:nxt]
            best_idx = 0
            best_key: tuple[int, str] | None = None
            for i in range(len(members)):
                key = (rank_of(rank_map, sources[prev + i]), doc_bin_to_id(members[i]))
                if best_key is None or key < best_key:
                    best_key = key
                    best_idx = i
            survivor = doc_bin_to_id(members[best_idx])
            for i in range(len(members)):
                if i != best_idx:
                    decisions.append((bucket, members[i].tobytes(), S2_DROP, survivor))
                    dropped += 1
        prev = nxt
    return decisions, groups, dropped


def _write_reduce_outputs(
    out_dir: Path,
    lo: int,
    hi: int,
    num_buckets: int,
    rng: int,
    decisions: list[tuple[int, bytes, str, str]],
    expected_sources: list[str],
) -> None:
    by_bucket: dict[int, list[tuple[bytes, str, str]]] = {}
    n_buckets = min(hi, num_buckets) - lo
    for b in range(lo, min(hi, num_buckets)):
        by_bucket[b] = []
    for bucket, doc_id, decision, dup_of in decisions:
        by_bucket[bucket].append((doc_id, decision, dup_of))
    out_dir.mkdir(parents=True, exist_ok=True)
    rank = 0
    for b in range(lo, min(hi, num_buckets)):
        rows = sorted(by_bucket[b], key=lambda r: r[0])
        path = out_dir / f"{_bucket4(b)}.parquet"
        tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
        if rows:
            table = pa.table(
                {
                    "doc_id": [doc_bin_to_id(r[0]) for r in rows],
                    "decision": [r[1] for r in rows],
                    "dup_of": [r[2] for r in rows],
                },
                schema=S2_DECISIONS_SCHEMA,
            )
            pq.write_table(table, tmp, compression="zstd")
        else:
            pq.write_table(empty_table(S2_DECISIONS_SCHEMA), tmp, compression="zstd")
        os.replace(tmp, path)
        summary = {
            "stage": "s2_reduce",
            "domain": "?",
            "bucket": b,
            "dropped": len(rows),
            "domain_sources": expected_sources,
            "rank": rank,
        }
        write_summary(out_dir / f"{_bucket4(b)}.summary.json", summary)
        rank += 1


def _merge_reduce_summaries(
    domain: str, lo: int, summaries: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "stage": "s2_reduce",
        "task": f"{domain}/{lo:04x}-skip",
        "domain": domain,
        "groups": sum(int(s.get("groups") or 0) for s in summaries),
        "dropped": sum(int(s.get("dropped") or 0) for s in summaries),
        "parts": 0,
        "elapsed_sec": 0.0,
        "errors": {},
        "skipped": True,
    }


# ---------------------------------------------------------------- finalize + pass 3

def build_s2_lookups(out_root: Path, domains: list[str]) -> list[dict[str, Any]]:
    summaries = []
    for domain in domains:
        out_dir = s2_lookup_dir(out_root, domain)
        marker = out_dir / "lookup.summary.json"
        current = 0
        for f in sorted(s2_decisions_dir(out_root, domain).glob("*.summary.json")):
            try:
                current += int(json.loads(f.read_text("utf-8")).get("dropped") or 0)
            except (json.JSONDecodeError, OSError):
                continue
        if marker.is_file():
            existing = json.loads(marker.read_text("utf-8"))
            if existing.get("dropped") == current:
                summaries.append(existing | {"skipped": True})
                continue
        started = time.monotonic()
        doc_ids: list[bytes] = []
        codes: list[int] = []
        for f in sorted(s2_decisions_dir(out_root, domain).glob("*.parquet")):
            table = pq.read_table(f, columns=["doc_id", "decision"])
            for did, dec in zip(table.column("doc_id").to_pylist(), table.column("decision").to_pylist()):
                doc_ids.append(bytes.fromhex(did))
                codes.append(0 if dec == S2_KEEP else 1)
        write_drops_lookup(out_dir, doc_ids, codes)
        summary = {
            "stage": "s2_lookup",
            "domain": domain,
            "kept": 0,
            "dropped": len(doc_ids),
            "elapsed_sec": round(time.monotonic() - started, 3),
            "errors": {},
            "skipped": False,
        }
        write_summary(marker, summary)
        summaries.append(summary)
    return summaries


def s2_decisions_dropped_total(out_root: Path) -> int:
    total = 0
    for f in sorted((out_root / "s2_lookup").rglob("lookup.summary.json")):
        try:
            total += int(json.loads(f.read_text("utf-8")).get("dropped") or 0)
        except (json.JSONDecodeError, OSError):
            continue
    return total


def process_shard_s2_merge(
    s1_path: Path, out_root: Path, source: str, relpath: str
) -> dict[str, Any]:
    out_path, summary_path = s2_output_paths(out_root, source, relpath)
    if summary_path.is_file() and out_path.is_file():
        summary = json.loads(summary_path.read_text("utf-8"))
        if (
            summary.get("decisions_dropped") == s2_decisions_dropped_total(out_root)
            and summary.get("input_fingerprint") == fingerprint(s1_path)
        ):
            summary["skipped"] = True
            return summary

    started = time.monotonic()
    in_rows = out_rows = error_rows = 0
    decisions: Counter[str] = Counter()
    writer = BatchWriter(out_path, S2_SCHEMA)
    lookups: dict[str, Any] = {}
    try:
        pf = pq.ParquetFile(s1_path)
        for batch in pf.iter_batches(batch_size=2048):
            rows = batch.to_pylist()
            by_domain: dict[str, list[int]] = {}
            for i, row in enumerate(rows):
                by_domain.setdefault(row["domain"], []).append(i)
            for domain, idxs in by_domain.items():
                lk = lookups.get(domain)
                if lk is None:
                    lk = lookups[domain] = load_drops_lookup(s2_lookup_dir(out_root, domain))
                ids = np.array([bytes.fromhex(rows[i]["doc_id"]) for i in idxs], dtype="V16")
                codes = drops_for(ids, lk)
                for i, code in zip(idxs, codes):
                    decision = S2_DROP if int(code) == 1 else S2_KEEP
                    decisions[decision] += 1
                    rows[i]["s2_decision"] = decision
                    writer.add(rows[i])
                    out_rows += 1
            in_rows += len(rows)
        out_bytes = writer.close()
    except Exception as exc:
        writer.abort()
        summary = {
            "stage": "s2_merge",
            "source": source,
            "relpath": relpath,
            "domain": "?",
            "in_rows": in_rows,
            "out_rows": out_rows,
            "error_rows": error_rows,
            "output_bytes": 0,
            "decisions": dict(decisions),
            "elapsed_sec": round(time.monotonic() - started, 3),
            "errors": {f"shard_error:{type(exc).__name__}": 1},
            "skipped": False,
        }
        write_summary(summary_path, summary)
        return summary

    summary = {
        "stage": "s2_merge",
        "source": source,
        "relpath": relpath,
        "domain": "?",
        "in_rows": in_rows,
        "out_rows": out_rows,
        "error_rows": error_rows,
        "output_bytes": out_bytes,
        "decisions": dict(decisions),
        "decisions_dropped": s2_decisions_dropped_total(out_root),
        "input_fingerprint": fingerprint(s1_path),
        "elapsed_sec": round(time.monotonic() - started, 3),
        "mb_per_sec": round(out_bytes / max(time.monotonic() - started, 1e-9) / 1e6, 2),
        "errors": {},
        "skipped": False,
    }
    write_summary(summary_path, summary)
    return summary


# ---------------------------------------------------------------- orchestration

def plan_s2_hash_tasks(out_root: Path, profiles: dict[str, dict[str, Any]], sources: list[str] | None) -> list[tuple]:
    tasks = []
    for source in profiles:
        if sources and source not in sources:
            continue
        src_dir = out_root / "s1" / source
        if not src_dir.is_dir():
            continue
        for path in sorted(src_dir.rglob("*.parquet")):
            rel = path.relative_to(src_dir).as_posix()
            tasks.append(("s2_hash", source, rel[: -len(".parquet")]))
    return sorted(tasks)


def plan_s2_reduce_tasks(
    out_root: Path, profiles: dict[str, dict[str, Any]], cfg: dict[str, Any], sources: list[str] | None
) -> tuple[list[tuple], dict[str, list[str]]]:
    num_buckets = int(cfg["s2"]["num_buckets"])
    rng = int(cfg["s2"]["reduce_range"])
    domains: dict[str, list[str]] = {}
    for source, prof in profiles.items():
        if sources and source not in sources:
            continue
        if not (out_root / "s1" / source).is_dir():
            continue
        domains.setdefault(prof["domain"], []).append(source)
    tasks = []
    for domain in sorted(domains):
        for lo in range(0, num_buckets, rng):
            tasks.append(("s2_reduce", domain, lo, lo + rng))
    return sorted(tasks), {d: sorted(s) for d, s in domains.items()}


def plan_s2_merge_tasks(out_root: Path, profiles: dict[str, dict[str, Any]], sources: list[str] | None) -> list[tuple]:
    return plan_s2_hash_tasks(out_root, profiles, sources)


def run_s2(
    out_root: Path,
    profiles: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
    workers: int,
    sources: list[str] | None = None,
    limit: int = 0,
) -> Path:
    summaries: list[dict[str, Any]] = []

    tasks = plan_s2_hash_tasks(out_root, profiles, sources)
    if limit:
        tasks = tasks[:limit]
    log.info("s2 pass1 (hash): %d shard tasks", len(tasks))
    summaries += run_tasks(tasks, {"out_root": out_root, "cfg": cfg}, _worker_s2_hash, workers, "s2_hash")

    reduce_tasks, domain_sources = plan_s2_reduce_tasks(out_root, profiles, cfg, sources)
    log.info("s2 pass2 (reduce): %d bucket-range tasks", len(reduce_tasks))
    summaries += run_tasks(
        reduce_tasks,
        {"out_root": out_root, "cfg": cfg, "domain_sources": domain_sources},
        _worker_s2_reduce,
        min(workers, 32),
        "s2_reduce",
    )

    domains = sorted(domain_sources)
    summaries += build_s2_lookups(out_root, domains)

    merge_tasks = plan_s2_merge_tasks(out_root, profiles, sources)
    if limit:
        merge_tasks = merge_tasks[:limit]
    log.info("s2 pass3 (merge): %d shard tasks", len(merge_tasks))
    summaries += run_tasks(
        merge_tasks, {"out_root": out_root}, _worker_s2_merge, workers, "s2_merge"
    )
    write_stage_summary(out_root, "s2", summaries)
    return summaries


def _worker_s2_hash(args: tuple[tuple, dict[str, Any]]) -> dict[str, Any]:
    task, ctx = args
    _, source, relpath = task
    try:
        return process_shard_s2_hash(
            ctx["out_root"] / "s1" / source / (relpath + ".parquet"),
            ctx["out_root"],
            source,
            relpath,
            int(ctx["cfg"]["s2"]["num_buckets"]),
        )
    except Exception as exc:
        return fatal_summary("s2_hash", f"{source}/{relpath}", exc, time.monotonic())


def _worker_s2_reduce(args: tuple[tuple, dict[str, Any]]) -> dict[str, Any]:
    task, ctx = args
    _, domain, lo, hi = task
    try:
        return process_s2_reduce(
            ctx["out_root"], domain, lo, hi, ctx["cfg"], ctx["domain_sources"][domain]
        )
    except Exception as exc:
        return fatal_summary("s2_reduce", f"{domain}/{lo}-{hi}", exc, time.monotonic())


def _worker_s2_merge(args: tuple[tuple, dict[str, Any]]) -> dict[str, Any]:
    task, ctx = args
    _, source, relpath = task
    try:
        return process_shard_s2_merge(
            ctx["out_root"] / "s1" / source / (relpath + ".parquet"),
            ctx["out_root"],
            source,
            relpath,
        )
    except Exception as exc:
        return fatal_summary("s2_merge", f"{source}/{relpath}", exc, time.monotonic())
