from __future__ import annotations

import json
import multiprocessing as mp
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
    CSIG_SUMMARY,
    CSig,
    CText,
    DocStore,
    S3_DROP,
    S3_KEEP,
    UnionFind,
    band_keys,
    build_csig,
    build_docstore,
    check_memory_cap,
    doc_bin_to_id,
    doc_id_to_bin,
    drops_for,
    empty_table,
    fatal_summary,
    fingerprint,
    first_bytes,
    jaccard_sets,
    load_drops_lookup,
    minhash_signature,
    rank_of,
    run_tasks,
    shingle_hashes,
    source_rank_map,
    stack_u8_to_void,
    unique_shingle_set,
    void_to_stack_u8,
    write_drops_lookup,
    write_stage_summary,
)
from .dedup_exact import S2_SCHEMA
from .ingest import BatchWriter, write_summary

log = logging.getLogger("yuformer.scale.dedup")

S3_SCHEMA: pa.Schema = S2_SCHEMA.append(
    pa.field("s3_decision", pa.string(), nullable=False)
)

SIG_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("sig", pa.binary(512), nullable=False),
    ]
)

LSH_PART_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("bucket", pa.int32(), nullable=False),
        pa.field("k0", pa.uint64(), nullable=False),
        pa.field("k1", pa.uint64(), nullable=False),
        pa.field("d0", pa.uint64(), nullable=False),
        pa.field("d1", pa.uint64(), nullable=False),
    ]
)

PAIRS_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("a0", pa.uint64(), nullable=False),
        pa.field("a1", pa.uint64(), nullable=False),
        pa.field("b0", pa.uint64(), nullable=False),
        pa.field("b1", pa.uint64(), nullable=False),
    ]
)

S3_DECISIONS_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("decision", pa.string(), nullable=False),
        pa.field("dup_of", pa.string(), nullable=False),
    ]
)


def _docbucket(doc_id_bin: bytes) -> str:
    return f"{doc_id_bin[0]:02x}"


def s3_sig_summary_path(out_root: Path, source: str, relpath: str) -> Path:
    base = out_root / "s3_sig" / "_summaries" / source / relpath
    return base.with_name(base.name + ".summary.json")


def s3_lsh_part_path(out_root: Path, domain: str, band: int, source: str, relpath: str) -> Path:
    return out_root / "s3_lsh" / domain / f"b{band}" / source / (relpath + ".parquet")


def s3_pairs_dir(out_root: Path, domain: str, band: int) -> Path:
    return out_root / "s3_pairs" / domain / f"b{band}"


def s3_cand_path(out_root: Path, domain: str) -> Path:
    return out_root / "s3_cand" / f"{domain}.npy"


def s3_cand_summary_path(out_root: Path, domain: str) -> Path:
    return out_root / "s3_cand" / f"{domain}.summary.json"


def s3_pairs_bydoc_dir(out_root: Path, domain: str) -> Path:
    return out_root / "s3_pairs_bydoc" / domain


def s3_cand_parts_dir(out_root: Path, domain: str, source: str, relpath: str) -> Path:
    return out_root / "s3_cand_parts" / domain / source / (relpath + ".parquet")


def s3_store_dir(out_root: Path, domain: str) -> Path:
    return out_root / "s3_store" / domain


def s3_verified_dir(out_root: Path, domain: str) -> Path:
    return out_root / "s3_verified" / domain


def s3_decisions_path(out_root: Path, domain: str) -> Path:
    return out_root / "s3_decisions" / f"{domain}.parquet"


def s3_lookup_dir(out_root: Path, domain: str) -> Path:
    return out_root / "s3_lookup" / domain


def s3_output_paths(out_root: Path, source: str, relpath: str) -> tuple[Path, Path]:
    base = out_root / "s3" / source / relpath
    return base.with_name(base.name + ".parquet"), base.with_name(base.name + ".summary.json")


# ---------------------------------------------------------------- pass 1: signature

class _DomainBuf:
    def __init__(self) -> None:
        self.sig_ids: list[str] = []
        self.sigs: list[bytes] = []
        self.sig_buckets: list[str] = []
        self.lsh: dict[int, list[tuple[int, int, int, int, int]]] = {}


def s3_candidate_total(out_root: Path, domains: list[str]) -> int:
    total = 0
    for domain in domains:
        p = s3_cand_summary_path(out_root, domain)
        if p.is_file():
            try:
                total += int(json.loads(p.read_text("utf-8")).get("candidate_docs") or 0)
            except (json.JSONDecodeError, OSError):
                continue
    return total


def process_shard_s3_sig(
    s2_path: Path, out_root: Path, source: str, relpath: str, cfg: dict[str, Any]
) -> dict[str, Any]:
    s3 = cfg["s3"]
    summary_path = s3_sig_summary_path(out_root, source, relpath)
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text("utf-8"))
        if summary.get("input_fingerprint") == fingerprint(s2_path):
            summary["skipped"] = True
            return summary

    started = time.monotonic()
    in_rows = sig_rows = truncated_docs = short_docs = 0
    text_bytes = 0
    outputs: list[str] = []
    bufs: dict[str, _DomainBuf] = {}
    try:
        pf = pq.ParquetFile(s2_path)
        for batch in pf.iter_batches(batch_size=2048):
            for row in batch.to_pylist():
                in_rows += 1
                if row.get("s2_decision") != "keep":
                    continue
                text = row["text"]
                text_bytes += len(text.encode("utf-8"))
                if len(text) > s3["truncate_chars"]:
                    text = text[: s3["truncate_chars"]]
                    truncated_docs += 1
                hashes = shingle_hashes(text, s3["shingle_n"])
                if len(hashes) == 0:
                    short_docs += 1
                    continue
                sig = minhash_signature(hashes, s3["sig_perms"])
                sig_bytes = sig.tobytes()
                d = doc_id_to_bin(row["doc_id"])
                d0 = int.from_bytes(d[:8], "little")
                d1 = int.from_bytes(d[8:], "little")
                buf = bufs.get(row["domain"])
                if buf is None:
                    buf = bufs[row["domain"]] = _DomainBuf()
                buf.sig_ids.append(row["doc_id"])
                buf.sigs.append(sig_bytes)
                buf.sig_buckets.append(_docbucket(d))
                keys = band_keys(sig_bytes, s3["bands"], s3["rows_per_band"])
                for band, key in enumerate(keys):
                    k0 = int.from_bytes(key[:8], "big")
                    k1 = int.from_bytes(key[8:], "big")
                    bucket = k0 % s3["lsh_buckets"]
                    buf.lsh.setdefault(band, []).append((bucket, k0, k1, d0, d1))
                sig_rows += 1

        for domain, buf in bufs.items():
            sig_base = out_root / "s3_sig" / domain
            for bucket in sorted(set(buf.sig_buckets)):
                idx = [i for i, b in enumerate(buf.sig_buckets) if b == bucket]
                path = sig_base / bucket / source / (relpath + ".parquet")
                _write_sig_file(path, [buf.sig_ids[i] for i in idx], [buf.sigs[i] for i in idx])
                outputs.append(str(path.relative_to(out_root)))
            for band, rows in buf.lsh.items():
                rows.sort()
                path = s3_lsh_part_path(out_root, domain, band, source, relpath)
                _write_lsh_part(path, rows)
                outputs.append(str(path.relative_to(out_root)))
    except Exception as exc:
        summary = {
            "stage": "s3_sig",
            "source": source,
            "relpath": relpath,
            "in_rows": in_rows,
            "sig_rows": sig_rows,
            "elapsed_sec": round(time.monotonic() - started, 3),
            "errors": {f"shard_error:{type(exc).__name__}": 1},
            "skipped": False,
        }
        write_summary(summary_path, summary)
        return summary

    elapsed = time.monotonic() - started
    summary = {
        "stage": "s3_sig",
        "source": source,
        "relpath": relpath,
        "input_fingerprint": fingerprint(s2_path),
        "domain": ",".join(sorted(bufs)) if bufs else "?",
        "in_rows": in_rows,
        "sig_rows": sig_rows,
        "truncated_docs": truncated_docs,
        "short_docs": short_docs,
        "text_bytes": text_bytes,
        "elapsed_sec": round(elapsed, 3),
        "mb_per_sec": round(text_bytes / max(elapsed, 1e-9) / 1e6, 2),
        "outputs": sorted(outputs),
        "errors": {},
        "skipped": False,
    }
    write_summary(summary_path, summary)
    return summary


def _write_sig_file(path: Path, doc_ids: list[str], sigs: list[bytes]) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    table = pa.table({"doc_id": doc_ids, "sig": sigs}, schema=SIG_SCHEMA)
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def _write_lsh_part(path: Path, rows: list[tuple[int, int, int, int, int]]) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    table = pa.table(
        {
            "bucket": np.array([r[0] for r in rows], dtype=np.int32),
            "k0": np.array([r[1] for r in rows], dtype=np.uint64),
            "k1": np.array([r[2] for r in rows], dtype=np.uint64),
            "d0": np.array([r[3] for r in rows], dtype=np.uint64),
            "d1": np.array([r[4] for r in rows], dtype=np.uint64),
        },
        schema=LSH_PART_SCHEMA,
    )
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


# ---------------------------------------------------------------- pass 2: LSH band reduce

def process_s3_band_reduce(
    out_root: Path,
    domain: str,
    band: int,
    lo: int,
    hi: int,
    cfg: dict[str, Any],
    expected_sources: list[str],
) -> dict[str, Any]:
    s3 = cfg["s3"]
    cap_gb = float(s3["memory_cap_gb"])
    lsh_buckets = int(s3["lsh_buckets"])
    group_cap = int(s3["group_pair_cap"])
    task_id = f"{domain}/b{band}/{lo}-{hi}"
    out_dir = s3_pairs_dir(out_root, domain, band)
    summary_paths = [
        out_dir / f"{b:04d}.summary.json" for b in range(lo, min(hi, lsh_buckets))
    ]
    if all(p.is_file() for p in summary_paths):
        existing = json.loads(summary_paths[0].read_text("utf-8"))
        if existing.get("domain_sources") == expected_sources:
            return {
                "stage": "s3_band",
                "task": task_id,
                "domain": domain,
                "pair_rows": sum(int(json.loads(p.read_text("utf-8")).get("pair_rows") or 0) for p in summary_paths),
                "groups": 0,
                "elapsed_sec": 0.0,
                "errors": {},
                "skipped": True,
            }

    started = time.monotonic()
    part_root = out_root / "s3_lsh" / domain / f"b{band}"
    files = sorted(part_root.rglob("*.parquet")) if part_root.is_dir() else []
    try:
        rec = _load_lsh_range(files, lo, hi, cap_gb)
        pairs, groups = _emit_pairs(rec, group_cap)
        _write_pairs(out_dir, lo, hi, lsh_buckets, pairs, expected_sources)
    except Exception as exc:
        summary = fatal_summary("s3_band", task_id, exc, started)
        summary["domain"] = domain
        write_summary(out_dir / f"reduce-{lo:04d}-{hi:04d}.error.json", summary)
        return summary

    summary = {
        "stage": "s3_band",
        "task": task_id,
        "domain": domain,
        "groups": groups,
        "pair_rows": len(pairs),
        "parts": len(files),
        "elapsed_sec": round(time.monotonic() - started, 3),
        "errors": {},
        "skipped": False,
    }
    return summary


def _load_lsh_range(files: list[Path], lo: int, hi: int, cap_gb: float) -> np.ndarray:
    import pyarrow.dataset as ds

    recs: list[np.ndarray] = []
    total = 0
    for f in files:
        dset = ds.dataset(f, format="parquet")
        expr = (ds.field("bucket") >= lo) & (ds.field("bucket") < hi)
        for batch in dset.to_batches(batch_size=65536, filter=expr):
            n = batch.num_rows
            if n == 0:
                continue
            cols = [batch.column(i).to_numpy(zero_copy_only=False) for i in range(5)]
            rec = np.empty(n, dtype=[("b", "<i4"), ("k", "V16"), ("d", "V16")])
            rec["b"] = cols[0]
            rec["k"] = stack_u8_to_void([cols[1], cols[2]], 16)
            rec["d"] = stack_u8_to_void([cols[3], cols[4]], 16)
            recs.append(rec)
            total += n * rec.dtype.itemsize
            check_memory_cap(total, cap_gb, f"s3 band reduce load [{lo},{hi})")
    if not recs:
        return np.empty(0, dtype=[("b", "<i4"), ("k", "V16"), ("d", "V16")])
    rec = np.concatenate(recs)
    return np.sort(rec)


def _emit_pairs(rec: np.ndarray, group_cap: int) -> tuple[list[tuple[int, bytes, bytes]], int]:
    pairs: list[tuple[int, bytes, bytes]] = []
    if len(rec) == 0:
        return pairs, 0
    same = (rec["b"][1:] == rec["b"][:-1]) & (rec["k"][1:] == rec["k"][:-1])
    ends = np.append(np.flatnonzero(~same) + 1, len(rec))
    groups = 0
    prev = 0
    for nxt in ends:
        g = nxt - prev
        if g >= 2:
            groups += 1
            bucket = int(rec["b"][prev])
            docs = rec["d"][prev:nxt]
            if g <= group_cap:
                for i in range(g):
                    di = docs[i].tobytes()
                    for j in range(i + 1, g):
                        pairs.append((bucket, di, docs[j].tobytes()))
            else:
                d0 = docs[0].tobytes()
                for i in range(1, g):
                    pairs.append((bucket, d0, docs[i].tobytes()))
        prev = nxt
    return pairs, groups


def _write_pairs(
    out_dir: Path,
    lo: int,
    hi: int,
    lsh_buckets: int,
    pairs: list[tuple[int, bytes, bytes]],
    expected_sources: list[str],
) -> None:
    import os

    by_bucket: dict[int, list[tuple[bytes, bytes]]] = {
        b: [] for b in range(lo, min(hi, lsh_buckets))
    }
    for bucket, a, b in pairs:
        by_bucket[bucket].append((a, b))
    out_dir.mkdir(parents=True, exist_ok=True)
    for b, rows in by_bucket.items():
        rows.sort()
        path = out_dir / f"{b:04d}.parquet"
        tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
        if rows:
            arr = np.frombuffer(
                b"".join(a + b_ for a, b_ in rows), dtype=np.uint64
            ).reshape(-1, 4)
            table = pa.table(
                {
                    "a0": arr[:, 0],
                    "a1": arr[:, 1],
                    "b0": arr[:, 2],
                    "b1": arr[:, 3],
                },
                schema=PAIRS_SCHEMA,
            )
        else:
            table = empty_table(PAIRS_SCHEMA)
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)
        write_summary(
            out_dir / f"{b:04d}.summary.json",
            {
                "stage": "s3_band",
                "bucket": b,
                "pair_rows": len(rows),
                "domain_sources": expected_sources,
            },
        )


# ---------------------------------------------------------------- candidates + bydoc pairs

def build_candidate_sets(
    out_root: Path, domains: list[str], cfg: dict[str, Any]
) -> list[dict[str, Any]]:
    cap_gb = float(cfg["s3"]["pairs_cap_gb"])
    # 2026-09-20 修复：原实现把"逐文件 unique 数组"全部驻留内存再 concat，同一文档在
    # N 个 pair 文件中出现就被计 N 次——web 域实测增长到 745GB+ 且仍在涨（病理线性）。
    # 改为分块归并：每 64 个文件做一次块内 np.unique 合并，峰值 ≈ 真实 unique 规模 × 小系数。
    consolidate_every = 64
    summaries = []
    for domain in domains:
        summary_path = s3_cand_summary_path(out_root, domain)
        if summary_path.is_file() and s3_cand_path(out_root, domain).is_file():
            summaries.append(json.loads(summary_path.read_text("utf-8")) | {"skipped": True})
            continue
        started = time.monotonic()
        chunk: list[np.ndarray] = []
        merged: list[np.ndarray] = []
        total_pairs = 0
        for f in sorted((out_root / "s3_pairs" / domain).rglob("*.parquet")):
            table = pq.read_table(f)
            if table.num_rows == 0:
                continue
            arr = np.column_stack(
                [table.column(c).to_numpy() for c in ("a0", "a1", "b0", "b1")]
            )
            a = stack_u8_to_void([arr[:, 0], arr[:, 1]], 16)
            b = stack_u8_to_void([arr[:, 2], arr[:, 3]], 16)
            chunk.append(np.unique(a))
            chunk.append(np.unique(b))
            if len(chunk) >= consolidate_every:
                merged.append(np.unique(np.concatenate(chunk)))
                chunk = []
            total_pairs += len(a)
            check_memory_cap(
                (sum(len(u) for u in merged) + sum(len(u) for u in chunk)) * 16,
                cap_gb,
                f"candidate set {domain}",
            )
        if chunk:
            merged.append(np.unique(np.concatenate(chunk)))
        cand = np.unique(np.concatenate(merged)) if merged else np.empty(0, dtype="V16")
        s3_cand_path(out_root, domain).parent.mkdir(parents=True, exist_ok=True)
        np.save(s3_cand_path(out_root, domain), cand)
        # 2026-09-20 修复：pair 数据不再驻留内存（web 域 pairs 内存态 ~0.5-1TB，曾触发
        # OOM killer SIGKILL）；bydoc 分桶改为从 pair 文件流式重写，内存与 pair 总量解耦。
        _write_pairs_bydoc(out_root, domain, sorted((out_root / "s3_pairs" / domain).rglob("*.parquet")))
        summary = {
            "stage": "s3_cand",
            "domain": domain,
            "candidate_docs": int(len(cand)),
            "pair_rows": int(total_pairs),
            "elapsed_sec": round(time.monotonic() - started, 3),
            "errors": {},
            "skipped": False,
        }
        write_summary(summary_path, summary)
        summaries.append(summary)
    return summaries


def _write_pairs_bydoc(out_root: Path, domain: str, pair_files: list[Path]) -> None:
    """流式 bydoc 分桶：按 a0 首字节分 256 桶，每桶缓冲到 FLUSH_ROWS 即落盘一个 chunk。
    内存与 pair 总量解耦（2026-09-20 修复，原实现全局 concat+argsort 在 web 域 OOM 被杀）。"""
    import os

    out_root_dir = s3_pairs_bydoc_dir(out_root, domain)
    FLUSH_ROWS = 200_000
    bufs: dict[int, list[np.ndarray]] = {}
    buf_rows: dict[int, int] = {}
    chunk_idx: dict[int, int] = {}

    def flush(byte: int) -> None:
        rows = bufs.pop(byte)
        buf_rows.pop(byte, None)
        arr = np.concatenate(rows, axis=0)  # rows 为 [k,2] 数组，沿行拼接
        d = out_root_dir / f"{byte:02x}"
        d.mkdir(parents=True, exist_ok=True)
        idx = chunk_idx.get(byte, 0)
        chunk_idx[byte] = idx + 1
        path = d / f"chunk-{idx:05d}.parquet"
        tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
        a = np.stack(void_to_stack_u8(arr[:, 0], 2), axis=1)
        b = np.stack(void_to_stack_u8(arr[:, 1], 2), axis=1)
        table = pa.table(
            {
                "a0": a[:, 0],
                "a1": a[:, 1],
                "b0": b[:, 0],
                "b1": b[:, 1],
            },
            schema=PAIRS_SCHEMA,
        )
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, path)

    for f in pair_files:
        table = pq.read_table(f)
        if table.num_rows == 0:
            continue
        arr = np.column_stack(
            [table.column(c).to_numpy() for c in ("a0", "a1", "b0", "b1")]
        )
        a = stack_u8_to_void([arr[:, 0], arr[:, 1]], 16)
        b = stack_u8_to_void([arr[:, 2], arr[:, 3]], 16)
        pairs = np.column_stack([a, b])
        first_bytes = pairs.view(np.uint8).reshape(len(pairs), 2, 16)[:, 0, 0]
        for byte in np.unique(first_bytes):
            rows = pairs[first_bytes == byte]
            bufs.setdefault(int(byte), []).append(rows)
            n = buf_rows.get(int(byte), 0) + len(rows)
            buf_rows[int(byte)] = n
            if n >= FLUSH_ROWS:
                flush(int(byte))
    for byte in list(bufs):
        flush(byte)


# ---------------------------------------------------------------- collect candidate texts

def process_shard_s3_collect(
    s2_path: Path, out_root: Path, source: str, relpath: str, cfg: dict[str, Any]
) -> dict[str, Any]:
    summary_path = out_root / "s3_collect" / "_summaries" / source / (relpath + ".summary.json")
    domains = [d for d in (out_root / "s3_cand").glob("*.npy")]
    domain_names = sorted(p.stem for p in domains)
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text("utf-8"))
        if (
            summary.get("input_fingerprint") == fingerprint(s2_path)
            and summary.get("candidate_total") == s3_candidate_total(out_root, domain_names)
        ):
            summary["skipped"] = True
            return summary

    started = time.monotonic()
    in_rows = out_rows = 0
    cands: dict[str, np.memmap] = {}
    writers: dict[str, Any] = {}
    try:
        pf = pq.ParquetFile(s2_path)
        for batch in pf.iter_batches(batch_size=2048, columns=["doc_id", "domain", "source", "text", "s2_decision"]):
            rows = batch.to_pylist()
            ids = np.array([doc_id_to_bin(r["doc_id"]) for r in rows], dtype="V16")
            keep = np.array([r["s2_decision"] == "keep" for r in rows])
            matched = np.zeros(len(rows), dtype=bool)
            for domain in sorted({r["domain"] for r in rows}):
                idx = [i for i, r in enumerate(rows) if r["domain"] == domain]
                arr = cands.get(domain)
                if arr is None:
                    arr = cands[domain] = np.load(s3_cand_path(out_root, domain), mmap_mode="r")
                if len(arr) == 0:
                    continue
                sub = ids[idx]
                pos = np.searchsorted(arr, sub)
                pos = np.clip(pos, 0, len(arr) - 1)
                hit = arr[pos] == sub
                for i, h in zip(idx, hit):
                    if h and keep[i]:
                        matched[i] = True
            for i, row in enumerate(rows):
                in_rows += 1
                if not matched[i]:
                    continue
                domain = row["domain"]
                writer = writers.get(domain)
                if writer is None:
                    path = s3_cand_parts_dir(out_root, domain, source, relpath)
                    schema = pa.schema(
                        [
                            pa.field("doc_id", pa.string(), nullable=False),
                            pa.field("source", pa.string(), nullable=False),
                            pa.field("text", pa.string(), nullable=False),
                        ]
                    )
                    writer = writers[domain] = _CollectWriter(path, schema)
                writer.add({"doc_id": row["doc_id"], "source": row["source"], "text": row["text"]})
                out_rows += 1
        for writer in writers.values():
            writer.finish()
    except Exception as exc:
        for writer in writers.values():
            writer.abort()
        summary = {
            "stage": "s3_collect",
            "source": source,
            "relpath": relpath,
            "in_rows": in_rows,
            "out_rows": out_rows,
            "elapsed_sec": round(time.monotonic() - started, 3),
            "errors": {f"shard_error:{type(exc).__name__}": 1},
            "skipped": False,
        }
        write_summary(summary_path, summary)
        return summary

    summary = {
        "stage": "s3_collect",
        "source": source,
        "relpath": relpath,
        "input_fingerprint": fingerprint(s2_path),
        "candidate_total": s3_candidate_total(out_root, domain_names),
        "domain": ",".join(sorted(writers)) if writers else "?",
        "in_rows": in_rows,
        "out_rows": out_rows,
        "candidate_rows": out_rows,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "errors": {},
        "skipped": False,
    }
    write_summary(summary_path, summary)
    return summary


class _CollectWriter:
    def __init__(self, path: Path, schema: pa.Schema):
        self.path = path
        self.schema = schema
        self.rows: list[dict[str, Any]] = []

    def add(self, row: dict[str, Any]) -> None:
        self.rows.append(row)

    def finish(self) -> None:
        if not self.rows:
            return
        self.rows.sort(key=lambda r: r["doc_id"])
        writer = BatchWriter(self.path, self.schema)
        try:
            for row in self.rows:
                writer.add(row)
            writer.close()
        except Exception:
            writer.abort()
            raise

    def abort(self) -> None:
        self.rows = []


# ---------------------------------------------------------------- docstore + verify + decide

def build_docstores(
    out_root: Path,
    domain_sources: dict[str, list[str]],
    profiles: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    summaries = []
    for domain in sorted(domain_sources):
        part_root = out_root / "s3_cand_parts" / domain
        files = sorted(part_root.rglob("*.parquet")) if part_root.is_dir() else []
        store_sources = sorted(
            s for s, p in profiles.items() if p.get("domain") == domain
        )
        payload = build_docstore(
            files,
            s3_store_dir(out_root, domain),
            int(cfg["s3"]["docstore_run_rows"]),
            float(cfg["s3"]["memory_cap_gb"]),
            store_sources,
        )
        summaries.append(
            {
                "stage": "s3_docstore",
                "domain": domain,
                "candidate_docs": int(payload["docs"]),
                "output_bytes": int(payload["text_bytes"]),
                "errors": {},
                "skipped": False,
            }
        )
    return summaries


def process_s3_verify(
    out_root: Path,
    domain: str,
    bucket_byte: int,
    cfg: dict[str, Any],
    expected_sources: list[str],
) -> dict[str, Any]:
    s3 = cfg["s3"]
    summary_path = s3_verified_dir(out_root, domain) / f"{bucket_byte:02x}.summary.json"
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text("utf-8"))
        if existing.get("domain_sources") == expected_sources:
            existing["skipped"] = True
            return existing

    started = time.monotonic()
    try:
        pairs = _load_pairs_bydoc(out_root, domain, bucket_byte, int(s3["verify_buckets"]))
        check_memory_cap(
            len(pairs) * 32, float(s3["pairs_cap_gb"]), f"verify pairs {domain}/{bucket_byte:02x}"
        )
        csig_path = out_root / "s3_csig" / domain
        if not (csig_path / CSIG_SUMMARY).is_file():
            raise FileNotFoundError(f"csig not built for {domain}: {csig_path}")
        cs = CSig(csig_path)
        verified = _verify_pairs_sig(pairs, cs, bucket_byte, int(s3["verify_buckets"]), s3)
        _write_verified(out_root, domain, bucket_byte, verified, expected_sources)
    except Exception as exc:
        summary = fatal_summary("s3_verify", f"{domain}/{bucket_byte:02x}", exc, started)
        summary["domain"] = domain
        write_summary(summary_path, summary)
        return summary

    summary = {
        "stage": "s3_verify",
        "domain": domain,
        "task": f"{domain}/{bucket_byte:02x}",
        "pair_rows": int(len(pairs)),
        "verified_pairs": int(len(pairs)),
        "verified_passed": int(len(verified)),
        "domain_sources": expected_sources,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "errors": {},
        "skipped": False,
    }
    write_summary(summary_path, summary)
    return summary


def _verify_pairs_sig(
    pairs: np.ndarray,
    cs: "CSig",
    bucket_byte: int,
    vb: int,
    s3: dict[str, Any],
) -> list[tuple[bytes, bytes]]:
    """签名一致率判定：J_est = mean(sig_a == sig_b) ≥ sig_agree_threshold（默认 0.8，软阈值）。

    按 (a 精确首字节, b 精确首字节) 分组向量化；a 桶 = 首字节 % vb == bucket_byte。
    """
    threshold = float(s3.get("sig_agree_threshold", 0.8))
    verified: list[tuple[bytes, bytes]] = []
    if len(pairs) == 0:
        return verified
    a = pairs[:, 0]
    b = pairs[:, 1]
    fa = first_bytes(a).astype(np.int32)
    fb = first_bytes(b).astype(np.int32)
    for a_byte in np.unique(fa):
        if int(a_byte) % vb != bucket_byte:
            continue
        cs.pin(int(a_byte))
        a_ids_bin, a_sigs = cs._bin(int(a_byte))
        if a_ids_bin is None:
            continue
        rows_a = np.flatnonzero(fa == a_byte)
        pos_a = cs.locate(int(a_byte), a[rows_a])
        ok_a = pos_a >= 0
        rows_a, pos_a = rows_a[ok_a], pos_a[ok_a]
        sig_a = np.ascontiguousarray(a_sigs[pos_a]).view(np.uint32).reshape(len(pos_a), 128)
        sub_b = b[rows_a]
        sub_fa = fb[rows_a]
        for b_byte in np.unique(sub_fa):
            sel = np.flatnonzero(sub_fa == b_byte)
            b_ids_bin, b_sigs = cs._bin(int(b_byte))
            if b_ids_bin is None:
                continue
            pos_b = cs.locate(int(b_byte), sub_b[sel])
            ok_b = pos_b >= 0
            sel = sel[ok_b]
            pos_b = pos_b[ok_b]
            if len(sel) == 0:
                continue
            sig_b = np.ascontiguousarray(b_sigs[pos_b]).view(np.uint32).reshape(len(pos_b), 128)
            agree = (sig_a[sel] == sig_b).mean(axis=1)
            for i in np.flatnonzero(agree >= threshold):
                verified.append((a[rows_a[sel[i]]].tobytes(), sub_b[sel[i]].tobytes()))
    return verified


def _load_pairs_bydoc(out_root: Path, domain: str, bucket_byte: int, verify_buckets: int = 256) -> np.ndarray:
    base = s3_pairs_bydoc_dir(out_root, domain)
    # bydoc 已按 doc_id 首字节分目录：vb=256 只读本桶目录；vb<256 才跨目录过滤
    if verify_buckets == 256:
        dirs = [base / f"{bucket_byte:02x}"]
    else:
        dirs = sorted(p for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
    chunks = []
    for d in dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("chunk-*.parquet")):
            table = pq.read_table(f)
            if table.num_rows == 0:
                continue
            arr = np.column_stack([table.column(c).to_numpy() for c in ("a0", "a1", "b0", "b1")])
            a = stack_u8_to_void([arr[:, 0], arr[:, 1]], 16)
            b = stack_u8_to_void([arr[:, 2], arr[:, 3]], 16)
            keep = first_bytes(a).astype(np.int32) % verify_buckets == bucket_byte
            if not keep.any():
                continue
            chunks.append(np.column_stack([a[keep], b[keep]]))
    if not chunks:
        return np.empty((0, 2), dtype="V16")
    return np.concatenate(chunks)


def _verify_pairs(
    pairs: np.ndarray,
    texts: dict[bytes, str],
    srcs: dict[bytes, int],
    s3: dict[str, Any],
) -> list[tuple[bytes, bytes]]:
    threshold = float(s3["jaccard_threshold"])
    cap = int(s3["verify_shingle_cap"])
    n = int(s3["shingle_n"])
    # 判定性近似：验证用正文前缀（verify_text_chars，默认 2048 字符 ≈ 400 shingles）。
    # 0.8 相似的两篇在前 2048 字符的 Jaccard 期望仍 >0.6，远高于随机对 ~0，判别力足够。
    trunc = int(s3.get("verify_text_chars") or s3["truncate_chars"])
    sets: dict[bytes, np.ndarray] = {}
    verified: list[tuple[bytes, bytes]] = []
    for a, b in pairs:
        a_b, b_b = a.tobytes(), b.tobytes()
        sa = sets.get(a_b)
        if sa is None:
            sa = sets[a_b] = unique_shingle_set(texts[a_b][:trunc], n, cap)
        sb = sets.get(b_b)
        if sb is None:
            sb = sets[b_b] = unique_shingle_set(texts[b_b][:trunc], n, cap)
        if jaccard_sets(sa, sb) >= threshold:
            verified.append((a_b, b_b))
    return verified


def _write_verified(
    out_root: Path,
    domain: str,
    bucket_byte: int,
    verified: list[tuple[bytes, bytes]],
    expected_sources: list[str],
) -> None:
    import os

    out_dir = s3_verified_dir(out_root, domain)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{bucket_byte:02x}.parquet"
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    if verified:
        arr = np.frombuffer(b"".join(a + b for a, b in verified), dtype=np.uint64).reshape(-1, 4)
        table = pa.table(
            {
                "a0": arr[:, 0],
                "a1": arr[:, 1],
                "b0": arr[:, 2],
                "b1": arr[:, 3],
            },
            schema=PAIRS_SCHEMA,
        )
    else:
        table = empty_table(PAIRS_SCHEMA)
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def s3_verified_total(out_root: Path, domains: list[str]) -> int:
    total = 0
    for domain in domains:
        vdir = s3_verified_dir(out_root, domain)
        if not vdir.is_dir():
            continue
        for f in sorted(vdir.glob("*.summary.json")):
            try:
                total += int(json.loads(f.read_text("utf-8")).get("verified_passed") or 0)
            except (json.JSONDecodeError, OSError):
                continue
    return total


def s3_decisions_dropped_total(out_root: Path) -> int:
    total = 0
    for f in sorted((out_root / "s3_decisions").glob("*.summary.json")):
        try:
            total += int(json.loads(f.read_text("utf-8")).get("dropped") or 0)
        except (json.JSONDecodeError, OSError):
            continue
    return total


def _csig_srcs_task(args: tuple[Path, Path, Path, int]) -> int:
    """per-byte srcs.bin：ctext _idx（与 csig bin 同序）loc→part 行取 source_code，
    按 sorted position 填平。ctext part 顺序读，无散点 IO。"""
    ctext_dir, csig_dir, out_dir, byte = args
    ids_path = csig_dir / "_idx" / f"{byte:02x}.ids.bin"
    if not ids_path.is_file():
        return 0
    idx_ids = np.memmap(ids_path, dtype="V16", mode="r")
    loc_path = ctext_dir / "_idx" / f"{byte:02x}.loc.bin"
    if not loc_path.is_file():
        return 0
    loc = np.memmap(loc_path, dtype="<u8", mode="r")
    srcs = np.zeros(len(idx_ids), dtype="<u2")
    byte_s = f"{byte:02x}"
    for part in sorted((ctext_dir / byte_s).glob("part-*.parquet")):
        run_idx = int(part.stem.split("-")[1])
        table = pq.read_table(part, columns=["a0", "a1", "source_code"])
        arr = np.column_stack([table.column("a0").to_numpy(), table.column("a1").to_numpy()])
        ids = stack_u8_to_void([arr[:, 0], arr[:, 1]], 16)
        pos = np.searchsorted(idx_ids, ids)
        pos = np.clip(pos, 0, len(idx_ids) - 1)
        hit = idx_ids[pos] == ids
        # loc 校验：part run_idx 必须与 loc 高 32 位一致
        p, c = pos[hit], table.column("source_code").to_numpy()[hit]
        ok = (loc[p] >> 32) == np.uint64(run_idx)
        srcs[p[ok]] = c[ok]
    tmp = out_dir / f"{byte_s}.srcs.bin.tmp-{os.getpid()}"
    tmp.write_bytes(srcs.tobytes())
    os.replace(tmp, out_dir / f"{byte_s}.srcs.bin")
    return len(srcs)


def _uf_direct_pairs(out_root: Path, domain: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """候选对直并查集聚类（web 规模路径）：8.4B 对逐对验证 IO 不可行（~400TB 散点读），
    LSH 16 band(t≈0.71) 候选即判重边（CCNet 做法），簇 survivor 规则保质量。"""
    s3 = cfg["s3"]
    threshold = int(s3.get("uf_direct_pair_threshold", 1_000_000_000))
    cand_summary = s3_cand_summary_path(out_root, domain)
    pair_rows = 0
    if cand_summary.is_file():
        pair_rows = int(json.loads(cand_summary.read_text("utf-8")).get("pair_rows") or 0)
    if pair_rows < threshold:
        return {"skipped": True, "pair_rows": pair_rows, "threshold": threshold}
    # 幂等：决定已产出即跳过（重跑安全）
    if s3_decisions_path(out_root, domain).is_file() and (
        out_root / "s3_decisions" / f"{domain}.summary.json"
    ).is_file():
        existing = json.loads((out_root / "s3_decisions" / f"{domain}.summary.json").read_text("utf-8"))
        existing["skipped"] = True
        return existing
    csig_path = out_root / "s3_csig" / domain
    if not (csig_path / CSIG_SUMMARY).is_file():
        raise FileNotFoundError(f"csig not built: {csig_path}")

    work_dir = out_root / "s3_uf_work" / domain
    done_dir = work_dir / "done"
    done_dir.mkdir(parents=True, exist_ok=True)
    offsets = np.zeros(257, dtype=np.int64)
    for b in range(256):
        idx_path = csig_path / "_idx" / f"{b:02x}.ids.bin"
        offsets[b + 1] = offsets[b] + (idx_path.stat().st_size // 16 if idx_path.is_file() else 0)
    n_docs = int(offsets[-1])
    # parent/rank 常驻 RAM（uint32：n<4.29e9；memmap 随机写会打爆网络 fs——实测 30min 零进度）。
    # 断点续跑：每 4 个 byte dir 顺序 flush 一次（~10s/次可接受），启动时按已完成 dir 恢复。
    parent_path = work_dir / "parent.bin"
    rank_path = work_dir / "rank.bin"
    n_done = len(list(done_dir.glob("*"))) if done_dir.is_dir() else 0
    if n_done and parent_path.is_file() and parent_path.stat().st_size == n_docs * 4:
        parent = np.fromfile(parent_path, dtype="<u4")
        rank = np.fromfile(rank_path, dtype=np.int8)
    else:
        if done_dir.is_dir():
            for m in done_dir.glob("*"):
                m.unlink()
        parent = np.arange(n_docs, dtype="<u4")
        rank = np.zeros(n_docs, dtype=np.int8)

    def find(x: int) -> int:
        while True:
            px = int(parent[x])
            if px == x:
                return x
            parent[x] = parent[px]
            x = int(parent[x])

    cs = CSig(csig_path)
    edges = 0
    pending: list[int] = []
    pairs_dir = s3_pairs_bydoc_dir(out_root, domain)
    started = time.monotonic()
    for byte_dir in sorted(pairs_dir.iterdir()):
        if not byte_dir.is_dir():
            continue
        a_byte = int(byte_dir.name, 16)
        marker = done_dir / f"{a_byte:02x}"
        if marker.is_file():
            continue
        a_off = int(offsets[a_byte])
        a_ids, _ = cs._bin(a_byte)
        if a_ids is None:
            marker.write_text("empty")
            continue
        for f in sorted(byte_dir.glob("chunk-*.parquet")):
            table = pq.read_table(f)
            if table.num_rows == 0:
                continue
            arr = np.column_stack([table.column(c).to_numpy() for c in ("a0", "a1", "b0", "b1")])
            a = stack_u8_to_void([arr[:, 0], arr[:, 1]], 16)
            b = stack_u8_to_void([arr[:, 2], arr[:, 3]], 16)
            pos_a = cs.locate(a_byte, a)
            fb = first_bytes(b).astype(np.int32)
            pos_b = np.full(len(b), -1, dtype=np.int64)
            for bb in np.unique(fb):
                ids_b, _ = cs._bin(int(bb))
                if ids_b is None:
                    continue
                mask = fb == bb
                pos_b[mask] = cs.locate(int(bb), b[mask])
            ok = (pos_a >= 0) & (pos_b >= 0)
            na = (pos_a[ok] + a_off).tolist()
            nb = (pos_b[ok] + offsets[fb[ok]]).tolist()
            par = parent
            rk = rank
            for i in range(len(na)):
                x, y = na[i], nb[i]
                while True:
                    px = int(par[x])
                    if px == x:
                        break
                    par[x] = par[px]
                    x = int(par[x])
                while True:
                    py = int(par[y])
                    if py == y:
                        break
                    par[y] = par[py]
                    y = int(par[y])
                if x == y:
                    continue
                if rk[x] < rk[y]:
                    x, y = y, x
                par[y] = np.uint32(x)
                if rk[x] == rk[y]:
                    rk[x] += 1
            edges += int(ok.sum())
        pending.append(a_byte)
        if edges and a_byte % 4 == 3:
            tmp_p = parent_path.with_name(parent_path.name + f".tmp-{os.getpid()}")
            tmp_r = rank_path.with_name(rank_path.name + f".tmp-{os.getpid()}")
            parent.tofile(tmp_p)
            rank.tofile(tmp_r)
            os.replace(tmp_p, parent_path)
            os.replace(tmp_r, rank_path)
            for m in pending:
                (done_dir / f"{m:02x}").write_text(str(edges))
            pending.clear()
        log.info("uf_direct %s: byte %s done, %d edges, %.0fs", domain, byte_dir.name, edges, time.monotonic() - started)

    # 收尾：最终 flush + 标记剩余 dir（与 parent 状态一致）
    tmp_p = parent_path.with_name(parent_path.name + f".tmp-{os.getpid()}")
    tmp_r = rank_path.with_name(rank_path.name + f".tmp-{os.getpid()}")
    parent.tofile(tmp_p)
    rank.tofile(tmp_r)
    os.replace(tmp_p, parent_path)
    os.replace(tmp_r, rank_path)
    for m in pending:
        (done_dir / f"{m:02x}").write_text(str(edges))
    pending.clear()

    # 向量化路径压缩到不动点（每轮路径折半）
    while True:
        p2 = parent[parent]
        if not (p2 != parent).any():
            break
        parent[:] = p2
    roots, counts = np.unique(parent, return_counts=True)
    big = counts >= 2
    clusters = int(big.sum())
    comp_member_pos = np.flatnonzero(big[parent - roots.searchsorted(parent)] if False else np.isin(parent, roots[big]))
    del rank

    from .dedup_common import source_rank_map
    rank_map = source_rank_map(list(cfg.get("source_priority") or []))
    doc_ids_global = np.empty(n_docs, dtype="V16")
    for b in range(256):
        ids_b, _ = cs._bin(b)
        if ids_b is not None:
            doc_ids_global[offsets[b] : offsets[b + 1]] = ids_b
    member_ids = doc_ids_global[comp_member_pos]
    del doc_ids_global

    # source：srcs.bin（与 csig bin 同序）；缺失则先从 ctext 构建
    srcs_paths = [csig_path / "_idx" / f"{b:02x}.srcs.bin" for b in range(256)]
    if not all(p.is_file() for p in srcs_paths):
        ctext_dir = out_root / "s3_ctext" / domain
        with mp.Pool(processes=min(48, 256)) as pool:
            pool.map(
                _csig_srcs_task,
                [(ctext_dir, csig_path, csig_path / "_idx", b) for b in range(256)],
            )
    member_codes = np.zeros(len(member_ids), dtype="<u2")
    fb_m = first_bytes(member_ids).astype(np.int32)
    missing_srcs = [b for b in range(256) if not (csig_path / "_idx" / f"{b:02x}.srcs.bin").is_file()]
    if missing_srcs:
        # srcs.bin 未全覆盖（无 ctext 的小域/合成场景）：回退 docstore 点查
        store_fb = DocStore(s3_store_dir(out_root, domain))
        member_codes = store_fb.sources(member_ids)
    else:
        for bb in np.unique(fb_m):
            mask = fb_m == bb
            ids_mem = np.memmap(csig_path / "_idx" / f"{int(bb):02x}.ids.bin", dtype="V16", mode="r")
            pos = np.searchsorted(ids_mem, member_ids[mask])
            pos = np.clip(pos, 0, len(ids_mem) - 1)
            hit = ids_mem[pos] == member_ids[mask]
            pos = np.where(hit, pos, 0)
            codes = np.memmap(csig_path / "_idx" / f"{int(bb):02x}.srcs.bin", dtype="<u2", mode="r")[pos]
            member_codes[mask] = np.where(hit, codes, 0)
    sources = _load_source_names(s3_store_dir(out_root, domain))
    del member_ids

    root_of_member = parent[comp_member_pos]
    del parent
    # survivor 逐簇计算：先收集每簇成员索引
    from collections import defaultdict
    members_of: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(root_of_member.tolist()):
        members_of[r].append(i)
    decisions: list[tuple[bytes, str, str]] = []
    per_source_drops: Counter[str] = Counter()
    doc_ids_global = np.empty(n_docs, dtype="V16")
    for b in range(256):
        idx_b = csig_path / "_idx" / f"{b:02x}.ids.bin"
        if idx_b.is_file():
            doc_ids_global[offsets[b] : offsets[b + 1]] = np.memmap(idx_b, dtype="V16", mode="r")
    for r, members in members_of.items():
        best_key: tuple[int, bytes] | None = None
        best_did: bytes | None = None
        for i in members:
            did = doc_ids_global[comp_member_pos[i]].tobytes()
            src = sources[int(member_codes[i])] if int(member_codes[i]) < len(sources) else "?"
            key = (rank_of(rank_map, src), did)
            if best_key is None or key < best_key:
                best_key = key
                best_did = did
        for i in members:
            did = doc_ids_global[comp_member_pos[i]].tobytes()
            if did == best_did:
                continue
            src = sources[int(member_codes[i])] if int(member_codes[i]) < len(sources) else "?"
            per_source_drops[src] += 1
            decisions.append((did, S3_DROP, best_did.hex()))
    decisions.sort(key=lambda r: r[0])
    out_path = s3_decisions_path(out_root, domain)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "doc_id": [d.hex() for d, _, _ in decisions],
            "decision": [dec for _, dec, _ in decisions],
            "dup_of": [dup for _, _, dup in decisions],
        },
        schema=S3_DECISIONS_SCHEMA,
    )
    tmp = out_path.with_name(out_path.name + f".tmp-{os.getpid()}")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, out_path)
    write_drops_lookup(s3_lookup_dir(out_root, domain), [d for d, _, _ in decisions], [1] * len(decisions))
    summary = {
        "stage": "s3_uf_direct",
        "domain": domain,
        "clusters": clusters,
        "dropped": len(decisions),
        "pair_rows": edges,
        "per_source_drops": dict(per_source_drops),
        "elapsed_sec": round(time.monotonic() - started, 3),
        "errors": {},
        "skipped": False,
    }
    write_summary(out_root / "s3_decisions" / f"{domain}.summary.json", summary)
    return summary


def build_s3_decisions(
    out_root: Path,
    domains: list[str],
    cfg: dict[str, Any],
    domain_sources: dict[str, list[str]],
) -> list[dict[str, Any]]:
    cap_gb = float(cfg["s3"]["pairs_cap_gb"])
    rank_map = source_rank_map(list(cfg.get("source_priority") or []))
    summaries = []
    for domain in domains:
        expected = domain_sources[domain]
        out_path = s3_decisions_path(out_root, domain)
        summary_path = out_root / "s3_decisions" / f"{domain}.summary.json"
        if summary_path.is_file() and out_path.is_file():
            existing = json.loads(summary_path.read_text("utf-8"))
            if (
                existing.get("domain_sources") == expected
                and existing.get("verified_total") == s3_verified_total(out_root, domains)
            ):
                existing["skipped"] = True
                summaries.append(existing)
                continue
        started = time.monotonic()
        uf = UnionFind()
        total_pairs = 0
        for f in sorted(s3_verified_dir(out_root, domain).glob("*.parquet")):
            table = pq.read_table(f)
            if table.num_rows == 0:
                continue
            arr = np.column_stack([table.column(c).to_numpy() for c in ("a0", "a1", "b0", "b1")])
            a = stack_u8_to_void([arr[:, 0], arr[:, 1]], 16)
            b = stack_u8_to_void([arr[:, 2], arr[:, 3]], 16)
            total_pairs += len(a)
            check_memory_cap(total_pairs * 32, cap_gb, f"verified pairs {domain}")
            for x, y in zip(a, b):
                uf.union(x.tobytes(), y.tobytes())
        store_dir = s3_store_dir(out_root, domain)
        # 只用 legacy DocStore：ctext 网络 fs 随机挂起（实测），小域四 bin 稳定可读
        store: Any = DocStore(store_dir)
        sources = _load_source_names(store_dir)
        decisions: list[tuple[bytes, str, str]] = []
        clusters = 0
        per_source_drops: Counter[str] = Counter()
        for comp in uf.components():
            if len(comp) < 2:
                continue
            clusters += 1
            ids = np.array(comp, dtype="V16")
            codes = store.sources(ids)
            best_idx = 0
            best_key: tuple[int, str] | None = None
            for i in range(len(comp)):
                src = sources[int(codes[i])] if int(codes[i]) < len(sources) else "?"
                key = (rank_of(rank_map, src), comp[i].hex())
                if best_key is None or key < best_key:
                    best_key = key
                    best_idx = i
            survivor = comp[best_idx].hex()
            for i, d in enumerate(comp):
                if i == best_idx:
                    continue
                src = sources[int(codes[i])] if int(codes[i]) < len(sources) else "?"
                per_source_drops[src] += 1
                decisions.append((d, S3_DROP, survivor))
        decisions.sort(key=lambda r: r[0])
        table = pa.table(
            {
                "doc_id": [doc_bin_to_id(d) for d, _, _ in decisions],
                "decision": [dec for _, dec, _ in decisions],
                "dup_of": [dup for _, _, dup in decisions],
            },
            schema=S3_DECISIONS_SCHEMA,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(out_path.name + f".tmp-{os.getpid()}")
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, out_path)
        write_drops_lookup(
            s3_lookup_dir(out_root, domain), [d for d, _, _ in decisions], [1] * len(decisions)
        )
        summary = {
            "stage": "s3_decide",
            "domain": domain,
            "clusters": clusters,
            "dropped": len(decisions),
            "verified_pairs": int(total_pairs),
            "verified_total": s3_verified_total(out_root, domains),
            "per_source_drops": dict(per_source_drops),
            "domain_sources": expected,
            "elapsed_sec": round(time.monotonic() - started, 3),
            "errors": {},
            "skipped": False,
        }
        write_summary(summary_path, summary)
        summaries.append(summary)
    return summaries


def _load_source_names(store_dir: Path) -> list[str]:
    with (store_dir / "sources.csv").open("r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


# ---------------------------------------------------------------- pass: materialize

def process_shard_s3_mat(
    s2_path: Path, out_root: Path, source: str, relpath: str
) -> dict[str, Any]:
    out_path, summary_path = s3_output_paths(out_root, source, relpath)
    if summary_path.is_file() and out_path.is_file():
        summary = json.loads(summary_path.read_text("utf-8"))
        if (
            summary.get("decisions_dropped") == s3_decisions_dropped_total(out_root)
            and summary.get("input_fingerprint") == fingerprint(s2_path)
        ):
            summary["skipped"] = True
            return summary

    started = time.monotonic()
    in_rows = out_rows = 0
    decisions: Counter[str] = Counter()
    writer = BatchWriter(out_path, S3_SCHEMA)
    lookups: dict[str, Any] = {}
    try:
        pf = pq.ParquetFile(s2_path)
        for batch in pf.iter_batches(batch_size=2048):
            rows = batch.to_pylist()
            by_domain: dict[str, list[int]] = {}
            for i, row in enumerate(rows):
                by_domain.setdefault(row["domain"], []).append(i)
            for domain, idxs in by_domain.items():
                lk = lookups.get(domain)
                if lk is None:
                    lk = lookups[domain] = load_drops_lookup(s3_lookup_dir(out_root, domain))
                ids = np.array([doc_id_to_bin(rows[i]["doc_id"]) for i in idxs], dtype="V16")
                codes = drops_for(ids, lk)
                for i, code in zip(idxs, codes):
                    decision = S3_DROP if int(code) == 1 else S3_KEEP
                    decisions[decision] += 1
                    rows[i]["s3_decision"] = decision
                    writer.add(rows[i])
                    out_rows += 1
            in_rows += len(rows)
        out_bytes = writer.close()
    except Exception as exc:
        writer.abort()
        summary = {
            "stage": "s3_mat",
            "source": source,
            "relpath": relpath,
            "in_rows": in_rows,
            "out_rows": out_rows,
            "output_bytes": 0,
            "decisions": dict(decisions),
            "elapsed_sec": round(time.monotonic() - started, 3),
            "errors": {f"shard_error:{type(exc).__name__}": 1},
            "skipped": False,
        }
        write_summary(summary_path, summary)
        return summary

    summary = {
        "stage": "s3_mat",
        "source": source,
        "relpath": relpath,
        "in_rows": in_rows,
        "out_rows": out_rows,
        "output_bytes": out_bytes,
        "decisions": dict(decisions),
        "decisions_dropped": s3_decisions_dropped_total(out_root),
        "input_fingerprint": fingerprint(s2_path),
        "elapsed_sec": round(time.monotonic() - started, 3),
        "mb_per_sec": round(out_bytes / max(time.monotonic() - started, 1e-9) / 1e6, 2),
        "errors": {},
        "skipped": False,
    }
    write_summary(summary_path, summary)
    return summary


# ---------------------------------------------------------------- orchestration

def _s3_domains(
    out_root: Path, profiles: dict[str, dict[str, Any]], sources: list[str] | None
) -> dict[str, list[str]]:
    domains: dict[str, list[str]] = {}
    for source, prof in profiles.items():
        if sources and source not in sources:
            continue
        if not (out_root / "s2" / source).is_dir():
            continue
        domains.setdefault(prof["domain"], []).append(source)
    return {d: sorted(s) for d, s in domains.items()}


def plan_s3_sig_tasks(out_root: Path, profiles: dict[str, dict[str, Any]], sources: list[str] | None) -> list[tuple]:
    tasks = []
    for source in profiles:
        if sources and source not in sources:
            continue
        src_dir = out_root / "s2" / source
        if not src_dir.is_dir():
            continue
        for path in sorted(src_dir.rglob("*.parquet")):
            rel = path.relative_to(src_dir).as_posix()
            tasks.append(("s3_sig", source, rel[: -len(".parquet")]))
    return sorted(tasks)


def plan_s3_band_tasks(
    domains: list[str], cfg: dict[str, Any]
) -> list[tuple]:
    s3 = cfg["s3"]
    rng = int(s3["reduce_range"])
    tasks = []
    for domain in domains:
        for band in range(int(s3["bands"])):
            for lo in range(0, int(s3["lsh_buckets"]), rng):
                tasks.append(("s3_band", domain, band, lo, lo + rng))
    return sorted(tasks)


def plan_s3_verify_tasks(domains: list[str], cfg: dict[str, Any]) -> list[tuple]:
    vb = int(cfg["s3"]["verify_buckets"])
    return [("s3_verify", d, k) for d in domains for k in range(vb)]


def plan_s3_mat_tasks(out_root: Path, profiles: dict[str, dict[str, Any]], sources: list[str] | None) -> list[tuple]:
    return plan_s3_sig_tasks(out_root, profiles, sources)


def run_s3(
    out_root: Path,
    profiles: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
    workers: int,
    sources: list[str] | None = None,
    limit: int = 0,
) -> Path:
    if not (out_root / "s2").is_dir():
        raise SystemExit("s3 依赖 s2 产物：请先运行 --stage s2（或 --stage dedup）")
    summaries: list[dict[str, Any]] = []
    domain_sources = _s3_domains(out_root, profiles, sources)
    domains = sorted(domain_sources)
    log.info("s3 domains: %s", domain_sources)

    sig_tasks = plan_s3_sig_tasks(out_root, profiles, sources)
    if limit:
        sig_tasks = sig_tasks[:limit]
    log.info("s3 pass1 (sig): %d shard tasks", len(sig_tasks))
    summaries += run_tasks(
        sig_tasks, {"out_root": out_root, "cfg": cfg}, _worker_s3_sig, workers, "s3_sig"
    )

    band_tasks = plan_s3_band_tasks(domains, cfg)
    log.info("s3 pass2 (band reduce): %d tasks", len(band_tasks))
    summaries += run_tasks(
        band_tasks,
        {"out_root": out_root, "cfg": cfg, "domain_sources": domain_sources},
        _worker_s3_band,
        min(workers, 64),
        "s3_band",
    )

    summaries += build_candidate_sets(out_root, domains, cfg)

    collect_tasks = plan_s3_mat_tasks(out_root, profiles, sources)
    if limit:
        collect_tasks = collect_tasks[:limit]
    log.info("s3 collect candidates: %d shard tasks", len(collect_tasks))
    summaries += run_tasks(
        collect_tasks, {"out_root": out_root, "cfg": cfg}, _worker_s3_collect, workers, "s3_collect"
    )

    summaries += build_docstores(out_root, domain_sources, profiles, cfg)

    for domain in domains:
        t0 = time.monotonic()
        payload = build_csig(
            out_root / "s3_sig" / domain,
            out_root / "s3_csig" / domain,
            min(workers, 96),
        )
        summaries.append(
            {
                "stage": "s3_csig",
                "domain": domain,
                "candidate_docs": int(payload["docs"]),
                "elapsed_sec": round(time.monotonic() - t0, 3),
                "errors": {},
                "skipped": False,
            }
        )

    # 超阈值域（web 8.4B 对）：verify 全跳过（逐对验证 IO 不可行），走 UF 直聚类
    uf_threshold = int(cfg["s3"].get("uf_direct_pair_threshold", 1_000_000_000))
    verify_domains = []
    for domain in domains:
        pair_rows = 0
        cs = s3_cand_summary_path(out_root, domain)
        if cs.is_file():
            pair_rows = int(json.loads(cs.read_text("utf-8")).get("pair_rows") or 0)
        if pair_rows < uf_threshold:
            verify_domains.append(domain)
    verify_tasks = plan_s3_verify_tasks(verify_domains, cfg)
    log.info("s3 verify: %d tasks (domains: %s; UF-direct: %s)", len(verify_tasks), verify_domains, sorted(set(domains) - set(verify_domains)))
    # 2026-09-22: 64→32 worker——verify 单任务上限 24GB × 并发数必须 << 机器内存，
    # 64×32GB=2TB 会触发系统 OOM（exit 137 实录）；32×24GB=768GB 留足余量
    summaries += run_tasks(
        verify_tasks,
        {"out_root": out_root, "cfg": cfg, "domain_sources": domain_sources},
        _worker_s3_verify,
        # 2026-09-22 v2: 32→16——cgroup 硬限 1TB（含 page cache），32×24GB=768GB 实测仍 OOM；
        # 16×16GB=256GB 余量充足，代价是 verify 时长翻倍（可接受， boring wins）
        min(workers, 16),
        "s3_verify",
    )

    # 超阈值域（web 8.4B 对）：跳过逐对验证，候选对直并查集聚类
    # 决定 summary 已存在（含 E-17 跳过策略）的域一律不重算
    uf_domains: set[str] = set()
    for domain in domains:
        dec_summary = out_root / "s3_decisions" / f"{domain}.summary.json"
        if dec_summary.is_file():
            existing = json.loads(dec_summary.read_text("utf-8"))
            existing["skipped"] = True
            summaries.append(existing)
            uf_domains.add(domain)
            continue
        uf_summary = _uf_direct_pairs(out_root, domain, cfg)
        uf_summary["domain"] = domain
        summaries.append(uf_summary)
        if not uf_summary.get("skipped"):
            uf_domains.add(domain)
    remaining = [d for d in domains if d not in uf_domains]
    if remaining:
        summaries += build_s3_decisions(out_root, remaining, cfg, domain_sources)

    mat_tasks = plan_s3_mat_tasks(out_root, profiles, sources)
    if limit:
        mat_tasks = mat_tasks[:limit]
    log.info("s3 materialize: %d shard tasks", len(mat_tasks))
    summaries += run_tasks(
        mat_tasks, {"out_root": out_root}, _worker_s3_mat, workers, "s3_mat"
    )
    write_stage_summary(out_root, "s3", summaries)
    return summaries


def _worker_s3_sig(args: tuple[tuple, dict[str, Any]]) -> dict[str, Any]:
    task, ctx = args
    _, source, relpath = task
    try:
        return process_shard_s3_sig(
            ctx["out_root"] / "s2" / source / (relpath + ".parquet"),
            ctx["out_root"],
            source,
            relpath,
            ctx["cfg"],
        )
    except Exception as exc:
        return fatal_summary("s3_sig", f"{source}/{relpath}", exc, time.monotonic())


def _worker_s3_band(args: tuple[tuple, dict[str, Any]]) -> dict[str, Any]:
    task, ctx = args
    _, domain, band, lo, hi = task
    try:
        return process_s3_band_reduce(
            ctx["out_root"], domain, band, lo, hi, ctx["cfg"], ctx["domain_sources"][domain]
        )
    except Exception as exc:
        return fatal_summary("s3_band", f"{domain}/b{band}/{lo}-{hi}", exc, time.monotonic())


def _worker_s3_collect(args: tuple[tuple, dict[str, Any]]) -> dict[str, Any]:
    task, ctx = args
    _, source, relpath = task
    try:
        return process_shard_s3_collect(
            ctx["out_root"] / "s2" / source / (relpath + ".parquet"),
            ctx["out_root"],
            source,
            relpath,
            ctx["cfg"],
        )
    except Exception as exc:
        return fatal_summary("s3_collect", f"{source}/{relpath}", exc, time.monotonic())


def _worker_s3_verify(args: tuple[tuple, dict[str, Any]]) -> dict[str, Any]:
    task, ctx = args
    _, domain, bucket_byte = task
    try:
        return process_s3_verify(
            ctx["out_root"], domain, bucket_byte, ctx["cfg"], ctx["domain_sources"][domain]
        )
    except Exception as exc:
        return fatal_summary("s3_verify", f"{domain}/{bucket_byte:02x}", exc, time.monotonic())


def _worker_s3_mat(args: tuple[tuple, dict[str, Any]]) -> dict[str, Any]:
    task, ctx = args
    _, source, relpath = task
    try:
        return process_shard_s3_mat(
            ctx["out_root"] / "s2" / source / (relpath + ".parquet"),
            ctx["out_root"],
            source,
            relpath,
        )
    except Exception as exc:
        return fatal_summary("s3_mat", f"{source}/{relpath}", exc, time.monotonic())
