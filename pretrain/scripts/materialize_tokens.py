#!/usr/bin/env python3
"""S6 契约物化：mix_order doc_id 顺序表 + s3 正文 → pretrain token 流。

流程（宪法 4：顺序完全确定）：
  pass 1 建 doc_id → 位置索引（扫描 s3 parquets；doc_id = blake2b(source\\0relpath\\0idx)
         不可逆，必须正排索引；索引 = 按 V16 doc_id 排序的 ids/文件号/行号）
  pass 2 按顺序表 position 逐批取 doc_id → 索引定位 → 按 parquet row group 批量读正文
         → tokenizer.encode_batch(add_special_tokens=False）并在每篇文档末尾追加 EOD
         → 按 position 顺序拼接 token 流（uint16/uint32 按词表大小，E-42/E-74）→ <name>.tokens.bin + meta.json
  val：val_sets/<domain>.parquet 同机制产出 <domain>.tokens.bin（held-out，永不进训练）

meta.json 契约（pretrain/src/yuformer_pretrain/data.py 消费）：
  {"total_tokens", "bytes", "sha256", "edge_sha256", "tokenizer_sha256",
   "document_boundary", "source"}

用法：
  PYTHONPATH=src python scripts/materialize_tokens.py \
    --order $YUFORMER_S6_ROOT/mix_order/<mix>_stage2.parquet \
    --s3-root $YUFORMER_S6_ROOT/s3 --tokenizer $YUFORMER_TOKENIZER \
    --out-dir $YUFORMER_TOKEN_STREAMS --name <mix>_stage2 --workers 8
"""
from __future__ import annotations

import argparse
import os
import hashlib
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from yuformer_pretrain.data import _edge_sha256, load_data_snapshot  # noqa: E402
from yuformer_pretrain.config import ConfigError  # noqa: E402
from yuformer_pretrain.lease import RunLease  # noqa: E402
from yuformer_pretrain.tracking import file_sha256  # noqa: E402
from source_document_binding import SOURCE_DOCUMENT_BINDING_VERSION  # noqa: E402

BATCH = 4096
DEFAULT_EOD_TOKEN = "<|endoftext|>"
PART_REFERENCE_TOKEN_LIMIT = 64_000_000
PART_DOCUMENT_LIMIT = 262_144
ENCODE_BATCH_CHAR_LIMIT = 4_000_000
ENCODE_BATCH_DOCUMENT_LIMIT = 2048


@contextmanager
def single_thread_worker_environment():
    settings = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                "RAYON_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
    previous = {key: os.environ.get(key) for key in settings}
    os.environ.update(settings)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def configure_materialize_worker():
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


def token_bounded_chunks(reference_tokens, positions, boundary, *,
                         max_tokens=PART_REFERENCE_TOKEN_LIMIT,
                         max_docs=PART_DOCUMENT_LIMIT, max_groups=2000):
    if min(max_tokens, max_docs, max_groups) < 1:
        raise ValueError("materialization chunk limits must be positive")
    if len(positions) == 0:
        return []
    if (len(boundary) < 2 or int(boundary[0]) != 0
            or int(boundary[-1]) != len(positions) or bool(np.any(np.diff(boundary) <= 0))):
        raise ValueError("row-group boundaries do not cover selected documents exactly")
    ordered_reference = np.asarray(reference_tokens, dtype=np.int64)[positions]
    if bool(np.any(ordered_reference < 0)):
        raise ValueError("reference token counts cannot be negative")
    counts = np.add.reduceat(ordered_reference, boundary[:-1], dtype=np.int64)
    docs = np.diff(boundary)
    result, start, tokens, rows = [], 0, 0, 0
    for group, (token_count, doc_count) in enumerate(zip(counts, docs, strict=True)):
        token_count, doc_count = int(token_count), int(doc_count)
        if group > start and (tokens + token_count > max_tokens
                              or rows + doc_count > max_docs or group - start >= max_groups):
            result.append((len(result), start, group))
            start, tokens, rows = group, 0, 0
        tokens += token_count
        rows += doc_count
    result.append((len(result), start, len(counts)))
    return result


def hex_to_v16(ids: list[str]) -> np.ndarray:
    return np.frombuffer(b"".join(bytes.fromhex(d) for d in ids), dtype="V16").reshape(-1)


def v16_to_hex(ids: np.ndarray) -> list[str]:
    return [bytes(x).hex() for x in ids]


def lookup_sorted_doc_ids(
    index_ids: np.ndarray,
    query_ids: np.ndarray,
    *,
    index_chunk_rows: int = 16_000_000,
    direct_query_threshold: int = 4_096,
) -> tuple[np.ndarray, np.ndarray]:
    """Locate query IDs in a sorted V16 index with storage-local reads.

    S6 order is a hash shuffle, so querying its IDs directly makes binary
    search jump randomly across the 97-GiB production index for every one of
    ~95M documents.  Sorting the queries first and copying the index in large
    sequential chunks avoids millions of sparse 4-KiB NFS faults.  Searches
    then happen against a resident chunk; ``positions[query_order] =
    sorted_positions`` restores the caller's logical order exactly.
    """
    if len(index_ids) == 0:
        return np.zeros(len(query_ids), dtype=np.int64), np.zeros(len(query_ids), dtype=bool)
    if index_chunk_rows < 1:
        raise ValueError("index_chunk_rows must be >= 1")
    if direct_query_threshold < 0:
        raise ValueError("direct_query_threshold must be >= 0")
    # Tiny audits should pay a few sparse page faults, not stream the complete
    # 97-GiB production index.  Formal train/val rosters exceed this threshold
    # and retain the sequential storage-local path below.
    if len(query_ids) <= direct_query_threshold:
        positions = np.searchsorted(index_ids, query_ids).astype(np.int64, copy=False)
        valid = positions < len(index_ids)
        hits = valid & (
            index_ids[np.minimum(positions, len(index_ids) - 1)] == query_ids
        )
        return positions, hits
    query_order = np.argsort(query_ids, kind="stable")
    sorted_queries = query_ids[query_order]
    sorted_positions = np.full(len(sorted_queries), len(index_ids), dtype=np.int64)
    sorted_hits = np.zeros(len(sorted_queries), dtype=bool)
    query_cursor = 0
    for index_start in range(0, len(index_ids), index_chunk_rows):
        index_stop = min(index_start + index_chunk_rows, len(index_ids))
        # Force one sequential read from the mmap.  Keeping a view here would
        # preserve searchsorted's sparse page-fault access pattern on NFS.
        chunk = np.array(index_ids[index_start:index_stop], copy=True)
        query_stop = int(np.searchsorted(sorted_queries, chunk[-1], side="right"))
        if query_stop <= query_cursor:
            continue
        local_queries = sorted_queries[query_cursor:query_stop]
        local_positions = np.searchsorted(chunk, local_queries)
        valid = local_positions < len(chunk)
        local_hits = valid & (
            chunk[np.minimum(local_positions, len(chunk) - 1)] == local_queries
        )
        sorted_positions[query_cursor:query_stop] = index_start + local_positions
        sorted_hits[query_cursor:query_stop] = local_hits
        query_cursor = query_stop
        if query_cursor == len(sorted_queries):
            break
    positions = np.empty_like(sorted_positions)
    hits = np.empty_like(sorted_hits)
    positions[query_order] = sorted_positions
    hits[query_order] = sorted_hits
    return positions, hits


def source_file_identity(path: Path) -> dict[str, int]:
    return _stat_identity(path.stat())


def _stat_identity(stat) -> dict[str, int]:
    return {"bytes": stat.st_size, "device": stat.st_dev,
            "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns}


def source_roster_fingerprint(s3_root: Path, files: list[Path]) -> tuple[str, list[dict[str, int]]]:
    identities = [source_file_identity(path) for path in files]
    rows = [{"path": path.relative_to(s3_root).as_posix(), **identity}
            for path, identity in zip(files, identities, strict=True)]
    fingerprint = hashlib.sha256(
        ("s3-doc-index-v2\0" + json.dumps(rows, sort_keys=True, separators=(",", ":"))).encode()
    ).hexdigest()
    return fingerprint, identities


def _load_doc_index_cache(cache_dir: Path, fingerprint: str):
    try:
        manifest = json.loads((cache_dir / f"index-v2-{fingerprint}.json").read_text())
        if (manifest.get("schema_version") != 2 or manifest.get("fingerprint") != fingerprint
                or manifest.get("unique_doc_ids") is not True):
            return None
        arrays = []
        for name, dtype in (("ids", "V16"), ("fidx", "uint32"), ("ridx", "uint32")):
            record = manifest["arrays"][name]
            path = cache_dir / f"{name}-{fingerprint}.npy"
            if source_file_identity(path) != record["identity"]:
                return None
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if (array.ndim != 1 or array.dtype != np.dtype(dtype)
                    or len(array) != int(manifest["rows"])
                    or source_file_identity(path) != record["identity"]):
                return None
            arrays.append(array)
        return tuple(arrays)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def build_doc_index(s3_root: Path, workers: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Path]]:
    """Build a source-bound immutable index generation with a final commit manifest."""
    files = sorted(s3_root.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"s3 下无 parquet: {s3_root}")
    cache_dir = s3_root / ".doc_index_cache"
    fingerprint, _ = source_roster_fingerprint(s3_root, files)
    cached = _load_doc_index_cache(cache_dir, fingerprint)
    if cached is not None:
        print(f"  index cache hit ({fingerprint[:16]}), memmap load", flush=True)
        return *cached, files
    deadline = time.monotonic() + 6 * 3600
    while True:
        lease = RunLease(cache_dir, "source-document-index", stale_seconds=600)
        try:
            lease.acquire()
            break
        except ConfigError:
            if time.monotonic() >= deadline:
                raise
            print("  waiting for the source document-index writer", flush=True)
            time.sleep(15)
    try:
        files = sorted(s3_root.rglob("*.parquet"))
        fingerprint, _ = source_roster_fingerprint(s3_root, files)
        cached = _load_doc_index_cache(cache_dir, fingerprint)
        if cached is not None:
            return *cached, files
        return _build_doc_index_generation(s3_root, files, cache_dir, fingerprint, lease, workers)
    finally:
        lease.release()


def _read_doc_index_file(path: Path) -> np.ndarray | None:
    parquet = pq.ParquetFile(path)
    rows = parquet.metadata.num_rows
    if rows == 0:
        return None
    if rows > (1 << 32):
        raise ValueError("source row index exceeds uint32 capacity")
    return hex_to_v16(parquet.read(columns=["doc_id"]).column("doc_id").to_pylist())


def _iter_doc_index_files(files: list[Path], workers: int):
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor

    readers = max(1, min(8, int(workers or 1), len(files)))
    with ThreadPoolExecutor(max_workers=readers) as pool:
        pending = deque()
        next_file = 0
        while next_file < min(len(files), 2 * readers):
            pending.append((next_file, pool.submit(_read_doc_index_file, files[next_file])))
            next_file += 1
        while pending:
            index, future = pending.popleft()
            value = future.result()
            if next_file < len(files):
                pending.append((next_file, pool.submit(_read_doc_index_file, files[next_file])))
                next_file += 1
            yield index, value


def _build_doc_index_generation(s3_root: Path, files: list[Path], cache_dir: Path,
                                fingerprint: str, lease: RunLease, workers: int):
    ids_parts, fidx_parts, ridx_parts = [], [], []
    t0 = time.monotonic()
    for fi, ids in _iter_doc_index_files(files, workers):
        if fi % 200 == 0:
            print(f"  index {fi}/{len(files)} shards, {time.monotonic()-t0:.0f}s", flush=True)
        if ids is None:
            continue
        n = len(ids)
        if fi >= (1 << 32):
            raise ValueError("source file/row index exceeds uint32 capacity")
        ids_parts.append(ids)
        fidx_parts.append(np.full(n, fi, dtype=np.uint32))
        ridx_parts.append(np.arange(n, dtype=np.uint32))
    if not ids_parts:
        raise ValueError("S3 source contains no document IDs")
    ids = np.concatenate(ids_parts)
    fidx = np.concatenate(fidx_parts)
    ridx = np.concatenate(ridx_parts)
    order = np.argsort(ids, kind="stable")
    ids, fidx, ridx = ids[order], fidx[order], ridx[order]
    for start in range(1, len(ids), 8_000_000):
        stop = min(len(ids), start + 8_000_000)
        if bool(np.any(ids[start:stop] == ids[start - 1:stop - 1])):
            raise ValueError("duplicate doc_id in S3 index; source mapping is ambiguous")
    manifest = {"schema_version": 2, "fingerprint": fingerprint,
                "rows": len(ids), "unique_doc_ids": True, "arrays": {}}
    for name, array in (("ids", ids), ("fidx", fidx), ("ridx", ridx)):
        path = cache_dir / f"{name}-{fingerprint}.npy"
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}.npy")
        np.save(temporary, array)
        lease.assert_owner()
        os.replace(temporary, path)
        manifest["arrays"][name] = {"identity": source_file_identity(path)}
    current_files = sorted(s3_root.rglob("*.parquet"))
    if current_files != files or source_roster_fingerprint(s3_root, files)[0] != fingerprint:
        raise ValueError("S3 source changed while building document index")
    temporary = cache_dir / f".index-v2-{fingerprint}.tmp-{os.getpid()}.json"
    temporary.write_text(json.dumps(manifest, sort_keys=True) + "\n")
    lease.assert_owner()
    os.replace(temporary, cache_dir / f"index-v2-{fingerprint}.json")
    print(f"  index cached ({fingerprint[:16]})", flush=True)
    return ids, fidx, ridx, files


def load_bound_tokenizer(path: Path, expected_sha256: str | None = None):
    from tokenizers import Tokenizer

    serialized = Path(path).read_bytes()
    actual = hashlib.sha256(serialized).hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError("tokenizer bytes differ from the bound tokenizer identity")
    return Tokenizer.from_str(serialized.decode("utf-8")), actual


class ShardTextFetcher:
    """按 (file, row_idx) 批量取正文：row group 粒度读，page cache 摊薄。"""

    def __init__(self, files: list[Path]) -> None:
        self.files = files
        self._rg_bounds: dict[int, np.ndarray] = {}

    def _row_groups(self, fi: int) -> np.ndarray:
        bounds = self._rg_bounds.get(fi)
        if bounds is None:
            md = pq.ParquetFile(self.files[fi]).metadata
            rows = [md.row_group(i).num_rows for i in range(md.num_row_groups)]
            bounds = np.concatenate([[0], np.cumsum(rows)])
            self._rg_bounds[fi] = bounds
        return bounds

    def fetch(self, fi: int, row_indices: np.ndarray) -> list[str]:
        bounds = self._row_groups(fi)
        texts: dict[int, str] = {}
        row_groups = np.searchsorted(bounds, row_indices, side="right") - 1
        parquet = pq.ParquetFile(self.files[fi])
        for rg in np.unique(row_groups):
            lo, hi = int(bounds[rg]), int(bounds[rg + 1])
            selected_rows = row_indices[row_groups == rg]
            local_rows = np.asarray(selected_rows - lo, dtype=np.int64)
            column = parquet.read_row_group(int(rg), columns=["text"]).column("text")
            # E-121: row groups can contain millions of documents.  Convert
            # only selected rows to Python strings; converting the full text
            # column before subsetting multiplies RAM by row-group size and by
            # concurrent reader count.
            selected_texts = column.take(pa.array(local_rows, type=pa.int64())).to_pylist()
            for row, text in zip(selected_rows, selected_texts):
                texts[int(row)] = text or ""
        return [texts[int(r)] for r in row_indices]


def _pass2a_chunk(task: tuple) -> tuple[int, int]:
    """E-54 worker：处理一个 chunk 组区间，写 part{i:06d}.npz（独立文件 → 并行安全）。"""
    idx, g0, g1, scan_dir, tokenizer_path, tmp_dir, tok_dtype, eod_id = task
    scan_dir = Path(scan_dir)
    # mmap-able arrays avoid loading the multi-GiB scan archive once per worker.
    sfi = np.load(scan_dir / "sfi.npy", mmap_mode="r")
    srg = np.load(scan_dir / "srg.npy", mmap_mode="r")
    srow = np.load(scan_dir / "srow.npy", mmap_mode="r")
    spos = np.load(scan_dir / "spos.npy", mmap_mode="r")
    expected_ids = np.load(scan_dir / "expected_ids.npy", mmap_mode="r")
    boundary = np.load(scan_dir / "boundary.npy", mmap_mode="r")
    files = [Path(x) for x in json.loads((scan_dir / "files.json").read_text("utf-8"))]
    file_identities = json.loads((scan_dir / "file_identities.json").read_text())
    tokenizer_sha = (scan_dir / "tokenizer_sha256.txt").read_text().strip()
    tok, _ = load_bound_tokenizer(Path(tokenizer_path), tokenizer_sha)
    import pyarrow.parquet as _pq
    batch_texts, batch_positions, token_rows = [], [], []
    batch_chars = 0

    def encode_pending():
        nonlocal batch_chars
        if not batch_texts:
            return
        encodings = tok.encode_batch(batch_texts, add_special_tokens=False)
        for position, encoding in zip(batch_positions, encodings, strict=True):
            ids = np.asarray(encoding.ids, dtype=tok_dtype)
            tokens = np.empty(len(ids) + 1, dtype=tok_dtype)
            tokens[:-1], tokens[-1] = ids, eod_id
            token_rows.append((position, tokens))
        batch_texts.clear()
        batch_positions.clear()
        batch_chars = 0
    # Each chunk is already sorted by (file, row-group).  Cache row-group
    # boundaries for the files touched by this worker so a late row-group does
    # not repeatedly sum all preceding groups (quadratic in groups per file).
    row_group_bounds: dict[int, np.ndarray] = {}
    touched_files: set[int] = set()
    for k in range(g0, g1):
        a, b = int(boundary[k]), int(boundary[k + 1])
        fi = int(sfi[a])
        rg = int(srg[a])
        if fi not in touched_files:
            if source_file_identity(files[fi]) != file_identities[fi]:
                raise ValueError(f"S3 source identity changed before encoding: {files[fi]}")
            touched_files.add(fi)
        with files[fi].open("rb") as source_handle:
            if _stat_identity(os.fstat(source_handle.fileno())) != file_identities[fi]:
                raise ValueError(f"S3 opened source identity differs from binding: {files[fi]}")
            pf = _pq.ParquetFile(source_handle)
            source_rows = pf.read_row_group(rg, columns=["doc_id", "text"])
            metadata = pf.metadata
            if _stat_identity(os.fstat(source_handle.fileno())) != file_identities[fi]:
                raise ValueError(f"S3 opened source changed during reading: {files[fi]}")
        col = source_rows.column("text")
        bounds = row_group_bounds.get(fi)
        if bounds is None:
            md = metadata
            bounds = np.concatenate([
                np.asarray([0], dtype=np.int64),
                np.cumsum(
                    [md.row_group(j).num_rows for j in range(md.num_row_groups)],
                    dtype=np.int64,
                ),
            ])
            row_group_bounds[fi] = bounds
        rg_start = int(bounds[rg])
        local_rows = np.asarray(srow[a:b], dtype=np.int64) - rg_start
        if bool(np.any(local_rows < 0)) or bool(np.any(local_rows >= len(col))):
            raise ValueError(f"S3 row index is outside the expected row group: {files[fi]}")
        actual_ids = hex_to_v16(source_rows.column("doc_id").take(pa.array(local_rows)).to_pylist())
        if not bool(np.all(actual_ids == expected_ids[spos[a:b]])):
            raise ValueError(f"S3 doc_id differs from requested order; refusing wrong text: {files[fi]}")
        for r, pp in zip(srow[a:b], spos[a:b]):
            text = col[int(r) - rg_start].as_py() or ""
            if batch_texts and (len(batch_texts) >= ENCODE_BATCH_DOCUMENT_LIMIT
                                or batch_chars + len(text) > ENCODE_BATCH_CHAR_LIMIT):
                encode_pending()
            batch_texts.append(text)
            batch_positions.append(int(pp))
            batch_chars += len(text)
    encode_pending()
    for fi in touched_files:
        if source_file_identity(files[fi]) != file_identities[fi]:
            raise ValueError(f"S3 source identity changed during encoding: {files[fi]}")
    token_rows.sort(key=lambda row: row[0])
    # E-74：按 tok_dtype 落盘（E-42 只改了声明 dtype，此处仍写死 <u2——128k 流 id≥65536
    # 被 numpy 静默回绕，2026-10-13 e0q2_yf-128k-bpe 整个流报废，据此修复）。
    doc_tokens = [tokens for _, tokens in token_rows]
    toks = (np.concatenate(doc_tokens)
            if token_rows else np.empty(0, dtype=tok_dtype))
    pos_arr = np.array([pp for pp, _ in token_rows], dtype=np.int64)
    offs = np.concatenate([[0], np.cumsum([len(tokens) for tokens in doc_tokens])]).astype(np.int64)
    if toks.size and tok_dtype == "<u2" and int(toks.max()) >= 65536:
        raise SystemExit(f"part{idx}: token id {int(toks.max())} 溢出 uint16（E-42 断言）")
    part_path = Path(tmp_dir) / f"part{idx:06d}.npz"
    part_tmp = part_path.with_name(f".{part_path.name}.tmp-{os.getpid()}.npz")
    np.savez(part_tmp, positions=pos_arr, offsets=offs, tokens=toks)
    os.replace(part_tmp, part_path)
    return idx, int(len(toks))


def merge_parts_by_position(
    part_files: list[Path],
    staging_bin: Path,
    *,
    expected_docs: int,
    tok_dtype: str,
    chunk_docs: int = 1_000_000,
    external_threshold_bytes: int = 32 << 30,
    doc_lengths_path: Path | None = None,
) -> int:
    """Merge storage-local token parts with a bounded Arrow gather working set.

    ``pyarrow.compute.take`` on the complete ChunkedArray may coalesce every
    source chunk before selecting a small logical window.  On the 100B stream
    that copied roughly 200 GB for each window and retained hundreds of GB in
    Arrow's allocator.  Select the required rows inside each source part first;
    the global reorder then sees only the current output window.
    """
    import pyarrow.compute as pc

    if chunk_docs < 1:
        raise ValueError("chunk_docs must be >= 1")
    if sum(path.stat().st_size for path in part_files) >= external_threshold_bytes:
        return merge_parts_external(part_files, staging_bin, expected_docs=expected_docs,
                                    tok_dtype=tok_dtype, doc_lengths_path=doc_lengths_path)
    part_dtype = np.uint16 if len(part_files) < np.iinfo(np.uint16).max else np.uint32
    missing_part = np.iinfo(part_dtype).max
    source_part = np.full(expected_docs, missing_part, dtype=part_dtype)
    source_local = np.empty(expected_docs, dtype=np.uint32)
    seen = np.zeros(expected_docs, dtype=bool)
    chunks: list[pa.LargeListArray] = []
    source_docs = 0
    item_type = pa.uint16() if np.dtype(tok_dtype).itemsize == 2 else pa.uint32()
    for part_index, part_path in enumerate(part_files):
        with np.load(part_path) as archive:
            positions = archive["positions"]
            offsets = archive["offsets"]
            tokens = archive["tokens"]
        if len(offsets) != len(positions) + 1:
            raise ValueError(f"part offsets/positions 长度不闭合: {part_path}")
        if tokens.dtype != np.dtype(tok_dtype):
            raise ValueError(f"part token dtype 不符: {part_path}: {tokens.dtype}")
        if len(positions) and (int(positions[0]) < 0 or int(positions[-1]) >= expected_docs):
            raise ValueError(f"part position 越界: {part_path}")
        if len(positions) > 1 and not bool(np.all(positions[1:] > positions[:-1])):
            raise ValueError(f"part position 非严格递增: {part_path}")
        if bool(np.any(seen[positions])):
            raise ValueError(f"part position 重复: {part_path}")
        seen[positions] = True
        source_part[positions] = part_index
        source_local[positions] = np.arange(len(positions), dtype=np.uint32)
        source_docs += len(positions)
        chunks.append(
            pa.LargeListArray.from_arrays(
                pa.array(offsets, type=pa.int64()),
                pa.array(tokens, type=item_type),
            )
        )
    if source_docs != expected_docs or not bool(np.all(seen)):
        raise ValueError(
            f"part roster 不闭合: docs={source_docs}, expected={expected_docs}, "
            f"missing={int((~seen).sum())}"
        )
    del seen
    written_tokens = 0
    lengths_output = doc_lengths_path.open("wb") if doc_lengths_path is not None else None
    try:
        with staging_bin.open("wb") as output:
            for start in range(0, expected_docs, chunk_docs):
                stop = min(start + chunk_docs, expected_docs)
                window_parts = source_part[start:stop]
                grouped_order = np.argsort(window_parts, kind="stable")
                grouped_parts = window_parts[grouped_order]
                grouped_local = source_local[start:stop][grouped_order]
                cuts = np.concatenate((
                    np.asarray([0], dtype=np.int64),
                    np.flatnonzero(np.diff(grouped_parts) != 0).astype(np.int64) + 1,
                    np.asarray([len(grouped_parts)], dtype=np.int64),
                ))
                selected: list[pa.LargeListArray] = []
                for cut in range(len(cuts) - 1):
                    left, right = int(cuts[cut]), int(cuts[cut + 1])
                    part_index = int(grouped_parts[left])
                    selected.append(pc.take(
                        chunks[part_index],
                        pa.array(grouped_local[left:right], type=pa.uint32()),
                    ))
                inverse_order = np.empty(len(grouped_order), dtype=np.uint32)
                inverse_order[grouped_order] = np.arange(len(grouped_order), dtype=np.uint32)
                gathered = pc.take(
                    pa.chunked_array(selected),
                    pa.array(inverse_order, type=pa.uint32()),
                )
                if lengths_output is not None:
                    lengths = pc.list_value_length(gathered).combine_chunks().to_numpy(
                        zero_copy_only=False
                    ).astype("<u4", copy=False)
                    lengths.tofile(lengths_output)
                flat = pc.list_flatten(gathered).combine_chunks()
                values = flat.to_numpy(zero_copy_only=True)
                values.tofile(output)
                written_tokens += len(values)
                del selected, gathered, flat, values
                if stop % 1_000_000 == 0 or stop == expected_docs:
                    print(f"  merged {stop}/{expected_docs} docs", flush=True)
    finally:
        if lengths_output is not None:
            lengths_output.flush()
            os.fsync(lengths_output.fileno())
            lengths_output.close()
    return written_tokens


def merge_parts_external(part_files: list[Path], staging_bin: Path, *,
                         expected_docs: int, tok_dtype: str,
                         doc_lengths_path: Path | None = None) -> int:
    """Bounded-memory k-way merge for 50B+ token streams."""
    import shutil
    import subprocess
    import zipfile

    dtype = np.dtype(tok_dtype)
    if dtype.str not in ("<u2", "<u4"):
        raise ValueError(f"unsupported token dtype: {dtype}")
    raw_dir = staging_bin.parent / f".{staging_bin.name}.merge-parts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for index, part_path in enumerate(part_files):
        with zipfile.ZipFile(part_path) as archive_file:
            source_zip = sorted((row.filename, row.CRC, row.file_size, row.compress_size)
                                for row in archive_file.infolist())
        prefix = raw_dir / f"part{index:06d}"
        positions_path = prefix.with_suffix(".positions.bin")
        offsets_path = prefix.with_suffix(".offsets.bin")
        tokens_path = prefix.with_suffix(".tokens.bin")
        marker_path = prefix.with_suffix(".json")
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            reusable = (
                marker.get("source_size") == part_path.stat().st_size
                and marker.get("source_zip") == [list(row) for row in source_zip]
                and marker.get("dtype") == dtype.str
                and positions_path.stat().st_size == int(marker["docs"]) * 8
                and offsets_path.stat().st_size == (int(marker["docs"]) + 1) * 8
                and tokens_path.stat().st_size == int(marker["tokens"]) * dtype.itemsize
            )
        except (OSError, ValueError, KeyError):
            reusable = False
        if not reusable:
            with np.load(part_path) as archive:
                positions = np.asarray(archive["positions"], dtype="<u8")
                offsets = np.asarray(archive["offsets"], dtype="<u8")
                tokens = archive["tokens"]
            if tokens.dtype != dtype or len(offsets) != len(positions) + 1:
                raise ValueError(f"invalid external-merge part: {part_path}")
            if len(positions) > 1 and not bool(np.all(positions[1:] > positions[:-1])):
                raise ValueError(f"part positions are not strictly increasing: {part_path}")
            if len(offsets) and (int(offsets[0]) != 0 or int(offsets[-1]) != len(tokens)
                                 or bool(np.any(offsets[1:] < offsets[:-1]))):
                raise ValueError(f"part offsets do not close: {part_path}")
            for target, values in ((positions_path, positions), (offsets_path, offsets),
                                   (tokens_path, tokens)):
                temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
                values.tofile(temporary)
                os.replace(temporary, target)
            marker = {"source": str(part_path), "source_size": part_path.stat().st_size,
                      "source_zip": source_zip, "docs": len(positions),
                      "tokens": len(tokens), "dtype": dtype.str}
            marker_tmp = marker_path.with_name(f".{marker_path.name}.tmp-{os.getpid()}")
            marker_tmp.write_text(json.dumps(marker, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(marker_tmp, marker_path)
        manifest_rows.append(
            f"{positions_path}\t{offsets_path}\t{tokens_path}\t{marker['docs']}\t{marker['tokens']}"
        )
    manifest = raw_dir / "manifest.tsv"
    manifest_tmp = manifest.with_name(f".{manifest.name}.tmp-{os.getpid()}")
    manifest_tmp.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    os.replace(manifest_tmp, manifest)
    source = Path(__file__).with_name("merge_token_parts.c")
    binary = raw_dir / "merge_token_parts"
    compiled = subprocess.run(["cc", "-O3", "-std=c11", str(source), "-o", str(binary)],
                              capture_output=True, text=True)
    if compiled.returncode:
        raise RuntimeError((compiled.stderr or compiled.stdout)[-2000:])
    command = [str(binary), str(manifest), str(staging_bin),
               str(expected_docs), str(dtype.itemsize)]
    if doc_lengths_path is not None:
        command.append(str(doc_lengths_path))
    merged = subprocess.run(command,
                            capture_output=True, text=True)
    if merged.returncode:
        raise RuntimeError((merged.stderr or merged.stdout)[-2000:])
    written_tokens = int(merged.stdout.strip())
    if staging_bin.stat().st_size != written_tokens * dtype.itemsize:
        raise ValueError("external merge byte/token count mismatch")
    if doc_lengths_path is not None:
        if doc_lengths_path.stat().st_size != expected_docs * 4:
            raise ValueError("external merge document-length count mismatch")
        lengths = np.memmap(doc_lengths_path, dtype="<u4", mode="r")
        if int(lengths.sum(dtype=np.uint64)) != written_tokens:
            raise ValueError("external merge document lengths do not sum to token count")
    shutil.rmtree(raw_dir, ignore_errors=True)
    return written_tokens


def materialize_stream(
    order_path: Path,
    s3_root: Path,
    tokenizer_path: Path,
    out_dir: Path,
    name: str,
    max_tokens: int = 0,
    eod_token: str = DEFAULT_EOD_TOKEN,
    data_snapshot: dict | None = None,
    workers: int | None = None,
    publish_guard: Callable[[], None] | None = None,
    tokenizer_release: dict | None = None,
    write_doc_lengths: bool = False,
) -> Path:
    tok, tok_sha = load_bound_tokenizer(tokenizer_path)
    if tokenizer_release is not None:
        if tokenizer_release.get("status") != "PASS":
            raise SystemExit("tokenizer release gate is not PASS")
        if tokenizer_release.get("tokenizer_sha256") != tok_sha:
            raise SystemExit("tokenizer hash differs from tokenizer release gate")
    # E-42：词表 >65536 的 tokenizer（如 yf-128k-bpe）必须走 u4 流——旧代码写死 uint16，
    # numpy 1.26 只发 DeprecationWarning 不报错，token id 静默按 65536 回绕喂错数据。
    vocab_size = int(tok.get_vocab_size())
    tok_dtype = "<u2" if vocab_size <= 65536 else "<u4"
    eod_id = tok.token_to_id(eod_token)
    if eod_id is None:
        raise SystemExit(f"tokenizer 未定义 EOD token: {eod_token!r}")

    implementation_sha = file_sha256(Path(__file__))
    table, order_sha, order_identity = load_bound_order(order_path, data_snapshot)

    source_files = sorted(s3_root.rglob("*.parquet"))
    source_fingerprint, source_identities = source_roster_fingerprint(s3_root, source_files)
    ids, fidx, ridx, files = build_doc_index(s3_root, workers=workers or 1)
    if files != source_files or source_roster_fingerprint(s3_root, files)[0] != source_fingerprint:
        raise ValueError("S3 source changed while preparing token materialization")
    fetcher = ShardTextFetcher(files)

    doc_ids = table.column("doc_id").to_pylist()
    reference_tokens = table.column("n_tokens").combine_chunks().to_numpy(zero_copy_only=False)
    n_docs = len(doc_ids)
    if max_tokens:
        keep = int(np.searchsorted(np.cumsum(reference_tokens, dtype=np.int64), max_tokens, side="right")) or 1
        doc_ids = doc_ids[:keep]
        print(f"max_tokens={max_tokens}: {n_docs} docs → {len(doc_ids)} docs")
    selected_reference_tokens = int(reference_tokens[:len(doc_ids)].sum(dtype=np.int64))
    print(f"materialize {name}: {len(doc_ids)} docs from {order_path.name}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"{name}.tokens.bin"
    # 边界策略是 part cache 身份的一部分，避免复用旧版“裸拼接”中间产物。
    boundary_key = hashlib.sha256(
        f"append-eod-v1\0{tok_sha}\0{eod_token}\0{eod_id}".encode("utf-8")
    ).hexdigest()[:12]

    # ---- pass 2（行组序两阶段，E-20/E-28 修复）----
    # 旧实现按 position 序（stage2=doc_id 均匀随机）逐批 fetch：每批触发 ~BATCH 个
    # 行组整读 + 64 条 FIFO 缓存对随机序无效——IO 放大 ~5000×。改为：
    # pass2a 按 (file, row_group) 顺序扫（每行组至多读一次；连续段切分 O(N)），
    # 每 CHUNK_GROUPS 个行组聚合落一个 part（块内 position 排序，避免千万小文件）；
    # pass2b heapq k 路归并 parts → 顺序拼接输出。
    import hashlib as _hl
    prep_dir = out_dir / ".prep_cache"
    prep_key = _hl.sha256(
        (f"{order_sha}\0{max_tokens}\0v8\0{implementation_sha}\0{source_fingerprint}\0"
         f"{SOURCE_DOCUMENT_BINDING_VERSION}\0"
         f"{PART_REFERENCE_TOKEN_LIMIT}:{PART_DOCUMENT_LIMIT}:"
         f"{ENCODE_BATCH_CHAR_LIMIT}:{ENCODE_BATCH_DOCUMENT_LIMIT}").encode("ascii")
    ).hexdigest()[:16]
    # A part cache belongs to one exact order and one boundary policy.  Without
    # prep_key, stale parts from an earlier order could be accepted on resume.
    tmp_dir = out_dir / f"{name}.parts-{prep_key}-{boundary_key}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    prep_npz = prep_dir / f"{prep_key}.npz"
    sfi = srg = srow = spos = boundary = None
    order_ids = hex_to_v16(doc_ids)
    if prep_npz.is_file():
        try:
            _z = np.load(prep_npz)
            sfi, srg, srow, spos = _z["sfi"], _z["srg"], _z["srow"], _z["spos"]
            boundary = _z["boundary"]
            n_groups = len(boundary) - 1
            print(f"  prep cache hit ({prep_key})", flush=True)
            print(f"pass2a: {len(doc_ids)} docs across {n_groups} row-groups（顺序扫描，每行组至多读 1 次）", flush=True)
        except Exception as exc:
            # E-68：被杀的写留下截断缓存——np.savez 非原子，必须校验后回退重算
            print(f"  prep cache corrupt ({type(exc).__name__}), recompute", flush=True)
            sfi = srg = srow = spos = boundary = None
    if sfi is None:
        pos, hit = lookup_sorted_doc_ids(ids, order_ids)
        if not bool(np.all(hit)):
            raise SystemExit(f"{int((~hit).sum())}/{len(order_ids)} doc_id 在 s3 索引中不存在（{order_path.name}）")
        sel_files = fidx[pos]
        sel_rows = ridx[pos]
        sel_pos = np.arange(len(order_ids), dtype=np.int64)

        rg_of = np.empty(len(order_ids), dtype=np.int64)
        # Group selected rows by file once.  The previous implementation built
        # a 95M-element boolean mask for every one of ~13K files, an accidental
        # O(n_files * n_docs) preparation path before tokenization even began.
        # Stable sorting preserves deterministic order inside a file.
        by_file = np.argsort(sel_files, kind="stable")
        sorted_files = sel_files[by_file]
        file_cuts = np.concatenate([
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(np.diff(sorted_files) != 0).astype(np.int64) + 1,
            np.asarray([len(sorted_files)], dtype=np.int64),
        ])
        for cut in range(len(file_cuts) - 1):
            a, b = int(file_cuts[cut]), int(file_cuts[cut + 1])
            selected = by_file[a:b]
            fi = int(sorted_files[a])
            md = pq.ParquetFile(files[fi]).metadata
            rows = [md.row_group(i).num_rows for i in range(md.num_row_groups)]
            bounds = np.concatenate([
                np.asarray([0], dtype=np.int64),
                np.cumsum(rows, dtype=np.int64),
            ])
            rg_of[selected] = np.searchsorted(
                bounds, sel_rows[selected], side="right"
            ) - 1

        scan_order = np.lexsort((sel_rows, rg_of, sel_files))
        sfi = sel_files[scan_order]
        srg = rg_of[scan_order]
        srow = sel_rows[scan_order]
        spos = sel_pos[scan_order]
        # 连续段边界（排序后同 (fi, rg) 连续）——E-54：必须 fi 或 rg 任一变都切段，
        # 旧代码只看 fi：同文件跨行组被并一段 → 用首个 rg 的 col 取后续行组 → 越界/取错正文
        boundary = np.flatnonzero((np.diff(sfi) != 0) | (np.diff(srg) != 0)) + 1
        boundary = np.concatenate([[0], boundary, [len(sfi)]])
        n_groups = len(boundary) - 1
        print(f"pass2a: {len(order_ids)} docs across {n_groups} row-groups（顺序扫描，每行组至多读 1 次）", flush=True)
        try:
            prep_dir.mkdir(parents=True, exist_ok=True)
            _tmp = prep_npz.with_name(prep_npz.name + f".tmp-{os.getpid()}.npz")  # E-68 原子写
            np.savez(_tmp, sfi=sfi, srg=srg, srow=srow, spos=spos, boundary=boundary)
            os.replace(_tmp, prep_npz)
            print(f"  prep cached ({prep_key})", flush=True)
        except OSError as exc:
            print(f"  [warn] prep cache write failed: {exc}", flush=True)

    # E-54：pass2a 并行化。单进程 tokenize 是 14-28h 级瓶颈（100B tokens）；chunk 组
    # 间零依赖（各自读行组/编码/写独立 part 文件），fork Pool 并行，确定性保持
    # （part 名=组序号固定，pass2b 归并逻辑不变）。scan 数组先落盘供 worker 共享。
    import multiprocessing as mp
    CHUNK_GROUPS = 2000
    scan_dir = tmp_dir / "_scan_arrays"
    scan_dir.mkdir(exist_ok=True)

    def _publish_scan_array(key: str, value: np.ndarray) -> None:
        dst = scan_dir / f"{key}.npy"
        if dst.is_file():
            try:
                cached = np.load(dst, mmap_mode="r")
                if cached.shape == value.shape and cached.dtype == value.dtype:
                    return
            except (OSError, ValueError):
                pass
        tmp = scan_dir / f".{key}.tmp-{os.getpid()}.npy"
        np.save(tmp, value)
        os.replace(tmp, dst)

    for key, value in (("sfi", sfi), ("srg", srg), ("srow", srow),
                       ("spos", spos), ("boundary", boundary), ("expected_ids", order_ids)):
        _publish_scan_array(key, value)
    files_tmp = scan_dir / f".files.tmp-{os.getpid()}.json"
    files_tmp.write_text(json.dumps([str(f) for f in files]), encoding="utf-8")
    os.replace(files_tmp, scan_dir / "files.json")
    identities_tmp = scan_dir / f".file_identities.tmp-{os.getpid()}.json"
    identities_tmp.write_text(json.dumps(source_identities, sort_keys=True))
    os.replace(identities_tmp, scan_dir / "file_identities.json")
    tokenizer_identity_tmp = scan_dir / f".tokenizer_sha256.tmp-{os.getpid()}.txt"
    tokenizer_identity_tmp.write_text(tok_sha + "\n")
    os.replace(tokenizer_identity_tmp, scan_dir / "tokenizer_sha256.txt")
    groups = token_bounded_chunks(reference_tokens, spos, boundary,
                                  max_tokens=PART_REFERENCE_TOKEN_LIMIT,
                                  max_docs=PART_DOCUMENT_LIMIT, max_groups=CHUNK_GROUPS)
    # E-55/57：part 级断点续跑——只补缺失或截断的 chunk。存在 ≠ 完好：被杀的 worker 会留
    # 半截 zip（np.savez 非原子），仅 is_file 判断会让 pass2b 在 np.load 处反复崩（2026-10-12
    # 实战 3 个截断 part 循环崩）。zip 中央目录校验很便宜。
    import zipfile as _zf
    def _part_ok(i: int) -> bool:
        pp = tmp_dir / f"part{i:06d}.npz"
        if not pp.is_file():
            return False
        try:
            with _zf.ZipFile(pp) as z:
                return set(z.namelist()) == {"positions.npy", "offsets.npy", "tokens.npy"}
        except Exception:
            return False
    before = len(groups)
    groups = [t for t in groups if not _part_ok(t[0])]
    if len(groups) < before:
        print(f"  part resume: {len(groups)} chunks remaining", flush=True)
    # An explicit function/CLI value is authoritative.  Keep the environment
    # variable only as the backwards-compatible default for direct callers.
    n_workers = (
        int(workers)
        if workers is not None
        else int(os.environ.get("YUFORMER_MAT_WORKERS", "32"))
    )
    n_workers = max(1, min(n_workers, len(groups)))
    print(f"pass2a parallel: {len(groups)} chunks, {n_workers} workers", flush=True)
    total = 0
    t0 = time.monotonic()
    # E-56：fork 死锁——pyarrow/numpy 已起线程后 fork，子进程继承锁状态 → 主进程
    # futex 永久等待（2026-10-12 实测）。spawn 子进程重导入本脚本（__main__ 守卫防重跑），
    # 慢 ~20s/worker 但无锁遗产。
    with single_thread_worker_environment(), mp.get_context("spawn").Pool(
            n_workers, initializer=configure_materialize_worker) as pool:
        done = 0
        for idx, n_tok in pool.imap_unordered(
            _pass2a_chunk,
            [(i, g0, g1, str(scan_dir), str(tokenizer_path), str(tmp_dir), tok_dtype, eod_id)
             for i, g0, g1 in groups],
        ):
            done += 1
            total += n_tok
            if done % 20 == 0 or done == len(groups):
                rate = total / max(1e-9, time.monotonic() - t0) / 1e6
                print(f"  pass2a {done}/{len(groups)} chunks, {total/1e6:.0f}M tok, {rate:.1f}M tok/s", flush=True)
    part_files = sorted(tmp_dir.glob("part*.npz"))

    # pass2b：Arrow LargeList gather + flatten，在 C++ 路径中按 position 归并。
    print(f"pass2b: merge {len(part_files)} parts by position", flush=True)
    staging_bin = out_dir / f".{name}.tokens.bin.tmp-{os.getpid()}"
    lengths_path = out_dir / f"{name}.doc_lengths.u32"
    staging_lengths = (out_dir / f".{name}.doc_lengths.u32.tmp-{os.getpid()}"
                       if write_doc_lengths else None)
    total = merge_parts_by_position(
        part_files,
        staging_bin,
        expected_docs=len(spos),
        tok_dtype=tok_dtype,
        doc_lengths_path=staging_lengths,
    )
    # E-59：total_tokens 以落盘 bin 为准——pass2a 断点续跑时 worker 回执只含本轮补跑
    # 的 chunk，旧累加口径会把 total 记成 0/部分值（2026-10-12 stage2 meta 实录 0）。
    disk_total = staging_bin.stat().st_size // np.dtype(tok_dtype).itemsize
    if total != disk_total:
        raise ValueError(f"pass2b token 计数与落盘字节不闭合: {total} != {disk_total}")

    meta = {
        "schema_version": 3 if write_doc_lengths else 2,
        "name": name,
        "total_tokens": int(total),
        "bytes": staging_bin.stat().st_size,
        "sha256": file_sha256(staging_bin),
        "edge_sha256": _edge_sha256(staging_bin),
        "tokenizer_sha256": tok_sha,
        "materializer_tool_sha256": implementation_sha,
        "native_merge_tool_sha256": file_sha256(Path(__file__).with_name("merge_token_parts.c")),
        "materialization": {"max_part_reference_tokens": PART_REFERENCE_TOKEN_LIMIT,
                            "max_part_documents": PART_DOCUMENT_LIMIT,
                            "max_encode_batch_characters": ENCODE_BATCH_CHAR_LIMIT,
                            "max_encode_batch_documents": ENCODE_BATCH_DOCUMENT_LIMIT,
                            "worker_threads": 1},
        "tokenizer": str(tokenizer_path),
        "source": f"s6:{order_path}",
        "s3_root": str(s3_root),
        "source_document_binding": {"schema_version": SOURCE_DOCUMENT_BINDING_VERSION,
                                    "order_bytes_verified": True,
                                    "source_handles_verified": True,
                                    "roster_fingerprint": source_fingerprint,
                                    "expected_doc_ids_verified": True,
                                    "tokenizer_bytes_verified": True},
        "stream_stat_identity": source_file_identity(staging_bin),
        "n_docs": len(doc_ids),
        "selection": {
            "requested_max_reference_tokens": int(max_tokens),
            "selected_docs": len(doc_ids),
            "selected_reference_tokens": selected_reference_tokens,
            "selected_text_tokens": int(total) - len(doc_ids),
            "reference_token_field": "n_tokens",
            "reference_tokens_are_selection_only": True,
            "actual_tokenizer_text_tokens": int(total) - len(doc_ids),
        },
        "seq_note": "正文 tokenize(add_special_tokens=False)，每篇文档末尾追加一个 EOD",
        "document_boundary": {
            "policy": "append_eod",
            "token": eod_token,
            "token_id": int(eod_id),
            "count": len(doc_ids),
        },
        "dtype": tok_dtype,
        "vocab_size": vocab_size,
        "order_sha256": order_sha,
        "order_bytes": order_identity["bytes"],
    }
    if staging_lengths is not None:
        if staging_lengths.stat().st_size != len(doc_ids) * 4:
            raise ValueError("逐文档 token 长度清单行数不闭合")
        lengths = np.memmap(staging_lengths, dtype="<u4", mode="r")
        if int(lengths.sum(dtype=np.uint64)) != int(total):
            raise ValueError("逐文档 token 长度之和与 token 流不闭合")
        meta["doc_lengths"] = {
            "file": lengths_path.name,
            "dtype": "<u4",
            "rows": len(doc_ids),
            "bytes": staging_lengths.stat().st_size,
            "sha256": file_sha256(staging_lengths),
            "edge_sha256": _edge_sha256(staging_lengths),
            "sum_tokens": int(total),
            "includes_appended_eod": True,
            "position_order": True,
        }
    if data_snapshot is not None:
        meta["data_snapshot_id"] = data_snapshot["snapshot_id"]
        meta["data_release_name"] = data_snapshot.get("release_name")
    if tokenizer_release is not None:
        meta["tokenizer_id"] = tokenizer_release.get("tokenizer_id")
        meta["tokenizer_gate_id"] = tokenizer_release.get("gate_id")
        meta["tokenizer_gate_sha256"] = tokenizer_release.get("_gate_sha256")
    meta_path = out_dir / f"{name}.meta.json"
    meta_tmp = out_dir / f".{name}.meta.json.tmp-{os.getpid()}"
    meta_tmp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    current_files = sorted(s3_root.rglob("*.parquet"))
    if current_files != files or source_roster_fingerprint(s3_root, files)[0] != source_fingerprint:
        raise ValueError("S3 source changed before token-stream publication")
    if file_sha256(Path(__file__)) != implementation_sha:
        raise ValueError("materializer source changed before token-stream publication")
    if file_sha256(tokenizer_path) != tok_sha:
        raise ValueError("tokenizer changed before token-stream publication")
    if source_file_identity(order_path) != order_identity:
        raise ValueError("order changed before token-stream publication")
    if publish_guard is not None:
        publish_guard()
    os.replace(staging_bin, bin_path)
    if staging_lengths is not None:
        if publish_guard is not None:
            publish_guard()
        os.replace(staging_lengths, lengths_path)
    if publish_guard is not None:
        publish_guard()
    os.replace(meta_tmp, meta_path)
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"done: {bin_path} ({total} tokens), meta {meta_path}", flush=True)
    return bin_path


def _hash_open_file(handle) -> str:
    handle.seek(0)
    digest = hashlib.sha256()
    while chunk := handle.read(16 << 20):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


def load_bound_order(path: Path, snapshot: dict | None = None):
    """Hash and parse one open inode, including across atomic path rollback."""
    with path.open("rb") as handle:
        identity = _stat_identity(os.fstat(handle.fileno()))
        digest = (_verify_snapshot_artifact(path, snapshot, handle=handle)
                  if snapshot is not None else _hash_open_file(handle))
        has_position = "position" in pq.ParquetFile(handle).schema_arrow.names
        columns = ["position", "doc_id", "n_tokens"] if has_position else ["doc_id", "n_tokens"]
        table = pq.read_table(handle, columns=columns)
        if (_stat_identity(os.fstat(handle.fileno())) != identity
                or source_file_identity(path) != identity):
            raise ValueError("order changed while loading verified input")
    if has_position:
        table = table.sort_by("position")
    else:
        table = table.append_column("position", pa.array(np.arange(table.num_rows), pa.int64()))
    return table, digest, identity


def _verify_snapshot_artifact(path: Path, snapshot: dict, *, handle=None) -> str:
    """Return SHA-256 after proving ``path`` is an exact snapshot artifact."""
    artifacts = snapshot.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise SystemExit("data snapshot 未包含带哈希的 artifacts，拒绝物化正式 token 流")
    resolved = path.resolve().as_posix()
    matches = [
        record for record in artifacts
        if isinstance(record, dict)
        and isinstance(record.get("path"), str)
        and (resolved == record["path"] or resolved.endswith("/" + record["path"]))
    ]
    if len(matches) != 1:
        raise SystemExit(f"输入表未唯一绑定到 data snapshot artifacts: {path}")
    record = matches[0]
    expected_sha = record.get("sha256")
    if not isinstance(expected_sha, str) or not expected_sha:
        raise SystemExit(f"data snapshot artifact 缺少 sha256: {record.get('path')}")
    identity = (_stat_identity(os.fstat(handle.fileno())) if handle is not None
                else source_file_identity(path))
    if int(record.get("bytes", -1)) != identity["bytes"]:
        raise SystemExit(f"输入表大小与 data snapshot 不一致: {path}")
    if all(key in record for key in identity) and any(
            int(record[key]) != value for key, value in identity.items()):
        raise ValueError(f"order stat identity differs from frozen release: {path}")
    actual = _hash_open_file(handle) if handle is not None else file_sha256(path)
    if actual != expected_sha:
        raise SystemExit(f"输入表 SHA-256 与 data snapshot 不一致: {path}")
    return actual


def _fetch_mixed(fetcher: ShardTextFetcher, fidx: np.ndarray, ridx: np.ndarray) -> list[str]:
    out: list[str | None] = [None] * len(fidx)
    for fi in np.unique(fidx):
        mask = fidx == fi
        rows = ridx[mask]
        got = fetcher.fetch(int(fi), rows)
        for j, t in zip(np.flatnonzero(mask), got):
            out[int(j)] = t
    return [t for t in out if t is not None]


def materialize_vals(val_dir: Path, s3_root: Path, tokenizer_path: Path, out_dir: Path,
                     max_tokens_per_domain: int = 0,
                     eod_token: str = DEFAULT_EOD_TOKEN,
                     data_snapshot: dict | None = None,
                     workers: int | None = None,
                     publish_guard: Callable[[], None] | None = None,
                     tokenizer_release: dict | None = None) -> list[str]:
    made = []
    for par in sorted(val_dir.glob("*.parquet")):
        domain = par.stem
        if domain.startswith("_"):
            continue
        materialize_stream(par, s3_root, tokenizer_path, out_dir, f"val_{domain}",
                           max_tokens=max_tokens_per_domain, eod_token=eod_token,
                           data_snapshot=data_snapshot, workers=workers,
                           publish_guard=publish_guard,
                           tokenizer_release=tokenizer_release)
        made.append(domain)
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description="Materialize pretrain token stream from S6 outputs")
    ap.add_argument("--order", required=True, help="mix_order/<mix>_stageN.parquet（可多次）")
    ap.add_argument("--s3-root", required=True)
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json 路径")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--eod-token", default=DEFAULT_EOD_TOKEN,
                    help="每篇文档末尾追加的 tokenizer token（默认 <|endoftext|>）")
    ap.add_argument("--data-snapshot", default="",
                    help="带 snapshot_id 的候选/正式 release manifest；身份写入所有 token stream meta")
    ap.add_argument("--tokenizer-gate", default="",
                    help="PASS TOKENIZER_READY.json；绑定 tokenizer id/hash 与选择证据")
    ap.add_argument("--val-dir", default="", help="val_sets 目录（有则逐域物化 val_<domain>.tokens.bin）")
    ap.add_argument("--val-only", action="store_true",
                    help="E-60：跳过主 order 物化只跑 val（编排器 val 步用——主流 meta 已存在时"
                         " 避免 main 先重跑整个 stage 再跑 val）")
    ap.add_argument("--val-max-tokens", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1,
                    help="tokenize worker processes (default: 1)")
    ap.add_argument("--write-doc-lengths", action="store_true",
                    help="发布 position 序 uint32 逐文档实际 token 长度清单（正式 1T overlap 门禁）")
    args = ap.parse_args()

    tok_path = Path(args.tokenizer)
    if tok_path.is_dir():
        tok_path = tok_path / "tokenizer.json"
    for p in ([Path(o) for o in [args.order]], [Path(args.s3_root)], [tok_path]):
        for x in p:
            if not x.exists():
                raise SystemExit(f"路径不存在: {x}")
    made = []
    data_snapshot = load_data_snapshot(args.data_snapshot) if args.data_snapshot else None
    tokenizer_release = None
    if args.tokenizer_gate:
        gate_path = Path(args.tokenizer_gate)
        tokenizer_release = json.loads(gate_path.read_text(encoding="utf-8"))
        tokenizer_release["_gate_sha256"] = file_sha256(gate_path)
    # Keep one writer over the shared part cache and final stream directory.
    with RunLease(Path(args.out_dir), f"materialize:{args.name}") as lease:
        if not args.val_only:
            made.append(str(materialize_stream(Path(args.order), Path(args.s3_root), tok_path,
                                               Path(args.out_dir), args.name, args.max_tokens,
                                               eod_token=args.eod_token,
                                               data_snapshot=data_snapshot,
                                               workers=args.workers,
                                               publish_guard=lease.assert_owner,
                                               tokenizer_release=tokenizer_release,
                                               write_doc_lengths=args.write_doc_lengths)))
        if args.val_dir:
            made += materialize_vals(Path(args.val_dir), Path(args.s3_root), tok_path,
                                     Path(args.out_dir), args.val_max_tokens,
                                     eod_token=args.eod_token,
                                     data_snapshot=data_snapshot,
                                     workers=args.workers,
                                     publish_guard=lease.assert_owner,
                                     tokenizer_release=tokenizer_release)
    print(json.dumps({"made": made}, indent=2))


if __name__ == "__main__":
    main()
