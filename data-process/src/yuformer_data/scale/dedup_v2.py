"""S3 web/code cross-source near-duplicate completion.

The production path uses the existing 128-permutation MinHash candidates, packs
candidate-only signatures to 4 bits per permutation (64 bytes/document), and
verifies each candidate by signature agreement.  The threshold is calibrated
against exact full-text 5-gram Jaccard samples; every domain must pass a
deterministic precision audit before any deletion decision is published.

Verified edges are reduced with external 16-way direct-edge dominance.  A
document is dropped only if it directly loses an edge under
``(source_priority, doc_id)``.  We intentionally do not take the graph's
transitive closure: approximate false-positive bridges therefore cannot merge
two otherwise unrelated families.  Raw partitions, partition reduction, and
streaming parquet/lookup writers keep memory bounded independently of total
edge count.  The old exact-text verifier and connected-component reducer remain
available as small-scale audit/reference paths.

All outputs use temporary files plus rename, and the decision summary is the
commit marker.  S5 consumes the sorted lookup directly.  ``ndv2_refresh`` is an
optional compatibility operation for consumers that still read the embedded
S3 ``s3_decision`` column; it is not on the S5/S6 critical path.
"""
from __future__ import annotations

import json
import hashlib
import logging
import math
import multiprocessing as mp
import os
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .dedup_common import (
    CSIG_SUMMARY,
    MemoryCapExceeded,
    S3_DROP,
    S3_KEEP,
    check_memory_cap,
    doc_id_to_bin,
    drops_for,
    fatal_summary,
    fingerprint,
    first_bytes,
    jaccard_sets,
    load_drops_lookup,
    rank_of,
    run_tasks,
    source_rank_map,
    stack_u8_to_void,
    write_summary,
    write_stage_summary,
    _word_spans,
)
from .dedup_minhash import (
    PAIRS_SCHEMA,
    S3_DECISIONS_SCHEMA,
    _load_source_names,
    _s3_domains,
    s3_decisions_path,
    s3_lookup_dir,
    s3_output_paths,
    s3_pairs_bydoc_dir,
    s3_store_dir,
)

log = logging.getLogger("yuformer.scale.dedup")

POLICY = "ndv2_minhash128_bbit4"
V2_STAGE = "ndv2"


def _malloc_trim() -> None:
    """把 glibc arena 空闲页归还内核。verify 单任务批量分配/释放 32MB 正文窗口
    （数万次），glibc 主 arena 只增不还——c0 桶实测 ru_maxrss 143GB（活动数据
    ~8GB），48 并发下 = E-13 灾难。每分块/每取文批后调用（E-14 口径：RSS 才是
    cgroup 计费对象，不是活动集）。"""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def s3_v2_verified_dir(out_root: Path, domain: str) -> Path:
    return out_root / "s3_v2" / "verified" / domain


def _rss_mb() -> int:
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
    except Exception:
        return -1


def _dir_fingerprint(path: Path) -> list[int]:
    """目录轻量指纹：(文件数, 总字节)。tmp+os.replace 语义下 size 变化即内容变化。"""
    n = 0
    size = 0
    if path.is_dir():
        for f in path.iterdir():
            if f.is_file():
                n += 1
                size += f.stat().st_size
    return [n, size]


def _write_verified_part(path: Path, verified: list[tuple[bytes, bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        table = pa.table(
            {f.name: pa.array([], type=f.type) for f in PAIRS_SCHEMA}, schema=PAIRS_SCHEMA
        )
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


def _write_verified_array(path: Path, verified: np.ndarray) -> None:
    """Write an (N,2) V16 edge array without constructing Python tuples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    if len(verified):
        u64 = np.ascontiguousarray(verified).view(np.uint64).reshape(-1, 4)
        table = pa.table(
            {"a0": u64[:, 0], "a1": u64[:, 1], "b0": u64[:, 2], "b1": u64[:, 3]},
            schema=PAIRS_SCHEMA,
        )
    else:
        table = pa.table(
            {field.name: pa.array([], type=field.type) for field in PAIRS_SCHEMA},
            schema=PAIRS_SCHEMA,
        )
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)


# ---------------------------------------------------------------- scalable b-bit MinHash verification

PACKED_CSIG_SUMMARY = "packed_csig.summary.json"


def packed_csig_dir(out_root: Path, domain: str) -> Path:
    return out_root / "s3_v2" / "packed_csig4" / domain


def _path_fp(path: Path) -> list[int]:
    if not path.is_file():
        return [-1, -1]
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def _pack_csig_byte(args: tuple[Path, str, int]) -> dict[str, Any]:
    out_root, domain, byte = args
    candidate_ids_path = flat_ctext_dir(out_root, domain) / f"{byte:02x}" / "ids.bin"
    source_root = out_root / "s3_csig" / domain / "_idx"
    source_ids_path = source_root / f"{byte:02x}.ids.bin"
    source_sig_path = source_root / f"{byte:02x}.sig.bin"
    dest_root = packed_csig_dir(out_root, domain)
    dest_root.mkdir(parents=True, exist_ok=True)
    dest_path = dest_root / f"{byte:02x}.sig4.bin"
    meta_path = dest_root / f"{byte:02x}.summary.json"
    inputs = {
        "candidate_ids": _path_fp(candidate_ids_path),
        "source_ids": _path_fp(source_ids_path),
        "source_signatures": _path_fp(source_sig_path),
    }
    if meta_path.is_file() and dest_path.is_file():
        prior = json.loads(meta_path.read_text("utf-8"))
        if prior.get("inputs") == inputs and prior.get("rows", 0) * 64 == dest_path.stat().st_size:
            prior["skipped"] = True
            return prior
    started = time.monotonic()
    if not candidate_ids_path.is_file():
        candidate_ids = np.empty(0, dtype="V16")
    else:
        candidate_ids = np.memmap(candidate_ids_path, dtype="V16", mode="r")
    if len(candidate_ids) == 0:
        tmp = dest_path.with_name(dest_path.name + f".tmp-{os.getpid()}")
        tmp.write_bytes(b"")
        os.replace(tmp, dest_path)
        payload = {
            "stage": "ndv2_pack_csig4", "domain": domain, "byte": byte,
            "rows": 0, "bytes": 0, "signature_bits": 4, "permutations": 128,
            "inputs": inputs, "missing": 0,
            "elapsed_sec": round(time.monotonic() - started, 3),
            "errors": {}, "skipped": False,
        }
        write_summary(meta_path, payload)
        return payload
    if not source_ids_path.is_file() or not source_sig_path.is_file():
        raise FileNotFoundError(f"csig source missing for {domain}/{byte:02x}")
    source_ids = np.memmap(source_ids_path, dtype="V16", mode="r")
    source_sigs = np.memmap(source_sig_path, dtype=np.uint32, mode="r")
    if source_sigs.size != len(source_ids) * 128:
        raise ValueError(f"csig shape mismatch for {domain}/{byte:02x}")
    source_sigs = source_sigs.reshape(-1, 128)
    tmp = dest_path.with_name(dest_path.name + f".tmp-{os.getpid()}")
    missing = 0
    with tmp.open("wb") as handle:
        for lo in range(0, len(candidate_ids), 500_000):
            query = np.asarray(candidate_ids[lo : lo + 500_000])
            pos = np.searchsorted(source_ids, query)
            valid = pos < len(source_ids)
            safe = np.minimum(pos, max(len(source_ids) - 1, 0))
            hit = valid & (source_ids[safe] == query) if len(source_ids) else np.zeros(len(query), bool)
            missing += int((~hit).sum())
            if not bool(np.all(hit)):
                raise ValueError(
                    f"packed csig {domain}/{byte:02x}: {int((~hit).sum())} candidate IDs missing"
                )
            sig = np.asarray(source_sigs[pos], dtype=np.uint32)
            low = np.asarray(sig & np.uint32(0xF), dtype=np.uint8)
            packed = low[:, 0::2] | (low[:, 1::2] << np.uint8(4))
            handle.write(np.ascontiguousarray(packed).tobytes())
    os.replace(tmp, dest_path)
    payload = {
        "stage": "ndv2_pack_csig4", "domain": domain, "byte": byte,
        "rows": int(len(candidate_ids)), "bytes": dest_path.stat().st_size,
        "signature_bits": 4, "permutations": 128, "inputs": inputs,
        "missing": missing, "elapsed_sec": round(time.monotonic() - started, 3),
        "errors": {}, "skipped": False,
    }
    write_summary(meta_path, payload)
    return payload


def build_packed_csig(out_root: Path, domain: str, workers: int) -> dict[str, Any]:
    """Pack candidate-only 128×uint32 signatures to 128×4-bit once per domain."""
    root = packed_csig_dir(out_root, domain)
    root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    with mp.Pool(processes=max(1, min(workers, 16))) as pool:
        for summary in pool.imap_unordered(
            _pack_csig_byte, [(out_root, domain, byte) for byte in range(256)], chunksize=1
        ):
            summaries.append(summary)
            if len(summaries) % 16 == 0 or len(summaries) == 256:
                log.info(
                    "ndv2 pack %s: %d/256 bins, %.2fB rows",
                    domain,
                    len(summaries),
                    sum(int(row.get("rows") or 0) for row in summaries) / 1e9,
                )
    summaries.sort(key=lambda row: int(row["byte"]))
    payload = {
        "stage": "ndv2_pack_csig4", "domain": domain,
        "bins": len(summaries), "rows": sum(int(row["rows"]) for row in summaries),
        "bytes": sum(int(row["bytes"]) for row in summaries),
        "errors": sum(1 for row in summaries if row.get("errors")),
        "signature_bits": 4, "permutations": 128,
    }
    write_summary(root / PACKED_CSIG_SUMMARY, payload)
    return payload


class PackedMinhashIndex:
    def __init__(self, out_root: Path, domain: str):
        self.out_root = out_root
        self.domain = domain
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _bin(self, byte: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._cache.get(byte)
        if cached is not None:
            return cached
        ids_path = flat_ctext_dir(self.out_root, self.domain) / f"{byte:02x}" / "ids.bin"
        sig_path = packed_csig_dir(self.out_root, self.domain) / f"{byte:02x}.sig4.bin"
        ids = np.memmap(ids_path, dtype="V16", mode="r") if ids_path.is_file() else np.empty(0, "V16")
        sigs_flat = np.memmap(sig_path, dtype=np.uint8, mode="r") if sig_path.is_file() else np.empty(0, np.uint8)
        if sigs_flat.size != len(ids) * 64:
            raise ValueError(f"packed signature shape mismatch: {self.domain}/{byte:02x}")
        cached = (ids, sigs_flat.reshape(-1, 64))
        self._cache[byte] = cached
        return cached

    def gather(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = np.empty((len(query), 64), dtype=np.uint8)
        hit_all = np.zeros(len(query), dtype=bool)
        fb = first_bytes(query)
        for byte in np.unique(fb):
            rows = np.flatnonzero(fb == byte)
            ids, sigs = self._bin(int(byte))
            if not len(ids):
                continue
            pos = np.searchsorted(ids, query[rows])
            valid = pos < len(ids)
            safe = np.minimum(pos, len(ids) - 1)
            hit = valid & (ids[safe] == query[rows])
            if bool(np.any(hit)):
                out[rows[hit]] = sigs[pos[hit]]
                hit_all[rows[hit]] = True
        return out, hit_all


_NIBBLE_MATCHES = np.array(
    [int((value & 0x0F) == 0) + int((value & 0xF0) == 0) for value in range(256)],
    dtype=np.uint8,
)


def _verify_pairs_minhash(
    pairs: np.ndarray, index: PackedMinhashIndex, cfg: dict[str, Any], counters: Counter[str]
) -> np.ndarray:
    v2 = cfg["s3_v2"]
    bits = int(v2.get("signature_bits", 4))
    if bits != 4:
        raise ValueError("only signature_bits=4 is implemented")
    jaccard = float(v2.get("minhash_jaccard_threshold", 0.8))
    collision = 1.0 / (1 << bits)
    required = int(np.ceil((collision + (1.0 - collision) * jaccard) * 128))
    chunk_size = int(v2.get("signature_chunk_pairs", 250_000))
    kept: list[np.ndarray] = []
    for lo in range(0, len(pairs), chunk_size):
        chunk = pairs[lo : lo + chunk_size]
        sig_a, hit_a = index.gather(chunk[:, 0])
        sig_b, hit_b = index.gather(chunk[:, 1])
        valid = hit_a & hit_b
        counters["signature_missing"] += int((~valid).sum())
        if bool(np.any(valid)):
            rows = np.flatnonzero(valid)
            matches = _NIBBLE_MATCHES[np.bitwise_xor(sig_a[rows], sig_b[rows])].sum(axis=1)
            accepted = rows[matches >= required]
            if len(accepted):
                kept.append(np.ascontiguousarray(chunk[accepted]))
        counters["signature_checked"] += int(len(chunk))
    counters["minhash_required_matches"] = required
    return np.concatenate(kept) if kept else np.empty((0, 2), dtype="V16")


def verify_pair_fulltext(text_a: str, text_b: str, cfg: dict[str, Any]) -> bool:
    """全文 5-gram Jaccard 判重（唯一判定入口，测试锚点）。禁用前缀截断（E-17 根因）。"""
    s3 = cfg["s3"]
    v2 = cfg.get("s3_v2") or {}
    n_gram = int(s3["shingle_n"])
    cap = int(v2.get("shingle_cap") or s3["verify_shingle_cap"])
    sa = fast_shingle_set(text_a, n_gram, cap)
    sb = fast_shingle_set(text_b, n_gram, cap)
    return jaccard_sets(sa, sb) >= float(s3["jaccard_threshold"])


# ---------------------------------------------------------------- 向量化 shingle 哈希
# verify 的 Jaccard 直接在正文 shingle 集合上计算，哈希函数是实现细节（对 (span 内容)
# 的确定性注入映射即可，碰撞贡献可忽略）；blake2b 逐 span Python 循环实测 ~0.7M spans/s
# 是全链 dominant cost（code 单桶 ~1 小时的根因）。改为 numpy 全向量化：词 token =
# (首 16 字节前缀 2×u64, 词长)（零填充，长度入混合——同前缀不同后缀词不碰撞），
# 5-gram = n_gram 个词 token 的乘加混合（uint64 自然溢出 = mod 2^64，确定性）。
# 词数 < n_gram 时整篇为单 shingle（与 S3 语义一致）。超标采样语义不变：
# np.unique（排序）后取前 cap = 确定性均匀采样。
_V2_P0 = np.uint64(0x9E3779B97F4A7C15)
_V2_P1 = np.uint64(0xC2B2AE3D27D4EB4F)
_V2_P2 = np.uint64(0x165667B19E3779F9)


def _word_tokens(enc: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.frombuffer(enc, dtype=np.uint8)
    starts, ends = _word_spans(enc)
    m = len(starts)
    if m == 0:
        e = np.empty(0, dtype=np.uint64)
        return e, e, e
    width = 16
    padded = np.zeros(len(arr) + width, dtype=np.uint8)
    padded[: len(arr)] = arr
    lens = (ends - starts).astype(np.int64)
    off = np.arange(width, dtype=np.int64)[None, :]
    idx = starts[:, None] + off
    # 词内字节取真值，词尾之后零填充——token 必须是词内容的纯函数（上下文无关），
    # 否则同一 5-gram 在两篇正文中因位置偏移得到不同哈希，Jaccard 系统性塌缩。
    # 下标无需 clip：max(starts)+15 ≤ len(enc)+14 < len(padded)。
    win = padded[idx]
    win = win * (off < lens[:, None])
    u = np.ascontiguousarray(win).view(np.uint64).reshape(m, 2)
    return u[:, 0].copy(), u[:, 1].copy(), lens.astype(np.uint64)


def _mix_tokens(t0: np.ndarray, t1: np.ndarray, tl: np.ndarray) -> np.ndarray:
    return ((t0 * _V2_P0 + t1) ^ (tl * _V2_P1)) * _V2_P2 + tl


def fast_shingle_set(text: str, n_gram: int, cap: int) -> np.ndarray:
    # uint64 标量乘法溢出告警静音：mod 2^64 溢出是混合设计的确定性组成部分
    with np.errstate(over="ignore"):
        return _fast_shingle_set_inner(text, n_gram, cap)


def _fast_shingle_set_inner(text: str, n_gram: int, cap: int) -> np.ndarray:
    enc = text.encode("utf-8")
    t0, t1, tl = _word_tokens(enc)
    m = len(t0)
    if m == 0:
        return np.empty(0, dtype=np.uint64)
    if m < n_gram:
        h = _mix_tokens(_mix_tokens(t0, t1, tl), tl, np.uint64(m))
    else:
        k = m - n_gram + 1
        acc0 = np.zeros(k, dtype=np.uint64)
        acc1 = np.zeros(k, dtype=np.uint64)
        for j in range(n_gram):
            w0 = t0[j : j + k]
            w1 = t1[j : j + k]
            wl = tl[j : j + k]
            acc0 = (acc0 ^ w0) * _V2_P0 + wl
            acc1 = (acc1 ^ w1) * _V2_P1 + wl
        h = acc0 * _V2_P2 + acc1
    u = np.unique(h)
    if cap is not None and len(u) > cap:
        u = u[:cap]
    return u


# ---------------------------------------------------------------- flat byte bins（s3_v2/ctext）
# runs2 parquet 的行组粒度（8192 行/组，text 列 ~40MB/组）对稠密区间读放大 ~2× 可接受，
# 但对 verify 的 b 侧稀疏伙伴点查是灾难：每点触发 1 个行组读 ≈ 40MB/篇 ⇒ 单任务 IO
# 达全语料级（faulthandler 实测：7k 对的小桶读 12GB 未出首个分块）。解法（E-12 排序键
# 区间化的专用布局）：候选正文按 doc_id 首字节预物化为 flat bin
#   s3_v2/ctext/<domain>/<byte>/{ids.bin(V16 排序), srcs.bin(u2), offs.bin(u8 n+1), texts.bin(拼接)}
# a 侧桶消费 = 顺序流式切片（零放大）；b 侧伙伴点查 = per-byte memmap searchsorted
# + 精确字节切片（16B id + 正文，零放大）；decide 的 source 查询同构。一次性构建：
# pass1 逐 run 顺序读（1× IO）写 per-byte 临时 part（run 内已按 doc_id 排序 ⇒ part 有序）；
# pass2 逐 byte 全量入内存排序归并（web 单 byte ≤ ~5M 文档 ≈ 26GB，cap 28GB、8 并发，
# E-13 核算 8×28=224GB << 1TB）。幂等：summary 齐在且 max_chars 一致即 skip。

FLAT_SUMMARY = "flat_ctext.summary.json"


def flat_ctext_dir(out_root: Path, domain: str) -> Path:
    return out_root / "s3_v2" / "ctext" / domain


def _flat_run_task(args: tuple[Path, Path, int, int]) -> int:
    run_path, tmp_dir, run_idx, max_chars = args
    table = pq.read_table(run_path, columns=["doc_id", "source_code", "text"])
    if table.num_rows == 0:
        return 0
    from .dedup_common import _hex_strings_to_void

    ids = _hex_strings_to_void(table.column("doc_id").to_numpy())
    srcs = table.column("source_code").to_numpy()
    texts = table.column("text").to_pylist()
    fb = first_bytes(ids).astype(np.int32)
    n_out = 0
    for byte in np.unique(fb):
        rows = np.flatnonzero(fb == byte)
        d = tmp_dir / f"{byte:02x}"
        d.mkdir(parents=True, exist_ok=True)
        sub_ids = np.ascontiguousarray(ids[rows])
        sub_srcs = np.ascontiguousarray(srcs[rows])
        chunks = [texts[int(r)][:max_chars].encode("utf-8") for r in rows]
        offs = np.zeros(len(chunks) + 1, dtype="<u8")
        np.cumsum([len(c) for c in chunks], out=offs[1:])
        stem = d / f"run-{run_idx:05d}"
        arrs = (
            (stem.with_suffix(".ids.bin"), sub_ids),
            (stem.with_suffix(".srcs.bin"), sub_srcs),
            (stem.with_suffix(".offs.bin"), offs),
        )
        for name, arr in arrs:
            tmp = name.with_name(name.name + f".tmp-{os.getpid()}")
            arr.tofile(tmp)
            os.replace(tmp, name)
        tpath = stem.with_suffix(".texts.bin")
        tmp = tpath.with_name(tpath.name + f".tmp-{os.getpid()}")
        tmp.write_bytes(b"".join(chunks))
        os.replace(tmp, tpath)
        n_out += len(rows)
    return n_out


def _flat_merge_task(args: tuple[Path, Path, int, float]) -> int:
    """逐 byte 归并：全量入内存（cap 检查）→ 稳定 argsort → 重排 texts → 写最终 bin。"""
    tmp_dir, out_dir, byte, cap_gb = args
    src = tmp_dir / f"{byte:02x}"
    if not src.is_dir():
        return 0
    ids_parts, srcs_parts, offs_parts, texts_parts = [], [], [], []
    est = 0
    for f in sorted(src.glob("run-*.ids.bin")):
        stem = f.name[: -len(".ids.bin")]
        ids_parts.append(np.fromfile(f, dtype="V16"))
        srcs_parts.append(np.fromfile(src / f"{stem}.srcs.bin", dtype="<u2"))
        offs_parts.append(np.fromfile(src / f"{stem}.offs.bin", dtype="<u8"))
        texts_parts.append((src / f"{stem}.texts.bin").read_bytes())
        est += len(ids_parts[-1]) * 26 + len(texts_parts[-1])
        check_memory_cap(est, cap_gb, f"flat merge byte={byte:02x}")
    ids = np.concatenate(ids_parts)
    srcs = np.concatenate(srcs_parts)
    # 各 part 的 offs 平移为全局字节区间
    starts, ends = [], []
    base = 0
    for offs, blob in zip(offs_parts, texts_parts):
        starts.append(offs[:-1] + base)
        ends.append(offs[1:] + base)
        base += len(blob)
    starts = np.concatenate(starts)
    ends = np.concatenate(ends)
    order = np.argsort(ids, kind="stable")
    ids, srcs = ids[order], srcs[order]
    # texts 按 order 重排：Python 循环一次性（5M 文档 ≈ 数秒），join 单趟
    blob_all = b"".join(texts_parts)
    del texts_parts
    texts_sorted = b"".join(blob_all[int(s) : int(e)] for s, e in zip(starts[order], ends[order]))
    del blob_all
    offs = np.zeros(len(ids) + 1, dtype="<u8")
    np.cumsum(ends[order] - starts[order], out=offs[1:])
    d = out_dir / f"{byte:02x}"
    d.mkdir(parents=True, exist_ok=True)
    for name, arr in (
        (d / "ids.bin", np.ascontiguousarray(ids)),
        (d / "srcs.bin", np.ascontiguousarray(srcs)),
        (d / "offs.bin", offs),
    ):
        tmp = name.with_name(name.name + f".tmp-{os.getpid()}")
        arr.tofile(tmp)
        os.replace(tmp, name)
    tpath = d / "texts.bin"
    tmp = tpath.with_name(tpath.name + f".tmp-{os.getpid()}")
    tmp.write_bytes(texts_sorted)
    os.replace(tmp, tpath)
    # 清理该 byte 的临时 part
    for f in src.iterdir():
        f.unlink()
    src.rmdir()
    _malloc_trim()
    return len(ids)


def build_flat_ctext(
    out_root: Path,
    domain: str,
    max_chars: int,
    workers: int,
    merge_cap_gb: float = 48.0,
) -> dict[str, Any]:
    """候选正文 → per-byte flat bins（幂等：summary 齐在且 max_chars 一致即 skip）。
    cap 48GB（2026-09-28：web byte=00 桶实测 28.0GiB 贴脸撞穿旧 cap 28——E-6 规则，
    按真实峰值 3-5 倍余量设界；8 并发 × 48 = 384GB << 1TB cgroup）。"""
    store_dir = s3_store_dir(out_root, domain)
    out_dir = flat_ctext_dir(out_root, domain)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / FLAT_SUMMARY
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text("utf-8"))
        if payload.get("max_chars") == max_chars:
            return payload
    run_files = sorted((store_dir / "_runs").glob("run-*.parquet"))
    if not run_files:
        payload = {"format": "flat_ctext", "docs": 0, "bytes": 256, "max_chars": max_chars}
        write_summary(summary_path, payload)
        return payload
    tmp_dir = out_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    n_docs = 0
    with mp.Pool(processes=max(1, workers)) as pool:
        for n in pool.imap_unordered(
            _flat_run_task,
            [(rf, tmp_dir, i, max_chars) for i, rf in enumerate(run_files)],
            chunksize=4,
        ):
            n_docs += n
    log.info("flat ctext pass1 %s: %d docs repartitioned in %.0fs", domain, n_docs, time.monotonic() - t0)
    n_final = 0
    with mp.Pool(processes=max(1, min(workers, 8))) as pool:
        for n in pool.imap_unordered(
            _flat_merge_task,
            [(tmp_dir, out_dir, b, merge_cap_gb) for b in range(256)],
        ):
            n_final += n
    if tmp_dir.is_dir():
        tmp_dir.rmdir()
    payload = {
        "format": "flat_ctext",
        "docs": int(n_final),
        "bytes": 256,
        "max_chars": max_chars,
        "elapsed_sec": round(time.monotonic() - t0, 3),
    }
    write_summary(summary_path, payload)
    return payload


class FlatCText:
    """per-byte flat bin 读取器：a 侧顺序切片 + b 侧 memmap 点查（零行组放大）。"""

    def __init__(self, base: Path):
        self.base = base
        summary = json.loads((base / FLAT_SUMMARY).read_text("utf-8"))
        self.n = int(summary["docs"])
        self.max_chars = int(summary["max_chars"])
        self._bins: dict[int, tuple[np.ndarray, np.ndarray, np.memmap, np.memmap]] = {}

    def _bin(self, byte: int):
        b = self._bins.get(byte)
        if b is None:
            d = self.base / f"{byte:02x}"
            ids = np.memmap(d / "ids.bin", dtype="V16", mode="r")
            srcs = np.memmap(d / "srcs.bin", dtype="<u2", mode="r")
            offs = np.memmap(d / "offs.bin", dtype="<u8", mode="r")
            texts = np.memmap(d / "texts.bin", dtype=np.uint8, mode="r")
            b = self._bins[byte] = (ids, srcs, offs, texts)
        return b

    def bucket_slice(
        self, byte: int, lo: int, hi: int
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """a 侧：bin 内 [lo,hi) 行（ids/srcs/texts）——顺序切片零放大。"""
        ids, srcs, offs, texts = self._bin(byte)
        hi = min(hi, len(ids))
        if lo >= hi:
            return np.empty(0, dtype="V16"), np.empty(0, dtype="<u2"), []
        t0, t1 = int(offs[lo]), int(offs[hi])
        blob = texts[t0:t1].tobytes()
        cuts = offs[lo + 1 : hi] - t0
        parts = np.split(np.frombuffer(blob, dtype=np.uint8), cuts)
        out = [p.tobytes().decode("utf-8") for p in parts]
        return np.ascontiguousarray(ids[lo:hi]), np.ascontiguousarray(srcs[lo:hi]), out

    def texts_for_ids(
        self, ids: np.ndarray, cap_gb: float = 2.0
    ) -> dict[bytes, str]:
        """b 侧：按首字节分组 searchsorted + 批量区间顺序读（32MB 窗口 flush，cap 检查）。

        禁止逐条 memmap 小读（网络 fs 上 111k 次 ~5KB 随机读实测 16ms/条，与 csig 页
        故障同病理）；命中稀疏跨全 bin 时窗口化防 span 放大（E-12）。
        内部再按 SEG 篇硬分片 + 存活字节账——单调用输入规模无上限时内存仍有界。"""
        out: dict[bytes, str] = {}
        if len(ids) == 0:
            return out
        out_bytes = 0  # 存活字节账 = 输出 dict + 当前窗口 blob（窗口 blob 读后即弃）
        SEG = 2_000
        for k0 in range(0, len(ids), SEG):
            out_bytes = self._texts_for_ids_seg(
                ids[k0 : k0 + SEG], out, out_bytes, cap_gb
            )
        return out

    def _texts_for_ids_seg(
        self, ids: np.ndarray, out: dict[bytes, str], out_bytes: int, cap_gb: float
    ) -> int:
        fb = first_bytes(ids).astype(np.int32)
        span_cap = 32 << 20
        for byte in np.unique(fb):
            q = ids[fb == byte]
            b_ids, _, b_offs, b_texts = self._bin(int(byte))
            if len(b_ids) == 0:
                continue
            pos = np.searchsorted(b_ids, q)
            pos = np.clip(pos, 0, len(b_ids) - 1)
            hit = np.flatnonzero(b_ids[pos] == q)
            if len(hit) == 0:
                continue
            hp = pos[hit]
            order = np.argsort(hp, kind="stable")
            hp, hit = hp[order], hit[order]
            i, n_hit = 0, len(hit)
            while i < n_hit:
                j = i
                lo = int(b_offs[hp[i]])
                while j < n_hit and int(b_offs[hp[j] + 1]) - lo <= span_cap:
                    j += 1
                hi = int(b_offs[hp[j - 1] + 1])
                blob = b_texts[lo:hi].tobytes()
                need = out_bytes + len(blob)
                if need > cap_gb * (1 << 30):
                    raise MemoryCapExceeded(
                        f"flat texts_for_ids byte={byte:02x}: needs {need / (1<<30):.2f} GiB "
                        f"> cap {cap_gb} GiB (n_ids={len(ids)}, out_bytes={out_bytes / (1<<30):.2f} GiB, "
                        f"blob={len(blob) / (1<<30):.2f} GiB, window=[{i},{j})/{n_hit})"
                    )
                # 逐篇 [start:end] 直切——np.split 无法表达不相交区间（其末段会吞掉
                # 倒数第二篇到最后一篇之间的全部非命中字节：window blob 32MB 时末篇
                # piece 达 4-11MB，out_bytes 虚增 ~150× 触发内存护栏，实测定位）
                starts = (b_offs[hp[i:j]] - lo).astype(np.int64)
                ends = (b_offs[hp[i:j] + 1] - lo).astype(np.int64)
                for k, h in enumerate(hit[i:j]):
                    piece = blob[starts[k] : ends[k]]
                    out[q[h].tobytes()] = piece.decode("utf-8")
                    out_bytes += len(piece)
                del blob
                i = j
        return out_bytes

    def sources_for_ids(self, ids: np.ndarray) -> np.ndarray:
        """decide 阶段批量 source 查询（int32，-1 = 未命中）。"""
        out = np.full(len(ids), -1, dtype=np.int32)
        if len(ids) == 0:
            return out
        fb = first_bytes(ids).astype(np.int32)
        for byte in np.unique(fb):
            mask = fb == byte
            q = ids[mask]
            qs = np.flatnonzero(mask)
            b_ids, b_srcs, _, _ = self._bin(int(byte))
            if len(b_ids) == 0:
                continue
            pos = np.searchsorted(b_ids, q)
            pos = np.clip(pos, 0, len(b_ids) - 1)
            hit = b_ids[pos] == q
            out[qs[hit]] = b_srcs[pos[hit]]
        return out


# ---------------------------------------------------------------- pass 1: 全文重验证

def _load_unique_pairs_bydoc(
    out_root: Path, domain: str, byte: int, cap_gb: float
) -> np.ndarray:
    """单桶候选对：流式读 chunk → 桶内 unique（跨 band 重复对消减，E-11 流式分桶的存量）。"""
    base = s3_pairs_bydoc_dir(out_root, domain) / f"{byte:02x}"
    chunks: list[np.ndarray] = []
    total = 0
    for f in sorted(base.glob("chunk-*.parquet")):
        table = pq.read_table(f)
        if table.num_rows == 0:
            continue
        arr = np.column_stack([table.column(c).to_numpy() for c in ("a0", "a1", "b0", "b1")])
        a = stack_u8_to_void([arr[:, 0], arr[:, 1]], 16)
        b = stack_u8_to_void([arr[:, 2], arr[:, 3]], 16)
        chunks.append(np.column_stack([a, b]))
        total += len(a) * 32
        check_memory_cap(total, cap_gb, f"ndv2 pairs {domain}/{byte:02x}")
    if not chunks:
        return np.empty((0, 2), dtype="V16")
    flat = np.ascontiguousarray(np.concatenate(chunks)).view("V32").ravel()
    del chunks
    uniq = np.unique(flat)  # 按 (a,b) 字典序排序，后续 a 切片依赖此序
    del flat
    return np.ascontiguousarray(uniq.view("V16").reshape(-1, 2))


def _open_csig(out_root: Path, domain: str) -> Path | None:
    p = out_root / "s3_csig" / domain
    if not (p / CSIG_SUMMARY).is_file():
        return None
    return p


def _load_csig_bin(csig_base: Path, byte: int) -> tuple[np.ndarray, np.ndarray] | None:
    """整 bin 批量入内存（np.fromfile 顺序读）——禁止 memmap 随机页故障读网络 fs：
    c0 桶实测 csig memmap gather 把进程打进 D-state 磁盘睡眠 20+ 分钟（rchar 停滞，
    与本仓库"ctext 网络 fs 随机挂起"同病理）。bin ≤ ~2.4GB（web）/455MB（code），
    每任务每 bin 仅装载一次，读后即弃（E-12 字节账由调用方控制）。"""
    ids_path = csig_base / "_idx" / f"{byte:02x}.ids.bin"
    if not ids_path.is_file():
        return None
    ids = np.fromfile(ids_path, dtype="V16")
    sigs = np.fromfile(csig_base / "_idx" / f"{byte:02x}.sig.bin", dtype=np.uint32)
    if len(ids) == 0:
        return None
    return ids, sigs.reshape(-1, 128)


def _sig_prefilter(
    cpairs: np.ndarray, csig_base: Path, byte: int, pre: float, counters: Counter[str]
) -> np.ndarray:
    """签名一致率预筛：agree < pre 的对直接拒绝（不经全文 intersect）。
    无签名（定位失败）保持 keep——预筛只能误放不能误杀（4.2σ 界）。"""
    keep = np.ones(len(cpairs), dtype=bool)
    a = cpairs[:, 0]
    b = cpairs[:, 1]
    a_bin = _load_csig_bin(csig_base, byte)
    if a_bin is None:
        return keep
    a_ids_bin, a_sigs = a_bin
    pos_a = np.searchsorted(a_ids_bin, a)
    pos_a = np.clip(pos_a, 0, len(a_ids_bin) - 1)
    pos_a = np.where(a_ids_bin[pos_a] == a, pos_a, -1)
    # E-43：a 侧签名缺失（-1）的行不得进入预筛——numpy 负索引会拿最后一条签名比对，
    # 意外"一致"时 keep=True 跳过 verify，真重复对被误放、非重复对被误杀（静默）。
    # 缺失 → 保持 keep=True？否：keep=True 语义是"预筛判重、跳过 verify"；缺失必须走
    # 完整 verify，即 keep 置 False。
    keep[pos_a < 0] = False
    fb = first_bytes(b).astype(np.int32)
    for b_byte in np.unique(fb):
        rows = np.flatnonzero(fb == b_byte)
        b_bin = _load_csig_bin(csig_base, int(b_byte))
        if b_bin is None:
            continue
        b_ids_bin, b_sigs = b_bin
        pos_b = np.searchsorted(b_ids_bin, b[rows])
        pos_b = np.clip(pos_b, 0, len(b_ids_bin) - 1)
        ok = (b_ids_bin[pos_b] == b[rows]) & (pos_a[rows] >= 0)
        sub = rows[ok]
        if len(sub) == 0:
            del b_bin, b_sigs
            continue
        sig_a = a_sigs[pos_a[sub]]
        sig_b = b_sigs[pos_b[ok]]
        agree = (sig_a == sig_b).mean(axis=1)
        drop = agree < pre
        keep[sub[drop]] = False
        counters["prefiltered"] += int(drop.sum())
        del b_bin, b_sigs, sig_a, sig_b
    return keep


def _verify_bucket(
    pairs: np.ndarray,
    fct: FlatCText,
    cs: Path | None,
    byte: int,
    cfg: dict[str, Any],
    counters: Counter[str],
) -> list[tuple[bytes, bytes]]:
    s3 = cfg["s3"]
    v2 = cfg["s3_v2"]
    n_gram = int(s3["shingle_n"])
    cap = int(v2.get("shingle_cap") or s3["verify_shingle_cap"])
    thresh = float(s3["jaccard_threshold"])
    pre = float(v2.get("pre_filter_min_agree") or 0.0)
    chunk_docs = int(v2["a_chunk_docs"])
    chunk_cap = float(v2["chunk_text_cap_gb"])
    b_cap = float(v2["b_cache_cap_gb"])

    a_ids = pairs[:, 0]  # np.unique 后非降序
    # 任务级预筛（一次性）：签名 bin 批量入内存（memmap 随机页故障会挂网络 fs，D-state
    # 实测——见 _load_csig_bin），与正文分块解耦。若按块预筛，
    # 每个 b-byte bin 会被重复加载（web：~235 块 × 全 b bin ≈ 141TB cache 读/任务——
    # E-14 page cache 预算不可能过）；任务级 = 每 bin 每任务仅装载一次。
    if cs is not None and pre > 0:
        t_pre = time.monotonic()
        keep = _sig_prefilter(pairs, cs, byte, pre, counters)
        counters["t_prefilter"] = int((time.monotonic() - t_pre) * 1000)
        pairs = pairs[keep]
        if len(pairs) == 0:
            return []
        a_ids = pairs[:, 0]
    a_unique = np.unique(a_ids)
    bin_ids = fct._bin(byte)[0]  # flat bin 按 doc_id 排序
    b_sets: dict[bytes, np.ndarray] = {}
    b_bytes = 0
    verified: list[tuple[bytes, bytes]] = []
    n_chunks = (len(bin_ids) + chunk_docs - 1) // chunk_docs

    for lo in range(0, len(bin_ids), chunk_docs):
        t_chunk = time.monotonic()
        ids_s, _, texts_s = fct.bucket_slice(byte, lo, lo + chunk_docs)
        est = sum(len(t) + 40 for t in texts_s)
        check_memory_cap(est, chunk_cap, f"ndv2 a-slice byte={byte:02x}")
        counters["t_fetch_a"] += int((time.monotonic() - t_chunk) * 1000)
        # 仅对出现在候选对 a 侧的文档算签名集（bin 内非 a 文档跳过）
        pos = np.searchsorted(a_unique, ids_s)
        pos = np.clip(pos, 0, len(a_unique) - 1)
        hit_rows = np.flatnonzero(a_unique[pos] == ids_s) if len(a_unique) else np.empty(0, np.int64)
        a_sets: dict[bytes, np.ndarray] = {}
        t_sh = time.monotonic()
        for i in hit_rows:
            a_sets[ids_s[i].tobytes()] = fast_shingle_set(texts_s[int(i)], n_gram, cap)
        del texts_s
        counters["t_shingle_a"] += int((time.monotonic() - t_sh) * 1000)
        if not a_sets:
            continue
        plo = int(np.searchsorted(a_ids, ids_s[0]))
        phi = int(np.searchsorted(a_ids, ids_s[-1], side="right"))
        cpairs = pairs[plo:phi]
        cand = cpairs
        del cpairs
        # b 侧签名集：FIFO 缓存（字节账），逐出后可再取，上限是重复计算而非正确性
        miss_b: list[bytes] = []
        for d in np.unique(cand[:, 1]):
            db = d.tobytes()
            if db not in b_sets:
                miss_b.append(db)
        t_bf = time.monotonic()
        if miss_b:
            # 分批取 b 正文（每批 ≤ FETCH_SLICE 篇）：单批正文有界（大文件域如 code
            # 单篇可达 100k 字符 ≈ 300KB，10k 篇/批实测 2GB+ 触顶——E-6：上限按真实
            # 峰值 3-5× 设），批间即算签名集并做缓存逐出，存活内存与批大小解耦（E-12）
            FETCH_SLICE = 4_000
            for s0 in range(0, len(miss_b), FETCH_SLICE):
                mtexts = fct.texts_for_ids(
                    np.frombuffer(b"".join(miss_b[s0 : s0 + FETCH_SLICE]), dtype="V16"),
                    cap_gb=4.0,
                )
                counters["b_fetched"] += len(mtexts)
                for d, t in mtexts.items():
                    s = fast_shingle_set(t, n_gram, cap)
                    b_sets[d] = s
                    b_bytes += s.nbytes + 64
                del mtexts
                while b_bytes > b_cap * (1 << 30) and b_sets:
                    _k = next(iter(b_sets))
                    b_bytes -= b_sets.pop(_k).nbytes + 64
                    counters["b_evicted"] += 1
                _malloc_trim()
        counters["t_fetch_b"] += int((time.monotonic() - t_bf) * 1000)
        del miss_b
        counters["fulltext_checked"] += len(cand)
        t_loop = time.monotonic()
        n_verified_before = len(verified)
        for row in cand:
            ab = row[0].tobytes()
            bb = row[1].tobytes()
            sa = a_sets.get(ab)
            if sa is None:
                continue
            sb = b_sets.get(bb)
            if sb is None:
                tb = fct.texts_for_ids(np.array([row[1]], dtype="V16")).get(bb)
                if tb is None:
                    counters["missing_text_b"] += 1
                    continue
                sb = fast_shingle_set(tb, n_gram, cap)
                b_sets[bb] = sb
                b_bytes += sb.nbytes + 64
            if jaccard_sets(sa, sb) >= thresh:
                verified.append((ab, bb))
        counters["t_pairloop"] += int((time.monotonic() - t_loop) * 1000)
        log.info(
            "ndv2 verify %s/%02x chunk %d/%d: pairs=%d checked=%d verified=%d b_cache=%.2fGB counters=%s",
            counters.get("_domain", "?"), byte, lo // chunk_docs + 1, n_chunks,
            phi - plo, len(cand), len(verified) - n_verified_before,
            b_bytes / (1 << 30), dict(counters),
        )
        del a_sets, cand
        _malloc_trim()
    return verified


def process_ndv2_verify(
    out_root: Path,
    domain: str,
    byte: int,
    cfg: dict[str, Any],
    expected_sources: list[str],
) -> dict[str, Any]:
    """单字节桶全文重验证任务。幂等：summary+part 齐在且 pairs 指纹未变即 skip；
    失败写 .error.json 且不落 part（E-16：skip 条件 = 产物存在且无错误）。"""
    v2 = cfg["s3_v2"]
    out_dir = s3_v2_verified_dir(out_root, domain)
    out_dir.mkdir(parents=True, exist_ok=True)
    part_path = out_dir / f"{byte:02x}.parquet"
    summary_path = out_dir / f"{byte:02x}.summary.json"
    pairs_fp = _dir_fingerprint(s3_pairs_bydoc_dir(out_root, domain) / f"{byte:02x}")
    method = str(v2.get("verification_method", "fulltext"))
    verification_contract = _verification_contract(out_root, domain, byte, v2)
    if summary_path.is_file() and part_path.is_file():
        existing = json.loads(summary_path.read_text("utf-8"))
        if (
            existing.get("pairs_fingerprint") == pairs_fp
            and existing.get("verification_contract") == verification_contract
            and not existing.get("errors")
            and existing.get("policy") == POLICY
        ):
            existing["skipped"] = True
            return existing

    started = time.monotonic()
    counters: Counter[str] = Counter()
    try:
        pairs = _load_unique_pairs_bydoc(
            out_root, domain, byte, float(v2["pairs_cap_gb"])
        )
        counters["pair_rows"] = int(len(pairs))
        if len(pairs):
            if method == "minhash_signature":
                verified_arr = _verify_pairs_minhash(
                    pairs, PackedMinhashIndex(out_root, domain), cfg, counters
                )
                verified = None
            elif method == "fulltext":
                fct = FlatCText(flat_ctext_dir(out_root, domain))
                cs = _open_csig(out_root, domain)
                counters["_domain"] = domain
                verified = _verify_bucket(pairs, fct, cs, byte, cfg, counters)
                counters.pop("_domain", None)
                verified_arr = None
            else:
                raise ValueError(f"unknown ndv2 verification_method={method!r}")
        else:
            verified = [] if method == "fulltext" else None
            verified_arr = np.empty((0, 2), dtype="V16") if method == "minhash_signature" else None
        del pairs
        if verified_arr is not None:
            _write_verified_array(part_path, verified_arr)
            counters["verified_passed"] = len(verified_arr)
        else:
            assert verified is not None
            _write_verified_part(part_path, verified)
            counters["verified_passed"] = len(verified)
    except Exception as exc:
        summary = fatal_summary("ndv2_verify", f"{domain}/{byte:02x}", exc, started)
        summary["domain"] = domain
        write_summary(out_dir / f"{byte:02x}.error.json", summary)
        return summary

    summary = {
        "stage": "ndv2_verify",
        "task": f"{domain}/{byte:02x}",
        "domain": domain,
        "policy": POLICY,
        "pairs_fingerprint": pairs_fp,
        "verification_contract": verification_contract,
        "domain_sources": expected_sources,
        "pair_rows": int(counters.get("pair_rows", 0)),
        "verified_passed": int(counters.get("verified_passed", 0)),
        "elapsed_sec": round(time.monotonic() - started, 3),
        "peak_rss_mb": _rss_mb(),
        "counters": dict(counters),
        "errors": {},
        "skipped": False,
    }
    write_summary(summary_path, summary)
    (out_dir / f"{byte:02x}.error.json").unlink(missing_ok=True)
    return summary


def _verification_contract(
    out_root: Path, domain: str, byte: int, v2: dict[str, Any]
) -> dict[str, Any]:
    method = str(v2.get("verification_method", "fulltext"))
    contract: dict[str, Any] = {
        "method": method,
        "policy": POLICY,
        "jaccard_threshold": float(v2.get("minhash_jaccard_threshold", 0.8)),
        "signature_bits": int(v2.get("signature_bits", 4)),
        "permutations": 128,
    }
    if method == "minhash_signature":
        contract["packed_index"] = _path_fp(
            packed_csig_dir(out_root, domain) / f"{byte:02x}.sig4.bin"
        )
    return contract


def _sample_verified_edges(verified_dir: Path, sample_size: int) -> np.ndarray:
    """Deterministic approximately-uniform sample over the published edge rows."""
    files: list[tuple[Path, int]] = []
    total = 0
    for path in sorted(verified_dir.rglob("*.parquet")):
        rows = int(pq.ParquetFile(path).metadata.num_rows)
        if rows:
            files.append((path, rows))
            total += rows
    if total == 0 or sample_size <= 0:
        return np.empty((0, 2), dtype="V16")
    n = min(sample_size, total)
    positions = ((np.arange(n, dtype=np.int64) * total) + total // 2) // n
    sampled: list[np.ndarray] = []
    offset = 0
    for path, rows in files:
        mask = (positions >= offset) & (positions < offset + rows)
        if not bool(np.any(mask)):
            offset += rows
            continue
        local = positions[mask] - offset
        table = pq.read_table(path, columns=["a0", "a1", "b0", "b1"])
        u64 = np.column_stack([table.column(c).to_numpy() for c in ("a0", "a1", "b0", "b1")])
        picked = u64[local]
        sampled.append(
            np.column_stack(
                [
                    stack_u8_to_void([picked[:, 0], picked[:, 1]], 16),
                    stack_u8_to_void([picked[:, 2], picked[:, 3]], 16),
                ]
            )
        )
        del table, u64, picked
        _malloc_trim()
        offset += rows
    return np.ascontiguousarray(np.concatenate(sampled))


def _wilson_lower(successes: int, trials: int, z: float = 1.96) -> float:
    if trials <= 0:
        return 0.0
    p = successes / trials
    z2 = z * z
    return float(
        (p + z2 / (2 * trials) - z * np.sqrt((p * (1 - p) + z2 / (4 * trials)) / trials))
        / (1 + z2 / trials)
    )


def _bbit_accept_probability(jaccard: float, required: int, bits: int = 4) -> float:
    """Binomial acceptance probability for 128 b-bit MinHash permutations."""
    collision = 1.0 / (1 << bits)
    p = collision + (1.0 - collision) * float(jaccard)
    return float(
        sum(
            math.comb(128, k) * (p**k) * ((1.0 - p) ** (128 - k))
            for k in range(required, 129)
        )
    )


def run_ndv2_precision_audit(
    out_root: Path,
    domain: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Gate approximate MinHash decisions with deterministic exact-text evidence."""
    v2 = cfg["s3_v2"]
    verified_dir = s3_v2_verified_dir(out_root, domain)
    edges_fp = _dir_fingerprint(verified_dir)
    sample_size = int(v2.get("precision_audit_sample_size", 1024))
    exact_threshold = float(cfg["s3"]["jaccard_threshold"])
    bits = int(v2.get("signature_bits", 4))
    approx_threshold = float(v2.get("minhash_jaccard_threshold", 0.8))
    collision = 1.0 / (1 << bits)
    required_matches = int(np.ceil((collision + (1.0 - collision) * approx_threshold) * 128))
    recall_reference = float(v2.get("recall_reference_jaccard", 0.85))
    theoretical_recall = _bbit_accept_probability(recall_reference, required_matches, bits)
    min_theoretical_recall = float(v2.get("min_theoretical_recall_at_reference", 0.70))
    contract = {
        "edges_fingerprint": edges_fp,
        "sample_method": "global_systematic_midpoint",
        "sample_size": sample_size,
        "exact_metric": f"word_{int(cfg['s3']['shingle_n'])}gram_jaccard",
        "exact_threshold": exact_threshold,
        "shingle_cap": int(v2["shingle_cap"]),
        "min_precision": float(v2.get("precision_audit_min", 0.97)),
        "min_wilson_lower_95": float(v2.get("precision_audit_wilson_min", 0.95)),
        "required_signature_matches": required_matches,
        "recall_reference_jaccard": recall_reference,
        "min_theoretical_recall_at_reference": min_theoretical_recall,
    }
    path = out_root / "s3_v2" / f"{domain}_minhash_precision_audit.json"
    if path.is_file():
        existing = json.loads(path.read_text("utf-8"))
        if existing.get("audit_contract") == contract and not existing.get("errors"):
            existing["skipped"] = True
            return existing

    started = time.monotonic()
    try:
        pairs = _sample_verified_edges(verified_dir, sample_size)
        if len(pairs) < sample_size:
            raise ValueError(
                f"precision audit requires {sample_size} verified edges, found {len(pairs)}"
            )
        fct = FlatCText(flat_ctext_dir(out_root, domain))
        ids = np.unique(np.ascontiguousarray(pairs).reshape(-1))
        texts = fct.texts_for_ids(
            ids, cap_gb=float(v2.get("precision_audit_text_cap_gb", 4.0))
        )
        log.info(
            "ndv2 precision audit %s: sampled=%d unique_docs=%d texts_found=%d",
            domain, len(pairs), len(ids), len(texts),
        )
        missing = [x.tobytes().hex() for x in ids if x.tobytes() not in texts]
        if missing:
            raise ValueError(
                f"precision audit missing {len(missing)}/{len(ids)} texts; first={missing[:3]}"
            )
        ngram = int(cfg["s3"]["shingle_n"])
        cap = int(v2["shingle_cap"])
        scores = np.empty(len(pairs), dtype=np.float64)
        for i, (a, b) in enumerate(pairs):
            scores[i] = jaccard_sets(
                fast_shingle_set(texts[a.tobytes()], ngram, cap),
                fast_shingle_set(texts[b.tobytes()], ngram, cap),
            )
            if (i + 1) % 128 == 0 or i + 1 == len(pairs):
                log.info("ndv2 precision audit %s: exact %d/%d", domain, i + 1, len(pairs))
        exact_positive = int((scores >= exact_threshold).sum())
        precision = exact_positive / len(scores)
        wilson = _wilson_lower(exact_positive, len(scores))
        passed = (
            precision >= contract["min_precision"]
            and wilson >= contract["min_wilson_lower_95"]
            and theoretical_recall >= min_theoretical_recall
        )
        summary = {
            "stage": "ndv2_precision_audit",
            "domain": domain,
            "audit_contract": contract,
            "sample_pairs": int(len(scores)),
            "texts_found": int(len(texts)),
            "exact_jaccard_ge_threshold": exact_positive,
            "precision": precision,
            "wilson_lower_95": wilson,
            "theoretical_acceptance_probability": {
                f"jaccard_{exact_threshold:g}": _bbit_accept_probability(
                    exact_threshold, required_matches, bits
                ),
                f"jaccard_{recall_reference:g}": theoretical_recall,
                "jaccard_0.9": _bbit_accept_probability(0.9, required_matches, bits),
            },
            "jaccard_quantiles": {
                str(q): float(np.quantile(scores, q))
                for q in (0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1)
            },
            "passed": passed,
            "elapsed_sec": round(time.monotonic() - started, 3),
            "errors": {} if passed else {"precision_gate_failed": 1},
            "skipped": False,
        }
    except Exception as exc:
        summary = fatal_summary("ndv2_precision_audit", domain, exc, started)
        summary.update({"domain": domain, "audit_contract": contract, "passed": False})
    write_summary(path, summary)
    return summary


# ---------------------------------------------------------------- pass 2: verified edges -> deterministic decisions

def _components_scipy(n: int, edges_idx: np.ndarray) -> np.ndarray:
    """scipy connected_components（C 实现；E-17 死因之二 = 纯 Python union-find 禁用）。"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    graph = coo_matrix(
        (np.ones(len(edges_idx), dtype=np.int8), (edges_idx[:, 0], edges_idx[:, 1])),
        shape=(n, n),
    ).tocsr()
    _n_comp, labels = connected_components(graph, directed=False)
    return labels.astype(np.int64)


def _cluster_survivors(
    nodes: np.ndarray,
    edges: np.ndarray,
    ranks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """返回 (drop_node_pos, dup_of_node_pos, comp_sizes, n_multi)。
    survivor = 簇内 (source_rank, doc_id 字典序) 最小者；全向量化（lexsort），
    无 Python 逐簇循环（web 规模簇数可达 1e8，逐簇循环 = E-17 式慢死）。"""
    n = len(nodes)
    pos = np.searchsorted(nodes, np.ascontiguousarray(edges).reshape(-1))
    edges_idx = np.clip(pos, 0, n - 1).reshape(-1, 2).astype(np.int32)
    labels = _components_scipy(n, edges_idx)
    del edges_idx, pos

    # doc_id 字典序决胜 = 字节序比较；uint64 默认小端解释 ≠ 字节序，需按 8 字节块 byteswap
    u = np.ascontiguousarray(nodes).view(np.uint64).reshape(n, 2)
    u_hi = u[:, 0].byteswap()
    u_lo = u[:, 1].byteswap()
    # lexsort 末键为主键：(label, rank, doc_id) 升序 → 每簇首行即 survivor
    order = np.lexsort((u_lo, u_hi, ranks, labels))
    del u, u_hi, u_lo
    lab_sorted = labels[order]
    first = np.flatnonzero(np.r_[True, lab_sorted[1:] != lab_sorted[:-1]])
    comp_sizes = np.diff(np.append(first, n))
    n_multi = int((comp_sizes >= 2).sum())
    surv_pos = order[first]
    is_surv = np.zeros(n, dtype=bool)
    is_surv[surv_pos] = True
    drop_pos = np.flatnonzero(~is_surv)
    lab_index = np.searchsorted(lab_sorted[first], labels[drop_pos])
    dup_of_pos = surv_pos[lab_index]
    return drop_pos, dup_of_pos, comp_sizes, n_multi


def _rank_lut(sources: list[str], rank_map: dict[str, int]) -> np.ndarray:
    """source_code → rank 查找表（末位 = 未知源兜底，len(rank_map)）。"""
    lut = np.array(
        [rank_of(rank_map, s) for s in sources] + [len(rank_map)], dtype=np.int64
    )
    return lut


def _write_lookup_vec(out_dir: Path, ids: np.ndarray, codes: np.ndarray) -> None:
    """向量化 drops 查找表（write_drops_lookup 的 np 版：web 规模 1e8 条不许 Python 循环）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    order = np.argsort(np.ascontiguousarray(ids), kind="stable")
    ids_s = np.ascontiguousarray(ids[order])
    codes_s = np.ascontiguousarray(np.asarray(codes, dtype=np.uint8)[order])
    ids_path = out_dir / "lookup.ids.bin"
    codes_path = out_dir / "lookup.codes.bin"
    tmp_i = ids_path.with_name(ids_path.name + f".tmp-{os.getpid()}")
    tmp_c = codes_path.with_name(codes_path.name + f".tmp-{os.getpid()}")
    ids_s.tofile(tmp_i)
    codes_s.tofile(tmp_c)
    os.replace(tmp_i, ids_path)
    os.replace(tmp_c, codes_path)


def _write_decisions_stream(
    path: Path, drop_ids: np.ndarray, dup_of_ids: np.ndarray, chunk_rows: int = 1_000_000
) -> None:
    """Bound Python string conversion and Arrow allocation for 100M+ decisions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    writer = pq.ParquetWriter(tmp, S3_DECISIONS_SCHEMA, compression="zstd")
    try:
        for lo in range(0, len(drop_ids), chunk_rows):
            hi = min(lo + chunk_rows, len(drop_ids))
            table = pa.table(
                {
                    "doc_id": [value.tobytes().hex() for value in drop_ids[lo:hi]],
                    "decision": pa.array([S3_DROP] * (hi - lo), pa.string()),
                    "dup_of": [value.tobytes().hex() for value in dup_of_ids[lo:hi]],
                },
                schema=S3_DECISIONS_SCHEMA,
            )
            writer.write_table(table)
    finally:
        writer.close()
    os.replace(tmp, path)


_EDGE_DECISION_DTYPE = np.dtype(
    [
        ("drop", "V16"),
        ("dup", "V16"),
        ("dup_rank", "<i4"),
        ("drop_source", "<i4"),
    ]
)


def _doc_order_words(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return big-endian uint64 words so numeric order equals doc-id byte order."""
    words = np.ascontiguousarray(ids).view(np.uint64).reshape(-1, 2)
    return words[:, 0].byteswap(), words[:, 1].byteswap()


def _direct_edge_losers(
    edges: np.ndarray,
    source_codes: np.ndarray,
    rank_lut: np.ndarray,
    n_sources: int,
) -> np.ndarray:
    """Turn each verified edge into one loser -> direct winner record.

    A document is dropped only when it directly loses a verified edge under
    ``(source_priority, doc_id)``.  This deliberately avoids connected-component
    chaining: one false-positive bridge cannot collapse two otherwise unrelated
    duplicate families.
    """
    if len(edges) == 0:
        return np.empty(0, dtype=_EDGE_DECISION_DTYPE)
    edges = np.ascontiguousarray(edges)
    non_self = edges[:, 0] != edges[:, 1]
    if not bool(np.all(non_self)):
        edges = edges[non_self]
        source_codes = source_codes.reshape(-1, 2)[non_self].reshape(-1)
    if len(edges) == 0:
        return np.empty(0, dtype=_EDGE_DECISION_DTYPE)
    src = source_codes.reshape(-1, 2)
    src_idx = np.where((src >= 0) & (src < n_sources), src, n_sources)
    ranks = rank_lut[src_idx]
    a_hi, a_lo = _doc_order_words(edges[:, 0])
    b_hi, b_lo = _doc_order_words(edges[:, 1])
    a_wins = (ranks[:, 0] < ranks[:, 1]) | (
        (ranks[:, 0] == ranks[:, 1])
        & ((a_hi < b_hi) | ((a_hi == b_hi) & (a_lo < b_lo)))
    )
    out = np.empty(len(edges), dtype=_EDGE_DECISION_DTYPE)
    out["drop"] = np.where(a_wins, edges[:, 1], edges[:, 0])
    out["dup"] = np.where(a_wins, edges[:, 0], edges[:, 1])
    out["dup_rank"] = np.where(a_wins, ranks[:, 0], ranks[:, 1]).astype(np.int32)
    out["drop_source"] = np.where(a_wins, src[:, 1], src[:, 0]).astype(np.int32)
    return out


def _partition_direct_edges(
    verified_dir: Path,
    work_dir: Path,
    fct: FlatCText,
    sources: list[str],
    rank_lut: np.ndarray,
    partitions: int,
) -> tuple[list[Path], int]:
    """Stream verified parquet files into a fixed number of raw decision partitions."""
    if partitions != 16:
        raise ValueError("edge_dominance_external currently requires 16 partitions")
    shutil.rmtree(work_dir, ignore_errors=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = [work_dir / f"part-{i:02x}.bin" for i in range(partitions)]
    handles = [p.open("wb") for p in paths]
    n_edges = 0
    try:
        for path in sorted(verified_dir.rglob("*.parquet")):
            table = pq.read_table(path, columns=["a0", "a1", "b0", "b1"])
            if table.num_rows == 0:
                continue
            u64 = np.column_stack([table.column(c).to_numpy() for c in ("a0", "a1", "b0", "b1")])
            edges = np.column_stack(
                [
                    stack_u8_to_void([u64[:, 0], u64[:, 1]], 16),
                    stack_u8_to_void([u64[:, 2], u64[:, 3]], 16),
                ]
            )
            src = fct.sources_for_ids(np.ascontiguousarray(edges).reshape(-1))
            records = _direct_edge_losers(edges, src, rank_lut, len(sources))
            if len(records):
                pidx = (first_bytes(records["drop"]) >> 4).astype(np.int8)
                for i in np.unique(pidx):
                    records[pidx == i].tofile(handles[int(i)])
            n_edges += int(len(records))
    finally:
        for handle in handles:
            handle.close()
    return paths, n_edges


def _reduce_direct_partition(path: Path, cap_gb: float) -> Path:
    """Select the best direct winner for each loser in one bounded partition."""
    if path.stat().st_size % _EDGE_DECISION_DTYPE.itemsize:
        raise ValueError(f"truncated direct-edge partition: {path}")
    check_memory_cap(path.stat().st_size * 3, cap_gb, f"ndv2 direct reduce {path.name}")
    out = path.with_name(path.stem + ".reduced.bin")
    tmp = out.with_name(out.name + f".tmp-{os.getpid()}")
    if path.stat().st_size == 0:
        tmp.write_bytes(b"")
        os.replace(tmp, out)
        return out
    rows = np.memmap(path, dtype=_EDGE_DECISION_DTYPE, mode="r")
    drop_hi, drop_lo = _doc_order_words(rows["drop"])
    dup_hi, dup_lo = _doc_order_words(rows["dup"])
    order = np.lexsort((dup_lo, dup_hi, rows["dup_rank"], drop_lo, drop_hi))
    ordered_drop = rows["drop"][order]
    first = np.flatnonzero(np.r_[True, ordered_drop[1:] != ordered_drop[:-1]])
    selected = np.asarray(rows[order[first]])
    selected.tofile(tmp)
    os.replace(tmp, out)
    return out


def _write_external_decisions(
    out_path: Path,
    lookup_dir: Path,
    reduced: list[Path],
    sources: list[str],
    dry_run: bool,
) -> tuple[int, Counter[str]]:
    """Stream reduced partitions to parquet and the sorted drop lookup."""
    dropped = 0
    per_source: Counter[str] = Counter()
    if dry_run:
        parquet_writer = None
        ids_fh = codes_fh = None
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lookup_dir.mkdir(parents=True, exist_ok=True)
        parquet_tmp = out_path.with_name(out_path.name + f".tmp-{os.getpid()}")
        ids_path = lookup_dir / "lookup.ids.bin"
        codes_path = lookup_dir / "lookup.codes.bin"
        ids_tmp = ids_path.with_name(ids_path.name + f".tmp-{os.getpid()}")
        codes_tmp = codes_path.with_name(codes_path.name + f".tmp-{os.getpid()}")
        parquet_writer = pq.ParquetWriter(parquet_tmp, S3_DECISIONS_SCHEMA, compression="zstd")
        ids_fh, codes_fh = ids_tmp.open("wb"), codes_tmp.open("wb")
    try:
        for path in reduced:
            if path.stat().st_size:
                rows = np.memmap(path, dtype=_EDGE_DECISION_DTYPE, mode="r")
            else:
                rows = np.empty(0, dtype=_EDGE_DECISION_DTYPE)
            if len(rows):
                uniq, counts = np.unique(rows["drop_source"], return_counts=True)
                for code, count in zip(uniq.tolist(), counts.tolist()):
                    name = sources[int(code)] if 0 <= int(code) < len(sources) else "?"
                    per_source[name] += int(count)
            dropped += int(len(rows))
            if dry_run:
                continue
            assert parquet_writer is not None and ids_fh is not None and codes_fh is not None
            for lo in range(0, len(rows), 1_000_000):
                chunk = rows[lo : lo + 1_000_000]
                parquet_writer.write_table(
                    pa.table(
                        {
                            "doc_id": [x.tobytes().hex() for x in chunk["drop"]],
                            "decision": pa.array([S3_DROP] * len(chunk), pa.string()),
                            "dup_of": [x.tobytes().hex() for x in chunk["dup"]],
                        },
                        schema=S3_DECISIONS_SCHEMA,
                    )
                )
                chunk["drop"].tofile(ids_fh)
                np.ones(len(chunk), dtype=np.uint8).tofile(codes_fh)
    finally:
        if not dry_run:
            assert parquet_writer is not None and ids_fh is not None and codes_fh is not None
            parquet_writer.close()
            ids_fh.close()
            codes_fh.close()
    if not dry_run:
        os.replace(parquet_tmp, out_path)
        os.replace(ids_tmp, ids_path)
        os.replace(codes_tmp, codes_path)
    return dropped, per_source


def _backup_e17_placeholder(out_path: Path) -> Path | None:
    bak = out_path.with_name(out_path.stem + ".e17_backup.parquet")
    if out_path.is_file() and not bak.is_file():
        shutil.copy2(out_path, bak)
        return bak
    return None


def _run_ndv2_direct_decide(
    out_root: Path,
    domain: str,
    cfg: dict[str, Any],
    expected_sources: list[str],
    dry_run: bool,
) -> dict[str, Any]:
    """External, bounded-memory direct-edge dominance decision path."""
    v2 = cfg["s3_v2"]
    verified_dir = s3_v2_verified_dir(out_root, domain)
    edges_fp = _dir_fingerprint(verified_dir)
    partitions = int(v2.get("decision_partitions", 16))
    contract = {
        "method": "edge_dominance_external",
        "partitions": partitions,
        "survivor_order": "source_priority_then_doc_id",
        "transitive_chaining": False,
    }
    summary_path = out_root / "s3_decisions" / f"{domain}.summary.json"
    if summary_path.is_file() and not dry_run:
        existing = json.loads(summary_path.read_text("utf-8"))
        if (
            existing.get("policy") == POLICY
            and existing.get("edges_fingerprint") == edges_fp
            and existing.get("decision_contract") == contract
            and not existing.get("errors")
        ):
            existing["skipped"] = True
            return existing
    if not dry_run:
        # The summary is the commit marker. Remove the prior generation before
        # mutating parquet/lookup files so downstream stages cannot observe a
        # mixed generation after interruption.
        summary_path.unlink(missing_ok=True)

    started = time.monotonic()
    work_dir = out_root / "s3_v2" / "decide_work" / domain
    try:
        fct = FlatCText(flat_ctext_dir(out_root, domain))
        sources = _load_source_names(s3_store_dir(out_root, domain))
        rank_map = source_rank_map(list(cfg.get("source_priority") or []))
        rank_lut = _rank_lut(sources, rank_map)
        parts, n_edges = _partition_direct_edges(
            verified_dir, work_dir, fct, sources, rank_lut, partitions
        )
        reduced = [
            _reduce_direct_partition(p, float(v2.get("decision_partition_cap_gb", 24.0)))
            for p in parts
        ]
        out_path = s3_decisions_path(out_root, domain)
        if not dry_run:
            _backup_e17_placeholder(out_path)
        dropped, per_source_drops = _write_external_decisions(
            out_path,
            s3_lookup_dir(out_root, domain),
            reduced,
            sources,
            dry_run,
        )
    except Exception as exc:
        summary = fatal_summary("ndv2_decide", domain, exc, started)
        summary.update({"domain": domain, "decision_contract": contract})
        write_summary(
            out_root / "s3_decisions" / f"{domain}.ndv2_decide.error.json", summary
        )
        return summary

    summary = {
        "stage": "ndv2_decide",
        "domain": domain,
        "policy": POLICY,
        "edges_fingerprint": edges_fp,
        "decision_contract": contract,
        "domain_sources": expected_sources,
        "clusters": None,
        "dropped": int(dropped),
        # Verification task summaries own `verified_passed`; using a distinct key
        # here prevents the stage aggregator from counting every edge twice.
        "verified_edge_rows": int(n_edges),
        "per_source_drops": dict(per_source_drops),
        "dry_run": dry_run,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "peak_rss_mb": _rss_mb(),
        "errors": {},
        "skipped": False,
    }
    if not dry_run:
        write_summary(summary_path, summary)
        (out_root / "s3_decisions" / f"{domain}.ndv2_decide.error.json").unlink(missing_ok=True)
        shutil.rmtree(work_dir, ignore_errors=True)
    return summary


def run_ndv2_cluster_decide(
    out_root: Path,
    domain: str,
    cfg: dict[str, Any],
    expected_sources: list[str],
    dry_run: bool = False,
) -> dict[str, Any]:
    """将已验证边转成 survivor 决定与 drops lookup。

    正式路径使用外排 direct-edge dominance；connected-components 仅保留为小规模
    对照。policy、边集指纹和决策合同完全一致时才允许幂等跳过。
    """
    v2 = cfg["s3_v2"]
    method = str(v2.get("clustering_method", "edge_dominance_external"))
    if method == "edge_dominance_external":
        return _run_ndv2_direct_decide(
            out_root, domain, cfg, expected_sources, dry_run
        )
    if method != "connected_components":
        raise ValueError(f"unknown ndv2 clustering_method={method!r}")
    verified_dir = s3_v2_verified_dir(out_root, domain)
    edges_fp = _dir_fingerprint(verified_dir)
    summary_path = out_root / "s3_decisions" / f"{domain}.summary.json"
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text("utf-8"))
        if (
            existing.get("policy") == POLICY
            and existing.get("edges_fingerprint") == edges_fp
            and not existing.get("errors")
        ):
            existing["skipped"] = True
            return existing

    started = time.monotonic()
    try:
        chunks: list[np.ndarray] = []
        est = 0
        for f in sorted(verified_dir.rglob("*.parquet")):
            table = pq.read_table(f)
            if table.num_rows == 0:
                continue
            arr = np.column_stack([table.column(c).to_numpy() for c in ("a0", "a1", "b0", "b1")])
            chunks.append(
                np.column_stack(
                    [
                        stack_u8_to_void([arr[:, 0], arr[:, 1]], 16),
                        stack_u8_to_void([arr[:, 2], arr[:, 3]], 16),
                    ]
                )
            )
            est += len(arr) * 32
            check_memory_cap(est, float(v2["edges_cap_gb"]), f"ndv2 edges {domain}")
        if chunks:
            edges = np.ascontiguousarray(np.unique(
                np.concatenate(chunks).view("V32").ravel()
            )).view("V16").reshape(-1, 2)
        else:
            edges = np.empty((0, 2), dtype="V16")
        del chunks
        n_edges = len(edges)
        nodes = (
            np.unique(np.ascontiguousarray(edges).view("V16").ravel())
            if n_edges
            else np.empty(0, dtype="V16")
        )
        n = len(nodes)
        check_memory_cap(
            n * 96 + n_edges * 64,
            float(v2["component_mem_cap_gb"]),
            f"ndv2 cluster nodes {domain}",
        )
        fct = FlatCText(flat_ctext_dir(out_root, domain))
        sources = _load_source_names(s3_store_dir(out_root, domain))
        rank_map = source_rank_map(list(cfg.get("source_priority") or []))
        lut = _rank_lut(sources, rank_map)
        if n:
            src_codes = fct.sources_for_ids(nodes)
            lut_idx = np.where(
                (src_codes >= 0) & (src_codes < len(sources)), src_codes, len(sources)
            )
            ranks = lut[lut_idx]
            drop_pos, dup_of_pos, comp_sizes, n_multi = _cluster_survivors(
                nodes, edges, ranks
            )
            del ranks, lut_idx
        else:
            src_codes = np.empty(0, dtype=np.int32)
            drop_pos = dup_of_pos = np.empty(0, dtype=np.int64)
            comp_sizes = np.empty(0, dtype=np.int64)
            n_multi = 0
        del edges
        drop_ids = np.ascontiguousarray(nodes[drop_pos])
        dup_of_ids = np.ascontiguousarray(nodes[dup_of_pos])
        # per_source_drops 向量化：复用节点级 src_codes，unique(code) → 计数 → 源名
        per_source_drops: Counter[str] = Counter()
        if len(drop_pos):
            src_codes_d = src_codes[drop_pos]
            uniq_codes, counts = np.unique(src_codes_d, return_counts=True)
            for c, k in zip(uniq_codes.tolist(), counts.tolist()):
                per_source_drops[
                    sources[int(c)] if 0 <= int(c) < len(sources) else "?"
                ] += int(k)
    except Exception as exc:
        summary = fatal_summary("ndv2_decide", domain, exc, started)
        summary["domain"] = domain
        write_summary(
            out_root / "s3_decisions" / f"{domain}.ndv2_decide.error.json", summary
        )
        return summary

    out_path = s3_decisions_path(out_root, domain)
    if not dry_run:
        _backup_e17_placeholder(out_path)
        _write_decisions_stream(out_path, drop_ids, dup_of_ids)
        _write_lookup_vec(
            s3_lookup_dir(out_root, domain), drop_ids,
            np.ones(len(drop_ids), dtype=np.uint8),
        )
    summary = {
        "stage": "ndv2_decide",
        "domain": domain,
        "policy": POLICY,
        "edges_fingerprint": edges_fp,
        "domain_sources": expected_sources,
        "clusters": n_multi,
        "dropped": int(len(drop_ids)),
        "verified_edge_rows": int(n_edges),
        "per_source_drops": dict(per_source_drops),
        "dry_run": dry_run,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "peak_rss_mb": _rss_mb(),
        "errors": {},
        "skipped": False,
    }
    if not dry_run:
        write_summary(summary_path, summary)
    return summary


# ---------------------------------------------------------------- pass 3: s3 产物刷新

def _lookup_fingerprint(out_root: Path, domain: str) -> list[int]:
    ids_path = s3_lookup_dir(out_root, domain) / "lookup.ids.bin"
    codes_path = s3_lookup_dir(out_root, domain) / "lookup.codes.bin"
    # Size alone is not an identity: two decision generations can contain the
    # same number of drops. Atomic replacement changes mtime_ns, which binds
    # refresh summaries to the exact published lookup generation.
    return [*_path_fp(ids_path), *_path_fp(codes_path)]


def process_shard_ndv2_refresh(
    out_root: Path,
    source: str,
    relpath: str,
    domain: str,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """流式读 s3/<source>/，按新 drops lookup 只重写 s3_decision 列。

    分片可达数十 GB，禁止 ``pq.read_table`` 整片驻内存。输入 batch 的其他列原样
    写入，summary 同时绑定原/新文件指纹和 doc_id 序列摘要，供下游证明此次原地
    迁移没有改变 decontam 所消费的文档集合。

    幂等：summary.ndv2_refresh_lookup 与当前 lookup 指纹一致即 skip；
    失败写 .ndv2_refresh.error.json 且不动原产物（E-16）。"""
    s3_path, summary_path = s3_output_paths(out_root, source, relpath)
    err_path = s3_path.with_name(s3_path.name + ".ndv2_refresh.error.json")
    lfp = _lookup_fingerprint(out_root, domain)
    current_fp = fingerprint(s3_path) if s3_path.is_file() else "missing"
    existing: dict[str, Any] = {}
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text("utf-8"))
        refresh_contract = (existing.get("ndv2_refresh") or {}).get("transform_contract") or {}
        if (
            existing.get("ndv2_refresh_lookup") == lfp
            and not existing.get("errors")
            and not err_path.exists()
            and refresh_contract.get("output_fingerprint") == current_fp
            and refresh_contract.get("method") == "stream_replace_s3_decision_v2"
        ):
            existing["skipped"] = True
            return existing
    started = time.monotonic()
    decisions: Counter[str] = Counter()
    tmp = s3_path.with_name(s3_path.name + f".tmp-{os.getpid()}")
    writer: pq.ParquetWriter | None = None
    rows_seen = 0
    doc_id_digest = hashlib.sha256()
    try:
        lookup = load_drops_lookup(s3_lookup_dir(out_root, domain))
        pf = pq.ParquetFile(s3_path)
        idx = pf.schema_arrow.get_field_index("s3_decision")
        if idx < 0:
            raise ValueError(f"s3_decision column missing: {s3_path}")
        writer = pq.ParquetWriter(tmp, pf.schema_arrow, compression="zstd")
        for batch in pf.iter_batches(batch_size=8192):
            doc_ids = batch.column(batch.schema.get_field_index("doc_id")).to_pylist()
            ids = np.array([doc_id_to_bin(d) for d in doc_ids], dtype="V16")
            codes = drops_for(ids, lookup)
            col = np.where(codes == 1, S3_DROP, S3_KEEP)
            arrays = list(batch.columns)
            arrays[idx] = pa.array(col, type=pf.schema_arrow.field(idx).type)
            writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=pf.schema_arrow))
            rows_seen += len(ids)
            doc_id_digest.update(np.ascontiguousarray(ids).tobytes())
            values, counts = np.unique(col, return_counts=True)
            decisions.update({str(v): int(n) for v, n in zip(values, counts)})
        writer.close()
        writer = None
        os.replace(tmp, s3_path)
        output_fp = fingerprint(s3_path)
        if err_path.exists():
            err_path.unlink()
    except Exception as exc:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        tmp.unlink(missing_ok=True)
        summary = fatal_summary("ndv2_refresh", f"{source}/{relpath}", exc, started)
        write_summary(err_path, summary)
        return summary

    # Preserve the original S3 materialization ledger. Refresh is an in-place
    # column update and must not erase S3 counts/provenance from its summary.
    summary = dict(existing)
    summary.update(
        {
            "source": source,
            "relpath": relpath,
            "domain": domain,
            "ndv2_refresh_lookup": lfp,
            "ndv2_refresh": {
                "rows": rows_seen,
                "decisions": dict(decisions),
                "elapsed_sec": round(time.monotonic() - started, 3),
                "transform_contract": {
                    "schema_version": 2,
                    "method": "stream_replace_s3_decision_v2",
                    "input_fingerprint": current_fp,
                    "output_fingerprint": output_fp,
                    "preserved_columns": [
                        name for name in pf.schema_arrow.names if name != "s3_decision"
                    ],
                    "doc_id_sha256": doc_id_digest.hexdigest(),
                },
            },
            "errors": {},
            "skipped": False,
        }
    )
    write_summary(summary_path, summary)
    return summary


# ---------------------------------------------------------------- orchestration

def _worker_ndv2_verify(args: tuple[tuple, dict[str, Any]]) -> dict[str, Any]:
    task, ctx = args
    _, domain, byte = task
    started = time.monotonic()
    try:
        return process_ndv2_verify(
            ctx["out_root"], domain, byte, ctx["cfg"], ctx["domain_sources"][domain]
        )
    except Exception as exc:
        summary = fatal_summary("ndv2_verify", f"{domain}/{byte:02x}", exc, started)
        summary["domain"] = domain
        return summary


def _worker_ndv2_refresh(args: tuple[tuple, dict[str, Any]]) -> dict[str, Any]:
    task, ctx = args
    _, source, relpath = task
    started = time.monotonic()
    try:
        return process_shard_ndv2_refresh(
            ctx["out_root"], source, relpath, ctx["src_domain"][source], ctx["cfg"]
        )
    except Exception as exc:
        summary = fatal_summary("ndv2_refresh", f"{source}/{relpath}", exc, started)
        summary["source"] = source
        return summary


def collect_published_ndv2(
    out_root: Path,
    profiles: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """Reconstruct complete ndv2 health after full or targeted retries."""
    v2 = cfg["s3_v2"]
    domain_sources = _s3_domains(out_root, profiles, None)
    rows: list[dict[str, Any]] = []
    for domain in v2.get("domains") or ["code", "web"]:
        if domain not in domain_sources:
            rows.append(
                {
                    "stage": "ndv2_pack_csig4", "domain": domain,
                    "shard_fatal": "configured domain has no published S3 sources",
                }
            )
            continue
        packed_path = packed_csig_dir(out_root, domain) / PACKED_CSIG_SUMMARY
        if packed_path.is_file():
            packed = json.loads(packed_path.read_text("utf-8"))
            rows.append(
                {
                    "stage": "ndv2_pack_csig4", "domain": domain,
                    "candidate_docs": int(packed.get("rows") or 0),
                    "errors": (
                        {"packed_bins_failed": int(packed.get("errors") or 0)}
                        if packed.get("errors") else {}
                    ),
                }
            )
        else:
            rows.append(
                {
                    "stage": "ndv2_pack_csig4", "domain": domain,
                    "shard_fatal": "packed signature summary missing",
                }
            )
        verified_dir = s3_v2_verified_dir(out_root, domain)
        for byte in range(256):
            summary_path = verified_dir / f"{byte:02x}.summary.json"
            part_path = verified_dir / f"{byte:02x}.parquet"
            if not summary_path.is_file() or not part_path.is_file():
                rows.append(
                    {
                        "stage": "ndv2_verify", "domain": domain,
                        "task": f"{domain}/{byte:02x}",
                        "shard_fatal": "published verify summary/part missing",
                    }
                )
                continue
            meta = json.loads(summary_path.read_text("utf-8"))
            if (
                meta.get("verification_contract")
                != _verification_contract(out_root, domain, byte, v2)
                or meta.get("pairs_fingerprint")
                != _dir_fingerprint(s3_pairs_bydoc_dir(out_root, domain) / f"{byte:02x}")
            ):
                meta = dict(meta)
                meta["shard_fatal"] = "published verification contract is stale"
            rows.append(meta)
        edges_fp = _dir_fingerprint(verified_dir)
        audit_path = out_root / "s3_v2" / f"{domain}_minhash_precision_audit.json"
        if audit_path.is_file():
            audit = json.loads(audit_path.read_text("utf-8"))
            if (
                not audit.get("passed")
                or (audit.get("audit_contract") or {}).get("edges_fingerprint") != edges_fp
            ):
                audit = dict(audit)
                audit["shard_fatal"] = "precision audit failed or is stale"
            rows.append(audit)
        else:
            rows.append(
                {
                    "stage": "ndv2_precision_audit", "domain": domain,
                    "shard_fatal": "precision audit missing",
                }
            )
        decision_path = out_root / "s3_decisions" / f"{domain}.summary.json"
        if decision_path.is_file():
            decision = json.loads(decision_path.read_text("utf-8"))
            if (
                decision.get("edges_fingerprint") != edges_fp
                or (decision.get("decision_contract") or {}).get("method")
                != str(v2.get("clustering_method", "edge_dominance_external"))
            ):
                decision = dict(decision)
                decision["shard_fatal"] = "published decision contract is stale"
            rows.append(decision)
        else:
            rows.append(
                {
                    "stage": "ndv2_decide", "domain": domain,
                    "shard_fatal": "decision summary missing",
                }
            )
    return rows


def run_ndv2(
    out_root: Path,
    profiles: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
    workers: int,
    sources: list[str] | None = None,
    domains: list[str] | None = None,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """ndv2 全链：256 桶/域验证（并行）→ 有界外排 survivor 决定（逐域串行）。"""
    v2 = cfg["s3_v2"]
    domain_sources = _s3_domains(out_root, profiles, sources)
    doms = [d for d in (domains or list(v2.get("domains") or ["web", "code"])) if d in domain_sources]
    if not doms:
        raise SystemExit(f"ndv2: no target domains with s2 products (wanted {domains or v2.get('domains')})")
    for d in doms:
        if not (s3_store_dir(out_root, d) / "docstore.summary.json").is_file():
            raise SystemExit(f"ndv2: docstore missing for {d}: {s3_store_dir(out_root, d)}")
        if not s3_pairs_bydoc_dir(out_root, d).is_dir():
            raise SystemExit(f"ndv2: pairs_bydoc missing for {d}: {s3_pairs_bydoc_dir(out_root, d)}")

    summaries: list[dict[str, Any]] = []
    n_workers = max(1, min(workers, int(v2["verify_workers"])))
    for domain in doms:
        t0 = time.monotonic()
        method = str(v2.get("verification_method", "fulltext"))
        # Candidate text bins already exist and provide the candidate-ID roster.
        # The production path packs candidate-only MinHash signatures once; the
        # historical exact-text verifier remains available for small audits.
        if method == "minhash_signature":
            payload = build_packed_csig(out_root, domain, max(1, min(workers, 16)))
            payload_stage = "ndv2_pack_csig4"
            payload_docs = int(payload["rows"])
        else:
            payload = build_flat_ctext(
                out_root, domain, int(v2["truncate_chars"]), max(1, min(workers, 48))
            )
            payload_stage = "ndv2_flat_ctext"
            payload_docs = int(payload["docs"])
        summaries.append(
            {
                "stage": payload_stage,
                "domain": domain,
                "candidate_docs": payload_docs,
                "elapsed_sec": round(time.monotonic() - t0, 3),
                "errors": {},
                "skipped": False,
            }
        )
        tasks = [("ndv2_verify", domain, b) for b in range(256)]
        log.info("ndv2 verify %s: %d byte-bucket tasks, %d workers", domain, len(tasks), n_workers)
        vsummaries = run_tasks(
            tasks,
            {"out_root": out_root, "cfg": cfg, "domain_sources": domain_sources},
            _worker_ndv2_verify,
            n_workers,
            "ndv2_verify",
        )
        summaries += vsummaries
        errs = sum(1 for s in vsummaries if s.get("errors") or s.get("shard_fatal"))
        passed = sum(int((s.get("counters") or {}).get("verified_passed") or 0) for s in vsummaries)
        if errs:
            # E-16：失败桶没落 part，此刻聚类会得到假"全 keep"——fail-fast 拒绝半成品
            write_stage_summary(
                out_root, "ndv2", collect_published_ndv2(out_root, profiles, cfg)
            )
            raise SystemExit(
                f"ndv2 verify {domain}: {errs}/256 buckets failed (see s3_v2/verified/{domain}/*.error.json); "
                "fix and rerun — cluster/decide refuses to run on partial edges"
            )
        log.info("ndv2 verify %s done: %d verified edges", domain, passed)
        if method == "minhash_signature":
            audit = run_ndv2_precision_audit(out_root, domain, cfg)
            summaries.append(audit)
            if audit.get("errors") or not audit.get("passed"):
                write_stage_summary(
                    out_root, "ndv2", collect_published_ndv2(out_root, profiles, cfg)
                )
                raise SystemExit(
                    f"ndv2 precision audit {domain} failed: precision={audit.get('precision')} "
                    f"wilson_lower_95={audit.get('wilson_lower_95')} errors={audit.get('errors')}"
                )
        decision = run_ndv2_cluster_decide(
            out_root, domain, cfg, domain_sources[domain], dry_run=dry_run
        )
        summaries.append(decision)
        if decision.get("errors") or decision.get("shard_fatal"):
            write_stage_summary(
                out_root, "ndv2", collect_published_ndv2(out_root, profiles, cfg)
            )
            raise SystemExit(
                f"ndv2 decide {domain} failed: {decision.get('error_message') or decision.get('errors')}"
            )
    write_stage_summary(
        out_root, "ndv2", collect_published_ndv2(out_root, profiles, cfg)
    )
    return summaries


def run_ndv2_refresh(
    out_root: Path,
    profiles: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
    workers: int,
    sources: list[str] | None = None,
    limit: int = 0,
) -> list[dict[str, Any]]:
    """s3/<source>/ 分片 s3_decision 列刷新（仅 web/code 域源）。"""
    v2 = cfg["s3_v2"]
    target_domains = set(v2.get("domains") or ["web", "code"])
    src_domain: dict[str, str] = {}
    for source, prof in profiles.items():
        if sources and source not in sources:
            continue
        if prof.get("domain") not in target_domains:
            continue
        src_dir = out_root / "s3" / source
        if src_dir.is_dir():
            src_domain[source] = prof["domain"]
    if not src_domain:
        raise SystemExit("ndv2_refresh: no web/code sources with s3 products")
    for source, domain in src_domain.items():
        lfp = _lookup_fingerprint(out_root, domain)
        if lfp[0] < 0:
            raise SystemExit(f"ndv2_refresh: drops lookup missing for {domain} (run --stage ndv2 first)")
    tasks: list[tuple] = []
    for source in src_domain:
        src_dir = out_root / "s3" / source
        for path in sorted(src_dir.rglob("*.parquet")):
            rel = path.relative_to(src_dir).as_posix()
            tasks.append(("ndv2_refresh", source, rel[: -len(".parquet")]))
    tasks.sort()
    if limit:
        tasks = tasks[:limit]
    n_workers = max(1, min(workers, int(v2["refresh_workers"])))
    log.info("ndv2_refresh: %d shard tasks (%d sources), %d workers", len(tasks), len(src_domain), n_workers)
    summaries = run_tasks(
        tasks,
        {"out_root": out_root, "cfg": cfg, "src_domain": src_domain},
        _worker_ndv2_refresh,
        n_workers,
        "ndv2_refresh",
    )
    # Rebuild the global summary from the complete published roster. Targeted
    # retries must never replace global health with a partial success view.
    published: list[dict[str, Any]] = []
    for source, prof in profiles.items():
        domain = prof.get("domain")
        if domain not in target_domains:
            continue
        src_dir = out_root / "s3" / source
        if not src_dir.is_dir():
            continue
        expected_fp = _lookup_fingerprint(out_root, str(domain))
        for s3_path in sorted(src_dir.rglob("*.parquet")):
            rel = s3_path.relative_to(src_dir).as_posix()[: -len(".parquet")]
            _, summary_path = s3_output_paths(out_root, source, rel)
            if not summary_path.is_file():
                published.append(
                    {
                        "stage": "ndv2_refresh", "source": source, "domain": domain,
                        "relpath": rel, "shard_fatal": "published S3 summary missing",
                    }
                )
                continue
            meta = json.loads(summary_path.read_text("utf-8"))
            row = dict(meta)
            row["stage"] = "ndv2_refresh"
            refresh = meta.get("ndv2_refresh") or {}
            row["in_rows"] = int(refresh.get("rows") or meta.get("in_rows") or 0)
            row["decisions"] = refresh.get("decisions") or meta.get("decisions") or {}
            if meta.get("ndv2_refresh_lookup") != expected_fp:
                row["shard_fatal"] = "refresh lookup generation is stale"
            published.append(row)
    write_stage_summary(out_root, "ndv2_refresh", published)
    errs = sum(1 for s in summaries if s.get("errors") or s.get("shard_fatal"))
    published_errs = sum(
        1 for s in published if s.get("errors") or s.get("shard_fatal")
    )
    if errs:
        log.warning("ndv2_refresh finished with %d/%d shard errors (rerun to retry)", errs, len(summaries))
    if published_errs:
        log.warning(
            "ndv2_refresh publication incomplete: %d/%d stale or failed shards",
            published_errs,
            len(published),
        )
    return summaries
