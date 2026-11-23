from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import os
import re
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from . import detectors
from .ingest import write_summary

log = logging.getLogger("yuformer.scale.dedup")

DEFAULT_DEDUP_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "dedup_config.yaml"
)

PROGRESS_EVERY = 50

_WS_ALL_RE = re.compile(r"\s+")

# 决策值（落 s2_decision / s3_decision 列与 decisions 文件）
S2_KEEP = "keep"
S2_DROP = "drop_exact_dup"
S3_KEEP = "keep"
S3_DROP = "drop_near_dup"


class MemoryCapExceeded(RuntimeError):
    pass


def load_dedup_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_DEDUP_CONFIG_PATH
    if not p.is_file():
        raise FileNotFoundError(f"dedup config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"dedup config not a mapping: {p}")
    cfg = dict(raw)
    cfg.setdefault("source_priority", [])
    cfg.setdefault("s2", {})
    cfg.setdefault("s3", {})
    s2 = cfg["s2"]
    s2.setdefault("num_buckets", 4096)
    s2.setdefault("reduce_range", 64)
    s2.setdefault("memory_cap_gb", 6.0)
    s3 = cfg["s3"]
    s3.setdefault("shingle_n", 5)
    s3.setdefault("sig_perms", 128)
    s3.setdefault("bands", 16)
    s3.setdefault("rows_per_band", 8)
    s3.setdefault("lsh_buckets", 2048)
    s3.setdefault("truncate_chars", 100_000)
    s3.setdefault("group_pair_cap", 50)
    s3.setdefault("jaccard_threshold", 0.8)
    s3.setdefault("verify_shingle_cap", 100_000)
    s3.setdefault("reduce_range", 64)
    s3.setdefault("memory_cap_gb", 6.0)
    s3.setdefault("docstore_run_rows", 100_000)
    s3.setdefault("verify_bucket_cap_gb", 32.0)
    s3.setdefault("verify_buckets", 256)
    s3.setdefault("pairs_cap_gb", 4.0)
    s3.setdefault("verify_text_chars", 2048)
    s3.setdefault("sig_agree_threshold", 0.8)
    if s3["sig_perms"] != s3["bands"] * s3["rows_per_band"]:
        raise ValueError("s3: sig_perms must equal bands * rows_per_band")
    # ndv2（2026-09-26，逆转 E-17）：web/code 全文 shingle 重验证 + scipy 聚类补齐。
    v2 = cfg.setdefault("s3_v2", {})
    v2.setdefault("domains", ["web", "code"])
    v2.setdefault("truncate_chars", 100_000)   # 全文验证输入截断（与签名一致；禁用 2048 前缀——E-17 根因）
    v2.setdefault("shingle_cap", 100_000)      # 单篇验证 shingle 上限（排序后取前 cap，确定性均匀采样）
    v2.setdefault("pre_filter_min_agree", 0.65)  # MinHash 签名一致率预筛：J_est<0.65 ⇒ 真实 J≥0.8 概率 ~1e-5（4.2σ）
    v2.setdefault("a_chunk_docs", 20_000)      # verify a 侧分块文档数（块内计算即弃，E-12）
    v2.setdefault("chunk_text_cap_gb", 2.0)    # 单块正文加载上限
    v2.setdefault("b_cache_cap_gb", 6.0)       # b 侧 shingle 集 FIFO 缓存字节账
    v2.setdefault("pairs_cap_gb", 8.0)         # 单字节桶候选对加载上限（web 单桶 ~1GB，余量 8×）
    v2.setdefault("edges_cap_gb", 32.0)        # decide 阶段存留边累加上限（预估 1-5 亿边 = 3-16GB）
    v2.setdefault("component_mem_cap_gb", 96.0)  # 聚类节点/lexsort 内存上限
    v2.setdefault("verify_workers", 48)        # E-13：单任务峰值 ~10GB × 48 = 480GB << 1TB cgroup
    v2.setdefault("refresh_workers", 64)
    return cfg


def source_rank_map(priority: list[str]) -> dict[str, int]:
    return {src: i for i, src in enumerate(priority)}


def rank_of(rank_map: dict[str, int], source: str) -> int:
    return rank_map.get(source, len(rank_map))


def normalize_for_hash(text: str) -> str:
    t = detectors.normalize_text(text).casefold()
    return _WS_ALL_RE.sub(" ", t).strip()


def text_sha256(text: str) -> bytes:
    return hashlib.sha256(normalize_for_hash(text).encode("utf-8")).digest()


def doc_id_to_bin(doc_id: str) -> bytes:
    return bytes.fromhex(doc_id)


def doc_bin_to_id(b: Any) -> str:
    if isinstance(b, np.void):
        return b.tobytes().hex()
    return b.hex()


def check_memory_cap(nbytes: float, cap_gb: float, what: str) -> None:
    if nbytes > cap_gb * (1 << 30):
        raise MemoryCapExceeded(
            f"{what}: needs {nbytes / (1<<30):.2f} GiB > cap {cap_gb} GiB"
        )


def stack_u8_to_void(cols: list[np.ndarray], size: int) -> np.ndarray:
    joined = np.stack(cols, axis=1)
    flat = np.ascontiguousarray(joined).reshape(len(cols[0]), size // 8)
    return flat.view(np.dtype(f"V{size}")).ravel()


def first_bytes(void_arr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(void_arr).view(np.uint8).reshape(-1, 16)[:, 0]


def fingerprint(path: Path) -> str:
    st = path.stat()
    return f"{st.st_size}:{st.st_mtime_ns}"


def void_to_stack_u8(arr: np.ndarray, n_cols: int) -> list[np.ndarray]:
    flat = np.ascontiguousarray(arr).view(np.uint64).reshape(len(arr), n_cols)
    return [flat[:, i].copy() for i in range(n_cols)]


# ---------------------------------------------------------------- MinHash

_SHINGLE_PERSON = b"yf-s3-shingle"
_PERM_PERSON = b"yf-s3-perm\x00\x00\x00\x00\x00\x00"
_BLAKE2B8_TEMPLATE = hashlib.blake2b(digest_size=8, person=_SHINGLE_PERSON)


_WS_BYTES = (9, 10, 11, 12, 13, 32)


def _word_spans(enc: bytes) -> tuple[np.ndarray, np.ndarray]:
    arr = np.frombuffer(enc, dtype=np.uint8)
    if len(arr) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    ws = np.zeros(len(arr), dtype=bool)
    for b in _WS_BYTES:
        ws |= arr == b
    start_flags = ~ws
    start_flags[1:] &= ws[:-1]
    end_flags = ~ws
    end_flags[:-1] &= ws[1:]
    return np.flatnonzero(start_flags), np.flatnonzero(end_flags) + 1


def shingle_hashes(text: str, n: int, cap: int | None = None) -> np.ndarray:
    enc = text.encode("utf-8")
    starts, ends = _word_spans(enc)
    m = len(starts)
    if m == 0:
        return np.empty(0, dtype=np.uint64)
    if m < n:
        spans = [(int(starts[0]), int(ends[-1]))]
    else:
        spans = [(int(starts[i]), int(ends[i + n - 1])) for i in range(m - n + 1)]
    view = memoryview(enc)
    out: list[bytes] = []
    ap = out.append
    for s, e in spans:
        h = _BLAKE2B8_TEMPLATE.copy()
        h.update(view[s:e])
        ap(h.digest())
    digests = b"".join(out)
    out = np.frombuffer(digests, dtype=np.uint64)
    if cap is not None and len(out) > cap:
        out = np.sort(out)[:cap]
    return out


def unique_shingle_set(text: str, n: int, cap: int) -> np.ndarray:
    h = shingle_hashes(text, n, cap=cap)
    if len(h) == 0:
        return h
    return np.unique(h)


@lru_cache(maxsize=1)
def minhash_perms(n_perms: int) -> tuple[np.ndarray, np.ndarray]:
    a = np.empty(n_perms, dtype=np.uint64)
    b = np.empty(n_perms, dtype=np.uint64)
    for i in range(n_perms):
        digest = hashlib.blake2b(
            i.to_bytes(4, "big"), digest_size=24, person=_PERM_PERSON
        ).digest()
        # numpy 1.x（系统 numpy 1.24）下 python int | np.uint64 对大整数抛 TypeError
        # （NEP 50 弱提升是 2.0 行为）；纯 Python 按位或再取模 2^64，数值与 numpy 2 口径逐位一致。
        a[i] = (int.from_bytes(digest[:8], "big") | 1) % (1 << 64)
        b[i] = int.from_bytes(digest[8:16], "big")
    return a, b


def minhash_signature(hashes: np.ndarray, n_perms: int) -> np.ndarray:
    if len(hashes) == 0:
        return np.zeros(n_perms, dtype=np.uint32)
    a, b = minhash_perms(n_perms)
    mins = np.full(n_perms, np.iinfo(np.uint64).max, dtype=np.uint64)
    prod = np.empty((8192, n_perms), dtype=np.uint64)
    for lo in range(0, len(hashes), 8192):
        chunk = hashes[lo : lo + 8192]
        m = len(chunk)
        np.multiply(chunk[:, None], a[None, :], out=prod[:m])
        prod[:m] += b
        np.minimum(mins, prod[:m].min(axis=0), out=mins)
    return (mins & np.uint64(0xFFFFFFFF)).astype(np.uint32)


_BAND_TEMPLATE = hashlib.blake2b(digest_size=16)


def band_keys(sig_bytes: bytes, bands: int, rows: int) -> list[bytes]:
    out = []
    for bidx in range(bands):
        h = _BAND_TEMPLATE.copy()
        h.update(sig_bytes[bidx * rows * 4 : (bidx + 1) * rows * 4])
        out.append(h.digest())
    return out


def jaccard_sets(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    inter = len(np.intersect1d(a, b, assume_unique=True))
    return inter / (len(a) + len(b) - inter)


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[bytes, bytes] = {}
        self.rank: dict[bytes, int] = {}

    def find(self, x: bytes) -> bytes:
        parent = self.parent
        if x not in parent:
            parent[x] = x
            self.rank[x] = 0
            return x
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(self, a: bytes, b: bytes) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def components(self) -> list[list[bytes]]:
        groups: dict[bytes, list[bytes]] = {}
        for x in list(self.parent):
            groups.setdefault(self.find(x), []).append(x)
        return list(groups.values())


# ---------------------------------------------------------------- drops lookup
# 每域一份：排序的 doc_id(16B) + 决策码(1B) 平铺 bin 文件，memmap + searchsorted 做流式 join。

DropsLookup = tuple[np.memmap, np.memmap]


def write_drops_lookup(
    out_dir: Path, doc_ids: list[bytes], codes: list[int]
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    order = np.argsort(np.array(doc_ids, dtype="V16"), kind="stable")
    ids = np.empty(len(order), dtype="V16")
    cd = np.empty(len(order), dtype=np.uint8)
    for i, o in enumerate(order):
        ids[i] = doc_ids[o]
        cd[i] = codes[o]
    ids_path = out_dir / "lookup.ids.bin"
    codes_path = out_dir / "lookup.codes.bin"
    tmp_i = ids_path.with_name(ids_path.name + f".tmp-{os.getpid()}")
    tmp_c = codes_path.with_name(codes_path.name + f".tmp-{os.getpid()}")
    tmp_i.write_bytes(ids.tobytes())
    tmp_c.write_bytes(cd.tobytes())
    os.replace(tmp_i, ids_path)
    os.replace(tmp_c, codes_path)
    return ids_path, codes_path


def load_drops_lookup(out_dir: Path) -> DropsLookup:
    ids_path = out_dir / "lookup.ids.bin"
    codes_path = out_dir / "lookup.codes.bin"
    if ids_path.stat().st_size == 0:
        return np.empty(0, dtype="V16"), np.empty(0, dtype=np.uint8)
    ids = np.memmap(ids_path, dtype="V16", mode="r")
    codes = np.memmap(codes_path, dtype=np.uint8, mode="r")
    return ids, codes


def drops_for(ids: np.ndarray, lookup: DropsLookup) -> np.ndarray:
    if len(lookup[0]) == 0:
        return np.zeros(len(ids), dtype=np.uint8)
    pos = np.searchsorted(lookup[0], ids)
    pos = np.clip(pos, 0, len(lookup[0]) - 1)
    hit = lookup[0][pos] == ids
    out = np.zeros(len(ids), dtype=np.uint8)
    out[hit] = lookup[1][pos[hit]]
    return out


# ---------------------------------------------------------------- docstore
# 候选文档正文外存 docstore（run 区间化，无全局归并）：
#   _runs/run-*.parquet        每 run 内按 doc_id（16B blake2b）排序（写 run 时保证）
#   _runs/run-*.ids.bin        每 run 的 doc_id 列副本（V16 平铺，memmap + searchsorted 定位）
#   docstore.summary.json      {"format": "runs2", "docs", "runs"}
# doc_id 首字节 = k 的文档在每个 run 内都是连续区间 ⇒ verify 按桶取文本只需 per-run
# searchsorted + 行组切片，无需跨 run 归并（旧 heapq.merge 全局物化既慢又触发 page-cache OOM）。
# 旧四 bin 格式（ids/srcs/offs/texts.bin）仍被兼容读取（6 个小域存量 docstore）。

SOURCE_CODE_CSV = "sources.csv"


def _source_codes(out_dir: Path, sources: list[str]) -> dict[str, int]:
    mapping = {s: i for i, s in enumerate(sorted(sources))}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / SOURCE_CODE_CSV).write_text(
        "\n".join(sorted(sources, key=lambda s: mapping[s])) + "\n", encoding="utf-8"
    )
    return mapping


_HEX_VAL_LO = np.full(256, -1, dtype=np.int16)
for _c in range(256):
    _ch = chr(_c)
    if "0" <= _ch <= "9":
        _HEX_VAL_LO[_c] = ord(_ch) - 48
    elif "a" <= _ch <= "f":
        _HEX_VAL_LO[_c] = ord(_ch) - 87


def _hex_strings_to_void(arr: np.ndarray) -> np.ndarray:
    raw = np.char.encode(arr.astype(str), "ascii").tobytes()
    vals = _HEX_VAL_LO[np.frombuffer(raw, dtype=np.uint8)].astype(np.uint8)
    out = (vals.reshape(-1, 32)[:, 0::2] << 4) | vals.reshape(-1, 32)[:, 1::2]
    return np.ascontiguousarray(out).view("V16").ravel()


def build_docstore(
    part_files: list[Path],
    out_dir: Path,
    run_rows: int,
    cap_gb: float,
    sources: list[str],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "docstore.summary.json"
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        # 存量四 bin 格式（无 format 标记）或已建好的 runs2 都直接复用
        if payload.get("format") == "runs2" or (out_dir / "texts.bin").is_file():
            return payload

    codes = _source_codes(out_dir, sources)
    run_dir = out_dir / "_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_files = sorted(run_dir.glob("run-*.parquet"))
    if not run_files:
        run_idx = 0
        buf_ids: list[bytes] = []
        buf_src: list[int] = []
        buf_text: list[str] = []
        for pf_path in sorted(part_files):
            pf = pq.ParquetFile(pf_path)
            for batch in pf.iter_batches(batch_size=8192):
                dcol = batch.column("doc_id").to_pylist()
                scol = batch.column("source").to_pylist()
                tcol = batch.column("text").to_pylist()
                for d, s, t in zip(dcol, scol, tcol):
                    buf_ids.append(doc_id_to_bin(d))
                    buf_src.append(codes[s])
                    buf_text.append(t)
                    if len(buf_ids) >= run_rows:
                        _write_run(run_dir, run_idx, buf_ids, buf_src, buf_text)
                        run_idx += 1
                        buf_ids, buf_src, buf_text = [], [], []
        if buf_ids:
            _write_run(run_dir, run_idx, buf_ids, buf_src, buf_text)
        run_files = sorted(run_dir.glob("run-*.parquet"))

    # run 区间化索引：每 run 写 doc_id 列副本（V16 平铺），行数取 parquet footer
    n_docs = 0
    n_bytes = 0
    for rf in run_files:
        sidecar = rf.with_suffix(".ids.bin")
        if not sidecar.is_file():
            ids = _hex_strings_to_void(
                pq.read_table(rf, columns=["doc_id"]).column("doc_id").to_numpy()
            )
            check_memory_cap(len(ids) * 16 + (1 << 26), cap_gb, f"docstore index {rf.name}")
            tmp = sidecar.with_name(sidecar.name + f".tmp-{os.getpid()}")
            tmp.write_bytes(ids.tobytes())
            os.replace(tmp, sidecar)
        n_docs += pq.ParquetFile(rf).metadata.num_rows
    payload = {"format": "runs2", "docs": n_docs, "runs": len(run_files), "text_bytes": n_bytes}
    write_summary(summary_path, payload)
    return payload


def _write_run(
    run_dir: Path, idx: int, ids: list[bytes], srcs: list[int], texts: list[str]
) -> None:
    order = np.argsort(np.array(ids, dtype="V16"), kind="stable")
    schema = pa.schema(
        [
            pa.field("doc_id", pa.string(), nullable=False),
            pa.field("source_code", pa.uint16(), nullable=False),
            pa.field("text", pa.string(), nullable=False),
        ]
    )
    path = run_dir / f"run-{idx:05d}.parquet"
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
        for lo in range(0, len(order), 8192):
            chunk = order[lo : lo + 8192]
            table = pa.table(
                {
                    "doc_id": [ids[i].hex() for i in chunk],
                    "source_code": np.uint16([srcs[i] for i in chunk]),
                    "text": [texts[i] for i in chunk],
                },
                schema=schema,
            )
            writer.write_table(table)
    os.replace(tmp, path)


class DocStore:
    """双模式候选正文读取器。

    legacy（四 bin 全局物化）：ids/srcs/offs/texts.bin 存在且无 format 标记（6 个存量小域）。
    runs2（run 区间化）：runs 已按 doc_id 排序，verify 按 doc_id 首字节分桶，
    每桶文本 = 各 run 内连续区间，per-run searchsorted + 行组切片，无全局归并。
    """

    def __init__(self, out_dir: Path):
        self.dir = out_dir
        summary = json.loads((out_dir / "docstore.summary.json").read_text("utf-8"))
        self.n = int(summary["docs"])
        self.legacy = summary.get("format") != "runs2" and (out_dir / "ids.bin").is_file()
        self.texts_path = out_dir / "texts.bin"
        if self.legacy:
            if self.n == 0:
                self.ids = np.empty(0, dtype="V16")
                self.srcs = np.empty(0, dtype="<u2")
                self.offs = np.zeros(1, dtype="<u8")
                return
            self.ids = np.memmap(out_dir / "ids.bin", dtype="V16", mode="r")
            self.srcs = np.memmap(out_dir / "srcs.bin", dtype="<u2", mode="r")
            self.offs = np.memmap(out_dir / "offs.bin", dtype="<u8", mode="r")
            return
        self.run_files = sorted((out_dir / "_runs").glob("run-*.parquet"))
        self.run_ids = [
            np.memmap(rf.with_suffix(".ids.bin"), dtype="V16", mode="r")
            for rf in self.run_files
        ]
        self._pf_cache: dict[int, pq.ParquetFile] = {}
        self._rg_rows: dict[int, int] = {}

    # ---------------- legacy 模式（存量小域，接口保持不变） ----------------

    def positions(self, doc_ids: np.ndarray) -> np.ndarray:
        if not self.legacy:
            raise NotImplementedError("positions() 仅 legacy docstore 支持")
        if self.n == 0:
            return np.full(len(doc_ids), -1, dtype=np.int64)
        pos = np.searchsorted(self.ids, doc_ids)
        pos = np.clip(pos, 0, self.n - 1)
        hit = self.ids[pos] == doc_ids
        out = np.full(len(doc_ids), -1, dtype=np.int64)
        out[hit] = pos[hit]
        return out

    def _legacy_texts(self, idxs: np.ndarray) -> list[str]:
        if len(idxs) == 0:
            return []
        starts = self.offs[idxs]
        ends = self.offs[idxs + 1]
        lo = int(starts.min())
        hi = int(ends.max())
        with self.texts_path.open("rb") as fh:
            fh.seek(lo)
            blob = fh.read(hi - lo)
        return [
            blob[int(s - lo) : int(e - lo)].decode("utf-8") for s, e in zip(starts, ends)
        ]

    def texts_for_bucket(
        self, bucket_byte: int, vb: int, max_chars: int, cap_gb: float
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        if not self.legacy:
            return self._runs_texts_for_bucket(bucket_byte, vb, max_chars, cap_gb)
        if self.n == 0:
            return np.empty(0, dtype="V16"), np.empty(0, dtype="<u2"), []
        fb = first_bytes(self.ids).astype(np.int32)
        idx = np.flatnonzero(fb % vb == bucket_byte)
        texts = [t[:max_chars] for t in self._legacy_texts(idx)]
        return (
            np.array(self.ids[idx], dtype="V16"),
            np.array(self.srcs[idx], dtype="<u2"),
            texts,
        )

    def texts_and_sources_for_ids(
        self, doc_ids: np.ndarray, max_chars: int, cap_gb: float
    ) -> tuple[dict[bytes, str], dict[bytes, int]]:
        if not self.legacy:
            return self._runs_texts_and_sources_for_ids(doc_ids, max_chars, cap_gb)
        texts: dict[bytes, str] = {}
        srcs: dict[bytes, int] = {}
        pos = self.positions(doc_ids)
        tt = self._legacy_texts(np.clip(pos, 0, max(self.n - 1, 0))) if self.n else []
        for i, p in enumerate(pos):
            if p < 0:
                continue
            d = doc_ids[i].tobytes()
            texts[d] = tt[i][:max_chars]
            srcs[d] = int(self.srcs[int(p)])
        return texts, srcs

    # ---------------- runs2 模式 ----------------

    def _pf(self, idx: int) -> pq.ParquetFile:
        pf = self._pf_cache.get(idx)
        if pf is None:
            if len(self._pf_cache) >= 64:
                self._pf_cache.pop(next(iter(self._pf_cache)))
            pf = self._pf_cache[idx] = pq.ParquetFile(self.run_files[idx])
            self._rg_rows[idx] = pf.metadata.row_group(0).num_rows
        return pf

    def _read_run_rows(
        self, idx: int, row_lo: int, row_hi: int, columns: list[str]
    ) -> dict[str, list[Any]]:
        pf = self._pf(idx)
        rg = self._rg_rows[idx]
        out: dict[str, list[Any]] = {c: [] for c in columns}
        g0, g1 = row_lo // rg, (row_hi - 1) // rg
        for g in range(g0, g1 + 1):
            tbl = pf.read_row_group(g, columns=columns)
            s = max(row_lo - g * rg, 0)
            e = min(row_hi - g * rg, tbl.num_rows)
            for c in columns:
                out[c].extend(tbl.column(c)[s:e].to_pylist())
        return out

    def _bucket_rows(self, ids: np.ndarray, bucket_byte: int, vb: int) -> np.ndarray:
        if vb == 256:
            lo_key = np.void(bytes([bucket_byte]) + b"\x00" * 15)
            hi_key = np.void(bytes([bucket_byte]) + b"\xff" * 15)
            lo = int(np.searchsorted(ids, lo_key))
            hi = int(np.searchsorted(ids, hi_key, side="right"))
            return np.arange(lo, hi, dtype=np.int64)
        fb = first_bytes(ids).astype(np.int32)
        return np.flatnonzero(fb % vb == bucket_byte).astype(np.int64)

    def _runs_texts_for_bucket(
        self, bucket_byte: int, vb: int, max_chars: int, cap_gb: float
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """取首字节 % vb == bucket_byte 的全部候选文档 (ids, srcs, texts[:max_chars])。"""
        ids_out: list[bytes] = []
        srcs_out: list[int] = []
        texts_out: list[str] = []
        est_bytes = 0
        for i, ids in enumerate(self.run_ids):
            rows = self._bucket_rows(ids, bucket_byte, vb)
            if len(rows) == 0:
                continue
            splits = np.flatnonzero(np.diff(rows) > 1) + 1
            run_srcs: list[int] = []
            run_texts: list[str] = []
            for seg in np.split(rows, splits):
                cols = self._read_run_rows(
                    i, int(seg[0]), int(seg[-1]) + 1, ["source_code", "text"]
                )
                est_bytes += sum(min(len(t), max_chars) + 40 for t in cols["text"])
                check_memory_cap(est_bytes, cap_gb, f"verify bucket byte={bucket_byte:02x}")
                run_srcs.extend(int(s) for s in cols["source_code"])
                run_texts.extend(t[:max_chars] for t in cols["text"])
            ids_out.append(np.ascontiguousarray(ids[rows]).tobytes())
            srcs_out.extend(run_srcs)
            texts_out.extend(run_texts)
        if not ids_out:
            return np.empty(0, dtype="V16"), np.empty(0, dtype="<u2"), []
        id_arr = np.frombuffer(b"".join(ids_out), dtype="V16")
        return id_arr, np.array(srcs_out, dtype="<u2"), texts_out

    def _runs_texts_and_sources_for_ids(
        self, doc_ids: np.ndarray, max_chars: int, cap_gb: float
    ) -> tuple[dict[bytes, str], dict[bytes, int]]:
        """跨 run 点查：逐 run searchsorted 命中后按行组读回（批量接口）。"""
        texts: dict[bytes, str] = {}
        srcs: dict[bytes, int] = {}
        if len(doc_ids) == 0:
            return texts, srcs
        est_bytes = 0
        for i, ids in enumerate(self.run_ids):
            if len(ids) == 0:
                continue
            pos = np.searchsorted(ids, doc_ids)
            pos = np.clip(pos, 0, len(ids) - 1)
            hit = np.flatnonzero(ids[pos] == doc_ids)
            if len(hit) == 0:
                continue
            rows = np.sort(pos[hit])
            # 合并相邻行为区间，减少行组重复读
            splits = np.flatnonzero(np.diff(rows) > 1) + 1
            for seg in np.split(rows, splits):
                cols = self._read_run_rows(i, int(seg[0]), int(seg[-1]) + 1, ["source_code", "text"])
                est_bytes += sum(min(len(t), max_chars) + 40 for t in cols["text"])
                check_memory_cap(est_bytes, cap_gb, "verify partner texts")
                for off, r in enumerate(range(int(seg[0]), int(seg[-1]) + 1)):
                    d = ids[r].tobytes()
                    texts[d] = cols["text"][off][:max_chars]
                    srcs[d] = int(cols["source_code"][off])
        return texts, srcs

    def sources(self, doc_ids: np.ndarray) -> np.ndarray:
        if self.legacy:
            pos = self.positions(doc_ids)
            out = np.zeros(len(doc_ids), dtype="<u2")
            ok = pos >= 0
            out[ok] = self.srcs[pos[ok]]
            return out
        out = np.zeros(len(doc_ids), dtype="<u2")
        if len(doc_ids) == 0:
            return out
        for i, ids in enumerate(self.run_ids):
            if len(ids) == 0:
                continue
            pos = np.searchsorted(ids, doc_ids)
            pos = np.clip(pos, 0, len(ids) - 1)
            hit = np.flatnonzero(ids[pos] == doc_ids)
            if len(hit) == 0:
                continue
            row_to_q = {int(pos[q]): int(q) for q in hit}
            rows = np.array(sorted(row_to_q), dtype=np.int64)
            splits = np.flatnonzero(np.diff(rows) > 1) + 1
            for seg in np.split(rows, splits):
                cols = self._read_run_rows(i, int(seg[0]), int(seg[-1]) + 1, ["source_code"])
                for off, r in enumerate(range(int(seg[0]), int(seg[-1]) + 1)):
                    out[row_to_q[r]] = int(cols["source_code"][off])
        return out


# ---------------------------------------------------------------- ctext（按 doc_id 首字节重分区的候选正文）
# verify 的专用布局：docstore runs 每 run 只读一次，行按首字节落 s3_ctext/<domain>/<byte>/part-*.parquet
# （part 内按 doc_id 排序），另建每字节 id 索引（ids 排序 + loc=part<<32|row）。
# verify 桶 k 读本桶 shard 全量（无行组放大）；跨桶伙伴点查 = 按首字节分组后 per-byte
# searchsorted，O(|Q| log n) 总量。正文在构建时截断 verify_text_chars（读时截断 runs 不动）。

CTEXT_SUMMARY = "ctext.summary.json"


def _ctext_part_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("a0", pa.uint64(), nullable=False),
            pa.field("a1", pa.uint64(), nullable=False),
            pa.field("source_code", pa.uint16(), nullable=False),
            pa.field("text", pa.string(), nullable=False),
        ]
    )


def write_ctext_part(path: Path, ids: np.ndarray, srcs: np.ndarray, texts: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    d0, d1 = void_to_stack_u8(np.ascontiguousarray(ids), 2)
    table = pa.table(
        {"a0": d0, "a1": d1, "source_code": srcs, "text": texts},
        schema=_ctext_part_schema(),
    )
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)
    return len(ids)


def _ctext_run_task(args: tuple[Path, Path, int, int]) -> int:
    run_path, out_dir, run_idx, max_chars = args
    table = pq.read_table(run_path, columns=["doc_id", "source_code", "text"])
    ids = _hex_strings_to_void(table.column("doc_id").to_numpy())
    srcs = table.column("source_code").to_numpy()
    texts = table.column("text").to_pylist()
    fb = first_bytes(ids).astype(np.int32)
    n_out = 0
    for b in range(256):
        rows = np.flatnonzero(fb == b)
        if len(rows) == 0:
            continue
        part = out_dir / f"{b:02x}" / f"part-{run_idx:05d}.parquet"
        write_ctext_part(
            part,
            np.ascontiguousarray(ids[rows]),
            np.ascontiguousarray(srcs[rows]),
            [texts[int(r)][:max_chars] for r in rows],
        )
        n_out += len(rows)
    return n_out


def _ctext_index_task(args: tuple[Path, Path, int]) -> int:
    # 从 run sidecar（每 run 已按 doc_id 排序）直接取 [首字节=b] 区间：
    # memmap searchsorted 只读 ~6KB/区间，避免读 3M 个小 part 文件。
    # loc = run_idx<<32 | row_within_part（part 内行号 = row_in_run - lo）。
    store_dir, out_dir, byte = args
    sidecars = sorted((store_dir / "_runs").glob("*.ids.bin"))
    ids_chunks: list[np.ndarray] = []
    loc_chunks: list[np.ndarray] = []
    lo_key = np.void(bytes([byte]) + b"\x00" * 15)
    hi_key = np.void(bytes([byte]) + b"\xff" * 15)
    for run_idx, sc in enumerate(sidecars):
        ids = np.memmap(sc, dtype="V16", mode="r")
        lo = int(np.searchsorted(ids, lo_key))
        hi = int(np.searchsorted(ids, hi_key, side="right"))
        if lo >= hi:
            continue
        ids_chunks.append(np.ascontiguousarray(ids[lo:hi]))
        loc_chunks.append(
            np.full(hi - lo, run_idx << 32, dtype=np.uint64)
            | np.arange(hi - lo, dtype=np.uint64)
        )
    if not ids_chunks:
        return 0
    ids = np.concatenate(ids_chunks)
    loc = np.concatenate(loc_chunks)
    order = np.argsort(ids, kind="stable")
    ids, loc = ids[order], loc[order]
    idx_dir = out_dir / "_idx"
    idx_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in ((f"{byte:02x}.ids.bin", ids), (f"{byte:02x}.loc.bin", loc)):
        tmp = idx_dir / (name + f".tmp-{os.getpid()}")
        tmp.write_bytes(np.ascontiguousarray(arr).tobytes())
        os.replace(tmp, idx_dir / name)
    return len(ids)


def build_ctext_index(store_dir: Path, out_dir: Path, workers: int) -> int:
    t0 = time.monotonic()
    total = 0
    with mp.Pool(processes=max(1, min(workers, 32))) as pool:
        for n in pool.imap_unordered(
            _ctext_index_task, [(store_dir, out_dir, b) for b in range(256)]
        ):
            total += n
    log.info("ctext index built: %d docs in %.0fs", total, time.monotonic() - t0)
    return total


def build_ctext(store_dir: Path, out_dir: Path, max_chars: int, workers: int) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / CTEXT_SUMMARY
    if summary_path.is_file():
        payload = json.loads(summary_path.read_text("utf-8"))
        if payload.get("max_chars") == max_chars:
            return payload
    run_files = sorted((store_dir / "_runs").glob("run-*.parquet"))
    if not run_files:
        payload = {"format": "ctext", "docs": 0, "runs": 0, "max_chars": max_chars}
        write_summary(summary_path, payload)
        return payload
    tasks = [(rf, out_dir, i, max_chars) for i, rf in enumerate(run_files)]
    n_docs = 0
    t0 = time.monotonic()
    with mp.Pool(processes=max(1, workers)) as pool:
        for i, n in enumerate(pool.imap_unordered(_ctext_run_task, tasks, chunksize=8)):
            n_docs += n
            if (i + 1) % 200 == 0:
                log.info("ctext repartition: %d/%d runs, %.0fs", i + 1, len(tasks), time.monotonic() - t0)
    n_indexed = build_ctext_index(store_dir, out_dir, workers)
    payload = {
        "format": "ctext",
        "docs": n_docs,
        "runs": len(run_files),
        "max_chars": max_chars,
        "indexed_docs": n_indexed,
        "elapsed_sec": round(time.monotonic() - t0, 3),
    }
    write_summary(summary_path, payload)
    return payload


class CText:
    """ctext 读取器：verify 桶文本全量 + 跨桶伙伴点查（per-byte 排序索引）。"""

    def __init__(self, base: Path):
        self.base = base
        summary = json.loads((base / CTEXT_SUMMARY).read_text("utf-8"))
        self.n = int(summary["docs"])
        self._pf_cache: dict[Path, pq.ParquetFile] = {}

    def _parts(self, byte: int) -> list[Path]:
        d = self.base / f"{byte:02x}"
        return sorted(d.glob("part-*.parquet")) if d.is_dir() else []

    def texts_for_bucket(
        self, byte: int, vb: int, cap_gb: float
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        # ctext part 按完整首字节（256 目录）落盘；vb<256 时需取所有 b%vb==byte 的目录
        dirs = (
            [self.base / f"{byte:02x}"]
            if vb == 256
            else [
                p
                for p in sorted(self.base.iterdir())
                if p.is_dir() and p.name != "_idx" and int(p.name, 16) % vb == byte
            ]
        )
        ids_out: list[np.ndarray] = []
        srcs_out: list[np.ndarray] = []
        texts_out: list[str] = []
        est = 0
        for d in dirs:
            for part in sorted(d.glob("part-*.parquet")):
                table = pq.read_table(part)
                if table.num_rows == 0:
                    continue
                arr = np.column_stack(
                    [table.column("a0").to_numpy(), table.column("a1").to_numpy()]
                )
                ids_out.append(stack_u8_to_void([arr[:, 0], arr[:, 1]], 16))
                srcs_out.append(table.column("source_code").to_numpy())
                texts_out.extend(table.column("text").to_pylist())
                est += sum(len(t) + 40 for t in texts_out[-table.num_rows :])
                check_memory_cap(est, cap_gb, f"ctext bucket byte={byte:02x}")
        if not ids_out:
            return np.empty(0, dtype="V16"), np.empty(0, dtype="<u2"), []
        return np.concatenate(ids_out), np.concatenate(srcs_out), texts_out

    def _byte_index(self, byte: int) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        ids_path = self.base / "_idx" / f"{byte:02x}.ids.bin"
        if not ids_path.is_file():
            return None, None
        loc_path = self.base / "_idx" / f"{byte:02x}.loc.bin"
        return (
            np.memmap(ids_path, dtype="V16", mode="r"),
            np.memmap(loc_path, dtype="<u8", mode="r"),
        )

    def texts_and_sources_for_ids(
        self, doc_ids: np.ndarray, max_chars: int, cap_gb: float
    ) -> tuple[dict[bytes, str], dict[bytes, int]]:
        texts: dict[bytes, str] = {}
        srcs: dict[bytes, int] = {}
        if len(doc_ids) == 0:
            return texts, srcs
        fb = first_bytes(doc_ids).astype(np.int32)
        est = 0
        for byte in np.unique(fb):
            idx_ids, loc = self._byte_index(int(byte))
            if idx_ids is None:
                continue
            mask = fb == byte
            qb = doc_ids[mask]
            qs = np.flatnonzero(mask)
            pos = np.searchsorted(idx_ids, qb)
            pos = np.clip(pos, 0, len(idx_ids) - 1)
            hit = np.flatnonzero(idx_ids[pos] == qb)
            if len(hit) == 0:
                continue
            cache: dict[int, pa.Table] = {}
            for h in hit:
                run_idx, row = int(loc[pos[h]] >> 32), int(loc[pos[h]] & 0xFFFFFFFF)
                table = cache.get(run_idx)
                if table is None:
                    if len(cache) >= 32:
                        cache.clear()
                    table = cache[run_idx] = pq.read_table(
                        self.base / f"{int(byte):02x}" / f"part-{run_idx:05d}.parquet"
                    )
                d = qb[h].tobytes()
                texts[d] = table.column("text")[row].as_py()
                srcs[d] = int(table.column("source_code")[row].as_py())
                est += len(texts[d]) + 40
                check_memory_cap(est, cap_gb, "ctext partner texts")
        return texts, srcs

    def sources(self, doc_ids: np.ndarray) -> np.ndarray:
        out = np.zeros(len(doc_ids), dtype="<u2")
        if len(doc_ids) == 0:
            return out
        fb = first_bytes(doc_ids).astype(np.int32)
        for byte in np.unique(fb):
            idx_ids, loc = self._byte_index(int(byte))
            if idx_ids is None:
                continue
            mask = fb == byte
            qb = doc_ids[mask]
            qs = np.flatnonzero(mask)
            pos = np.searchsorted(idx_ids, qb)
            pos = np.clip(pos, 0, len(idx_ids) - 1)
            hit = np.flatnonzero(idx_ids[pos] == qb)
            if len(hit) == 0:
                continue
            cache: dict[int, pa.Table] = {}
            for h in hit:
                run_idx, row = int(loc[pos[h]] >> 32), int(loc[pos[h]] & 0xFFFFFFFF)
                table = cache.get(run_idx)
                if table is None:
                    if len(cache) >= 32:
                        cache.clear()
                    table = cache[run_idx] = pq.read_table(
                        self.base / f"{int(byte):02x}" / f"part-{run_idx:05d}.parquet",
                        columns=["source_code"],
                    )
                out[qs[h]] = int(table.column("source_code")[row].as_py())
        return out


# ---------------------------------------------------------------- csig（按 doc_id 首字节重分区的签名 + 源）
# 8.4B 候选对下逐对取正文重算 Jaccard 不可行（TB 级散点 IO + ~30-50h 计算）。
# 判定改用 128 排列 MinHash 签名一致率（J_est = mean(sig_a == sig_b)，无偏估计 J，
# σ≈0.036@0.8，阈值 0.8 为软阈值——Dolma/text-dedup 同规模标准做法）。
# 签名按 doc_id 首字节重分区为 flat bin（sigs.bin n×512B + srcs.bin n×2B + 排序 ids），
# 伙伴签名 = memmap 精确 512B 读，无小文件放大；桶内全量顺序读。
# 每 bin 内按 doc_id 排序，与 ctext _idx 同一稳定排序 ⇒ 跨结构 id 顺序一致。

CSIG_SUMMARY = "csig.summary.json"


def _csig_byte_task(args: tuple[Path, Path, int]) -> int:
    """s3_sig 已按 doc_id 首字节分目录（<byte>/<source>/<relpath>.parquet）——
    无需重分区：每字节并行读全量 → 按 doc_id 排序 → 写 flat bin（ids + sigs）。"""
    byte_dir, out_dir, byte = args
    ids_chunks: list[np.ndarray] = []
    sig_chunks: list[np.ndarray] = []
    for dirpath, _dirnames, filenames in os.walk(byte_dir):
        for name in filenames:
            if not name.endswith(".parquet"):
                continue
            table = pq.read_table(Path(dirpath) / name, columns=["doc_id", "sig"])
            if table.num_rows == 0:
                continue
            ids_chunks.append(_hex_strings_to_void(table.column("doc_id").to_numpy()))
            sig_chunks.append(
                np.frombuffer(b"".join(table.column("sig").to_pylist()), dtype=np.uint8).reshape(-1, 512)
            )
    if not ids_chunks:
        return 0
    ids = np.concatenate(ids_chunks)
    sigs = np.concatenate(sig_chunks)
    check_memory_cap(len(ids) * 528 + (1 << 28), 12.0, f"csig byte={byte:02x}")
    order = np.argsort(ids, kind="stable")
    ids, sigs = ids[order], sigs[order]
    idx_dir = out_dir / "_idx"
    idx_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in ((f"{byte:02x}.ids.bin", ids), (f"{byte:02x}.sig.bin", sigs)):
        tmp = idx_dir / (name + f".tmp-{os.getpid()}")
        tmp.write_bytes(np.ascontiguousarray(arr).tobytes())
        os.replace(tmp, idx_dir / name)
    return len(ids)


def build_csig(sig_root: Path, out_dir: Path, workers: int) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / CSIG_SUMMARY
    if summary_path.is_file():
        return json.loads(summary_path.read_text("utf-8"))
    t0 = time.monotonic()
    # 2026-09-22: 容缺——某领域在 S0/S1 历史故障中产物为空（如 math/academic），
    # 该领域整体缺席 S3，csig 应视为空集而非崩溃（崩溃会让整个 run_s3 陪葬）。
    if not sig_root.is_dir():
        payload = {"docs": 0, "format": "csig", "elapsed_sec": 0.0, "empty": True}
        write_summary(summary_path, payload)
        return payload
    byte_dirs = [p for p in sorted(sig_root.iterdir()) if p.is_dir() and len(p.name) == 2]
    n_docs = 0
    with mp.Pool(processes=max(1, min(workers, 96))) as pool:
        for n in pool.imap_unordered(
            _csig_byte_task, [(d, out_dir, int(d.name, 16)) for d in byte_dirs]
        ):
            n_docs += n
    log.info("csig built: %d docs in %.0fs", n_docs, time.monotonic() - t0)
    payload = {
        "format": "csig",
        "docs": n_docs,
        "elapsed_sec": round(time.monotonic() - t0, 3),
    }
    write_summary(summary_path, payload)
    return payload


class CSig:
    """按 doc_id 首字节的签名 bin 读取器（verify 判定 + decide 源查询）。"""

    def __init__(self, base: Path):
        self.base = base
        summary = json.loads((base / CSIG_SUMMARY).read_text("utf-8"))
        self.n = int(summary["docs"])
        self._bins: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def _bin(self, byte: int) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        if byte in self._bins:
            return self._bins[byte]
        # b 侧逐组加载用后即弃（256 bin × 2.4GB 全缓存会撑爆内存），只保留 a 侧桶
        if len(self._bins) >= 4:
            for k in [k for k in self._bins if k != getattr(self, "_own_byte", -1)]:
                del self._bins[k]
                break
        idx_path = self.base / "_idx" / f"{byte:02x}.ids.bin"
        if not idx_path.is_file():
            return None, None
        ids = np.memmap(idx_path, dtype="V16", mode="r")
        sigs = np.memmap(self.base / "_idx" / f"{byte:02x}.sig.bin", dtype=np.uint8, mode="r").reshape(-1, 512)
        self._bins[byte] = (ids, sigs)
        return self._bins[byte]

    def pin(self, byte: int) -> None:
        """标记 a 侧桶为常驻（verify 桶内重复用于全部对）。"""
        self._own_byte = byte

    def sigs_at(self, byte: int, pos: np.ndarray) -> np.ndarray:
        _, sigs = self._bin(byte)
        return np.ascontiguousarray(sigs[pos]).view(np.uint32).reshape(len(pos), 128)

    def locate(self, byte: int, doc_ids: np.ndarray) -> np.ndarray:
        """doc_ids（均属该首字节）在 bin 内的位置；未命中 = -1。"""
        ids, _ = self._bin(byte)
        if ids is None:
            return np.full(len(doc_ids), -1, dtype=np.int64)
        pos = np.searchsorted(ids, doc_ids)
        pos = np.clip(pos, 0, len(ids) - 1)
        hit = ids[pos] == doc_ids
        out = np.full(len(doc_ids), -1, dtype=np.int64)
        out[hit] = pos[hit]
        return out

# ---------------------------------------------------------------- 任务执行框架

_Ctx = dict[str, Any]


def run_tasks(
    tasks: list[tuple],
    ctx: _Ctx,
    worker_fn: Callable[[tuple, _Ctx], dict[str, Any]],
    workers: int,
    stage: str,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    started = time.monotonic()
    done = 0
    counters: Counter[str] = Counter()
    with mp.Pool(processes=workers) as pool:
        it = pool.imap_unordered(worker_fn, [(t, ctx) for t in tasks], chunksize=1)
        while True:
            try:
                summary = next(it)
            except StopIteration:
                break
            except Exception as exc:
                log.error("pool-level shard failure: %s", exc)
                summaries.append({"shard_fatal": f"pool:{type(exc).__name__}: {exc}"})
                continue
            summaries.append(summary)
            done += 1
            for key, value in (summary.get("counters") or {}).items():
                counters[key] += int(value)
            if done % PROGRESS_EVERY == 0 or done == len(tasks):
                elapsed = time.monotonic() - started
                log.info(
                    "progress[%s]: %d/%d tasks, %.2f tasks/s, counters: %s",
                    stage, done, len(tasks), done / max(elapsed, 1e-9), dict(counters),
                )
    return summaries


NUMERIC_SUMMARY_KEYS = (
    "in_rows",
    "out_rows",
    "audit_rows",
    "error_rows",
    "input_bytes",
    "output_bytes",
    "text_bytes",
    "sig_rows",
    "lsh_rows",
    "pair_rows",
    "candidate_docs",
    "verified_pairs",
    "verified_passed",
    "groups",
    "clusters",
    "dropped",
    "kept",
    "truncated_docs",
    "short_docs",
)


def write_stage_summary(
    out_root: Path, stage: str, summaries: list[dict[str, Any]]
) -> Path:
    per_source: dict[str, Counter[str]] = {}
    per_domain: dict[str, Counter[str]] = {}
    decision_totals: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    for s in summaries:
        src = s.get("source") or s.get("domain") or "?"
        dom = s.get("domain", "?")
        acc = per_source.setdefault(src, Counter())
        for key in NUMERIC_SUMMARY_KEYS:
            if key in s:
                acc[key] += int(s.get(key) or 0)
                totals[key] += int(s.get(key) or 0)
        acc["elapsed_sec"] += float(s.get("elapsed_sec") or 0.0)
        for key, v in (s.get("decisions") or {}).items():
            decision_totals[key] += int(v)
            totals[f"decisions.{key}"] += int(v)
        dacc = per_domain.setdefault(dom, Counter())
        for key in NUMERIC_SUMMARY_KEYS:
            if key in s:
                dacc[key] += int(s.get(key) or 0)
    payload = {
        "stage": stage,
        "tasks": len(summaries),
        "task_errors": sum(1 for s in summaries if s.get("errors") or s.get("shard_fatal")),
        "totals": dict(totals),
        "decisions": dict(decision_totals),
        "per_source": {k: dict(v) for k, v in sorted(per_source.items())},
        "per_domain": {k: dict(v) for k, v in sorted(per_domain.items())},
    }
    # All stage consumers use one canonical failure field. Keep task_errors as
    # a compatibility alias for old reports.
    payload["shard_errors"] = payload["task_errors"]
    path = out_root / f"stage_{stage}_global_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1), "utf-8")
    return path


def empty_table(schema: pa.Schema) -> pa.Table:
    return pa.table(
        {f.name: pa.array([], type=f.type) for f in schema}, schema=schema
    )


def fatal_summary(stage: str, task_id: str, exc: Exception, started: float) -> dict[str, Any]:    return {
        "stage": stage,
        "task": task_id,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "errors": {f"task_error:{type(exc).__name__}": 1},
        "error_message": str(exc)[:500],
        "skipped": False,
    }
