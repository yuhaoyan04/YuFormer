from __future__ import annotations

import hashlib
import itertools
import json
import logging
import multiprocessing as mp
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .dedup_common import fingerprint, normalize_for_hash
from .ingest import safe_iter_batches, write_summary

log = logging.getLogger("yuformer.scale.decontam")

DEFAULT_DECONTAM_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "decontam.yaml"
)

DECISION_KEEP = "keep"
DECISION_DROP = "drop_contaminated"

# ---------------------------------------------------------------- 哈希方案（判定性，两侧同源）
# 词 → uint64 id：FNV-1a（numpy 批量，纯算术、跨进程确定）。
# 词组（n-gram 滑窗）→ uint64：词 id 的多项式 rolling hash（mod 2^64，前缀和法向量化）。
# 不用 blake2b-8：无法向量化，达不到 ≥5 万行/秒/核目标（E-12：全量扫描必须向量化）。
# 64-bit 空间下全池 ~1e12 次 shingle 查询 vs 2^64，假命中期望 <1e-4；
# 决策表落 hit_benchmark 供审计抽样复核。
_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_ROLL_PRIME = 0xFF51AFD7ED558CCD  # 奇数 ⇒ 2^64 环内存在乘法逆元（前缀和滑窗法的前提）
_MOD = 1 << 64

_ROLL_POW_CACHE: dict[int, int] = {}

# 词级缓存（per-process，纯函数缓存：清缓存不改变任何输出，确定性不受影响）。
_WORD_CACHE: dict[str, int] = {}
_WORD_CACHE_CAP = 4_000_000
# Bound each benchmark-index lookup independently of parquet batch size.  Full
# document scanning makes a 4096-row batch potentially contain hundreds of
# millions of shingles (especially books); concatenating that whole batch was
# only safe while the legacy 50k prefix cap existed.
_MAX_QUERY_SHINGLES = 2_000_000


def _fnv1a64_batch(words: list[str]) -> np.ndarray:
    if not words:
        return np.empty(0, dtype="<u8")
    enc = [w.encode("utf-8") for w in words]
    lens = np.fromiter((len(e) for e in enc), dtype=np.int64, count=len(enc))
    maxlen = int(lens.max())
    arr = np.zeros((len(enc), maxlen), dtype=np.uint8)
    for i, e in enumerate(enc):
        arr[i, : len(e)] = np.frombuffer(e, dtype=np.uint8)
    h = np.full(len(enc), _FNV_OFFSET, dtype="<u8")
    prime = np.uint64(_FNV_PRIME)
    for j in range(maxlen):
        m = lens > j
        hv = h[m]
        hv ^= arr[m, j].astype("<u8")
        hv *= prime
        h[m] = hv
    return h


def word_hashes(words: list[str]) -> np.ndarray:
    # E-22：原实现先算 missing 再 clear 缓存——clear 删掉的老词不在 missing 里，
    # 随后 __getitem__ 必 KeyError（高频词 'the' 首当其冲）。修复：判定"填充后超容"
    # 则 clear 并把 missing 扩为全批词，保证查询必命中。
    missing = {w for w in words if w not in _WORD_CACHE}
    if len(_WORD_CACHE) + len(missing) > _WORD_CACHE_CAP:
        _WORD_CACHE.clear()
        missing = set(words)
    if missing:
        uniq = sorted(missing)
        hv = _fnv1a64_batch(uniq)
        for w, v in zip(uniq, hv.tolist()):
            _WORD_CACHE[w] = v
    # map(C 层循环) 快于生成器表达式；词缓存命中时零 FNV 计算
    return np.fromiter(map(_WORD_CACHE.__getitem__, words), dtype="<u8", count=len(words))


def shingle_hashes_u64(word_h: np.ndarray, n: int) -> np.ndarray:
    """词 id 序列 → n-gram rolling hash（uint64）。少于 n 个词返回空数组。"""
    m = len(word_h)
    if m < n:
        return np.empty(0, dtype="<u8")
    # ``np.convolve`` performs the fixed-width polynomial in compiled code and
    # avoids four full-length prefix/power temporaries.  With coefficients in
    # ascending order, NumPy's convolution reversal yields
    # sum(h[k+j] * P**(n-1-j)), exactly the historical rolling-hash value.
    powers = np.asarray(
        [pow(_ROLL_PRIME, j, _MOD) for j in range(n)], dtype="<u8"
    )
    return np.convolve(word_h, powers, mode="valid").astype("<u8", copy=False)


_CALC_RE = re.compile(r"<<[^>]*>>")
# 中文（CJK）文本无空格分词：整句 = 1 个"词" ⇒ 8-gram 永远构不出来（C-Eval 会整体漏检）。
# 判定：CJK 表意文字逐字成 token，拉丁/数字等非空白连续段成词——中英文同一管线，
# 确定性，两侧（索引/扫描）同一 tokenizer。
_CJK_RUN = re.compile(
    r"[^\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002a6df\s]+"
    r"|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002a6df]"
)
_CJK_ANY = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002a6df]")


def _tokenize(norm_text: str, max_tokens: int | None = None) -> list[str]:
    if not _CJK_ANY.search(norm_text):
        if not norm_text:
            return []
        if max_tokens is None:
            return norm_text.split(" ")
        # maxsplit produces one extra tail item; slicing preserves exactly the
        # same first tokens without materializing the rest of a long document.
        return norm_text.split(" ", max_tokens)[:max_tokens]
    matches = _CJK_RUN.finditer(norm_text)
    if max_tokens is not None:
        matches = itertools.islice(matches, max_tokens)
    return [match.group(0) for match in matches]


def _fast_normalize(text: str) -> str:
    """与 dedup_common.normalize_for_hash 输出逐字节一致。ASCII 纯文本走等价快速路径：
    NFKC 对 ASCII 是恒等映射、casefold 即 lower()、空白折叠即 split/join——
    NFKC 整表归一是全流水线第一热点（实测 ~60µs/2KB 文档），此路径 ~6µs。"""
    if text.isascii():
        from . import detectors

        return " ".join(text.translate(detectors._DELETE_TABLE).split()).lower()
    return normalize_for_hash(text)


def doc_shingle_hashes(
    text: str, n: int, cap: int | None
) -> tuple[np.ndarray, bool]:
    """单文档 n-gram 哈希流。

    Production uses ``cap=None`` and scans the full S1-bounded document.  A
    finite cap remains available only for controlled experiments and backwards
    compatibility; prefix-only decontamination is not a release-safe policy.
    """
    norm = _fast_normalize(text)
    if not norm:
        return np.empty(0, dtype="<u8"), False
    # We only publish the first ``cap`` shingles.  Read one extra token beyond
    # the required cap+n-1 prefix to preserve the exact truncated flag, then
    # avoid hashing the unused tail (critical for book-length documents).
    token_limit = cap + n if cap is not None else None
    words = _tokenize(norm, max_tokens=token_limit)
    truncated = cap is not None and len(words) >= cap + n
    if truncated:
        words = words[: cap + n - 1]
    sh = shingle_hashes_u64(word_hashes(words), n)
    return sh, truncated


def exceeds_shingle_cap(text: str, n: int, cap: int) -> bool:
    """Whether the legacy prefix scan omitted at least one shingle.

    This intentionally performs normalization/tokenization only; it avoids word
    hashing and benchmark-index lookup for the billions of documents whose old
    prefix already covered the full text.
    """
    # Every tokenizer output consumes at least one input code point. Therefore
    # fewer than cap+n characters cannot possibly produce cap+n tokens. This
    # exact O(1) lower bound avoids normalizing/splitting virtually every short
    # document during a legacy-cap expansion audit.
    if len(text) < cap + n:
        return False
    if text.isascii():
        # Case folding does not affect token boundaries.  Avoid the legacy
        # normalize ``split→join`` followed by a second split, and stop the
        # whitespace split as soon as cap+n tokens are known to exist.
        from . import detectors

        cleaned = text.translate(detectors._DELETE_TABLE)
        return len(cleaned.split(None, cap + n - 1)) >= cap + n
    norm = _fast_normalize(text)
    if not norm:
        return False
    return len(_tokenize(norm, max_tokens=cap + n)) >= cap + n


# ---------------------------------------------------------------- 配置

def load_decontam_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_DECONTAM_CONFIG_PATH
    if not p.is_file():
        raise FileNotFoundError(f"decontam config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"decontam config not a mapping: {p}")
    cfg = dict(raw)
    # Config paths are part of the project layout, not the caller's cwd.  The
    # documented value ``../data/...`` is relative to data-process/.
    index_dir = Path(str(cfg.get("index_dir", "../data/decontam_index")))
    if not index_dir.is_absolute():
        index_dir = (p.parent.parent / index_dir).resolve()
    cfg["index_dir"] = str(index_dir)
    cfg.setdefault("ngram_n", 8)
    cfg.setdefault("max_shingles_per_doc", None)
    cfg.setdefault("monotonic_from_max_shingles_per_doc", None)
    cfg.setdefault("input_stage", "s3")
    cfg.setdefault("index_dir", "../data/decontam_index")
    cfg.setdefault("memory_cap_gb", 4.0)
    cfg.setdefault("require_pinned_revisions", False)
    cfg.setdefault("benchmarks", [])
    if int(cfg["ngram_n"]) < 1:
        raise ValueError("ngram_n must be >= 1")
    if cfg["max_shingles_per_doc"] is not None and int(cfg["max_shingles_per_doc"]) < 1:
        raise ValueError("max_shingles_per_doc must be null or >= 1")
    previous_cap = cfg["monotonic_from_max_shingles_per_doc"]
    if previous_cap is not None:
        if cfg["max_shingles_per_doc"] is not None or int(previous_cap) < 1:
            raise ValueError(
                "monotonic_from_max_shingles_per_doc requires a positive old cap and current null cap"
            )
    if not cfg["benchmarks"]:
        raise ValueError("decontam config: benchmarks list empty")
    for b in cfg["benchmarks"]:
        for key in ("name", "repo", "paths", "fields"):
            if key not in b:
                raise ValueError(f"benchmark entry missing {key}: {b}")
        b.setdefault("min_distinct_hits", 1)
        if int(b["min_distinct_hits"]) < 1:
            raise ValueError(f"benchmark {b['name']}: min_distinct_hits must be >= 1")
        if cfg["require_pinned_revisions"]:
            revision = str(b.get("revision") or "")
            if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower()):
                raise ValueError(
                    f"benchmark {b['name']}: production config requires a 40-hex revision"
                )
    benchmark_names = {str(b["name"]) for b in cfg["benchmarks"]}
    source_overrides = cfg.setdefault("source_min_distinct_hits", {})
    if not isinstance(source_overrides, dict):
        raise ValueError("source_min_distinct_hits must be a mapping")
    for source, overrides in source_overrides.items():
        if not isinstance(overrides, dict):
            raise ValueError(f"source_min_distinct_hits.{source} must be a mapping")
        unknown = set(overrides) - benchmark_names
        if unknown:
            raise ValueError(
                f"source_min_distinct_hits.{source} has unknown benchmarks: {sorted(unknown)}"
            )
        for benchmark, value in overrides.items():
            if int(value) < 1:
                raise ValueError(
                    f"source_min_distinct_hits.{source}.{benchmark} must be >= 1"
                )
    return cfg


def _config_echo(cfg: dict[str, Any]) -> str:
    """Identity of benchmark-index construction inputs only.

    Per-source hit thresholds affect scanning, not index bytes; they belong in
    ``_scan_config_echo`` so a threshold correction does not rebuild/download
    the benchmark corpus.
    """
    payload = {
        "ngram_n": cfg["ngram_n"],
        "benchmarks": cfg["benchmarks"],
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _effective_min_distinct_hits(cfg: dict[str, Any], source: str) -> dict[str, int]:
    """Resolve scan thresholds without weakening unrelated sources.

    CJK boilerplate needs a higher C-Eval threshold for the three Chinese corpora,
    while a global C-Eval threshold would silently reduce decontamination recall for
    English/code sources.  Keep the override in the source contract.
    """
    resolved = {
        str(b["name"]): int(b.get("min_distinct_hits", 1))
        for b in cfg["benchmarks"]
    }
    resolved.update(
        {
            str(name): int(value)
            for name, value in (
                (cfg.get("source_min_distinct_hits") or {}).get(source) or {}
            ).items()
        }
    )
    return resolved


def effective_min_distinct_hits(cfg: dict[str, Any], source: str) -> dict[str, int]:
    """Public audit-facing view of the production per-source hit policy."""
    return _effective_min_distinct_hits(cfg, source)


def _scan_config_echo(cfg: dict[str, Any], source: str) -> str:
    """Per-source scan identity so a source override invalidates only that source."""
    payload = {
        "ngram_n": cfg["ngram_n"],
        "max_shingles_per_doc": cfg["max_shingles_per_doc"],
        "benchmarks": [
            {
                "name": b["name"],
                "repo": b["repo"],
                "revision": b.get("revision"),
                "paths": b["paths"],
                "fields": b["fields"],
                "clean": b.get("clean", []),
            }
            for b in cfg["benchmarks"]
        ],
        "effective_min_distinct_hits": _effective_min_distinct_hits(cfg, source),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _scan_policy(cfg: dict[str, Any]) -> dict[str, Any]:
    cap = cfg.get("max_shingles_per_doc")
    old_cap = cfg.get("monotonic_from_max_shingles_per_doc")
    policy: dict[str, Any] = {
        "schema_version": 1,
        "scope": "full_document" if cap is None else "prefix",
        "max_shingles_per_doc": cap,
    }
    if cap is None and old_cap is not None:
        policy["transition"] = {
            "from_scope": "prefix",
            "from_max_shingles_per_doc": int(old_cap),
            "monotonic_effect": "only_add_drops",
            "proof": "full-document shingle set is a superset of the former prefix",
        }
    return policy


# ---------------------------------------------------------------- 基准语料获取与文本抽取

def _resolve_revision(repo_id: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi(endpoint=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    return api.dataset_info(repo_id).sha


def _fetch_benchmark_file(repo_id: str, revision: str, relpath: str, cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            filename=relpath,
            local_dir=str(cache_dir),
        )
    )


def _item_text(row: dict[str, Any], fields: list[Any], clean: list[str]) -> str:
    pieces: list[str] = []
    for spec in fields:
        if isinstance(spec, str):
            name, join = spec, False
        else:
            name, join = spec["name"], bool(spec.get("join"))
        val = row.get(name)
        if val is None:
            continue
        if join and isinstance(val, (list, tuple)):
            pieces.extend(str(v) for v in val if isinstance(v, str) and v)
        elif isinstance(val, str) and val:
            pieces.append(val)
    text = "\n".join(pieces)
    if "strip_calculator_annotations" in clean:
        text = _CALC_RE.sub(" ", text)
    return text


def iter_benchmark_texts(
    bench: dict[str, Any], cache_dir: Path, revision: str
) -> Iterator[tuple[str, str]]:
    """yield (path, item_text)。题面+选项+答案全纳入（fields 配置声明）。"""
    for relpath in bench["paths"]:
        local = _fetch_benchmark_file(bench["repo"], revision, relpath, cache_dir)
        pf = pq.ParquetFile(local)
        for batch in pf.iter_batches(batch_size=4096):
            for row in batch.to_pylist():
                text = _item_text(row, bench["fields"], bench.get("clean", []))
                if text.strip():
                    yield relpath, text


# ---------------------------------------------------------------- 索引构建

DECONTAM_INDEX_HASHES = "index.hashes.bin"  # <u8 排序去重（memmap + searchsorted 判定）
DECONTAM_INDEX_CODES = "index.codes.bin"  # uint8 基准码（与 hashes 对齐，配置顺序 = 优先级）


def _index_content_sha256(index_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in (DECONTAM_INDEX_HASHES, DECONTAM_INDEX_CODES):
        path = index_dir / name
        if not path.is_file():
            return "missing"
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


class BenchmarkIndex:
    """基准 8-gram 索引（只读，memmap 共享 page cache——160 worker 同文件不重复占内存）。"""

    def __init__(self, index_dir: Path):
        self.dir = index_dir
        meta = json.loads((index_dir / "index.meta.json").read_text(encoding="utf-8"))
        self.config_sha256 = str(meta.get("config_sha256") or "")
        self.index_sha256 = str(
            meta.get("index_sha256")
            or _index_content_sha256(index_dir)
        )
        self.names: list[str] = meta["names"]
        self.n = int(meta["n_unique_shingles"])
        self.hashes = np.memmap(index_dir / DECONTAM_INDEX_HASHES, dtype="<u8", mode="r")
        self.codes = np.memmap(index_dir / DECONTAM_INDEX_CODES, dtype=np.uint8, mode="r")

    def hit_codes(self, sh: np.ndarray) -> np.ndarray:
        """命中的基准码（去重排序；空 sh 或空索引返回空）。"""
        codes = self.hit_codes_batch(sh)
        return np.unique(codes[codes >= 0])

    def hit_codes_batch(self, sh: np.ndarray) -> np.ndarray:
        """对齐输入的命中码查询：返回 int16（-1=未命中）——整批一次 searchsorted，
        避免逐文档小数组查询的 numpy 调用开销（吞吐关键路径）。"""
        out = np.full(len(sh), -1, dtype=np.int16)
        if len(sh) == 0 or self.n == 0:
            return out
        pos = np.searchsorted(self.hashes, sh)
        pos = np.clip(pos, 0, self.n - 1)
        hit = self.hashes[pos] == sh
        out[hit] = self.codes[pos[hit]]
        return out

    def hit_benchmarks(self, sh: np.ndarray) -> list[str]:
        return [self.names[int(c)] for c in self.hit_codes(sh)]


def write_index(
    cfg: dict[str, Any],
    index_dir: Path,
    bench_texts: dict[str, list[str]],
    per_bench_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """由 {基准名: [文本,...]} 物化索引（排序去重 bin + meta）。测试注入假数据走此路径。"""
    index_dir.mkdir(parents=True, exist_ok=True)
    names = [b["name"] for b in cfg["benchmarks"]]
    n = int(cfg["ngram_n"])
    hashes_chunks: list[np.ndarray] = []
    codes_chunks: list[np.ndarray] = []
    per_bench: dict[str, Any] = {}
    for code, name in enumerate(names):
        texts = bench_texts.get(name, [])
        uniq: set[int] = set()
        items = 0
        skipped_items = 0
        for text in texts:
            sh, _ = doc_shingle_hashes(text, n, cap=None)
            if len(sh) == 0:
                skipped_items += 1
                continue
            items += 1
            uniq.update(sh.tolist())
        arr = np.fromiter(uniq, dtype="<u8", count=len(uniq))
        hashes_chunks.append(arr)
        codes_chunks.append(np.full(len(arr), code, dtype=np.uint8))
        per_bench[name] = {"items": items, "unique_shingles": len(uniq), "skipped_items": skipped_items}
    if hashes_chunks:
        hashes = np.concatenate(hashes_chunks)
        codes = np.concatenate(codes_chunks)
    else:
        hashes = np.empty(0, dtype="<u8")
        codes = np.empty(0, dtype=np.uint8)
    # 同一 shingle 出现在多个基准：按 (hash 主, code 次) 排序后取每组首个
    # （低 code = 配置顺序优先）；lexsort 最后一键为主键
    order = np.lexsort((codes, hashes))
    hashes, codes = hashes[order], codes[order]
    keep = np.ones(len(hashes), dtype=bool)
    if len(hashes) > 1:
        keep[1:] = hashes[1:] != hashes[:-1]
    hashes, codes = hashes[keep], codes[keep]

    index_hash = hashlib.sha256()
    index_hash.update(memoryview(np.ascontiguousarray(hashes)))
    index_hash.update(memoryview(np.ascontiguousarray(codes)))

    tmp_h = index_dir / (DECONTAM_INDEX_HASHES + f".tmp-{os.getpid()}")
    tmp_c = index_dir / (DECONTAM_INDEX_CODES + f".tmp-{os.getpid()}")
    tmp_h.write_bytes(np.ascontiguousarray(hashes).tobytes())
    tmp_c.write_bytes(np.ascontiguousarray(codes).tobytes())
    os.replace(tmp_h, index_dir / DECONTAM_INDEX_HASHES)
    os.replace(tmp_c, index_dir / DECONTAM_INDEX_CODES)

    meta = {
        "names": names,
        "n_unique_shingles": int(len(hashes)),
        "ngram_n": n,
        "config_sha256": _config_echo(cfg),
        "index_sha256": index_hash.hexdigest(),
        "per_benchmark": per_bench,
        "sources": per_bench_meta or {},
    }
    write_summary(index_dir / "index.meta.json", meta)
    payload = {
        "stage": "decontam_index",
        "config_sha256": meta["config_sha256"],
        "index_sha256": meta["index_sha256"],
        "n_unique_shingles": int(len(hashes)),
        "per_benchmark": per_bench,
        "elapsed_sec": 0.0,
        "skipped": False,
    }
    write_summary(index_dir / "index.summary.json", payload)
    return payload


def build_benchmark_index(cfg: dict[str, Any], index_dir: Path) -> dict[str, Any]:
    """下载/读取 5 基准测试集 → 8-gram 索引。幂等：config echo 一致且产物齐在即 skip。"""
    index_dir = Path(index_dir)
    summary_path = index_dir / "index.summary.json"
    if (
        summary_path.is_file()
        and (index_dir / DECONTAM_INDEX_HASHES).is_file()
        and (index_dir / DECONTAM_INDEX_CODES).is_file()
        and (index_dir / "index.meta.json").is_file()
    ):
        prev = json.loads(summary_path.read_text(encoding="utf-8"))
        if prev.get("config_sha256") == _config_echo(cfg) and not prev.get("errors"):
            # Older valid indexes did not publish the content digest. Upgrade
            # metadata in place so every scan summary can bind exact bytes.
            meta_path = index_dir / "index.meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if not meta.get("index_sha256"):
                meta["index_sha256"] = _index_content_sha256(index_dir)
                write_summary(meta_path, meta)
            prev["index_sha256"] = meta["index_sha256"]
            prev["skipped"] = True
            return prev

    started = time.monotonic()
    cache_dir = index_dir / "repo_cache"
    bench_texts: dict[str, list[str]] = {}
    per_bench_meta: dict[str, Any] = {}
    for bench in cfg["benchmarks"]:
        name = bench["name"]
        revision = bench.get("revision") or _resolve_revision(bench["repo"])
        per_path: list[dict[str, Any]] = []
        texts: list[str] = []
        for relpath in bench["paths"]:
            try:
                n_before = len(texts)
                for _path, text in iter_benchmark_texts(
                    {**bench, "paths": [relpath]}, cache_dir, revision
                ):
                    texts.append(text)
                per_path.append(
                    {"path": relpath, "status": "ok", "items": len(texts) - n_before}
                )
            except Exception as exc:
                # 单路径失败不拖垮整个索引：记录失败、其余路径继续（下次运行自动重试补齐）
                per_path.append(
                    {
                        "path": relpath,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                    }
                )
                log.warning("benchmark %s path %s failed: %s", name, relpath, exc)
        bench_texts[name] = texts
        per_bench_meta[name] = {
            "repo": bench["repo"],
            "revision": revision,
            "paths": per_path,
        }
        log.info(
            "benchmark %s: %d items from %d/%d paths (repo=%s @ %s)",
            name, len(texts), sum(1 for p in per_path if p["status"] == "ok"),
            len(per_path), bench["repo"], revision[:8],
        )
    payload = write_index(cfg, index_dir, bench_texts, per_bench_meta)
    payload["elapsed_sec"] = round(time.monotonic() - started, 3)
    write_summary(summary_path, payload)
    n_failed = sum(
        1
        for m in per_bench_meta.values()
        for p in m["paths"]
        if p["status"] != "ok"
    )
    if n_failed:
        payload["errors"] = {f"benchmark_paths_failed": n_failed}
    return payload


# ---------------------------------------------------------------- 分片扫描（决策表）

DECONTAM_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("decision", pa.string(), nullable=False),
        pa.field("hit_benchmark", pa.string(), nullable=False),
    ]
)


def decontam_output_paths(out_root: Path, source: str, relpath: str) -> tuple[Path, Path]:
    base = out_root / "decontam" / source / relpath
    return base.with_name(base.name + ".parquet"), base.with_name(base.name + ".summary.json")


def current_index_sha256(cfg: dict[str, Any]) -> str:
    meta_path = Path(str(cfg["index_dir"])) / "index.meta.json"
    if not meta_path.is_file():
        return "missing"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return str(meta.get("index_sha256") or _index_content_sha256(meta_path.parent))
    except Exception:
        return "invalid"


def validate_published_summary(
    in_path: Path,
    out_path: Path,
    summary: dict[str, Any],
    cfg: dict[str, Any],
    source: str,
) -> str | None:
    """Return a fail-closed reason when a decontam shard is stale or incomplete."""
    if summary.get("errors") or summary.get("shard_fatal"):
        return "published summary contains errors"
    if not in_path.is_file() or not out_path.is_file():
        return "input or output parquet missing"
    current_input_fp = fingerprint(in_path)
    if summary.get("input_fingerprint") != current_input_fp:
        # ndv2_refresh is a controlled in-place update of s3_decision only.
        # Decontam consumes doc_id/text, so a v2 transform contract can carry an
        # older exact input fingerprint across that semantic-preserving rewrite.
        suffix = ".parquet"
        s3_summary = in_path.with_name(
            in_path.name[: -len(suffix)] + ".summary.json"
        )
        try:
            s3_meta = json.loads(s3_summary.read_text(encoding="utf-8"))
            transform = (s3_meta.get("ndv2_refresh") or {}).get("transform_contract") or {}
        except Exception:
            transform = {}
        if not (
            transform.get("method") == "stream_replace_s3_decision_v2"
            and transform.get("input_fingerprint") == summary.get("input_fingerprint")
            and transform.get("output_fingerprint") == current_input_fp
            and {"doc_id", "text"}.issubset(set(transform.get("preserved_columns") or []))
        ):
            return "input fingerprint is missing or stale"
    if summary.get("config_sha256") != _scan_config_echo(cfg, source):
        return "per-source decontam config is missing or stale"
    expected_index = current_index_sha256(cfg)
    if expected_index in ("missing", "invalid"):
        return f"benchmark index metadata is {expected_index}"
    if summary.get("index_sha256") != expected_index:
        return "benchmark index fingerprint is missing or stale"
    if int(summary.get("in_rows") or 0) != int(summary.get("out_rows") or 0):
        return "decision row count does not close against input rows"
    if int(summary.get("output_bytes") or -1) != out_path.stat().st_size:
        return "decision parquet size is missing or stale"
    return None


def migrate_legacy_summary(
    in_path: Path,
    out_path: Path,
    summary_path: Path,
    cfg: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    """Bind a structurally verified legacy shard to current unchanged semantics.

    The 2026-09 production run predates lineage fields.  A metadata-only migration
    is valid for sources whose effective scan thresholds did not change. Sources
    with any source-specific override are refused and must be rescanned.
    """
    overrides = (cfg.get("source_min_distinct_hits") or {}).get(source) or {}
    if overrides:
        raise ValueError(
            f"{source} has source-specific decontam thresholds and must be rescanned"
        )
    if not in_path.is_file() or not out_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"legacy decontam output incomplete: {source}/{in_path.name}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    reason = validate_published_summary(in_path, out_path, summary, cfg, source)
    if reason is None:
        result = dict(summary)
        result["skipped"] = True
        return result
    if any(summary.get(k) for k in ("input_fingerprint", "config_sha256", "index_sha256")):
        raise ValueError(f"refusing to migrate partially versioned stale summary: {reason}")
    if summary.get("errors") or summary.get("shard_fatal"):
        raise ValueError("legacy decontam summary contains errors")
    if in_path.stat().st_mtime_ns > out_path.stat().st_mtime_ns:
        raise ValueError("S3 input is newer than legacy decontam output; rescan required")

    in_rows = int(pq.ParquetFile(in_path).metadata.num_rows)
    out_pf = pq.ParquetFile(out_path)
    out_rows = int(out_pf.metadata.num_rows)
    if out_pf.schema_arrow.names != DECONTAM_SCHEMA.names:
        raise ValueError(f"legacy decontam schema mismatch: {out_pf.schema_arrow}")
    if in_rows != out_rows:
        raise ValueError(f"input/output row mismatch: {in_rows} != {out_rows}")
    if in_rows != int(summary.get("in_rows") or 0):
        raise ValueError("legacy summary input row count mismatch")
    if out_rows != int(summary.get("out_rows") or 0):
        raise ValueError("legacy summary output row count mismatch")
    decisions = {k: int(v) for k, v in (summary.get("decisions") or {}).items()}
    if sum(decisions.values()) != out_rows:
        raise ValueError("legacy decision ledger does not close")
    if int(summary.get("output_bytes") or -1) != out_path.stat().st_size:
        raise ValueError("legacy output byte count mismatch")
    index_sha256 = current_index_sha256(cfg)
    if index_sha256 in ("missing", "invalid"):
        raise ValueError(f"benchmark index metadata is {index_sha256}")

    summary.update(
        {
            "input_fingerprint": fingerprint(in_path),
            "config_sha256": _scan_config_echo(cfg, source),
            "index_sha256": index_sha256,
            "effective_min_distinct_hits": _effective_min_distinct_hits(cfg, source),
            "lineage_migration": {
                "schema_version": 1,
                "method": "legacy_structural_binding",
                "verified": [
                    "input_precedes_output",
                    "input_output_parquet_rows_equal",
                    "summary_row_ledger_closed",
                    "decision_ledger_closed",
                    "output_schema_and_size",
                    "benchmark_index_content_sha256",
                    "unchanged_effective_thresholds",
                ],
            },
            "skipped": False,
        }
    )
    write_summary(summary_path, summary)
    return summary


def _decide_batch(
    texts: list[str], index: BenchmarkIndex, n: int, cap: int | None,
    min_distinct_hits: dict[str, int] | None = None,
) -> tuple[list[tuple[bool, str, bool]], int, int]:
    """返回 [(is_drop, hit_benchmark, truncated)], truncated_docs, short_docs。
    累积至固定 shingle 预算后合并 searchsorted：保留 memmap 顺序读吞吐，同时使
    ``cap=None`` 的全文扫描内存有界，不再由 parquet batch 内长文档总长度决定峰值。"""
    out: list[tuple[bool, str, bool]] = []
    truncated_docs = 0
    short_docs = 0
    pending: list[tuple[np.ndarray, bool]] = []
    pending_shingles = 0

    def flush() -> None:
        nonlocal pending, pending_shingles
        if not pending:
            return
        all_sh = np.concatenate([sh for sh, _ in pending])
        all_codes = index.hit_codes_batch(all_sh)
        off = 0
        for sh, truncated in pending:
            doc_codes = all_codes[off : off + len(sh)]
            off += len(sh)
            hit = doc_codes[doc_codes >= 0]
            qualified: list[int] = []
            for code in np.unique(hit):
                name = index.names[int(code)]
                required = int((min_distinct_hits or {}).get(name, 1))
                # 同一句在文档内重复不能凑阈值，必须是不同 benchmark shingle。
                if len(np.unique(sh[doc_codes == code])) >= required:
                    qualified.append(int(code))
            if qualified:
                out.append((True, index.names[min(qualified)], truncated))
            else:
                out.append((False, "", truncated))
        pending = []
        pending_shingles = 0

    for text in texts:
        sh, truncated = doc_shingle_hashes(text, n, cap)
        if len(sh) == 0:
            flush()
            short_docs += 1
            out.append((False, "", False))
            continue
        if truncated:
            truncated_docs += 1
        if pending and pending_shingles + len(sh) > _MAX_QUERY_SHINGLES:
            flush()
        pending.append((sh, truncated))
        pending_shingles += len(sh)
        if pending_shingles >= _MAX_QUERY_SHINGLES:
            flush()
    flush()
    return out, truncated_docs, short_docs


def _legacy_threshold_delta(
    in_path: Path,
    out_path: Path,
    summary: dict[str, Any],
    cfg: dict[str, Any],
    source: str,
) -> tuple[pa.Table, set[str]] | None:
    """Return an aligned legacy decision table for a pure threshold increase.

    A threshold increase can only change rows whose previously selected
    benchmark is one of the overridden benchmarks.  All keeps and all other
    benchmark drops are invariant.  This path is deliberately limited to the
    unversioned 2026-09 artifacts whose baseline threshold was one distinct
    shingle; anything partially versioned or structurally ambiguous falls back
    to a full scan.
    """
    overrides = {
        str(name): int(value)
        for name, value in (
            (cfg.get("source_min_distinct_hits") or {}).get(source) or {}
        ).items()
    }
    if not overrides or any(value <= 1 for value in overrides.values()):
        return None
    if any(int(b.get("min_distinct_hits", 1)) != 1 for b in cfg["benchmarks"]):
        return None
    if any(summary.get(key) for key in ("input_fingerprint", "config_sha256", "index_sha256")):
        return None
    if summary.get("errors") or summary.get("shard_fatal"):
        return None
    if not in_path.is_file() or not out_path.is_file():
        return None
    if in_path.stat().st_mtime_ns > out_path.stat().st_mtime_ns:
        return None
    in_rows = int(pq.ParquetFile(in_path).metadata.num_rows)
    out_pf = pq.ParquetFile(out_path)
    out_rows = int(out_pf.metadata.num_rows)
    if out_pf.schema_arrow.names != DECONTAM_SCHEMA.names or in_rows != out_rows:
        return None
    if in_rows != int(summary.get("in_rows") or -1) or out_rows != int(summary.get("out_rows") or -1):
        return None
    decisions = {k: int(v) for k, v in (summary.get("decisions") or {}).items()}
    hit_counts = {k: int(v) for k, v in (summary.get("hit_benchmarks") or {}).items()}
    if sum(decisions.values()) != out_rows:
        return None
    if sum(hit_counts.values()) != int(decisions.get(DECISION_DROP, 0)):
        return None
    if int(summary.get("output_bytes") or -1) != out_path.stat().st_size:
        return None
    return out_pf.read(use_threads=False), set(overrides)


def _legacy_cap_expansion_delta(
    in_path: Path,
    out_path: Path,
    summary: dict[str, Any],
    cfg: dict[str, Any],
    source: str,
) -> tuple[pa.Table, int] | None:
    """Return old aligned decisions for a certified prefix→full scan.

    Full-document shingles are a strict superset of the old prefix shingles.
    Only documents that actually exceeded the former cap need a new hash/index
    pass; all other decisions and selected benchmark labels are identical.
    """
    old_cap = cfg.get("monotonic_from_max_shingles_per_doc")
    if cfg.get("max_shingles_per_doc") is not None or old_cap is None:
        return None
    old_cap = int(old_cap)
    old_cfg = dict(cfg)
    old_cfg["max_shingles_per_doc"] = old_cap
    old_cfg["monotonic_from_max_shingles_per_doc"] = None
    if validate_published_summary(in_path, out_path, summary, old_cfg, source) is not None:
        return None
    out_pf = pq.ParquetFile(out_path)
    if out_pf.schema_arrow.names != DECONTAM_SCHEMA.names:
        return None
    return out_pf.read(use_threads=False), old_cap


def process_shard_decontam(
    in_path: Path,
    out_root: Path,
    source: str,
    relpath: str,
    index: BenchmarkIndex,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """读 input_stage 分片（doc_id,text 列）→ 文档 8-gram 命中即 drop → 决策表落盘。"""
    out_path, summary_path = decontam_output_paths(out_root, source, relpath)
    input_exists = in_path.is_file()
    input_fingerprint = fingerprint(in_path) if input_exists else "missing"
    config_sha256 = _scan_config_echo(cfg, source)
    index_sha256 = index.index_sha256
    published_summary: dict[str, Any] | None = None
    if summary_path.is_file() and out_path.is_file():
        published_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if input_exists and validate_published_summary(
            in_path, out_path, published_summary, cfg, source
        ) is None:
            # Metadata-only upgrade for outputs completed by the same decision
            # config before scan_policy was added.  The config hash and output
            # identity have already been validated above.
            policy = _scan_policy(cfg)
            if published_summary.get("scan_policy") != policy:
                published_summary["scan_policy"] = policy
                published_summary["skipped"] = False
                write_summary(summary_path, published_summary)
            published_summary["skipped"] = True
            return published_summary

    started = time.monotonic()
    n = int(cfg["ngram_n"])
    cap_raw = cfg.get("max_shingles_per_doc")
    cap = None if cap_raw is None else int(cap_raw)
    min_hits = _effective_min_distinct_hits(cfg, source)
    cap_delta: tuple[pa.Table, int] | None = None
    delta: tuple[pa.Table, set[str]] | None = None
    if published_summary is not None:
        cap_delta = _legacy_cap_expansion_delta(
            in_path, out_path, published_summary, cfg, source
        )
        if cap_delta is None:
            delta = _legacy_threshold_delta(
                in_path, out_path, published_summary, cfg, source
            )
    legacy_table = cap_delta[0] if cap_delta is not None else (delta[0] if delta is not None else None)
    legacy_cap = cap_delta[1] if cap_delta is not None else None
    delta_benchmarks = delta[1] if delta is not None else set()
    delta_recomputed_rows = 0
    if cap_delta is not None and int((published_summary or {}).get("truncated_docs") or 0) == 0:
        # The old summary already proves that its prefix covered every document
        # in this shard.  Full-scan decisions are byte-for-byte identical, so a
        # metadata-only publication is sufficient and avoids reading S3 text.
        summary = dict(published_summary or {})
        summary.update(
            {
                "input_fingerprint": input_fingerprint,
                "config_sha256": config_sha256,
                "index_sha256": index_sha256,
                "effective_min_distinct_hits": min_hits,
                "scan_policy": _scan_policy(cfg),
                "truncated_docs": 0,
                "elapsed_sec": round(time.monotonic() - started, 3),
                "incremental_rescan": {
                    "schema_version": 1,
                    "method": "legacy_prefix_to_full_document",
                    "previous_max_shingles_per_doc": int(legacy_cap),
                    "recomputed_rows": 0,
                    "verified": [
                        "old_output_valid_under_exact_previous_config",
                        "old_summary_truncated_docs_zero",
                        "full_shingle_set_equals_prefix_for_every_document",
                    ],
                },
                "errors": {},
                "skipped": False,
            }
        )
        write_summary(summary_path, summary)
        return summary
    in_rows = 0
    decisions: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    previous_decisions = {
        k: int(v) for k, v in ((published_summary or {}).get("decisions") or {}).items()
    }
    truncated_docs = int((published_summary or {}).get("truncated_docs") or 0) if delta else 0
    short_docs = int((published_summary or {}).get("short_docs") or 0) if (delta or cap_delta) else 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + f".tmp-{os.getpid()}")
    writer: pq.ParquetWriter | None = None
    try:
        pf = pq.ParquetFile(in_path)
        for batch in safe_iter_batches(pf, batch_size=4096, columns=["doc_id", "text"]):
            rows = batch.to_pylist()
            texts = [r["text"] if isinstance(r["text"], str) else "" for r in rows]
            if legacy_table is None:
                verdicts, n_trunc, n_short = _decide_batch(
                    texts, index, n, cap, min_distinct_hits=min_hits
                )
                truncated_docs += n_trunc
                short_docs += n_short
            elif cap_delta is not None:
                count = len(rows)
                old = legacy_table.slice(in_rows, count)
                old_ids = old.column("doc_id").combine_chunks()
                if len(old) != count or not batch.column("doc_id").equals(old_ids):
                    raise ValueError(
                        f"legacy cap delta doc_id alignment failed at row {in_rows}"
                    )
                old_decisions = old.column("decision").to_pylist()
                old_hits = old.column("hit_benchmark").to_pylist()
                candidates = [
                    i for i, text in enumerate(texts)
                    if exceeds_shingle_cap(text, n, int(legacy_cap))
                ]
                if candidates:
                    revised, _n_trunc, _n_short = _decide_batch(
                        [texts[i] for i in candidates],
                        index,
                        n,
                        None,
                        min_distinct_hits=min_hits,
                    )
                    delta_recomputed_rows += len(candidates)
                    for i, (is_drop, benchmark, _truncated) in zip(candidates, revised):
                        old_decisions[i] = DECISION_DROP if is_drop else DECISION_KEEP
                        old_hits[i] = benchmark if is_drop else ""
                verdicts = [
                    (decision == DECISION_DROP, benchmark, False)
                    for decision, benchmark in zip(old_decisions, old_hits)
                ]
            else:
                count = len(rows)
                old = legacy_table.slice(in_rows, count)
                old_ids = old.column("doc_id").combine_chunks()
                if len(old) != count or not batch.column("doc_id").equals(old_ids):
                    raise ValueError(
                        f"legacy delta doc_id alignment failed at row {in_rows}"
                    )
                old_decisions = old.column("decision").to_pylist()
                old_hits = old.column("hit_benchmark").to_pylist()
                candidates = [
                    i
                    for i, (decision, benchmark) in enumerate(
                        zip(old_decisions, old_hits)
                    )
                    if decision == DECISION_DROP and benchmark in delta_benchmarks
                ]
                if candidates:
                    revised, _n_trunc, _n_short = _decide_batch(
                        [texts[i] for i in candidates],
                        index,
                        n,
                        cap,
                        min_distinct_hits=min_hits,
                    )
                    delta_recomputed_rows += len(candidates)
                    for i, (is_drop, benchmark, truncated) in zip(candidates, revised):
                        if is_drop and benchmark not in delta_benchmarks:
                            raise ValueError(
                                "threshold-only delta selected an unexpected benchmark"
                            )
                        old_decisions[i] = DECISION_DROP if is_drop else DECISION_KEEP
                        old_hits[i] = benchmark if is_drop else ""
                verdicts = [
                    (decision == DECISION_DROP, benchmark, False)
                    for decision, benchmark in zip(old_decisions, old_hits)
                ]
            in_rows += len(rows)
            doc_ids: list[str] = []
            dec_col: list[str] = []
            hit_col: list[str] = []
            for row, (is_drop, bench, _t) in zip(rows, verdicts):
                dec = DECISION_DROP if is_drop else DECISION_KEEP
                decisions[dec] += 1
                if is_drop:
                    hits[bench] += 1
                doc_ids.append(row["doc_id"])
                dec_col.append(dec)
                hit_col.append(bench)
            table = pa.table(
                {"doc_id": doc_ids, "decision": dec_col, "hit_benchmark": hit_col},
                schema=DECONTAM_SCHEMA,
            )
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, DECONTAM_SCHEMA, compression="zstd")
            writer.write_table(table)
        if cap_delta is not None:
            expected_recomputed = int((published_summary or {}).get("truncated_docs") or 0)
            if delta_recomputed_rows != expected_recomputed:
                raise ValueError(
                    f"legacy truncated-doc ledger mismatch: detected {delta_recomputed_rows}, "
                    f"summary declared {expected_recomputed}"
                )
            previous_drop = int(previous_decisions.get(DECISION_DROP, 0))
            current_drop = int(decisions.get(DECISION_DROP, 0))
            if current_drop < previous_drop:
                raise ValueError(
                    "prefix-to-full monotonicity violated: "
                    f"previous_drop={previous_drop}, current_drop={current_drop}"
                )
        if writer is not None:
            writer.close()
            os.replace(tmp_path, out_path)
            out_bytes = out_path.stat().st_size
        else:
            # E-37（2026-10-08 修订）：合法空分片（上游 0 行）也落 schema-only parquet。
            # S5 硬前置按"summary+parquet 双在"判定 decontam 覆盖——与其他 stage 的
            # 产物契约一致；不落盘会让 54 个空分片永远卡 gate。0-row-group 读取由
            # ingest.safe_iter_batches 统一兜住。E-16"不落空产物"语义仅保留给失败路径。
            writer = pq.ParquetWriter(tmp_path, DECONTAM_SCHEMA, compression="zstd")
            writer.close()
            os.replace(tmp_path, out_path)
            out_bytes = out_path.stat().st_size
    except Exception as exc:
        if writer is not None:
            writer.close()
        if tmp_path.exists():
            tmp_path.unlink()  # 失败空产物不落盘，下次运行自动重试
        elapsed = time.monotonic() - started
        summary = {
            "stage": "decontam",
            "source": source,
            "relpath": relpath,
            "input_fingerprint": input_fingerprint,
            "config_sha256": config_sha256,
            "index_sha256": index_sha256,
            "in_rows": in_rows,
            "out_rows": 0,
            "decisions": dict(decisions),
            "hit_benchmarks": dict(hits),
            "truncated_docs": truncated_docs,
            "short_docs": short_docs,
            "output_bytes": 0,
            "elapsed_sec": round(elapsed, 3),
            "errors": {f"shard_error:{type(exc).__name__}": 1},
            "error_message": str(exc)[:500],
            "skipped": False,
        }
        write_summary(summary_path, summary)
        return summary

    elapsed = time.monotonic() - started
    summary = {
        "stage": "decontam",
        "source": source,
        "relpath": relpath,
        "input_fingerprint": input_fingerprint,
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
        "effective_min_distinct_hits": min_hits,
        "scan_policy": _scan_policy(cfg),
        "in_rows": in_rows,
        "out_rows": int(sum(decisions.values())),
        "decisions": dict(decisions),
        "hit_benchmarks": dict(hits),
        "truncated_docs": truncated_docs,
        "short_docs": short_docs,
        "output_bytes": out_bytes,
        "elapsed_sec": round(elapsed, 3),
        "rows_per_sec": round(in_rows / max(elapsed, 1e-9), 1),
        "errors": {},
        "skipped": False,
    }
    if delta is not None:
        summary["incremental_rescan"] = {
            "schema_version": 1,
            "method": "legacy_selected_benchmark_threshold_increase",
            "baseline_min_distinct_hits": 1,
            "changed_benchmarks": sorted(delta_benchmarks),
            "recomputed_rows": delta_recomputed_rows,
            "verified": [
                "input_precedes_legacy_output",
                "input_output_rows_equal",
                "legacy_summary_and_decision_ledgers_closed",
                "doc_id_row_alignment_exact",
                "non_selected_decisions_invariant_under_threshold_increase",
            ],
        }
    if cap_delta is not None:
        previous_drop = int(previous_decisions.get(DECISION_DROP, 0))
        current_drop = int(decisions.get(DECISION_DROP, 0))
        summary["incremental_rescan"] = {
            "schema_version": 1,
            "method": "legacy_prefix_to_full_document",
            "previous_max_shingles_per_doc": int(legacy_cap),
            "recomputed_rows": delta_recomputed_rows,
            "previous_decisions": previous_decisions,
            "newly_dropped": current_drop - previous_drop,
            "verified": [
                "old_output_valid_under_exact_previous_config",
                "input_output_rows_equal",
                "doc_id_row_alignment_exact",
                "untruncated_shingle_sets_identical",
                "full_shingle_set_superset_of_prefix",
            ],
        }
    write_summary(summary_path, summary)
    return summary


# ---------------------------------------------------------------- driver 接入

_WORKER: dict[str, Any] = {}


def _init_worker(index_dir: str, cfg_path: str) -> None:
    _WORKER["index"] = BenchmarkIndex(Path(index_dir))
    _WORKER["cfg"] = load_decontam_config(cfg_path)


def _run_shard(task: tuple[Path, str, str, str]) -> dict[str, Any]:
    out_root, source, relpath, in_path = task
    try:
        return process_shard_decontam(
            Path(in_path), Path(out_root), source, relpath,
            _WORKER["index"], _WORKER["cfg"],
        )
    except Exception as exc:
        return {
            "stage": "decontam",
            "source": source,
            "relpath": relpath,
            "shard_fatal": f"{type(exc).__name__}: {exc}",
        }


def shard_tasks(
    out_root: Path, cfg: dict[str, Any], sources: list[str] | None, limit: int
) -> list[tuple[Path, str, str, str]]:
    root = out_root / str(cfg["input_stage"])
    if not root.is_dir():
        raise FileNotFoundError(
            f"decontam 输入阶段产物不存在: {root}（先跑 --stage {cfg['input_stage']}，"
            f"或在 decontam.yaml 改 input_stage）"
        )
    tasks = []
    src_filter = set(sources) if sources else None
    for src_dir in sorted(root.iterdir()):
        if not src_dir.is_dir():
            continue
        if src_filter is not None and src_dir.name not in src_filter:
            continue
        for path in sorted(src_dir.rglob("*.parquet")):
            rel = path.relative_to(src_dir).as_posix()
            tasks.append((out_root, src_dir.name, rel[: -len(".parquet")], str(path)))
    tasks.sort(key=lambda t: (t[1], t[2]))
    if limit:
        tasks = tasks[:limit]
    return tasks


def collect_published_summaries(
    out_root: Path, cfg: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Rebuild the global view from the complete published shard roster.

    A targeted retry must never replace the global summary with only the retried
    sources. Missing/corrupt summary or parquet pairs become explicit failed rows.
    """
    summaries: list[dict[str, Any]] = []
    coverage = {"expected": 0, "complete": 0, "missing_summary": 0, "missing_output": 0}
    for _root, source, relpath, input_path in shard_tasks(out_root, cfg, None, 0):
        coverage["expected"] += 1
        out_path, summary_path = decontam_output_paths(out_root, source, relpath)
        if not summary_path.is_file():
            coverage["missing_summary"] += 1
            summaries.append({
                "stage": "decontam", "source": source, "relpath": relpath,
                "shard_fatal": "published summary missing",
            })
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            coverage["missing_summary"] += 1
            summaries.append({
                "stage": "decontam", "source": source, "relpath": relpath,
                "shard_fatal": f"published summary unreadable: {type(exc).__name__}: {exc}",
            })
            continue
        if not out_path.is_file():
            coverage["missing_output"] += 1
            summary = dict(summary)
            summary["shard_fatal"] = "published parquet missing"
        else:
            coverage["complete"] += 1
            stale_reason = validate_published_summary(
                Path(input_path), out_path, summary, cfg, source
            )
            if stale_reason:
                summary = dict(summary)
                summary["shard_fatal"] = stale_reason
        summaries.append(summary)
    return summaries, coverage


def write_global_summary(
    out_root: Path,
    summaries: list[dict[str, Any]],
    *,
    coverage: dict[str, int] | None = None,
    cfg: dict[str, Any] | None = None,
) -> Path:
    # E-16：失败计数置顶；attempt/success 口径分离
    per_source: dict[str, Counter[str]] = {}
    totals: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    hits: Counter[str] = Counter()
    for s in summaries:
        src = s.get("source", "?")
        acc = per_source.setdefault(src, Counter())
        acc["in_rows"] += int(s.get("in_rows") or 0)
        acc["out_rows"] += int(s.get("out_rows") or 0)
        acc["truncated_docs"] += int(s.get("truncated_docs") or 0)
        acc["short_docs"] += int(s.get("short_docs") or 0)
        acc["elapsed_sec"] += float(s.get("elapsed_sec") or 0.0)
        for key in ("in_rows", "out_rows", "truncated_docs", "short_docs"):
            totals[key] += int(s.get(key) or 0)
        decisions.update(s.get("decisions") or {})
        hits.update(s.get("hit_benchmarks") or {})
        for b, v in (s.get("hit_benchmarks") or {}).items():
            acc[f"hit.{b}"] += int(v)
    payload = {
        "stage": "decontam",
        "shards": len(summaries),
        "shard_errors": sum(1 for s in summaries if s.get("errors") or s.get("shard_fatal")),
        "totals": dict(totals),
        "decisions": dict(decisions),
        "hit_benchmarks": dict(sorted(hits.items())),
        "per_source": {k: dict(v) for k, v in sorted(per_source.items())},
    }
    if coverage is not None:
        payload["coverage"] = coverage
    if cfg is not None:
        payload["publication_contract"] = {
            "schema_version": 2,
            "config_sha256": _config_echo(cfg),
            "index_sha256": current_index_sha256(cfg),
            "scan_policy": _scan_policy(cfg),
        }
    path = out_root / "stage_decontam_global_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1),
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return path


def audit_fullscan_transition(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    """Prove that a capped decontam generation was replaced monotonically.

    The prefix-to-full policy may only convert keep→drop. Benchmark labels can
    change when a newly discovered lower-code benchmark wins, so monotonicity is
    checked on total drops globally and per source rather than per benchmark.
    """
    failures: list[str] = []
    for label, value in (("before", before), ("after", after)):
        if value.get("stage") != "decontam":
            failures.append(f"{label}.stage != decontam")
        if int(value.get("shard_errors") or 0):
            failures.append(f"{label}.shard_errors != 0")
        coverage = value.get("coverage") or {}
        if int(coverage.get("expected") or 0) != int(coverage.get("complete") or -1):
            failures.append(f"{label}.coverage incomplete")
    before_totals = before.get("totals") or {}
    after_totals = after.get("totals") or {}
    for key in ("in_rows", "out_rows"):
        if int(before_totals.get(key) or 0) != int(after_totals.get(key) or -1):
            failures.append(f"{key} changed across transition")
    if int(before_totals.get("truncated_docs") or 0) <= 0:
        failures.append("before.truncated_docs does not prove a capped generation")
    if int(after_totals.get("truncated_docs") or 0) != 0:
        failures.append("after.truncated_docs != 0")
    scan_policy = (
        (after.get("publication_contract") or {}).get("scan_policy") or {}
    )
    if scan_policy.get("scope") != "full_document" or scan_policy.get(
        "max_shingles_per_doc", "missing"
    ) is not None:
        failures.append("after publication is not a full-document scan")
    if (before.get("publication_contract") or {}).get("index_sha256") != (
        after.get("publication_contract") or {}
    ).get("index_sha256"):
        failures.append("benchmark index changed across transition")

    def _drops(summary: dict[str, Any]) -> int:
        return int((summary.get("decisions") or {}).get(DECISION_DROP) or 0)

    before_drop, after_drop = _drops(before), _drops(after)
    if after_drop < before_drop:
        failures.append("global drop count decreased")
    sources = sorted(set(before.get("per_source") or {}) | set(after.get("per_source") or {}))
    per_source: dict[str, Any] = {}
    for source in sources:
        old = (before.get("per_source") or {}).get(source) or {}
        new = (after.get("per_source") or {}).get(source) or {}
        old_drop = sum(int(v) for k, v in old.items() if k.startswith("hit."))
        new_drop = sum(int(v) for k, v in new.items() if k.startswith("hit."))
        if new_drop < old_drop:
            failures.append(f"{source} drop count decreased")
        per_source[source] = {
            "before_dropped": old_drop,
            "after_dropped": new_drop,
            "newly_dropped": new_drop - old_drop,
        }
    return {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "rows": int(after_totals.get("out_rows") or 0),
        "previous_truncated_docs": int(before_totals.get("truncated_docs") or 0),
        "after_truncated_docs": int(after_totals.get("truncated_docs") or 0),
        "before_dropped": before_drop,
        "after_dropped": after_drop,
        "newly_dropped": after_drop - before_drop,
        "index_sha256": (after.get("publication_contract") or {}).get("index_sha256"),
        "scan_policy": scan_policy,
        "per_source": per_source,
    }


def release_readiness_errors(summary: dict[str, Any]) -> list[str]:
    """Return fail-closed reasons why a decontam generation is not consumable."""
    errors: list[str] = []
    if summary.get("stage") != "decontam":
        errors.append("stage is not decontam")
    if int(summary.get("shard_errors") or 0):
        errors.append(f"shard_errors={summary.get('shard_errors')}")
    coverage = summary.get("coverage") or {}
    expected = int(coverage.get("expected") or 0)
    complete = int(coverage.get("complete") or 0)
    if expected <= 0 or complete != expected:
        errors.append(f"coverage incomplete: complete={complete}, expected={expected}")
    if int(coverage.get("missing_summary") or 0) or int(coverage.get("missing_output") or 0):
        errors.append("coverage reports missing summaries or outputs")
    totals = summary.get("totals") or {}
    if int(totals.get("truncated_docs") or 0):
        errors.append(f"truncated_docs={totals.get('truncated_docs')}")
    policy = (summary.get("publication_contract") or {}).get("scan_policy") or {}
    if policy.get("scope") != "full_document" or policy.get(
        "max_shingles_per_doc", "missing"
    ) is not None:
        errors.append("publication scan policy is not full_document")
    return errors


def run_decontam_scan(
    tasks: list[tuple[Path, str, str, str]],
    cfg_path: str,
    index_dir: Path,
    workers: int,
    guard: tuple[Any, Any],
    log_fn: Any = log.info,
) -> list[dict[str, Any]]:
    beat, guard_stop = guard
    summaries: list[dict[str, Any]] = []
    started = time.monotonic()
    done = 0
    rows = 0
    dropped = 0
    try:
        with mp.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(str(index_dir), str(cfg_path)),
        ) as pool:
            it = pool.imap_unordered(_run_shard, tasks, chunksize=1)
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
                beat()
                rows += int(summary.get("in_rows") or 0)
                dropped += int((summary.get("decisions") or {}).get(DECISION_DROP) or 0)
                if done % 50 == 0 or done == len(tasks):
                    elapsed = time.monotonic() - started
                    log_fn(
                        "progress[decontam]: %d/%d shards, %.2f shards/s, %.0f docs/s, dropped=%d",
                        done, len(tasks), done / max(elapsed, 1e-9),
                        rows / max(elapsed, 1e-9), dropped,
                    )
    finally:
        guard_stop.set()
    return summaries
