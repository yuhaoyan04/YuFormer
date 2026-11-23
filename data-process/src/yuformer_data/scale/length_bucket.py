"""S5 计长分桶（2026-09-26 实现）。

输入 = s3 keep 池行，三个决策源过滤（与 S1 三态审计哲学一致：决策落审计链，
过滤在物化前显式执行）：
  ① s2_decision 列（s3 分片自带）：drop_exact_dup 剔除；
  ② s3 近重：s3_decisions/<domain>.parquet 决策表（或其 lookup bin 物化，
     memmap + searchsorted，全 worker 共享 page cache）按 doc_id 左连，
     drop_near_dup 剔除；
  ③ 评测去污染：decontam/<source>/<relpath>.parquet（doc_id, decision），
     drop_contaminated 剔除。

计长：自训 yf-64k-bpe（HF `tokenizers` 库直读 tokenizer.json，每 worker 进程
加载一次），encode_batch 批量编码、add_special_tokens=False（裸文本 token 数，
与 pretrain 侧 packing 口径一致）。输出 n_tokens uint32（S1 max_chars=2M ⇒
token 数 << 2^32）。

分桶与资格：沿用 training.py 互斥前缀嵌套语义——doc 落唯一互斥桶
（B4K/B16K/B64K/gt_64K，按 cleaning_config.yaml context_lengths），
mix_eligibility = 该 doc 有资格进入的 mix 阶段列表（n_tokens <= L 的每个
context length 一个；短桶有资格进长阶段，每条最终只进一个 mix——最终选取
物化在 S6）。gt_64K doc 对前缀嵌套各阶段无资格（后续 midtrain 截断消费，
资格表留空是可审计的有意识取舍）。

工程约束沿用事故账本：文件级幂等（parquet+summary 齐在即 skip）、tmp+os.replace、
失败分片只落 error summary 不落空产物（E-16）、driver 停滞守卫（driver.py 同款，
stage 结束必须 stop，否则旧守卫误杀后续 stage）、分片流式（无 accumulate-then-
process，E-12）、lookup memmap 共享 page cache（E-13/E-14）。
"""
from __future__ import annotations

import json
import hashlib
import logging
import multiprocessing as mp
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from ..training import assign_context_bucket
from .dedup_common import drops_for, fingerprint
from .dedup_minhash import s3_decisions_path, s3_lookup_dir, s3_output_paths
from .ingest import BatchWriter, safe_iter_batches, write_summary

log = logging.getLogger("yuformer.scale.length_bucket")

DEFAULT_CLEANING_CONFIG = (
    Path(__file__).resolve().parents[3] / "configs" / "cleaning_config.yaml"
)
# tokenizer 为外部资产（AGENTS.md 宪法 3）：默认取仓库旁 tokenizer 模块产物，
# YUFORMER_TOKENIZER 环境变量可覆盖（HF 目录或 tokenizer.json 文件路径）。
DEFAULT_TOKENIZER_DIR = (
    Path(__file__).resolve().parents[4] / "tokenizer" / "tokenizers" / "yf-64k-bpe"
)

S5_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("n_tokens", pa.uint32(), nullable=False),
        pa.field("length_bucket", pa.string(), nullable=False),
        pa.field("mix_eligibility", pa.list_(pa.string()), nullable=False),
    ]
)

ENCODE_BATCH = 2048
ENCODE_ROW_CAP = 2048
ENCODE_CHAR_CAP = 16_000_000
READ_CHAR_CAP = 32_000_000

_WORKER: dict[str, Any] = {}
TOKEN_CACHE_AUDIT_ROWS = 8


def s5_output_paths(out_root: Path, source: str, relpath: str) -> tuple[Path, Path]:
    base = out_root / "s5" / source / relpath
    return base.with_name(base.name + ".parquet"), base.with_name(base.name + ".summary.json")


def load_context_lengths(path: str | Path | None = None) -> list[int]:
    p = Path(path) if path else DEFAULT_CLEANING_CONFIG
    if not p.is_file():
        raise FileNotFoundError(f"cleaning config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    lengths = (cfg or {}).get("context_lengths") or [4096, 16384, 65536]
    out = sorted({int(x) for x in lengths if int(x) > 0})
    if not out:
        raise ValueError(f"context_lengths empty in {p}")
    return out


def resolve_tokenizer_path(path: str | Path | None = None) -> Path:
    raw = path or os.environ.get("YUFORMER_TOKENIZER") or DEFAULT_TOKENIZER_DIR
    p = Path(raw)
    if p.is_dir():
        p = p / "tokenizer.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"tokenizer.json not found: {p}（训练 yf-64k-bpe 或设 YUFORMER_TOKENIZER）"
        )
    return p


def eligibility_flags(n_tokens: int, context_lengths: list[int]) -> list[str]:
    return [f"B{L // 1024}K" for L in context_lengths if n_tokens <= L]


def _ids_to_void(doc_ids: list[str]) -> np.ndarray:
    return np.frombuffer(
        b"".join(bytes.fromhex(d) for d in doc_ids), dtype="V16"
    ).reshape(-1)


def _arrow_hex_ids_to_void(doc_ids: pa.Array) -> np.ndarray:
    """Decode a non-null fixed-width hex Arrow string column in two C loops."""
    if doc_ids.null_count:
        raise ValueError("doc_id contains nulls")
    offsets_buffer = doc_ids.buffers()[1]
    if offsets_buffer is None:
        return np.empty(0, dtype="V16")
    offsets = np.frombuffer(
        offsets_buffer,
        dtype="<i4",
        count=len(doc_ids) + 1,
        offset=int(doc_ids.offset) * 4,
    )
    if len(offsets) != len(doc_ids) + 1 or np.any(np.diff(offsets) != 32):
        raise ValueError("doc_id must contain 32-character hex strings")
    data = doc_ids.buffers()[2]
    if data is None:
        return np.empty(0, dtype="V16")
    raw = memoryview(data)[int(offsets[0]) : int(offsets[-1])].tobytes()
    try:
        decoded = bytes.fromhex(raw.decode("ascii"))
    except ValueError as exc:
        raise ValueError("doc_id contains non-hex characters") from exc
    return np.frombuffer(decoded, dtype="V16")


def load_domain_drops(out_root: Path, domain: str) -> tuple[np.ndarray, np.ndarray]:
    """s3 近重 drops lookup (ids V16 排序, codes)：优先 lookup bin（memmap），
    退回决策 parquet。worker 进程内缓存（key 含 out_root 防多 out_root 串场）。"""
    ids_path = s3_lookup_dir(out_root, domain) / "lookup.ids.bin"
    codes_path = s3_lookup_dir(out_root, domain) / "lookup.codes.bin"
    if ids_path.is_file() and codes_path.is_file():
        generation = (*_lookup_fingerprint(out_root, domain),)
    else:
        decision_path = s3_decisions_path(out_root, domain)
        if decision_path.is_file():
            stat = decision_path.stat()
            generation = (-1, -1, -1, -1, stat.st_size, stat.st_mtime_ns)
        else:
            generation = (-1, -1, -1, -1, -1, -1)
    key = ("s3drops", str(out_root), domain, generation)
    if key in _WORKER:
        return _WORKER[key]
    if ids_path.is_file() and codes_path.is_file():
        from .dedup_common import load_drops_lookup

        lookup = load_drops_lookup(s3_lookup_dir(out_root, domain))
        arr = (np.asarray(lookup[0]), np.asarray(lookup[1]))
    else:
        dec = s3_decisions_path(out_root, domain)
        if not dec.is_file():
            raise FileNotFoundError(
                f"s3 decisions missing for domain {domain}: {dec}（先跑 --stage s3/ndv2）"
            )
        ids = pq.read_table(dec, columns=["doc_id"]).column("doc_id").to_pylist()
        ids_v = np.sort(_ids_to_void(ids)) if ids else np.empty(0, dtype="V16")
        arr = (ids_v, np.ones(len(ids_v), dtype=np.uint8))
    _WORKER[key] = arr
    return arr


def load_decontam_drops(out_root: Path, source: str, relpath: str) -> set[str]:
    from .decontam import DECISION_DROP, decontam_output_paths

    out_path, _ = decontam_output_paths(out_root, source, relpath)
    if not out_path.is_file():
        raise FileNotFoundError(
            f"decontam shard missing: {out_path}（先跑 --stage decontam；S5 输入含去污染 join）"
        )
    pf = pq.ParquetFile(out_path)
    drops: set[str] = set()
    for batch in safe_iter_batches(pf, batch_size=8192, columns=["doc_id", "decision"]):
        d = batch.to_pydict()
        drops.update(
            doc for doc, dec in zip(d["doc_id"], d["decision"]) if dec == DECISION_DROP
        )
    return drops


def _encode_counts(tokenizer: Any, texts: list[str]) -> list[int]:
    """按字节自适应子批（E-3 家族修复）：ENCODE_BATCH=2048 行对长文域（dolma-books
    ~300KB/行）是 600MB/批的病态输入——内存峰值与 encode_batch 内部并行都受害。
    累积字符超过 ENCODE_CHAR_CAP 即切子批，行数上限保留。"""
    counts: list[int] = []
    sub: list[str] = []
    sub_chars = 0
    for t in texts:
        sub.append(t)
        sub_chars += len(t)
        if len(sub) >= ENCODE_ROW_CAP or sub_chars >= ENCODE_CHAR_CAP:
            encs = tokenizer.encode_batch(sub, add_special_tokens=False)
            counts.extend(len(e.ids) for e in encs)
            sub = []
            sub_chars = 0
    if sub:
        encs = tokenizer.encode_batch(sub, add_special_tokens=False)
        counts.extend(len(e.ids) for e in encs)
    return counts


def _lookup_fingerprint(out_root: Path, domain: str) -> list[int]:
    """domain 的 s3 drops lookup 代际指纹（ids/codes 的 size + mtime_ns）——ndv2 decide
    重建 lookup 后指纹变化，S5 旧产物即过期需重算（E-23 防静默过期）。"""
    from .dedup_minhash import s3_lookup_dir

    d = s3_lookup_dir(out_root, domain)
    result: list[int] = []
    for name in ("lookup.ids.bin", "lookup.codes.bin"):
        path = d / name
        if path.is_file():
            stat = path.stat()
            result.extend((stat.st_size, stat.st_mtime_ns))
        else:
            result.extend((-1, -1))
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _LegacyTokenCache:
    """Stream an older S5 shard in original S3 order without loading it in RAM.

    S5 preserves input order while filtering rows.  Comparing the cache cursor at
    every S3 row therefore supports both newly dropped and newly rescued records.
    """

    def __init__(self, path: Path):
        self._batches = iter(
            pq.ParquetFile(path).iter_batches(
                batch_size=8192, columns=["doc_id", "n_tokens"]
            )
        )
        self._ids: list[str] = []
        self._tokens: list[int] = []
        self._pos = 0
        self._eof = False

    def _fill(self) -> None:
        while self._pos >= len(self._ids) and not self._eof:
            try:
                batch = next(self._batches).to_pydict()
            except StopIteration:
                self._eof = True
                self._ids = []
                self._tokens = []
                self._pos = 0
                return
            self._ids = batch["doc_id"]
            self._tokens = [int(v) for v in batch["n_tokens"]]
            self._pos = 0
            if self._ids:
                return

    def consume_if(self, doc_id: str) -> int | None:
        self._fill()
        if self._eof or self._ids[self._pos] != doc_id:
            return None
        value = self._tokens[self._pos]
        self._pos += 1
        return value

    def exhausted(self) -> bool:
        self._fill()
        return self._eof


def _legacy_token_cache(
    out_path: Path, summary: dict[str, Any], tokenizer_sha256: str
) -> _LegacyTokenCache | None:
    contract = summary.get("token_count_cache_contract") or {}
    schema_version = int(contract.get("schema_version") or 0)
    trusted = bool(contract.get("dataset_token_audit_sha256")) or (
        schema_version == 2
        and contract.get("provenance") == "current_pipeline_tokenization_or_certified_reuse"
        and (summary.get("processing_contract") or {}).get("tokenizer_sha256")
        == tokenizer_sha256
        and bool(summary.get("content_sha256"))
    )
    if (
        schema_version not in (1, 2)
        or contract.get("tokenizer_sha256") != tokenizer_sha256
        or contract.get("output_fingerprint") != fingerprint(out_path)
        or not contract.get("structural_validation_passed")
        or not trusted
    ):
        return None
    return _LegacyTokenCache(out_path)


def _can_apply_restrictive_lookup_delta(
    domain: str, summary: dict[str, Any], token_cache: _LegacyTokenCache | None
) -> bool:
    """Whether an empty-old-lookup S5 shard can be filtered without S3 text."""
    decisions = summary.get("decisions") or {}
    return (
        domain in {"web", "code"}
        and token_cache is not None
        and not summary.get("errors")
        and int(decisions.get("drop_s3_near_dup") or 0) == 0
        and int(decisions.get("keep") or -1) == int(summary.get("out_rows") or -2)
    )


def _can_apply_fullscan_decontam_delta(
    out_root: Path,
    source: str,
    relpath: str,
    previous_summary: dict[str, Any],
    current_contract: dict[str, Any],
    token_cache: _LegacyTokenCache | None,
) -> bool:
    """Prove that the changed decontam policy can only remove old S5 rows."""
    if token_cache is None or previous_summary.get("errors"):
        return False
    old_contract = previous_summary.get("processing_contract") or {}
    invariant_keys = (
        "s3_fingerprint",
        "s3_lookup_fingerprint",
        "tokenizer_sha256",
        "context_lengths",
    )
    if any(old_contract.get(key) != current_contract.get(key) for key in invariant_keys):
        return False
    if old_contract.get("decontam_summary_sha256") == current_contract.get(
        "decontam_summary_sha256"
    ):
        return False
    decisions = previous_summary.get("decisions") or {}
    if int(decisions.get("keep") or -1) != int(previous_summary.get("out_rows") or -2):
        return False
    from .decontam import decontam_output_paths

    _out, summary_path = decontam_output_paths(out_root, source, relpath)
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    transition = (summary.get("scan_policy") or {}).get("transition") or {}
    return (
        (summary.get("scan_policy") or {}).get("scope") == "full_document"
        and transition.get("from_scope") == "prefix"
        and int(transition.get("from_max_shingles_per_doc") or 0) > 0
        and transition.get("monotonic_effect") == "only_add_drops"
    )


def _renew_token_cache_contract(
    previous_summary: dict[str, Any], out_path: Path, tokenizer_sha256: str
) -> dict[str, Any] | None:
    old = previous_summary.get("token_count_cache_contract") or {}
    audit_sha = old.get("dataset_token_audit_sha256")
    schema_version = int(old.get("schema_version") or 0)
    direct = (
        schema_version == 2
        and old.get("provenance") == "current_pipeline_tokenization_or_certified_reuse"
    )
    if schema_version not in (1, 2) or (not audit_sha and not direct):
        return None
    renewed = {
        "schema_version": schema_version,
        "tokenizer_sha256": tokenizer_sha256,
        "output_fingerprint": fingerprint(out_path),
        "structural_validation_passed": True,
    }
    if audit_sha:
        renewed["dataset_token_audit_sha256"] = audit_sha
    if direct:
        renewed["provenance"] = old["provenance"]
    return renewed


def _apply_fullscan_decontam_delta(
    out_root: Path,
    source: str,
    relpath: str,
    domain: str,
    out_path: Path,
    out_summary: Path,
    previous_summary: dict[str, Any],
    contract: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    """Filter a certified S5 table after prefix→full decontamination.

    The full-document shingle set strictly contains the old prefix set.  Old
    decontam drops therefore remain absent from S5 and only newly discovered
    drops among the old keep rows need to be removed.  Token counts and all
    other decision inputs are invariant by the caller's proof.
    """
    drops = load_decontam_drops(out_root, source, relpath)
    pf = pq.ParquetFile(out_path)
    expected_old_rows = int(previous_summary.get("out_rows") or 0)
    if int(pf.metadata.num_rows) != expected_old_rows:
        raise ValueError(f"legacy S5 cache row closure failed: {source}/{relpath}")
    tmp_path = out_path.with_name(out_path.name + f".decontam-delta-tmp-{os.getpid()}")
    writer = pq.ParquetWriter(tmp_path, S5_SCHEMA, compression="zstd")
    content_digest = hashlib.sha256()
    buckets: Counter[str] = Counter()
    bucket_tokens: Counter[str] = Counter()
    kept = 0
    newly_dropped = 0
    total_tokens = 0
    try:
        for batch in safe_iter_batches(pf, batch_size=32768):
            ids_py = batch.column("doc_id").to_pylist()
            keep_mask = np.fromiter(
                (doc_id not in drops for doc_id in ids_py),
                dtype=bool,
                count=len(ids_py),
            )
            newly_dropped += int(len(keep_mask) - np.count_nonzero(keep_mask))
            if not np.any(keep_mask):
                continue
            kept_batch = batch.filter(pa.array(keep_mask))
            nts = kept_batch.column("n_tokens").to_numpy(
                zero_copy_only=False
            ).astype(np.uint32, copy=False)
            digest_rows = np.empty(
                len(nts), dtype=np.dtype([("doc_id", "V16"), ("n_tokens", "<u4")])
            )
            digest_rows["doc_id"] = _arrow_hex_ids_to_void(kept_batch.column("doc_id"))
            digest_rows["n_tokens"] = nts
            content_digest.update(digest_rows.tobytes())
            actual_buckets = kept_batch.column("length_bucket").to_numpy(
                zero_copy_only=False
            )
            for label in np.unique(actual_buckets):
                mask = actual_buckets == label
                buckets[str(label)] += int(np.count_nonzero(mask))
                bucket_tokens[str(label)] += int(nts[mask].sum(dtype=np.uint64))
            total_tokens += int(nts.sum(dtype=np.uint64))
            kept += len(nts)
            writer.write_batch(kept_batch)
        writer.close()
        pf.close()
        old_decisions = previous_summary.get("decisions") or {}
        decisions = {
            "drop_s2_exact_dup": int(old_decisions.get("drop_s2_exact_dup") or 0),
            "drop_s3_near_dup": int(old_decisions.get("drop_s3_near_dup") or 0),
            "drop_decontam": int(old_decisions.get("drop_decontam") or 0) + newly_dropped,
            "keep": kept,
        }
        in_rows = int(previous_summary.get("in_rows") or 0)
        if sum(decisions.values()) != in_rows:
            raise ValueError(
                f"fullscan decontam delta ledger does not close: {source}/{relpath}"
            )
        out_bytes = tmp_path.stat().st_size
        os.replace(tmp_path, out_path)
    except BaseException:
        try:
            writer.close()
        except Exception:
            pass
        tmp_path.unlink(missing_ok=True)
        raise

    elapsed = time.monotonic() - started
    summary: dict[str, Any] = {
        "stage": "s5",
        "source": source,
        "relpath": relpath,
        "domain": domain,
        "in_rows": in_rows,
        "out_rows": kept,
        "total_tokens": total_tokens,
        "decisions": decisions,
        "buckets": dict(buckets),
        "bucket_tokens": dict(bucket_tokens),
        "content_sha256": content_digest.hexdigest(),
        "content_hash_algorithm": "sha256(sequence(doc_id_bytes16 || n_tokens_le_u32))",
        "output_bytes": out_bytes,
        "elapsed_sec": round(elapsed, 3),
        "rows_per_sec": round(expected_old_rows / max(elapsed, 1e-9), 1),
        "s3_lookup_fp": contract["s3_lookup_fingerprint"],
        "processing_contract": contract,
        "token_cache_reuse": {
            "enabled": True,
            "reused_rows": kept,
            "newly_tokenized_rows": 0,
            "exact_audited_rows": 0,
        },
        "incremental_filter": {
            "schema_version": 1,
            "method": "prefix_to_full_decontam_only_adds_drops",
            "old_rows": expected_old_rows,
            "newly_dropped_rows": newly_dropped,
            "s3_text_read": False,
            "proof": "full-document shingle set is a superset of the former prefix",
        },
        "errors": {},
        "skipped": False,
    }
    renewed = _renew_token_cache_contract(
        previous_summary, out_path, contract["tokenizer_sha256"]
    )
    if renewed is not None:
        renewed.update({
            "context_lengths": list(contract["context_lengths"]),
            "rows": kept,
            "total_tokens": total_tokens,
        })
        summary["token_count_cache_contract"] = renewed
    write_summary(out_summary, summary)
    return summary


def _apply_restrictive_lookup_delta(
    out_root: Path,
    source: str,
    relpath: str,
    domain: str,
    out_path: Path,
    out_summary: Path,
    previous_summary: dict[str, Any],
    contract: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    """Filter a certified legacy S5 table when near-dedup only adds drops.

    The old E-17 lookup was empty.  Current ndv2 therefore cannot rescue an S5
    row; it can only remove rows.  Decontam decisions are compact and let us
    account for ndv2/decontam overlap without reading the multi-terabyte S3 text
    column.  The retained table already carries exact certified token counts.
    """
    lookup = load_domain_drops(out_root, domain)
    decontam_drops = load_decontam_drops(out_root, source, relpath)
    decontam_overlap = 0
    if decontam_drops:
        decontam_ids = _ids_to_void(list(decontam_drops))
        decontam_overlap = int(np.count_nonzero(drops_for(decontam_ids, lookup) == 1))

    pf = pq.ParquetFile(out_path)
    expected_old_rows = int(previous_summary.get("out_rows") or 0)
    if int(pf.metadata.num_rows) != expected_old_rows:
        raise ValueError(f"legacy S5 cache row closure failed: {source}/{relpath}")
    tmp_path = out_path.with_name(out_path.name + f".delta-tmp-{os.getpid()}")
    writer = pq.ParquetWriter(tmp_path, S5_SCHEMA, compression="zstd")
    content_digest = hashlib.sha256()
    buckets: Counter[str] = Counter()
    bucket_tokens: Counter[str] = Counter()
    kept = 0
    dropped_from_old_keep = 0
    total_tokens = 0
    try:
        for batch in safe_iter_batches(pf, batch_size=32768):
            ids = _arrow_hex_ids_to_void(batch.column("doc_id"))
            keep_mask = drops_for(ids, lookup) != 1
            dropped_from_old_keep += int(len(keep_mask) - np.count_nonzero(keep_mask))
            if not np.any(keep_mask):
                continue
            kept_batch = batch.filter(pa.array(keep_mask))
            nts = kept_batch.column("n_tokens").to_numpy(
                zero_copy_only=False
            ).astype(np.uint32, copy=False)
            digest_rows = np.empty(
                len(nts), dtype=np.dtype([("doc_id", "V16"), ("n_tokens", "<u4")])
            )
            digest_rows["doc_id"] = _arrow_hex_ids_to_void(
                kept_batch.column("doc_id")
            )
            digest_rows["n_tokens"] = nts
            content_digest.update(digest_rows.tobytes())
            actual_buckets = kept_batch.column("length_bucket").to_numpy(
                zero_copy_only=False
            )
            for label in np.unique(actual_buckets):
                mask = actual_buckets == label
                count = int(np.count_nonzero(mask))
                buckets[str(label)] += count
                bucket_tokens[str(label)] += int(nts[mask].sum(dtype=np.uint64))
            total_tokens += int(nts.sum(dtype=np.uint64))
            kept += len(nts)
            writer.write_batch(kept_batch)
        writer.close()
        pf.close()
    except BaseException:
        try:
            writer.close()
        except Exception:
            pass
        tmp_path.unlink(missing_ok=True)
        raise

    try:
        old_decisions = previous_summary.get("decisions") or {}
        drop_s2 = int(old_decisions.get("drop_s2_exact_dup") or 0)
        old_decontam = int(old_decisions.get("drop_decontam") or 0)
        if decontam_overlap > old_decontam:
            raise ValueError(
                f"ndv2/decontam overlap exceeds old ledger: {source}/{relpath}"
            )
        decisions = {
            "drop_s2_exact_dup": drop_s2,
            "drop_s3_near_dup": dropped_from_old_keep + decontam_overlap,
            "drop_decontam": old_decontam - decontam_overlap,
            "keep": kept,
        }
        in_rows = int(previous_summary.get("in_rows") or 0)
        if sum(decisions.values()) != in_rows:
            raise ValueError(
                f"restrictive ndv2 decision ledger does not close: {source}/{relpath}"
            )
        out_bytes = tmp_path.stat().st_size
        os.replace(tmp_path, out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    elapsed = time.monotonic() - started
    summary = {
        "stage": "s5",
        "source": source,
        "relpath": relpath,
        "domain": domain,
        "in_rows": in_rows,
        "out_rows": kept,
        "total_tokens": total_tokens,
        "decisions": decisions,
        "buckets": dict(buckets),
        "bucket_tokens": dict(bucket_tokens),
        "content_sha256": content_digest.hexdigest(),
        "content_hash_algorithm": "sha256(sequence(doc_id_bytes16 || n_tokens_le_u32))",
        "output_bytes": out_bytes,
        "elapsed_sec": round(elapsed, 3),
        "rows_per_sec": round(expected_old_rows / max(elapsed, 1e-9), 1),
        "s3_lookup_fp": contract["s3_lookup_fingerprint"],
        "processing_contract": contract,
        "token_cache_reuse": {
            "enabled": True,
            "reused_rows": kept,
            "newly_tokenized_rows": 0,
            "exact_audited_rows": 0,
        },
        "incremental_filter": {
            "schema_version": 1,
            "method": "empty_legacy_lookup_to_restrictive_ndv2",
            "old_rows": expected_old_rows,
            "newly_dropped_rows": dropped_from_old_keep,
            "ndv2_decontam_overlap": decontam_overlap,
            "s3_text_read": False,
        },
        "errors": {},
        "skipped": False,
    }
    renewed = _renew_token_cache_contract(
        previous_summary, out_path, contract["tokenizer_sha256"]
    )
    if renewed is not None:
        renewed.update({
            "context_lengths": list(contract["context_lengths"]),
            "rows": kept,
            "total_tokens": total_tokens,
        })
        summary["token_count_cache_contract"] = renewed
    write_summary(out_summary, summary)
    return summary


def _inspect_s5_output(
    out_path: Path,
    source: str,
    relpath: str,
    context_lengths: list[int],
    *,
    collect_ids: bool = False,
) -> dict[str, Any]:
    """Validate compact S5 structure and compute its semantic identity."""
    pf = pq.ParquetFile(out_path)
    if pf.schema_arrow.names != S5_SCHEMA.names:
        raise ValueError(f"S5 schema mismatch for {source}/{relpath}: {pf.schema_arrow}")
    rows = 0
    total_tokens = 0
    buckets: Counter[str] = Counter()
    bucket_tokens: Counter[str] = Counter()
    digest = hashlib.sha256()
    id_chunks: list[np.ndarray] = []
    boundaries = np.asarray(context_lengths, dtype=np.uint32)
    bucket_labels = np.asarray(
        [f"B{value // 1024}K" for value in context_lengths]
        + [f"gt_{context_lengths[-1] // 1024}K"],
        dtype=object,
    )
    eligible_labels = bucket_labels[:-1]
    for batch in safe_iter_batches(
        pf, batch_size=32768,
        columns=["doc_id", "n_tokens", "length_bucket", "mix_eligibility"],
    ):
        ids = batch.column("doc_id")
        ids_v = _arrow_hex_ids_to_void(ids)
        if collect_ids:
            id_chunks.append(ids_v.copy())
        nts = batch.column("n_tokens").to_numpy(zero_copy_only=False).astype(
            np.uint32, copy=False
        )
        bucket_idx = np.searchsorted(boundaries, nts, side="left")
        actual_buckets = batch.column("length_bucket").to_numpy(zero_copy_only=False)
        if not np.array_equal(actual_buckets, bucket_labels[bucket_idx]):
            raise ValueError(f"S5 bucket mismatch: {source}/{relpath}")
        eligibility = batch.column("mix_eligibility")
        expected_lengths = np.maximum(len(context_lengths) - bucket_idx, 0)
        actual_offsets = eligibility.offsets.to_numpy(zero_copy_only=False)
        if not np.array_equal(np.diff(actual_offsets), expected_lengths):
            raise ValueError(f"S5 eligibility mismatch: {source}/{relpath}")
        flat = eligibility.flatten().to_numpy(zero_copy_only=False)
        if len(flat):
            row_index = np.repeat(np.arange(len(nts)), expected_lengths)
            within_row = np.arange(len(flat)) - np.repeat(
                np.cumsum(expected_lengths) - expected_lengths,
                expected_lengths,
            )
            expected_flat = eligible_labels[bucket_idx[row_index] + within_row]
            if not np.array_equal(flat, expected_flat):
                raise ValueError(f"S5 eligibility mismatch: {source}/{relpath}")
        digest_rows = np.empty(
            len(nts), dtype=np.dtype([("doc_id", "V16"), ("n_tokens", "<u4")])
        )
        digest_rows["doc_id"] = ids_v
        digest_rows["n_tokens"] = nts
        digest.update(digest_rows.tobytes())
        rows += len(nts)
        total_tokens += int(nts.sum(dtype=np.uint64))
        counts = np.bincount(bucket_idx, minlength=len(bucket_labels))
        for idx, count in enumerate(counts):
            if count:
                label = str(bucket_labels[idx])
                buckets[label] += int(count)
                bucket_tokens[label] += int(nts[bucket_idx == idx].sum(dtype=np.uint64))
    return {
        "rows": rows,
        "total_tokens": total_tokens,
        "buckets": dict(buckets),
        "bucket_tokens": dict(bucket_tokens),
        "content_sha256": digest.hexdigest(),
        "doc_ids": (
            np.concatenate(id_chunks) if id_chunks else np.empty(0, dtype="V16")
        ) if collect_ids else None,
    }


def certify_legacy_s5_output(
    out_root: Path,
    source: str,
    relpath: str,
    domain: str,
    context_lengths: list[int],
    tokenizer_path: Path,
    *,
    promote_processing_contract: bool = False,
    audit_sha256: str | None = None,
) -> dict[str, Any]:
    """Add exact structural/content lineage to a legacy S5 shard.

    This reads only the compact S5 parquet.  It proves row/count/bucket/content
    closure and binds the token-count cache to an immutable tokenizer.  Promotion
    to the current processing contract is allowed only when the caller also
    provides a dataset-level token audit identity.
    """
    out_path, summary_path = s5_output_paths(out_root, source, relpath)
    if not out_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"legacy S5 output incomplete: {source}/{relpath}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("errors") or summary.get("shard_fatal"):
        raise ValueError(f"legacy S5 summary has errors: {source}/{relpath}")
    if promote_processing_contract and not audit_sha256:
        raise ValueError("processing-contract promotion requires audit_sha256")

    tokenizer_sha256 = _sha256(tokenizer_path)
    output_fp = fingerprint(out_path)
    existing = summary.get("token_count_cache_contract") or {}
    expected_processing_contract = _processing_contract(
        out_root, source, relpath, domain, context_lengths, tokenizer_sha256
    )
    if (
        existing.get("output_fingerprint") == output_fp
        and existing.get("tokenizer_sha256") == tokenizer_sha256
        and existing.get("context_lengths") == list(context_lengths)
        and existing.get("structural_validation_passed")
        and existing.get("dataset_token_audit_sha256") == audit_sha256
        and (
            not promote_processing_contract
            or summary.get("processing_contract") == expected_processing_contract
        )
    ):
        result = dict(summary)
        result["skipped"] = True
        return result

    if tokenizer_path.stat().st_mtime_ns > out_path.stat().st_mtime_ns:
        raise ValueError(
            f"tokenizer is newer than legacy S5 output; cannot certify cache: {source}/{relpath}"
        )
    inspected = _inspect_s5_output(out_path, source, relpath, context_lengths)
    rows = int(inspected["rows"])
    total_tokens = int(inspected["total_tokens"])
    buckets = inspected["buckets"]
    bucket_tokens = inspected["bucket_tokens"]

    decisions = Counter({k: int(v) for k, v in (summary.get("decisions") or {}).items()})
    expected_in = int(summary.get("in_rows") or 0)
    expected_out = int(summary.get("out_rows") or 0)
    if sum(decisions.values()) != expected_in:
        raise ValueError(f"legacy S5 decision ledger does not close: {source}/{relpath}")
    if decisions.get("keep", 0) != expected_out or rows != expected_out:
        raise ValueError(f"legacy S5 output row count does not close: {source}/{relpath}")
    if total_tokens != int(summary.get("total_tokens") or 0):
        raise ValueError(f"legacy S5 token total does not close: {source}/{relpath}")
    if buckets != (summary.get("buckets") or {}):
        raise ValueError(f"legacy S5 bucket ledger does not close: {source}/{relpath}")

    summary.update(
        {
            "bucket_tokens": bucket_tokens,
            "content_sha256": inspected["content_sha256"],
            "content_hash_algorithm": "sha256(sequence(doc_id_bytes16 || n_tokens_le_u32))",
            "output_bytes": out_path.stat().st_size,
            "token_count_cache_contract": {
                "schema_version": 1,
                "output_fingerprint": output_fp,
                "tokenizer_sha256": tokenizer_sha256,
                "tokenizer_mtime_precedes_output": True,
                "context_lengths": list(context_lengths),
                "rows": rows,
                "total_tokens": total_tokens,
                "structural_validation_passed": True,
                "dataset_token_audit_sha256": audit_sha256,
            },
            "skipped": False,
        }
    )
    if promote_processing_contract:
        summary["processing_contract"] = expected_processing_contract
        summary["lineage_migration"] = {
            "schema_version": 1,
            "method": "legacy_s5_structural_scan",
            "dataset_token_audit_sha256": audit_sha256,
        }
    write_summary(summary_path, summary)
    return summary


def recover_s5_summary_from_certified_output(
    out_root: Path,
    source: str,
    relpath: str,
    domain: str,
    context_lengths: list[int],
    tokenizer_path: Path,
    *,
    audit_sha256: str,
) -> dict[str, Any]:
    """Recover a damaged summary without trusting or re-tokenizing S3 text.

    The compact table is accepted only after its ordered doc-id stream exactly
    equals a fresh evaluation of every current S2/S3/decontam decision.  Token
    counts are then structurally revalidated and rebound to the immutable
    tokenizer plus the dataset-level exact token audit.
    """
    if not audit_sha256:
        raise ValueError("S5 summary recovery requires dataset token audit sha256")
    out_path, summary_path = s5_output_paths(out_root, source, relpath)
    if not out_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"S5 recovery input incomplete: {source}/{relpath}")
    damaged = json.loads(summary_path.read_text(encoding="utf-8"))
    if not damaged.get("errors") and not damaged.get("shard_fatal"):
        raise ValueError(f"S5 summary is not damaged: {source}/{relpath}")

    started = time.monotonic()
    inspected = _inspect_s5_output(
        out_path, source, relpath, context_lengths, collect_ids=True
    )
    actual_ids = inspected.pop("doc_ids")
    s3_path, _ = s3_output_paths(out_root, source, relpath)
    s3_lookup = load_domain_drops(out_root, domain)
    decontam_drops = load_decontam_drops(out_root, source, relpath)
    decisions: Counter[str] = Counter()
    in_rows = 0
    out_offset = 0
    pf = pq.ParquetFile(s3_path)
    for batch in safe_iter_batches(
        pf, batch_size=65536, columns=["doc_id", "s2_decision"]
    ):
        ids = batch.column("doc_id")
        ids_v = _arrow_hex_ids_to_void(ids)
        s2_values = batch.column("s2_decision").to_pylist()
        s2_keep = np.fromiter(
            (value == "keep" for value in s2_values), dtype=bool, count=len(ids_v)
        )
        s3_drop = drops_for(ids_v, s3_lookup) == 1
        ids_py = ids.to_pylist()
        decontam_drop = np.fromiter(
            (doc_id in decontam_drops for doc_id in ids_py),
            dtype=bool,
            count=len(ids_v),
        )
        keep = s2_keep & ~s3_drop & ~decontam_drop
        decisions["drop_s2_exact_dup"] += int(np.count_nonzero(~s2_keep))
        decisions["drop_s3_near_dup"] += int(np.count_nonzero(s2_keep & s3_drop))
        decisions["drop_decontam"] += int(
            np.count_nonzero(s2_keep & ~s3_drop & decontam_drop)
        )
        expected = ids_v[keep]
        end = out_offset + len(expected)
        if end > len(actual_ids) or not np.array_equal(expected, actual_ids[out_offset:end]):
            raise ValueError(
                f"S5 recovery keep-set mismatch at output row {out_offset}: "
                f"{source}/{relpath}"
            )
        out_offset = end
        in_rows += len(ids_v)
    if out_offset != len(actual_ids):
        raise ValueError(
            f"S5 recovery output has {len(actual_ids) - out_offset} extra rows: "
            f"{source}/{relpath}"
        )
    decisions["keep"] = out_offset
    if sum(decisions.values()) != in_rows:
        raise ValueError(f"S5 recovery decision ledger does not close: {source}/{relpath}")
    decision_payload = {key: int(value) for key, value in decisions.items() if value}

    tokenizer_sha256 = _sha256(tokenizer_path)
    contract = _processing_contract(
        out_root, source, relpath, domain, context_lengths, tokenizer_sha256
    )
    elapsed = time.monotonic() - started
    summary = {
        "stage": "s5",
        "source": source,
        "relpath": relpath,
        "domain": domain,
        "in_rows": in_rows,
        "out_rows": out_offset,
        "total_tokens": int(inspected["total_tokens"]),
        "decisions": decision_payload,
        "buckets": inspected["buckets"],
        "bucket_tokens": inspected["bucket_tokens"],
        "content_sha256": inspected["content_sha256"],
        "content_hash_algorithm": "sha256(sequence(doc_id_bytes16 || n_tokens_le_u32))",
        "output_bytes": out_path.stat().st_size,
        "elapsed_sec": round(elapsed, 3),
        "rows_per_sec": round(in_rows / max(elapsed, 1e-9), 1),
        "s3_lookup_fp": contract["s3_lookup_fingerprint"],
        "processing_contract": contract,
        "token_cache_reuse": {
            "enabled": True,
            "reused_rows": out_offset,
            "newly_tokenized_rows": 0,
            "exact_audited_rows": 0,
        },
        "token_count_cache_contract": {
            "schema_version": 1,
            "output_fingerprint": fingerprint(out_path),
            "tokenizer_sha256": tokenizer_sha256,
            "tokenizer_mtime_precedes_output": (
                tokenizer_path.stat().st_mtime_ns <= out_path.stat().st_mtime_ns
            ),
            "context_lengths": list(context_lengths),
            "rows": out_offset,
            "total_tokens": int(inspected["total_tokens"]),
            "structural_validation_passed": True,
            "dataset_token_audit_sha256": audit_sha256,
        },
        "lineage_recovery": {
            "schema_version": 1,
            "method": "current_keep_set_and_certified_compact_cache",
            "ordered_keep_set_verified": True,
            "s3_text_read": False,
            "dataset_token_audit_sha256": audit_sha256,
            "replaced_error_message": damaged.get("error_message"),
        },
        "errors": {},
        "skipped": False,
    }
    write_summary(summary_path, summary)
    return summary


def s5_global_input_contract(
    out_root: Path, tokenizer_path: Path, context_config: Path | None = None
) -> dict[str, Any]:
    """Small control-plane identity used by orchestration to reject stale S5."""
    controls = [
        out_root / "stage_decontam_global_summary.json",
        *[out_root / "s3_decisions" / f"{domain}.summary.json" for domain in ("code", "web")],
        *[
            out_root / "s3_v2" / f"{domain}_minhash_precision_audit.json"
            for domain in ("code", "web")
        ],
    ]
    return {
        "schema_version": 2,
        "control_sha256": {
            path.relative_to(out_root).as_posix(): (
                _sha256(path) if path.is_file() else "missing"
            )
            for path in controls
        },
        "tokenizer_sha256": _sha256(tokenizer_path),
        "context_config_sha256": _sha256(context_config or DEFAULT_CLEANING_CONFIG),
        "lookup_fingerprint": {
            domain: _lookup_fingerprint(out_root, domain) for domain in ("code", "web")
        },
    }


def _processing_contract(
    out_root: Path,
    source: str,
    relpath: str,
    domain: str,
    context_lengths: list[int],
    tokenizer_sha256: str,
) -> dict[str, Any]:
    """Identity of every input that can change S5 decisions or token counts."""
    from .decontam import decontam_output_paths

    s3_path, _ = s3_output_paths(out_root, source, relpath)
    decontam_path, decontam_summary = decontam_output_paths(out_root, source, relpath)
    return {
        "schema_version": 2,
        "s3_fingerprint": fingerprint(s3_path) if s3_path.is_file() else "missing",
        "decontam_fingerprint": (
            fingerprint(decontam_path) if decontam_path.is_file() else "missing"
        ),
        "decontam_summary_sha256": (
            _sha256(decontam_summary) if decontam_summary.is_file() else "missing"
        ),
        "s3_lookup_fingerprint": _lookup_fingerprint(out_root, domain),
        "tokenizer_sha256": tokenizer_sha256,
        "context_lengths": list(context_lengths),
    }


def process_shard_s5(
    out_root: Path,
    source: str,
    relpath: str,
    domain: str,
    context_lengths: list[int],
    tokenizer: Any | None = None,
    tokenizer_sha256: str | None = None,
) -> dict[str, Any]:
    """单 s3 分片 → s5 计长分桶分片（全部输入身份一致时才幂等 skip）。"""
    s3_path, _ = s3_output_paths(out_root, source, relpath)
    out_path, out_summary = s5_output_paths(out_root, source, relpath)
    if tokenizer_sha256 is None:
        tokenizer_sha256 = _sha256(resolve_tokenizer_path())
    contract = _processing_contract(
        out_root, source, relpath, domain, context_lengths, tokenizer_sha256
    )
    previous_summary: dict[str, Any] = {}
    if out_summary.is_file() and out_path.is_file():
        previous_summary = json.loads(out_summary.read_text("utf-8"))
        if (
            previous_summary.get("processing_contract") == contract
            and not previous_summary.get("errors")
            and int(previous_summary.get("output_bytes") or -1) == out_path.stat().st_size
            and bool(previous_summary.get("content_sha256"))
        ):
            previous_summary["skipped"] = True
            return previous_summary
        log.debug("s5 shard %s/%s: processing contract changed, recomputing", source, relpath)

    started = time.monotonic()
    token_cache = (
        _legacy_token_cache(out_path, previous_summary, tokenizer_sha256)
        if previous_summary and out_path.is_file()
        else None
    )
    if _can_apply_fullscan_decontam_delta(
        out_root, source, relpath, previous_summary, contract, token_cache
    ):
        return _apply_fullscan_decontam_delta(
            out_root,
            source,
            relpath,
            domain,
            out_path,
            out_summary,
            previous_summary,
            contract,
            started,
        )
    if _can_apply_restrictive_lookup_delta(domain, previous_summary, token_cache):
        return _apply_restrictive_lookup_delta(
            out_root,
            source,
            relpath,
            domain,
            out_path,
            out_summary,
            previous_summary,
            contract,
            started,
        )
    if tokenizer is None:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(resolve_tokenizer_path()))

    in_rows = 0
    kept = 0
    total_tokens = 0
    decisions: Counter[str] = Counter()
    buckets: Counter[str] = Counter()
    bucket_tokens: Counter[str] = Counter()
    content_digest = hashlib.sha256()
    cache_reused = 0
    cache_misses = 0
    cache_exact_audited = 0
    writer = BatchWriter(out_path, S5_SCHEMA)
    try:
        s3_drops = load_domain_drops(out_root, domain)
        decontam_drops = load_decontam_drops(out_root, source, relpath)
        pf = pq.ParquetFile(s3_path)
        md = pf.metadata
        total_bytes = sum(md.row_group(i).total_byte_size for i in range(md.num_row_groups))
        avg_row_bytes = max(1, total_bytes // max(md.num_rows, 1))
        read_batch = max(32, min(ENCODE_BATCH, READ_CHAR_CAP // avg_row_bytes))
        for batch in safe_iter_batches(pf, batch_size=read_batch):
            rows = batch.to_pydict()
            n = len(rows["doc_id"])
            if n == 0:
                continue
            in_rows += n
            ids_v = _ids_to_void(rows["doc_id"])
            s3_codes = drops_for(ids_v, s3_drops)
            sel_pos: list[int] = []
            cached_counts: dict[int, int] = {}
            for i in range(n):
                cached = token_cache.consume_if(rows["doc_id"][i]) if token_cache else None
                if rows["s2_decision"][i] != "keep":
                    decisions["drop_s2_exact_dup"] += 1
                elif int(s3_codes[i]) == 1:
                    decisions["drop_s3_near_dup"] += 1
                elif rows["doc_id"][i] in decontam_drops:
                    decisions["drop_decontam"] += 1
                else:
                    sel_pos.append(i)
                    if cached is not None:
                        cached_counts[i] = cached
            encode_pos = [i for i in sel_pos if i not in cached_counts]
            audit_pos = [
                i for i in sel_pos if i in cached_counts
            ][: max(0, TOKEN_CACHE_AUDIT_ROWS - cache_exact_audited)]
            exact_pos = encode_pos + audit_pos
            exact_counts = _encode_counts(
                tokenizer,
                [
                    rows["text"][i] if isinstance(rows["text"][i], str) else ""
                    for i in exact_pos
                ],
            )
            exact = dict(zip(exact_pos, exact_counts))
            for i in audit_pos:
                if int(exact[i]) != int(cached_counts[i]):
                    raise ValueError(
                        f"legacy token cache mismatch for {source}/{relpath} doc_id={rows['doc_id'][i]}: "
                        f"cached={cached_counts[i]} current={exact[i]}"
                    )
            cache_exact_audited += len(audit_pos)
            counts = []
            for i in sel_pos:
                if i in cached_counts:
                    counts.append(cached_counts[i])
                    cache_reused += 1
                else:
                    counts.append(exact[i])
                    cache_misses += 1
            doc_ids: list[str] = []
            nt_col: list[int] = []
            bucket_col: list[str] = []
            elig_col: list[list[str]] = []
            for i, c in zip(sel_pos, counts):
                nt = min(int(c), 0xFFFFFFFF)
                doc_ids.append(rows["doc_id"][i])
                nt_col.append(nt)
                total_tokens += nt
                bucket = assign_context_bucket(nt, context_lengths)
                buckets[bucket] += 1
                bucket_tokens[bucket] += nt
                bucket_col.append(bucket)
                elig_col.append(eligibility_flags(nt, context_lengths))
            if doc_ids:
                # Stable semantic digest of the S5 rows. Interleaved fixed-width
                # records make it independent of parquet row-group boundaries.
                digest_rows = np.empty(
                    len(doc_ids), dtype=np.dtype([("doc_id", "V16"), ("n_tokens", "<u4")])
                )
                digest_rows["doc_id"] = _ids_to_void(doc_ids)
                digest_rows["n_tokens"] = np.asarray(nt_col, dtype="<u4")
                content_digest.update(digest_rows.tobytes())
                writer.extend(
                    {
                        "doc_id": doc_ids,
                        "n_tokens": nt_col,
                        "length_bucket": bucket_col,
                        "mix_eligibility": elig_col,
                    }
                )
            kept += len(doc_ids)
            decisions["keep"] += len(doc_ids)
        if token_cache is not None and not token_cache.exhausted():
            raise ValueError(
                f"legacy token cache contains doc_ids absent or out of order in current S3: "
                f"{source}/{relpath}"
            )
        out_bytes = writer.close()
    except Exception as exc:
        writer.abort()
        summary = {
            "stage": "s5",
            "source": source,
            "relpath": relpath,
            "domain": domain,
            "in_rows": in_rows,
            "out_rows": 0,
            "total_tokens": 0,
            "decisions": dict(decisions),
            "buckets": {},
            "bucket_tokens": {},
            "content_sha256": None,
            "output_bytes": 0,
            "elapsed_sec": round(time.monotonic() - started, 3),
            "errors": {f"shard_error:{type(exc).__name__}": 1},
            "error_message": str(exc)[:500],
            "processing_contract": contract,
            "skipped": False,
        }
        write_summary(out_summary, summary)
        return summary

    elapsed = time.monotonic() - started
    summary = {
        "stage": "s5",
        "source": source,
        "relpath": relpath,
        "domain": domain,
        "in_rows": in_rows,
        "out_rows": kept,
        "total_tokens": total_tokens,
        "decisions": dict(decisions),
        "buckets": dict(buckets),
        "bucket_tokens": dict(bucket_tokens),
        "content_sha256": content_digest.hexdigest(),
        "content_hash_algorithm": "sha256(sequence(doc_id_bytes16 || n_tokens_le_u32))",
        "output_bytes": out_bytes,
        "elapsed_sec": round(elapsed, 3),
        "rows_per_sec": round(kept / max(elapsed, 1e-9), 1),
        "s3_lookup_fp": contract["s3_lookup_fingerprint"],
        "processing_contract": contract,
        "token_cache_reuse": {
            "enabled": token_cache is not None,
            "reused_rows": cache_reused,
            "newly_tokenized_rows": cache_misses,
            "exact_audited_rows": cache_exact_audited,
        },
        "errors": {},
        "skipped": False,
    }
    summary["token_count_cache_contract"] = {
        "schema_version": 2,
        "provenance": "current_pipeline_tokenization_or_certified_reuse",
        "output_fingerprint": fingerprint(out_path),
        "tokenizer_sha256": tokenizer_sha256,
        "context_lengths": list(context_lengths),
        "rows": kept,
        "total_tokens": total_tokens,
        "structural_validation_passed": True,
    }
    write_summary(out_summary, summary)
    return summary


# ---------------------------------------------------------------- driver 接入


def _init_worker(tok_path: str | None, out_root: str, ctx: list[int]) -> None:
    from tokenizers import Tokenizer

    resolved = resolve_tokenizer_path(tok_path)
    _WORKER["tok"] = Tokenizer.from_file(str(resolved))
    _WORKER["tokenizer_sha256"] = _sha256(resolved)
    _WORKER["out_root"] = Path(out_root)
    _WORKER["ctx"] = list(ctx)


def _run_shard(task: tuple[str, str, str]) -> dict[str, Any]:
    source, relpath, domain = task
    try:
        return process_shard_s5(
            _WORKER["out_root"], source, relpath, domain,
            _WORKER["ctx"], tokenizer=_WORKER["tok"],
            tokenizer_sha256=_WORKER["tokenizer_sha256"],
        )
    except Exception as exc:
        return {
            "stage": "s5",
            "source": source,
            "relpath": relpath,
            "shard_fatal": f"{type(exc).__name__}: {exc}",
        }


def s5_tasks(
    out_root: Path,
    profiles: dict[str, dict[str, Any]],
    sources: list[str] | None = None,
    limit: int = 0,
) -> list[tuple[str, str, str]]:
    """S5 任务集 = s3 产物分片（与 decontam 同一输入域）。"""
    tasks = []
    for source, prof in profiles.items():
        if sources and source not in sources:
            continue
        src_dir = out_root / "s3" / source
        if not src_dir.is_dir():
            continue
        for path in sorted(src_dir.rglob("*.parquet")):
            rel = path.relative_to(src_dir).as_posix()
            tasks.append((source, rel[: -len(".parquet")], prof["domain"]))
    tasks.sort()
    if limit:
        tasks = tasks[:limit]
    return tasks


def check_decontam_coverage(
    out_root: Path, tasks: list[tuple[str, str, str]]
) -> list[str]:
    from .decontam import (
        decontam_output_paths,
        load_decontam_config,
        validate_published_summary,
    )

    cfg = load_decontam_config()
    missing = []
    for source, relpath, _domain in tasks:
        out_path, summary_path = decontam_output_paths(out_root, source, relpath)
        if not (out_path.is_file() and summary_path.is_file()):
            missing.append(f"{source}/{relpath}")
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            s3_path, _ = s3_output_paths(out_root, source, relpath)
            if validate_published_summary(s3_path, out_path, summary, cfg, source):
                missing.append(f"{source}/{relpath}")
        except Exception:
            missing.append(f"{source}/{relpath}")
    return missing


def check_ndv2_done(out_root: Path, tasks: list[tuple[str, str, str]], allow_e17_web: bool = False, allow_e17_code: bool = False) -> list[str]:
    """Require the current near-dedup decision and matching exact-text audit.

    This prevents S5 from silently consuming an older placeholder, connected-
    component generation, or a lookup produced from a different verified edge
    set. Explicit E-17 flags remain available only for legacy experiment mixes.
    """
    from .dedup_minhash import s3_decisions_path

    domains = {domain for _s, _r, domain in tasks if domain in ("web", "code")}
    pending = []
    for domain in sorted(domains):
        if domain == "web" and allow_e17_web:
            continue
        if domain == "code" and allow_e17_code:
            continue
        summary_path = s3_decisions_path(out_root, domain).with_suffix(".summary.json")
        if not summary_path.is_file():
            pending.append(f"{domain}: s3_decisions summary 缺失")
            continue
        meta = json.loads(summary_path.read_text("utf-8"))
        if meta.get("policy") != "ndv2_minhash128_bbit4":
            pending.append(
                f"{domain}: policy={meta.get('policy')!r}，需要当前 ndv2 MinHash 决策"
            )
            continue
        contract = meta.get("decision_contract") or {}
        if contract.get("method") != "edge_dominance_external":
            pending.append(
                f"{domain}: decision_contract={contract.get('method')!r}，"
                "需要无传递级联的 edge_dominance_external 决策"
            )
            continue
        audit_path = out_root / "s3_v2" / f"{domain}_minhash_precision_audit.json"
        if not audit_path.is_file():
            pending.append(f"{domain}: MinHash 全文精度审计缺失")
            continue
        audit = json.loads(audit_path.read_text("utf-8"))
        if (
            not audit.get("passed")
            or audit.get("errors")
            or (audit.get("audit_contract") or {}).get("edges_fingerprint")
            != meta.get("edges_fingerprint")
        ):
            pending.append(f"{domain}: MinHash 全文精度审计未通过或与当前边集不匹配")
    return pending


def run_s5_scan(
    tasks: list[tuple[str, str, str]],
    tok_path: str | None,
    out_root: Path,
    ctx: list[int],
    workers: int,
    guard: tuple[Any, Any],
) -> list[dict[str, Any]]:
    beat, guard_stop = guard
    summaries: list[dict[str, Any]] = []
    started = time.monotonic()
    done = 0
    rows = 0
    tokens = 0
    try:
        with mp.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(tok_path, str(out_root), list(ctx)),
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
                rows += int(summary.get("out_rows") or 0)
                tokens += int(summary.get("total_tokens") or 0)
                if done % 50 == 0 or done == len(tasks):
                    elapsed = time.monotonic() - started
                    log.info(
                        "progress[s5]: %d/%d shards, %.2f shards/s, %.0f docs/s, %.2fM tokens/s",
                        done, len(tasks), done / max(elapsed, 1e-9),
                        rows / max(elapsed, 1e-9), tokens / max(elapsed, 1e-9) / 1e6,
                    )
    finally:
        guard_stop.set()
    return summaries


def _global_summary_payload(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    # E-16：shard_errors 置顶
    per_source: dict[str, Counter[str]] = {}
    per_domain: dict[str, Counter[str]] = {}
    decisions: Counter[str] = Counter()
    buckets: Counter[str] = Counter()
    bucket_tokens: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    for s in summaries:
        src = s.get("source", "?")
        dom = s.get("domain", "?")
        acc = per_source.setdefault(src, Counter())
        dacc = per_domain.setdefault(dom, Counter())
        for key in ("in_rows", "out_rows", "total_tokens", "output_bytes"):
            acc[key] += int(s.get(key) or 0)
            dacc[key] += int(s.get(key) or 0)
            totals[key] += int(s.get(key) or 0)
        acc["elapsed_sec"] += float(s.get("elapsed_sec") or 0.0)
        decisions.update(s.get("decisions") or {})
        buckets.update(s.get("buckets") or {})
        bucket_tokens.update(s.get("bucket_tokens") or {})
        for k, v in (s.get("decisions") or {}).items():
            dacc[f"decisions.{k}"] += int(v)
        for k, v in (s.get("buckets") or {}).items():
            dacc[f"bucket_docs.{k}"] += int(v)
        for k, v in (s.get("bucket_tokens") or {}).items():
            dacc[f"bucket_tokens.{k}"] += int(v)
    return {
        "stage": "s5",
        "shards": len(summaries),
        "shard_errors": sum(1 for s in summaries if s.get("errors") or s.get("shard_fatal")),
        "totals": dict(totals),
        "decisions": dict(decisions),
        "buckets": dict(sorted(buckets.items())),
        "bucket_tokens": dict(sorted(bucket_tokens.items())),
        "per_source": {k: dict(v) for k, v in sorted(per_source.items())},
        "per_domain": {k: dict(v) for k, v in sorted(per_domain.items())},
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    return path


def write_global_summary(out_root: Path, summaries: list[dict[str, Any]]) -> Path:
    payload = _global_summary_payload(summaries)
    path = out_root / "stage_s5_global_summary.json"
    return _write_json_atomic(path, payload)


def collect_published_summaries(
    out_root: Path,
    tasks: list[tuple[str, str, str]],
    *,
    context_lengths: list[int] | None = None,
    tokenizer_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load and validate the complete S5 publication set after targeted retries."""
    summaries: list[dict[str, Any]] = []
    coverage = {"expected": len(tasks), "complete": 0, "missing_summary": 0, "missing_output": 0}
    for source, relpath, domain in tasks:
        out_path, summary_path = s5_output_paths(out_root, source, relpath)
        if not summary_path.is_file():
            coverage["missing_summary"] += 1
            summaries.append({
                "stage": "s5", "source": source, "relpath": relpath, "domain": domain,
                "shard_fatal": "published summary missing",
            })
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            coverage["missing_summary"] += 1
            summaries.append({
                "stage": "s5", "source": source, "relpath": relpath, "domain": domain,
                "shard_fatal": f"published summary unreadable: {type(exc).__name__}: {exc}",
            })
            continue
        if not out_path.is_file():
            coverage["missing_output"] += 1
            summary = dict(summary)
            summary["shard_fatal"] = "published parquet missing"
        else:
            coverage["complete"] += 1
            if context_lengths is not None and tokenizer_sha256 is not None:
                expected = _processing_contract(
                    out_root,
                    source,
                    relpath,
                    domain,
                    context_lengths,
                    tokenizer_sha256,
                )
                if summary.get("processing_contract") != expected:
                    summary = dict(summary)
                    summary["shard_fatal"] = "published S5 processing contract is stale"
                elif (
                    int(summary.get("output_bytes") or -1) != out_path.stat().st_size
                    or not summary.get("content_sha256")
                ):
                    summary = dict(summary)
                    summary["shard_fatal"] = "published S5 output identity is incomplete or stale"
        summaries.append(summary)
    return summaries, coverage


def run_s5(
    out_root: Path,
    profiles: dict[str, dict[str, Any]],
    workers: int,
    tokenizer: str | Path | None = None,
    sources: list[str] | None = None,
    limit: int = 0,
    guard: tuple[Any, Any] | None = None,
    context_lengths: list[int] | None = None,
    allow_e17_web: bool = False,
    allow_e17_code: bool = False,
) -> Path:
    if not (out_root / "s3").is_dir():
        raise SystemExit(f"s5 依赖 s3 产物：{out_root / 's3'} 不存在，请先运行 --stage s3")
    if guard is None:
        import threading

        guard = (lambda: None, threading.Event())
    ctx = context_lengths or load_context_lengths()
    tok_path = resolve_tokenizer_path(tokenizer)
    tasks = s5_tasks(out_root, profiles, sources=sources, limit=limit)
    if not tasks:
        raise SystemExit("s5: no s3 shards found")
    decontam_global = out_root / "stage_decontam_global_summary.json"
    if not decontam_global.is_file():
        raise SystemExit(
            f"s5 硬前置：decontam 未覆盖（全局 summary 缺失：{decontam_global}）"
        )
    from .decontam import release_readiness_errors

    decontam_errors = release_readiness_errors(
        json.loads(decontam_global.read_text(encoding="utf-8"))
    )
    if decontam_errors:
        raise SystemExit(
            "s5 硬前置：decontam 未形成可消费的全文发布（"
            + "; ".join(decontam_errors)
            + "）"
        )
    missing = check_decontam_coverage(out_root, tasks)
    if missing:
        raise SystemExit(
            f"s5 硬前置：decontam 未覆盖全部 s3 输入分片（缺 {len(missing)}/{len(tasks)}，"
            f"如 {missing[:3]}）——先跑完 --stage decontam（global summary shard_errors=0）"
        )
    ndv2_pending = check_ndv2_done(out_root, tasks, allow_e17_web=allow_e17_web, allow_e17_code=allow_e17_code)
    if ndv2_pending:
        raise SystemExit(
            "s5 硬前置：ndv2 近重决定未落定（" + "; ".join(ndv2_pending) +
            "）——lookup 为空会把 web/code 近重全部当 keep，ndv2 后产物静默过期（E-23）。"
            "fast track 交付可 --allow-e17-web（web 域 FineWeb 立场，code 仍强制）"
        )
    start_contract = s5_global_input_contract(
        out_root, tok_path, DEFAULT_CLEANING_CONFIG
    )
    global_path = out_root / "stage_s5_global_summary.json"
    if global_path.is_file():
        try:
            previous_global = json.loads(global_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous_global = {}
        if previous_global.get("input_contract") != start_contract:
            # Revoke the old commit marker before mutating any shard. Readers
            # cannot mistake an in-progress mixed generation for a release.
            global_path.unlink(missing_ok=True)
    log.info("stage s5: %d shards, ctx=%s, tokenizer=%s", len(tasks), ctx, tok_path)
    summaries = run_s5_scan(tasks, str(tok_path), out_root, ctx, workers, guard)
    published, coverage = collect_published_summaries(
        out_root,
        s5_tasks(out_root, profiles, sources=None, limit=0),
        context_lengths=ctx,
        tokenizer_sha256=_sha256(tok_path),
    )
    final_contract = s5_global_input_contract(
        out_root, tok_path, DEFAULT_CLEANING_CONFIG
    )
    if final_contract != start_contract:
        global_path.unlink(missing_ok=True)
        raise SystemExit(
            "s5 upstream changed during processing; global publication was revoked. "
            "Rerun after upstream writers finish."
        )
    payload = _global_summary_payload(published)
    payload["coverage"] = coverage
    payload["input_contract"] = start_contract
    published_errors = int(payload.get("shard_errors") or 0)
    coverage_ok = (
        int(coverage.get("expected") or 0) > 0
        and int(coverage.get("complete") or 0) == int(coverage.get("expected") or 0)
        and int(coverage.get("missing_summary") or 0) == 0
        and int(coverage.get("missing_output") or 0) == 0
    )
    if published_errors or not coverage_ok:
        failed_path = out_root / "audits" / "stage_s5_last_failed_summary.json"
        _write_json_atomic(failed_path, payload)
        global_path.unlink(missing_ok=True)
        raise SystemExit(
            "stage s5 publication rejected: "
            f"shard_errors={published_errors}, coverage={coverage}; "
            f"diagnostics={failed_path}"
        )
    path = _write_json_atomic(global_path, payload)
    log.info("stage s5 global summary: %s", path)
    errors = sum(1 for s in summaries if s.get("errors") or s.get("shard_fatal"))
    if errors:
        log.warning("stage s5 finished with %d shard errors (rerun to retry)", errors)
    return path
