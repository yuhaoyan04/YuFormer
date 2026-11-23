"""S6 配比物化（2026-09-26 实现，E1 实验前最后一块代码）。

流程（driver `--stage s6 --mix-config <yaml>`）：
  硬门 fail-fast（缺一即拒绝物化，全部缺失一次性报告）：
    ① s4_scored 覆盖全部 s5 输入分片（composite 分来源）+ s4 全局 summary
      shard_errors=0 + 分位表已构建且覆盖 mix 领域；
    ② decontam 全部完成：全局 summary shard_errors=0 且逐分片产物覆盖 s3 输入集；
    ③ ndv2 web/code lookup 已发布：decision 必须是当前 b-bit MinHash policy、直接边
      dominance 策略，且全文精度审计与边集指纹一致；lookup 行数与决策账本闭合。
      S5 直接 join lookup，不重写 10TB S3 正文；
    ④ 每领域 held-out val 已切出（ensure_val_sets，文件级幂等；确定性 blake2b
      抽样，永不进训练）；
    ⑤ s5 产物覆盖全部 s3 输入分片（读入数据源自身的完整性前提）。
  配比：领域内配置指定的 score_field 降序（doc_id 字节序决胜，全向量化 lexsort，无 Python
  逐行排序）取 top-k（阈值由 s4 分位表给出：threshold = table[n-⌈top_k·n⌉]，
  同值同分入选 ⇒ 决胜无关），按候选中的 token 前缀截到领域配额（权重 × 总量目标，
  最大余数法 = training.allocate_mix_quotas）。
  正式顺序仅物化 stage2：全局按 doc_id 字节序排列。doc_id 本身是
  blake2b(source\0relpath\0idx) 均匀哈希，因此该顺序等价于可复现的无偏随机顺序。
  旧 upstream_score 两段 curriculum 仅在显式 experimental 配置下保留。
  val 切出（gate ④ 的实现）：两遍流式——pass1 按 blake2b(doc_id) 前 12 bit
  分 4096 桶累计 token 直方图定边界桶 B；pass2 桶 <B 全取、桶 =B 按全哈希排序
  取前缀到配额。确定性、流式、内存有界（直方图 4096×16B/域）。
  全局 summary：各领域/各源入选 token 数、drop 原因统计（s2 精确去重 / s3
  近重 / 去污染剔除量来自 s5 per-domain 决策；质量分位数截断量 = 候选 tokens −
  入选 tokens；配额缺口 = 配额 − 入选）。

工程约束沿用事故账本：文件级幂等（val 逐域、mix 按 config echo 整体）、
tmp+os.replace、失败任务只落 error summary 不落空产物（E-16）、停滞守卫
（guard_factory 由 driver 提供，每 phase 一个，结束必须 stop）、流式分片 +
numpy 向量化（E-12；doc_id 字符串↔V16/hex 全部向量化，无 Python 逐行热循环；
候选收集的 val 排除用整批 searchsorted）、候选 parts 落盘（mix_work/）驱动
进程归并（E-13：单域峰值 ≈ 76B/候选行 + lexsort 4×8B/行序键，2TB 机 OK）、
val ids bin memmap 全 worker 共享 page cache（E-14）。
"""
from __future__ import annotations

import hashlib
import json
import logging
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
import yaml

from ..training import allocate_mix_quotas
from .dedup_common import _hex_strings_to_void
from .length_bucket import (
    resolve_tokenizer_path,
    s5_global_input_contract,
    s5_output_paths,
    s5_tasks,
)
from .ingest import safe_iter_batches, write_summary
from .score import (
    effective_hybrid_v2,
    load_scorer_config,
    validate_selection_contract,
)

log = logging.getLogger("yuformer.scale.materialize")

NDV2_DOMAINS = ("web", "code")
E17_POLICY = "near_dup_clustering_skipped"
NDV2_POLICY = "ndv2_minhash128_bbit4"
NDV2_DECISION_METHOD = "edge_dominance_external"
VAL_BUCKETS = 4096
DEFAULT_VAL_TOKENS_PER_DOMAIN = 50_000_000
DEFAULT_VAL_MAX_FRACTION = 0.01  # 安全阀：微域池过小时 val 不超池 1%

VAL_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("doc_id", pa.large_string(), nullable=False),  # E-51: int64 offsets 需 large_string
        pa.field("domain", pa.string(), nullable=False),
        pa.field("n_tokens", pa.uint32(), nullable=False),
    ]
)
MEMBERS_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("doc_id", pa.large_string(), nullable=False),  # E-51: int64 offsets 需 large_string
        pa.field("domain", pa.string(), nullable=False),
        pa.field("composite", pa.float32(), nullable=False),
        pa.field("n_tokens", pa.uint32(), nullable=False),
    ]
)
ORDER_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("position", pa.int64(), nullable=False),
        pa.field("doc_id", pa.large_string(), nullable=False),  # E-51: int64 offsets 需 large_string
        pa.field("domain", pa.string(), nullable=False),
        pa.field("n_tokens", pa.uint32(), nullable=False),
        pa.field("upstream_score", pa.float64(), nullable=True),
    ]
)

_WORKER: dict[str, Any] = {}

MERGE_BYTES_PER_CANDIDATE = 120
MERGE_MEMORY_BUDGET = 500_000_000_000
S6_COLLECT_SCHEMA_VERSION = 4

# C4 research artifact. Keep historical tracks for ablations, while making the
# three signals used by the current scorer explicit hard requirements. Nullable
# historical/upstream tracks remain useful when their coverage is reported.
GOLDRULE_TRACK_COLUMNS = (
    "self_score",
    "composite",
    "upstream_score",
    "upstream_norm",
    "fasttext_score",
    "bert_score",
    "composite_v2",
    "rule_score",
    "rule_composite",
    "hybrid_composite",
    "llm_distill",
    "hybrid_v2",
)
GOLDRULE_REQUIRED_FINITE = ("rule_score", "llm_distill", "hybrid_v2")
GOLDRULE_TRACK_DTYPES = {
    name: (np.float64 if name == "upstream_score" else np.float32)
    for name in GOLDRULE_TRACK_COLUMNS
}


def _goldrule_hash_selected(hash64: int, sample_rate: float) -> bool:
    """Deterministic full-width Bernoulli threshold on an unsigned 64-bit hash."""
    if sample_rate <= 0.0:
        return False
    if sample_rate >= 1.0:
        return True
    return hash64 < int(sample_rate * (1 << 64))


def _prepare_collect_work_root(work_root: Path, contract: dict[str, Any]) -> bool:
    """Prepare a generation-bound resumable collect directory.

    Returns True only when an existing work generation is safe to resume.
    Any config, upstream, roster, or collector-schema change revokes all parts.
    """
    marker = work_root / "_generation.json"
    if marker.is_file():
        try:
            if json.loads(marker.read_text(encoding="utf-8")) == contract:
                return True
        except (OSError, json.JSONDecodeError):
            pass
    if work_root.is_dir():
        shutil.rmtree(work_root, ignore_errors=True)
    work_root.mkdir(parents=True, exist_ok=True)
    write_summary(marker, contract)
    return False


def _collect_done_path(work_root: Path, source: str, relpath: str, domain: str) -> Path:
    key = hashlib.sha256(
        f"{source}\0{relpath}\0{domain}".encode("utf-8")
    ).hexdigest()
    return work_root / "_done" / f"{key}.json"


def load_mix_config(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"mix config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"mix config not a mapping: {p}")
    for key in ("mix_name", "total_tokens", "domains"):
        if key not in raw:
            raise ValueError(f"mix config missing {key}: {p}")
    domains = raw["domains"]
    if not isinstance(domains, dict) or not domains:
        raise ValueError("mix config domains must be a non-empty mapping")
    for dom, spec in domains.items():
        if not isinstance(spec, dict) or "weight" not in spec:
            raise ValueError(f"mix config domain {dom}: missing weight")
        spec.setdefault("top_k", 1.0)
        if not (0.0 < float(spec["top_k"]) <= 1.0):
            raise ValueError(f"mix config domain {dom}: top_k must be in (0, 1]")
    quota_fill_policy = str(raw.get("quota_fill_policy", "strict_top_k"))
    if quota_fill_policy not in {"strict_top_k", "deepen_to_quota"}:
        raise ValueError(
            "quota_fill_policy must be 'strict_top_k' or 'deepen_to_quota'"
        )
    max_total_shortfall_tokens = int(
        raw.get(
            "max_total_shortfall_tokens",
            int(raw["total_tokens"]) if quota_fill_policy == "strict_top_k" else 0,
        )
    )
    if max_total_shortfall_tokens < 0:
        raise ValueError("max_total_shortfall_tokens must be >= 0")
    raw["quota_fill_policy"] = quota_fill_policy
    raw["max_total_shortfall_tokens"] = max_total_shortfall_tokens
    raw.setdefault("score_field", "composite_v2")
    raw.setdefault("seed", "yuformer")
    cur = raw.get("curriculum") or {}
    curriculum_enabled = bool(cur.get("enabled", False))
    curriculum_method = str(cur.get("method", "random"))
    curriculum_fraction = float(cur.get("stage1_fraction", 0.0))
    curriculum_experimental = bool(cur.get("experimental", False))
    if not (0.0 <= curriculum_fraction < 1.0):
        raise ValueError("curriculum.stage1_fraction must be in [0, 1)")
    if curriculum_enabled:
        if curriculum_method != "upstream_score_legacy" or not curriculum_experimental:
            raise ValueError(
                "curriculum is not a production default: the only implemented ordering is "
                "method='upstream_score_legacy' and it must be marked experimental=true"
            )
        if curriculum_fraction <= 0.0:
            raise ValueError("enabled curriculum requires stage1_fraction > 0")
    elif curriculum_method != "random" or curriculum_fraction != 0.0:
        raise ValueError(
            "disabled curriculum must declare method='random' and stage1_fraction=0.0"
        )
    raw["curriculum"] = {
        "enabled": curriculum_enabled,
        "method": curriculum_method,
        "stage1_fraction": curriculum_fraction,
        "experimental": curriculum_experimental,
    }
    val = raw.get("val") or {}
    raw["val"] = {
        "tokens_per_domain": int(val.get("tokens_per_domain", DEFAULT_VAL_TOKENS_PER_DOMAIN)),
        "max_fraction": float(val.get("max_fraction", DEFAULT_VAL_MAX_FRACTION)),
    }
    goldrule = raw.get("goldrule") or {}
    raw["goldrule"] = {
        "per_domain_quota": int(goldrule.get("per_domain_quota", 0)),
        "sample_rate": float(goldrule.get("sample_rate", 0.0)),
        "min_total_rows": int(goldrule.get("min_total_rows", 0)),
    }
    if raw["goldrule"]["per_domain_quota"] < 0 or raw["goldrule"]["min_total_rows"] < 0:
        raise ValueError("goldrule quotas must be >= 0")
    if not 0.0 <= raw["goldrule"]["sample_rate"] <= 1.0:
        raise ValueError("goldrule.sample_rate must be in [0, 1]")
    if raw["goldrule"]["min_total_rows"] and (
        not raw["goldrule"]["per_domain_quota"] or not raw["goldrule"]["sample_rate"]
    ):
        raise ValueError(
            "goldrule.min_total_rows requires positive per_domain_quota and sample_rate"
        )
    included_sources = raw.get("included_sources")
    if included_sources is not None:
        if (
            not isinstance(included_sources, list)
            or not included_sources
            or any(not isinstance(source, str) or not source for source in included_sources)
            or len(set(included_sources)) != len(included_sources)
        ):
            raise ValueError("included_sources must be a non-empty list of unique source names")
        raw["included_sources"] = sorted(included_sources)
    raw["_config_path"] = str(p)
    return raw


def config_echo(cfg: dict[str, Any]) -> str:
    payload = {
        "mix_name": cfg["mix_name"],
        "total_tokens": int(cfg["total_tokens"]),
        "domains": cfg["domains"],
        "score_field": cfg["score_field"],
        "seed": cfg["seed"],
        "curriculum": cfg["curriculum"],
        "val": {k: v for k, v in cfg["val"].items()},
        "goldrule": cfg["goldrule"],
        "quota_fill_policy": cfg["quota_fill_policy"],
        "max_total_shortfall_tokens": int(cfg["max_total_shortfall_tokens"]),
        "included_sources": cfg.get("included_sources"),
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _resolve_release_sources(
    cfg: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
    requested: list[str] | None,
) -> list[str] | None:
    """Make a configured release roster authoritative and fail closed.

    A CLI source filter is useful for development runs, but allowing it to
    silently narrow a formal release would make the same config identity refer
    to different corpora.  Once ``included_sources`` is declared, callers must
    either omit the filter or provide the exact same set.
    """
    configured = cfg.get("included_sources")
    if configured is None:
        if requested is not None:
            unknown = sorted(set(requested) - set(profiles))
            if unknown:
                raise ValueError(f"unknown requested sources: {unknown}")
        return requested
    unknown = sorted(set(configured) - set(profiles))
    if unknown:
        raise ValueError(f"included_sources contains unknown sources: {unknown}")
    if requested is not None and set(requested) != set(configured):
        raise ValueError(
            "CLI/source filter must exactly match mix included_sources for a formal release"
        )
    return list(configured)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _published_artifact(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": stat.st_size,
        "sha256": _file_sha256(path),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
    }


def _published_artifacts_current(root: Path, rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    for row in rows:
        try:
            path = root / str(row["path"])
            stat = path.stat()
            if (stat.st_size, stat.st_dev, stat.st_ino, stat.st_mtime_ns) != (
                int(row["bytes"]), int(row["device"]), int(row["inode"]),
                int(row["mtime_ns"]),
            ):
                return False
        except (OSError, KeyError, TypeError, ValueError):
            return False
    return True


def _stat_snapshot(paths: list[Path], root: Path) -> dict[str, Any]:
    """大产物身份快照。

    JSON/NPY 控制产物取内容哈希；TB 级 parquet/bin 绑定相对路径与大小，其内容身份由
    同快照中的逐分片 summary 内容哈希承接。该口径跨目录复制保持稳定。
    """
    h = hashlib.sha256()
    count = 0
    total_bytes = 0
    for path in sorted(set(paths)):
        if not path.is_file():
            try:
                missing_name = path.relative_to(root).as_posix()
            except ValueError:
                missing_name = str(path)
            h.update(f"missing\0{missing_name}".encode("utf-8"))
            continue
        st = path.stat()
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            rel = str(path)
        content_hash = _file_sha256(path) if path.suffix in (".json", ".npy") else "size-only"
        h.update(f"{rel}\0{st.st_size}\0{content_hash}\n".encode("utf-8"))
        count += 1
        total_bytes += st.st_size
    return {
        "algorithm": "sha256(relative_path\\0size\\0json_or_npy_content_sha256)",
        "files": count,
        "bytes": total_bytes,
        "sha256": h.hexdigest(),
    }


def _s5_input_snapshot(
    out_root: Path, tasks: list[tuple[str, str, str]]
) -> dict[str, Any]:
    paths: list[Path] = []
    for source, relpath, _domain in tasks:
        par, summary = s5_output_paths(out_root, source, relpath)
        paths.extend((par, summary))
    return _stat_snapshot(paths, out_root)


def _s6_upstream_snapshot(
    out_root: Path, tasks: list[tuple[str, str, str]], score_field: str
) -> dict[str, Any]:
    """绑定 S6 实际读取的数据文件身份和小型控制产物内容哈希。"""
    data_paths: list[Path] = []
    for source, relpath, _domain in tasks:
        s5_par, s5_summary = s5_output_paths(out_root, source, relpath)
        data_paths.extend((s5_par, s5_summary))
        data_paths.extend((
            out_root / "s4_scored" / source / f"{relpath}.parquet",
            out_root / "s4_scored" / source / f"{relpath}.summary.json",
        ))
    data_paths.extend((out_root / "val_sets").glob("*.parquet"))
    data_paths.extend((out_root / "val_sets").glob("*.ids.bin"))
    # Bind only the score table this mix actually reads.  Hashing every historic
    # score track made an S6 start stream 121 GB of unrelated immutable NPY files.
    suffix = {
        "composite_v2": "",
        "rule_composite": ".rule",
        "hybrid_composite": ".hybrid",
        "hybrid_v2": ".hybrid2",
    }[score_field]
    tables_root = out_root / "s4_quantiles" / "tables"
    quantile_summary = tables_root / f"_build{suffix}.summary.json"
    try:
        quantile_payload = json.loads(quantile_summary.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data_paths.append(quantile_summary)
    else:
        for info in (quantile_payload.get("domains") or {}).values():
            data_paths.append(tables_root / str(info.get("npy") or ""))
    active_domains = {domain for _source, _relpath, domain in tasks}
    for domain in NDV2_DOMAINS:
        if domain not in active_domains:
            continue
        lookup_root = out_root / "s3_lookup" / domain
        data_paths.extend(
            (lookup_root / "lookup.ids.bin", lookup_root / "lookup.codes.bin")
        )

    control_paths = [
        out_root / "stage_s4_global_summary.json",
        out_root / "stage_s5_global_summary.json",
        out_root / "stage_decontam_global_summary.json",
        out_root / "stage_ndv2_global_summary.json",
        out_root / "val_sets" / "_val_global_summary.json",
    ]
    for domain in NDV2_DOMAINS:
        if domain not in active_domains:
            continue
        control_paths.extend(
            (
                out_root / "s3_decisions" / f"{domain}.summary.json",
                out_root / "s3_v2" / f"{domain}_minhash_precision_audit.json",
            )
        )
    control_paths.append(quantile_summary)
    control_paths.extend((out_root / "val_sets").glob("*.summary.json"))
    control_sha256 = {
        p.relative_to(out_root).as_posix(): _file_sha256(p)
        for p in sorted(set(control_paths)) if p.is_file()
    }
    payload: dict[str, Any] = {
        "data_artifacts": _stat_snapshot(data_paths, out_root),
        "control_sha256": control_sha256,
    }
    payload["snapshot_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


# ---------------------------------------------------------------- 工具

def val_hash(doc_id: str) -> bytes:
    return hashlib.blake2b(doc_id.encode("ascii"), digest_size=16).digest()


def _hash_bucket(h: bytes) -> int:
    return int.from_bytes(h[:2], "big") >> 4  # 前 12 bit → 4096 桶


def _ids_be_keys(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """V16 doc_id → 字节序比较键（uint64 小端解释 ≠ 字节序，需按 8B 块 byteswap）。"""
    u = np.ascontiguousarray(ids).view(np.uint64).reshape(-1, 2)
    return u[:, 0].byteswap(), u[:, 1].byteswap()


def hexlify_ids(ids: np.ndarray) -> np.ndarray:
    """V16 → S32 hex ascii（向量化，无 Python 逐行 hex）。"""
    u = np.ascontiguousarray(ids).view(np.uint8).reshape(-1, 16)
    pairs = np.empty((len(u), 32), dtype=np.uint8)
    pairs[:, 0::2] = u >> 4
    pairs[:, 1::2] = u & 15
    ascii_arr = pairs + np.where(pairs < 10, np.uint8(48), np.uint8(87))
    return np.ascontiguousarray(ascii_arr).view("S32").ravel()


DOC_ID_BUILD_CHUNK = 200_000_000  # E-73：单块上限（2 亿行实测通过）；大表分块构建降峰值内存


def _doc_id_chunked(ids: np.ndarray) -> pa.ChunkedArray:
    """E-73：大表 doc_id 列分块构建——mix_m2 选中 ~15 亿行时整列构建在内存紧张机器上
    触发了未能定位的规模相关损坏（ArrowInvalid: Negative offsets，2 亿行合成验证通过）。
    分块 = 每块走已验证路径 + 块级校验 + 峰值内存有界；任何一块坏即带精确行号 fail-fast。"""
    if len(ids) <= DOC_ID_BUILD_CHUNK:
        return pa.chunked_array([_doc_id_array(ids)])
    chunks = []
    for v0 in range(0, len(ids), DOC_ID_BUILD_CHUNK):
        c = _doc_id_array(ids[v0:v0 + DOC_ID_BUILD_CHUNK])
        if c.type != pa.large_string():
            raise ValueError(f"doc_id chunk @{v0}: unexpected type {c.type}")
        chunks.append(c)
    return pa.chunked_array(chunks)


def _doc_id_array(ids: np.ndarray) -> pa.Array:
    """V16 → pa.string（from_buffers 零 Python 列表；hex ascii 必为合法 utf8）。"""
    ascii_flat = np.ascontiguousarray(hexlify_ids(ids).view(np.uint8).reshape(-1, 32)).reshape(-1)
    n = len(ids)
    # E-40：offsets 必须 int64——int32 在 n > 2^31/32 ≈ 6700 万行时乘 32 溢出为负，
    # pyarrow 报 ArrowInvalid: Negative offsets（2026-10-10 mix_m3_e1_100b ~2 亿行触发）。
    offsets = np.arange(0, n + 1, dtype=np.int64) * 32  # 每行 32B
    return pa.Array.from_buffers(
        pa.large_string(),  # E-51：int64 offsets 必须 large_string（pa.string 按 int32 解释
                            # int64 buffer → "Negative offsets"；写 parquet 时按行组切片回 int32，
                            # 列块 <2GB 即安全——与 row_group_size=50M 配套）
        n,
        [None, pa.py_buffer(np.ascontiguousarray(offsets)), pa.py_buffer(ascii_flat)],
    )


DOMAIN_BUILD_MAX_BYTES = 64 << 20
DOMAIN_BUILD_MAX_ROWS = 1_000_000


def _domain_array(codes: np.ndarray, dom_list: list[str]) -> pa.ChunkedArray:
    """Decode domains in bounded chunks before schema casting expands the dictionary."""
    dictionary = pa.array(dom_list, type=pa.string())
    width = max((len(domain.encode("utf-8")) for domain in dom_list), default=1)
    rows = max(1, min(DOMAIN_BUILD_MAX_ROWS, DOMAIN_BUILD_MAX_BYTES // max(1, width)))
    chunks = []
    for start in range(0, len(codes), rows):
        encoded = pa.DictionaryArray.from_arrays(
            pa.array(codes[start:start + rows].astype(np.int32)), dictionary
        )
        chunks.append(encoded.dictionary_decode())
    return pa.chunked_array(chunks, type=pa.string())


def active_domains(
    out_root: Path, profiles: dict[str, dict[str, Any]], sources: list[str] | None = None
) -> dict[str, list[str]]:
    """有 s3 产物的源 → 领域映射（配比/val/硬门的作用域）。"""
    doms: dict[str, list[str]] = {}
    for source, prof in profiles.items():
        if sources and source not in sources:
            continue
        if (out_root / "s3" / source).is_dir():
            doms.setdefault(prof["domain"], []).append(source)
    return {d: sorted(s) for d, s in sorted(doms.items())}


# ---------------------------------------------------------------- 硬门

def gate_errors(
    out_root: Path,
    profiles: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
    sources: list[str] | None = None,
) -> dict[str, list[str]]:
    """前置硬门校验（S6 首次物化前的 fail-fast 清单，2026-09-25 裁定）。"""
    errors: dict[str, list[str]] = {}
    doms = active_domains(out_root, profiles, sources)

    def add(gate: str, msg: str) -> None:
        errors.setdefault(gate, []).append(msg)

    # ① s4_scored 覆盖全部 s5 输入分片（s5 输入集 = s3 分片集）
    s4_missing: list[str] = []
    s4_selection_stale: list[str] = []
    s5_missing: list[str] = []
    decontam_missing: list[str] = []
    scorer_cfg: dict[str, Any] | None = None
    s4_audit_sha = ""
    if cfg["score_field"] == "hybrid_v2":
        scorer_cfg = load_scorer_config()
        s4_audit = out_root / "audits" / "s4_hybrid_v2_legacy_audit.json"
        if s4_audit.is_file():
            audit_payload = json.loads(s4_audit.read_text(encoding="utf-8"))
            if audit_payload.get("passed"):
                s4_audit_sha = _file_sha256(s4_audit)
    for source in [s for ds in doms.values() for s in ds]:
        src_dir = out_root / "s3" / source
        for path in sorted(src_dir.rglob("*.parquet")):
            rel = path.relative_to(src_dir).as_posix()[: -len(".parquet")]
            for stage, bucket in (("s4_scored", s4_missing), ("s5", s5_missing)):
                par = out_root / stage / source / (rel + ".parquet")
                summ = out_root / stage / source / (rel + ".summary.json")
                if not (par.is_file() and summ.is_file()):
                    bucket.append(f"{source}/{rel}")
            if cfg["score_field"] == "hybrid_v2":
                s1_path = out_root / "s1" / source / f"{rel}.parquet"
                scored_path = out_root / "s4_scored" / source / f"{rel}.parquet"
                score_summary = out_root / "s4_scored" / source / f"{rel}.summary.json"
                pairs_path = out_root / "s4_quantiles" / "pairs" / source / f"{rel}.parquet"
                if all(p.is_file() for p in (s1_path, scored_path, score_summary, pairs_path)):
                    reason = validate_selection_contract(
                        s1_path,
                        scored_path,
                        pairs_path,
                        json.loads(score_summary.read_text(encoding="utf-8")),
                        scorer_cfg or {},
                        s4_audit_sha,
                    )
                    if reason:
                        s4_selection_stale.append(f"{source}/{rel}: {reason}")
            dpar = out_root / "decontam" / source / (rel + ".parquet")
            dsumm = out_root / "decontam" / source / (rel + ".summary.json")
            if not (dpar.is_file() and dsumm.is_file()):
                decontam_missing.append(f"{source}/{rel}")
    if s4_missing:
        add("① s4_scored 覆盖", f"缺 {len(s4_missing)} 分片（如 {s4_missing[:3]}）——先跑完 --stage s4")
    if s4_selection_stale:
        add(
            "① s4_scored 覆盖",
            f"{len(s4_selection_stale)} 个 hybrid_v2 选择合同缺失或陈旧（如 {s4_selection_stale[:3]}）",
        )
    s4_global = out_root / "stage_s4_global_summary.json"
    s4_meta: dict[str, Any] = {}
    if s4_global.is_file():
        s4_meta = json.loads(s4_global.read_text("utf-8"))
        if int(s4_meta.get("shard_errors") or 0):
            add("① s4_scored 覆盖", f"stage_s4_global_summary shard_errors={s4_meta['shard_errors']} ≠ 0")
        score_shards = int(s4_meta.get("score_shards") or 0)
        quantile_shards = int(s4_meta.get("quantile_shards") or 0)
        total_rows = int(s4_meta.get("total_rows") or 0)
        if score_shards <= 0 or quantile_shards != score_shards:
            add(
                "① s4_scored 覆盖",
                f"S4 分片账不闭合: score_shards={score_shards}, quantile_shards={quantile_shards}",
            )
        if int(s4_meta.get("quantile_rows") or 0) != total_rows:
            add(
                "① s4_scored 覆盖",
                f"S4 行数账不闭合: total_rows={total_rows}, quantile_rows={s4_meta.get('quantile_rows')}",
            )
    else:
        add("① s4_scored 覆盖", f"全局 summary 缺失: {s4_global}")
    if cfg["score_field"] in ("composite_v2", "rule_composite", "hybrid_composite", "hybrid_v2"):
        suffix = {"composite_v2": "", "rule_composite": ".rule", "hybrid_composite": ".hybrid", "hybrid_v2": ".hybrid2"}[cfg["score_field"]]
        qt = out_root / "s4_quantiles" / "tables" / f"_build{suffix}.summary.json"
        if not qt.is_file():
            add("① s4_scored 覆盖", f"分位表未构建: {qt}（--stage s4 全量含 quantile build；rule/hybrid 表由 build_quantile_tables(score_field=...) 构建）")
        else:
            build_meta = json.loads(qt.read_text("utf-8"))
            built_meta = build_meta.get("domains") or {}
            built = set(built_meta)
            expected_score_field = {
                "composite_v2": "composite",  # base table 的历史无后缀别名
                "rule_composite": "rule_composite",
                "hybrid_composite": "hybrid_composite",
                "hybrid_v2": "hybrid_v2",
            }[cfg["score_field"]]
            if build_meta.get("score_field") != expected_score_field:
                add(
                    "① s4_scored 覆盖",
                    f"分位表 score_field={build_meta.get('score_field')!r}，期望 {expected_score_field!r}",
                )
            if cfg["score_field"] == "hybrid_v2" and not (
                (build_meta.get("input_contract") or {}).get(
                    "selection_contract_manifest_sha256"
                )
            ):
                add("① s4_scored 覆盖", "hybrid_v2 分位表未绑定选择合同 manifest")
            if s4_meta and (
                int(build_meta.get("pair_shards") or 0) != int(s4_meta.get("score_shards") or 0)
                or int(build_meta.get("total_rows") or 0) != int(s4_meta.get("total_rows") or 0)
            ):
                add("① s4_scored 覆盖", "分位表与 S4 全局分片数/行数不一致，禁止消费陈旧表")
            for dom in cfg["domains"]:
                if dom not in built and dom in doms:
                    add("① s4_scored 覆盖", f"分位表缺领域 {dom}（score_field={cfg['score_field']}）")
                    continue
                if dom in built and dom in doms:
                    info = built_meta[dom]
                    npy = out_root / "s4_quantiles" / "tables" / str(info.get("npy") or "")
                    if not npy.is_file() or int(info.get("n") or 0) <= 0:
                        add("① s4_scored 覆盖", f"分位表领域 {dom} 的 npy 缺失或为空")
    else:
        add("① s4_scored 覆盖", f"score_field={cfg['score_field']}：仅支持 composite_v2 / rule_composite")

    # ② decontam 全部完成且无 error shard
    dg = out_root / "stage_decontam_global_summary.json"
    if not dg.is_file():
        add("② decontam 完成", f"全局 summary 缺失: {dg}")
    else:
        gs = json.loads(dg.read_text("utf-8"))
        if int(gs.get("shard_errors") or 0):
            add("② decontam 完成", f"stage_decontam_global_summary shard_errors={gs['shard_errors']} ≠ 0")
        from .decontam import release_readiness_errors

        for reason in release_readiness_errors(gs):
            add("② decontam 完成", reason)
    if decontam_missing:
        add("② decontam 完成", f"缺 {len(decontam_missing)} 分片产物（如 {decontam_missing[:3]}）——先跑完 --stage decontam")

    # ③ ndv2 的 s3_decisions/web+code 必须来自当前 MinHash 精度门与直接边策略。
    # S5 直接读取发布 lookup；刷新 S3 中冗余的 s3_decision 列不是消费前置。
    # 保留 legacy fast-track 开关仅供旧实验复现；正式配置不允许豁免。
    allow_e17_web = bool(cfg.get("allow_e17_web"))
    allow_e17_code = bool(cfg.get("allow_e17_code"))
    for dom in NDV2_DOMAINS:
        if dom not in doms:
            continue
        if dom == "web" and allow_e17_web:
            continue
        if dom == "code" and allow_e17_code:
            continue
        summ = out_root / "s3_decisions" / f"{dom}.summary.json"
        if not summ.is_file():
            add("③ ndv2 lookup", f"s3_decisions/{dom}.summary.json 缺失（先跑 --stage s3/ndv2）")
            continue
        decision = json.loads(summ.read_text("utf-8"))
        policy = decision.get("policy")
        if policy == E17_POLICY:
            add("③ ndv2 lookup", f"{dom} 仍是 E-17 占位 policy={E17_POLICY}——先跑 --stage ndv2（fast track 可配 allow_e17_web: true）")
            continue
        if policy != NDV2_POLICY or decision.get("errors"):
            add("③ ndv2 lookup", f"{dom} policy={policy} 未识别（期望 {NDV2_POLICY}）")
            continue
        method = (decision.get("decision_contract") or {}).get("method")
        if method != NDV2_DECISION_METHOD:
            add(
                "③ ndv2 lookup",
                f"{dom} decision method={method!r}（期望 {NDV2_DECISION_METHOD}，禁止近似边传递闭包）",
            )
            continue
        audit_path = out_root / "s3_v2" / f"{dom}_minhash_precision_audit.json"
        if not audit_path.is_file():
            add("③ ndv2 lookup", f"{dom} 全文精度审计缺失: {audit_path}")
            continue
        audit = json.loads(audit_path.read_text("utf-8"))
        if (
            audit.get("passed") is not True
            or audit.get("errors")
            or (audit.get("audit_contract") or {}).get("edges_fingerprint")
            != decision.get("edges_fingerprint")
        ):
            add("③ ndv2 lookup", f"{dom} 全文精度审计失败或与当前边集不匹配")
            continue
        dropped = int(decision.get("dropped") or 0)
        lookup_root = out_root / "s3_lookup" / dom
        ids_path = lookup_root / "lookup.ids.bin"
        codes_path = lookup_root / "lookup.codes.bin"
        if (
            not ids_path.is_file()
            or not codes_path.is_file()
            or ids_path.stat().st_size != dropped * 16
            or codes_path.stat().st_size != dropped
        ):
            add(
                "③ ndv2 lookup",
                f"{dom} lookup 文件缺失或行数未与 dropped={dropped} 闭合",
            )
    required_ndv2_doms = [
        d for d in NDV2_DOMAINS
        if d in doms
        and not (d == "web" and allow_e17_web)
        and not (d == "code" and allow_e17_code)
    ]
    if required_ndv2_doms:
        ng = out_root / "stage_ndv2_global_summary.json"
        if not ng.is_file():
            add("③ ndv2 lookup", f"全局 summary 缺失: {ng}")
        else:
            gs = json.loads(ng.read_text("utf-8"))
            if int(gs.get("shard_errors") or gs.get("task_errors") or 0):
                add("③ ndv2 lookup", "stage_ndv2_global_summary 含 shard/task errors")
            for dom in required_ndv2_doms:
                expected = int(
                    ((gs.get("per_domain") or {}).get(dom) or {}).get("dropped") or -1
                )
                decision_path = out_root / "s3_decisions" / f"{dom}.summary.json"
                actual = int(json.loads(decision_path.read_text()).get("dropped") or 0)
                if expected != actual:
                    add(
                        "③ ndv2 lookup",
                        f"{dom} 全局 dropped={expected} 与决定表 dropped={actual} 不一致",
                    )

    # ⑤ s5 覆盖（读入数据源自身的完整性前提）
    if s5_missing:
        add("⑤ s5 覆盖", f"缺 {len(s5_missing)} 分片（如 {s5_missing[:3]}）——先跑完 --stage s5")
    s5_global = out_root / "stage_s5_global_summary.json"
    if not s5_global.is_file():
        add("⑤ s5 覆盖", f"全局 summary 缺失: {s5_global}")
    else:
        gs = json.loads(s5_global.read_text("utf-8"))
        if int(gs.get("shard_errors") or 0):
            add("⑤ s5 覆盖", f"stage_s5_global_summary shard_errors={gs['shard_errors']} ≠ 0")
        current_s5_contract = s5_global_input_contract(
            out_root, resolve_tokenizer_path()
        )
        if gs.get("input_contract") != current_s5_contract:
            add("⑤ s5 覆盖", "stage_s5_global_summary input_contract 与当前上游不一致")

    # mix 配置 vs 实际领域
    unknown = set(cfg["domains"]) - set(doms)
    if unknown:
        add("⑥ mix 配置", f"domains 不在活动领域（有 s3 产物）内: {sorted(unknown)}")
    return errors


def assert_gates(errors: dict[str, list[str]]) -> None:
    if not errors:
        return
    lines = ["S6 硬门拒绝物化（fail-fast，全部缺失如下）："]
    for gate in sorted(errors):
        lines.append(f"  {gate}: {len(errors[gate])} 项")
        for msg in errors[gate][:5]:
            lines.append(f"    - {msg}")
    raise SystemExit("\n".join(lines))


# ---------------------------------------------------------------- val 切出（gate ④）

def _pool_stats_for_tasks(
    out_root: Path, tasks: list[tuple[str, str, str]]
) -> dict[str, dict[str, int]]:
    """Exact S5 pool ledger for the release's source-scoped task roster."""
    result: dict[str, dict[str, int]] = {}
    for source, relpath, domain in tasks:
        _parquet, summary_path = s5_output_paths(out_root, source, relpath)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        row = result.setdefault(domain, {"docs": 0, "tokens": 0})
        row["docs"] += int(summary.get("out_rows") or 0)
        row["tokens"] += int(summary.get("total_tokens") or 0)
        for decision, count in (summary.get("decisions") or {}).items():
            key = f"decisions.{decision}"
            row[key] = row.get(key, 0) + int(count or 0)
    return result


def _init_worker(out_root: str) -> None:
    _WORKER["out_root"] = Path(out_root)


def _run_val_pass1(task: tuple[str, str, str]) -> dict[str, Any]:
    source, relpath, domain = task
    started = time.monotonic()
    out_root = _WORKER["out_root"]
    try:
        out_path, _ = s5_output_paths(out_root, source, relpath)
        hist_t = np.zeros(VAL_BUCKETS, dtype=np.uint64)
        hist_d = np.zeros(VAL_BUCKETS, dtype=np.uint64)
        pf = pq.ParquetFile(out_path)
        for batch in safe_iter_batches(pf, batch_size=8192, columns=["doc_id", "n_tokens"]):
            d = batch.to_pydict()
            for doc_id, nt in zip(d["doc_id"], d["n_tokens"]):
                b = _hash_bucket(val_hash(doc_id))
                hist_t[b] += int(nt)
                hist_d[b] += 1
        return {
            "stage": "s6_val_pass1", "source": source, "relpath": relpath, "domain": domain,
            "hist_tokens": hist_t.tolist(), "hist_docs": hist_d.tolist(),
            "elapsed_sec": round(time.monotonic() - started, 3), "errors": {},
        }
    except Exception as exc:
        return {
            "stage": "s6_val_pass1", "source": source, "relpath": relpath,
            "elapsed_sec": round(time.monotonic() - started, 3),
            "errors": {f"shard_error:{type(exc).__name__}": 1}, "error_message": str(exc)[:500],
        }


def _run_val_pass2(args: tuple[tuple[str, str, str], dict[str, int]]) -> dict[str, Any]:
    task, boundaries = args
    source, relpath, domain = task
    started = time.monotonic()
    out_root = _WORKER["out_root"]
    B = boundaries[domain]
    try:
        out_path, _ = s5_output_paths(out_root, source, relpath)
        low: dict[str, int] = {}
        edge: list[tuple[bytes, str, int]] = []
        pf = pq.ParquetFile(out_path)
        for batch in safe_iter_batches(pf, batch_size=8192, columns=["doc_id", "n_tokens"]):
            d = batch.to_pydict()
            for doc_id, nt in zip(d["doc_id"], d["n_tokens"]):
                b = _hash_bucket(val_hash(doc_id))
                if b < B:
                    low[doc_id] = int(nt)
                elif b == B:
                    edge.append((val_hash(doc_id), doc_id, int(nt)))
        return {
            "stage": "s6_val_pass2", "source": source, "relpath": relpath, "domain": domain,
            "low": low, "edge": edge,
            "elapsed_sec": round(time.monotonic() - started, 3), "errors": {},
        }
    except Exception as exc:
        return {
            "stage": "s6_val_pass2", "source": source, "relpath": relpath,
            "elapsed_sec": round(time.monotonic() - started, 3),
            "errors": {f"shard_error:{type(exc).__name__}": 1}, "error_message": str(exc)[:500],
        }


def _scan(
    stage: str,
    tasks: list[Any],
    fn: Any,
    workers: int,
    guard: tuple[Any, Any],
    initargs: tuple = (),
) -> list[dict[str, Any]]:
    beat, guard_stop = guard
    summaries: list[dict[str, Any]] = []
    started = time.monotonic()
    done = 0
    try:
        with mp.Pool(processes=workers, initializer=_init_worker, initargs=initargs) as pool:
            it = pool.imap_unordered(fn, tasks, chunksize=1)
            while True:
                try:
                    summary = next(it)
                except StopIteration:
                    break
                except Exception as exc:
                    log.error("pool-level task failure: %s", exc)
                    summaries.append({"shard_fatal": f"pool:{type(exc).__name__}: {exc}"})
                    continue
                summaries.append(summary)
                done += 1
                beat()
                if done % 200 == 0 or done == len(tasks):
                    log.info(
                        "progress[%s]: %d/%d tasks, %.2f tasks/s",
                        stage, done, len(tasks), done / max(time.monotonic() - started, 1e-9),
                    )
    finally:
        guard_stop.set()
    return summaries


def _write_errors(errs: list[dict[str, Any]], err_root: Path, stage: str) -> None:
    err_root.mkdir(parents=True, exist_ok=True)
    for s in errs[:10]:
        name = f"{s.get('source') or s.get('domain') or 'task'}__{s.get('relpath') or s.get('task') or 'unknown'}"
        write_summary(err_root / f"{name}.error.json", s)


def _prepare_val_contract(
    out_root: Path,
    profiles: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
    sources: list[str] | None,
) -> tuple[
    dict[str, list[str]],
    list[tuple[str, str, str]],
    dict[str, dict[str, int]],
    dict[str, Any],
    str,
    dict[str, int],
]:
    doms = active_domains(out_root, profiles, sources)
    tasks = s5_tasks(out_root, profiles, sources=sources)
    pool = _pool_stats_for_tasks(out_root, tasks)
    input_snapshot = _s5_input_snapshot(out_root, tasks)
    # Validation is a fixed held-out set shared by every curriculum stage.
    # Training-order seeds differ across stages, but the deterministic
    # blake2b(doc_id) sampler must remain bound to the original release seed.
    val_seed = str(cfg.get("val_seed", "yuformer"))
    config_sha256 = hashlib.sha256(
        json.dumps(
            {"seed": val_seed, "val": cfg["val"], "domains": doms},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    quotas = {
        dom: min(
            int(cfg["val"]["tokens_per_domain"]),
            int(cfg["val"]["max_fraction"] * pool.get(dom, {"tokens": 0})["tokens"]),
        )
        for dom in doms
    }
    return doms, tasks, pool, input_snapshot, config_sha256, quotas


def _val_outputs_current(
    out_root: Path,
    doms: dict[str, list[str]],
    input_snapshot_sha256: str,
    config_sha256: str,
    quotas: dict[str, int],
) -> bool:
    val_root = out_root / "val_sets"
    for dom in doms:
        par = val_root / f"{dom}.parquet"
        summary_path = val_root / f"{dom}.summary.json"
        ids = val_root / f"{dom}.ids.bin"
        if not (par.is_file() and summary_path.is_file() and ids.is_file()):
            return False
        try:
            summary = json.loads(summary_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not (
            not summary.get("errors")
            and summary.get("input_snapshot_sha256") == input_snapshot_sha256
            and summary.get("config_sha256") == config_sha256
            and int(summary.get("quota_tokens") or 0) == quotas[dom]
        ):
            return False
    return True


def ensure_val_sets(
    out_root: Path,
    profiles: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
    workers: int,
    guard_factory: Any,
    sources: list[str] | None = None,
    _prepared: tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    """逐域确定性 val 切出（文件级幂等：parquet+summary+ids.bin 齐在即 skip）。

    抽样语义：blake2b(doc_id, 128bit) 升序 = 无放回等概率（确定性零随机数）；
    配额 = min(tokens_per_domain, max_fraction × 域池 tokens)（微域安全阀）。"""
    val_root = out_root / "val_sets"
    val_root.mkdir(parents=True, exist_ok=True)
    prepared = _prepared or _prepare_val_contract(out_root, profiles, cfg, sources)
    doms, tasks, pool, input_snapshot, val_config_sha256, quotas = prepared

    def dom_done(dom: str) -> bool:
        par = val_root / f"{dom}.parquet"
        summary_path = val_root / f"{dom}.summary.json"
        ids = val_root / f"{dom}.ids.bin"
        if not (par.is_file() and summary_path.is_file() and ids.is_file()):
            return False
        summary = json.loads(summary_path.read_text("utf-8"))
        return (
            not summary.get("errors")
            and summary.get("input_snapshot_sha256") == input_snapshot["sha256"]
            and summary.get("config_sha256") == val_config_sha256
            and int(summary.get("quota_tokens") or 0) == quotas[dom]
        )

    missing = [dom for dom in doms if not dom_done(dom)]
    if missing:
        log.info("s6_val: cutting val sets for %s (quotas=%s)", missing, quotas)
        summaries = _scan(
            "s6_val_pass1", tasks, _run_val_pass1, workers,
            guard_factory("s6_val_pass1"), initargs=(str(out_root),),
        )
        errs = [s for s in summaries if s.get("errors") or s.get("shard_fatal")]
        if errs:
            _write_errors(errs, val_root / "_errors", "s6_val_pass1")
            raise SystemExit(
                f"s6_val pass1: {len(errs)} shard failures（error summary 已落 {val_root}/_errors，修复后重跑）"
            )
        hist = {dom: np.zeros(VAL_BUCKETS, dtype=np.uint64) for dom in missing}
        for s in summaries:
            dom = s.get("domain")
            if dom in hist:
                hist[dom] += np.array(s["hist_tokens"], dtype=np.uint64)
        boundaries = {}
        for dom in missing:
            cum = np.cumsum(hist[dom])
            B = int(np.searchsorted(cum, quotas[dom], side="left"))
            boundaries[dom] = min(B, VAL_BUCKETS - 1)
        summaries = _scan(
            "s6_val_pass2", [(t, boundaries) for t in tasks if t[2] in missing],
            _run_val_pass2, workers,
            guard_factory("s6_val_pass2"), initargs=(str(out_root),),
        )
        errs = [s for s in summaries if s.get("errors") or s.get("shard_fatal")]
        if errs:
            _write_errors(errs, val_root / "_errors", "s6_val_pass2")
            raise SystemExit(f"s6_val pass2: {len(errs)} shard failures（见 {val_root}/_errors）")
        low: dict[str, dict[str, int]] = {dom: {} for dom in missing}
        edge: dict[str, list[tuple[bytes, str, int]]] = {dom: [] for dom in missing}
        for s in summaries:
            dom = s.get("domain")
            if dom not in missing:
                continue
            low[dom].update(s["low"])
            edge[dom].extend(s["edge"])
        for dom in missing:
            low_ids = sorted(low[dom])
            low_nt = [low[dom][d] for d in low_ids]
            low_tokens = int(sum(low_nt))
            fill = quotas[dom] - low_tokens
            add_ids: list[str] = []
            add_nt: list[int] = []
            if fill > 0:
                acc = 0
                for _h, doc_id, nt in sorted(edge[dom], key=lambda t: (t[0], t[1])):
                    add_ids.append(doc_id)
                    add_nt.append(nt)
                    acc += nt
                    if acc >= fill:
                        break
            ids_v = _hex_strings_to_void(np.array(low_ids + add_ids, dtype=object))
            nt_arr = np.array(low_nt + add_nt, dtype=np.uint32)
            order = np.argsort(ids_v, kind="stable")
            _write_val_domain(
                val_root, dom, np.ascontiguousarray(ids_v[order]),
                nt_arr[order], quotas[dom], pool.get(dom, {}),
                input_snapshot["sha256"], val_config_sha256,
            )
    else:
        log.info("s6_val: all domains already cut (skip)")

    per_domain = {}
    for dom in doms:
        summ = json.loads((val_root / f"{dom}.summary.json").read_text("utf-8"))
        per_domain[dom] = {
            "val_docs": int(summ.get("val_docs") or 0),
            "val_tokens": int(summ.get("val_tokens") or 0),
            "quota_tokens": int(summ.get("quota_tokens") or 0),
            "pool_docs": int(summ.get("pool_docs") or 0),
            "pool_tokens": int(summ.get("pool_tokens") or 0),
        }
    payload = {
        "stage": "s6_val",
        "seed": str(cfg.get("val_seed", "yuformer")),
        "hash": "blake2b-128(doc_id) → 4096 桶直方图边界 + 边界桶内全哈希前缀",
        "input_snapshot": input_snapshot,
        "config_sha256": val_config_sha256,
        "per_domain": per_domain,
    }
    write_summary(val_root / "_val_global_summary.json", payload)
    return payload


def _write_val_domain(
    val_root: Path,
    dom: str,
    ids_v: np.ndarray,
    nt_arr: np.ndarray,
    quota: int,
    pool_info: dict[str, int],
    input_snapshot_sha256: str,
    config_sha256: str,
) -> None:
    table = pa.table(
        {
            "doc_id": _doc_id_array(ids_v),
            "domain": pa.array([dom] * len(ids_v), type=pa.string()),
            "n_tokens": pa.array(nt_arr, type=pa.uint32()),
        },
        schema=VAL_SCHEMA,
    )
    par_path = val_root / f"{dom}.parquet"
    tmp = par_path.with_name(par_path.name + f".tmp-{os.getpid()}")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, par_path)
    ids_path = val_root / f"{dom}.ids.bin"
    tmp_i = ids_path.with_name(ids_path.name + f".tmp-{os.getpid()}")
    np.ascontiguousarray(ids_v).tofile(tmp_i)
    os.replace(tmp_i, ids_path)
    summary = {
        "stage": "s6_val",
        "domain": dom,
        "quota_tokens": int(quota),
        "val_docs": int(len(ids_v)),
        "val_tokens": int(nt_arr.astype(np.uint64).sum()) if len(nt_arr) else 0,
        "pool_docs": int(pool_info.get("docs") or 0),
        "pool_tokens": int(pool_info.get("tokens") or 0),
        "sampling": "blake2b(doc_id) 升序 = 无放回等概率（确定性）",
        "input_snapshot_sha256": input_snapshot_sha256,
        "config_sha256": config_sha256,
        "errors": {},
        "skipped": False,
    }
    write_summary(val_root / f"{dom}.summary.json", summary)


# ---------------------------------------------------------------- 阈值与候选收集

def load_quantile_table(out_root: Path, domain: str, score_field: str = "composite_v2") -> np.ndarray:
    """score_field 后缀约定（score.py build_quantile_tables）：<domain>.npy /
    .rule.npy / .hybrid.npy / .hybrid2.npy。表缺失/域缺失即报错（fail-fast）。"""
    allowed = ("composite_v2", "rule_composite", "hybrid_composite", "hybrid_v2")
    if score_field not in allowed:
        raise ValueError(f"unsupported score_field: {score_field}")
    suffix = {"composite_v2": "", "rule_composite": ".rule", "hybrid_composite": ".hybrid", "hybrid_v2": ".hybrid2"}[score_field]
    tables_dir = out_root / "s4_quantiles" / "tables"
    meta_path = tables_dir / f"_build{suffix}.summary.json"
    meta = json.loads(meta_path.read_text("utf-8"))["domains"]
    if domain not in meta:
        raise FileNotFoundError(f"quantile table missing for domain {domain} (score_field={score_field})")
    return np.load(tables_dir / meta[domain]["npy"])


def topk_threshold(table: np.ndarray, top_k: float) -> float:
    """composite ≥ 返回值 ⇔ 进入该域 top-⌈top_k·n⌉（同值同分 ⇒ 决胜无关）。"""
    n = len(table)
    if n == 0:
        raise ValueError("empty quantile table")
    k = max(1, int(np.ceil(top_k * n)))
    return float(table[n - k])


def load_candidate_parts(
    parts: list[Path], expected_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Copy one archive at a time into fixed-size arrays, closing it immediately."""
    if expected_rows < 0:
        raise ValueError("candidate row count must be nonnegative")
    dtypes = {"ids": np.dtype("V16"), "comp": np.dtype("float32"),
              "nt": np.dtype("uint32"), "up": np.dtype("float64")}
    columns = {key: np.empty(expected_rows, dtype=dtype) for key, dtype in dtypes.items()}
    src_idx = np.empty(expected_rows, dtype=np.uint16)
    src_keys = sorted({path.name.split("__", 1)[0] for path in parts})
    if len(src_keys) > np.iinfo(np.uint16).max + 1:
        raise ValueError("too many candidate sources for uint16 source indices")
    src_lut = {source: index for index, source in enumerate(src_keys)}
    cursor = 0
    for path in parts:
        with np.load(path, allow_pickle=False) as archive:
            ids = archive["ids"]
            count = len(ids)
            stop = cursor + count
            if stop > expected_rows:
                raise ValueError("candidate parts exceed the committed row count")
            for key, dtype in dtypes.items():
                values = ids if key == "ids" else archive[key]
                if values.ndim != 1 or len(values) != count or values.dtype != dtype:
                    raise ValueError(f"invalid candidate column {key}: {path}")
                columns[key][cursor:stop] = values
            src_idx[cursor:stop] = src_lut[path.name.split("__", 1)[0]]
            cursor = stop
    if cursor != expected_rows:
        raise ValueError("candidate parts do not cover the committed row count")
    return columns["ids"], columns["comp"], columns["nt"], columns["up"], src_idx, src_keys


def candidate_prefix_count(sorted_n_tokens: np.ndarray, quota_tokens: int) -> int:
    """Count the complete-document prefix that fits a token quota.

    ``top_k`` is a document-quality quantile and has already been applied when
    candidates reach this function.  It must not be multiplied by the pool's
    token count as a second cap: document length can correlate with quality, so
    the top 10% of documents need not contain 10% of tokens.
    """
    if quota_tokens <= 0 or not len(sorted_n_tokens):
        return 0
    cumulative = np.cumsum(sorted_n_tokens.astype(np.uint64))
    count = int(np.searchsorted(cumulative, quota_tokens, side="right"))
    return count or 1


def _collect_worker(
    args: tuple[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    task, ctx = args
    source, relpath, domain = task
    started = time.monotonic()
    out_root = Path(ctx["out_root"])
    if domain not in ctx["thresholds"]:
        return {
            "stage": "s6_collect", "source": source, "relpath": relpath, "domain": domain,
            "candidates": 0, "candidate_tokens": 0,
            "elapsed_sec": round(time.monotonic() - started, 3), "errors": {},
        }
    work_dir = Path(ctx["work_dir"]) / domain
    work_root = Path(ctx["work_dir"])
    done_path = _collect_done_path(work_root, source, relpath, domain)
    if done_path.is_file():
        try:
            committed = json.loads(done_path.read_text(encoding="utf-8"))
            artifacts = committed.get("artifacts") or []
            if (
                committed.get("schema_version") == S6_COLLECT_SCHEMA_VERSION
                and isinstance(artifacts, list)
                and all((work_root / rel).is_file() for rel in artifacts)
                and isinstance(committed.get("summary"), dict)
            ):
                resumed = dict(committed["summary"])
                resumed["skipped"] = True
                return resumed
        except (OSError, json.JSONDecodeError, TypeError):
            pass
        done_path.unlink(missing_ok=True)
    try:
        s5_path, _ = s5_output_paths(out_root, source, relpath)
        scored = out_root / "s4_scored" / source / (relpath + ".parquet")
        score_field = ctx["score_field"]
        goldrule_quota = int(ctx.get("goldrule_quota") or 0)
        threshold = ctx["thresholds"][domain]
        # With C4 enabled, read all numeric tracks in the existing S4 pass, but
        # retain them only for rows that pass the quality threshold *and* hash
        # sample. This avoids both the old all-row tuple cache and a second scan
        # over billions of S4 rows.
        score_cols = list(dict.fromkeys([
            "doc_id", score_field, "upstream_score",
            *(GOLDRULE_TRACK_COLUMNS if goldrule_quota else ()),
        ]))
        comp: dict[str, float] = {}
        up: dict[str, float] = {}
        gold_tracks: dict[str, tuple[bytes, tuple[float, ...], int]] = {}
        pf = pq.ParquetFile(scored)
        if goldrule_quota:
            missing = sorted(set(GOLDRULE_TRACK_COLUMNS) - set(pf.schema_arrow.names))
            if missing:
                raise ValueError(f"goldrule score columns missing from {scored}: {missing}")
        representation = "current_v2"
        if score_field == "hybrid_v2":
            summary_path = out_root / "s4_scored" / source / (
                relpath + ".summary.json"
            )
            selection = (
                json.loads(summary_path.read_text(encoding="utf-8"))
                .get("selection_contracts", {})
                .get("hybrid_v2", {})
            )
            representation = str(selection.get("representation") or "")
        for batch in safe_iter_batches(pf, batch_size=8192, columns=score_cols):
            d = batch.to_pydict()
            scores = d[score_field]
            effective_hybrid: np.ndarray | None = None
            if score_field == "hybrid_v2":
                effective_hybrid = effective_hybrid_v2(
                    np.asarray(scores, dtype=np.float32),
                    d["upstream_score"],
                    ctx["hybrid_v2_scorer_cfg"],
                    representation,
                )
                scores = effective_hybrid.tolist()
            elif goldrule_quota:
                effective_hybrid = effective_hybrid_v2(
                    np.asarray(d["hybrid_v2"], dtype=np.float32),
                    d["upstream_score"],
                    ctx["hybrid_v2_scorer_cfg"],
                    representation,
                )
            for i, (doc_id, c, u) in enumerate(
                zip(d["doc_id"], scores, d["upstream_score"])
            ):
                comp[doc_id] = float(c)
                up[doc_id] = float(u) if u is not None else float("nan")
                if not goldrule_quota or float(c) < threshold:
                    continue
                doc_bytes = bytes.fromhex(doc_id)
                h = int.from_bytes(
                    hashlib.blake2b(
                        b"goldrule|" + doc_bytes, digest_size=8
                    ).digest(),
                    "big",
                )
                if not _goldrule_hash_selected(
                    h, float(ctx.get("goldrule_rate", 0.0))
                ):
                    continue
                vals: list[float] = []
                for name in GOLDRULE_TRACK_COLUMNS:
                    value = effective_hybrid[i] if name == "hybrid_v2" else d[name][i]
                    if name == "upstream_norm" and u is None:
                        value = None
                    vals.append(float(value) if value is not None else float("nan"))
                gold_tracks[doc_id] = (doc_bytes, tuple(vals), h)
        val_ids = ctx["val_ids"].get(domain)
        ids_l: list[bytes] = []
        comp_l: list[float] = []
        nt_l: list[int] = []
        up_l: list[float] = []
        gold_l: list[tuple[bytes, tuple[float, ...], int, int]] = []
        n_cand = 0
        cand_tokens = 0
        pf5 = pq.ParquetFile(s5_path)
        for batch in safe_iter_batches(pf5, batch_size=8192, columns=["doc_id", "n_tokens"]):
            d = batch.to_pydict()
            ids_v = _hex_strings_to_void(np.array(d["doc_id"], dtype=object))
            comp_hit = [comp.get(doc_id) for doc_id in d["doc_id"]]
            # None（s4_scored 无此行，如 S1 review/drop 行未计长）与 0.0 分都必须区分
            keep = np.fromiter(
                (c is not None and c >= threshold for c in comp_hit),
                dtype=bool, count=len(comp_hit),
            )
            if val_ids is not None and len(val_ids):
                pos = np.searchsorted(val_ids, ids_v)
                pos = np.clip(pos, 0, len(val_ids) - 1)
                in_val = (val_ids[pos] == ids_v)
                keep &= ~in_val
            for i in np.flatnonzero(keep):
                n_cand += 1
                cand_tokens += int(d["n_tokens"][int(i)])
                ids_l.append(ids_v[int(i)].tobytes())
                comp_l.append(comp_hit[int(i)])
                nt_l.append(int(d["n_tokens"][int(i)]))
                up_l.append(up.get(d["doc_id"][int(i)], float("nan")))
                if goldrule_quota:
                    track = gold_tracks.get(d["doc_id"][int(i)])
                    if track is not None:
                        gold_l.append((
                            track[0], track[1], int(d["n_tokens"][int(i)]), track[2]
                        ))
        committed_artifacts: list[str] = []
        if ids_l:
            work_dir.mkdir(parents=True, exist_ok=True)
            slug = hashlib.blake2b(relpath.encode("utf-8"), digest_size=8).hexdigest()
            part = work_dir / f"{source}__{slug}.npz"
            # np.savez 对不带 .npz 后缀的名字会自行追加——tmp 名必须已带 .npz
            tmp = part.with_name(part.name + f".tmp-{os.getpid()}.npz")
            np.savez(
                tmp,
                ids=np.frombuffer(b"".join(ids_l), dtype="V16").reshape(-1),
                comp=np.array(comp_l, dtype=np.float32),
                nt=np.array(nt_l, dtype=np.uint32),
                up=np.array(up_l, dtype=np.float64),
            )
            os.replace(tmp, part)
            committed_artifacts.append(part.relative_to(work_root).as_posix())
        if gold_l:
            gr_dir = Path(ctx["work_dir"]) / "goldrule"
            gr_dir.mkdir(parents=True, exist_ok=True)
            slug = hashlib.blake2b((domain + "|" + relpath).encode("utf-8"), digest_size=8).hexdigest()
            gr_part = gr_dir / f"{domain}__{source}__{slug}.npz"
            tmp = gr_part.with_name(gr_part.name + f".tmp-{os.getpid()}.npz")
            np.savez(
                tmp,
                ids=np.frombuffer(b"".join(g[0] for g in gold_l), dtype="V16").reshape(-1),
                **{
                    name: np.array(
                        [g[1][j] for g in gold_l], dtype=GOLDRULE_TRACK_DTYPES[name]
                    )
                    for j, name in enumerate(GOLDRULE_TRACK_COLUMNS)
                },
                nt=np.array([g[2] for g in gold_l], dtype=np.uint32),
                sample_hash=np.array([g[3] for g in gold_l], dtype=np.uint64),
                domain=np.array([domain.encode("utf-8")] * len(gold_l), dtype="S24"),
                source=np.array([source.encode("utf-8")] * len(gold_l), dtype="S32"),
            )
            os.replace(tmp, gr_part)
            committed_artifacts.append(gr_part.relative_to(work_root).as_posix())
        result = {
            "stage": "s6_collect", "source": source, "relpath": relpath, "domain": domain,
            "candidates": n_cand, "candidate_tokens": cand_tokens,
            "elapsed_sec": round(time.monotonic() - started, 3), "errors": {},
        }
        done_path.parent.mkdir(parents=True, exist_ok=True)
        write_summary(done_path, {
            "schema_version": S6_COLLECT_SCHEMA_VERSION,
            "artifacts": committed_artifacts,
            "summary": result,
        })
        return result
    except Exception as exc:
        return {
            "stage": "s6_collect", "source": source, "relpath": relpath,
            "elapsed_sec": round(time.monotonic() - started, 3),
            "errors": {f"shard_error:{type(exc).__name__}": 1}, "error_message": str(exc)[:500],
        }


# ---------------------------------------------------------------- 领域排序选取

def sort_key_quality(ids: np.ndarray, comp: np.ndarray) -> np.ndarray:
    """(composite 降序, doc_id 字节序升序) 的 lexsort 序（全向量化）。"""
    comp_inv = np.uint32(0xFFFFFFFF) - np.ascontiguousarray(comp).view(np.uint32)
    hi, lo = _ids_be_keys(ids)
    return np.lexsort((lo, hi, comp_inv))


# ---------------------------------------------------------------- 课程两段 + 落盘

def stage1_order_key(ids: np.ndarray, up: np.ndarray) -> np.ndarray:
    """(upstream 升序 NaN 殿后, doc_id 字节序)——单域内 edu 低→高课程序。"""
    up_k = np.where(np.isnan(up), np.inf, up)
    hi, lo = _ids_be_keys(ids)
    return np.lexsort((lo, hi, up_k))


def stage2_order_key(ids: np.ndarray) -> np.ndarray:
    """doc_id 字节序 = blake2b 均匀哈希序（确定性全局"打乱"）。"""
    hi, lo = _ids_be_keys(ids)
    return np.lexsort((lo, hi))


def _write_order_table(
    path: Path,
    ids: np.ndarray,
    dom_codes: np.ndarray,
    dom_list: list[str],
    nt: np.ndarray,
    up: np.ndarray,
) -> None:
    table = pa.table(
        {
            "position": np.arange(len(ids), dtype=np.int64),
            "doc_id": _doc_id_chunked(ids),
            "domain": _domain_array(dom_codes, dom_list),
            "n_tokens": pa.array(np.ascontiguousarray(nt), type=pa.uint32()),
            "upstream_score": pa.array(np.ascontiguousarray(up), type=pa.float64()),
        },
        schema=ORDER_SCHEMA,
    )
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    # E-45：同 members——stage2 全量 ~1.5 亿行 × 32B = 4.8GB > parquet 2GB 列块上限。
    pq.write_table(table, tmp, compression="zstd", row_group_size=50_000_000)
    os.replace(tmp, path)


def _candidate_token_totals(summaries: list[dict[str, Any]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for summary in summaries:
        domain = str(summary.get("domain"))
        totals[domain] = totals.get(domain, 0) + int(summary.get("candidate_tokens") or 0)
    return totals


def _assert_quota_capacity(out_root: Path, mix_name: str, quotas: dict[str, int],
                           available: dict[str, int], maximum: int, phase: str) -> None:
    shortfalls = {domain: max(0, quota - available.get(domain, 0))
                  for domain, quota in quotas.items()}
    total = sum(shortfalls.values())
    if total <= maximum:
        return
    for relative in (f"mix_members/{mix_name}.parquet",
                     f"mix_order/{mix_name}_stage1.parquet",
                     f"mix_order/{mix_name}_stage2.parquet"):
        (out_root / relative).unlink(missing_ok=True)
    write_summary(out_root / "audits" / f"s6_{mix_name}_quota_failure.json", {
        "status": "FAIL", "phase": phase, "mix_name": mix_name,
        "total_shortfall_tokens": total, "max_total_shortfall_tokens": maximum,
        "per_domain": {domain: {"quota_tokens": quota,
                                "available_tokens": available.get(domain, 0),
                                "shortfall_tokens": shortfalls[domain]}
                       for domain, quota in quotas.items()},
    })
    raise SystemExit(f"s6 quota shortfall {total:,} exceeds configured maximum {maximum:,} "
                     f"at {phase}: { {d: n for d, n in shortfalls.items() if n} }")


def materialize(
    out_root: Path,
    profiles: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
    workers: int,
    guard_factory: Any,
    sources: list[str] | None = None,
) -> Path:
    mix_name = str(cfg["mix_name"])
    echo = config_echo(cfg)
    mix_root = out_root / "mix_members"
    order_root = out_root / "mix_order"
    # E-46：work 目录加 __v2 后缀——跨 pod 幽灵 driver（旧代码 checkout）启动时会对
    # 标准名 mix_work/<mix> 做 strict rmtree，实测把我们的 collect npz 删掉导致
    # FileNotFoundError 崩溃（2026-10-10 两次）。幽灵永不使用 __v2 名 → 彻底隔离；
    # 本 pod 重启沿用 __v2 → npz 断点续跑不受影响。
    work_root = out_root / "mix_work" / f"{mix_name}__v2"
    manifest_path = order_root / f"{mix_name}.manifest.json"
    members_path = mix_root / f"{mix_name}.parquet"

    # Validation files are release artifacts.  Revoke an older commit marker
    # before ensure_val_sets can overwrite them, otherwise readers can observe
    # an old manifest paired with a new validation generation.
    prepared_val = _prepare_val_contract(out_root, profiles, cfg, sources)
    doms, tasks, pool, val_input_snapshot, val_config_sha256, val_quotas = prepared_val
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = {}
        val_current = _val_outputs_current(
            out_root,
            doms,
            val_input_snapshot["sha256"],
            val_config_sha256,
            val_quotas,
        )
        if previous.get("config_echo") != echo or not val_current or not members_path.is_file():
            manifest_path.unlink(missing_ok=True)
            global_summary = out_root / "stage_s6_global_summary.json"
            if global_summary.is_file():
                try:
                    if json.loads(global_summary.read_text("utf-8")).get("mix_name") == mix_name:
                        global_summary.unlink()
                except (OSError, json.JSONDecodeError):
                    global_summary.unlink(missing_ok=True)

    val_summary = ensure_val_sets(
        out_root, profiles, cfg, workers, guard_factory, sources, _prepared=prepared_val
    )
    upstream_snapshot = _s6_upstream_snapshot(out_root, tasks, cfg["score_field"])

    if manifest_path.is_file() and members_path.is_file():
        prev = json.loads(manifest_path.read_text("utf-8"))
        published = [
            members_path,
            out_root / "stage_s6_global_summary.json",
            *[order_root / name for name in prev.get("order_tables", [])],
        ]
        if prev.get("goldrule_research"):
            published.append(out_root / str(prev["goldrule_research"]))
        if (
            prev.get("config_echo") == echo
            and prev.get("upstream_snapshot_sha256") == upstream_snapshot["snapshot_sha256"]
            and all(path.is_file() for path in published)
            and (
                not prev.get("published_artifacts")
                or _published_artifacts_current(out_root, prev["published_artifacts"])
            )
            and (
                not prev.get("goldrule_research")
                or prev.get("goldrule_research_sha256")
                == _file_sha256(out_root / str(prev["goldrule_research"]))
            )
        ):
            log.info("s6: mix %s already materialized (config + upstream snapshot match, skip)", mix_name)
            return out_root / "stage_s6_global_summary.json"
        # The manifest is the commit marker.  Revoke a stale generation before
        # overwriting any member/order file so readers cannot mistake mixed
        # generations for a complete publication.
        manifest_path.unlink(missing_ok=True)
        global_summary = out_root / "stage_s6_global_summary.json"
        if global_summary.is_file():
            try:
                if json.loads(global_summary.read_text("utf-8")).get("mix_name") == mix_name:
                    global_summary.unlink()
            except (OSError, json.JSONDecodeError):
                global_summary.unlink(missing_ok=True)

    quotas = allocate_mix_quotas(
        {d: float(s["weight"]) for d, s in cfg["domains"].items()},
        int(cfg["total_tokens"]),
    )
    nominal_cap_tokens = {
        d: int(float(cfg["domains"][d]["top_k"]) * pool.get(d, {"tokens": 0})["tokens"])
        for d in cfg["domains"]
    }
    # A quality fraction is a preferred band, while the token quota is the
    # release contract.  For domains whose preferred band cannot satisfy the
    # quota, deepen_to_quota scans the complete scored pool and still selects
    # the highest-quality prefix.  Scanning the full deficient domain avoids an
    # approximate score/token conversion and guarantees a one-pass result after
    # validation IDs are removed.
    effective_top_k = {
        d: (
            1.0
            if cfg["quota_fill_policy"] == "deepen_to_quota"
            and int(quotas[d]) > nominal_cap_tokens[d]
            else float(cfg["domains"][d]["top_k"])
        )
        for d in cfg["domains"]
    }
    thresholds = {
        d: topk_threshold(
            load_quantile_table(out_root, d, cfg["score_field"]),
            effective_top_k[d],
        )
        for d in cfg["domains"]
    }
    val_ids = {
        d: np.memmap(out_root / "val_sets" / f"{d}.ids.bin", dtype="V16", mode="r")
        for d in active_domains(out_root, profiles, sources)
        if (out_root / "val_sets" / f"{d}.ids.bin").is_file()
    }

    log.info(
        "s6 collect: %d shards, thresholds=%s", len(tasks),
        {d: round(t, 6) for d, t in thresholds.items()},
    )
    task_identity = hashlib.sha256(
        json.dumps(tasks, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    work_contract = {
        "schema_version": S6_COLLECT_SCHEMA_VERSION,
        "config_echo": echo,
        "upstream_snapshot_sha256": upstream_snapshot["snapshot_sha256"],
        "task_identity_sha256": task_identity,
        "task_count": len(tasks),
    }
    resumed = _prepare_collect_work_root(work_root, work_contract)
    log.info(
        "s6 collect work generation: %s (%s)",
        work_root,
        "resume" if resumed else "new",
    )
    gr_cfg = cfg.get("goldrule") or {}
    collect_ctx = {
        "out_root": str(out_root), "work_dir": str(work_root),
        "score_field": cfg["score_field"], "thresholds": thresholds, "val_ids": val_ids,
        "goldrule_quota": int(gr_cfg.get("per_domain_quota", 0)),
        "goldrule_rate": float(gr_cfg.get("sample_rate", 0.0)),
    }
    if cfg["score_field"] == "hybrid_v2" or collect_ctx["goldrule_quota"]:
        collect_ctx["hybrid_v2_scorer_cfg"] = load_scorer_config()
    summaries = _scan(
        "s6_collect", [(t, collect_ctx) for t in tasks],
        _collect_worker, workers, guard_factory("s6_collect"),
        initargs=(str(out_root),),
    )
    errs = [s for s in summaries if s.get("errors") or s.get("shard_fatal")]
    if errs:
        _write_errors(errs, work_root / "_errors", "s6_collect")
        raise SystemExit(
            f"s6 collect: {len(errs)} shard failures（error summary 已落盘，未物化任何 mix 产物）"
        )

    measured = _candidate_token_totals(summaries)
    deep_domains = {
        domain for domain, quota in quotas.items()
        if cfg["quota_fill_policy"] == "deepen_to_quota"
        and effective_top_k[domain] < 1.0 and measured.get(domain, 0) < quota
    }
    deep_root = work_root / "_measured_deepfill"
    if deep_domains:
        log.info("s6 measured deep fill: %s", {
            domain: {"quota": quotas[domain], "candidate_tokens": measured.get(domain, 0)}
            for domain in sorted(deep_domains)})
        for domain in deep_domains:
            effective_top_k[domain] = 1.0
            thresholds[domain] = topk_threshold(
                load_quantile_table(out_root, domain, cfg["score_field"]), 1.0)
        _prepare_collect_work_root(deep_root, {
            **work_contract, "domains": sorted(deep_domains),
            "thresholds": {domain: thresholds[domain] for domain in sorted(deep_domains)},
        })
        deep_context = {**collect_ctx, "work_dir": str(deep_root), "thresholds": thresholds}
        deep_summaries = _scan(
            "s6_deepfill", [(task, deep_context) for task in tasks if task[2] in deep_domains],
            _collect_worker, workers, guard_factory("s6_deepfill"), initargs=(str(out_root),),
        )
        deep_errors = [row for row in deep_summaries if row.get("errors") or row.get("shard_fatal")]
        if deep_errors:
            _write_errors(deep_errors, deep_root / "_errors", "s6_deepfill")
            raise SystemExit(f"s6 deep fill: {len(deep_errors)} shard failures")
        summaries = [row for row in summaries if row.get("domain") not in deep_domains] + deep_summaries
    _assert_quota_capacity(out_root, mix_name, quotas, _candidate_token_totals(summaries),
                          int(cfg["max_total_shortfall_tokens"]), "candidate_capacity")

    # 逐域归并 parts → 质量序 token 前缀选取
    # E-15/E-13 内存预算守卫：归并峰值 ≈ 候选行 × ~120B（npz 全载 + concatenate
    # 翻倍 + lexsort 3 键临时）。超预算即 fail-fast 拒跑（E-6：护栏是干净报错，
    # 不是把上限当目标）——需分域分批归并或提高预算后重跑。
    cand_by_dom: dict[str, int] = {}
    for s in summaries:
        d = str(s.get("domain"))
        cand_by_dom[d] = cand_by_dom.get(d, 0) + int(s.get("candidates") or 0)
    for dom, n in cand_by_dom.items():
        est_bytes = n * MERGE_BYTES_PER_CANDIDATE
        if est_bytes > MERGE_MEMORY_BUDGET:
            raise SystemExit(
                f"s6 merge memory guard: domain {dom} 预估 {est_bytes / 1e9:.0f}GB "
                f"({n:,} 候选 × ~{MERGE_BYTES_PER_CANDIDATE}B) 超预算 "
                f"{MERGE_MEMORY_BUDGET / 1e9:.0f}GB——分域物化（--sources）或调大 "
                f"MERGE_MEMORY_BUDGET 后重跑（E-6/E-13）"
            )
    dom_list = sorted(cfg["domains"])
    selected: dict[str, dict[str, Any]] = {}
    for dom in dom_list:
        dom_dir = (deep_root if dom in deep_domains else work_root) / dom
        parts = sorted(dom_dir.glob("*.npz")) if dom_dir.is_dir() else []
        ids, comp, nt, up, src_idx, src_keys = load_candidate_parts(
            parts, cand_by_dom.get(dom, 0))
        cap = nominal_cap_tokens[dom]
        order = sort_key_quality(ids, comp) if len(ids) else np.array([], dtype=np.int64)
        # Candidate collection already enforces the requested/effective top-k
        # document band.  Apply only the domain token quota here.  The former
        # min(quota, top_k * pool_tokens) double-capped the selection using a
        # dimensionally invalid estimate and truncated long-document domains.
        limit = int(quotas[dom])
        count = candidate_prefix_count(nt[order], limit)
        sel = order[:count]
        sel_nt = nt[sel] if count else nt[:0]
        tok_by_src: Counter[str] = Counter()
        if count:
            src_i = src_idx[sel].astype(np.int64)
            for i, name in enumerate(src_keys):
                mask = src_i == i
                if mask.any():
                    tok_by_src[name] = int(sel_nt[mask].astype(np.uint64).sum())
        selected[dom] = {
            "ids": ids[sel] if count else ids[:0],
            "comp": comp[sel] if count else comp[:0],
            "nt": sel_nt,
            "up": up[sel] if count else up[:0],
            "src_keys": src_keys,
            "quota": int(quotas[dom]),
            "cap_tokens": cap,
            "requested_top_k": float(cfg["domains"][dom]["top_k"]),
            "effective_top_k": float(effective_top_k[dom]),
            "stats": {
                "candidate_docs": int(len(ids)),
                "candidate_tokens": int(nt.astype(np.uint64).sum()) if len(ids) else 0,
                "selected_docs": int(count),
                "selected_tokens": int(sel_nt.astype(np.uint64).sum()) if count else 0,
            },
            "selected_tokens_by_source": dict(sorted(tok_by_src.items())),
        }

    _assert_quota_capacity(
        out_root, mix_name, quotas,
        {domain: value["stats"]["selected_tokens"] for domain, value in selected.items()},
        int(cfg["max_total_shortfall_tokens"]), "selected_quality_prefix",
    )

    # mix_members 全表（domain, doc_id 字节序；doc_id 序是纯确定性总序）
    dom_codes_global = {d: i for i, d in enumerate(dom_list)}
    if dom_list:
        all_ids = np.concatenate([selected[d]["ids"] for d in dom_list])
        all_comp = np.concatenate([selected[d]["comp"] for d in dom_list])
        all_nt = np.concatenate([selected[d]["nt"] for d in dom_list])
        all_codes = np.concatenate(
            [np.full(len(selected[d]["ids"]), dom_codes_global[d], dtype=np.int64) for d in dom_list]
        )
        hi, lo = _ids_be_keys(all_ids)
        mem_order = np.lexsort((lo, hi, all_codes))
    else:
        all_ids = np.empty(0, dtype="V16")
        all_comp = np.empty(0, dtype=np.float32)
        all_nt = np.empty(0, dtype=np.uint32)
        all_codes = np.array([], dtype=np.int64)
        mem_order = np.array([], dtype=np.int64)
    mix_root.mkdir(parents=True, exist_ok=True)
    order_root.mkdir(parents=True, exist_ok=True)
    members_table = pa.table(
        {
            "doc_id": _doc_id_chunked(all_ids[mem_order]),
            "domain": _domain_array(all_codes[mem_order], dom_list),
            "composite": pa.array(np.ascontiguousarray(all_comp[mem_order]), type=pa.float32()),
            "n_tokens": pa.array(np.ascontiguousarray(all_nt[mem_order]), type=pa.uint32()),
        },
        schema=MEMBERS_SCHEMA,
    )
    tmp = members_path.with_name(members_path.name + f".tmp-{os.getpid()}")
    # E-45：parquet 单列块上限 2GB——doc_id hex 串 32B/行，members ~2 亿行 = 6.4GB
    # 单次 write_table 直接 ArrowInvalid（2026-10-10 实测）。row_group_size 50M 行
    # → 每列块 1.6GB < 2GB（4 个行组）；读侧契约不变（pq.read_table 透明跨行组）。
    pq.write_table(members_table, tmp, compression="zstd", row_group_size=50_000_000)
    os.replace(tmp, members_path)

    # 课程两段物化
    cur = cfg["curriculum"]
    frac = float(cur["stage1_fraction"])
    s1_ids = np.empty(0, dtype="V16")
    s1_nt = np.empty(0, dtype=np.uint32)
    s1_up = np.empty(0, dtype=np.float64)
    s1_codes = np.array([], dtype=np.int64)
    s2_ids = np.empty(0, dtype="V16")
    s2_nt = np.empty(0, dtype=np.uint32)
    s2_up = np.empty(0, dtype=np.float64)
    s2_codes = np.array([], dtype=np.int64)
    with_stage1 = bool(cur["enabled"] and frac > 0.0)
    for d in dom_list:
        sel = selected[d]
        if not len(sel["ids"]):
            continue
        if with_stage1:
            okey = stage1_order_key(sel["ids"], sel["up"])
            cum = np.cumsum(sel["nt"][okey].astype(np.uint64))
            cut = int(np.searchsorted(cum, frac * int(cum[-1]), side="right")) if len(cum) else 0
            s1_ids = np.concatenate([s1_ids, sel["ids"][okey[:cut]]])
            s1_nt = np.concatenate([s1_nt, sel["nt"][okey[:cut]]])
            s1_up = np.concatenate([s1_up, sel["up"][okey[:cut]]])
            s1_codes = np.concatenate([s1_codes, np.full(cut, dom_codes_global[d], dtype=np.int64)])
            tail = okey[cut:]
        else:
            tail = np.arange(len(sel["ids"]), dtype=np.int64)
        s2_ids = np.concatenate([s2_ids, sel["ids"][tail]])
        s2_nt = np.concatenate([s2_nt, sel["nt"][tail]])
        s2_up = np.concatenate([s2_up, sel["up"][tail]])
        s2_codes = np.concatenate(
            [s2_codes, np.full(len(tail), dom_codes_global[d], dtype=np.int64)]
        )
    stage_stats: dict[str, Any] = {}
    stage1_path = order_root / f"{mix_name}_stage1.parquet"
    stage2_path = order_root / f"{mix_name}_stage2.parquet"
    if with_stage1:
        if len(s1_ids):
            hi, lo = _ids_be_keys(s1_ids)
            o = np.lexsort((
                lo, hi, s1_codes, np.where(np.isnan(s1_up), np.inf, s1_up),
            ))
            s1_ids, s1_nt, s1_up, s1_codes = s1_ids[o], s1_nt[o], s1_up[o], s1_codes[o]
        _write_order_table(
            stage1_path, s1_ids, s1_codes, dom_list, s1_nt, s1_up
        )
        stage_stats["stage1"] = {
            "docs": int(len(s1_ids)),
            "tokens": int(s1_nt.astype(np.uint64).sum()) if len(s1_nt) else 0,
            "ordering": "upstream edu 低→高（NaN 殿后），跨域再按 domain/doc_id 决胜",
        }
    else:
        # A config change from the legacy two-stage order to the random baseline must
        # not leave a stale stage1 table that a glob-based consumer could replay.
        stage1_path.unlink(missing_ok=True)
    if len(s2_ids):
        o = stage2_order_key(s2_ids)
        s2_ids, s2_nt, s2_up, s2_codes = s2_ids[o], s2_nt[o], s2_up[o], s2_codes[o]
    _write_order_table(
        stage2_path, s2_ids, s2_codes, dom_list, s2_nt, s2_up
    )
    stage_stats["stage2"] = {
        "docs": int(len(s2_ids)),
        "tokens": int(s2_nt.astype(np.uint64).sum()) if len(s2_nt) else 0,
        "ordering": "doc_id 字节序 = blake2b 均匀哈希序（确定性全局打乱）",
    }

    # Close the time-of-check/time-of-use window.  Upstream stages may be run by
    # another orchestrator while a long S6 collect is in flight.  Publishing a
    # manifest in that case would certify a mix assembled from multiple input
    # generations.  Recompute the exact snapshot before the commit marker and
    # discard the uncommitted outputs on any drift.
    final_upstream_snapshot = _s6_upstream_snapshot(
        out_root, tasks, cfg["score_field"]
    )
    if final_upstream_snapshot["snapshot_sha256"] != upstream_snapshot["snapshot_sha256"]:
        members_path.unlink(missing_ok=True)
        stage1_path.unlink(missing_ok=True)
        stage2_path.unlink(missing_ok=True)
        raise SystemExit(
            "s6 upstream changed during materialization; uncommitted members/order "
            "were removed. Wait for upstream writers to finish, then rerun."
        )

    manifest = {
        "mix_name": mix_name,
        "included_sources": sorted(sources) if sources is not None else sorted(profiles),
        "config_echo": echo,
        "upstream_snapshot_sha256": upstream_snapshot["snapshot_sha256"],
        "upstream_snapshot": upstream_snapshot,
        "config_path": cfg.get("_config_path"),
        "score_field": cfg["score_field"],
        "seed": cfg["seed"],
        "val_global_summary": "val_sets/_val_global_summary.json",
        "thresholds": {d: round(t, 8) for d, t in thresholds.items()},
        "requested_top_k": {
            d: float(cfg["domains"][d]["top_k"]) for d in dom_list
        },
        "effective_top_k": {d: float(effective_top_k[d]) for d in dom_list},
        "quota_fill_policy": cfg["quota_fill_policy"],
        "max_total_shortfall_tokens": int(cfg["max_total_shortfall_tokens"]),
        "quotas_tokens": {d: selected[d]["quota"] for d in dom_list},
        "cap_tokens": {d: selected[d]["cap_tokens"] for d in dom_list},
        "selected": {d: selected[d]["stats"] for d in dom_list},
        "selected_tokens_by_source": {d: selected[d]["selected_tokens_by_source"] for d in dom_list},
        "curriculum": cur,
        "stage_stats": stage_stats,
        "members": f"mix_members/{mix_name}.parquet",
        "order_tables": [
            *([stage1_path.name] if with_stage1 else []),
            stage2_path.name,
        ],
        "tokenize": "doc_id 顺序表已物化；tokenize 由 pretrain 阶段执行（本模块不落 .bin）",
    }
    if int((cfg.get("goldrule") or {}).get("per_domain_quota", 0)):
        manifest["goldrule_research"] = f"goldrule_research/{mix_name}.parquet"

    # 全局 summary（drop 原因统计：s5 决策链 + 分位数截断 + 配额缺口）
    per_domain: dict[str, Any] = {}
    for d in dom_list:
        st = selected[d]["stats"]
        quota = selected[d]["quota"]
        sel_tokens = st["selected_tokens"]
        # Use the exact release task roster.  The global S5 summary may also
        # contain deliberately excluded sources and must not leak their counts
        # into a source-scoped release certificate.
        s5_dom = pool.get(d) or {}
        drops = {
            k[len("decisions."):]: int(v)
            for k, v in s5_dom.items() if k.startswith("decisions.")
        }
        per_domain[d] = {
            "quota_tokens": quota,
            "pool_docs": int(s5_dom.get("docs") or 0),
            "pool_tokens": int(s5_dom.get("tokens") or 0),
            "drop_s2_exact_dup": int(drops.get("drop_s2_exact_dup") or 0),
            "drop_s3_near_dup": int(drops.get("drop_s3_near_dup") or 0),
            "drop_decontam": int(drops.get("drop_decontam") or 0),
            "candidate_docs": st["candidate_docs"],
            "candidate_tokens": st["candidate_tokens"],
            "selected_docs": st["selected_docs"],
            "selected_tokens": sel_tokens,
            "quality_truncated_tokens": st["candidate_tokens"] - sel_tokens,
            "quota_shortfall_tokens": max(0, quota - sel_tokens),
            "val_tokens": val_summary["per_domain"].get(d, {}).get("val_tokens", 0),
            "selected_tokens_by_source": selected[d]["selected_tokens_by_source"],
        }
    total_quota_shortfall = sum(v["quota_shortfall_tokens"] for v in per_domain.values())
    max_total_shortfall = int(cfg["max_total_shortfall_tokens"])
    if total_quota_shortfall > max_total_shortfall:
        members_path.unlink(missing_ok=True)
        stage1_path.unlink(missing_ok=True)
        stage2_path.unlink(missing_ok=True)
        raise SystemExit(
            f"s6 quota shortfall {total_quota_shortfall:,} exceeds configured maximum "
            f"{max_total_shortfall:,}; uncommitted members/order were removed"
        )
    payload = {
        "stage": "s6",
        "mix_name": mix_name,
        "included_sources": sorted(sources) if sources is not None else sorted(profiles),
        "config_echo": echo,
        "upstream_snapshot_sha256": upstream_snapshot["snapshot_sha256"],
        "total_tokens_target": int(cfg["total_tokens"]),
        "total_selected_tokens": sum(v["selected_tokens"] for v in per_domain.values()),
        "total_quota_shortfall_tokens": total_quota_shortfall,
        "max_total_shortfall_tokens": max_total_shortfall,
        "quota_fill_policy": cfg["quota_fill_policy"],
        "curriculum": stage_stats,
        "per_domain": per_domain,
        "collect_shards": len(summaries),
        "collect_errors": 0,
    }
    # goldrule 研究子集归并：collect 期确定性 hash 抽样的候选分数快照，研究轨专用，
    # 与训练集互斥性无要求（只是研究样本，不进训练 mix——doc_id 也未进 members）。
    gr_parts = sorted((work_root / "goldrule").glob("*.npz")) if (work_root / "goldrule").is_dir() else []
    if deep_domains:
        gr_parts = [path for path in gr_parts if path.name.split("__", 1)[0] not in deep_domains]
        gr_parts += sorted((deep_root / "goldrule").glob("*.npz"))
    min_total_rows = int(gr_cfg.get("min_total_rows", 0))
    if not gr_parts and min_total_rows:
        raise SystemExit(
            "goldrule research subset has 0 rows; "
            f"requires >= {min_total_rows:,}. Increase sample_rate/per_domain_quota."
        )
    if gr_parts:
        gr_root = out_root / "goldrule_research"
        gr_root.mkdir(parents=True, exist_ok=True)
        # Production has one part per shard. Close each NPZ promptly instead of
        # retaining roughly 13k open file descriptors until finalization.
        arrs: list[dict[str, np.ndarray]] = []
        for part in gr_parts:
            with np.load(part, allow_pickle=False) as archive:
                arrs.append({name: archive[name] for name in archive.files})
        gr_ids = np.concatenate([a["ids"] for a in arrs])
        gr_domains = np.concatenate([a["domain"] for a in arrs])
        gr_hashes = np.concatenate([a["sample_hash"] for a in arrs])
        per_domain_quota = int(gr_cfg.get("per_domain_quota", 0))
        selected_idx: list[np.ndarray] = []
        per_domain_available: dict[str, int] = {}
        per_domain_selected: dict[str, int] = {}
        for domain_raw in sorted(set(gr_domains.tolist())):
            domain = domain_raw.decode("utf-8")
            idx = np.flatnonzero(gr_domains == domain_raw)
            per_domain_available[domain] = int(len(idx))
            if per_domain_quota and len(idx) > per_domain_quota:
                ids_u64 = np.ascontiguousarray(gr_ids[idx]).view(np.uint64).reshape(-1, 2)
                order = np.lexsort((ids_u64[:, 1], ids_u64[:, 0], gr_hashes[idx]))
                idx = idx[order[:per_domain_quota]]
            selected_idx.append(idx)
            per_domain_selected[domain] = int(len(idx))
        take = np.sort(np.concatenate(selected_idx)) if selected_idx else np.empty(0, dtype=np.int64)
        if len(take) < min_total_rows:
            raise SystemExit(
                f"goldrule research subset has {len(take):,} rows after stratified caps; "
                f"requires >= {min_total_rows:,}. Increase sample_rate/per_domain_quota."
            )
        def _cat(name: str) -> np.ndarray:
            return np.concatenate([a[name] for a in arrs])[take]
        track_arrays = {name: _cat(name) for name in GOLDRULE_TRACK_COLUMNS}
        gr_sources = _cat("source")
        coverage: dict[str, dict[str, float | int]] = {}
        for name, values in track_arrays.items():
            finite = int(np.isfinite(values).sum())
            coverage[name] = {
                "finite": finite,
                "nonfinite": int(len(values) - finite),
                "fraction": finite / len(values) if len(values) else 0.0,
            }
        invalid_required = {
            name: int(coverage[name]["nonfinite"])
            for name in GOLDRULE_REQUIRED_FINITE
            if coverage[name]["nonfinite"]
        }
        if invalid_required:
            raise SystemExit(
                "goldrule current-track finite-value gate failed: "
                + ", ".join(f"{k}={v:,}" for k, v in invalid_required.items())
            )
        table_cols: dict[str, pa.Array] = {
            "doc_id": pa.array([x.tobytes().hex() for x in gr_ids[take]], pa.string()),
            "domain": pa.array(gr_domains[take].astype(str).tolist(), pa.string()),
            "source": pa.array(gr_sources.astype(str).tolist(), pa.string()),
            "n_tokens": pa.array(_cat("nt"), pa.uint32()),
            "sample_hash": pa.array(gr_hashes[take], pa.uint64()),
        }
        table_cols.update({
            name: pa.array(
                values,
                type=pa.float64() if name == "upstream_score" else pa.float32(),
                from_pandas=True,
            )
            for name, values in track_arrays.items()
        })
        gr_tbl = pa.table(table_cols)
        gr_path = gr_root / f"{mix_name}.parquet"
        tmp = gr_path.with_name(gr_path.name + f".tmp-{os.getpid()}")
        pq.write_table(gr_tbl, tmp, compression="zstd")
        os.replace(tmp, gr_path)
        gr_sha256 = _file_sha256(gr_path)
        manifest["goldrule_research_sha256"] = gr_sha256
        payload["goldrule_research"] = {
            "path": str(gr_path.relative_to(out_root)),
            "sha256": gr_sha256,
            "bytes": gr_path.stat().st_size,
            "rows": gr_tbl.num_rows,
            "sample_rate": float(gr_cfg.get("sample_rate", 0.0)),
            "per_domain_quota": per_domain_quota,
            "min_total_rows": min_total_rows,
            "per_domain_available": per_domain_available,
            "per_domain_selected": per_domain_selected,
            "per_source_selected": {
                source: int(count)
                for source, count in zip(
                    *np.unique(gr_sources.astype(str), return_counts=True)
                )
            },
            "track_coverage": coverage,
            "required_finite_tracks": list(GOLDRULE_REQUIRED_FINITE),
            "hybrid_v2_semantics": "effective current_v2 (legacy missing-upstream rows repaired)",
            "sampling": "blake2b-64('goldrule|' || doc_id_bytes16); hash-fraction then per-domain lowest-hash cap",
            "note": "全轨分数快照（研究轨专用，不进训练 mix）；doc_id 可回查 s1/s4_scored 正文",
        }
        log.info("s6 goldrule research subset: %d rows -> %s", gr_tbl.num_rows, gr_path)
    path = out_root / "stage_s6_global_summary.json"
    write_summary(path, payload)
    # stage_s6_global_summary.json is a shared convenience pointer overwritten
    # by the next mix.  Bind only mix-owned artifacts in this manifest.
    published_paths = [members_path, stage2_path]
    if with_stage1:
        published_paths.append(stage1_path)
    if manifest.get("goldrule_research"):
        published_paths.append(out_root / str(manifest["goldrule_research"]))
    manifest["published_artifacts"] = [
        _published_artifact(artifact_path, out_root)
        for artifact_path in sorted(published_paths)
    ]
    # Manifest is the generation commit marker and must be published last,
    # after members, orders, C4 research data, and the global summary exist.
    write_summary(manifest_path, manifest)
    shutil.rmtree(work_root, ignore_errors=True)
    log.info("stage s6 global summary: %s", path)
    return path


# ---------------------------------------------------------------- driver 入口

def run_s6(
    out_root: Path,
    profiles: dict[str, dict[str, Any]],
    mix_config: str | Path,
    workers: int,
    guard_factory: Any,
    sources: list[str] | None = None,
) -> Path:
    cfg = load_mix_config(mix_config)
    sources = _resolve_release_sources(cfg, profiles, sources)
    errors = gate_errors(out_root, profiles, cfg, sources)
    assert_gates(errors)
    log.info("S6 硬门全部通过（① s4_scored 覆盖 ② decontam 完成 ③ ndv2 lookup ④ val 留出 ⑤ s5 覆盖）")
    return materialize(out_root, profiles, cfg, workers, guard_factory, sources)


def run_s6_val(
    out_root: Path,
    profiles: dict[str, dict[str, Any]],
    mix_config: str | Path,
    workers: int,
    guard_factory: Any,
    sources: list[str] | None = None,
) -> Path:
    cfg = load_mix_config(mix_config)
    sources = _resolve_release_sources(cfg, profiles, sources)
    s5_global = out_root / "stage_s5_global_summary.json"
    if not s5_global.is_file():
        raise SystemExit(f"s6_val 依赖 s5 全局 summary：{s5_global} 缺失（先跑 --stage s5）")
    gs = json.loads(s5_global.read_text("utf-8"))
    if int(gs.get("shard_errors") or 0):
        raise SystemExit("s6_val 依赖 s5 无错误分片：stage_s5_global_summary shard_errors ≠ 0")
    current_contract = s5_global_input_contract(out_root, resolve_tokenizer_path())
    if gs.get("input_contract") != current_contract:
        raise SystemExit("s6_val 依赖当前上游生成的 s5；input_contract 已过期（先重跑 --stage s5）")
    decontam_global = out_root / "stage_decontam_global_summary.json"
    if not decontam_global.is_file():
        raise SystemExit("s6_val 依赖已发布的全文 decontam 全局 summary")
    from .decontam import release_readiness_errors

    readiness = release_readiness_errors(
        json.loads(decontam_global.read_text(encoding="utf-8"))
    )
    if readiness:
        raise SystemExit("s6_val decontam 门禁失败：" + "; ".join(readiness))
    ensure_val_sets(out_root, profiles, cfg, workers, guard_factory, sources)
    path = out_root / "val_sets" / "_val_global_summary.json"
    log.info("s6_val global summary: %s", path)
    return path
