from __future__ import annotations

import argparse
import gc
import hashlib
import heapq
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

DEFAULT_GOLDSET_CONFIG = (
    Path(__file__).resolve().parents[3] / "configs" / "scorer_goldset.yaml"
)

HQ_LABEL = "__label__hq"
LQ_LABEL = "__label__lq"
GOLD_COLUMNS = ("doc_id", "text", "label", "source", "rule")
GOLD_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("label", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("rule", pa.string(), nullable=False),
    ]
)


def load_goldset_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_GOLDSET_CONFIG
    if not p.is_file():
        raise FileNotFoundError(f"scorer goldset config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict) or not cfg.get("positive") or not cfg.get("negative"):
        raise ValueError(f"scorer goldset config invalid: {p}")
    return cfg


def normalize_scorer_text(text: str, max_chars: int) -> str:
    # 2026-09-26: CJK 逐字切分——fasttext 按空白分词，中文整句=单个词完全无特征
    # （wikipedia-zh LOO AUC 0.289 反向排名的结构根因）。逐字切分后中文成为
    # 字符 n-gram 序列，与 lid 检测器同款处理。训练/推理共用本函数，两侧同步生效。
    raw = text[: max_chars * 2]  # CJK 插空格后膨胀，先放宽截断
    out: list[str] = []
    n = 0
    for ch in raw.lower():
        if n >= max_chars:
            break
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0xF900 <= code <= 0xFAFF:
            out.append(" ")
            out.append(ch)
            out.append(" ")
            n += 1
        else:
            out.append(ch)
            n += 1
    return " ".join("".join(out).split())


def scorer_windows(text: str, max_chars: int, long_doc_chars: int) -> list[str]:
    """打分窗（fasttext 轨；BERT 轨在 score_bert 用 token 窗同策略）。

    短文档（归一化后 ≤ long_doc_chars）单窗 = 头部 max_chars 字符；
    长文档 head/mid/tail 三窗各 max_chars 字符，均分。2026-09-20 用户裁定。
    """
    norm = normalize_scorer_text(text, 1 << 30)
    if len(norm) <= long_doc_chars:
        return [norm[:max_chars]]
    half = max_chars // 2
    mid = len(norm) // 2
    return [norm[:max_chars], norm[mid - half : mid + half], norm[-max_chars:]]


def _rule_rng(seed: int, key: str) -> random.Random:
    digest = hashlib.blake2b(f"yuformer-s4-goldset|{seed}|{key}".encode(), digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _s1_shards(out_root: Path, source: str) -> list[Path]:
    src_dir = out_root / "s1" / source
    if not src_dir.is_dir():
        return []
    return sorted(p for p in src_dir.rglob("*.parquet") if p.is_file())


def _audit_shards(out_root: Path) -> list[Path]:
    audit_root = out_root / "s1_audit"
    if not audit_root.is_dir():
        return []
    return sorted(p for p in audit_root.rglob("*.parquet") if p.is_file())


def _reservoir_sample(
    paths: list[Path],
    quota: int,
    rng: random.Random,
    predicate: Callable[[dict[str, Any]], bool],
    text_field: str,
    min_chars: int,
    upstream_min: float | None,
    upstream_max: float | None,
) -> list[dict[str, Any]]:
    reservoir: list[dict[str, Any]] = []
    seen = 0
    for path in paths:
        try:
            pf = pq.ParquetFile(path)
        except Exception:
            continue
        for batch in pf.iter_batches(batch_size=2048):
            cols = batch.to_pylist()
            for row in cols:
                text = row.get(text_field) or ""
                if len(text) < min_chars:
                    continue
                score = row.get("upstream_score")
                if upstream_min is not None and (score is None or score < upstream_min):
                    continue
                if upstream_max is not None and (score is None or score > upstream_max):
                    continue
                if not predicate(row):
                    continue
                seen += 1
                if len(reservoir) < quota:
                    reservoir.append(row)
                else:
                    j = rng.randrange(seen)
                    if j < quota:
                        reservoir[j] = row
    return reservoir


def _goldset_workers() -> int:
    env = os.environ.get("YUFORMER_GOLDSET_WORKERS")
    if env:
        return max(1, int(env))
    return max(1, min(32, os.cpu_count() or 1))


def _batch_mask(
    batch: pa.RecordBatch,
    text_field: str,
    min_chars: int,
    upstream_min: float | None,
    upstream_max: float | None,
) -> np.ndarray:
    """向量化过滤：返回本 batch 合格行的 bool mask（与逐行版语义一致：
    文本长度 ≥ min_chars；上游分越界/缺失淘汰；null 文本淘汰）。"""
    text_col = batch.column(text_field)
    mask = pc.greater_equal(pc.utf8_length(text_col), min_chars)
    mask = pc.and_(mask, pc.is_valid(text_col))
    if upstream_min is not None or upstream_max is not None:
        score_np = batch.column("upstream_score").to_numpy(zero_copy_only=False)
        keep = np.ones(len(batch), dtype=bool)
        if upstream_min is not None:
            keep &= score_np >= upstream_min
        if upstream_max is not None:
            keep &= score_np <= upstream_max
        mask = pc.and_(mask, pa.array(keep))
    return pc.fill_null(mask, False).to_numpy(zero_copy_only=False)


def _reservoir_worker_rows(
    spec: dict[str, Any],
) -> tuple[int, list[tuple[float, int, dict[str, Any]]]]:
    """带 reasons_any 规则（audit）的逐行过滤 + 堆采样路径。"""
    quota = int(spec["quota"])
    min_chars = int(spec["min_chars"])
    text_field = spec["text_field"]
    id_field = spec["id_field"]
    upstream_min = spec["upstream_min"]
    upstream_max = spec["upstream_max"]
    reasons_any = frozenset(spec["reasons_any"] or ())
    rng = random.Random(int(spec["rng_seed"]))
    # 本函数不产生引用环（list/tuple/dict/str 均为树形），关掉 GC 避免
    # "巨大常驻 reservoir × 高分配率" 触发反复 gen2 全量扫描（实测 ~100× 降速）
    gc.disable()
    heap: list[tuple[float, int, dict[str, Any]]] = []
    seen = 0
    for path_str in spec["paths"]:
        try:
            pf = pq.ParquetFile(path_str)
        except Exception:
            continue
        cols = [id_field, text_field]
        if upstream_min is not None or upstream_max is not None:
            cols.append("upstream_score")
        cols.append("reasons")
        for batch in pf.iter_batches(batch_size=4096, columns=cols):
            for row in batch.to_pylist():
                text = row.get(text_field) or ""
                if len(text) < min_chars:
                    continue
                score = row.get("upstream_score")
                if upstream_min is not None and (score is None or score < upstream_min):
                    continue
                if upstream_max is not None and (score is None or score > upstream_max):
                    continue
                if reasons_any.isdisjoint(row.get("reasons") or ()):
                    continue
                seen += 1
                neg_u = -rng.random()
                if len(heap) < quota:
                    heapq.heappush(
                        heap, (neg_u, seen, {"doc_id": row.get(id_field), "text": text})
                    )
                elif neg_u > heap[0][0]:
                    # 新 u < 当前最大 u（min-heap 存 -u，heap[0] 为最大 u）才替换；
                    # 落选行不建 dict（绝大多数行走此分支）
                    heapq.heapreplace(
                        heap, (neg_u, seen, {"doc_id": row.get(id_field), "text": text})
                    )
    return seen, heap


def _reservoir_worker(spec: dict[str, Any]) -> tuple[int, list[tuple[float, int, dict[str, Any]]]]:
    """均匀 key 流式水库抽样：第 ℓ 个合格行取 u=rng.random()，保留 quota 个最小 u。

    N 个 iid 均匀随机变量的最小 quota 个 = 精确无放回等概率抽样；且全局按 u 排名
    与 worker 内按 u 排名一致，各 worker 对全局 top-quota 的贡献必为其本地
    top-k'（k'≤quota）⊆ 本地 top-quota，故 _merge_worker_reservoirs 精确可合并。
    u 来自 worker 专属 rng 子流（blake2b(seed|规则名|w{i})），各 worker 独立。
    返回 (合格行总数 n_i, [(-u, 本地序号 ℓ, {doc_id, text}), ...])；heap 第二元
    ℓ 在 worker 内唯一，tuple 比较永不落到 row dict。只投影所需列减少解码量。
    无 reasons_any 时用向量化过滤（arrow compute 求 mask，只物化合格行），
    与逐行版逐字节同序、同一 rng 消费序，结果完全一致。
    """
    quota = int(spec["quota"])
    min_chars = int(spec["min_chars"])
    text_field = spec["text_field"]
    id_field = spec["id_field"]
    upstream_min = spec["upstream_min"]
    upstream_max = spec["upstream_max"]
    reasons_any = frozenset(spec["reasons_any"] or ())
    if reasons_any:
        return _reservoir_worker_rows(spec)
    rng = random.Random(int(spec["rng_seed"]))
    gc.disable()
    heap: list[tuple[float, int, dict[str, Any]]] = []
    seen = 0
    cols = [id_field, text_field]
    if upstream_min is not None or upstream_max is not None:
        cols.append("upstream_score")
    for path_str in spec["paths"]:
        try:
            pf = pq.ParquetFile(path_str)
        except Exception:
            continue
        for batch in pf.iter_batches(batch_size=8192, columns=cols):
            idx = np.flatnonzero(
                _batch_mask(batch, text_field, min_chars, upstream_min, upstream_max)
            )
            if idx.size == 0:
                continue
            take = pa.array(idx)
            ids = pc.take(batch.column(id_field), take).to_pylist()
            texts = pc.take(batch.column(text_field), take).to_pylist()
            for doc_id, text in zip(ids, texts):
                seen += 1
                neg_u = -rng.random()
                if len(heap) < quota:
                    heapq.heappush(heap, (neg_u, seen, {"doc_id": doc_id, "text": text or ""}))
                elif neg_u > heap[0][0]:
                    heapq.heapreplace(
                        heap, (neg_u, seen, {"doc_id": doc_id, "text": text or ""})
                    )
    return seen, heap


def _reservoir_worker_process(spec: dict[str, Any], out_path: str) -> None:
    """子进程入口：采样结果落盘（parquet），不经 mp 管道回传大对象——管道回传
    大 payload 会触发 SimpleQueue._wlock 死锁（2026-09-26 实测，mp.Pool 方案
    在 32 worker × 大批量结果下复现：worker 卡 outqueue.put，父进程卡 get）。"""
    n_i, heap = _reservoir_worker(spec)
    table = pa.table(
        {
            "neg_u": pa.array([-k for k, _, _ in heap], type=pa.float64()),
            "pos": pa.array([p for _, p, _ in heap], type=pa.int64()),
            "doc_id": pa.array([r["doc_id"] for _, _, r in heap], type=pa.string()),
            "text": pa.array([r["text"] for _, _, r in heap], type=pa.string()),
        }
    )
    pq.write_table(table, out_path, compression="zstd")
    # n_i 写入 sidecar（merge 不用，但保留供调试/审计）
    Path(out_path).with_suffix(".n").write_text(str(n_i))


def _load_worker_reservoir(
    path: Path,
) -> list[tuple[float, int, dict[str, Any]]]:
    t = pq.read_table(path, columns=["neg_u", "pos", "doc_id", "text"])
    return [
        (-u, pos, {"doc_id": doc_id, "text": text})
        for u, pos, doc_id, text in zip(
            t.column("neg_u").to_pylist(),
            t.column("pos").to_pylist(),
            t.column("doc_id").to_pylist(),
            t.column("text").to_pylist(),
        )
    ]


def _merge_worker_reservoirs(
    parts: list[list[tuple[float, int, dict[str, Any]]]],
    quota: int,
) -> list[dict[str, Any]]:
    """确定性精确合并：全部 worker reservoir 按 u 升序取最小 quota 个。

    全局 top-quota（按 u）的每个 worker 贡献 ⊆ 该 worker 返回的本地 top-quota，
    故合并结果与单流全局"最小 quota 个 u"完全一致（等概率抽样的并行实现）。
    最终按 (worker 序 = shard 分片序, 本地序号 ℓ) 排序输出，跨次运行逐行确定。
    """
    candidates: list[tuple[float, int, int, dict[str, Any]]] = []
    for i, items in enumerate(parts):
        for neg_u, pos, row in items:
            candidates.append((-neg_u, i, pos, row))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]))
    selected = sorted(candidates[:quota], key=lambda x: (x[1], x[2]))
    return [row for _, _, _, row in selected]


def _rule_worker_specs(
    paths: list[Path],
    quota: int,
    seed: int,
    key: str,
    text_field: str,
    id_field: str,
    min_chars: int,
    upstream_min: float | None,
    upstream_max: float | None,
    reasons_any: Iterable[str] | None,
    n_workers: int,
) -> list[dict[str, Any]]:
    """按分片字节数确定性均分（size 降序轮询），避免大分片堆在一个 worker 上拖尾；
    同一快照下文件大小稳定，分片方案跨次运行一致。worker 内仍按路径序扫描。"""
    by_size = sorted(paths, key=lambda p: (-p.stat().st_size, p.as_posix()))
    chunks = [[] for _ in range(n_workers)]
    for i, p in enumerate(by_size):
        chunks[i % n_workers].append(p)
    chunks = [sorted(c) for c in chunks if c]
    return [
        {
            "paths": [p.as_posix() for p in chunk],
            "quota": quota,
            "rng_seed": int.from_bytes(
                hashlib.blake2b(
                    f"yuformer-s4-goldset|{seed}|{key}|w{i}".encode(), digest_size=8
                ).digest(),
                "big",
            ),
            "text_field": text_field,
            "id_field": id_field,
            "min_chars": min_chars,
            "upstream_min": upstream_min,
            "upstream_max": upstream_max,
            "reasons_any": sorted(reasons_any) if reasons_any else None,
        }
        for i, chunk in enumerate(chunks)
    ]


def _parallel_reservoir(
    paths: list[Path],
    quota: int,
    seed: int,
    key: str,
    text_field: str,
    id_field: str,
    min_chars: int,
    upstream_min: float | None,
    upstream_max: float | None,
    reasons_any: Iterable[str] | None,
    n_workers: int,
    tmp_dir: Path,
) -> list[dict[str, Any]]:
    if not paths or quota <= 0:
        return []

    def predicate(row: dict[str, Any], reasons_any: Iterable[str] | None = reasons_any) -> bool:
        if not reasons_any:
            return True
        reasons = row.get("reasons") or []
        return any(r in reasons_any for r in reasons)

    if n_workers <= 1 or len(paths) < 2:
        # 单 worker 退化：保持旧单线程语义（同一 rng 驱动），输出统一瘦行
        rows = _reservoir_sample(
            paths, quota, _rule_rng(seed, key), predicate,
            text_field=text_field, min_chars=min_chars,
            upstream_min=upstream_min, upstream_max=upstream_max,
        )
        return [{"doc_id": r.get(id_field), "text": r.get(text_field) or ""} for r in rows]
    specs = _rule_worker_specs(
        paths, quota, seed, key, text_field, id_field, min_chars,
        upstream_min, upstream_max, reasons_any, n_workers,
    )
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_paths = [tmp_dir / f"{key}.w{i}.parquet" for i in range(len(specs))]
    # 每规则一次性 spawn n 个子进程（spawn 避免 pyarrow 线程池 × fork 风险），
    # 结果走磁盘不经管道；join 后按 exitcode fail-fast
    procs = [
        mp.get_context("spawn").Process(
            target=_reservoir_worker_process, args=(spec, out.as_posix())
        )
        for spec, out in zip(specs, out_paths)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    failed = [p.exitcode for p in procs if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"goldset workers failed for rule {key}: exitcode={failed}")
    parts = [_load_worker_reservoir(out) for out in out_paths]
    for out in out_paths:
        out.unlink()
        out.with_suffix(".n").unlink(missing_ok=True)
    return _merge_worker_reservoirs(parts, quota)


def _rule_shards(out_root: Path, rule: dict[str, Any]) -> tuple[list[Path], str, str]:
    source = str(rule["source"])
    if source == "@s1_audit":
        paths = _audit_shards(out_root)
        text_field, id_field = "text_head", "doc_id"
    else:
        paths = _s1_shards(out_root, source)
        text_field, id_field = "text", "doc_id"
    max_shards = rule.get("max_shards")
    if max_shards and len(paths) > int(max_shards):
        # 按路径 blake2b 稳定哈希选 shard 子集（跨 dump/年份无偏，同下载采样策略），
        # 再按路径排序扫描保证行序确定
        paths = sorted(paths, key=lambda p: hashlib.blake2b(p.as_posix().encode(), digest_size=16).hexdigest())
        paths = sorted(paths[: int(max_shards)])
    return paths, text_field, id_field


def build_goldset(
    cfg: dict[str, Any],
    out_root: Path,
    gold_dir: Path,
) -> dict[str, Any]:
    seed = int(cfg.get("seed", 0))
    min_chars = int(cfg.get("min_text_chars", 100))
    max_chars = int(cfg.get("max_text_chars", 20000))
    gold_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    rule_stats: dict[str, dict[str, Any]] = {}
    # 规则内 shard 列表按字节数均分到 spawn 子进程（每规则一批，结果落盘不经
    # 管道，见 _reservoir_worker_process）；worker 数=min(32, 核数)，可用
    # YUFORMER_GOLDSET_WORKERS 覆盖
    n_workers = _goldset_workers()
    tmp_dir = gold_dir / "_reservoir_tmp"
    # records 累积 ~1M dict 且分配率高，GC gen2 会反复全量扫描；本阶段无引用环，关掉
    gc_was_enabled = gc.isenabled()
    gc.disable()
    started_all = time.monotonic()
    try:
        for label, rules in (("hq", cfg["positive"]), ("lq", cfg["negative"])):
            for rule in rules:
                key = str(rule.get("rule") or f"{rule['source']}-{label}")
                paths, text_field, id_field = _rule_shards(out_root, rule)
                reasons_any = rule.get("reasons_any")
                started_rule = time.monotonic()
                sampled = _parallel_reservoir(
                    paths,
                    int(rule["quota"]),
                    seed,
                    key,
                    text_field=text_field,
                    id_field=id_field,
                    min_chars=min_chars,
                    upstream_min=rule.get("upstream_min"),
                    upstream_max=rule.get("upstream_max"),
                    reasons_any=reasons_any,
                    n_workers=n_workers,
                    tmp_dir=tmp_dir,
                )
                print(
                    f"[goldset] rule={key} label={label} shards={len(paths)} "
                    f"sampled={len(sampled)} sec={time.monotonic() - started_rule:.1f}",
                    flush=True,
                )
                for row in sampled:
                    # _parallel_reservoir 返回瘦行 {doc_id, text}（text 已按 text_field 取值）
                    records.append(
                        {
                            "doc_id": str(row.get("doc_id") or hashlib.blake2b(
                                f"{key}|{len(records)}".encode(), digest_size=16
                            ).hexdigest()),
                            "text": normalize_scorer_text(str(row.get("text") or ""), max_chars),
                            "label": label,
                            "source": str(rule["source"]),
                            "rule": key,
                        }
                    )
                rule_stats[key] = {
                    "label": label,
                    "source": str(rule["source"]),
                    "quota": int(rule["quota"]),
                    "sampled": len(sampled),
                    "shards_scanned": len(paths),
                    "satisfied": len(sampled) >= int(rule["quota"]),
                }
    finally:
        if gc_was_enabled:
            gc.enable()
        if tmp_dir.is_dir() and not any(tmp_dir.iterdir()):
            tmp_dir.rmdir()
    print(f"[goldset] rules done total_sec={time.monotonic() - started_all:.1f} "
          f"workers={n_workers}")

    shuffle_rng = _rule_rng(seed, "shuffle")
    shuffle_rng.shuffle(records)

    table = pa.table(
        {name: [r[name] for r in records] for name in GOLD_COLUMNS}, schema=GOLD_SCHEMA
    )
    gold_path = gold_dir / "gold.parquet"
    tmp = gold_path.with_name(gold_path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(gold_path)

    txt_path = gold_dir / "gold.fasttext.txt"
    tmp_txt = txt_path.with_name(txt_path.name + ".tmp")
    # 2026-09-25: fasttext train NaN 根因——s1_audit 负样本来自 S2 之前的未去重文档，
    # 精确重复行（同 text 同 label 大量重复 / 少数同 text 冲突 label）触发训练器数值崩溃。
    # 写训练文件时按归一化文本去重；冲突行（同文本双标签）两边都弃并计数。
    seen: dict[str, str | None] = {}
    n_dup = n_conflict = 0
    with tmp_txt.open("w", encoding="utf-8") as fh:
        for r in records:
            label_token = HQ_LABEL if r["label"] == "hq" else LQ_LABEL
            norm_text = normalize_scorer_text(r["text"], 1024)
            key = norm_text[:4096]
            if key in seen:
                if seen[key] is None or seen[key] != label_token:
                    # 冲突（或已标记冲突）的行：若首次冲突则计数，永不再写
                    if seen[key] is not None:
                        n_conflict += 1
                        seen[key] = None
                else:
                    n_dup += 1
                continue
            seen[key] = label_token
            fh.write(f"{label_token} {norm_text}\n")
    tmp_txt.replace(txt_path)
    n_written = sum(1 for v in seen.values() if v is not None)
    print(f"gold.fasttext.txt dedup: dropped_dup={n_dup} dropped_conflict={n_conflict} "
          f"kept={n_written}")

    counts = Counter(r["label"] for r in records)
    per_source = Counter(f"{r['label']}:{r['source']}" for r in records)
    summary = {
        "stage": "s4_goldset",
        "seed": seed,
        "records": len(records),
        "labels": dict(sorted(counts.items())),
        "per_source": dict(sorted(per_source.items())),
        "rules": {k: rule_stats[k] for k in sorted(rule_stats)},
        "gold_md5": hashlib.md5(gold_path.read_bytes()).hexdigest(),
        "txt_md5": hashlib.md5(txt_path.read_bytes()).hexdigest(),
        "skipped": False,
    }
    (gold_dir / "goldset.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8"
    )
    return summary


def _load_gold(gold_dir: Path) -> list[dict[str, Any]]:
    gold_path = gold_dir / "gold.parquet"
    if not gold_path.is_file():
        raise FileNotFoundError(f"gold set not found: {gold_path} (run goldset step first)")
    return pq.read_table(gold_path).to_pylist()


def predict_hq_proba(model: Any, texts: list[str], batch_size: int = 1000) -> np.ndarray:
    probs = np.zeros(len(texts), dtype=np.float64)
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        try:
            labels, pvals = model.predict(chunk, k=2)
        except ValueError:
            # numpy>=2 × fasttext 0.9.x 包装器不兼容（同 detectors.LangDetector）：
            # 批量走底层 f.predict 绕过。返回 (prob, label) 对列表。
            pairs = model.f.predict(list(chunk), 2, 0.0, "strict")
            labels = [[p[1] for p in ps] for ps in pairs]
            pvals = [[p[0] for p in ps] for ps in pairs]
        for j, (labs, pv) in enumerate(zip(labels, pvals)):
            d = dict(zip(labs, pv))
            probs[i + j] = d.get(HQ_LABEL, 0.0)
    return probs


def _auc_ap(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    out: dict[str, float] = {}
    out["auc"] = float(roc_auc_score(y_true, scores)) if len(set(y_true)) > 1 else float("nan")
    out["ap"] = float(average_precision_score(y_true, scores)) if len(set(y_true)) > 1 else float("nan")
    out["n"] = int(len(y_true))
    out["n_pos"] = int(int(y_true.sum()))
    out["acc_at_0.5"] = float((scores >= 0.5).astype(int).__eq__(y_true).mean())
    return out


def _train_fasttext(txt_path: Path, params: dict[str, Any]) -> Any:
    """Train in a fresh process and return a read-only loaded model.

    fastText 0.9.x can retain unstable native state across repeated supervised
    fits in one Python process.  Cross-validation performs several fits, so a
    process boundary is part of the reproducibility contract rather than a
    retry after NaN.
    """
    import fasttext

    kwargs = {
        "input": str(txt_path.resolve()),
        "dim": int(params.get("dim", 100)),
        "wordNgrams": int(params.get("wordNgrams", 2)),
        "epoch": int(params.get("epoch", 5)),
        "lr": float(params.get("lr", 0.1)),
        "minCount": int(params.get("minCount", 5)),
        "loss": str(params.get("loss", "softmax")),
        "thread": int(params.get("thread", 1)),
        "seed": int(params.get("seed", 0)),
        "verbose": 0,
    }
    fd, model_name = tempfile.mkstemp(prefix="yuformer-fasttext-", suffix=".bin")
    os.close(fd)
    model_path = Path(model_name)
    code = (
        "import fasttext,json,sys;"
        "m=fasttext.train_supervised(**json.loads(sys.argv[1]));"
        "m.save_model(sys.argv[2])"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code, json.dumps(kwargs, sort_keys=True), str(model_path)],
            capture_output=True,
            text=True,
            timeout=int(params.get("timeout_seconds", 3600)),
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-2000:]
            if "Encountered NaN" in detail:
                raise RuntimeError(
                    "fastText isolated training diverged with NaN "
                    f"(loss={kwargs['loss']!r}, lr={kwargs['lr']}, epoch={kwargs['epoch']}, "
                    f"seed={kwargs['seed']})."
                )
            raise RuntimeError(
                f"fastText isolated training failed with exit {completed.returncode}: {detail}"
            )
        return fasttext.load_model(str(model_path))
    finally:
        model_path.unlink(missing_ok=True)


def leave_one_source_cv(
    cfg: dict[str, Any],
    gold_dir: Path,
    out_json: Path | None = None,
) -> dict[str, Any]:
    seed = int(cfg.get("seed", 0))
    params = cfg.get("model_params", {})
    rows = _load_gold(gold_dir)
    pos_sources = sorted({r["source"] for r in rows if r["label"] == "hq"})
    lq_rows = [r for r in rows if r["label"] == "lq"]

    fold_summaries: dict[str, Any] = {}
    for held in pos_sources:
        train_rows = [r for r in rows if r["label"] == "lq" or r["source"] != held]
        held_hq = [r for r in rows if r["label"] == "hq" and r["source"] == held]
        rng = _rule_rng(seed, f"cv-test-{held}")
        k = len(held_hq)
        test_lq = list(lq_rows)
        rng.shuffle(test_lq)
        test_rows = held_hq + test_lq[:k]

        work = gold_dir / f"cv_{held}.train.txt"
        # 同 gold.fasttext.txt：fold 训练文件也去重，防 fasttext NaN（2026-09-25）
        seen_fold: set[str] = set()
        with work.open("w", encoding="utf-8") as fh:
            for r in train_rows:
                norm_text = normalize_scorer_text(r["text"], 1024)
                key = norm_text[:4096]
                if key in seen_fold:
                    continue
                seen_fold.add(key)
                fh.write(f"{HQ_LABEL if r['label'] == 'hq' else LQ_LABEL} {norm_text}\n")
        model = _train_fasttext(work, params)
        scores = predict_hq_proba(model, [normalize_scorer_text(r["text"], 1024) for r in test_rows])  # 训练/评估预处理一致（CJK）
        y_true = np.array([1 if r["label"] == "hq" else 0 for r in test_rows], dtype=int)
        metrics = _auc_ap(y_true, scores)
        metrics["held_source"] = held
        metrics["train_rows"] = len(train_rows)
        fold_summaries[held] = metrics
        work.unlink(missing_ok=True)

    aucs = [m["auc"] for m in fold_summaries.values() if m["auc"] == m["auc"]]
    result = {
        "stage": "s4_scorer_cv",
        "seed": seed,
        "training_process_isolation": "one_fresh_process_per_fold",
        "positive_sources": pos_sources,
        "folds": {k: fold_summaries[k] for k in sorted(fold_summaries)},
        "mean_auc": float(np.mean(aucs)) if aucs else None,
        "auc_target": float(cfg.get("auc_target", 0.85)),
        "target_met": bool(aucs and float(np.mean(aucs)) >= float(cfg.get("auc_target", 0.85))),
    }
    if not result["target_met"]:
        result["recommendation"] = (
            "mean leave-one-source AUC below target: expand negatives with stratified "
            "low/mid/high upstream-score tails (per-source quantile sampling) and/or add "
            "more positive sources; do NOT silently accept the classifier."
        )
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(
            json.dumps(result, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8"
        )
    return result


def train_final_model(
    cfg: dict[str, Any],
    gold_dir: Path,
    model_dir: Path,
) -> dict[str, Any]:
    params = cfg.get("model_params", {})
    txt_path = gold_dir / "gold.fasttext.txt"
    if not txt_path.is_file():
        raise FileNotFoundError(f"gold fasttext file not found: {txt_path}")
    model_dir.mkdir(parents=True, exist_ok=True)
    bin_path = model_dir / "yuformer_quality_scorer.bin"
    ftz_path = model_dir / "yuformer_quality_scorer.ftz"

    started = time.monotonic()
    model = _train_fasttext(txt_path, params)
    train_sec = time.monotonic() - started
    model.save_model(str(bin_path))

    quantized = False
    if cfg.get("quantize", True):
        model.quantize(input=None, retrain=False, cutoff=0, qnorm=False, dsub=2, verbose=0)
        model.save_model(str(ftz_path))
        quantized = True

    rows = _load_gold(gold_dir)
    eval_scores = predict_hq_proba(model, [normalize_scorer_text(r["text"], 1024) for r in rows])  # 训练/评估预处理一致（CJK）
    y_true = np.array([1 if r["label"] == "hq" else 0 for r in rows], dtype=int)
    overall = _auc_ap(y_true, eval_scores)
    lq_idx = [i for i, r in enumerate(rows) if r["label"] == "lq"]
    per_source: dict[str, dict[str, float]] = {}
    sources = sorted({r["source"] for r in rows})
    for src in sources:
        pos_idx = [i for i, r in enumerate(rows) if r["source"] == src and r["label"] == "hq"]
        if not pos_idx:
            continue
        idx = pos_idx + lq_idx
        per_source[src] = _auc_ap(y_true[idx], eval_scores[idx])

    summary = {
        "stage": "s4_scorer_train",
        "training_process_isolation": "fresh_process",
        "train_rows": len(rows),
        "model_params": {k: params[k] for k in sorted(params)},
        "train_sec": round(train_sec, 3),
        "bin_sha256": hashlib.sha256(bin_path.read_bytes()).hexdigest(),
        "bin_md5": hashlib.md5(bin_path.read_bytes()).hexdigest(),
        "bin_bytes": bin_path.stat().st_size,
        "quantized": quantized,
        "ftz_sha256": hashlib.sha256(ftz_path.read_bytes()).hexdigest() if quantized else None,
        "ftz_md5": hashlib.md5(ftz_path.read_bytes()).hexdigest() if quantized else None,
        "ftz_bytes": ftz_path.stat().st_size if quantized else None,
        "overall_auc": overall["auc"],
        "overall_ap": overall["ap"],
        "per_source": {k: per_source[k] for k in sorted(per_source)},
        "skipped": False,
    }
    (model_dir / "scorer_train.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8"
    )
    return summary


# 公共别名：BERT 档（bert_train.py / score_bert.py）复用同一 gold-set 构造、
# CV 划分与指标，禁止复制实现。
load_gold_rows = _load_gold
auc_ap = _auc_ap
rule_rng = _rule_rng


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="yuformer_data.scale.scorer_train")
    parser.add_argument("--config", default=str(DEFAULT_GOLDSET_CONFIG))
    parser.add_argument("--out-root", default="../data/processed")
    parser.add_argument("--gold-dir", default=None, help="default <out-root>/scorer_goldset")
    parser.add_argument("--model-dir", default="../data/models")
    parser.add_argument("--step", choices=["goldset", "cv", "train", "all"], default="all")
    args = parser.parse_args(argv)

    cfg = load_goldset_config(args.config)
    out_root = Path(args.out_root)
    gold_dir = Path(args.gold_dir) if args.gold_dir else out_root / "scorer_goldset"
    model_dir = Path(args.model_dir)

    if args.step in ("goldset", "all"):
        s = build_goldset(cfg, out_root, gold_dir)
        print(json.dumps({"goldset": {"records": s["records"], "labels": s["labels"]}}, sort_keys=True))
    if args.step in ("cv", "all"):
        r = leave_one_source_cv(cfg, gold_dir, gold_dir / "cv_metrics.json")
        print(json.dumps({"cv": {"mean_auc": r["mean_auc"], "target_met": r["target_met"]}}, sort_keys=True))
    if args.step in ("train", "all"):
        t = train_final_model(cfg, gold_dir, model_dir)
        print(json.dumps({"train": {"overall_auc": t["overall_auc"], "bin_sha256": t["bin_sha256"]}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
