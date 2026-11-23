"""并行 gold-set 采样的确定性 / 分布等价性 / 契约测试。

数学等价依据（随机 key 水库抽样 + top-k 可合并性）：
  每个合格行取 iid 均匀随机数 u，保留 quota 个最小 u —— N 个 iid 均匀变量的
  最小 quota 个即精确无放回等概率抽样（对称性），入选概率 = quota/N。
  全局按 u 排名与 worker 内按 u 排名一致：任一 worker 对全局 top-quota 的贡献
  必为其本地 top-k'（k'≤quota）⊆ 其返回的本地 top-quota，故按 shard 分片并行
  后合并与单流全局抽样数学等价。同 seed 重跑逐行一致（u 由 blake2b 子流确定）。
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from yuformer_data.scale.schema import S1_SCHEMA
from yuformer_data.scale.scorer_train import (
    _merge_worker_reservoirs,
    _reservoir_sample,
    _reservoir_worker,
    _rule_rng,
    _rule_worker_specs,
    build_goldset,
)

GOOD_TEXT = (
    "the quick brown fox jumps over the lazy dog while seventeen curious astronomers "
    "measure distant galaxies with remarkable patience and steady hands every night "
)

PARALLEL_WORKERS = 4


def _write_s1_shard(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    full = []
    for i, r in enumerate(rows):
        full.append(
            {
                "doc_id": r.get("doc_id", f"doc-{path.stem}-{i}"),
                "text": r.get("text", GOOD_TEXT + f"variant{i}"),
                "domain": r.get("domain", "web"),
                "source": r.get("source", "synt"),
                "lang": r.get("lang", "en"),
                "upstream_score": r.get("upstream_score"),
                "meta": "{}",
                "decision": "keep",
                "reasons": r.get("reasons", []),
                "metrics": "{}",
            }
        )
    pq.write_table(
        pa.table({k: [r[k] for r in full] for k in S1_SCHEMA.names}, schema=S1_SCHEMA), path
    )


def _write_audit_shard(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {
        "doc_id": [f"audit-{path.stem}-{i}" for i in range(len(rows))],
        "reasons": [r["reasons"] for r in rows],
        "text_head": [r["text"][:200] for r in rows],
        "source": [r.get("source", "synt") for r in rows],
    }
    pq.write_table(pa.table(cols), path)


def _make_out_root(tmp_path: Path, n_shards: int, rows_per_shard: int) -> Path:
    out_root = tmp_path / "processed"
    for s in range(n_shards):
        _write_s1_shard(
            out_root / "s1" / "src-a" / f"shard-{s:03d}.parquet",
            [
                {"text": GOOD_TEXT + f"doc{s}x{i}y", "upstream_score": 0.5}
                for i in range(rows_per_shard)
            ],
        )
    return out_root


def _paths(out_root: Path, source: str = "src-a") -> list[Path]:
    return sorted((out_root / "s1" / source).rglob("*.parquet"))


@pytest.fixture
def workers(monkeypatch):
    monkeypatch.setenv("YUFORMER_GOLDSET_WORKERS", str(PARALLEL_WORKERS))
    return PARALLEL_WORKERS


def _run_parallel_inproc(out_root: Path, quota: int, seed: int, workers: int, key: str = "k"):
    """进程内跑完整并行采样核心（分片→worker 采样→merge），与生产路径同代码，
    仅不经子进程 spawn——用于高频重复（统计检验）而省启动开销；mp 集成由
    build_goldset 端到端测试覆盖。"""
    paths = _paths(out_root)
    specs = _rule_worker_specs(
        paths, quota, seed, key, "text", "doc_id", 50, None, None, None, workers
    )
    parts = [_reservoir_worker(spec)[1] for spec in specs]
    return _merge_worker_reservoirs(parts, quota)


def test_same_seed_twice_row_identical(tmp_path, workers):
    out_root = _make_out_root(tmp_path, n_shards=10, rows_per_shard=150)
    cfg = {
        "seed": 4242,
        "min_text_chars": 50,
        "max_text_chars": 2000,
        "positive": [{"rule": "p-a", "source": "src-a", "quota": 200}],
        "negative": [{"rule": "n-a", "source": "src-a", "quota": 150, "upstream_max": 0.9}],
    }
    gold1 = tmp_path / "gold1"
    gold2 = tmp_path / "gold2"
    s1 = build_goldset(cfg, out_root, gold1)
    s2 = build_goldset(cfg, out_root, gold2)
    assert s1["gold_md5"] == s2["gold_md5"]
    assert s1["txt_md5"] == s2["txt_md5"]
    rows1 = pq.read_table(gold1 / "gold.parquet").to_pylist()
    rows2 = pq.read_table(gold2 / "gold.parquet").to_pylist()
    assert rows1 == rows2  # 逐行相同（含顺序）


def test_parallel_distribution_matches_theory(tmp_path, workers):
    """并行版 inclusion 频率 vs 理论 quota/N：χ² 拟合优度 + 单线程参考对拍。"""
    n_shards, rows_per_shard = 8, 150
    n_rows = n_shards * rows_per_shard
    quota = 120
    out_root = _make_out_root(tmp_path, n_shards, rows_per_shard)
    paths = _paths(out_root)
    doc_ids = [f"doc-{p.stem}-{i}" for p in paths for i in range(rows_per_shard)]

    repeats = 400
    freq_par = dict.fromkeys(doc_ids, 0)
    freq_seq = dict.fromkeys(doc_ids, 0)
    for r in range(repeats):
        seed = 1000 + r
        got = _run_parallel_inproc(out_root, quota, seed, workers)
        assert len(got) == quota  # quota 截断正确（N > quota）
        for row in got:
            freq_par[row["doc_id"]] += 1
        # 单线程参考实现（旧 _reservoir_sample），同 seed 不同实现
        def predicate(row):
            return True

        seq = _reservoir_sample(
            paths, quota, _rule_rng(seed, "k"), predicate,
            text_field="text", min_chars=50,
            upstream_min=None, upstream_max=None,
        )
        assert len(seq) == quota
        for row in seq:
            freq_seq[row["doc_id"]] += 1

    p = quota / n_rows
    expected = repeats * p
    chi2_par = sum((v - expected) ** 2 / expected for v in freq_par.values())
    chi2_seq = sum((v - expected) ** 2 / expected for v in freq_seq.values())
    df = n_rows - 1
    # χ²(df) 均值 df，std sqrt(2df)；留 6σ 余量，且两实现统计量应同量级
    limit = df + 6 * math.sqrt(2 * df)
    assert chi2_par < limit, f"parallel chi2={chi2_par:.0f} vs df={df}"
    assert chi2_seq < limit, f"sequential chi2={chi2_seq:.0f} vs df={df}"
    assert abs(chi2_par - chi2_seq) < 6 * math.sqrt(2 * df) + limit
    # 总体包含率贴近理论值
    assert abs(sum(freq_par.values()) / (repeats * n_rows) - p) < 0.01


def test_quota_truncation_and_shortfall(tmp_path, workers):
    out_root = _make_out_root(tmp_path, n_shards=6, rows_per_shard=100)
    full = _run_parallel_inproc(out_root, quota=500, seed=7, workers=workers)
    assert len(full) == 500  # N=600 > quota → 截断到 quota
    short = _run_parallel_inproc(out_root, quota=1000, seed=7, workers=workers)
    assert len(short) == 600  # quota > N → 全取（不重复、不越界）
    doc_ids = {r["doc_id"] for r in short}
    assert len(doc_ids) == 600  # 无放回


def test_audit_rule_parallel_deterministic(tmp_path, workers):
    """@s1_audit 规则（按文件分片 + reasons_any 过滤）并行化：确定性 + 计数契约。"""
    out_root = tmp_path / "processed"
    n_per_shard = 60
    for s in range(8):
        _write_audit_shard(
            out_root / "s1_audit" / "sub" / f"audit-{s}.parquet",
            [
                {
                    "text": (GOOD_TEXT + f"audits{s}i{i}")[:200],
                    "reasons": ["garbled_latin1_mojibake"] if i % 2 == 0 else ["other_reason"],
                }
                for i in range(n_per_shard)
            ],
        )
    cfg = {
        "seed": 99,
        "min_text_chars": 50,
        "max_text_chars": 2000,
        "positive": [{"rule": "p", "source": "src-x", "quota": 10}],
        "negative": [
            {
                "rule": "audit-par",
                "source": "@s1_audit",
                "quota": 100,
                "reasons_any": ["garbled_latin1_mojibake"],
            }
        ],
    }
    # src-x 不存在 → 正样本 0；audit 合格行 = 8*30=240，quota=100 → 截断 100
    gold_a = tmp_path / "gold-a"
    gold_b = tmp_path / "gold-b"
    sa = build_goldset(cfg, out_root, gold_a)
    sb = build_goldset(cfg, out_root, gold_b)
    rule = sa["rules"]["audit-par"]
    assert rule["sampled"] == 100 and rule["satisfied"] and rule["shards_scanned"] == 8
    assert sa["labels"] == {"lq": 100}  # Counter 不记录 0 计数标签
    assert sa["gold_md5"] == sb["gold_md5"] and sa["txt_md5"] == sb["txt_md5"]
    rows = pq.read_table(gold_a / "gold.parquet").to_pylist()
    assert all(r["source"] == "@s1_audit" and r["label"] == "lq" for r in rows)
    assert len({r["doc_id"] for r in rows}) == 100


def test_idempotent_and_summary_contract(tmp_path, workers):
    out_root = _make_out_root(tmp_path, n_shards=5, rows_per_shard=80)
    cfg = {
        "seed": 123,
        "min_text_chars": 50,
        "max_text_chars": 2000,
        "positive": [{"rule": "p1", "source": "src-a", "quota": 100}],
        "negative": [],
    }
    gold = tmp_path / "gold"
    s1 = build_goldset(cfg, out_root, gold)
    first_bytes = (gold / "gold.parquet").read_bytes()
    s2 = build_goldset(cfg, out_root, gold)
    assert s2["gold_md5"] == s1["gold_md5"]
    assert (gold / "gold.parquet").read_bytes() == first_bytes
    for key in ("stage", "seed", "records", "labels", "per_source", "rules", "gold_md5", "txt_md5", "skipped"):
        assert key in s1, f"summary missing {key}"
    assert s1["stage"] == "s4_goldset" and s1["skipped"] is False
    assert (gold / "gold.fasttext.txt").is_file()
    assert json.loads((gold / "goldset.summary.json").read_text())["records"] == s1["records"]
    # 环境变量覆盖生效（测试 fixture 设置了 4 worker）
    assert os.environ["YUFORMER_GOLDSET_WORKERS"] == str(workers)
