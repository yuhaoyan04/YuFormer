"""S4 BERT 回归档测试：冒烟训练 / 推理数值 / 幂等 / 降级 / composite_v2 融合。

GPU 不可用自动 skip（pytest.mark.skipif）；composite_v2 纯 CPU 测试不依赖 GPU。
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    import importlib.util

    import torch

    HAS_GPU = (
        torch.cuda.is_available()
        and importlib.util.find_spec("transformers") is not None
    )
except ImportError:
    torch = None  # type: ignore[assignment]
    HAS_GPU = False

from yuformer_data.scale.bert_train import (  # noqa: E402
    train_bert_classifier,
    train_final_bert,
)
from yuformer_data.scale.score import composite_v2_vector  # noqa: E402
from yuformer_data.scale.score_bert import (  # noqa: E402
    BERT_SCORED_SCHEMA,
    bert_output_paths,
    process_shard_bert,
)
from yuformer_data.scale.schema import S1_SCHEMA  # noqa: E402
from yuformer_data.scale.scorer_train import normalize_scorer_text  # noqa: E402

needs_gpu = pytest.mark.skipif(not HAS_GPU, reason="CUDA GPU not available")

GOOD_TEXT = (
    "The history of the steam engine stretches back into the first century AD; "
    "the aeolipile described by Hero of Alexandria is the earliest known example. "
    "In the following centuries the fundamentals were gradually developed."
)
BAD_TEXT = "<div><span>Ã Â ï¿½</span></div>" * 40

S3_SCHEMA = S1_SCHEMA.append(pa.field("s2_decision", pa.string(), nullable=False)).append(
    pa.field("s3_decision", pa.string(), nullable=False)
)


def _write_s1_shard(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    full = []
    for i, r in enumerate(rows):
        rec = {
            "doc_id": r.get("doc_id", f"doc-{path.stem}-{i}"),
            "text": r.get("text", GOOD_TEXT),
            "domain": r.get("domain", "web"),
            "source": r.get("source", "synt"),
            "lang": r.get("lang", "en"),
            "upstream_score": r.get("upstream_score"),
            "meta": "{}",
            "decision": "keep",
            "reasons": [],
            "metrics": "{}",
        }
        full.append(rec)
    pq.write_table(
        pa.table({k: [r[k] for r in full] for k in S1_SCHEMA.names}, schema=S1_SCHEMA), path
    )


def _gold_rows(per_class: int = 200) -> list[dict]:
    rows = []
    for i in range(per_class):
        rows.append(
            {
                "doc_id": f"hq-{i}",
                "text": normalize_scorer_text(GOOD_TEXT + f" variant {i}", 1024),
                "label": "hq",
                "source": "synt-pos",
                "rule": "pos",
            }
        )
        rows.append(
            {
                "doc_id": f"lq-{i}",
                "text": normalize_scorer_text(BAD_TEXT + f" variant {i}", 1024),
                "label": "lq",
                "source": "@s1_audit",
                "rule": "neg",
            }
        )
    return rows


def _write_s3_shard(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    full = []
    for i, r in enumerate(rows):
        rec = {
            "doc_id": r.get("doc_id", f"doc-{i}"),
            "text": r.get("text", GOOD_TEXT),
            "domain": r.get("domain", "web"),
            "source": r.get("source", "synt"),
            "lang": r.get("lang", "en"),
            "upstream_score": r.get("upstream_score"),
            "meta": "{}",
            "decision": "keep",
            "reasons": [],
            "metrics": "{}",
            "s2_decision": "keep",
            "s3_decision": r.get("s3_decision", "keep"),
        }
        full.append(rec)
    pq.write_table(
        pa.table({k: [r[k] for r in full] for k in S3_SCHEMA.names}, schema=S3_SCHEMA), path
    )


def _bert_cfg(model_dir: Path, **over) -> dict:
    cfg = {
        "bert_model": str(model_dir),
        "bert": {"max_len": 64, "max_score_chars": 512, "batch_size": 64, "score_quantum": 1e-4},
    }
    cfg["bert"].update(over)
    return cfg


# ---------- 纯 CPU：composite_v2 融合语义 ----------


def test_composite_v2_three_track_and_fallback():
    cfg = {
        "composite_v2": {"upstream_weight": 0.25, "fasttext_weight": 0.35, "bert_weight": 0.40},
        "upstream_normalize": {"synt": [0.0, 5.0]},
    }
    sources = ["synt", "synt", "synt", "synt"]
    upstream = [5.0, 5.0, None, 2.5]
    ft = np.array([0.8, 0.2, 0.8, 0.8])
    bert = np.array([0.9, 0.1, 0.6, np.nan])

    comp, nu, used = composite_v2_vector(sources, upstream, ft, bert, cfg)
    assert nu[[0, 1, 3]] == pytest.approx([1.0, 1.0, 0.5])
    assert np.isnan(nu[2])
    assert used.tolist() == [True, True, True, False]
    # 全轨可用：0.25*1 + 0.35*0.8 + 0.40*0.9
    assert comp[0] == pytest.approx(0.25 + 0.28 + 0.36, abs=1e-9)
    # 上游缺失：权重在 f/b 间重归一 → (0.35*0.8 + 0.40*0.6) / 0.75
    assert comp[2] == pytest.approx((0.35 * 0.8 + 0.40 * 0.6) / 0.75, abs=1e-9)
    # bert 行缺失且上游半权：(0.25*0.5 + 0.35*0.8) / 0.60
    assert comp[3] == pytest.approx((0.25 * 0.5 + 0.35 * 0.8) / 0.60, abs=1e-9)

    # 整轨 bert 缺失 → u:f 按原比例重归一（(0.25·1 + 0.35·0.8)/0.60）
    comp2, _, _ = composite_v2_vector(sources, upstream, ft, None, cfg)
    assert comp2[0] == pytest.approx((0.25 * 1.0 + 0.35 * 0.8) / 0.60, abs=1e-9)
    assert comp2[2] == pytest.approx(0.8, abs=1e-9)  # 只剩 fasttext

    # 无 composite_v2 配置 → 回退 legacy 双轨
    legacy = {"composite": {"upstream_weight": 0.5, "self_weight": 0.5},
              "upstream_normalize": {"synt": [0.0, 5.0]}}
    comp3, _, _ = composite_v2_vector(sources, upstream, ft, None, legacy)
    assert comp3[0] == pytest.approx(0.5 * 1.0 + 0.5 * 0.8, abs=1e-9)
    assert comp3[2] == pytest.approx(0.8, abs=1e-9)  # 上游缺失 → 只剩 fasttext
    assert comp3[3] == pytest.approx(0.5 * 0.5 + 0.5 * 0.8, abs=1e-9)  # 上游有效


# ---------- GPU 冒烟训练 ----------


@needs_gpu
def test_smoke_train_loss_decreases(tmp_path):
    rows = _gold_rows(200)
    params = {
        "max_len": 64,
        "batch_size": 32,
        "lr": 2e-5,
        "epochs": 1,
        "warmup_ratio": 0.05,
        "seed": 1234,
        "amp_fp16": True,
    }
    model, tokenizer, info = train_bert_classifier(rows, params, device="cuda")
    assert info["steps"] > 0
    assert info["last_loss"] < info["first_loss"], (
        f"loss did not decrease: first={info['first_loss']} last={info['last_loss']}"
    )
    # 保存/重载往返
    out_dir = tmp_path / "bert_model"
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    reloaded = AutoModelForSequenceClassification.from_pretrained(out_dir)
    tok2 = AutoTokenizer.from_pretrained(out_dir)
    assert reloaded.config.num_labels == 2 and tok2 is not None


@needs_gpu
def test_final_train_writes_summary(tmp_path):
    from yuformer_data.scale.bert_train import DEFAULT_BERT_PARAMS

    cfg = {"bert_train": dict(DEFAULT_BERT_PARAMS, epochs=1, batch_size=32, max_len=64)}
    summary = train_final_bert(cfg, _gold_dir(tmp_path), tmp_path / "yuformer_quality_bert")
    assert summary["train_rows"] == 400
    assert summary["overall_auc"] == summary["overall_auc"]  # not NaN
    assert (tmp_path / "yuformer_quality_bert" / "bert_train.summary.json").is_file()


def _gold_dir(tmp_path: Path) -> Path:
    from yuformer_data.scale.scorer_train import GOLD_COLUMNS, GOLD_SCHEMA

    rows = _gold_rows(200)
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({k: [r[k] for r in rows] for k in GOLD_COLUMNS}, schema=GOLD_SCHEMA),
        gold_dir / "gold.parquet",
    )
    return gold_dir


# ---------- GPU 推理：数值正确性 / 幂等 / 降级 ----------


@needs_gpu
def test_inference_numeric_and_quantized(tmp_path):
    model_dir = tmp_path / "yuformer_quality_bert"
    summary = train_final_bert(
        {"bert_train": {"epochs": 1, "batch_size": 32, "max_len": 64}},
        _gold_dir(tmp_path),
        model_dir,
    )
    assert summary["overall_auc"] >= 0.9  # 冒烟数据线性可分

    out_root = tmp_path / "processed"
    rows = [{"text": GOOD_TEXT * 2, "domain": "web"} for _ in range(50)] + [
        {"text": BAD_TEXT, "domain": "web", "s3_decision": "drop"} for _ in range(10)
    ]
    _write_s3_shard(out_root / "s3" / "synt" / "synt-0.parquet", rows)
    cfg = _bert_cfg(model_dir)

    s = process_shard_bert(
        out_root / "s3" / "synt" / "synt-0.parquet", out_root, "synt", "synt-0.parquet", cfg
    )
    assert s["in_rows"] == 60 and s["scored_rows"] == 50 and not s.get("errors")
    out_path, summary_path = bert_output_paths(out_root, "synt", "synt-0.parquet")
    got = pq.read_table(out_path).to_pylist()
    assert len(got) == 50
    scores = np.array([r["bert_score"] for r in got])
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
    # 量化到 1e-4
    assert np.allclose(scores / 1e-4, np.round(scores / 1e-4), atol=1e-6)
    # keep 的 GOOD_TEXT 分数应整体高于（被 drop 的）垃圾——用垃圾文本混入 keep 验证
    rows2 = [{"text": BAD_TEXT, "domain": "web"} for _ in range(50)]
    _write_s3_shard(out_root / "s3" / "synt" / "synt-1.parquet", rows2)
    s2 = process_shard_bert(
        out_root / "s3" / "synt" / "synt-1.parquet", out_root, "synt", "synt-1.parquet", cfg
    )
    got2 = pq.read_table(bert_output_paths(out_root, "synt", "synt-1.parquet")[0]).to_pylist()
    assert np.mean([r["bert_score"] for r in got]) > np.mean([r["bert_score"] for r in got2])


@needs_gpu
def test_inference_idempotent(tmp_path):
    model_dir = tmp_path / "yuformer_quality_bert"
    train_final_bert(
        {"bert_train": {"epochs": 1, "batch_size": 32, "max_len": 64}},
        _gold_dir(tmp_path),
        model_dir,
    )
    out_root = tmp_path / "processed"
    _write_s3_shard(
        out_root / "s3" / "synt" / "synt-0.parquet",
        [{"text": GOOD_TEXT * 2, "domain": "web"} for _ in range(30)],
    )
    cfg = _bert_cfg(model_dir)
    s1 = process_shard_bert(
        out_root / "s3" / "synt" / "synt-0.parquet", out_root, "synt", "synt-0.parquet", cfg
    )
    assert s1["skipped"] is False
    out_path, _ = bert_output_paths(out_root, "synt", "synt-0.parquet")
    first_md5 = hashlib.md5(out_path.read_bytes()).hexdigest()
    s2 = process_shard_bert(
        out_root / "s3" / "synt" / "synt-0.parquet", out_root, "synt", "synt-0.parquet", cfg
    )
    assert s2["skipped"] is True
    assert hashlib.md5(out_path.read_bytes()).hexdigest() == first_md5


@needs_gpu
def test_missing_s3_decision_fails_with_hint(tmp_path):
    model_dir = tmp_path / "yuformer_quality_bert"
    train_final_bert(
        {"bert_train": {"epochs": 1, "batch_size": 32, "max_len": 64}},
        _gold_dir(tmp_path),
        model_dir,
    )
    out_root = tmp_path / "processed"
    # 只有 s1 列（无 s2/s3 decision）→ 报错提示先跑 dedup
    _write_s3_shard(out_root / "s3" / "synt" / "synt-0.parquet", [{"text": GOOD_TEXT}])
    raw = pq.read_table(out_root / "s3" / "synt" / "synt-0.parquet")
    pq.write_table(raw.drop(["s2_decision", "s3_decision"]), out_root / "s3" / "synt" / "synt-0.parquet")
    cfg = _bert_cfg(model_dir)
    s = process_shard_bert(
        out_root / "s3" / "synt" / "synt-0.parquet", out_root, "synt", "synt-0.parquet", cfg
    )
    assert s.get("errors") and "dedup" in s.get("error_detail", "")


@needs_gpu
def test_score_pass_fuses_bert_and_recompute(tmp_path):
    """score.py 融合：s4_bert 存在时 composite_v2 三轨；bert 产物出现后触发重算。"""
    from yuformer_data.scale.score import process_shard_score, scored_output_paths
    from yuformer_data.scale.scorer_train import _train_fasttext

    model_dir = tmp_path / "yuformer_quality_bert"
    train_final_bert(
        {"bert_train": {"epochs": 1, "batch_size": 32, "max_len": 64}},
        _gold_dir(tmp_path),
        model_dir,
    )
    out_root = tmp_path / "processed"
    # fasttext 占位模型；使用 softmax 并通过生产 helper 隔离 native 训练进程。
    rng = np.random.default_rng(0)
    vocab = GOOD_TEXT.lower().split()
    junk = ["Ã", "Â", "ï¿½", "<div>", "<span>", "xxx", "yyy", "zzz", "http", "click"]
    lines = []
    for i in range(200):
        hq_words = rng.choice(vocab, size=120, replace=True)
        lq_words = rng.choice(junk, size=120, replace=True)
        lines.append(f"__label__hq {' '.join(hq_words)}")
        lines.append(f"__label__lq {' '.join(lq_words)}")
    txt = tmp_path / "ft.txt"
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ft_model = _train_fasttext(
        txt,
        {"dim": 16, "wordNgrams": 2, "epoch": 2, "lr": 0.02,
         "minCount": 1, "loss": "softmax", "thread": 1, "seed": 0},
    )
    ft_path = tmp_path / "model.ftz"
    ft_model.save_model(str(ft_path))

    scorer_cfg = {
        "model": str(ft_path),
        "composite": {"upstream_weight": 0.5, "self_weight": 0.5},
        "composite_v2": {"upstream_weight": 0.25, "fasttext_weight": 0.35, "bert_weight": 0.40},
        "upstream_normalize": {"synt": [0.0, 5.0]},
        "inference": {"max_score_chars": 512},
    }

    rows = [{"text": GOOD_TEXT * 2, "domain": "web", "upstream_score": 4.0} for _ in range(16)]
    _write_s1_shard(out_root / "s1" / "synt" / "synt-0.parquet", rows)
    doc_ids = [f"doc-synt-0-{i}" for i in range(16)]

    # pass 1：无 bert → composite_v2 = 双轨重归一，bert_score 为 null
    s1 = process_shard_score(out_root / "s1" / "synt" / "synt-0.parquet",
                             out_root, "synt", "synt-0.parquet", scorer_cfg, model=ft_model)
    assert s1["skipped"] is False and s1["bert_joined"] is False
    scored_path, _, _ = scored_output_paths(out_root, "synt", "synt-0.parquet")
    got = pq.read_table(scored_path).to_pylist()
    assert all(r["bert_score"] is None for r in got)
    ft_vals = np.array([r["fasttext_score"] for r in got])
    assert np.allclose(ft_vals, np.array([r["self_score"] for r in got]), atol=1e-6)
    for r in got:
        # 双轨重归一：(0.25·0.8 + 0.35·ft) / 0.60
        assert r["composite_v2"] == pytest.approx((0.25 * 0.8 + 0.35 * r["fasttext_score"]) / 0.60, abs=1e-5)
        assert r["upstream_norm"] == pytest.approx(0.8, abs=1e-6)

    # pass 2：bert 产物出现 → 同一 shard 重算（非 skip），三轨融合
    from yuformer_data.scale.score_bert import BERT_SCORED_SCHEMA

    bert_pq = out_root / "s4_bert" / "synt" / "synt-0.parquet.parquet"
    bert_pq.parent.mkdir(parents=True, exist_ok=True)
    bscores = np.round(np.linspace(0.1, 0.9, 16) / 1e-4) * 1e-4
    pq.write_table(
        pa.table({"doc_id": doc_ids, "bert_score": bscores.astype(np.float32)}, schema=BERT_SCORED_SCHEMA),
        bert_pq,
    )
    s2 = process_shard_score(out_root / "s1" / "synt" / "synt-0.parquet",
                             out_root, "synt", "synt-0.parquet", scorer_cfg, model=ft_model)
    assert s2["skipped"] is False and s2["bert_joined"] is True
    got2 = pq.read_table(scored_path).to_pylist()
    for i, r in enumerate(got2):
        assert r["bert_score"] is not None
        assert r["composite_v2"] == pytest.approx(
            0.25 * 0.8 + 0.35 * r["fasttext_score"] + 0.40 * r["bert_score"], abs=1e-5
        )
    # pass 3：幂等恢复
    s3 = process_shard_score(out_root / "s1" / "synt" / "synt-0.parquet",
                             out_root, "synt", "synt-0.parquet", scorer_cfg, model=ft_model)
    assert s3["skipped"] is True


@needs_gpu
def test_domain_allowlist_filter(tmp_path):
    model_dir = tmp_path / "yuformer_quality_bert"
    train_final_bert(
        {"bert_train": {"epochs": 1, "batch_size": 32, "max_len": 64}},
        _gold_dir(tmp_path),
        model_dir,
    )
    out_root = tmp_path / "processed"
    rows = [{"text": GOOD_TEXT, "domain": "web"} for _ in range(20)] + [
        {"text": GOOD_TEXT, "domain": "code"} for _ in range(20)
    ]
    _write_s3_shard(out_root / "s3" / "synt" / "synt-0.parquet", rows)
    cfg = _bert_cfg(model_dir, domain_allowlist=["web"])
    s = process_shard_bert(
        out_root / "s3" / "synt" / "synt-0.parquet", out_root, "synt", "synt-0.parquet", cfg
    )
    assert s["scored_rows"] == 20 and s["skipped_domain"] == 20


# ---------- 截断策略（2026-09-20 修订）：分窗 helper / fasttext 三窗 / BERT 三窗 / 校准 ----------


def test_scorer_windows_char_slices():
    from yuformer_data.scale.scorer_train import scorer_windows

    short = "the quick brown fox " * 20  # ~380 chars ≤ 4096 → 单窗
    wins = scorer_windows(short, max_chars=100, long_doc_chars=4096)
    assert len(wins) == 1 and wins[0] == " ".join(short.lower().split())[:100]

    long_text = "abcdefghij" * 2000  # 20k chars > 4096 → 三窗
    norm = " ".join(long_text.lower().split())
    h, m, t = scorer_windows(long_text, max_chars=100, long_doc_chars=4096)
    assert h == norm[:100]
    mid = len(norm) // 2
    assert m == norm[mid - 50 : mid + 50]
    assert t == norm[-100:]
    # 确定性
    assert scorer_windows(long_text, 100, 4096) == [h, m, t]


def test_token_windows():
    from yuformer_data.scale.bert_train import token_windows

    ids = list(range(100))
    assert token_windows(ids, 256) == [ids]  # 短序列单窗
    long_ids = list(range(1000))
    h, m, t = token_windows(long_ids, 256)
    assert h == long_ids[:256]
    assert m == long_ids[500 - 128 : 500 - 128 + 256]
    assert t == long_ids[-256:]
    assert token_windows(long_ids, 256) == [h, m, t]


def test_fasttext_window_scoring():
    """fasttext 轨三窗：头部好/尾部垃圾的长文，三窗分 < 头部窗分；短文档两模式一致。"""
    from yuformer_data.scale.score import predict_self_scores
    from yuformer_data.scale.scorer_train import _train_fasttext

    rng = np.random.default_rng(1)
    vocab = GOOD_TEXT.lower().split()
    junk = ["Ã", "Â", "ï¿½", "<div>", "xxx"]
    lines = []
    for _ in range(100):
        lines.append("__label__hq " + " ".join(rng.choice(vocab, size=150)))
        lines.append("__label__lq " + " ".join(rng.choice(junk, size=150)))
    p = Path("/tmp") / f"ft_win_{os.getpid()}.txt"
    p.write_text("\n".join(lines) + "\n")
    model = _train_fasttext(
        p,
        {"dim": 16, "wordNgrams": 2, "epoch": 2, "lr": 0.02,
         "minCount": 1, "loss": "softmax", "thread": 1, "seed": 1},
    )
    p.unlink()

    good_head = " ".join(rng.choice(vocab, size=800))       # ~4.5k 字符好文
    bad_tail = " ".join(rng.choice(junk, size=800))
    long_doc = good_head + " " + bad_tail                   # >4096 字符
    short_doc = " ".join(rng.choice(vocab, size=100))       # 短文档

    head_only = predict_self_scores(model, [long_doc, short_doc], 1024, 4096, False)
    windowed = predict_self_scores(model, [long_doc, short_doc], 1024, 4096, True)
    # 长文：尾部垃圾把三窗均分拉低
    assert windowed[0] < head_only[0]
    # 短文档：单窗，两模式数值一致
    assert windowed[1] == pytest.approx(head_only[1], abs=1e-6)
    assert np.all(windowed >= 0.0) and np.all(windowed <= 1.0)


@needs_gpu
def test_bert_window_scoring(tmp_path):
    """BERT 轨三窗：head-good/tail-bad 长文三窗分低于 head-only；短文档两模式一致。"""
    from yuformer_data.scale.score_bert import BertInferencer

    model_dir = tmp_path / "yuformer_quality_bert"
    train_final_bert(
        {"bert_train": {"epochs": 1, "batch_size": 32, "max_len": 64}},
        _gold_dir(tmp_path),
        model_dir,
    )
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).half().cuda().eval()

    rng = np.random.default_rng(2)
    vocab = GOOD_TEXT.lower().split()
    junk = ["Ã", "Â", "ï¿½", "<div>", "xxx"]
    good_head = " ".join(rng.choice(vocab, size=700))   # ~4k 字符
    bad_tail = " ".join(rng.choice(junk, size=700))
    long_doc = good_head + " " + bad_tail               # >4096 字符
    short_doc = " ".join(rng.choice(vocab, size=120))

    head_inf = BertInferencer(model, tok, "cuda", max_len=64, batch_size=32,
                              quantum=1e-4, max_chars=512, long_doc_chars=4096,
                              window_scoring=False)
    win_inf = BertInferencer(model, tok, "cuda", max_len=64, batch_size=32,
                             quantum=1e-4, max_chars=512, long_doc_chars=4096,
                             window_scoring=True)
    s_head = head_inf.score([long_doc, short_doc])
    s_win = win_inf.score([long_doc, short_doc])
    assert np.all(s_win >= 0) and np.all(s_win <= 1)
    assert s_win[0] < s_head[0]  # 尾部垃圾拉低三窗均分
    assert s_win[1] == pytest.approx(s_head[1], abs=5e-3)  # 短文档单窗：fp16 + 窗尾无 SEP，容差放宽
    head_inf.shutdown()
    win_inf.shutdown()


@needs_gpu
def test_truncation_calibration(tmp_path):
    """校准脚本小样本端到端：60 篇分层抽样 → 三种分数 → Spearman 矩阵 + 判定建议。"""
    import json as jsonlib

    from yuformer_data.scale.truncation_calibration import run_calibration

    model_dir = tmp_path / "yuformer_quality_bert"
    train_final_bert(
        {"bert_train": {"epochs": 1, "batch_size": 32, "max_len": 64}},
        _gold_dir(tmp_path),
        model_dir,
    )
    out_root = tmp_path / "processed"
    rng = np.random.default_rng(3)
    vocab = GOOD_TEXT.lower().split()
    junk = ["Ã", "Â", "ï¿½", "<div>", "xxx"]
    rows = (
        [{"text": " ".join(rng.choice(vocab, size=80)), "domain": "web"} for _ in range(30)]
        + [{"text": " ".join(rng.choice(vocab, size=400)), "domain": "books"} for _ in range(30)]
        + [{"text": " ".join(rng.choice(vocab, size=700))
            + " " + " ".join(rng.choice(junk, size=700)), "domain": "books"} for _ in range(20)]
    )
    _write_s1_shard(out_root / "s1" / "synt" / "synt-0.parquet", rows)

    out_json = tmp_path / "truncation_calibration.json"
    cfg = {"bert": {"max_len": 64, "batch_size": 32, "score_quantum": 1e-4,
                    "batch_size_eval": 32}, "long_doc_chars": 4096}
    result = run_calibration(cfg, out_root, model_dir, out_json, n_total=60, seed=7)
    assert out_json.is_file()
    disk = jsonlib.loads(out_json.read_text())
    assert disk["recommendation"] == result["recommendation"]
    assert result["n_total"] == 60
    assert result["sample_source"] == "s1"  # 无 s3 → 回退 s1
    overall = result["spearman"]["overall"]
    assert set(overall) == {"head_vs_three", "head_vs_full", "three_vs_full"}
    assert result["recommendation"] in (
        "head_only", "three_window_long", "full512_three_window", "insufficient_data",
    )
    assert result["strata_actual"]["long"] == 12  # 配额 20%×60
