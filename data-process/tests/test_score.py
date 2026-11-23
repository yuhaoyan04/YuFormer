from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from yuformer_data.scale.score import (
    build_quantile_tables,
    composite_vector,
    effective_hybrid_v2,
    load_quantile_tables,
    load_scorer_config,
    normalize_upstream,
    process_shard_quantile,
    process_shard_score,
    quantile_of,
    s4_output_paths,
    scored_output_paths,
)
from yuformer_data.scale.schema import S1_SCHEMA
from yuformer_data.scale.scorer_train import (
    build_goldset,
    leave_one_source_cv,
    load_goldset_config,
    predict_hq_proba,
    train_final_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
S1_ROOT = REPO_ROOT / "data" / "processed" / "s1"

GOOD_TEXT = (
    "The history of the steam engine stretches back into the first century AD; "
    "the aeolipile described by Hero of Alexandria is the earliest known example. "
    "In the following centuries the fundamentals were gradually developed."
)
BAD_TEXT = "<div><span>Ã Â ï¿½</span></div>" * 40


def _write_s1_shard(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    full = []
    for i, r in enumerate(rows):
        rec = {
            "doc_id": f"doc-{path.stem}-{i}",
            "text": r.get("text", GOOD_TEXT),
            "domain": r.get("domain", "web"),
            "source": r.get("source", "synt"),
            "lang": r.get("lang", "en"),
            "upstream_score": r.get("upstream_score"),
            "meta": "{}",
            "decision": "keep",
            "reasons": r.get("reasons", []),
            "metrics": "{}",
        }
        full.append(rec)
    pq.write_table(pa.table({k: [r[k] for r in full] for k in S1_SCHEMA.names}, schema=S1_SCHEMA), path)


def _write_audit_shard(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {
        "doc_id": [f"audit-{i}" for i in range(len(rows))],
        "reasons": [r["reasons"] for r in rows],
        "text_head": [r["text"][:200] for r in rows],
        "source": [r.get("source", "synt") for r in rows],
    }
    pq.write_table(pa.table(cols), path)


def _synth_out_root(tmp_path: Path) -> Path:
    out_root = tmp_path / "processed"
    hq_text = "the history of science and engineering " * 30
    lq_text = "Ã Â ï¿½ <div> xxx yyy zzz" * 40
    for src, dom in (("pos-a", "web"), ("pos-b", "books"), ("pos-c", "academic")):
        _write_s1_shard(
            out_root / "s1" / src / f"{src}-0.parquet",
            [
                {"text": hq_text + f" variant {i}", "domain": dom, "source": src}
                for i in range(200)
            ],
        )
    _write_s1_shard(
        out_root / "s1" / "negtail" / "negtail-0.parquet",
        [{"text": lq_text + f" variant {i}", "domain": "web", "source": "negtail", "upstream_score": 0.005}
         for i in range(200)],
    )
    _write_audit_shard(
        out_root / "s1_audit" / "negtail" / "negtail-0.parquet",
        [{"text": lq_text + f" audit {i}", "reasons": ["garbled_latin1_mojibake"]} for i in range(200)],
    )
    return out_root


def _synth_goldset_cfg(tmp_path: Path, seed: int = 7) -> dict:
    return {
        "seed": seed,
        "min_text_chars": 50,
        "max_text_chars": 2000,
        # Tiny synthetic corpora use a conservative LR; each native fastText
        # fit is also process-isolated by the production trainer.
        "model_params": {"dim": 16, "wordNgrams": 2, "epoch": 3, "lr": 0.02, "minCount": 1, "loss": "softmax", "thread": 1},
        "quantize": True,
        "auc_target": 0.85,
        "positive": [
            {"rule": f"pos-{s}", "source": f"pos-{s}", "quota": 60} for s in ("a", "b", "c")
        ],
        "negative": [
            {"rule": "audit", "source": "@s1_audit", "quota": 80,
             "reasons_any": ["garbled_latin1_mojibake"]},
            {"rule": "tail", "source": "negtail", "quota": 80, "upstream_max": 0.02},
        ],
    }


def _synth_scorer_cfg(tmp_path: Path, model_path: Path) -> dict:
    cfg = {
        "model": str(model_path),
        "composite": {"upstream_weight": 0.5, "self_weight": 0.5},
        "upstream_normalize": {"synt": [0.0, 5.0], "negtail": [0.0, 1.0]},
        "inference": {"batch_size": 1000, "max_score_chars": 2000},
        "quantile": {"precision": 1e-6},
    }
    return cfg


def test_scorer_config_resolves_assets_from_data_process_root(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config = config_dir / "scorer.yaml"
    config.write_text(
        "composite: {upstream_weight: 0.5, self_weight: 0.5}\n"
        "model: models/scorer.bin\n"
        "goldrule_model: models/goldrule.joblib\n",
        encoding="utf-8",
    )
    loaded = load_scorer_config(config)
    assert loaded["model"] == str((tmp_path / "models/scorer.bin").resolve())
    assert loaded["goldrule_model"] == str(
        (tmp_path / "models/goldrule.joblib").resolve()
    )


def test_goldset_construction_ratios(tmp_path):
    out_root = _synth_out_root(tmp_path)
    cfg = _synth_goldset_cfg(tmp_path)
    gold_dir = tmp_path / "gold"
    summary = build_goldset(cfg, out_root, gold_dir)

    assert summary["labels"] == {"hq": 180, "lq": 160}
    per_rule = {k: v["sampled"] for k, v in summary["rules"].items()}
    assert per_rule == {"audit": 80, "pos-a": 60, "pos-b": 60, "pos-c": 60, "tail": 80}
    assert all(v["satisfied"] for v in summary["rules"].values())

    table = pq.read_table(gold_dir / "gold.parquet")
    assert table.num_rows == 340
    labels = table.column("label").to_pylist()
    assert labels.count("hq") == 180 and labels.count("lq") == 160
    sources = set(table.column("source").to_pylist())
    assert sources == {"pos-a", "pos-b", "pos-c", "@s1_audit", "negtail"}

    summary2 = build_goldset(cfg, out_root, gold_dir)
    assert summary2["gold_md5"] == summary["gold_md5"]


def test_leave_one_source_cv_structure(tmp_path):
    out_root = _synth_out_root(tmp_path)
    cfg = _synth_goldset_cfg(tmp_path)
    gold_dir = tmp_path / "gold"
    build_goldset(cfg, out_root, gold_dir)
    out_json = gold_dir / "cv_metrics.json"
    result = leave_one_source_cv(cfg, gold_dir, out_json)

    assert out_json.is_file()
    loaded = json.loads(out_json.read_text())
    assert loaded["stage"] == "s4_scorer_cv"
    assert loaded["positive_sources"] == ["pos-a", "pos-b", "pos-c"]
    for held in ("pos-a", "pos-b", "pos-c"):
        fold = loaded["folds"][held]
        for key in ("auc", "ap", "acc_at_0.5", "n", "n_pos", "held_source", "train_rows"):
            assert key in fold
        assert fold["held_source"] == held
        assert fold["n_pos"] == fold["n"] - fold["n_pos"] or fold["n_pos"] > 0
    assert 0.0 <= loaded["mean_auc"] <= 1.0
    assert result["auc_target"] == 0.85


def test_training_deterministic_md5(tmp_path):
    out_root = _synth_out_root(tmp_path)
    cfg = _synth_goldset_cfg(tmp_path, seed=11)
    gold_dir = tmp_path / "gold"
    build_goldset(cfg, out_root, gold_dir)

    m1 = train_final_model(cfg, gold_dir, tmp_path / "m1")
    m2 = train_final_model(cfg, gold_dir, tmp_path / "m2")
    assert m1["bin_md5"] == m2["bin_md5"], "fasttext thread=1 training must be byte-deterministic"
    assert m1["ftz_md5"] == m2["ftz_md5"]
    assert m1["bin_sha256"] == m2["bin_sha256"]
    assert m1["ftz_sha256"] == m2["ftz_sha256"]
    assert (tmp_path / "m1" / "yuformer_quality_scorer.bin").is_file()
    assert (tmp_path / "m1" / "yuformer_quality_scorer.ftz").is_file()


def test_composite_and_quantile_numeric(tmp_path):
    out_root = _synth_out_root(tmp_path)
    cfg = _synth_goldset_cfg(tmp_path, seed=13)
    gold_dir = tmp_path / "gold"
    build_goldset(cfg, out_root, gold_dir)
    train_final_model(cfg, gold_dir, tmp_path / "models")
    scorer_cfg = _synth_scorer_cfg(tmp_path, tmp_path / "models" / "yuformer_quality_scorer.ftz")

    rows = [
        {"text": GOOD_TEXT * 3, "domain": "web", "source": "synt", "upstream_score": 4.0},
        {"text": GOOD_TEXT * 2, "domain": "web", "source": "synt", "upstream_score": None},
        {"text": GOOD_TEXT, "domain": "books", "source": "synt", "upstream_score": 2.5},
        {"text": GOOD_TEXT + " tail", "domain": "books", "source": "synt", "upstream_score": None},
    ]
    _write_s1_shard(out_root / "s1" / "synt" / "synt-0.parquet", rows)

    from yuformer_data.scale.scorer_train import load_goldset_config  # noqa: F401
    import fasttext

    model = fasttext.load_model(str(scorer_cfg["model"]))
    summary = process_shard_score(out_root / "s1" / "synt" / "synt-0.parquet", out_root, "synt", "synt-0.parquet", scorer_cfg, model=model)
    assert summary["in_rows"] == 4 and not summary.get("errors")

    scored_path, _, _ = scored_output_paths(out_root, "synt", "synt-0.parquet")
    scored = pq.read_table(scored_path).to_pylist()

    texts_norm = [r["text"][:2000].lower() for r in rows]
    expected_self = predict_hq_proba(model, [" ".join(t.split()) for t in texts_norm])
    for i, row in enumerate(scored):
        assert row["self_score"] == pytest.approx(expected_self[i], abs=1e-6)
        nu = normalize_upstream(rows[i]["source"], rows[i]["upstream_score"], scorer_cfg)
        if nu is None:
            expected_comp = expected_self[i]
        else:
            expected_comp = 0.5 * nu + 0.5 * expected_self[i]
        assert row["composite"] == pytest.approx(expected_comp, abs=1e-5)

    # second shard to make quantiles non-trivial: web domain gets 5 rows total
    rows2 = [
        {"text": GOOD_TEXT * 4, "domain": "web", "source": "synt", "upstream_score": 5.0},
        {"text": GOOD_TEXT * 5, "domain": "web", "source": "synt", "upstream_score": 1.0},
    ]
    _write_s1_shard(out_root / "s1" / "synt" / "synt-1.parquet", rows2)
    process_shard_score(out_root / "s1" / "synt" / "synt-1.parquet", out_root, "synt", "synt-1.parquet", scorer_cfg, model=model)

    build = build_quantile_tables(out_root)
    assert build["total_rows"] == 6
    tables = load_quantile_tables(out_root)
    web = tables["web"]
    assert len(web) == 4 and web.dtype == np.float32

    qsummary = process_shard_quantile(out_root, "synt", "synt-0.parquet", scorer_cfg, tables=tables)
    assert qsummary["in_rows"] == 4 and not qsummary.get("errors")
    out_path, _ = s4_output_paths(out_root, "synt", "synt-0.parquet")
    final = pq.read_table(out_path).to_pylist()
    assert "domain_quantile" in final[0]

    def expected_q(sorted_vals: np.ndarray, x: float) -> float:
        arr = np.sort(sorted_vals.astype(np.float64))
        n = len(arr)
        left = int(np.searchsorted(arr, x, side="left"))
        if left == 0:
            return 0.0
        if left >= n:
            return 1.0
        lo, hi = arr[left - 1], arr[left]
        frac = 0.0 if hi <= lo else min(1.0, max(0.0, (x - lo) / (hi - lo)))
        return min(1.0, max(0.0, (left - 1 + frac) / (n - 1)))

    all_scored = scored + pq.read_table(scored_output_paths(out_root, "synt", "synt-1.parquet")[0]).to_pylist()
    web_vals = np.array([r["composite"] for r in all_scored if r["domain"] == "web"], dtype=np.float64)
    books_vals = np.array([r["composite"] for r in all_scored if r["domain"] == "books"], dtype=np.float64)
    for row in final:
        vals = web_vals if row["domain"] == "web" else books_vals
        assert row["domain_quantile"] == pytest.approx(expected_q(vals, row["composite"]), abs=1e-5)

    # direct unit check of the quantile kernel on a hand-picked array
    arr = np.array([0.2, 0.5, 0.8], dtype=np.float32)
    got = quantile_of(arr, np.array([0.2, 0.5, 0.8, 0.35, 0.9], dtype=np.float32), 1e-6)
    assert got == pytest.approx([0.0, 0.5, 1.0, 0.25, 1.0], abs=1e-6)


def test_score_and_quantile_idempotent(tmp_path):
    out_root = _synth_out_root(tmp_path)
    cfg = _synth_goldset_cfg(tmp_path, seed=17)
    gold_dir = tmp_path / "gold"
    build_goldset(cfg, out_root, gold_dir)
    train_final_model(cfg, gold_dir, tmp_path / "models")
    scorer_cfg = _synth_scorer_cfg(tmp_path, tmp_path / "models" / "yuformer_quality_scorer.ftz")
    _write_s1_shard(
        out_root / "s1" / "synt" / "synt-0.parquet",
        [{"text": GOOD_TEXT * 3, "domain": "web", "source": "synt", "upstream_score": 4.0}],
    )
    s1_path = out_root / "s1" / "synt" / "synt-0.parquet"
    s1 = process_shard_score(s1_path, out_root, "synt", "synt-0.parquet", scorer_cfg)
    scored_path, _, _ = scored_output_paths(out_root, "synt", "synt-0.parquet")
    first_bytes = scored_path.read_bytes()
    s2 = process_shard_score(s1_path, out_root, "synt", "synt-0.parquet", scorer_cfg)
    assert s2["skipped"] is True and scored_path.read_bytes() == first_bytes
    assert s1["processing_contract"]["schema_version"] == 2

    b1 = build_quantile_tables(out_root)
    b2 = build_quantile_tables(out_root)
    assert b1["input_contract"]["schema_version"] == 2
    assert b1["build_id"]
    assert all(
        f".{b1['build_id']}.npy" in info["npy"]
        for info in b1["domains"].values()
    )
    assert b2["skipped"] is True
    hybrid_build = build_quantile_tables(
        out_root, score_field="hybrid_v2", cfg=scorer_cfg
    )
    assert hybrid_build["input_contract"][
        "selection_contract_manifest_sha256"
    ]
    q1 = process_shard_quantile(out_root, "synt", "synt-0.parquet", scorer_cfg)
    out_path, _ = s4_output_paths(out_root, "synt", "synt-0.parquet")
    first_final = out_path.read_bytes()
    q2 = process_shard_quantile(out_root, "synt", "synt-0.parquet", scorer_cfg)
    assert q2["skipped"] is True and out_path.read_bytes() == first_final
    assert q1["processing_contract"]["schema_version"] == 2
    assert s1["skipped"] is False and q1["skipped"] is False

    # Any semantic scoring-config change invalidates the score shard even when
    # the model bytes and BERT-presence boolean are unchanged.
    changed_cfg = dict(scorer_cfg)
    changed_cfg["composite"] = {"upstream_weight": 0.4, "self_weight": 0.6}
    s3 = process_shard_score(
        s1_path, out_root, "synt", "synt-0.parquet", changed_cfg
    )
    assert s3["skipped"] is False
    assert (
        s3["processing_contract"]["scorer_config_sha256"]
        != s1["processing_contract"]["scorer_config_sha256"]
    )


def _real_s1_shards() -> list[Path]:
    if not S1_ROOT.is_dir():
        return []
    shards = sorted(
        p for p in S1_ROOT.rglob("*.parquet")
        if 50_000_000 <= p.stat().st_size <= 200_000_000
    )
    picked = []
    seen_src: set[str] = set()
    for p in shards:
        src = p.relative_to(S1_ROOT).parts[0]
        if src not in seen_src:
            picked.append(p)
            seen_src.add(src)
        if len(picked) >= 3:
            break
    return picked


@pytest.mark.skipif(
    os.environ.get("YUFORMER_RUN_REAL_DATA_TESTS") != "1",
    reason="multi-GB real-data throughput test is opt-in",
)
def test_end_to_end_throughput_real(tmp_path):
    shards = _real_s1_shards()
    if not shards:
        pytest.skip(f"no real s1 shards under {S1_ROOT}")
    out_root = _synth_out_root(tmp_path)
    cfg = _synth_goldset_cfg(tmp_path, seed=19)
    gold_dir = tmp_path / "gold"
    build_goldset(cfg, out_root, gold_dir)
    train_final_model(cfg, gold_dir, tmp_path / "models")
    scorer_cfg = _synth_scorer_cfg(tmp_path, tmp_path / "models" / "yuformer_quality_scorer.ftz")

    import fasttext

    model = fasttext.load_model(str(scorer_cfg["model"]))
    rates = []
    total_rows = 0
    for path in shards:
        rel_parent = path.relative_to(S1_ROOT)
        source = rel_parent.parts[0]
        relpath = rel_parent.relative_to(source).as_posix()[: -len(".parquet")]
        started = time.monotonic()
        summary = process_shard_score(path, tmp_path / "out", source, relpath, scorer_cfg, model=model)
        elapsed = time.monotonic() - started
        assert summary["in_rows"] > 0 and not summary.get("errors")
        rates.append(summary["in_rows"] / max(elapsed, 1e-9))
        total_rows += summary["in_rows"]

    best = max(rates)
    print(f"\nS4 throughput: per-shard rates {[f'{r:.0f}' for r in rates]} docs/s/core (best {best:.0f}), rows={total_rows}")
    # E-36（2026-10-08）：fasttext 轨实际一直运行（yaml off→False 陷阱），fasttext 三窗
    # 推理是吞吐瓶颈（生产文档 112~1300 rows/s/核量级）；旧 20000 目标对应 E-25 意图的
    # track-off 路径，永不达标。阈值对齐 track-on 现实：留 2 倍余量防 10× 级性能回归。
    assert best >= 800, f"throughput {best:.0f} docs/s/core below 800 (E-36 track-on floor)"


# ---------------------------------------------------------------- hybrid_composite（2026-09-27 语言感知混合轨）

def test_hybrid_composite_semantics():
    import numpy as np
    from yuformer_data.scale.score import hybrid_composite_vector

    rule_comp = np.array([0.624, 0.500, 0.700, 0.400], dtype=np.float64)
    fasttext = np.array([0.000, 0.900, 0.000, 0.800], dtype=np.float64)
    langs = ["en", "en", "zh", "zh"]
    h = hybrid_composite_vector(langs, rule_comp, fasttext)
    # en：0.6·rule + 0.4·ft —— SEO 垃圾（ft=0）被拉低
    assert abs(h[0] - 0.6 * 0.624) < 1e-9
    assert abs(h[1] - (0.6 * 0.500 + 0.4 * 0.900)) < 1e-9
    # zh：rule_composite 原样（fasttext 不参与——中文近零不破坏域内排序）
    assert h[2] == 0.700 and h[3] == 0.400
    # 语义验证：英文垃圾 < 英文好文
    assert h[0] < h[1]


# ---------------------------------------------------------------- hybrid_v2（2026-09-28 生产轨：goldrule 蒸馏 + upstream + rule）

def test_hybrid_v2_semantics():
    import numpy as np
    from yuformer_data.scale.score import hybrid_v2_vector

    cfg = {"hybrid_v2": {"llm_distill_weight": 0.5, "upstream_weight": 0.3, "rule_weight": 0.2}}
    llm = np.array([0.01, 0.90, 0.50, 0.80], dtype=np.float64)
    up = np.array([np.nan, np.nan, 0.60, 0.80], dtype=np.float64)
    rule = np.array([0.624, 0.500, 0.700, 0.600], dtype=np.float64)
    h = hybrid_v2_vector(llm, up, rule, cfg)
    # 垃圾（llm 深杀）+ 无 upstream：归一后 (0.5·0.01 + 0.2·0.624)/0.7
    assert abs(h[0] - (0.5 * 0.01 + 0.2 * 0.624) / 0.7) < 1e-9
    # 好文 + 无 upstream
    assert abs(h[1] - (0.5 * 0.90 + 0.2 * 0.500) / 0.7) < 1e-9
    # 全轨：三路和
    assert abs(h[3] - (0.5 * 0.80 + 0.3 * 0.80 + 0.2 * 0.600)) < 1e-9
    assert h[0] < h[3]

    # llm_distill 整轨缺失 → 退化为 upstream+rule
    h_none = hybrid_v2_vector(None, up, rule, cfg)
    assert abs(h_none[3] - (0.3 * 0.80 + 0.2 * 0.600) / 0.5) < 1e-9


def test_effective_hybrid_v2_repairs_legacy_missing_upstream():
    cfg = {
        "hybrid_v2": {
            "llm_distill_weight": 0.5,
            "upstream_weight": 0.3,
            "rule_weight": 0.2,
        }
    }
    stored = np.array([0.35, 0.80], dtype=np.float32)
    got = effective_hybrid_v2(
        stored, [None, 0.5], cfg, "legacy_zero_upstream_v1"
    )
    assert got == pytest.approx([0.5, 0.8])
    assert effective_hybrid_v2(
        stored, [None, 0.5], cfg, "current_v2"
    ) == pytest.approx(stored)
