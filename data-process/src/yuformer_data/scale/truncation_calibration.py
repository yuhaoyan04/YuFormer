"""截断策略校准：head-only vs 三窗 vs full-doc(max_len 512) 的 Spearman 相关矩阵。

2026-09-20 用户裁定。确定性抽 5,000 篇（短 ≤1024 / 中 1024-4096 / 长 >4096 字符分层，
blake2b 驱动水库抽样，同快照重跑一致），用已训练 BERT 档每篇算三种 bert_score，
输出相关矩阵与内置判定建议到 data/models/truncation_calibration.json：

  Spearman(head, full) ≥ 0.90 → 全池 head-only（最便宜）
  0.75 – 0.90            → 长文（>long_doc_chars）三窗
  < 0.75                 → 全池 max_len 512 三窗

只跑小样本（5k 篇单卡秒级~分钟级），不占 CPU。采样源：优先 s3 keep 池（若已产出），
否则回退 s1（去重对截断校准影响可忽略，summary 记录实际来源）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from .score import load_scorer_config, spearman
from .score_bert import BertInferencer, _load_bert, resolve_bert_model_dir
from .scorer_train import rule_rng

STRATA = (("short", 0, 1024), ("medium", 1024, 4096), ("long", 4096, 1 << 62))
# 分层配额：短/中/长 = 40/40/20%（长文是截断策略的关注对象，但占比低）
STRATA_QUOTA = {"short": 0.4, "medium": 0.4, "long": 0.2}


def _candidate_shards(out_root: Path) -> tuple[str, list[Path]]:
    s3_root = out_root / "s3"
    if s3_root.is_dir() and any(s3_root.rglob("*.parquet")):
        return "s3", sorted(p for p in s3_root.rglob("*.parquet") if p.is_file())
    s1_root = out_root / "s1"
    if s1_root.is_dir():
        return "s1", sorted(p for p in s1_root.rglob("*.parquet") if p.is_file())
    raise FileNotFoundError(
        f"calibration 需要打分池：{s3_root} 与 {s1_root} 均不存在（先跑 S1/S3）"
    )


def _reservoir_strata(
    shards: list[Path], quotas: dict[str, int], seed: int
) -> dict[str, list[dict[str, str]]]:
    """按 (归一化长度分层) 确定性水库抽样。行序 = shard 排序序 + 批序，重跑一致。"""
    rngs = {name: rule_rng(seed, f"trunc-cal-{name}") for name, _lo, _hi in STRATA}
    reservoirs: dict[str, list[dict[str, str]]] = {name: [] for name, _lo, _hi in STRATA}
    seen = {name: 0 for name, _lo, _hi in STRATA}

    def try_add(stratum: str, row: dict[str, str]) -> None:
        rng = rngs[stratum]
        reservoir = reservoirs[stratum]
        quota = quotas[stratum]
        seen[stratum] += 1
        if len(reservoir) < quota:
            reservoir.append(row)
        else:
            j = rng.randrange(seen[stratum])
            if j < quota:
                reservoir[j] = row

    for path in shards:
        try:
            pf = pq.ParquetFile(path)
        except Exception:
            continue
        cols = ["doc_id", "text"] + (["s3_decision"] if "s3_decision" in pf.schema_arrow.names else [])
        for batch in pf.iter_batches(batch_size=2048, columns=cols):
            d = batch.to_pydict()
            n = len(d["doc_id"])
            for i in range(n):
                if "s3_decision" in d and d["s3_decision"][i] != "keep":
                    continue
                text = d["text"][i] or ""
                ln = len(" ".join(text.lower().split()))  # 归一化长度（与打分口径一致）
                for name, lo, hi in STRATA:
                    if lo < ln <= hi:
                        try_add(name, {"doc_id": d["doc_id"][i], "text": text})
                        break
    return reservoirs


def _spearman_matrix(
    scores: dict[str, np.ndarray], strata_idx: dict[str, np.ndarray]
) -> dict[str, Any]:
    """overall + 每层的 (head|three|full) 两两 Spearman。"""
    keys = ("head", "three", "full")
    out: dict[str, Any] = {"overall": {}, "per_stratum": {}}
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            ka, kb = keys[a], keys[b]
            out["overall"][f"{ka}_vs_{kb}"] = spearman(
                scores[ka].tolist(), scores[kb].tolist()
            )
    for stratum, idx in sorted(strata_idx.items()):
        per: dict[str, Any] = {"n": int(len(idx))}
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                ka, kb = keys[a], keys[b]
                per[f"{ka}_vs_{kb}"] = spearman(
                    scores[ka][idx].tolist(), scores[kb][idx].tolist()
                )
        out["per_stratum"][stratum] = per
    return out


def recommend(head_vs_full: float | None) -> tuple[str, str]:
    """内置判定建议（阈值 2026-09-20 用户裁定）。"""
    if head_vs_full is None:
        return "insufficient_data", "head 与 full 分数无变异，无法计算相关"
    if head_vs_full >= 0.90:
        return "head_only", f"head vs full Spearman {head_vs_full:.3f} ≥ 0.90 → 全池 head-only"
    if head_vs_full >= 0.75:
        return "three_window_long", (
            f"head vs full Spearman {head_vs_full:.3f} ∈ [0.75, 0.90) → 长文（>long_doc_chars）三窗"
        )
    return "full512_three_window", (
        f"head vs full Spearman {head_vs_full:.3f} < 0.75 → 全池 max_len 512 三窗"
    )


def run_calibration(
    cfg: dict[str, Any],
    out_root: Path,
    model_dir: Path,
    out_json: Path,
    n_total: int = 5000,
    seed: int = 20260920,
    device: str | None = None,
    log_fn: Any = None,
) -> dict[str, Any]:
    log = log_fn or (lambda m: print(m, file=sys.stderr, flush=True))
    quotas = {
        name: int(n_total * STRATA_QUOTA[name]) for name, _lo, _hi in STRATA
    }
    stage, shards = _candidate_shards(Path(out_root))
    log(f"truncation_calibration: sampling from {stage} ({len(shards)} shards), quotas={quotas}")
    sampled = _reservoir_strata(shards, quotas, seed)
    texts = [r["text"] for rows in sampled.values() for r in rows]
    strata_idx = {}
    off = 0
    for name, _lo, _hi in STRATA:
        k = len(sampled[name])
        strata_idx[name] = np.arange(off, off + k)
        off += k
    actual = {name: len(rows) for name, rows in sampled.items()}
    log(f"truncation_calibration: sampled {len(texts)} docs {actual}")

    model, tok, dev = _load_bert(Path(model_dir))
    params = cfg.get("bert") or {}
    max_len = int(params.get("max_len", 256))
    bs = int(params.get("batch_size", 512))
    quantum = float(params.get("score_quantum", 1e-4))
    long_doc_chars = int(params.get("long_doc_chars", cfg.get("long_doc_chars", 4096)))
    if device:
        model = model.to(device)
        dev = device

    started = time.monotonic()
    # 三种口径（共享权重；仅截断/分窗不同）
    head = BertInferencer(model, tok, dev, max_len=max_len, batch_size=bs, quantum=quantum,
                          max_chars=1024, long_doc_chars=long_doc_chars, window_scoring=False)
    three = BertInferencer(model, tok, dev, max_len=max_len, batch_size=bs, quantum=quantum,
                           max_chars=1024, long_doc_chars=long_doc_chars, window_scoring=True)
    full = BertInferencer(model, tok, dev, max_len=512, batch_size=bs, quantum=quantum,
                          max_chars=8192, long_doc_chars=1 << 62, window_scoring=False)
    scores = {
        "head": head.score(texts),
        "three": three.score(texts),
        "full": full.score(texts),
    }
    for inf in (head, three, full):
        inf.shutdown()
    elapsed = time.monotonic() - started
    log(f"truncation_calibration: scored 3 variants × {len(texts)} docs in {elapsed:.1f}s")

    matrix = _spearman_matrix(scores, strata_idx)
    head_vs_full = matrix["overall"].get("head_vs_full")
    rec_code, rec_reason = recommend(head_vs_full)
    result = {
        "stage": "truncation_calibration",
        "date": "2026-09-20",
        "sample_source": stage,
        "n_total": len(texts),
        "strata_quotas": quotas,
        "strata_actual": {k: int(len(v)) for k, v in strata_idx.items()},
        "seed": seed,
        "model": str(model_dir),
        "variants": {
            "head": {"max_len": max_len, "max_chars": 1024, "window_scoring": False},
            "three": {"max_len": max_len, "max_chars": 1024, "window_scoring": True,
                      "long_doc_chars": long_doc_chars},
            "full": {"max_len": 512, "max_chars": 8192, "window_scoring": False},
        },
        "spearman": matrix,
        "recommendation": rec_code,
        "recommendation_reason": rec_reason,
        "elapsed_sec": round(elapsed, 3),
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_json.with_name(out_json.name + ".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8")
    tmp.replace(out_json)
    log(f"truncation_calibration: recommendation={rec_code} ({rec_reason})")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="yuformer_data.scale.truncation_calibration")
    parser.add_argument("--config", default=None, help="scorer.yaml（默认 scale/score.py 内置路径）")
    parser.add_argument("--out-root", default="../data/processed")
    parser.add_argument("--model-dir", default=None,
                        help="BERT 档模型目录（默认 scorer.yaml bert_model）")
    parser.add_argument("--out-json", default=None,
                        help="default <model-dir 的父目录>/truncation_calibration.json")
    parser.add_argument("--n-total", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args(argv)

    from .score import DEFAULT_SCORER_CONFIG

    cfg = load_scorer_config(args.config or DEFAULT_SCORER_CONFIG)
    model_dir = Path(args.model_dir) if args.model_dir else resolve_bert_model_dir(cfg)
    if args.out_json:
        out_json = Path(args.out_json)
    else:
        out_json = model_dir.parent / "truncation_calibration.json"
    result = run_calibration(
        cfg, Path(args.out_root), model_dir, out_json,
        n_total=args.n_total, seed=args.seed,
    )
    print(json.dumps(
        {"recommendation": result["recommendation"],
         "head_vs_full": result["spearman"]["overall"].get("head_vs_full"),
         "n": result["n_total"]},
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
