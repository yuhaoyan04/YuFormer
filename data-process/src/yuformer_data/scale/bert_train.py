"""S4 BERT 回归档训练：与 fasttext 档共用 gold-set 与留一源 CV 门禁。

模型：bert-base-multilingual-cased（HF: google-bert/bert-base-multilingual-cased，
经 HF_ENDPOINT=https://hf-mirror.com 拉取），max_len 256 token，二分类 CE
（hq=1 / lq=0），sigmoid 概率即 bert_score。

复用 scorer_train 的 gold-set 构造与 CV 划分（load_gold_rows / rule_rng / auc_ap），
不复制实现。CV 门禁与 fasttext 档一致：mean AUC ≥0.85 且每个 holdout ≥0.75。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .scorer_train import (
    DEFAULT_GOLDSET_CONFIG,
    auc_ap,
    build_goldset,
    load_gold_rows,
    load_goldset_config,
    rule_rng,
)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

BERT_MODEL_NAME = "google-bert/bert-base-multilingual-cased"

DEFAULT_BERT_PARAMS: dict[str, Any] = {
    "max_len": 256,
    "batch_size": 64,
    "lr": 2e-5,
    "epochs": 2,
    "warmup_ratio": 0.05,
    "seed": 20260917,
    "amp_fp16": True,
}

# CV 门禁（与 fasttext 档一致的留一源划分；阈值见 PIPELINE_DESIGN S4 预检门）
CV_MEAN_AUC_TARGET = 0.85
CV_MIN_FOLD_AUC = 0.75


def _bert_params(cfg: dict[str, Any]) -> dict[str, Any]:
    params = dict(DEFAULT_BERT_PARAMS)
    params.update(cfg.get("bert_train") or {})
    return params


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(BERT_MODEL_NAME)


def _load_model(num_labels: int = 2):
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME, num_labels=num_labels
    )


def token_windows(ids: list[int], window: int) -> list[list[int]]:
    """token 序列的三窗切分（含 special token；短序列单窗）。

    head/mid/tail 各 window 个 token；mid 取序列中点对齐窗口。2026-09-20 用户裁定。
    """
    if len(ids) <= window:
        return [ids]
    half = window // 2
    mid = len(ids) // 2
    lo = max(0, mid - half)
    return [ids[:window], ids[lo : lo + window], ids[-window:]]


def encode_texts(tokenizer: Any, texts: list[str], max_len: int) -> dict[str, Any]:
    # padding="longest"（非 max_length）：批内按最长文档对齐。attention mask 保证
    # 数值与 max_length 完全一致，而 FLOP 随批内实际长度缩放——真实池短文档多，
    # 相比无脑 pad 到 256 省 1.3-1.8× 算力。
    return tokenizer(
        texts,
        truncation=True,
        max_length=max_len,
        padding=True,
        return_tensors="pt",
    )


class _GoldDataset:
    """极简 map-style 数据集：避免 DataLoader worker 进程（CPU 受限环境）。"""

    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_len: int) -> None:
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def collate(self, idxs: list[int]) -> dict[str, Any]:
        import torch

        texts = [self.rows[i]["text"] for i in idxs]
        enc = encode_texts(self.tok, texts, self.max_len)
        labels = torch.tensor(
            [1 if self.rows[i]["label"] == "hq" else 0 for i in idxs], dtype=torch.long
        )
        enc["labels"] = labels
        return enc


def bert_hq_proba(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    max_len: int,
    batch_size: int,
    device: str,
    amp_fp16: bool,
) -> np.ndarray:
    """FP16 批量推理：返回 P(hq) ∈ [0,1]，量化到 1e-4。

    模型本身已是 fp16（推理档 .half() 加载）时跳过 autocast——实测纯 fp16 比
    autocast 快 ~33%（autocast 会把 LayerNorm/softmax 留在 fp32）。
    """
    import torch

    model.eval()
    is_fp16_model = next(model.parameters()).dtype == torch.float16
    use_autocast = amp_fp16 and device.startswith("cuda") and not is_fp16_model
    probs = np.zeros(len(texts), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = encode_texts(tokenizer, texts[i : i + batch_size], max_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_autocast):
                logits = model(**enc).logits
            p = torch.softmax(logits.float(), dim=-1)[:, 1]
            probs[i : i + len(p)] = p.detach().cpu().numpy()
    return np.round(probs / 1e-4) * 1e-4


def train_bert_classifier(
    train_rows: list[dict[str, Any]],
    params: dict[str, Any],
    device: str | None = None,
    log_fn: Any = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """训练二分类头，返回 (model, tokenizer, train_info)。单卡即可。"""
    import torch
    from transformers import get_linear_schedule_with_warmup

    seed = int(params.get("seed", 0))
    _seed_everything(seed)
    max_len = int(params.get("max_len", 256))
    batch_size = int(params.get("batch_size", 64))
    epochs = int(params.get("epochs", 2))
    lr = float(params.get("lr", 2e-5))
    warmup_ratio = float(params.get("warmup_ratio", 0.05))
    amp_fp16 = bool(params.get("amp_fp16", True)) and torch.cuda.is_available()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = _load_tokenizer()
    model = _load_model(num_labels=2).to(device)
    model.train()

    ds = _GoldDataset(train_rows, tokenizer, max_len)
    n = len(ds)
    steps_per_epoch = max(1, -(-n // batch_size))
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)

    started = time.monotonic()
    first_loss: float | None = None
    last_loss: float | None = None
    step = 0
    for epoch in range(epochs):
        order = list(range(n))
        random.Random(seed + epoch).shuffle(order)
        for i in range(0, n, batch_size):
            batch = ds.collate(order[i : i + batch_size])
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=amp_fp16
            ):
                out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if first_loss is None:
                first_loss = float(loss.detach().cpu())
            last_loss = float(loss.detach().cpu())
            if log_fn and step % 100 == 0:
                log_fn(f"bert_train: step {step}/{total_steps} loss {last_loss:.4f}")
    train_sec = time.monotonic() - started
    info = {
        "train_rows": n,
        "steps": step,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "train_sec": round(train_sec, 3),
        "device": device,
        "amp_fp16": amp_fp16,
    }
    return model, tokenizer, info


def leave_one_source_cv_bert(
    cfg: dict[str, Any],
    gold_dir: Path,
    out_json: Path | None = None,
    device: str | None = None,
    log_fn: Any = None,
) -> dict[str, Any]:
    """与 fasttext 档完全相同的留一源划分（rule_rng("cv-test-<held>")）。"""
    seed = int(cfg.get("seed", 0))
    params = _bert_params(cfg)
    rows = load_gold_rows(gold_dir)
    pos_sources = sorted({r["source"] for r in rows if r["label"] == "hq"})
    lq_rows = [r for r in rows if r["label"] == "lq"]

    folds: dict[str, Any] = {}
    for held in pos_sources:
        train_rows = [r for r in rows if r["label"] == "lq" or r["source"] != held]
        held_hq = [r for r in rows if r["label"] == "hq" and r["source"] == held]
        rng = rule_rng(seed, f"cv-test-{held}")
        k = len(held_hq)
        test_lq = list(lq_rows)
        rng.shuffle(test_lq)
        test_rows = held_hq + test_lq[:k]

        model, tokenizer, _info = train_bert_classifier(train_rows, params, device=device, log_fn=log_fn)
        scores = bert_hq_proba(
            model,
            tokenizer,
            [r["text"] for r in test_rows],
            int(params["max_len"]),
            int(params.get("eval_batch_size", 256)),
            device or ("cuda"),
            bool(params.get("amp_fp16", True)),
        )
        del model
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        y_true = np.array([1 if r["label"] == "hq" else 0 for r in test_rows], dtype=int)
        metrics = auc_ap(y_true, scores)
        metrics["held_source"] = held
        metrics["train_rows"] = len(train_rows)
        folds[held] = metrics
        if log_fn:
            log_fn(f"bert_cv: held={held} auc={metrics['auc']:.4f} ap={metrics['ap']:.4f}")

    aucs = {k: m["auc"] for k, m in folds.items()}
    mean_auc = float(np.mean(list(aucs.values()))) if aucs else None
    min_fold = float(min(aucs.values())) if aucs else None
    target_met = bool(
        aucs and mean_auc is not None
        and mean_auc >= CV_MEAN_AUC_TARGET
        and min_fold >= CV_MIN_FOLD_AUC
    )
    result = {
        "stage": "s4_bert_cv",
        "model": BERT_MODEL_NAME,
        "seed": seed,
        "positive_sources": pos_sources,
        "folds": {k: folds[k] for k in sorted(folds)},
        "mean_auc": mean_auc,
        "min_fold_auc": min_fold,
        "auc_target": CV_MEAN_AUC_TARGET,
        "min_fold_auc_target": CV_MIN_FOLD_AUC,
        "target_met": target_met,
        "bert_params": {k: params[k] for k in sorted(params)},
    }
    if not target_met:
        result["recommendation"] = (
            "leave-one-source gate failed (mean AUC >= 0.85 AND every fold >= 0.75 "
            "required): expand negatives / add positive sources; do NOT silently accept."
        )
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_json.with_name(out_json.name + ".tmp")
        tmp.write_text(
            json.dumps(result, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8"
        )
        tmp.replace(out_json)
    return result


def train_final_bert(
    cfg: dict[str, Any],
    gold_dir: Path,
    model_dir: Path,
    device: str | None = None,
    log_fn: Any = None,
) -> dict[str, Any]:
    """全量 gold-set 训练 + 保存 HuggingFace 目录 + summary。"""
    params = _bert_params(cfg)
    rows = load_gold_rows(gold_dir)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer, info = train_bert_classifier(rows, params, device=device, log_fn=log_fn)

    # 训练集内指标（in-sample，仅供参考；质量门以留一源 CV 为准）
    scores = bert_hq_proba(
        model,
        tokenizer,
        [r["text"] for r in rows],
        int(params["max_len"]),
        int(params.get("eval_batch_size", 256)),
        info["device"],
        bool(params.get("amp_fp16", True)),
    )
    y_true = np.array([1 if r["label"] == "hq" else 0 for r in rows], dtype=int)
    overall = auc_ap(y_true, scores)

    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # GPU 训练存在数值不确定性：记录产物内容的稳定指纹（量化后分数）而非权重哈希
    score_md5 = hashlib.md5(scores.astype(np.float32).tobytes()).hexdigest()
    summary = {
        "stage": "s4_bert_train",
        "model": BERT_MODEL_NAME,
        "train_rows": len(rows),
        "bert_params": {k: params[k] for k in sorted(params)},
        **info,
        "overall_auc": overall["auc"],
        "overall_ap": overall["ap"],
        "train_scores_md5": score_md5,
        "skipped": False,
        "notes": (
            "GPU 数值不确定：权重与 in-sample 分数的 md5 重跑可能不同；"
            "推理 bert_score 量化到 1e-4，重跑抖动 ±1e-4 属预期。"
        ),
    }
    (model_dir / "bert_train.summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="yuformer_data.scale.bert_train")
    parser.add_argument("--config", default=str(DEFAULT_GOLDSET_CONFIG))
    parser.add_argument("--out-root", default="../data/processed")
    parser.add_argument("--gold-dir", default=None, help="default <out-root>/scorer_goldset")
    parser.add_argument("--model-dir", default="../data/models")
    parser.add_argument("--step", choices=["goldset", "cv", "train", "all"], default="all")
    args = parser.parse_args(argv)

    logging_fn = lambda msg: print(msg, file=sys.stderr, flush=True)  # noqa: E731

    cfg = load_goldset_config(args.config)
    out_root = Path(args.out_root)
    gold_dir = Path(args.gold_dir) if args.gold_dir else out_root / "scorer_goldset"
    model_dir = Path(args.model_dir)
    bert_dir = model_dir / "yuformer_quality_bert"
    cv_json = model_dir / "bert_scorer_cv_metrics.json"

    if args.step in ("goldset", "all"):
        s = build_goldset(cfg, out_root, gold_dir)
        print(json.dumps({"goldset": {"records": s["records"], "labels": s["labels"]}}, sort_keys=True))
    if args.step in ("cv", "all"):
        r = leave_one_source_cv_bert(cfg, gold_dir, cv_json, log_fn=logging_fn)
        print(json.dumps(
            {"cv": {"mean_auc": r["mean_auc"], "min_fold_auc": r["min_fold_auc"], "target_met": r["target_met"]}},
            sort_keys=True,
        ))
        if not r["target_met"]:
            print(json.dumps({"cv_warning": r["recommendation"]}, sort_keys=True))
    if args.step in ("train", "all"):
        t = train_final_bert(cfg, gold_dir, bert_dir, log_fn=logging_fn)
        print(json.dumps({"train": {"overall_auc": t["overall_auc"], "rows": t["train_rows"]}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
