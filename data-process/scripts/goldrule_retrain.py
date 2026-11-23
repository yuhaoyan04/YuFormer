"""goldrule 重训：fasttext 在 LLM 标注上重训 + 病理指标验证（对照旧模型的四个缺陷）。

验证协议（修复 E-25 指标错位——holdout 是 LLM 标注集而非源间 CV）：
  1. holdout AUC：总体 + 分域 + 分语言（目标 ≥0.85）
  2. 零膨胀率 P(<1e-3)：旧模型 fineweb 85% / 中文 96% —— 新目标 <20%
  3. 垃圾判杀：硬编码已知垃圾样本（VPN spinner / 中文医疗软文）必须低分
  4. 排序质量：Spearman(fasttext, llm_score) on holdout
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

KNOWN_SPAM = [
    ("en_vpn_spinner", "Which is the very best VPN for the purpose of Firestick? You'll glad to be familiar with that you can't really make a mistake with your decision. There are several excellent quality options, that exist quite easily and can be used to help ensure your net. Buy now best price click here www.deal.com subscribe newsletter."),
    ("zh_hospital_ad", "不同体质的牛皮癣患者如何治疗护理？牛皮癣患者的体质不同，所采用的治疗方式和护理方式也是不同的。郑州银屑病医院专家指出可以从下面的一些方法入手。专家经过多年临床经验总结，治愈患者数万例，欢迎在线咨询预约挂号。"),
    ("zh_seo_filler", "那么如何选择好的装修公司呢？选择装修公司一定要注意这几点。第一，要看装修公司的资质。第二，要看装修公司的口碑。第三，要看装修公司的案例。了解更多装修知识，请关注我们的网站，每天更新最新装修资讯。"),
]


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra = (ra - ra.mean()) / (ra.std() + 1e-12)
    rb = (rb - rb.mean()) / (rb.std() + 1e-12)
    return float((ra * rb).mean())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", default="../data/models/goldrule/annotations.jsonl")
    ap.add_argument("--hardneg-annotations", default="../data/models/goldrule/hardneg_annotations.jsonl")
    ap.add_argument("--annotate-set", default="../data/models/goldrule/annotate_set.parquet")
    ap.add_argument("--hardneg-set", default="../data/models/goldrule/hard_neg_set.parquet")
    ap.add_argument("--model-dir", default="../data/models")
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--hq-min-score", type=int, default=4, help="hq 判定阈值（4=对齐 top-10% 语义；3=FineWeb-Edu 宽松档）")
    ap.add_argument("--balance", action="store_true", help="hq 降采样到 lq 的 2 倍（类平衡）")
    args = ap.parse_args(argv)

    import fasttext

    ann = [json.loads(l) for l in Path(args.annotations).read_text().splitlines() if l.strip()]
    hardneg_path = Path(args.hardneg_annotations)
    if hardneg_path.is_file():
        ann += [json.loads(l) for l in hardneg_path.read_text().splitlines() if l.strip()]
        print(f"merged hardneg annotations: {len(ann)} total", file=sys.stderr)
    ann = [a for a in ann if a.get("llm_score", -1) >= 0]
    text_by_id = {}
    for set_path in (args.annotate_set, args.hardneg_set):
        p = Path(set_path)
        if p.is_file():
            for r in pq.read_table(p).to_pylist():
                text_by_id[r["doc_id"]] = r
    rows = []
    for a in ann:
        r = text_by_id.get(a["doc_id"])
        if r:
            rows.append({**a, "text": r["text"]})
    if len(rows) < 1000:
        print(f"insufficient annotations: {len(rows)}", file=sys.stderr)
        return 2
    hq_min = args.hq_min_score
    n_hq = sum(1 for r in rows if r["llm_score"] >= hq_min)
    print(f"annotations: {len(rows)} (hq(>={hq_min})={n_hq} {n_hq/len(rows):.0%}, lq={len(rows)-n_hq})")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(rows))
    n_hold = int(len(rows) * args.holdout_frac)
    hold_idx = set(idx[:n_hold].tolist())
    train = [rows[i] for i in range(len(rows)) if i not in hold_idx]
    hold = [rows[i] for i in sorted(hold_idx)]

    train_file = Path(args.model_dir) / "goldrule" / "train_llm.txt"
    train_file.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with open(train_file, "w") as f:
        for r in train:
            key = hash(r["text"])
            if key in seen:
                continue
            seen.add(key)
            label = "__label__hq" if r["llm_score"] >= hq_min else "__label__lq"
            f.write(f"{label} {r['text'].replace(chr(10), ' ')}\n")

    model = fasttext.train_supervised(
        input=str(train_file), dim=50, wordNgrams=2, epoch=6, lr=0.1,
        minCount=3, loss="softmax", thread=1, verbose=0,
    )

    def ft_score(text: str) -> float:
        labels, probs = model.predict([text.replace("\n", " ")[:1024]])
        for l, p in zip(labels[0], probs[0]):
            if "hq" in l:
                return float(p)
        return 0.0

    def auc(pos, neg):
        if not len(pos) or not len(neg):
            return float("nan")
        combined = np.concatenate([pos, neg])
        ranks = np.argsort(np.argsort(combined)).astype(float) + 1
        rp = ranks[: len(pos)]
        return float((rp.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))

    hold_scores = np.array([ft_score(r["text"]) for r in hold])
    hold_labels = np.array([1 if r["llm_score"] >= hq_min else 0 for r in hold])
    pos = hold_scores[hold_labels == 1]
    neg = hold_scores[hold_labels == 0]
    overall_auc = auc(pos, neg)

    per_domain = {}
    for dom in sorted({r["domain"] for r in hold}):
        m = np.array([r["domain"] == dom for r in hold])
        per_domain[dom] = round(auc(hold_scores[m & (hold_labels == 1)], hold_scores[m & (hold_labels == 0)]), 3)
    zh_mask = np.array([str(r.get("lang", "")).startswith("zh") for r in hold])
    zh_auc = auc(hold_scores[zh_mask & (hold_labels == 1)], hold_scores[zh_mask & (hold_labels == 0)])

    mid_mask = np.array([0.4 <= r.get("rule_score", 0) <= 0.7 for r in hold])
    metrics = {
        "stage": "goldrule_retrain",
        "n_train": len(train), "n_holdout": len(hold),
        "holdout_auc": round(overall_auc, 4),
        "holdout_auc_zh": round(zh_auc, 4) if zh_auc == zh_auc else None,
        "holdout_auc_per_domain": per_domain,
        "zero_inflation_holdout": round(float((hold_scores < 1e-3).mean()), 4),
        "zero_inflation_midrule_docs": round(float((hold_scores[mid_mask] < 1e-3).mean()), 4),
        "spearman_vs_llm": round(spearman(hold_scores, np.array([r["llm_score"] for r in hold], dtype=float)), 4),
        "spam_checks": {},
        "baseline_note": "旧模型：fineweb 判零 85%、中文 96%、中段文档大量误零（E-25 病理）",
    }
    for name, text in KNOWN_SPAM:
        s = ft_score(text)
        metrics["spam_checks"][name] = round(s, 4)

    model_path = Path(args.model_dir) / "goldrule_scorer_v1.bin"
    model.save_model(str(model_path))
    metrics["model"] = str(model_path)
    out = Path(args.model_dir) / "goldrule" / "retrain_metrics.json"
    out.write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=1))
    print(json.dumps(metrics, ensure_ascii=False, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
