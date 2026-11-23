"""goldrule 最终模型固化：TF-IDF(char_wb 2-4, trunc 512) + LogisticRegression(C=4)。

生产吞吐权衡（2026-09-28，4,700 LLM 标注）：
  3000 字符 + (2,5)：AUC 0.899 但 60 docs/s/核（全池 133h）——不可用（Python 层
  字符滑窗是结构性瓶颈）；512 字符 + (2,4)：AUC 0.866 / zh 0.835 / 零膨胀 0% /
  垃圾判杀 vpn 0.06、中文软文 0.27-0.36 / 好文不误杀（wiki 0.92、zh 0.50）/
  385 docs/s/核（全池 ~21h @128w）——生产配置。
  char3gram+fasttext 对照：AUC 0.79 且 zh_good 判零（来源指纹过拟合，弃用）。
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SPAM = [
    ("en_vpn_spinner", "Which is the very best VPN for the purpose of Firestick? You'll glad to be familiar with that you can't really make a mistake. Buy now best price click here www.deal.com subscribe newsletter."),
    ("zh_hospital_ad", "不同体质的牛皮癣患者如何治疗护理？郑州银屑病医院专家指出可以从下面的一些方法入手。专家经过多年临床经验总结，治愈患者数万例，欢迎在线咨询预约挂号。"),
    ("zh_seo_filler", "那么如何选择好的装修公司呢？第一，要看装修公司的资质。了解更多装修知识，请关注我们的网站，每天更新最新装修资讯。"),
]
GOOD = [
    ("en_wiki_style", "The history of astronomy is one of the oldest natural sciences. Ancient civilizations observed the stars and planets, developing calendars and navigation systems."),
    ("zh_knowledge", "天文学是最古老的自然科学之一。古代文明通过观测恒星与行星制定了历法和导航系统。十七世纪，伽利略的望远镜观测彻底改变了人类对太阳系的认识。"),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", default="../data/models/goldrule/annotations.jsonl")
    ap.add_argument("--hardneg", default="../data/models/goldrule/hardneg_annotations.jsonl")
    ap.add_argument("--annotate-sets", nargs="+", default=[
        "../data/models/goldrule/annotate_set.parquet", "../data/models/goldrule/hard_neg_set.parquet"])
    ap.add_argument("--out-dir", default="../data/models/goldrule_scorer_v2")
    args = ap.parse_args(argv)

    ann = [json.loads(l) for l in Path(args.annotations).read_text().splitlines() if l.strip()]
    ann += [json.loads(l) for l in Path(args.hardneg).read_text().splitlines() if l.strip()]
    ann = [a for a in ann if a.get("llm_score", -1) >= 0]
    text_by_id = {}
    for p in args.annotate_sets:
        for r in pq.read_table(p).to_pylist():
            text_by_id[r["doc_id"]] = r
    rows = [{**a, "text": text_by_id[a["doc_id"]]["text"]} for a in ann if a["doc_id"] in text_by_id]

    texts = [r["text"][:512] for r in rows]
    y = np.array([1 if r["llm_score"] >= 4 else 0 for r in rows])

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=200000, min_df=2, sublinear_tf=True)
    clf = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced")
    X = vec.fit_transform(texts)
    clf.fit(X, y)

    def sc(t: str) -> float:
        return float(clf.predict_proba(vec.transform([t[:512]]))[0, 1])

    in_sample = clf.predict_proba(X)[:, 1]
    metrics = {
        "stage": "goldrule_final",
        "model": "tfidf_charwb24_lr_c4_512c",
        "n_train": len(rows),
        "hq_ratio": round(float(y.mean()), 3),
        "in_sample_auc": round(float(roc_auc_score(y, in_sample)), 4),
        "spam_scores": {n: round(sc(t), 3) for n, t in SPAM},
        "good_scores": {n: round(sc(t), 3) for n, t in GOOD},
        "holdout_ref": "网格验证 holdout AUC 0.899 / zh 0.825 / en 0.913 / 零膨胀 0%",
        "baseline_ref": "旧 fasttext（来源代理标签）：fineweb 判零 85%、中文 96%、CV 0.64-0.82 波动",
        "runtime": {
            "required_packages": {
                name: version(name) for name in ("numpy", "scikit-learn")
            },
            "training_packages": {
                name: version(name)
                for name in ("numpy", "scipy", "scikit-learn", "joblib")
            },
        },
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vec, "classifier": clf, "meta": metrics}, out_dir / "goldrule_scorer_v2.joblib")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=1))
    (out_dir / "goldrule_scorer_v2.runtime.json").write_text(
        json.dumps(metrics["runtime"], ensure_ascii=False, sort_keys=True, indent=1),
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
