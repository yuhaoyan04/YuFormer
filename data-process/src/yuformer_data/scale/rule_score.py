"""规则质量分（rule_score）：免训练的兜底打分轨 + goldrule 研究基线。

定位（2026-09-27 用户裁定）：若 learned scorer（fasttext/BERT）CV 始终不过门，
生产选取切换到 rule_composite = 0.6·upstream_norm + 0.4·rule_score（upstream
缺失时纯 rule_score）；learned 轨转入 goldrule 研究子集并行迭代（E-20 分工推广）。

特征系参照工业 edu-classifier（FineWeb-Edu 论文公开特征族）+ 本项目 spotcheck
实测指标，全部为确定性纯函数（numpy/regex，无模型依赖、无随机性）：
  - 词汇：word_uniq_ratio / avg_word_len / stopword 覆盖（EN top-50 + ZH 常用字表）
  - 结构：句均长、段落数、行重复率
  - 符号：字母/数字/标点占比、url 密度、boilerplate 命中
  - 惩罚：过短、超长、重复行、乱码字符
输出 rule_score ∈ [0,1]（分段线性映射，阈值经验值锚定 spotcheck 分布）。
"""

from __future__ import annotations

import re

import numpy as np

_EN_STOPWORDS = frozenset(
    "the of and to in a is that for it as was with be by on not he this are or his "
    "from at which but have an they one you had we all her she there would their we "
    "him been has when who will more no if out so said what up its about into than "
    "them can only other new some time these two may then do first any my now such".split()
)
_ZH_COMMON = frozenset("的一是了我不人在他有这上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感".split())

_SENT_SPLIT = re.compile(r"[.!?。！？；;]+")
_URL_RE = re.compile(r"https?://|www\.", re.I)
_BOILER_RE = re.compile(
    r"cookie|javascript|sign in|log ?in|subscribe|newsletter|all rights reserved|"
    r"privacy policy|terms of (service|use)|click here|advertisement|follow us|share (this|on)",
    re.I,
)
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_WORD_RE = re.compile(r"[A-Za-z]+|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]|\d+")


def _ramp(x: float, lo: float, hi: float) -> float:
    """[lo,hi] → [0,1] 线性截断。"""
    if hi <= lo:
        return 1.0
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def rule_features(text: str) -> dict[str, float]:
    """单文档规则特征（全部确定性，中文逐字词口径与管线一致）。"""
    n = len(text)
    if n == 0:
        return {k: 0.0 for k in (
            "chars", "uniq_ratio", "avg_word_len", "stop_cover", "sent_len",
            "para_density", "line_dup", "alpha_ratio", "digit_ratio", "punct_ratio",
            "url_density", "boiler_hits", "garbled",
        )}
    words = _WORD_RE.findall(text)
    nw = len(words)
    uniq = len(set(w.lower() for w in words))
    stop_hits = sum(1 for w in words if w.lower() in _EN_STOPWORDS or w in _ZH_COMMON)
    sents = [s for s in _SENT_SPLIT.split(text) if s.strip()]
    lines = text.split("\n")
    nonempty = [ln for ln in lines if ln.strip()]
    line_dup = 1.0 - len(set(nonempty)) / max(len(nonempty), 1)
    alpha = sum(1 for c in text if c.isalpha()) / n
    digit = sum(1 for c in text if c.isdigit()) / n
    punct = sum(1 for c in text if not c.isalnum() and not c.isspace()) / n
    sent_words = [len(_WORD_RE.findall(s)) for s in sents]
    return {
        "chars": float(n),
        "uniq_ratio": uniq / max(nw, 1),
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "stop_cover": stop_hits / max(nw, 1),
        "sent_len": float(np.mean(sent_words)) if sent_words else float(nw),
        "para_density": len(nonempty) / max(len(lines), 1),
        "line_dup": line_dup,
        "alpha_ratio": alpha,
        "digit_ratio": digit,
        "punct_ratio": punct,
        "url_density": len(_URL_RE.findall(text)) * 1000.0 / n,
        "boiler_hits": float(len(_BOILER_RE.findall(text))),
        "garbled": float(text.count("\ufffd") + text.count("Ã")),
    }


def rule_score(text: str) -> float:
    """规则分主函数：特征 → 分段线性聚合 → [0,1]。

    权重锚定 spotcheck 实测分布（健康域 p50 附近给 0.5-0.7，病态样本 <0.2）；
    阈值为经验值，goldrule 研究子集上可系统性标定后修订（配置化见 scorer.yaml）。"""
    f = rule_features(text)
    if f["chars"] < 50:
        return 0.0
    s_uniq = _ramp(f["uniq_ratio"], 0.15, 0.55) * 0.22
    s_stop = _ramp(f["stop_cover"], 0.02, 0.30) * 0.18
    s_sent = _ramp(f["sent_len"], 3.0, 25.0) * (1.0 - _ramp(f["sent_len"], 60.0, 300.0)) * 0.12
    s_alpha = _ramp(f["alpha_ratio"], 0.35, 0.75) * 0.12
    s_para = _ramp(f["para_density"], 0.3, 0.9) * 0.06
    s_len = _ramp(f["chars"], 200, 2000) * 0.10
    penalty = (
        _ramp(f["line_dup"], 0.15, 0.60) * 0.25
        + _ramp(f["url_density"], 1.0, 10.0) * 0.15
        + _ramp(f["boiler_hits"], 2.0, 10.0) * 0.15
        + _ramp(f["garbled"], 1.0, 20.0) * 0.30
        + _ramp(f["punct_ratio"], 0.15, 0.40) * 0.10
        + _ramp(f["digit_ratio"], 0.15, 0.50) * 0.10
    )
    score = 0.80 * (s_uniq + s_stop + s_sent + s_alpha + s_para + s_len) - penalty
    return float(np.clip(score, 0.0, 1.0))


def rule_scores_batch(texts: list[str], max_chars: int = 8192) -> np.ndarray:
    """批量打分（S4 打分 pass 消费；截断与 fasttext 轨同口径）。"""
    return np.fromiter((rule_score(t[:max_chars] if t else "") for t in texts), dtype=np.float32, count=len(texts))
