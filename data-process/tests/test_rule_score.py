"""rule_score 单测：健康文本 > 垃圾文本、中文/英文双语、确定性、截断稳定。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yuformer_data.scale.rule_score import rule_score, rule_scores_batch  # noqa: E402

GOOD_EN = (
    "The history of astronomy is one of the oldest natural sciences. Ancient civilizations "
    "observed the stars and planets, developing calendars and navigation systems. In the "
    "seventeenth century, Galileo's telescopic observations revolutionized our understanding "
    "of the solar system. Newton later provided the mathematical framework that explained "
    "both terrestrial and celestial mechanics. Modern astronomy spans observational and "
    "theoretical branches, studying objects from nearby planets to distant galaxies."
)
GOOD_ZH = (
    "天文学是最古老的自然科学之一。古代文明通过观测恒星与行星制定了历法和导航系统。"
    "十七世纪，伽利略的望远镜观测彻底改变了人类对太阳系的认识。牛顿随后提出了解释"
    "天体运动的数学框架。现代天文学分为观测与理论两大分支，研究对象从行星到遥远星系。"
)
BAD_GARBLED = "Ã©Â£Ã¨ Ã¢â‚¬Â¦ Ã¯Â¼ Ã¢â„¢\ufffd\ufffd xÃ©Â£Ã¨ Ã¢â‚¬Â¦ Ã¯Â¼ Ã¢â„¢\ufffd\ufffd xÃ©Â£Ã¨ Ã¢â‚¬Â¦ Ã¯Â¼"
BAD_SPAM = (
    "buy now!!! click here www.example.com best price!!! buy now!!! "
    "www.deal.com subscribe newsletter cookie policy buy now!!! click here "
    "www.example.com best price!!! advertisement share this follow us"
)
BAD_DUP = "the same line the same line the same line the same line the same line the same line " * 8


def test_good_beats_bad():
    g_en, g_zh = rule_score(GOOD_EN), rule_score(GOOD_ZH)
    b1, b2, b3 = rule_score(BAD_GARBLED), rule_score(BAD_SPAM), rule_score(BAD_DUP)
    assert g_en > 0.35, f"good en too low: {g_en}"
    assert g_zh > 0.35, f"good zh too low: {g_zh}"
    assert b1 < g_en and b2 < g_en and b3 < 0.4
    assert b1 < 0.3 and b2 < 0.45


def test_short_empty_zero():
    assert rule_score("") == 0.0
    assert rule_score("too short") == 0.0


def test_deterministic():
    assert rule_score(GOOD_EN) == rule_score(GOOD_EN)
    batch = rule_scores_batch([GOOD_EN, BAD_SPAM, GOOD_ZH, ""])
    assert batch.dtype == np.float32 and batch.shape == (4,)
    assert batch[0] == np.float32(rule_score(GOOD_EN))


def test_truncation_stable_for_long():
    long_text = (GOOD_EN + " ") * 400
    s_full = rule_score(long_text)
    s_trunc = rule_score(long_text[:8192])
    assert abs(s_full - s_trunc) < 0.05
