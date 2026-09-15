from __future__ import annotations

import re

from ..core import Step
from ..models import Decision, Sample

_BOILERPLATE = re.compile(r"(?i)(cookie policy|accept all cookies|privacy policy|sign in|subscribe now)")
_UNKNOWN = re.compile(r"(?i)^\s*(?:unknown|n/?a|i don'?t know|无法回答|不知道)\s*[.!。！]?$", re.MULTILINE)


def normalize_web_schema(sample: Sample, context) -> Sample:
    data = sample.data
    question = data.get("question")
    answer = data.get("answer")
    if isinstance(question, str) and isinstance(answer, str):
        sample.text_fields = ("question", "answer")
        sample.buckets["subtype"] = "web_qa"
    else:
        sample.buckets.setdefault("subtype", "web_text")
    return sample


def filter_web_boilerplate(sample: Sample, context) -> Sample:
    text = sample.text()
    matches = len(_BOILERPLATE.findall(text))
    limit = int(context.config.get("web_boilerplate_limit", 4))
    if matches >= limit:
        sample.mark(Decision.REVIEW, "web_boilerplate")
    lines = [line for line in text.splitlines() if line.strip()]
    if lines and len(set(lines)) / len(lines) < 0.45:
        sample.mark(Decision.DROP, "repetitive_web_text")
    return sample


def validate_web_qa(sample: Sample, context) -> Sample:
    if sample.buckets.get("subtype") != "web_qa":
        return sample
    data = sample.data
    question = str(data.get("question") or "")
    answer = str(data.get("answer") or "")
    if len(question) < 4 or not answer:
        sample.mark(Decision.DROP, "incomplete_qa_pair")
    elif _UNKNOWN.match(answer):
        sample.mark(Decision.DROP, "unknown_answer")
    if data.get("grounded") is False:
        sample.mark(Decision.REVIEW, "ungrounded_answer")
    return sample


def bucket_web(sample: Sample, context) -> Sample:
    data = sample.data
    sample.buckets["topic"] = str(data.get("topic_top1") or data.get("topic") or "unknown")
    score = data.get("quality_score")
    if isinstance(score, (int, float)):
        sample.buckets["model_quality"] = "high" if score >= 0.8 else "medium" if score >= 0.5 else "low"
    return sample


STEPS: tuple[Step, ...] = (
    Step("normalize_web_schema", normalize_web_schema),
    Step("filter_web_boilerplate", filter_web_boilerplate),
    Step("validate_web_qa", validate_web_qa),
    Step("bucket_web", bucket_web),
)
