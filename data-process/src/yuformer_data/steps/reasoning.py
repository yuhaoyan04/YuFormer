from __future__ import annotations

import re

from ..core import Step
from ..models import Decision, Sample

_CONTROL_ARTIFACT = re.compile(
    r"(?is)<\|(?:system|assistant|user|endoftext)[^>]*\|>|\[system prompt\]|api[_ ]error"
)
_REASON_SIGNAL = re.compile(r"(?i)(?:because|therefore|first,|step \d+|we need to|由于|因此|首先|步骤)")


def normalize_reasoning_schema(sample: Sample, context) -> Sample:
    data = sample.data
    if "prompt" in data and "problem" not in data:
        data["problem"] = data["prompt"]
    if "response" in data and "reasoning" not in data:
        data["reasoning"] = data["response"]
    problem = data.get("problem")
    reasoning = data.get("reasoning")
    if isinstance(problem, str) and isinstance(reasoning, str):
        fields = ["problem", "reasoning"]
        if isinstance(data.get("final_answer"), str):
            fields.append("final_answer")
        sample.text_fields = tuple(fields)
    return sample


def detect_control_artifacts(sample: Sample, context) -> Sample:
    if _CONTROL_ARTIFACT.search(sample.text()):
        sample.mark(Decision.REVIEW, "control_or_prompt_artifact")
    return sample


def validate_reasoning_signal(sample: Sample, context) -> Sample:
    reasoning = str(sample.data.get("reasoning") or "")
    if not reasoning:
        sample.mark(Decision.DROP, "missing_reasoning")
    elif len(reasoning) < 64 or not _REASON_SIGNAL.search(reasoning):
        sample.mark(Decision.REVIEW, "weak_reasoning_signal")
    if not sample.data.get("final_answer") and not sample.data.get("answer"):
        sample.mark(Decision.REVIEW, "missing_final_answer")
    return sample


def bucket_reasoning(sample: Sample, context) -> Sample:
    data = sample.data
    sample.buckets["origin"] = "teacher_rewrite" if data.get("teacher_model") else str(data.get("origin") or "raw")
    sample.buckets["domain"] = str(data.get("domain") or data.get("source_group") or "general")
    sample.buckets["verified"] = "verified" if data.get("verified") else "unverified"
    upstream = data.get("source_row_hash")
    if upstream:
        data.setdefault("lineage_hash", str(upstream))
    return sample


STEPS: tuple[Step, ...] = (
    Step("normalize_reasoning_schema", normalize_reasoning_schema),
    Step("detect_control_artifacts", detect_control_artifacts),
    Step("validate_reasoning_signal", validate_reasoning_signal),
    Step("bucket_reasoning", bucket_reasoning),
)
