from __future__ import annotations

import re

from ..core import Step
from ..models import Decision, Sample

_IMAGE_DEP = re.compile(r"(?i)(?:<img|\.(?:png|jpg|jpeg|gif|svg)\b|\\includegraphics|see (?:the )?figure)")
_ANSWER_SIGNAL = re.compile(r"(?i)(?:final answer|answer\s*:|therefore|thus|\\boxed\{|答案|所以)")


def normalize_math_schema(sample: Sample, context) -> Sample:
    data = sample.data
    if "problem" in data and "question" not in data:
        data["question"] = data["problem"]
    if "response" in data and "solution" not in data:
        data["solution"] = data["response"]
    question = data.get("question")
    solution = data.get("solution")
    if isinstance(question, str) and isinstance(solution, str):
        sample.text_fields = ("question", "solution")
    return sample


def validate_problem_solution(sample: Sample, context) -> Sample:
    data = sample.data
    problem = str(data.get("question") or "")
    solution = str(data.get("solution") or "")
    if not problem or not solution:
        sample.mark(Decision.DROP, "missing_problem_or_solution")
    elif _IMAGE_DEP.search(problem + "\n" + solution) and not data.get("image_text"):
        sample.mark(Decision.REVIEW, "missing_multimodal_context")
    return sample


def validate_math_answer_signal(sample: Sample, context) -> Sample:
    solution = str(sample.data.get("solution") or "")
    if len(solution) < 32:
        sample.mark(Decision.REVIEW, "solution_too_short")
    if not _ANSWER_SIGNAL.search(solution) and not sample.data.get("answer"):
        sample.mark(Decision.REVIEW, "missing_final_answer_signal")
    verified = sample.data.get("verified")
    sample.buckets["verified"] = "verified" if verified is True else "failed" if verified is False else "unknown"
    if verified is False:
        sample.mark(Decision.REVIEW, "verifier_failed")
    return sample


def bucket_math(sample: Sample, context) -> Sample:
    data = sample.data
    sample.buckets["form"] = str(data.get("form") or ("textbook" if data.get("chapter") else "qa"))
    sample.buckets["difficulty"] = str(data.get("difficulty") or "unknown")
    sample.buckets["domain"] = str(data.get("domain") or data.get("subject") or "math")
    return sample


STEPS: tuple[Step, ...] = (
    Step("normalize_math_schema", normalize_math_schema),
    Step("validate_problem_solution", validate_problem_solution),
    Step("validate_math_answer_signal", validate_math_answer_signal),
    Step("bucket_math", bucket_math),
)
