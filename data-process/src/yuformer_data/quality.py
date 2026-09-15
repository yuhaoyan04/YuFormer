from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .core import PipelineContext
from .models import Decision, Sample


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    score_field: str
    keep_min: float | None = None
    review_min: float | None = None
    hard_error_field: str | None = None
    hard_error_max: float | None = None
    require_score: bool = False


def quality_policy_key(sample: Sample) -> str:
    return str(
        sample.data.get("source_key")
        or sample.data.get("source")
        or sample.dataset
        or sample.category.value
    )


def resolve_quality_policy(sample: Sample, policies: Mapping[str, Any]) -> QualityPolicy | None:
    if not isinstance(policies, Mapping):
        return None
    key = quality_policy_key(sample)
    raw = policies.get(key) or policies.get(sample.category.value) or policies.get("default")
    if not raw:
        return None
    return QualityPolicy(
        score_field=raw.get("score_field", "quality_score"),
        keep_min=raw.get("keep_min"),
        review_min=raw.get("review_min"),
        hard_error_field=raw.get("hard_error_field"),
        hard_error_max=raw.get("hard_error_max"),
        require_score=bool(raw.get("require_score", False)),
    )


def apply_quality_policy(sample: Sample, context: PipelineContext) -> Sample:
    policies = context.config.get("quality_policies")
    if not policies:
        return sample
    policy = resolve_quality_policy(sample, policies)
    if policy is None:
        return sample
    score = _number(sample.data.get(policy.score_field))
    if score is None:
        if policy.require_score:
            sample.flags.add("missing_quality_score")
            sample.mark(Decision.REVIEW, "missing_quality_score")
    else:
        sample.metrics["quality_score"] = score
        if policy.review_min is not None and score < policy.review_min:
            sample.mark(Decision.DROP, "quality_score_below_drop_threshold")
        elif policy.keep_min is not None and score < policy.keep_min:
            sample.mark(Decision.REVIEW, "quality_score_below_keep_threshold")
    if policy.hard_error_field and policy.hard_error_max is not None:
        hard_error = _number(sample.data.get(policy.hard_error_field))
        if hard_error is not None:
            sample.metrics["hard_error_score"] = hard_error
            if hard_error > policy.hard_error_max:
                sample.mark(Decision.DROP, "hard_error_score_exceeded")
    return sample
