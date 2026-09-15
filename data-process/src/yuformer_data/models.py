from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


class _StrEnum(str, enum.Enum):
    def __new__(cls, value: str):
        obj = str.__new__(cls, value)
        obj._value_ = value
        return obj

    def __str__(self) -> str:
        return self.value


class Stage(_StrEnum):
    PRETRAIN = "pretrain"
    MIDTRAIN = "midtrain"
    POSTTRAIN = "posttrain"
    SFT = "sft"


class DataCategory(_StrEnum):
    CODE = "code"
    WEB = "web"
    AGENTIC = "agentic"
    INSTRUCTION = "instruction"
    MATH = "math"
    REASONING = "reasoning"
    GENERAL_TEXT = "general_text"


class Decision(_StrEnum):
    KEEP = "keep"
    REVIEW = "review"
    DROP = "drop"


_PRIORITY = {Decision.DROP: 0, Decision.REVIEW: 1, Decision.KEEP: 2}


@dataclass(slots=True)
class Sample:
    dataset: str
    source_id: str
    stage: Stage
    category: DataCategory
    data: dict[str, Any]
    text_fields: tuple[str, ...] = ("text",)
    decision: Decision = Decision.KEEP
    reasons: list[str] = field(default_factory=list)
    flags: set[str] = field(default_factory=set)
    buckets: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    def text(self) -> str:
        parts = []
        for name in self.text_fields:
            val = self.data.get(name)
            if isinstance(val, str) and val:
                parts.append(val)
        return "\n".join(parts)

    def mark(self, decision: Decision, reason: str) -> None:
        if decision == Decision.DROP or (
            decision == Decision.REVIEW and self.decision == Decision.KEEP
        ):
            self.decision = decision
        if reason and reason not in self.reasons:
            self.reasons.append(reason)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "source_id": self.source_id,
            "stage": self.stage.value,
            "category": self.category.value,
            "decision": self.decision.value,
            "reasons": list(self.reasons),
            "flags": sorted(self.flags),
            "buckets": dict(self.buckets),
            "metrics": dict(self.metrics),
            "data": self.data,
        }


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def release_record(sample: Sample) -> dict[str, Any]:
    result = sample.as_dict()
    result["record_sha256"] = stable_hash(
        json.dumps(result["data"], ensure_ascii=False, sort_keys=True, default=str)
    )
    return result
