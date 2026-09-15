from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from .models import Decision, Sample


@dataclass(slots=True)
class Step:
    name: str
    fn: Callable[[Sample, "PipelineContext"], Sample]
    description: str = ""

    def __call__(self, sample: Sample, context: "PipelineContext") -> Sample:
        context.count(f"step.{self.name}.seen")
        before = sample.decision
        sample = self.fn(sample, context)
        if sample.decision != before:
            context.count(f"step.{self.name}.{sample.decision.value}")
        return sample


@dataclass(slots=True)
class PipelineContext:
    config: dict[str, Any] = field(default_factory=dict)
    counters: Counter = field(default_factory=Counter)
    seen_hashes: set[str] = field(default_factory=set)
    near_dedup_state: Any = None

    def count(self, name: str, amount: int = 1) -> None:
        self.counters[name] += amount


@dataclass(slots=True)
class Pipeline:
    steps: tuple[Step, ...]
    context: PipelineContext

    def run(self, sample: Sample) -> Sample:
        self.context.count("input")
        for step in self.steps:
            sample = step(sample, self.context)
            if sample.decision == Decision.DROP:
                break
        self.context.count(f"output.{sample.decision.value}")
        return sample

    def step_names(self) -> list[str]:
        return [s.name for s in self.steps]
