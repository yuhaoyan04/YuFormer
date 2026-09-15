from __future__ import annotations

from .models import DataCategory
from .core import Pipeline, PipelineContext, Step
from .steps import common, code, web, agentic, instruction, math, reasoning

CATEGORY_STEPS: dict[DataCategory, tuple[Step, ...]] = {
    DataCategory.CODE: code.STEPS,
    DataCategory.WEB: web.STEPS,
    DataCategory.AGENTIC: agentic.STEPS,
    DataCategory.INSTRUCTION: instruction.STEPS,
    DataCategory.MATH: math.STEPS,
    DataCategory.REASONING: reasoning.STEPS,
    DataCategory.GENERAL_TEXT: (),
}


def build_pipeline(category: DataCategory, config: dict | None = None) -> Pipeline:
    category_steps = CATEGORY_STEPS.get(category, ())
    if category_steps:
        steps = (
            category_steps[:1]
            + common.COMMON_PREFIX
            + category_steps[1:]
            + common.COMMON_SUFFIX
        )
    else:
        steps = common.COMMON_PREFIX + common.COMMON_SUFFIX
    return Pipeline(steps=steps, context=PipelineContext(config=config or {}))
