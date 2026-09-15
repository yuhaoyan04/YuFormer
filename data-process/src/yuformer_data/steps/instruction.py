from __future__ import annotations

import re

from ..core import Step
from ..models import Decision, Sample

_TEMPLATE = re.compile(r"(?i)(?:as an ai language model|i cannot assist with|here is the requested response)")


def normalize_instruction_schema(sample: Sample, context) -> Sample:
    data = sample.data
    if "instruction" in data and "prompt" not in data:
        data["prompt"] = data["instruction"]
    if "output" in data and "response" not in data:
        data["response"] = data["output"]
    prompt = data.get("prompt")
    response = data.get("response")
    if isinstance(prompt, str) and isinstance(response, str):
        sample.text_fields = ("prompt", "response")
    elif isinstance(data.get("messages"), list):
        messages = data["messages"]
        turns = []
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content")
                if isinstance(content, str) and content:
                    turns.append(f"{role or 'unknown'}: {content}")
        if turns:
            data["text"] = "\n".join(turns)
            sample.text_fields = ("text",)
    return sample


def validate_instruction_pair(sample: Sample, context) -> Sample:
    data = sample.data
    messages = data.get("messages")
    if isinstance(messages, list) and messages:
        roles = {str(m.get("role") or "") for m in messages if isinstance(m, dict)}
        if "user" not in roles or "assistant" not in roles:
            sample.mark(Decision.DROP, "incomplete_instruction_messages")
        return sample
    prompt = str(data.get("prompt") or "")
    response = str(data.get("response") or "")
    if not prompt or not response:
        sample.mark(Decision.DROP, "incomplete_instruction_pair")
    elif prompt.casefold() == response.casefold():
        sample.mark(Decision.DROP, "prompt_response_identical")
    return sample


def filter_instruction_templates(sample: Sample, context) -> Sample:
    if _TEMPLATE.search(str(sample.data.get("response") or "")):
        sample.mark(Decision.REVIEW, "generic_model_template")
    return sample


def bucket_instruction(sample: Sample, context) -> Sample:
    data = sample.data
    sample.buckets["task"] = str(data.get("task") or data.get("category") or "general")
    sample.buckets["risk"] = str(data.get("risk_category") or "normal")
    messages = data.get("messages")
    sample.buckets["turns"] = "multi" if isinstance(messages, list) and len(messages) > 2 else "single"
    return sample


STEPS: tuple[Step, ...] = (
    Step("normalize_instruction_schema", normalize_instruction_schema),
    Step("validate_instruction_pair", validate_instruction_pair),
    Step("filter_instruction_templates", filter_instruction_templates),
    Step("bucket_instruction", bucket_instruction),
)
