from __future__ import annotations

from ..core import Step
from ..models import Decision, Sample


def _messages(sample: Sample) -> list[dict]:
    value = sample.data.get("messages") or sample.data.get("trajectory") or []
    if isinstance(value, list):
        return [m for m in value if isinstance(m, dict)]
    return []


def normalize_agentic_schema(sample: Sample, context) -> Sample:
    data = sample.data
    messages = _messages(sample)
    if messages and not isinstance(data.get("text"), str):
        turns = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if isinstance(content, str) and content:
                turns.append(f"{role or 'unknown'}: {content}")
        if turns:
            data["text"] = "\n".join(turns)
            sample.text_fields = ("text",)
    if not messages and isinstance(data.get("text"), str) and not data.get("pre_rendered_agent_trace"):
        sample.mark(Decision.REVIEW, "unstructured_agent_trace")
    sample.metrics["turns"] = float(len(messages))
    sample.buckets["subtype"] = str(data.get("sample_type") or "trajectory")
    return sample


def validate_role_order(sample: Sample, context) -> Sample:
    roles = [str(m.get("role") or "") for m in _messages(sample)]
    if not roles:
        return sample
    if roles[0] not in {"system", "user"}:
        sample.mark(Decision.REVIEW, "invalid_first_role")
    if any(not role for role in roles):
        sample.mark(Decision.DROP, "missing_message_role")
    return sample


def validate_tool_call_pairs(sample: Sample, context) -> Sample:
    if sample.data.get("pre_rendered_agent_trace"):
        return sample
    pending = 0
    tool_calls = 0
    observations = 0
    for msg in _messages(sample):
        role = str(msg.get("role") or "")
        has_call = bool(msg.get("tool_calls")) or role in {"assistant_tool", "tool_call"}
        if has_call:
            pending += 1
            tool_calls += 1
        if role in {"tool", "tool_response", "observation"}:
            observations += 1
            pending = max(0, pending - 1)
    sample.metrics["tool_calls"] = float(tool_calls)
    sample.metrics["tool_observations"] = float(observations)
    if tool_calls == 0:
        sample.mark(Decision.REVIEW, "no_tool_call")
    elif pending:
        sample.mark(Decision.REVIEW, "unpaired_tool_call")
    return sample


def detect_trajectory_outcome(sample: Sample, context) -> Sample:
    text = sample.text().casefold()
    failure_markers = ("exit due to", "permission denied", "tool error", "timed out")
    success_markers = ("submitted", "tests passed", "task completed", "success")
    failed = any(m in text for m in failure_markers)
    verified = bool(
        sample.data.get("verified")
        or sample.data.get("tests_passed")
        or any(m in text for m in success_markers)
    )
    sample.buckets["outcome"] = "verified" if verified else "failed" if failed else "weak"
    if failed and not verified:
        sample.mark(Decision.REVIEW, "failed_or_environment_trajectory")
    return sample


def bucket_agentic(sample: Sample, context) -> Sample:
    tool_names = sample.data.get("tool_names") or sample.data.get("target_tools") or []
    if isinstance(tool_names, list) and tool_names:
        sample.buckets["tool_family"] = ",".join(sorted({str(n) for n in tool_names}))
    else:
        sample.buckets.setdefault("tool_family", "unknown")
    return sample


STEPS: tuple[Step, ...] = (
    Step("normalize_agentic_schema", normalize_agentic_schema),
    Step("validate_role_order", validate_role_order),
    Step("validate_tool_call_pairs", validate_tool_call_pairs),
    Step("detect_trajectory_outcome", detect_trajectory_outcome),
    Step("bucket_agentic", bucket_agentic),
)
