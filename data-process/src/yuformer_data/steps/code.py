from __future__ import annotations

import re

from ..core import Step
from ..models import Decision, Sample

_GENERATED = re.compile(r"(?i)(?:generated file|do not edit|auto[- ]generated)")
_CODE_SIGNAL = re.compile(r"[{}();]|\b(?:def|class|function|import|package|SELECT|FROM)\b")


def normalize_code_schema(sample: Sample, context) -> Sample:
    data = sample.data
    if "content" in data and "text" not in data:
        data["text"] = data["content"]
        sample.text_fields = tuple(
            "text" if f == "content" else f for f in sample.text_fields
        )
    sample.buckets.setdefault("subtype", str(data.get("subtype") or "code"))
    return sample


def classify_code_language(sample: Sample, context) -> Sample:
    data = sample.data
    language = data.get("language")
    if not language:
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            language = metadata.get("language")
    sample.buckets["language"] = str(language or "unknown")
    return sample


def filter_non_source_assets(sample: Sample, context) -> Sample:
    data = sample.data
    path = str(data.get("path") or data.get("file_path") or "").casefold()
    denied = tuple(context.config.get("code_denied_path_parts", ("node_modules/", "vendor/", ".min.js", ".map")))
    if path and any(str(part).casefold() in path for part in denied):
        sample.mark(Decision.DROP, "non_core_or_vendored_file")
        return sample
    if _GENERATED.search(sample.text()[:4096]):
        sample.mark(Decision.REVIEW, "possibly_generated_code")
    return sample


def validate_code_signal(sample: Sample, context) -> Sample:
    text = sample.text()
    if not _CODE_SIGNAL.search(text) and len(text.splitlines()) < 3:
        sample.mark(Decision.REVIEW, "weak_code_signal")
    data = sample.data
    if any(k in data for k in ("prefix", "middle", "suffix")):
        if not all(isinstance(data.get(k), str) and data.get(k) for k in ("prefix", "middle", "suffix")):
            sample.mark(Decision.DROP, "incomplete_fim_triplet")
        else:
            sample.buckets["subtype"] = "fim"
    return sample


def bucket_code_task(sample: Sample, context) -> Sample:
    data = sample.data
    if data.get("patch") or data.get("diff"):
        sample.buckets["subtype"] = "patch"
    elif data.get("prompt") and data.get("completion"):
        sample.buckets["subtype"] = "code_qa"
    sample.buckets["verified"] = "verified" if data.get("verified") or data.get("tests_passed") else "unverified"
    return sample


STEPS: tuple[Step, ...] = (
    Step("normalize_code_schema", normalize_code_schema),
    Step("classify_code_language", classify_code_language),
    Step("filter_non_source_assets", filter_non_source_assets),
    Step("validate_code_signal", validate_code_signal),
    Step("bucket_code_task", bucket_code_task),
)
