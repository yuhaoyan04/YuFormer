from __future__ import annotations

import hashlib
import re
import unicodedata

from ..core import PipelineContext, Step
from ..dedup import NearDeduper
from ..models import Decision, Sample, release_record, stable_hash

_WHITESPACE = re.compile(r"[\t\f\v ]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SECRET = re.compile(
    r"(?i)(?:api[_-]?key|secret|password|access[_-]?token)\s*[:=]\s*[\"']?[A-Za-z0-9_\-/.]{12,}"
)


def _normalize_text_value(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = unicodedata.normalize("NFKC", value)
    value = _CONTROL.sub("", value)
    lines = [_WHITESPACE.sub(" ", line).strip() for line in value.split("\n")]
    value = "\n".join(lines)
    value = _BLANK_LINES.sub("\n\n", value).strip()
    return value


def normalize_schema_text(sample: Sample, context: PipelineContext) -> Sample:
    for name in sample.text_fields:
        val = sample.data.get(name)
        if isinstance(val, str):
            sample.data[name] = _normalize_text_value(val)
    return sample


def validate_basic_integrity(sample: Sample, context: PipelineContext) -> Sample:
    text = sample.text()
    if not text:
        sample.mark(Decision.DROP, "empty_training_text")
        return sample
    minimum = int(context.config.get("min_chars", 16))
    maximum = int(context.config.get("max_chars", 2_000_000))
    if len(text) < minimum:
        sample.mark(Decision.DROP, "too_short")
    elif len(text) > maximum:
        sample.mark(Decision.REVIEW, "too_long")
    return sample


def validate_provenance(sample: Sample, context: PipelineContext) -> Sample:
    required = bool(context.config.get("require_license", False))
    license_name = sample.data.get("license")
    if not license_name:
        metadata = sample.data.get("metadata")
        if isinstance(metadata, dict):
            license_name = metadata.get("license")
    if required and not license_name:
        sample.flags.add("missing_license")
        sample.mark(Decision.REVIEW, "missing_license")
    if not sample.data.get("source"):
        sample.flags.add("missing_source")
        sample.mark(Decision.REVIEW, "missing_source")
    return sample


def scan_sensitive_content(sample: Sample, context: PipelineContext) -> Sample:
    text = sample.text()
    if _SECRET.search(text):
        sample.flags.add("possible_secret")
        sample.mark(Decision.REVIEW, "possible_secret")
    terms = context.config.get("sensitive_terms", ())
    if isinstance(terms, (list, tuple, set)) and terms:
        lowered = text.casefold()
        if any(str(term).casefold() in lowered for term in terms if str(term)):
            sample.flags.add("sensitive_term")
            sample.mark(Decision.REVIEW, "sensitive_term")
    return sample


def exact_deduplicate(sample: Sample, context: PipelineContext) -> Sample:
    normalized = _normalize_text_value(sample.text()).casefold()
    digest = stable_hash(normalized)
    sample.data.setdefault("text_sha256", digest)
    if digest in context.seen_hashes:
        sample.mark(Decision.DROP, "exact_duplicate")
    else:
        context.seen_hashes.add(digest)
    return sample


def near_deduplicate(sample: Sample, context: PipelineContext) -> Sample:
    if not bool(context.config.get("near_dedup", False)):
        return sample
    if context.near_dedup_state is None:
        context.near_dedup_state = NearDeduper(
            threshold=int(context.config.get("near_dedup_threshold", 3)),
            bands=int(context.config.get("near_dedup_bands", 4)),
        )
    deduper: NearDeduper = context.near_dedup_state
    ngram = int(context.config.get("near_dedup_ngram", 5))
    if deduper.check_and_add(sample.text(), ngram=ngram):
        sample.mark(Decision.DROP, "near_duplicate")
    return sample


def estimate_tokens_and_length_bucket(sample: Sample, context: PipelineContext) -> Sample:
    text = sample.text()
    estimator = context.config.get("token_estimator")
    if callable(estimator):
        tokens = int(estimator(text))
    else:
        tokens = max(1, (len(text) + 3) // 4)
    tokens = max(1, tokens)
    sample.metrics["estimated_tokens"] = float(tokens)

    context_lengths = context.config.get("context_lengths", [4096, 16384, 65536])
    if not isinstance(context_lengths, (list, tuple)) or not context_lengths:
        context_lengths = [4096, 16384, 65536]
    context_lengths = sorted(set(int(x) for x in context_lengths if int(x) > 0))

    label = f"gt_{context_lengths[-1] // 1024}K"
    lower = 0
    for upper in context_lengths:
        if tokens <= upper:
            label = f"{lower + 1}_{upper}"
            break
        lower = upper
    sample.buckets["length"] = label

    ctx_bucket = f"gt_{context_lengths[-1] // 1024}K"
    lower = 0
    for upper in context_lengths:
        if tokens <= upper:
            ctx_bucket = f"B{upper // 1024}K"
            break
        lower = upper
    sample.buckets["context"] = ctx_bucket
    return sample


def finalize_quality_bucket(sample: Sample, context: PipelineContext) -> Sample:
    if sample.decision == Decision.KEEP:
        bucket = "main"
    elif sample.decision == Decision.REVIEW:
        bucket = "review"
    else:
        bucket = "drop"
    sample.buckets.setdefault("quality", bucket)
    return sample


from ..quality import apply_quality_policy

COMMON_PREFIX: tuple[Step, ...] = (
    Step("normalize_schema_text", normalize_schema_text),
    Step("validate_basic_integrity", validate_basic_integrity),
    Step("validate_provenance", validate_provenance),
    Step("scan_sensitive_content", scan_sensitive_content),
)

COMMON_SUFFIX: tuple[Step, ...] = (
    Step("apply_quality_policy", apply_quality_policy),
    Step("exact_deduplicate", exact_deduplicate),
    Step("near_deduplicate", near_deduplicate),
    Step("estimate_tokens_and_length_bucket", estimate_tokens_and_length_bucket),
    Step("finalize_quality_bucket", finalize_quality_bucket),
)
