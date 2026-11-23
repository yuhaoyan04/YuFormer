"""Fail-closed validation for multi-turn SFT JSONL records."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


REVISION_RE = re.compile(r"^[0-9a-fA-F]{40,64}$")
ROLES = {"system", "user", "assistant", "tool"}
SPLITS = {"train", "validation", "test"}


def _canonical_dialogue(messages: list[dict[str, Any]]) -> str:
    normalized = []
    for message in messages:
        normalized.append(
            {
                "role": message.get("role"),
                "content": " ".join(str(message.get("content") or "").split()),
                "reasoning_content": " ".join(
                    str(message.get("reasoning_content") or "").split()
                ),
                "tool_calls": message.get("tool_calls") or [],
                "tool_call_id": message.get("tool_call_id"),
            }
        )
    return hashlib.sha256(
        json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_record(record: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(record, dict):
        return ["record must be an object"]
    for field in ("sample_id", "source", "source_revision", "license", "split"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            errors.append(f"{field} must be a non-empty string")
    if isinstance(record.get("source_revision"), str) and not REVISION_RE.fullmatch(
        record["source_revision"]
    ):
        errors.append("source_revision must be a 40-64 hex immutable revision/content hash")
    if record.get("split") not in SPLITS:
        errors.append(f"split must be one of {sorted(SPLITS)}")

    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        errors.append("messages must be a non-empty list")
        return errors
    first_message = messages[0] if isinstance(messages[0], dict) else {}
    if first_message.get("role") == "system":
        start = 1
    else:
        start = 0
    first_non_system = messages[start] if start < len(messages) else {}
    if not isinstance(first_non_system, dict) or first_non_system.get("role") != "user":
        errors.append("the first non-system message must be user")
    assistant_targets = 0
    pending_tool_calls: set[str] = set()
    seen_tool_calls: set[str] = set()
    previous_role: str | None = None
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"messages[{index}] must be an object")
            continue
        role = message.get("role")
        if role not in ROLES:
            errors.append(f"messages[{index}].role is invalid: {role!r}")
            continue
        if pending_tool_calls and role != "tool":
            errors.append(
                f"messages[{index}] appears before pending tool calls are answered: "
                f"{sorted(pending_tool_calls)}"
            )
        if role == "system" and index != 0:
            errors.append("system message is only allowed at index 0")
        content = message.get("content", "")
        if not isinstance(content, str):
            errors.append(f"messages[{index}].content must be a string")
            content = ""
        reasoning = message.get("reasoning_content", "")
        if reasoning is not None and not isinstance(reasoning, str):
            errors.append(f"messages[{index}].reasoning_content must be a string or null")
            reasoning = ""
        calls = message.get("tool_calls") or []
        if calls and role != "assistant":
            errors.append(f"messages[{index}].tool_calls is only valid for assistant")
        if not isinstance(calls, list):
            errors.append(f"messages[{index}].tool_calls must be a list")
            calls = []
        for call_index, call in enumerate(calls):
            if not isinstance(call, dict):
                errors.append(f"messages[{index}].tool_calls[{call_index}] must be an object")
                continue
            call_id = call.get("id")
            function = call.get("function")
            if not isinstance(call_id, str) or not call_id:
                errors.append(f"messages[{index}].tool_calls[{call_index}].id is required")
                continue
            if call_id in seen_tool_calls:
                errors.append(f"duplicate tool call id: {call_id}")
            seen_tool_calls.add(call_id)
            pending_tool_calls.add(call_id)
            if (
                not isinstance(function, dict)
                or not isinstance(function.get("name"), str)
                or not function["name"].strip()
            ):
                errors.append(
                    f"messages[{index}].tool_calls[{call_index}].function.name is required"
                )
            elif not isinstance(function.get("arguments", {}), (str, dict)):
                errors.append(
                    f"messages[{index}].tool_calls[{call_index}].function.arguments "
                    "must be a JSON object or a string encoding one"
                )
            elif isinstance(function.get("arguments", {}), str):
                try:
                    parsed_arguments = json.loads(function["arguments"])
                except json.JSONDecodeError:
                    errors.append(
                        f"messages[{index}].tool_calls[{call_index}].function.arguments "
                        "is not valid JSON"
                    )
                else:
                    if not isinstance(parsed_arguments, dict):
                        errors.append(
                            f"messages[{index}].tool_calls[{call_index}].function.arguments "
                            "must encode a JSON object"
                        )
        if role == "assistant" and calls:
            assistant_targets += 1
        if role == "tool":
            call_id = message.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                errors.append(f"messages[{index}].tool_call_id is required")
            elif call_id not in pending_tool_calls:
                errors.append(f"messages[{index}] references unknown/consumed tool_call_id")
            else:
                pending_tool_calls.remove(call_id)
            if previous_role not in {"assistant", "tool"}:
                errors.append(f"messages[{index}] tool response must follow assistant/tool")
        if role == "assistant" and (content.strip() or str(reasoning or "").strip()):
            assistant_targets += 1
        previous_role = role
    if pending_tool_calls:
        errors.append(f"unanswered tool calls: {sorted(pending_tool_calls)}")
    if assistant_targets == 0:
        errors.append("record has no non-empty assistant supervision target")
    last_message = messages[-1] if isinstance(messages[-1], dict) else {}
    if last_message.get("role") != "assistant":
        errors.append("conversation must end with an assistant message")
    return errors


def audit_jsonl(path: Path) -> dict[str, Any]:
    rows = 0
    valid_rows = 0
    ids: dict[str, int] = {}
    dialogue_splits: dict[str, set[str]] = {}
    errors: list[dict[str, Any]] = []
    per_split: dict[str, int] = {}
    per_source: dict[str, int] = {}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            rows += 1
            try:
                record = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                errors.append({"line": line_number, "errors": [f"invalid JSON/UTF-8: {exc}"]})
                continue
            row_errors = validate_record(record)
            sample_id = record.get("sample_id") if isinstance(record, dict) else None
            if isinstance(sample_id, str):
                if sample_id in ids:
                    row_errors.append(f"duplicate sample_id; first seen at line {ids[sample_id]}")
                else:
                    ids[sample_id] = line_number
            # Leakage detection is independent from metadata and ID validity.  A
            # duplicate ID must not hide an otherwise parseable dialogue that
            # crosses a split boundary.
            if (
                isinstance(record, dict)
                and record.get("split") in SPLITS
                and isinstance(record.get("messages"), list)
                and record["messages"]
                and all(isinstance(message, dict) for message in record["messages"])
            ):
                dialogue_hash = _canonical_dialogue(record["messages"])
                dialogue_splits.setdefault(dialogue_hash, set()).add(record["split"])
            if not row_errors:
                valid_rows += 1
                split = record["split"]
                source = record["source"]
                per_split[split] = per_split.get(split, 0) + 1
                per_source[source] = per_source.get(source, 0) + 1
            else:
                errors.append({"line": line_number, "sample_id": sample_id, "errors": row_errors})
    cross_split_duplicates = [
        {"dialogue_sha256": value, "splits": sorted(splits)}
        for value, splits in sorted(dialogue_splits.items())
        if len(splits) > 1
    ]
    if cross_split_duplicates:
        errors.append(
            {
                "line": None,
                "errors": [
                    f"{len(cross_split_duplicates)} normalized dialogues occur across splits"
                ],
            }
        )
    return {
        "schema_version": 1,
        "status": "PASS" if not errors and rows else "FAIL",
        "path": str(path),
        "sha256": digest.hexdigest(),
        "rows": rows,
        "valid_rows": valid_rows,
        "unique_sample_ids": len(ids),
        "per_split": dict(sorted(per_split.items())),
        "per_source": dict(sorted(per_source.items())),
        "cross_split_duplicates": cross_split_duplicates,
        "errors": errors[:10_000],
        "errors_truncated": len(errors) > 10_000,
    }
