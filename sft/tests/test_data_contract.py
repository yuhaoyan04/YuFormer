import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from yuformer_sft.data_contract import audit_jsonl, validate_record


def _record(sample_id="a", split="train"):
    return {
        "sample_id": sample_id,
        "source": "fixture",
        "source_revision": "a" * 40,
        "license": "Apache-2.0",
        "split": split,
        "messages": [
            {"role": "user", "content": "What is 1+1?"},
            {"role": "assistant", "content": "2"},
        ],
    }


def test_valid_record_and_tool_roundtrip():
    record = _record()
    record["messages"] = [
        {"role": "user", "content": "weather"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1", "function": {"name": "weather", "arguments": {}}}],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "sunny"},
        {"role": "assistant", "content": "It is sunny."},
    ]
    assert validate_record(record) == []


def test_rejects_unpinned_source_and_unanswered_tool():
    record = _record()
    record["source_revision"] = "main"
    record["messages"][-1] = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call-1", "function": {"name": "x"}}],
    }
    errors = validate_record(record)
    assert any("source_revision" in error for error in errors)
    assert any("unanswered tool calls" in error for error in errors)


def test_non_object_messages_are_reported_without_crashing():
    record = _record()
    record["messages"] = ["bad", {"role": "assistant", "content": "answer"}]
    errors = validate_record(record)
    assert any("must be an object" in error for error in errors)
    assert any("first non-system message" in error for error in errors)


def test_tool_responses_must_be_contiguous_and_arguments_structured():
    record = _record()
    record["messages"] = [
        {"role": "user", "content": "run"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call-1", "function": {"name": "x", "arguments": 42}}
            ],
        },
        {"role": "assistant", "content": "too early"},
    ]
    errors = validate_record(record)
    assert any("arguments" in error for error in errors)
    assert any("before pending tool calls" in error for error in errors)


def test_string_tool_arguments_must_encode_a_json_object():
    record = _record()
    record["messages"] = [
        {"role": "user", "content": "run"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call-1", "function": {"name": "x", "arguments": "[1, 2]"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "done"},
        {"role": "assistant", "content": "done"},
    ]
    errors = validate_record(record)
    assert any("must encode a JSON object" in error for error in errors)
    record["messages"][1]["tool_calls"][0]["function"]["arguments"] = '{"x": 1}'
    assert validate_record(record) == []


def test_audit_rejects_duplicate_ids_and_cross_split_dialogues(tmp_path):
    path = tmp_path / "sft.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in (_record("same", "train"), _record("same", "test")))
        + "\n"
    )
    result = audit_jsonl(path)
    assert result["status"] == "FAIL"
    assert result["cross_split_duplicates"]
