import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "recover_source_provenance.py"
SPEC = importlib.util.spec_from_file_location("recover_source_provenance", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_hf_recovery_binds_s0_size_commit_and_etag(tmp_path):
    revision = "a" * 40
    etag = "b" * 64
    raw = tmp_path / "raw" / "source" / "data" / "part.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"payload")
    metadata = (
        tmp_path / "raw" / "source" / ".cache" / "huggingface" / "download"
        / "data" / "part.jsonl.metadata"
    )
    metadata.parent.mkdir(parents=True)
    metadata.write_text(f"{revision}\n{etag}\n123.0\n", encoding="utf-8")
    summary = tmp_path / "s0" / "source" / "part.summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({
        "stage": "s0", "source": "source", "relpath": "data/part.jsonl",
        "input_bytes": 7, "error_rows": 0, "errors": {},
    }), encoding="utf-8")

    roster = MODULE.load_s0_roster(tmp_path / "s0", "source")
    record = MODULE.recover_hf_record(tmp_path / "raw", "source", revision, roster[0])
    assert record["content_etag"] == etag
    assert record["bytes"] == 7


def test_hf_recovery_rejects_wrong_revision(tmp_path):
    raw = tmp_path / "raw" / "source" / "part"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"x")
    metadata = tmp_path / "raw" / "source" / ".cache/huggingface/download/part.metadata"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(f"{'a' * 40}\n{'b' * 64}\n", encoding="utf-8")
    row = {"relpath": "part", "input_bytes": 1, "s0_error_rows": 0, "summary": "s.json"}
    try:
        MODULE.recover_hf_record(tmp_path / "raw", "source", "c" * 40, row)
    except ValueError as exc:
        assert "configured revision" in str(exc)
    else:
        raise AssertionError("revision mismatch must fail")
