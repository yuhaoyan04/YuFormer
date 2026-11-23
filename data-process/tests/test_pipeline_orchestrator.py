import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "pipeline_orchestrator.py"
SPEC = importlib.util.spec_from_file_location("pipeline_orchestrator", SCRIPT)
orchestrator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(orchestrator)


def test_candidate_tokenization_allows_eod_and_review_gates_to_remain():
    assert orchestrator.candidate_ready_for_tokenization(
        {"snapshot_id": "snapshot", "failed_gates": ["E4"]}
    )
    assert orchestrator.candidate_ready_for_tokenization(
        {"snapshot_id": "snapshot", "failed_gates": []}
    )
    assert orchestrator.candidate_ready_for_tokenization(
        {"snapshot_id": "snapshot", "failed_gates": ["E1", "E2", "E4", "G1", "G2"]}
    )
    assert not orchestrator.candidate_ready_for_tokenization(
        {"snapshot_id": "snapshot", "failed_gates": ["B1", "E4"]}
    )
    assert not orchestrator.candidate_ready_for_tokenization(
        {"snapshot_id": None, "failed_gates": ["E4"]}
    )


def test_stream_meta_requires_snapshot_and_eod(tmp_path):
    path = tmp_path / "stream.meta.json"
    path.write_text(
        json.dumps(
            {
                "data_snapshot_id": "snapshot-a",
                "n_docs": 3,
                "document_boundary": {"policy": "append_eod", "count": 3},
            }
        )
    )
    assert orchestrator.stream_meta_valid(path, "snapshot-a")
    assert not orchestrator.stream_meta_valid(path, "snapshot-b")
    payload = json.loads(path.read_text())
    payload["document_boundary"]["count"] = 2
    path.write_text(json.dumps(payload))
    assert not orchestrator.stream_meta_valid(path, "snapshot-a")


def test_formal_token_stream_audit_must_be_fresh_and_match_meta(tmp_path, monkeypatch):
    streams = tmp_path / "streams"
    streams.mkdir()
    monkeypatch.setattr(orchestrator, "STREAMS", streams)
    binary = streams / "train.tokens.bin"
    meta = streams / "train.meta.json"
    binary.write_bytes(b"tokens")
    meta.write_text(json.dumps({"sha256": "stream-sha"}))
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({
        "status": "PASS",
        "dataset_id": "data-a",
        "snapshot_id": "snapshot-a",
        "val_max_tokens_per_domain": 20_000_000,
        "roster_exact": True,
        "streams": [{
            "name": "train", "status": "PASS", "actual_bytes": 6,
            "actual_sha256": "stream-sha",
        }],
    }))
    assert orchestrator.token_stream_audit_valid(audit, "data-a", "snapshot-a")
    assert not orchestrator.token_stream_audit_valid(audit, "data-b", "snapshot-a")
    binary.write_bytes(b"changed")
    assert not orchestrator.token_stream_audit_valid(audit, "data-a", "snapshot-a")
