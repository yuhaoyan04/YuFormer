import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "finalize_pretrain_data_gate.py"
SPEC = importlib.util.spec_from_file_location("finalize_pretrain_data_gate", SCRIPT)
finalize = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(finalize)


def write(path, payload):
    path.write_text(json.dumps(payload))
    return path


def inputs(tmp_path, *, intersection=0):
    release = write(tmp_path / "release.json", {
        "status": "PASS", "dataset_id": "data-1", "snapshot_id": "snap-1",
    })
    stream = write(tmp_path / "stream.json", {
        "status": "PASS", "dataset_id": "data-1", "snapshot_id": "snap-1",
        "roster_exact": True,
        "streams": [{"name": "train", "status": "PASS"}],
    })
    corpus = write(tmp_path / "manifest.json", {
        "status": "PASS", "dataset_id": "data-1", "snapshot_id": "snap-1",
        "name": "corpus", "totals": {"docs": 2},
    })
    audit = write(tmp_path / "corpus-audit.json", {
        "status": "PASS", "dataset_id": "data-1", "snapshot_id": "snap-1",
        "validation_intersection": intersection,
    })
    return release, stream, corpus, audit


def test_certificate_closes_all_data_identities(tmp_path):
    release, stream, corpus, audit = inputs(tmp_path)
    result = finalize.build_certificate(
        release_path=release,
        token_stream_audit_path=stream,
        corpus_manifest_path=corpus,
        corpus_audit_path=audit,
    )
    assert result["status"] == "PASS"
    assert result["gate_id"].startswith("yuformer-pretrain-data-")


def test_certificate_rejects_validation_leakage(tmp_path):
    release, stream, corpus, audit = inputs(tmp_path, intersection=1)
    with pytest.raises(ValueError, match="intersects validation"):
        finalize.build_certificate(
            release_path=release,
            token_stream_audit_path=stream,
            corpus_manifest_path=corpus,
            corpus_audit_path=audit,
        )


def test_certificate_rejects_token_stream_dataset_mismatch(tmp_path):
    release, stream, corpus, audit = inputs(tmp_path)
    payload = json.loads(stream.read_text())
    payload["dataset_id"] = "data-other"
    write(stream, payload)
    with pytest.raises(ValueError, match="data identity mismatch"):
        finalize.build_certificate(
            release_path=release,
            token_stream_audit_path=stream,
            corpus_manifest_path=corpus,
            corpus_audit_path=audit,
        )
