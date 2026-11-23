import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_tokenizer_corpus.py"
SPEC = importlib.util.spec_from_file_location("audit_tokenizer_corpus", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


def test_validation_ids_are_exact_and_unique(tmp_path):
    val = tmp_path / "val"
    val.mkdir()
    pq.write_table(pa.table({"doc_id": ["a" * 32, "b" * 32]}), val / "web.parquet")
    assert audit.load_validation_ids(val) == {"a" * 32, "b" * 32}


def test_snapshot_artifact_requires_unique_binding(tmp_path):
    order = tmp_path / "order.parquet"
    order.write_bytes(b"order")
    snapshot = {"artifacts": [{"path": "order.parquet", "bytes": 5, "sha256": "x"}]}
    assert audit.snapshot_artifact(snapshot, order)["bytes"] == 5


def test_atomic_json_replaces_complete_document(tmp_path):
    out = tmp_path / "audit.json"
    audit.atomic_json(out, {"status": "PASS"})
    assert json.loads(out.read_text()) == {"status": "PASS"}
