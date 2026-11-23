import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_token_stream_release.py"
SPEC = importlib.util.spec_from_file_location("token_stream_release_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_selected_val_contract_respects_whole_document_boundary(tmp_path):
    path = tmp_path / "val.parquet"
    pq.write_table(pa.table({"n_tokens": pa.array([4, 5, 20], type=pa.uint32())}), path)
    assert MODULE.selected_val_contract(path, 10) == (2, 9)
    assert MODULE.selected_val_contract(path, 3) == (1, 4)
    assert MODULE.selected_val_contract(path, 0) == (3, 29)


def test_snapshot_artifact_requires_unique_exact_suffix(tmp_path):
    path = tmp_path / "data" / "order.parquet"
    path.parent.mkdir()
    path.touch()
    snapshot = {"artifacts": [{"path": "data/order.parquet", "bytes": 0, "sha256": "abc"}]}
    assert MODULE.snapshot_artifact(snapshot, path)["sha256"] == "abc"


def test_v3_stream_audit_closes_document_length_ledger(tmp_path):
    order = tmp_path / "order.parquet"
    pq.write_table(pa.table({"n_tokens": pa.array([2, 3], type=pa.uint32())}), order)
    stream_dir = tmp_path / "streams"
    stream_dir.mkdir()
    binary = stream_dir / "train.tokens.bin"
    np.asarray([1, 2, 0, 3, 4, 5, 0], dtype="<u2").tofile(binary)
    lengths_path = stream_dir / "train.doc_lengths.u32"
    np.asarray([3, 4], dtype="<u4").tofile(lengths_path)
    order_record = {"path": str(order), "bytes": order.stat().st_size,
                    "sha256": MODULE.sha256_file(order)}
    snapshot = {"snapshot_id": "snapshot", "release_name": "release",
                "artifacts": [order_record]}
    meta = {
        "schema_version": 3, "name": "train", "dtype": "<u2", "bytes": binary.stat().st_size,
        "total_tokens": 7, "n_docs": 2, "sha256": MODULE.sha256_file(binary),
        "edge_sha256": MODULE.edge_sha256(binary), "tokenizer_sha256": "tok",
        "data_snapshot_id": "snapshot", "data_release_name": "release",
        "order_sha256": order_record["sha256"], "order_bytes": order_record["bytes"],
        "document_boundary": {"policy": "append_eod", "count": 2},
        "selection": {"selected_docs": 2, "selected_reference_tokens": 5,
                      "actual_tokenizer_text_tokens": 5, "selected_text_tokens": 5,
                      "reference_tokens_are_selection_only": True},
        "doc_lengths": {"file": lengths_path.name, "dtype": "<u4", "rows": 2,
                        "bytes": lengths_path.stat().st_size,
                        "sha256": MODULE.sha256_file(lengths_path),
                        "edge_sha256": MODULE.edge_sha256(lengths_path), "sum_tokens": 7,
                        "includes_appended_eod": True, "position_order": True},
    }
    (stream_dir / "train.meta.json").write_text(json.dumps(meta), encoding="utf-8")
    result = MODULE.audit_one(
        stream_dir=stream_dir, name="train", order_path=order, expected_docs=2,
        expected_reference_tokens=5, snapshot=snapshot, tokenizer_sha256="tok",
    )
    assert result["status"] == "PASS"
    assert result["actual_doc_lengths_sha256"] == meta["doc_lengths"]["sha256"]
    assert result["actual_doc_lengths_bytes"] == 8
    np.asarray([3, 3], dtype="<u4").tofile(lengths_path)
    result = MODULE.audit_one(
        stream_dir=stream_dir, name="train", order_path=order, expected_docs=2,
        expected_reference_tokens=5, snapshot=snapshot, tokenizer_sha256="tok",
    )
    assert result["status"] == "FAIL"
    assert result["checks"]["doc_lengths_sum_exact"] is False
