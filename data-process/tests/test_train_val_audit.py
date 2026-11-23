import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_train_val_ids.py"
SPEC = importlib.util.spec_from_file_location("audit_train_val_ids", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


def _write_fixture(tmp_path, *, overlap=False):
    members = tmp_path / "members.parquet"
    pq.write_table(
        pa.table({"doc_id": ["a", "b", "c"]}),
        members,
    )
    val = tmp_path / "val"
    val.mkdir()
    val_id = "b" if overlap else "z"
    pq.write_table(
        pa.table({"doc_id": [val_id], "domain": ["web"], "n_tokens": [12]}),
        val / "web.parquet",
    )
    (val / "_val_global_summary.json").write_text(
        json.dumps(
            {
                "per_domain": {
                    "web": {"val_docs": 1, "val_tokens": 12, "quota_tokens": 10}
                }
            }
        )
    )
    return members, val


def test_train_val_audit_passes_disjoint_ids(tmp_path):
    members, val = _write_fixture(tmp_path)
    result = audit.audit_train_val_ids(members, val, batch_size=2)
    assert result["status"] == "PASS"
    assert result["train_val_intersection_rows"] == 0


def test_train_val_audit_fails_overlap(tmp_path):
    members, val = _write_fixture(tmp_path, overlap=True)
    result = audit.audit_train_val_ids(members, val, batch_size=2)
    assert result["status"] == "FAIL"
    assert result["train_val_intersection_rows"] == 1
