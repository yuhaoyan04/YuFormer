import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_mix_identity.py"
SPEC = importlib.util.spec_from_file_location("audit_mix_identity", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


def _id(number):
    return f"{number:032x}"


def _write(tmp_path, *, duplicate=False):
    members = tmp_path / "members.parquet"
    stage1 = tmp_path / "stage1.parquet"
    stage2 = tmp_path / "stage2.parquet"
    member_ids = [_id(1), _id(2), _id(3)]
    if duplicate:
        member_ids[-1] = _id(2)
    pq.write_table(pa.table({"doc_id": member_ids}), members)
    pq.write_table(pa.table({"doc_id": [_id(1)], "position": [0]}), stage1)
    pq.write_table(
        pa.table({"doc_id": [_id(2), _id(3)], "position": [0, 1]}), stage2
    )
    return members, [stage1, stage2]


def test_mix_identity_exact_partition_audit(tmp_path):
    members, orders = _write(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    result = audit.audit_mix_identity(members, orders, work, batch_size=2)
    assert result["status"] == "PASS"
    assert result["member_rows"] == result["order_rows"] == 3


def test_mix_identity_detects_duplicate_and_multiset_mismatch(tmp_path):
    members, orders = _write(tmp_path, duplicate=True)
    work = tmp_path / "work"
    work.mkdir()
    result = audit.audit_mix_identity(members, orders, work, batch_size=2)
    assert result["status"] == "FAIL"
    assert result["duplicate_member_ids"] == 1
    assert result["mismatched_buckets"]
