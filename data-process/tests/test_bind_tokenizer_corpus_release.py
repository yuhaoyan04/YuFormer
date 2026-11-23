import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "bind_tokenizer_corpus_release.py"
SPEC = importlib.util.spec_from_file_location("bind_tokenizer_corpus_release", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def fixtures(tmp_path):
    order = tmp_path / "order.parquet"
    order.write_bytes(b"immutable order")
    order_sha = MODULE.sha256_file(order)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "status": "PASS", "dataset_id": "old", "snapshot_id": "snap",
        "order": {"bytes": order.stat().st_size, "sha256": order_sha},
    }))
    release = tmp_path / "release.json"
    release.write_text(json.dumps({
        "status": "PASS", "dataset_id": "new", "snapshot_id": "snap",
        "artifacts": [{"path": str(order), "bytes": order.stat().st_size, "sha256": order_sha}],
    }))
    return manifest, release, order


def test_bind_allows_governance_only_release_change(tmp_path):
    manifest, release, order = fixtures(tmp_path)
    result = MODULE.bind(manifest, release, order)
    assert result["dataset_id"] == "new"
    assert result["release_binding"]["previous_dataset_id"] == "old"
    assert json.loads(manifest.read_text())["dataset_id"] == "new"


def test_bind_rejects_snapshot_change(tmp_path):
    manifest, release, order = fixtures(tmp_path)
    payload = json.loads(release.read_text())
    payload["snapshot_id"] = "other"
    release.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="snapshot changed"):
        MODULE.bind(manifest, release, order)
