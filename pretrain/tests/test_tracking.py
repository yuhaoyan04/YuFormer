"""experiments/_common/tracker.py 对接：metrics.jsonl + manifest 哈希链。"""
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "experiments" / "_common"))

from tracker import RunTracker, canonical_sha256  # noqa: E402


def test_metrics_jsonl_roundtrip(tmp_path):
    tr = RunTracker(tmp_path, "arm1", {"b": 2, "a": 1}, seed=42)
    tr.log_metrics(step=1, loss=3.5, lr=1e-4)
    tr.log_metrics(step=2, loss=3.0, val_loss_web=4.1)
    rows = tr.read_metrics()
    assert [r["step"] for r in rows] == [1, 2]
    assert rows[1]["val_loss_web"] == 4.1
    line = (tr.run_dir / "metrics.jsonl").read_text().splitlines()[0]
    assert json.loads(line)["loss"] == 3.5


def test_manifest_identity(tmp_path):
    cfg = {"lr": 2e-4, "seed": 1}
    tr = RunTracker(tmp_path, "arm2", cfg, seed=1)
    m = tr.write_manifest(tokenizer_sha256="abc", data_sha256=["def"])
    payload = json.loads(m.read_text())
    assert payload["config_sha256"] == canonical_sha256(cfg)
    assert payload["tokenizer_sha256"] == "abc"
    assert payload["seed"] == 1
    # config 键序不影响 hash（宪法 4：sort_keys 序列化）
    assert canonical_sha256({"seed": 1, "lr": 2e-4}) == canonical_sha256(cfg)


def test_config_hash_changes_with_semantics(tmp_path):
    a = RunTracker(tmp_path, "a", {"x": 1}, 1).config_sha256
    b = RunTracker(tmp_path, "b", {"x": 2}, 1).config_sha256
    assert a != b


def test_metrics_reject_non_monotonic_append(tmp_path):
    tr = RunTracker(tmp_path, "arm", {"x": 1}, 1)
    tr.log_metrics(step=10, loss=1.0)
    tr.log_metrics(step=10, val_loss=1.1)
    with pytest.raises(ValueError, match="non-monotonic"):
        tr.log_metrics(step=9, loss=1.2)


def test_tracker_rejects_existing_corrupt_timeline(tmp_path):
    run_dir = tmp_path / "arm"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text('{"step": 3}\n{"step": 2}\n')
    with pytest.raises(ValueError, match="non-monotonic"):
        RunTracker(tmp_path, "arm", {"x": 1}, 1)


def test_resume_rewinds_uncommitted_metrics_suffix_and_archives_it(tmp_path):
    tr = RunTracker(tmp_path, "arm", {"x": 1}, 1)
    for step in (90, 100, 101, 111):
        tr.log_metrics(step=step, loss=float(step))

    archive = tr.rewind_to_step(100, reason="resume_from_checkpoint")
    assert archive is not None
    assert [row["step"] for row in tr.read_metrics()] == [90, 100]
    archived = archive.read_text().splitlines()
    assert json.loads(archived[0]) == {
        "checkpoint_step": 100,
        "reason": "resume_from_checkpoint",
    }
    assert [json.loads(line)["step"] for line in archived[1:]] == [101, 111]

    tr.log_metrics(step=101, loss=0.5)
    assert [row["step"] for row in tr.read_metrics()] == [90, 100, 101]
