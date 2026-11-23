from pathlib import Path

import pytest

from yuformer_data.release import GATE_IDS, ReleaseGateError, evaluate_release


def _spec(tmp_path, status="PASS"):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "data.bin").write_bytes(b"data")
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}\n")
    return {
        "schema_version": 1,
        "release_name": "test-release",
        "snapshot_root": "snapshot",
        "audit_scope": {
            "included_sources": ["source-a"],
            "excluded_sources": {"source-old": "superseded"},
        },
        "artifacts": ["data.bin"],
        "gates": {
            gate_id: {
                "status": status,
                "detail": "test",
                "verification": "manual",
                "reviewer": "tester",
                "evidence": ["evidence.json"],
            }
            for gate_id in GATE_IDS
        },
    }


def test_passed_release_gets_content_addressed_id(tmp_path):
    spec = _spec(tmp_path)
    result = evaluate_release(spec, repo_root=tmp_path)
    assert result["status"] == "PASS"
    assert result["dataset_id"].startswith("yuformer-data-")
    assert result["snapshot_id"].startswith("yuformer-snapshot-")
    assert result["artifacts"][0]["sha256"]

    (tmp_path / "snapshot" / "data.bin").write_bytes(b"changed")
    changed = evaluate_release(spec, repo_root=tmp_path)
    assert changed["dataset_id"] != result["dataset_id"]

    spec["audit_scope"]["included_sources"].append("source-b")
    changed_lineage = evaluate_release(spec, repo_root=tmp_path)
    assert changed_lineage["dataset_id"] != changed["dataset_id"]


def test_blocked_release_does_not_hash_large_artifacts(tmp_path):
    spec = _spec(tmp_path, status="PENDING")
    result = evaluate_release(spec, repo_root=tmp_path)
    assert result["status"] == "BLOCKED"
    assert result["dataset_id"] is None
    assert "sha256" not in result["artifacts"][0]
    assert result["snapshot_id"] is None

    candidate = evaluate_release(
        spec, repo_root=tmp_path, hash_artifacts_when_blocked=True
    )
    assert candidate["status"] == "BLOCKED"
    assert candidate["snapshot_id"].startswith("yuformer-snapshot-")
    assert candidate["dataset_id"] is None


def test_pass_requires_evidence(tmp_path):
    spec = _spec(tmp_path)
    spec["gates"]["A1"]["evidence"] = []
    result = evaluate_release(spec, repo_root=tmp_path)
    assert result["gates"]["A1"]["status"] == "BLOCKED"


def test_machine_verified_pass_requires_successful_json_assertion(tmp_path):
    spec = _spec(tmp_path)
    spec["gates"]["A1"].update(
        {
            "verification": "machine",
            "assertions": [
                {
                    "path": "evidence.json",
                    "json_pointer": "/status",
                    "equals": "PASS",
                }
            ],
        }
    )
    result = evaluate_release(spec, repo_root=tmp_path)
    assert result["gates"]["A1"]["status"] == "BLOCKED"
    assert result["gates"]["A1"]["assertions"][0]["passed"] is False

    (tmp_path / "evidence.json").write_text('{"status":"PASS"}\n')
    result = evaluate_release(spec, repo_root=tmp_path)
    assert result["gates"]["A1"]["status"] == "PASS"
    assert result["gates"]["A1"]["assertions"][0]["passed"] is True


def test_exact_gate_set_is_required(tmp_path):
    spec = _spec(tmp_path)
    del spec["gates"]["F3"]
    with pytest.raises(ReleaseGateError, match="gate set mismatch"):
        evaluate_release(spec, repo_root=tmp_path)


def test_release_requires_explicit_non_overlapping_lineage(tmp_path):
    spec = _spec(tmp_path)
    scope = spec.pop("audit_scope")
    with pytest.raises(ReleaseGateError, match="audit_scope"):
        evaluate_release(spec, repo_root=tmp_path)

    spec["audit_scope"] = scope
    spec["audit_scope"]["excluded_sources"]["source-a"] = "bad overlap"
    with pytest.raises(ReleaseGateError, match="both included and excluded"):
        evaluate_release(spec, repo_root=tmp_path)


def test_release_rejects_paths_outside_declared_roots(tmp_path):
    spec = _spec(tmp_path)
    spec["artifacts"] = ["../evidence.json"]
    with pytest.raises(ReleaseGateError, match="artifact escapes"):
        evaluate_release(spec, repo_root=tmp_path)

    second = tmp_path / "second"
    second.mkdir()
    spec = _spec(second)
    spec["gates"]["A1"]["evidence"] = ["../outside.json"]
    with pytest.raises(ReleaseGateError, match="evidence escapes"):
        evaluate_release(spec, repo_root=second)
