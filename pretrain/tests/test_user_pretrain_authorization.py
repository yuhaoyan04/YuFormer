import hashlib
import json
from pathlib import Path

from pretrain.scripts.user_pretrain_authorization import (
    AUTHORIZATION,
    identity_exception_is_bound,
    source_drift_only,
)


def _bound_inputs(root: Path) -> tuple[dict, dict]:
    path = root / AUTHORIZATION
    path.parent.mkdir(parents=True)
    authorization = {
        "status": "APPROVED_BY_USER",
        "authorization_scope": "skip_e1_and_launch_full_pretrain",
        "research_decision": "E1 remains failed and is waived.",
    }
    path.write_text(json.dumps(authorization))
    frozen = {
        "scale_evidence": {"status": "WAIVED_BY_USER"},
        "inputs": {AUTHORIZATION: hashlib.sha256(path.read_bytes()).hexdigest()},
    }
    return frozen, authorization


def test_only_uniform_training_source_drift_is_eligible_for_explicit_waiver(tmp_path):
    frozen, _ = _bound_inputs(tmp_path)
    audit = {
        "status": "BLOCKED", "roster_exact": True, "total": 2,
        "rows": [
            {"status": "FAIL", "errors": ["training_source_drift"]},
            {"status": "FAIL", "errors": ["training_source_drift"]},
        ],
    }
    assert source_drift_only(audit)
    assert identity_exception_is_bound(tmp_path, frozen, audit)


def test_waiver_does_not_hide_other_identity_failures(tmp_path):
    frozen, _ = _bound_inputs(tmp_path)
    audit = {
        "status": "BLOCKED", "roster_exact": True, "total": 1,
        "rows": [{"status": "FAIL", "errors": ["dataset_id_drift"]}],
    }
    assert not source_drift_only(audit)
    assert not identity_exception_is_bound(tmp_path, frozen, audit)


def test_waiver_must_be_hash_bound_to_frozen_recipe(tmp_path):
    frozen, _ = _bound_inputs(tmp_path)
    frozen["inputs"][AUTHORIZATION] = "0" * 64
    audit = {"status": "PASS"}
    assert not identity_exception_is_bound(tmp_path, frozen, audit)
