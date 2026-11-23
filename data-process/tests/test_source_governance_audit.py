import importlib.util
import hashlib
import json
from pathlib import Path

import yaml


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_source_governance.py"
SPEC = importlib.util.spec_from_file_location("audit_source_governance", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


def _write(path, value):
    path.write_text(yaml.safe_dump(value), encoding="utf-8")


def test_governance_audit_is_fail_closed(tmp_path):
    scope = tmp_path / "scope.yaml"
    governance = tmp_path / "governance.yaml"
    _write(scope, {"audit_scope": {"included_sources": ["source-a"]}})
    _write(
        governance,
        {
            "sources": {
                "source-a": {
                    "upstream": "org/data",
                    "immutable_revision": None,
                    "declared_license": "ODC-By-1.0",
                    "evidence_url": "https://example.test/card",
                    "training_use_review": "PENDING",
                    "redistribution_review": "PENDING",
                    "pii_review": "PENDING",
                    "opt_out_review": "PENDING",
                    "reviewer": None,
                }
            }
        },
    )
    result = audit.audit_source_governance(governance, scope)
    assert result["status"] == "BLOCKED"
    assert result["gates"]["G1"]["status"] == "BLOCKED"
    assert result["gates"]["G2"]["status"] == "BLOCKED"


def test_governance_audit_passes_only_complete_signed_records(tmp_path):
    scope = tmp_path / "scope.yaml"
    governance = tmp_path / "governance.yaml"
    inventory = tmp_path / "source-a.inventory.jsonl"
    inventory.write_text('{"relpath":"part"}\n', encoding="utf-8")
    inventory_sha = hashlib.sha256(inventory.read_bytes()).hexdigest()
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({
        "status": "PASS",
        "sources": {"source-a": {
            "status": "PASS",
            "immutable_revision": "a" * 40,
            "inventory_path": str(inventory),
            "inventory_sha256": inventory_sha,
        }},
    }), encoding="utf-8")
    _write(scope, {"audit_scope": {"included_sources": ["source-a"]}})
    _write(
        governance,
        {
            "provenance_report": str(provenance),
            "provenance_report_sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
            "sources": {
                "source-a": {
                    "upstream": "org/data",
                    "immutable_revision": "a" * 40,
                    "declared_license": "ODC-By-1.0",
                    "evidence_url": "https://example.test/card",
                    "training_use_review": "PASS",
                    "redistribution_review": "PASS",
                    "pii_review": "PASS",
                    "opt_out_review": "PASS",
                    "reviewer": "data-governance-owner",
                }
            }
        },
    )
    result = audit.audit_source_governance(governance, scope)
    assert result["status"] == "PASS"
    assert result["gates"]["G1"]["status"] == "PASS"
    assert result["gates"]["G2"]["status"] == "PASS"


def test_governance_allows_documented_excluded_source_records(tmp_path):
    scope = tmp_path / "scope.yaml"
    governance = tmp_path / "governance.yaml"
    _write(
        scope,
        {"audit_scope": {
            "included_sources": [],
            "excluded_sources": {"source-incomplete": "immutable shard missing"},
        }},
    )
    _write(governance, {"sources": {"source-incomplete": {}}})
    result = audit.audit_source_governance(governance, scope)
    assert result["unexpected_sources"] == []
    assert result["excluded_sources"] == ["source-incomplete"]
