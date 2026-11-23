#!/usr/bin/env python3
"""Audit immutable source provenance and explicit data-use governance (G1/G2)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


REVISION_RE = re.compile(r"^[0-9a-f]{40,64}$")
REVIEW_FIELDS = (
    "training_use_review",
    "redistribution_review",
    "pii_review",
    "opt_out_review",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_evidence_path(value: object, repo_root: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def audit_source_governance(
    governance_path: Path,
    scope_path: Path,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    governance = yaml.safe_load(governance_path.read_text(encoding="utf-8")) or {}
    scope = yaml.safe_load(scope_path.read_text(encoding="utf-8")) or {}
    root = (repo_root or Path.cwd()).resolve()
    provenance_path = _resolve_evidence_path(governance.get("provenance_report"), root)
    provenance: dict[str, Any] = {}
    provenance_report_verified = False
    if provenance_path is not None and provenance_path.is_file():
        expected = governance.get("provenance_report_sha256")
        provenance_report_verified = (
            isinstance(expected, str)
            and bool(re.fullmatch(r"[0-9a-f]{64}", expected))
            and _sha256_file(provenance_path) == expected
        )
        if provenance_report_verified:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            provenance_report_verified = provenance.get("status") == "PASS"
    included = sorted((scope.get("audit_scope") or {}).get("included_sources") or [])
    excluded = sorted(((scope.get("audit_scope") or {}).get("excluded_sources") or {}).keys())
    records = governance.get("sources") or {}
    unexpected = sorted(set(records) - set(included) - set(excluded))
    missing = sorted(set(included) - set(records))
    per_source: dict[str, dict[str, Any]] = {}
    for source in included:
        record = records.get(source) or {}
        revision = record.get("immutable_revision")
        source_provenance = (provenance.get("sources") or {}).get(source) or {}
        inventory_path = _resolve_evidence_path(source_provenance.get("inventory_path"), root)
        provenance_verified = (
            provenance_report_verified
            and source_provenance.get("status") == "PASS"
            and source_provenance.get("immutable_revision") == revision
            and inventory_path is not None
            and inventory_path.is_file()
            and _sha256_file(inventory_path) == source_provenance.get("inventory_sha256")
        )
        checks = {
            "immutable_revision": isinstance(revision, str)
            and bool(REVISION_RE.fullmatch(revision)),
            "declared_license": isinstance(record.get("declared_license"), str)
            and bool(record["declared_license"].strip()),
            "evidence_url": isinstance(record.get("evidence_url"), str)
            and record["evidence_url"].startswith("https://"),
            "provenance_inventory": provenance_verified,
            "reviewer": isinstance(record.get("reviewer"), str)
            and bool(record["reviewer"].strip()),
            **{field: record.get(field) == "PASS" for field in REVIEW_FIELDS},
        }
        per_source[source] = {
            "status": "PASS" if all(checks.values()) else "BLOCKED",
            "upstream": record.get("upstream"),
            "immutable_revision": revision,
            "declared_license": record.get("declared_license"),
            "evidence_url": record.get("evidence_url"),
            "provenance_inventory": source_provenance,
            "reviewer": record.get("reviewer"),
            "checks": checks,
            "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        }
    g1_pass = not missing and not unexpected and all(
        row["checks"]["immutable_revision"]
        and row["checks"]["declared_license"]
        and row["checks"]["evidence_url"]
        and row["checks"]["provenance_inventory"]
        for row in per_source.values()
    )
    g2_pass = not missing and all(
        row["checks"]["reviewer"]
        and all(row["checks"][field] for field in REVIEW_FIELDS)
        for row in per_source.values()
    )
    return {
        "schema_version": 1,
        "status": "PASS" if g1_pass and g2_pass else "BLOCKED",
        "included_sources": included,
        "excluded_sources": excluded,
        "missing_sources": missing,
        "unexpected_sources": unexpected,
        "provenance_report": str(provenance_path) if provenance_path is not None else None,
        "provenance_report_verified": provenance_report_verified,
        "gates": {
            "G1": {
                "status": "PASS" if g1_pass else "BLOCKED",
                "detail": "every included source has an immutable revision and license evidence",
            },
            "G2": {
                "status": "PASS" if g2_pass else "BLOCKED",
                "detail": "training, redistribution, PII, and opt-out reviews are explicitly signed",
            },
        },
        "per_source": per_source,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--governance", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit_source_governance(args.governance, args.scope)
    destination = args.out.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    print(json.dumps({"status": result["status"], "gates": result["gates"]}, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
