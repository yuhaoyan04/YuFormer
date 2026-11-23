"""Narrow checks for explicit full-pretrain authorization and the E1 waiver."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

AUTHORIZATION = "data/health/pretrain-user-full-pretrain-authorization.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_authorization(root: Path) -> dict[str, Any]:
    value = json.loads((root / AUTHORIZATION).read_text(encoding="utf-8"))
    if (value.get("status") != "APPROVED_BY_USER"
            or value.get("authorization_scope") != "skip_e1_and_launch_full_pretrain"):
        raise ValueError("explicit user authorization for the E1 waiver/full pretrain is absent")
    return value


def source_drift_only(identity_audit: dict[str, Any]) -> bool:
    rows = identity_audit.get("rows")
    return (
        identity_audit.get("status") == "BLOCKED"
        and identity_audit.get("roster_exact") is True
        and isinstance(rows, list)
        and bool(rows)
        and len(rows) == int(identity_audit.get("total", -1))
        and all(row.get("status") == "FAIL"
                and row.get("errors") == ["training_source_drift"] for row in rows)
    )


def identity_exception_is_bound(root: Path, frozen: dict[str, Any],
                                identity_audit: dict[str, Any]) -> bool:
    """Only waive historical source drift, and only when bound by the user waiver."""
    try:
        authorization = load_authorization(root)
        bound = (frozen.get("inputs") or {}).get(AUTHORIZATION)
        if bound != sha256(root / AUTHORIZATION):
            return False
        if frozen.get("scale_evidence", {}).get("status") != "WAIVED_BY_USER":
            return False
        return identity_audit.get("status") == "PASS" or (
            source_drift_only(identity_audit)
            and bool(authorization.get("research_decision"))
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False
