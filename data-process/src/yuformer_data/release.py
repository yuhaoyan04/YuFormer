"""Machine-verifiable data release gate and content-addressed dataset identity."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml


GATE_IDS = tuple(
    f"{section}{index}"
    for section, count in (
        ("A", 4),
        ("B", 3),
        ("C", 4),
        ("D", 4),
        ("E", 4),
        ("F", 3),
        ("G", 2),
    )
    for index in range(1, count + 1)
)
VALID_STATUS = {"PASS", "BLOCKED", "PENDING"}
VALID_VERIFICATION = {"machine", "manual"}


class ReleaseGateError(ValueError):
    pass


def require_within(path: Path, parent: Path, label: str) -> None:
    try:
        path.relative_to(parent)
    except ValueError as exc:
        raise ReleaseGateError(f"{label} escapes declared root: {path}") from exc


def json_pointer_get(payload: Any, pointer: str) -> Any:
    if pointer == "":
        return payload
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ReleaseGateError(f"invalid JSON pointer: {pointer!r}")
    value = payload
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            try:
                value = value[int(part)]
            except (ValueError, IndexError) as exc:
                raise ReleaseGateError(f"JSON pointer not found: {pointer}") from exc
        elif isinstance(value, dict) and part in value:
            value = value[part]
        else:
            raise ReleaseGateError(f"JSON pointer not found: {pointer}")
    return value


def validate_audit_scope(spec: dict[str, Any]) -> dict[str, Any]:
    """Require an explicit source lineage; silent source omission is not releasable."""
    scope = spec.get("audit_scope")
    if not isinstance(scope, dict):
        raise ReleaseGateError("audit_scope must be a mapping")
    included = scope.get("included_sources")
    excluded = scope.get("excluded_sources", {})
    if (
        not isinstance(included, list)
        or not included
        or not all(isinstance(source, str) and source for source in included)
    ):
        raise ReleaseGateError(
            "audit_scope.included_sources must be a non-empty string list"
        )
    if len(included) != len(set(included)):
        raise ReleaseGateError("audit_scope.included_sources contains duplicates")
    if not isinstance(excluded, dict) or not all(
        isinstance(source, str)
        and source
        and isinstance(reason, str)
        and reason.strip()
        for source, reason in excluded.items()
    ):
        raise ReleaseGateError(
            "audit_scope.excluded_sources must map sources to non-empty reasons"
        )
    overlap = sorted(set(included) & set(excluded))
    if overlap:
        raise ReleaseGateError(
            f"audit_scope sources cannot be both included and excluded: {overlap}"
        )
    return {
        "included_sources": sorted(included),
        "excluded_sources": dict(sorted(excluded.items())),
    }


def canonical_sha256(value: Any) -> str:
    blob = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_release_spec(path: str | Path) -> dict[str, Any]:
    spec_path = Path(path)
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReleaseGateError("release spec must be a mapping")
    if payload.get("schema_version") != 1:
        raise ReleaseGateError("release spec schema_version must be 1")
    return payload


def evaluate_release(
    spec: dict[str, Any],
    *,
    repo_root: str | Path,
    hash_artifacts_when_blocked: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    snapshot_root = (root / spec["snapshot_root"]).resolve()
    require_within(snapshot_root, root, "snapshot_root")
    if not isinstance(spec.get("release_name"), str) or not spec["release_name"].strip():
        raise ReleaseGateError("release_name must be a non-empty string")
    audit_scope = validate_audit_scope(spec)
    gates = spec.get("gates")
    if not isinstance(gates, dict):
        raise ReleaseGateError("gates must be a mapping")
    missing_gate_ids = sorted(set(GATE_IDS) - set(gates))
    extra_gate_ids = sorted(set(gates) - set(GATE_IDS))
    if missing_gate_ids or extra_gate_ids:
        raise ReleaseGateError(
            f"gate set mismatch: missing={missing_gate_ids}, extra={extra_gate_ids}"
        )

    gate_results: dict[str, Any] = {}
    all_pass = True
    for gate_id in GATE_IDS:
        gate = gates[gate_id]
        status = str(gate.get("status", "PENDING")).upper()
        if status not in VALID_STATUS:
            raise ReleaseGateError(
                f"{gate_id}.status must be one of {sorted(VALID_STATUS)}, got {status}"
            )
        evidence = []
        evidence_missing = []
        evidence_paths: set[str] = set()
        for raw in gate.get("evidence", []):
            path = (root / raw).resolve()
            require_within(path, root, f"{gate_id}.evidence")
            if not path.is_file():
                evidence_missing.append(str(raw))
                continue
            relative_path = path.relative_to(root).as_posix()
            evidence_paths.add(relative_path)
            evidence.append(
                {
                    "path": relative_path,
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
        verification = str(gate.get("verification", "manual")).lower()
        if verification not in VALID_VERIFICATION:
            raise ReleaseGateError(
                f"{gate_id}.verification must be one of {sorted(VALID_VERIFICATION)}"
            )
        assertion_results = []
        assertions_ok = True
        assertions = gate.get("assertions", [])
        if not isinstance(assertions, list):
            raise ReleaseGateError(f"{gate_id}.assertions must be a list")
        for index, assertion in enumerate(assertions):
            if not isinstance(assertion, dict):
                raise ReleaseGateError(f"{gate_id}.assertions[{index}] must be a mapping")
            raw = assertion.get("path")
            pointer = assertion.get("json_pointer")
            if not isinstance(raw, str) or not raw:
                raise ReleaseGateError(f"{gate_id}.assertions[{index}].path is required")
            path = (root / raw).resolve()
            require_within(path, root, f"{gate_id}.assertion")
            relative_path = path.relative_to(root).as_posix()
            record = {
                "path": relative_path,
                "json_pointer": pointer,
                "expected": assertion.get("equals"),
                "passed": False,
            }
            try:
                if relative_path not in evidence_paths:
                    raise ReleaseGateError("assertion path is not listed as gate evidence")
                payload = json.loads(path.read_text(encoding="utf-8"))
                actual = json_pointer_get(payload, pointer)
                record["actual"] = actual
                record["passed"] = actual == assertion.get("equals")
                if not record["passed"]:
                    record["error"] = "actual value does not equal expected value"
            except Exception as exc:
                record["error"] = str(exc)
            assertions_ok &= bool(record["passed"])
            assertion_results.append(record)
        effective = status
        if status == "PASS":
            if not evidence or evidence_missing:
                effective = "BLOCKED"
            elif verification == "machine" and (not assertion_results or not assertions_ok):
                effective = "BLOCKED"
            elif verification == "manual" and not gate.get("reviewer"):
                effective = "BLOCKED"
        all_pass &= effective == "PASS"
        gate_results[gate_id] = {
            "status": effective,
            "declared_status": status,
            "detail": str(gate.get("detail", "")),
            "reviewer": gate.get("reviewer"),
            "verification": verification,
            "evidence": evidence,
            "missing_evidence": evidence_missing,
            "assertions": assertion_results,
        }

    artifacts: list[dict[str, Any]] = []
    missing_artifacts: list[str] = []
    declared_artifacts = spec.get("artifacts", [])
    if not isinstance(declared_artifacts, list) or not all(
        isinstance(path, str) and path for path in declared_artifacts
    ):
        raise ReleaseGateError("artifacts must be a list of non-empty relative paths")
    if len(declared_artifacts) != len(set(declared_artifacts)):
        raise ReleaseGateError("artifacts contains duplicate paths")
    for raw in declared_artifacts:
        path = (snapshot_root / raw).resolve()
        require_within(path, snapshot_root, "artifact")
        if not path.is_file():
            missing_artifacts.append(str(raw))
            continue
        record = {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
        }
        if all_pass or hash_artifacts_when_blocked:
            record["sha256"] = file_sha256(path)
        artifacts.append(record)
    if not declared_artifacts or missing_artifacts:
        all_pass = False

    result: dict[str, Any] = {
        "schema_version": 1,
        "release_name": spec["release_name"],
        "snapshot_root": snapshot_root.relative_to(root).as_posix(),
        "audit_scope": audit_scope,
        "status": "PASS" if all_pass else "BLOCKED",
        "gates": gate_results,
        "artifacts": artifacts,
        "missing_artifacts": missing_artifacts,
        "failed_gates": [
            gate_id
            for gate_id, gate in gate_results.items()
            if gate["status"] != "PASS"
        ],
        "spec_sha256": canonical_sha256(spec),
    }
    hashed_artifacts = bool(artifacts) and not missing_artifacts and all(
        "sha256" in artifact for artifact in artifacts
    )
    if hashed_artifacts:
        result["snapshot_id"] = "yuformer-snapshot-" + canonical_sha256(
            {"audit_scope": audit_scope, "artifacts": artifacts}
        )[:20]
    else:
        result["snapshot_id"] = None
    if all_pass:
        result["dataset_id"] = "yuformer-data-" + canonical_sha256(
            {
                "release_name": result["release_name"],
                "audit_scope": audit_scope,
                "artifacts": artifacts,
                "gates": gate_results,
                "spec_sha256": result["spec_sha256"],
                "snapshot_id": result["snapshot_id"],
            }
        )[:20]
    else:
        result["dataset_id"] = None
    return result


def write_release_result(path: str | Path, result: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    tmp.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, destination)
