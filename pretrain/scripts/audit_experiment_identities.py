#!/usr/bin/env python3
"""Prove every completed pilot/E1/E2/E3 result belongs to its current contract."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lr_pilot_contract import extensions, EXTENSION_GROUP, CONTRACT
import batch_pilot_contract as batch_contract
from research_scope import group_enabled, POLICY as SCOPE_POLICY


ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {"e1_pilot": 15, "batch_pilot": 9, "e1": 24, "e2": 30, "e3": 8,
            "e1_performance": 5, "e1_seedrep": 48}
ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expected_e1_names() -> set[str]:
    e1 = {f"e1_ln_{size}" for size in ("50m", "100m", "200m", "400m", "800m")}
    e1 |= {f"e1_ld_300m_{size}" for size in ("100m", "300m", "1b", "3b", "9p85b")}
    e1 |= {f"e1_iso_c{line}_{size}"
           for line, sizes in ((0, ("50m", "100m", "200m", "400m", "800m")),
                               (1, ("50m", "100m", "200m", "400m", "800m")),
                               (2, ("100m", "200m", "400m", "800m")))
           for size in sizes}
    return e1


def validate_e1_roster(names: set[str]) -> None:
    expected = expected_e1_names()
    if names != expected:
        missing, unexpected = sorted(expected - names), sorted(names - expected)
        raise ValueError(f"E1 design roster mismatch: missing={missing}, unexpected={unexpected}")


def expected_names(root: Path) -> dict[str, set[str]]:
    pilots = {f"e1_lr_{size}_{level}" for size in ("50m", "200m", "800m")
              for level in ("vlow", "low", "mid", "high", "vhigh")}
    e1 = expected_e1_names()
    architectures = (
        "e2_l16", "e2_l32", "e2_l48", "e2_ffn2x", "e2_ffn4x",
        "e2_nogate", "e2_noqkn", "e2_fullrope", "e2_swaoff", "e2_untied",
    )
    e2 = {name if seed == 1 else f"{name}_s{seed}"
          for name in architectures for seed in (1, 2, 3)}
    matrix = yaml.safe_load(
        (root / "experiments/e3_data_mix/configs/e3_matrix.yaml").read_text(encoding="utf-8")
    ) if group_enabled(root, 'e3') else {'arms': {}}
    batches = {f"batch_{batch}_{level}" for batch in ("1m", "2m", "4m")
               for level in ("low", "mid", "high")}
    names = {"e1_pilot": pilots, "batch_pilot": batches, "e1": e1,
            "e2": e2, "e3": set(matrix["arms"]),
            "e1_performance": {f"e1_perf_{size}" for size in ("50m", "100m", "200m", "400m", "800m")}}
    seedrep_manifest = root / "data/health/e1_seedrep-contracts.json"
    seedrep = json.loads(seedrep_manifest.read_text()) if seedrep_manifest.is_file() else {}
    names["e1_seedrep"] = {row["name"] for row in seedrep.get("runs", [])}
    if extensions(root):
        names[EXTENSION_GROUP] = {row['run_name'] for row in extensions(root)}
    if batch_contract.extensions(root):
        names[batch_contract.GROUP] = {row['run_name'] for row in batch_contract.extensions(root)}
    return {group: roster for group, roster in names.items() if group_enabled(root, group)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_key(root: Path, path: Path) -> str:
    """Return a stable key for an artifact covered by the identity audit."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def snapshot_artifacts(root: Path, paths: set[Path]) -> dict[str, str]:
    """Hash every mutable input used to prove experiment identity."""
    return {
        evidence_key(root, path): sha256(path)
        for path in sorted(paths, key=lambda value: str(value.resolve()))
    }


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"invalid env line {number}: {path}")
        key, value = line.split("=", 1)
        if not ENV_KEY.fullmatch(key) or any(token in value for token in ("`", "$(")):
            raise ValueError(f"unsafe env line {number}: {path}")
        values[key] = value
    return values


def expand_config(path: Path, values: dict[str, str]) -> str:
    merged = {**os.environ, **values}
    missing = sorted(set(VARIABLE.findall(path.read_text(encoding="utf-8"))) - set(merged))
    if missing:
        raise ValueError(f"unresolved variables in {path}: {missing}")
    return VARIABLE.sub(lambda match: merged[match.group(1)], path.read_text(encoding="utf-8"))


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def source_sha256(root: Path) -> str:
    validator_path = root / "pretrain/backends/torchtitan/validate_contract.py"
    spec = importlib.util.spec_from_file_location("identity_validator", validator_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {validator_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._source_sha256(root)


def attempt_binding_errors(identity: dict[str, Any], attempt: dict[str, Any],
                           completion: dict[str, Any], result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    attempt_id = attempt.get("attempt_id")
    if (not isinstance(attempt_id, str) or not attempt_id
            or completion.get("attempt_id") != attempt_id
            or result.get("attempt_id") != attempt_id):
        errors.append("attempt_id_mismatch")
    if attempt.get("identity_sha256") != identity.get("identity_sha256"):
        errors.append("attempt_identity_mismatch")
    if attempt.get("status") != "PASS":
        errors.append("attempt_not_pass")
    return errors


def audit(root: Path, groups: dict[str, set[str]] | None = None) -> dict[str, Any]:
    current_source_sha = source_sha256(root)
    exact_names = expected_names(root) if groups is None else groups
    if "e1" in exact_names:
        validate_e1_roster(exact_names["e1"])
    expected_counts = {key: len(value) for key, value in exact_names.items()}
    if groups is None and extensions(root):
        expected_counts[EXTENSION_GROUP] = len(extensions(root))
    if groups is None and batch_contract.extensions(root):
        expected_counts[batch_contract.GROUP] = len(batch_contract.extensions(root))
    rows: list[dict[str, Any]] = []
    group_counts: dict[str, int] = {}
    evidence_paths: set[Path] = set()
    if groups is None and (root / SCOPE_POLICY).is_file():
        evidence_paths.add(root / SCOPE_POLICY)
    if groups is None and (root / CONTRACT).is_file():
        evidence_paths.add(root / CONTRACT)
    if groups is None and (root / batch_contract.CONTRACT).is_file():
        evidence_paths.add(root / batch_contract.CONTRACT)
    for group, expected_count in expected_counts.items():
        config_dir = root / "pretrain/configs" / group
        env_paths = sorted(config_dir.glob("*.env"))
        group_counts[group] = len(env_paths)
        for env_path in env_paths:
            errors: list[str] = []
            try:
                evidence_paths.add(env_path)
                values = parse_env(env_path)
                config_value = values["YUFORMER_CONFIG"]
                config_path = Path(config_value)
                if not config_path.is_absolute():
                    config_path = root / "pretrain" / config_path
                rendered = expand_config(config_path, values)
                config = yaml.safe_load(rendered)
                run_dir = Path(values["YUFORMER_PRETRAIN_OUT"])
                identity_path = run_dir / "run_identity.json"
                attempt_path = run_dir / "current_attempt.json"
                completion_path = run_dir / "training_complete.json"
                result_path = run_dir / "result.json"
                identity = load_object(identity_path)
                attempt = load_object(attempt_path)
                completion = load_object(completion_path)
                result = load_object(result_path)
                evidence_paths.update({config_path, identity_path, attempt_path,
                                       completion_path, result_path})
                expected_config_sha = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
                if identity.get("run_name") != config.get("run_name") or result.get("run_name") != config.get("run_name"):
                    errors.append("run_name_mismatch")
                if identity.get("resolved_config_sha256") != expected_config_sha:
                    errors.append("resolved_config_drift")
                if identity.get("source_sha256") != current_source_sha:
                    errors.append("training_source_drift")
                model_configs_path = Path(config["model"]["configs"])
                if identity.get("model_configs_sha256") != sha256(model_configs_path):
                    errors.append("model_configs_drift")
                if completion.get("identity_sha256") != identity.get("identity_sha256"):
                    errors.append("completion_identity_mismatch")
                if result.get("identity_sha256") != identity.get("identity_sha256"):
                    errors.append("result_identity_mismatch")
                errors.extend(attempt_binding_errors(identity, attempt, completion, result))
                if completion.get("status") != "PASS" or result.get("status") != "PASS":
                    errors.append("run_not_pass")
                release = load_object(Path(config["data"]["release_manifest"]))
                gate = load_object(Path(config["data"]["tokenizer_gate"]))
                stream_audit_path = Path(config["data"]["token_stream_audit"])
                stream_audit = load_object(stream_audit_path)
                evidence_paths.update({Path(config["model"]["configs"]),
                                       Path(config["data"]["release_manifest"]),
                                       Path(config["data"]["tokenizer_gate"]),
                                       stream_audit_path})
                for key, expected in (
                    ("dataset_id", release.get("dataset_id")),
                    ("snapshot_id", release.get("snapshot_id")),
                    ("tokenizer_id", gate.get("tokenizer_id")),
                    ("tokenizer_sha256", gate.get("tokenizer_sha256")),
                ):
                    if identity.get(key) != expected:
                        errors.append(f"{key}_drift")
                if identity.get("token_stream_audit_sha256") != sha256(stream_audit_path):
                    errors.append("token_stream_audit_drift")
                if any(value.get("status") != "PASS" for value in (release, gate, stream_audit)):
                    errors.append("data_gate_not_pass")
            except Exception as exc:
                errors.append(f"{type(exc).__name__}:{exc}")
                config_path = Path("")
                run_dir = Path("")
            rows.append({"group": group, "name": env_path.stem,
                         "env": str(env_path.relative_to(root)),
                         "config": str(config_path), "run_dir": str(run_dir),
                         "status": "PASS" if not errors else "FAIL", "errors": errors})
    observed_names = {
        group: {row["name"] for row in rows if row["group"] == group}
        for group in expected_counts
    }
    roster_exact = group_counts == expected_counts and observed_names == exact_names
    passed = sum(row["status"] == "PASS" for row in rows)
    total_expected = sum(expected_counts.values())
    status = "PASS" if roster_exact and passed == total_expected else "BLOCKED"
    evidence_snapshot = snapshot_artifacts(
        root, {path for path in evidence_paths if path.is_file()}
    )
    return {"schema_version": 1, "status": status,
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "audit_tool_sha256": sha256(Path(__file__)),
            "source_sha256": current_source_sha, "expected_by_group": expected_counts,
            "observed_by_group": group_counts, "roster_exact": roster_exact,
            "missing_by_group": {group: sorted(exact_names[group] - observed_names[group])
                                 for group in expected_counts},
            "unexpected_by_group": {group: sorted(observed_names[group] - exact_names[group])
                                    for group in expected_counts},
            "passed": passed, "total": total_expected, "rows": rows,
            "evidence_snapshot": evidence_snapshot,
            "evidence_snapshot_count": len(evidence_snapshot)}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            old_semantic = {key: item for key, item in previous.items() if key != "observed_at"}
            new_semantic = {key: item for key, item in value.items() if key != "observed_at"}
            if old_semantic == new_semantic:
                return
        except (OSError, ValueError, AttributeError):
            pass
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "report/experiment_identity_audit_latest.json")
    args = parser.parse_args()
    result = audit(args.root.resolve())
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "passed": result["passed"],
                      "total": result["total"], "roster_exact": result["roster_exact"]}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
