#!/usr/bin/env python3
"""Compute the immutable input identity for TorchTitan qualification evidence."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

from production_dependency_identity import production_asset_paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip() and not raw.lstrip().startswith("#"):
            key, value = raw.split("=", 1)
            values[key] = value
    return values


def _resolve(value: str, root: Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def qualification_identity(root: Path) -> dict[str, Any]:
    """Hash code, contracts, and small data manifests used by qualification.

    Large token binaries are represented by their audited metadata and audit
    manifest. The formal launcher independently verifies the binary edge/full
    hashes before every qualification run.
    """
    root = root.resolve()
    paths: set[Path] = set()
    tokenizer_gates: set[Path] = set()
    suffixes = {".py", ".sh", ".json"}
    for base in (root / "pretrain/backends/torchtitan", root / "model/src"):
        if base.is_dir():
            paths.update(
                path.resolve() for path in base.rglob("*")
                if path.is_file() and path.suffix in suffixes and "__pycache__" not in path.parts
            )
    for relative in (
        "model/configs/model_configs.yaml",
        "pretrain/configs/pretrain_base.yaml",
        "pretrain/scripts/run_torchtitan_qualification.py",
        "pretrain/scripts/collect_torchtitan_result.py",
        "pretrain/scripts/run_with_lease.py",
        "pretrain/scripts/assert_run_lease.py",
        "pretrain/scripts/qualification_identity.py",
        "pretrain/scripts/process_control.py",
        "pretrain/scripts/artifact_archive.py",
        "pretrain/src/yuformer_pretrain/lease.py",
    ):
        paths.add((root / relative).resolve())

    paths.update(production_asset_paths(root))
    paths.add((root / "pretrain/scripts/production_dependency_identity.py").resolve())

    config_dir = root / "pretrain/configs/qualification"
    env_by_stem: dict[str, dict[str, str]] = {}
    for path in sorted(config_dir.glob("*.env")):
        paths.add(path.resolve())
        env_by_stem[path.stem] = _parse_env(path)
    for path in sorted(config_dir.glob("*.yaml")):
        paths.add(path.resolve())
        variables = env_by_stem.get(path.stem, {})
        rendered = path.read_text(encoding="utf-8")
        for key, value in variables.items():
            rendered = rendered.replace(f"${{{key}}}", value)
        cfg = yaml.safe_load(os.path.expandvars(rendered))
        if not isinstance(cfg, dict):
            continue
        model_configs = (cfg.get("model") or {}).get("configs")
        if isinstance(model_configs, str) and "${" not in model_configs:
            paths.add(_resolve(model_configs, root))
        data = cfg.get("data") or {}
        for key in ("tokenizer_gate", "release_manifest", "token_stream_audit"):
            value = data.get(key)
            if isinstance(value, str) and "${" not in value:
                dependency = _resolve(value, root)
                paths.add(dependency)
                if key == "tokenizer_gate":
                    tokenizer_gates.add(dependency)
        for value in data.get("meta", []):
            if isinstance(value, str) and "${" not in value:
                paths.add(_resolve(value, root))

    # Bind the tokenizer bytes that the gate points to, not only its declared hash.
    for gate in tokenizer_gates:
        if not gate.is_file():
            continue
        try:
            value = json.loads(gate.read_text(encoding="utf-8"))
            tokenizer = value.get("tokenizer_path")
            if isinstance(tokenizer, str):
                paths.add(_resolve(tokenizer, root))
        except (OSError, ValueError):
            pass

    rows: list[dict[str, Any]] = []
    for path in sorted(paths, key=str):
        label = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        if path.is_file():
            rows.append({"path": label, "bytes": path.stat().st_size, "sha256": _sha256(path)})
        else:
            rows.append({"path": label, "missing": True})
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {"schema_version": 1, "sha256": digest, "files": rows}


def report_matches(report: dict[str, Any], current: dict[str, Any]) -> bool:
    recorded = report.get("qualification_identity")
    return (
        isinstance(recorded, dict)
        and recorded.get("schema_version") == current.get("schema_version")
        and recorded.get("sha256") == current.get("sha256")
    )
