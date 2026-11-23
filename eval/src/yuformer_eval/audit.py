"""Fail-closed audit for publishable evaluation results."""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

HEX40_64 = re.compile(r"^[0-9a-f]{40,64}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_result_manifest(manifest: Any, *, root: Path) -> dict[str, Any]:
    errors: list[str] = []
    if not isinstance(manifest, dict):
        return {"schema_version": 1, "status": "FAIL", "errors": ["manifest must be an object"]}
    if manifest.get("schema_version") != 1:
        errors.append("schema_version must equal 1")

    backend = manifest.get("backend") or {}
    if backend.get("name") != "lm-evaluation-harness" or not HEX40_64.fullmatch(
        str(backend.get("commit") or "")
    ):
        errors.append("backend must name lm-evaluation-harness and pin a 40-64 hex commit")

    model = manifest.get("model") or {}
    for field in ("checkpoint_sha256", "tokenizer_sha256", "chat_template_sha256"):
        if not HEX64.fullmatch(str(model.get(field) or "")):
            errors.append(f"model.{field} must be a 64-hex content hash")

    run = manifest.get("run") or {}
    if not isinstance(run.get("seed"), int):
        errors.append("run.seed must be an integer")
    if run.get("limit") is not None:
        errors.append("publishable runs must set run.limit to null")
    if run.get("fewshot_as_multiturn") not in (True, False):
        errors.append("run.fewshot_as_multiturn must be explicit")

    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        errors.append("tasks must be a non-empty list")
        tasks = []
    names: set[str] = set()
    for index, task in enumerate(tasks):
        prefix = f"tasks[{index}]"
        if not isinstance(task, dict):
            errors.append(f"{prefix} must be an object")
            continue
        name = task.get("name")
        if not isinstance(name, str) or not name:
            errors.append(f"{prefix}.name is required")
        elif name in names:
            errors.append(f"duplicate task name: {name}")
        else:
            names.add(name)
        if not HEX40_64.fullmatch(str(task.get("dataset_revision") or "")):
            errors.append(f"{prefix}.dataset_revision must be an immutable 40-64 hex revision")
        for field in ("task_config_sha256", "prompt_template_sha256"):
            if not HEX64.fullmatch(str(task.get(field) or "")):
                errors.append(f"{prefix}.{field} must be a 64-hex content hash")
        if not isinstance(task.get("n_samples"), int) or task.get("n_samples", 0) <= 0:
            errors.append(f"{prefix}.n_samples must be positive")
        value = task.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            errors.append(f"{prefix}.value must be finite")
        if task.get("higher_is_better") not in (True, False):
            errors.append(f"{prefix}.higher_is_better must be explicit")
        contamination = task.get("contamination") or {}
        if contamination.get("checked") is not True:
            errors.append(f"{prefix}.contamination.checked must be true")
        report_hash = contamination.get("report_sha256")
        report_path = contamination.get("report_path")
        if not HEX64.fullmatch(str(report_hash or "")):
            errors.append(f"{prefix}.contamination.report_sha256 must be a 64-hex hash")
        if not isinstance(report_path, str) or not report_path:
            errors.append(f"{prefix}.contamination.report_path is required")
        else:
            relative = Path(report_path)
            if relative.is_absolute() or ".." in relative.parts:
                errors.append(f"{prefix}.contamination.report_path must stay within the evaluation root")
            else:
                resolved = root / relative
                if not resolved.is_file():
                    errors.append(f"{prefix}.contamination report is missing: {report_path}")
                elif HEX64.fullmatch(str(report_hash or "")) and _hash(resolved) != report_hash:
                    errors.append(f"{prefix}.contamination report sha256 mismatch")

    artifacts = manifest.get("artifacts") or {}
    samples_path = artifacts.get("samples_path")
    expected = artifacts.get("samples_sha256")
    if not isinstance(samples_path, str) or not samples_path:
        errors.append("artifacts.samples_path is required")
    elif not HEX64.fullmatch(str(expected or "")):
        errors.append("artifacts.samples_sha256 must be a 64-hex hash")
    else:
        path = Path(samples_path)
        if path.is_absolute() or ".." in path.parts:
            errors.append("artifacts.samples_path must stay within the evaluation root")
        else:
            resolved = root / path
            if not resolved.is_file():
                errors.append(f"samples artifact is missing: {samples_path}")
            elif _hash(resolved) != expected:
                errors.append("samples artifact sha256 mismatch")

    return {
        "schema_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "task_count": len(tasks),
        "errors": errors,
    }
