#!/usr/bin/env python3
"""Select a bracketed best stable LR pilot independently for each model size."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lr_pilot_contract import extensions, contract_sha256, EXTENSION_GROUP


NAME = re.compile(r"^e1_lr_(50m|200m|800m)_(vlow|low|mid|high|vhigh|ext[0-9]+)$")
SIZES = ("50m", "200m", "800m")
LEVELS = ("vlow", "low", "mid", "high", "vhigh")
TOKENS_PER_STEP = 4096 * 4 * 16
MATCHED_TRAINED_TOKENS = 500_000_000 // TOKENS_PER_STEP * TOKENS_PER_STEP


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
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def select(result_paths: list[Path], extra: list[dict] | None = None) -> dict[str, Any]:
    declared = {(row['size'], row['level']): row for row in (extra or [])}
    expected = {(size, level) for size in SIZES for level in LEVELS} | set(declared)
    roster: dict[tuple[str, str], dict[str, Any]] = {}
    rejected: list[dict[str, str]] = []
    duplicates: list[dict[str, str]] = []
    for path in result_paths:
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
            match = NAME.match(str(result.get("run_name", "")))
            if not match:
                rejected.append({"path": str(path), "reason": "unexpected_run_name"})
                continue
            key = (match.group(1), match.group(2))
            if key not in expected:
                rejected.append({"path": str(path), "reason": "undeclared_run_name"})
                continue
            if key in roster:
                duplicates.append({"size": key[0], "level": key[1], "path": str(path)})
                continue
            roster[key] = result
        except (OSError, ValueError) as exc:
            rejected.append({"path": str(path), "reason": f"{type(exc).__name__}:{exc}"})
    missing = sorted(f"{size}:{level}" for size, level in expected - set(roster))
    selections: dict[str, Any] = {}
    curves: dict[str, list[dict[str, Any]]] = {}
    invalid: list[dict[str, str]] = []
    boundary_selections: list[dict[str, Any]] = []
    for size in SIZES:
        eligible = []
        curves[size] = []
        levels = [*LEVELS, *(level for row_size, level in declared if row_size == size)]
        for level in levels:
            result = roster.get((size, level))
            metric = (result or {}).get("metrics", {}).get("validation_metrics/loss", {})
            loss = metric.get("last")
            lr = (result or {}).get("hyperparameters", {}).get("learning_rate")
            hyper = (result or {}).get("hyperparameters", {})
            tokens_per_step = hyper.get("tokens_per_step")
            trained_steps = hyper.get("trained_steps")
            trained_tokens = hyper.get("trained_tokens")
            identity = (result or {}).get("identity_sha256")
            valid = (result and result.get("status") == "PASS"
                     and isinstance(loss, (int, float)) and math.isfinite(float(loss))
                     and float(loss) > 0 and isinstance(lr, (int, float))
                     and math.isfinite(float(lr)) and float(lr) > 0
                     and tokens_per_step == TOKENS_PER_STEP
                     and isinstance(trained_steps, int) and not isinstance(trained_steps, bool)
                     and trained_steps > 0
                     and trained_tokens == MATCHED_TRAINED_TOKENS
                     and trained_steps * tokens_per_step == trained_tokens
                     and isinstance(identity, str) and bool(identity))
            if valid and (size, level) in declared:
                valid = math.isclose(float(lr), declared[size, level]['learning_rate'], rel_tol=1e-12)
            if valid:
                eligible.append((float(loss), level, result))
                curves[size].append({"level": level, "validation_loss": float(loss),
                                     "learning_rate": float(lr),
                                     "trained_tokens": trained_tokens,
                                     "run_name": result["run_name"]})
            elif result is not None:
                invalid.append({"run_name": str(result.get("run_name")),
                                "reason": "invalid_status_loss_lr_or_identity"})
        if eligible:
            loss, level, result = min(
                eligible,
                key=lambda row: (row[0], abs(levels.index(row[1]) - levels.index("mid"))),
            )
            selections[size] = {"level": level, "validation_loss": loss,
                                "run_name": result["run_name"],
                                "learning_rate": result.get("hyperparameters", {}).get("learning_rate"),
                                "trained_tokens": result.get("hyperparameters", {}).get("trained_tokens"),
                                "identity_sha256": result.get("identity_sha256")}
            rates = [float(row[2]['hyperparameters']['learning_rate']) for row in eligible]
            chosen_lr = float(selections[size]['learning_rate'])
            if not (any(rate < chosen_lr for rate in rates) and any(rate > chosen_lr for rate in rates)):
                boundary_selections.append({"size": size, "level": level,
                                            "learning_rate": selections[size]["learning_rate"]})
    status = ("PASS" if not missing and not duplicates and not invalid and not rejected
              and not boundary_selections and len(selections) == len(SIZES)
              else "BLOCKED")
    return {
        "schema_version": 1, "status": status,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "expected_results": len(expected), "observed_results": len(roster), "missing": missing,
        "expected_run_names": sorted(f'e1_lr_{size}_{level}' for size, level in expected),
        "duplicates": duplicates, "rejected": rejected, "invalid": invalid,
        "boundary_selections": boundary_selections, "curves": curves,
        "selections": selections,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    extra = extensions(args.root)
    paths = list(args.results_root.glob("*/result.json"))
    if extra:
        paths.extend((args.root / 'data/experiments' / EXTENSION_GROUP).glob('*/result.json'))
    result = select(sorted(paths), extra)
    result['search_contract_sha256'] = contract_sha256(args.root)
    result["analysis_tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "observed_results": result["observed_results"],
                      "selections": result["selections"]}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
