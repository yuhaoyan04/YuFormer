#!/usr/bin/env python3
"""Select a bracketed tokens/update and LR pair from the exact nine-arm pilot."""
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
import batch_pilot_contract as contract
from research_scope import load_scope, scope_sha256


NAME = re.compile(r"^batch_(1m|2m|4m)_(low|mid|high|ext[0-9]+)$")
BATCHES = ("1m", "2m", "4m")
LEVELS = ("low", "mid", "high")
MATCHED_TRAINED_TOKENS = 2_000_000_000 // (4096 * 4 * 256) * (4096 * 4 * 256)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            old_semantic = {key: item for key, item in previous.items() if key != "observed_at"}
            new_semantic = {key: item for key, item in value.items() if key != "observed_at"}
            if old_semantic == new_semantic:
                return
            # Recheck qualification on refresh without changing the observation
            # time merely because its promotion wrapper was removed.
            if (previous.get('status') == 'PASS'
                    and previous.get('loss_evidence_qualification')
                    and value.get('status') == 'PASS_PROVISIONAL'):
                base = {k: v for k, v in old_semantic.items()
                        if k != 'loss_evidence_qualification'}
                base['status'] = 'PASS_PROVISIONAL'
                if base == new_semantic:
                    value['observed_at'] = previous.get('observed_at')
        except (OSError, ValueError, AttributeError):
            pass
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def select(paths: list[Path], extra: list[dict] | None = None,
           selection_policy: dict | None = None) -> dict[str, Any]:
    if selection_policy is not None and (selection_policy.get('mode') != 'fixed_baseline'
            or selection_policy.get('batch') != '1m' or selection_policy.get('tokens_per_step') != 1048576):
        raise ValueError('Unsupported fixed batch baseline policy')
    declared = {(row['batch'], row['level']): row for row in (extra or [])}
    expected = {(batch, level) for batch in BATCHES for level in LEVELS} | set(declared)
    roster: dict[tuple[str, str], dict[str, Any]] = {}
    rejected, duplicates, invalid = [], [], []
    for path in paths:
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            rejected.append({"path": str(path), "reason": f"{type(exc).__name__}:{exc}"})
            continue
        match = NAME.fullmatch(str(result.get("run_name", "")))
        if match is None:
            rejected.append({"path": str(path), "reason": "unexpected_run_name"})
            continue
        key = (match.group(1), match.group(2))
        if key not in expected:
            rejected.append({'path': str(path), 'reason': 'undeclared_run_name'})
            continue
        if key in roster:
            duplicates.append({"batch": key[0], "level": key[1], "path": str(path)})
        else:
            roster[key] = result
    missing = sorted(f"{batch}:{level}" for batch, level in expected - set(roster))
    curves: dict[str, list[dict[str, Any]]] = {batch: [] for batch in BATCHES}
    optimized = []
    boundary = []
    for batch in BATCHES:
        eligible = []
        levels = [*LEVELS, *(level for batch_name, level in declared if batch_name == batch)]
        for level in levels:
            result = roster.get((batch, level))
            metrics = (result or {}).get("metrics") or {}
            loss = (metrics.get("validation_metrics/loss") or {}).get("last")
            throughput = (metrics.get("throughput(tps)") or {}).get("last")
            loss_only = bool((result or {}).get("execution_context", {}).get("loss_only"))
            hyper = (result or {}).get("hyperparameters") or {}
            lr, tps = hyper.get("learning_rate"), hyper.get("tokens_per_step")
            trained_steps, trained_tokens = hyper.get("trained_steps"), hyper.get("trained_tokens")
            numeric_values = (loss, lr, tps) if loss_only else (loss, throughput, lr, tps)
            valid = (result and result.get("status") == "PASS"
                     and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                             and math.isfinite(float(value)) and float(value) > 0
                             for value in numeric_values)
                     and isinstance(trained_steps, int) and not isinstance(trained_steps, bool)
                     and trained_steps > 0
                     and trained_tokens == MATCHED_TRAINED_TOKENS
                     and trained_steps * int(tps) == trained_tokens
                     and int(tps) == {'1m': 1048576, '2m': 2097152, '4m': 4194304}[batch]
                     and isinstance(result.get("identity_sha256"), str)
                     and bool(result.get("identity_sha256")))
            if valid and (batch, level) in declared:
                valid = math.isclose(float(lr), declared[batch, level]['learning_rate'], rel_tol=1e-12)
            if not valid:
                if result is not None:
                    invalid.append({"run_name": str(result.get("run_name")),
                                    "reason": "invalid_status_metrics_hyperparameters_or_identity"})
                continue
            row = {"run_name": result["run_name"], "batch": batch, "level": level,
                   "validation_loss": float(loss),
                   "throughput_tps_per_device": (None if loss_only else float(throughput)),
                   "learning_rate": float(lr), "tokens_per_step": int(tps),
                   "trained_steps": trained_steps, "trained_tokens": trained_tokens,
                   "identity_sha256": result["identity_sha256"],
                   "loss_only": loss_only,
                   "performance_metrics_excluded": (
                       ["throughput(tps)", "mfu(%)", "wall_time"] if loss_only else []
                   )}
            curves[batch].append(row)
            eligible.append(row)
        if len(eligible) == len(levels):
            best = min(eligible, key=lambda row: (row["validation_loss"],
                                                  abs(levels.index(row["level"]) - 1)))
            optimized.append(best)
            rates = [row['learning_rate'] for row in eligible]
            if not (any(lr < best['learning_rate'] for lr in rates)
                    and any(lr > best['learning_rate'] for lr in rates)):
                boundary.append({"batch": batch, "level": best["level"],
                                 "learning_rate": best["learning_rate"]})
    selection = (min(optimized, key=lambda row: (row["validation_loss"], row["tokens_per_step"]))
                 if len(optimized) == len(BATCHES) else None)
    blocking_boundary = boundary
    if selection_policy is not None:
        selection = next((row for row in optimized if row['batch'] == selection_policy['batch']), None)
        blocking_boundary = [edge for edge in boundary if edge['batch'] == selection_policy['batch']]
    ready = (not missing and not duplicates and not invalid and not rejected
             and not blocking_boundary and selection is not None)
    status = (("PASS_PROVISIONAL" if any(row["loss_only"] for row in optimized) else "PASS")
              if ready else "BLOCKED")
    return {
        "schema_version": 1, "status": status,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "expected_results": len(expected), "observed_results": len(roster), "missing": missing,
        'expected_run_names': sorted(f'batch_{batch}_{level}' for batch, level in expected),
        "duplicates": duplicates, "rejected": rejected, "invalid": invalid,
        "boundary_selections": boundary, "curves": curves,
        'blocking_boundary_selections': blocking_boundary,
        'selection_policy': selection_policy,
        'selection_claim': ('fixed batch baseline with bracketed learning rate; '
                            'no globally optimal batch-size claim' if selection_policy else
                            'best observed result among LR-bracketed batch curves'),
        "per_batch_optima": optimized, "selection": selection,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    paths = list(args.results_root.glob('*/result.json'))
    extra = contract.extensions(args.root)
    if extra:
        paths.extend((args.root / 'data/experiments' / contract.GROUP).glob('*/result.json'))
    result = select(sorted(paths), extra, load_scope(args.root).get('batch_selection_policy'))
    result['research_scope_sha256'] = scope_sha256(args.root)
    result['search_contract_sha256'] = contract.contract_sha256(args.root)
    result["analysis_tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "selection": result["selection"]}, sort_keys=True))
    return 0 if result["status"] in {"PASS", "PASS_PROVISIONAL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
