#!/usr/bin/env python3
"""Fail-closed static and source-capacity audit for the E3 mixture matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'pretrain/scripts'))
from research_budget import screen_budget
EXPECTED_ARMS = {
    "e3_m2_base", "e3_code_2x", "e3_math_2x", "e3_chinese_2x",
    "e3_academic_2x", "e3_encyclopedia_2x", "e3_m1_native", "e3_m3_quality",
}


def allocate(weights: dict[str, float], total: int) -> dict[str, int]:
    raw = {key: total * value for key, value in weights.items()}
    quotas = {key: int(value) for key, value in raw.items()}
    remaining = total - sum(quotas.values())
    order = sorted(weights, key=lambda key: (-(raw[key] - quotas[key]), key))
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def audit(root: Path, matrix_path: Path, policy_path: Path | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    policy_path = policy_path or matrix_path.with_name("e3_analysis_policy.yaml")
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    defects = []
    if int(policy.get("schema_version", 0)) != 1:
        defects.append("analysis policy schema is not supported")
    for key in ("practical_equivalence_absolute_loss",
                "maximum_domain_relative_regression_vs_m2"):
        value = policy.get(key)
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not 0 <= float(value) <= 0.1):
            defects.append(f"analysis policy {key} is outside [0, 0.1]")
    domains = list(cfg.get("domains", []))
    arms = cfg.get("arms", {})
    if set(arms) != EXPECTED_ARMS:
        defects.append("arm roster is not exact")
    release = json.loads((root / cfg["source_release"]).read_text(encoding="utf-8"))
    manifest = json.loads((root / cfg["source_manifest"]).read_text(encoding="utf-8"))
    gate = json.loads((root / cfg["tokenizer_gate"]).read_text(encoding="utf-8"))
    if release.get("status") != "PASS" or gate.get("status") != "PASS":
        defects.append("source release or tokenizer gate is not PASS")
    if release.get("snapshot_id") != gate.get("snapshot_id"):
        defects.append("source release and tokenizer snapshot differ")
    available = {key: int(value["selected_tokens"])
                 for key, value in manifest.get("selected", {}).items()}
    rows = []
    total = int(cfg["budget_tokens_per_arm"])
    training_budget = screen_budget(root, total)
    if not 0 < training_budget <= total:
        defects.append('training budget must be positive and within stream capacity')
    for name, arm in sorted(arms.items()):
        weights = {key: float(value) for key, value in arm.get("weights", {}).items()}
        if set(weights) != set(domains):
            defects.append(f"{name}: domain roster differs")
            continue
        if abs(sum(weights.values()) - 1.0) > 1e-9 or any(value <= 0 for value in weights.values()):
            defects.append(f"{name}: weights are not positive and normalized")
        if arm.get("selection") not in {"deterministic_broad", "quality_prefix"}:
            defects.append(f"{name}: unsupported selection policy")
        quotas = allocate(weights, total)
        shortages = {domain: quota - available.get(domain, 0)
                     for domain, quota in quotas.items() if quota > available.get(domain, 0)}
        if shortages:
            defects.append(f"{name}: source capacity shortfall {shortages}")
        rows.append({"arm": name, "selection": arm.get("selection"),
                     "quotas": quotas, "shortages": shortages})
    return {"schema_version": 1, "status": "PASS" if not defects else "FAIL",
            "observed_at": datetime.now(timezone.utc).isoformat(),
            "source_snapshot_id": release.get("snapshot_id"),
            "analysis_policy": {"path": str(policy_path.relative_to(root)),
                                "sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
                                **policy},
            "budget_tokens_per_arm": total, "train_budget_tokens_per_arm": training_budget,
            "arms": rows, "defects": defects}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=ROOT / "experiments/e3_data_mix/configs/e3_matrix.yaml")
    parser.add_argument("--policy", type=Path,
                        default=ROOT / "experiments/e3_data_mix/configs/e3_analysis_policy.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/e3_data_mix/results/matrix_audit.json")
    args = parser.parse_args()
    result = audit(ROOT, args.matrix, args.policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({"status": result["status"], "arms": len(result["arms"]),
                      "defects": result["defects"]}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
