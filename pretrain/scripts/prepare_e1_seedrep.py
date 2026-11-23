#!/usr/bin/env python3
"""Freeze paired, multi-seed replications of every E1 design point."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml

from audit_experiment_identities import audit, parse_env, sha256, validate_e1_roster

ROOT = Path(__file__).resolve().parents[2]
GROUP = "e1_seedrep"


def require_frozen_budget_binding(root: Path, existing: dict) -> None:
    relative = "pretrain/configs/research_budget.yaml"
    current = sha256(root / relative)
    if existing.get("inputs", {}).get(relative) != current:
        raise ValueError(
            "research_budget.yaml changed after seedrep contract freeze; "
            "record an explicit contract amendment before reusing or rebuilding"
        )


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.is_file() and path.read_text() == payload:
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    temporary.write_text(payload)
    os.replace(temporary, path)


def prepare(root: Path) -> dict:
    budget_path = root / "pretrain/configs/research_budget.yaml"
    budget_file = yaml.safe_load(budget_path.read_text())
    policy = budget_file["e1_seed_replication"]
    screening_fit_path = root / "experiments/e1_scaling/results/scaling_fit_screening_20261107.json"
    if (policy.get("scope") != "all_24_e1_design_points"
            or policy.get("token_budget") != "inherit_exact_per_arm_e1_screen_budget"
            or policy.get("total_seeds_including_screen") != [20261001, 20261002, 20261003]):
        raise ValueError("E1 seed replication policy is not the frozen three-seed protocol")
    configs = sorted((root / "pretrain/configs/e1").glob("*.yaml"))
    source_names = {path.stem for path in configs}
    validate_e1_roster(source_names)
    identity = audit(root, {"e1": source_names})
    if identity.get("status") != "PASS" or identity.get("expected_by_group", {}).get("e1") != 24:
        raise ValueError("the original 24 E1 runs must pass current identity audit before replication")

    destination = root / f"pretrain/configs/{GROUP}"
    manifest_path = root / f"data/health/{GROUP}-contracts.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("status") == "PASS":
            require_frozen_budget_binding(root, existing)
        if (existing.get("status") == "PASS"
                and existing.get("inputs", {}).get(
                    "experiments/e1_scaling/results/scaling_fit_screening_20261107.json"
                ) == (sha256(screening_fit_path) if screening_fit_path.is_file() else None)
                and all(
                (root / row["config"]).is_file()
                and sha256(root / row["config"]) == row["config_sha256"]
                and sha256(root / row["env"]) == row["env_sha256"]
                for row in existing.get("runs", []))):
            return existing

    inputs = {"pretrain/configs/research_budget.yaml": sha256(budget_path)}
    if not screening_fit_path.is_file():
        raise ValueError(f"immutable one-seed screening fit snapshot is missing: {screening_fit_path}")
    inputs[screening_fit_path.relative_to(root).as_posix()] = sha256(screening_fit_path)
    rows = []
    for source_config_path in configs:
        source_name = source_config_path.stem
        source_env_path = source_config_path.with_suffix(".env")
        source_config = yaml.safe_load(source_config_path.read_text())
        source_env = parse_env(source_env_path)
        source_run_dir = Path(source_env["YUFORMER_PRETRAIN_OUT"])
        source_result_path = source_run_dir / "result.json"
        source_result = json.loads(source_result_path.read_text())
        if source_result.get("status") != "PASS" or source_result.get("identity_sha256") is None:
            raise ValueError(f"source E1 run is not a complete passing result: {source_name}")
        if int(source_config.get("seed", -1)) != 20261001:
            raise ValueError(f"source seed drift for {source_name}")
        inputs[source_config_path.relative_to(root).as_posix()] = sha256(source_config_path)
        inputs[source_env_path.relative_to(root).as_posix()] = sha256(source_env_path)
        inputs[source_result_path.relative_to(root).as_posix()] = sha256(source_result_path)

        for seed_index, seed in enumerate(policy["additional_seeds"], start=2):
            name = f"{GROUP}_{source_name}_s{seed_index}"
            config = dict(source_config)
            config["run_name"] = name
            config["seed"] = int(seed)
            if (config["train"]["budget_tokens"] != source_config["train"]["budget_tokens"]
                    or config["train"]["expected_iters"] != source_config["train"]["expected_iters"]):
                raise ValueError(f"replication changed token/step budget: {name}")
            target_config = destination / f"{name}.yaml"
            target_env = destination / f"{name}.env"
            run_dir = root / f"data/experiments/{GROUP}/{name}"
            if (run_dir / "current_attempt.json").exists():
                if target_config.exists() and yaml.safe_load(target_config.read_text()) != config:
                    raise ValueError(f"refusing to mutate an already launched contract: {name}")
            env = dict(source_env)
            env.update({
                "YUFORMER_CONFIG": f"configs/{GROUP}/{name}.yaml",
                "YUFORMER_PRETRAIN_OUT": str(run_dir),
                "MASTER_PORT": str(29900 + len(rows)),
            })
            destination.mkdir(parents=True, exist_ok=True)
            config_text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
            env_text = "".join(f"{key}={value}\n" for key, value in env.items())
            if target_config.exists() and (run_dir / "current_attempt.json").exists():
                if target_config.read_text() != config_text or target_env.read_text() != env_text:
                    raise ValueError(f"replication contract drift after launch: {name}")
            else:
                target_config.write_text(config_text)
                target_env.write_text(env_text)
            rows.append({
                "name": name, "source_run": source_name, "seed": int(seed),
                "source_result_sha256": sha256(source_result_path),
                "config": target_config.relative_to(root).as_posix(),
                "env": target_env.relative_to(root).as_posix(),
                "config_sha256": sha256(target_config), "env_sha256": sha256(target_env),
                "budget_tokens": int(config["train"]["budget_tokens"]),
            })
    if len(rows) != 48:
        raise ValueError(f"expected 48 seed replication contracts, generated {len(rows)}")
    total_tokens = sum(row["budget_tokens"] for row in rows)
    if total_tokens != int(policy["maximum_additional_tokens"]):
        raise ValueError(f"replication token budget mismatch: {total_tokens}")
    manifest = {
        "schema_version": 1, "status": "PASS", "group": GROUP,
        "design_points": 24, "existing_seed": 20261001,
        "additional_seeds": policy["additional_seeds"],
        "total_seed_count_per_point": 3, "paired_seed_design": True,
        "token_budget_policy": policy["token_budget"],
        "maximum_additional_tokens": total_tokens,
        "source_identity_audit": {
            "status": identity["status"], "passed": identity["passed"],
            "total": identity["total"],
        },
        "source_identity_evidence": identity.get("evidence_snapshot", {}),
        "inputs": inputs, "runs": rows,
        "prepare_tool_sha256": sha256(Path(__file__)),
    }
    atomic_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    value = prepare(args.root.resolve())
    print(json.dumps({"status": value["status"], "runs": len(value["runs"]),
                      "additional_tokens": value["maximum_additional_tokens"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
