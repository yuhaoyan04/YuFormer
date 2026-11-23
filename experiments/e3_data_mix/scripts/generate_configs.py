#!/usr/bin/env python3
"""Freeze E3 TorchTitan contracts after E2 selection and stream audit pass."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'pretrain/scripts'))
from research_budget import screen_budget


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def aligned_matched_budget(declared_budget: int, minimum_total_tokens: int,
                           seq_len: int, samples_per_step: int) -> tuple[int, int, int]:
    available_samples = (minimum_total_tokens - 1) // seq_len
    iterations = declared_budget // (seq_len * samples_per_step)
    required_samples = iterations * samples_per_step
    if iterations <= 0 or available_samples < required_samples:
        raise ValueError(
            f"E3 streams cannot supply the declared matched budget: "
            f"available_samples={available_samples}, required_samples={required_samples}"
        )
    return iterations, iterations * seq_len * samples_per_step, available_samples


def generate(root: Path) -> dict[str, Any]:
    selection = json.loads((root / "experiments/e2_architecture/results/selection.json").read_text())
    stream_audit_path = root / "data/processed/audits/e3_token_streams_20261021.json"
    audit = json.loads(stream_audit_path.read_text())
    matrix = yaml.safe_load((root / "experiments/e3_data_mix/configs/e3_matrix.yaml").read_text())
    if (selection.get("status") not in {"PASS", "PASS_PROVISIONAL"}
            or audit.get("status") != "PASS" or audit.get("roster_exact") is not True):
        raise ValueError("E2 selection and exact E3 stream audit must both PASS")
    selected_name = selection["selection"]["run_name"]
    e2_config_path = root / f"pretrain/configs/e2/{selected_name}.yaml"
    e2_config = yaml.safe_load(e2_config_path.read_text())
    preset, lr = e2_config["model"]["preset"], float(e2_config["optim"]["lr"])
    stream_rows = {row["name"]: row for row in audit["streams"] if row["name"] in matrix["arms"]}
    if set(stream_rows) != set(matrix["arms"]):
        raise ValueError("E3 train stream audit roster differs from matrix")
    seq_len, samples_per_step = 4096, 4 * 16
    # Match PretokenizedDataLoader exactly: consecutive samples share the
    # boundary token, so capacity is floor((total_tokens - 1) / seq_len).
    minimum_total_tokens = min(int(row["actual_total_tokens_with_eod"])
                               for row in stream_rows.values())
    declared_budget = screen_budget(root, int(matrix['budget_tokens_per_arm']))
    if not 0 < declared_budget <= int(matrix['budget_tokens_per_arm']):
        raise ValueError('training budget exceeds materialized stream budget')
    common_iters, budget, common_samples = aligned_matched_budget(
        declared_budget, minimum_total_tokens, seq_len, samples_per_step
    )
    config_dir = root / "pretrain/configs/e3"
    config_dir.mkdir(parents=True, exist_ok=True)
    for stale in (*config_dir.glob("*.yaml"), *config_dir.glob("*.env")):
        stale.unlink()
    generated = []
    for index, name in enumerate(sorted(matrix["arms"])):
        config = {
            "run_name": name, "backend": "torchtitan-v0.3.0", "seed": 20261021,
            "model": {"preset": preset, "configs": "${YUFORMER_MODEL_CONFIGS}"},
            "out_dir": "${YUFORMER_PRETRAIN_OUT}",
            "data": {
                "require_release": True, "require_eod": True,
                "release_manifest": "${YUFORMER_DATA_RELEASE}",
                "tokenizer_gate": "${YUFORMER_TOKENIZER_GATE}",
                "token_stream_audit": "${YUFORMER_TOKEN_STREAM_AUDIT}",
                "stages": ["${YUFORMER_TOKEN_STREAMS}/${YUFORMER_ARM_DATA}.tokens.bin"],
                "meta": ["${YUFORMER_TOKEN_STREAMS}/${YUFORMER_ARM_DATA}.meta.json"],
                "val_dir": "${YUFORMER_VAL_STREAMS}", "seq_len": seq_len, "epochs": 1,
                "val_samples_per_domain": 64, "val_max_tokens_per_domain": 20_000_000,
            },
            "optim": {"optimizer": "adamw", "lr": lr, "weight_decay": 0.1,
                      "warmup_steps": max(5, min(2000, common_iters // 100)),
                      "lr_schedule": "cosine", "min_lr_ratio": 0.1,
                      "adam_betas": [0.9, 0.95], "muon_momentum": 0.95,
                      "muon_nesterov": True, "muon_ns_steps": 5, "clip_grad": 1.0},
            "train": {"budget_tokens": budget, "micro_batch_size": 4, "grad_accum": 16,
                      "expected_iters": common_iters, "activation_checkpointing": False,
                      "bf16": True, "eval_every": max(20, min(500, common_iters // 4)), "log_every": 10,
                      "save_every": max(20, min(1000, common_iters // 4)), "keep_last_n": 2, "keep_every": 50000},
            "parallel": {"world_size": 1, "sharding": "full_shard"},
        }
        yaml_path = config_dir / f"{name}.yaml"
        atomic_text(yaml_path, yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        env = {
            "YUFORMER_PYTHON": "/opt/yuformer-torchtitan/bin/python",
            "YUFORMER_TORCHTITAN_ROOT": "/opt/torchtitan-v0.3.0",
            "YUFORMER_CONFIG": f"configs/e3/{name}.yaml", "GPUS_PER_NODE": "1",
            "MASTER_PORT": str(29700 + index),
            "YUFORMER_MODEL_CONFIGS": str(root / "model/configs/model_configs.yaml"),
            "YUFORMER_PRETRAIN_OUT": str(root / f"data/experiments/e3/{name}"),
            "YUFORMER_TOKEN_STREAMS": str(root / "data/token_streams/e3_20261021_yf64k"),
            "YUFORMER_VAL_STREAMS": str(root / "data/token_streams/m3_20261020_yf64k"),
            "YUFORMER_ARM_DATA": name,
            "YUFORMER_DATA_RELEASE": str(root / "data/experiments/e3_data_mix/orders/manifest.json"),
            "YUFORMER_TOKENIZER_GATE": str(root / "data/TOKENIZER_READY.json"),
            "YUFORMER_TOKEN_STREAM_AUDIT": str(stream_audit_path),
        }
        env_path = config_dir / f"{name}.env"
        atomic_text(env_path, "".join(f"{key}={value}\n" for key, value in env.items()))
        validation_env = dict(os.environ)
        validation_env.update(env)
        checked = subprocess.run([
            sys.executable, str(root / "pretrain/backends/torchtitan/validate_contract.py"),
            "--repo", str(root), "--config", str(yaml_path), "--print-out-dir",
        ], cwd=root, env=validation_env, capture_output=True, text=True)
        if checked.returncode:
            raise ValueError(f"{name}: runtime contract validation failed: "
                             f"{(checked.stderr or checked.stdout)[-1000:]}")
        generated.append({"run_name": name, "config": str(yaml_path.relative_to(root)),
                          "config_sha256": sha256(yaml_path),
                          "env": str(env_path.relative_to(root)),
                          "env_sha256": sha256(env_path), "contract_status": "PASS"})
    return {"status": "PASS", "selected_e2": selected_name, "preset": preset,
            "learning_rate": lr, "common_iters": common_iters,
            "declared_budget_tokens": declared_budget, "budget_tokens": budget,
            "budget_rounding_shortfall_tokens": declared_budget - budget,
            "minimum_stream_capacity_samples": common_samples,
            "inputs": {
                "experiments/e2_architecture/results/selection.json": sha256(
                    root / "experiments/e2_architecture/results/selection.json"),
                "data/processed/audits/e3_token_streams_20261021.json": sha256(stream_audit_path),
                "experiments/e3_data_mix/configs/e3_matrix.yaml": sha256(
                    root / "experiments/e3_data_mix/configs/e3_matrix.yaml"),
                'pretrain/configs/research_budget.yaml': sha256(root / 'pretrain/configs/research_budget.yaml'),
                str(e2_config_path.relative_to(root)): sha256(e2_config_path),
            },
            "generated": generated}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments/e3_data_mix/results/config_generation.json")
    args = parser.parse_args()
    atomic_text(args.output, json.dumps({
        "schema_version": 1, "status": "IN_PROGRESS", "generated": [],
    }, indent=2, sort_keys=True) + "\n")
    result = generate(args.root.resolve())
    atomic_text(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "configs": len(result["generated"]),
                      "budget_tokens": result["budget_tokens"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
