#!/usr/bin/env python3
"""Generate the pinned TorchTitan runtime qualification contracts."""
from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "pretrain/configs/qualification"
STREAMS = ROOT / "data/token_streams/m3_20261020_yf64k"


CASES = {
    "q1_single_gpu": {"world": 1, "steps": 12, "save": 6, "eval": 6},
    "q2_fsdp4_dcp": {"world": 4, "steps": 12, "save": 4, "eval": 6},
    "q3_recovery_reference": {"world": 4, "steps": 24, "save": 4, "eval": 8},
    "q3_train_interrupt": {"world": 4, "steps": 24, "save": 4, "eval": 8},
    "q3_save_interrupt": {"world": 4, "steps": 24, "save": 4, "eval": 8},
    "q3_load_interrupt": {"world": 4, "steps": 24, "save": 4, "eval": 8},
    "q4_reshard_4to2": {"world": 4, "steps": 24, "save": 4, "eval": 8},
    "q5_reshard_4to8": {"world": 4, "steps": 24, "save": 4, "eval": 8},
    'q6_large_accumulation': {'world': 1, 'steps': 2, 'save': 1, 'eval': 2},
}


def contract(name: str, spec: dict[str, int]) -> dict:
    seq, micro, accum = 4096, 1, 1
    large = name == 'q6_large_accumulation'
    if large:
        accum = 64
    budget = seq * micro * accum * spec["world"] * spec["steps"]
    return {
        "run_name": name,
        "backend": "torchtitan-v0.3.0",
        "experiment_status": "qualification",
        "seed": 20261021,
        "model": {"preset": 'e1-800m' if large else "e1-50m", "configs": "${YUFORMER_MODEL_CONFIGS}"},
        "out_dir": "${YUFORMER_PRETRAIN_OUT}",
        "data": {
            "require_release": True, "require_eod": True,
            "release_manifest": "${YUFORMER_DATA_RELEASE}",
            "tokenizer_gate": "${YUFORMER_TOKENIZER_GATE}",
            "token_stream_audit": "${YUFORMER_TOKEN_STREAM_AUDIT}",
            "stages": ["${YUFORMER_TOKEN_STREAMS}/${YUFORMER_ARM_DATA}.tokens.bin"],
            "meta": ["${YUFORMER_TOKEN_STREAMS}/${YUFORMER_ARM_DATA}.meta.json"],
            "val_dir": "${YUFORMER_TOKEN_STREAMS}", "seq_len": seq, "epochs": 1,
            "val_samples_per_domain": 1 if large else 8, "val_max_tokens_per_domain": 20_000_000,
        },
        "optim": {
            "optimizer": "adamw", "lr": 0.0001, "weight_decay": 0.1,
            "warmup_steps": 2, "lr_schedule": "cosine", "min_lr_ratio": 0.1,
            "adam_betas": [0.9, 0.95], "clip_grad": 1.0,
        },
        "train": {
            "budget_tokens": budget, "micro_batch_size": micro, "grad_accum": accum,
            "expected_iters": spec["steps"], "activation_checkpointing": large,
            "cuda_graphs": False,
            "bf16": True, "eval_every": spec["eval"], "log_every": 1,
            "save_every": spec["save"], "keep_last_n": 3, "keep_every": 1000,
        },
        "parallel": {"world_size": spec["world"], "sharding": "full_shard"},
    }


def environment(name: str, port: int) -> str:
    values = {
        "YUFORMER_PYTHON": "/opt/yuformer-torchtitan/bin/python",
        "YUFORMER_TORCHTITAN_ROOT": "/opt/torchtitan-v0.3.0",
        "YUFORMER_CONFIG": f"configs/qualification/{name}.yaml",
        "GPUS_PER_NODE": str(CASES[name]["world"]), "MASTER_PORT": str(port),
        "YUFORMER_MODEL_CONFIGS": str(ROOT / "model/configs/model_configs.yaml"),
        "YUFORMER_PRETRAIN_OUT": str(ROOT / "data/experiments/qualification" / name),
        "YUFORMER_TOKEN_STREAMS": str(STREAMS), "YUFORMER_ARM_DATA": "mix_m3_e1_100b_stage2",
        "YUFORMER_DATA_RELEASE": str(ROOT / "data/processed/releases/mix_m3_e1_100b-20261019.release.json"),
        "YUFORMER_TOKENIZER_GATE": str(ROOT / "data/TOKENIZER_READY.json"),
        "YUFORMER_TOKEN_STREAM_AUDIT": str(ROOT / "data/processed/audits/formal_token_stream_release_mix_m3_e1_100b_m3_20261020.json"),
    }
    return "".join(f"{key}={value}\n" for key, value in values.items())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    expected = {f"{name}{suffix}" for name in CASES for suffix in (".yaml", ".env")}
    for path in OUT.glob("q*.*"):
        if path.name not in expected:
            path.unlink()
    for offset, (name, spec) in enumerate(CASES.items()):
        (OUT / f"{name}.yaml").write_text(
            yaml.safe_dump(contract(name, spec), sort_keys=False), encoding="utf-8"
        )
        (OUT / f"{name}.env").write_text(environment(name, 29700 + offset), encoding="utf-8")
    print(f"generated {len(CASES)} qualification contracts in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
