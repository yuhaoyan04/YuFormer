#!/usr/bin/env python3
"""Fit a power-law LR rule from pilots and freeze it into E1/E2 contracts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from research_scope import group_enabled, scope_sha256
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lr_pilot_contract import selection_is_current
from batch_pilot_contract import selection_is_current as batch_selection_is_current
from audit_experiment_identities import parse_env


ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def non_embedding_params(preset: dict[str, Any]) -> int:
    h, layers, ffn = (int(preset[key]) for key in ("hidden_size", "num_layers", "intermediate_size"))
    heads, kv_heads = int(preset["num_heads"]), int(preset["num_kv_heads"])
    head_dim, kv_width = h // heads, kv_heads * (h // heads)
    full = sum((index + 1) % int(preset["attention_every"]) == 0 for index in range(layers))
    gated = layers - full if preset.get("attention_output_gate", False) else 0
    attention = layers * (h * (h + 2 * kv_width) + h * h) + gated * h * h
    mlp = layers * 3 * h * ffn
    norms = 2 * layers * h + h + (2 * layers * head_dim if preset.get("qk_norm", False) else 0)
    return attention + mlp + norms


def immutable_started_config(path: Path, cfg: dict) -> str:
    """Keep executed provenance intact; reject recipe changes after start."""
    text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    env_path = path.with_suffix('.env')
    if not env_path.exists():
        return text
    run = parse_env(env_path).get('YUFORMER_PRETRAIN_OUT')
    if not run or not (Path(run) / 'run_identity.json').exists():
        return text
    old_text = path.read_text(encoding='utf-8')
    old = yaml.safe_load(old_text)
    def recipe(value):
        return {**value, 'optim': {k: v for k, v in value['optim'].items()
                                 if k != 'lr_resolution'}}
    if recipe(old) != recipe(cfg):
        raise ValueError(f'refusing to change started research contract: {path}')
    return old_text


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    x_mean, y_mean = sum(xs) / len(xs), sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("pilot model sizes are degenerate")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)) / denominator
    return y_mean - slope * x_mean, slope


def resolve(root: Path, selection_path: Path, batch_selection_path: Path) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "PASS":
        raise ValueError("LR pilot selection is not PASS")
    if not selection_is_current(root, selection):
        raise ValueError("LR pilot selection is not the current declared bracketed contract")
    batch = json.loads(batch_selection_path.read_text(encoding="utf-8"))
    if not batch_selection_is_current(root, batch):
        raise ValueError("batch pilot selection is not the current declared bracketed contract")
    model_doc = yaml.safe_load((root / "model/configs/model_configs.yaml").read_text(encoding="utf-8"))
    presets = model_doc["presets"]
    anchors = {"50m": "e1-50m", "200m": "e1-200m", "800m": "e1-800m"}
    points = []
    for size, preset_name in anchors.items():
        lr = selection["selections"][size].get("learning_rate")
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError(f"selected learning rate is absent or invalid: {size}")
        points.append((non_embedding_params(presets[preset_name]), float(lr)))
    intercept, exponent = linear_fit(
        [math.log(params) for params, _ in points], [math.log(lr) for _, lr in points]
    )
    selection_sha = sha256(selection_path)
    batch_selection_sha = sha256(batch_selection_path)
    loss_only = (batch.get('status') == 'PASS_PROVISIONAL'
                 or bool(batch.get('selection', {}).get('loss_only')))
    selected_tps = int(batch["selection"]["tokens_per_step"])
    selected_batch_lr = float(batch["selection"]["learning_rate"])
    params_200m = non_embedding_params(presets["e1-200m"])
    predicted_base_200m_lr = math.exp(intercept + exponent * math.log(params_200m))
    batch_lr_factor = selected_batch_lr / predicted_base_200m_lr
    if (selected_tps <= 0 or not math.isfinite(batch_lr_factor) or batch_lr_factor <= 0):
        raise ValueError("batch pilot selected invalid tokens/update or LR factor")
    rows, pending = [], []
    policy_path = root / 'pretrain/configs/research_budget.yaml'
    policy = yaml.safe_load(policy_path.read_text()) if policy_path.is_file() else {}
    screen_tps = min(selected_tps, int(policy.get('screen_tokens_per_update_cap', selected_tps)))
    screen_lr_factor = batch_lr_factor * (screen_tps / selected_tps) ** 0.5
    for group in ("e1", "e2"):
        if not group_enabled(root, group):
            continue
        for path in sorted((root / "pretrain/configs" / group).glob("*.yaml")):
            cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
            preset_name = cfg["model"]["preset"]
            params = non_embedding_params(presets[preset_name])
            base_unit = (int(cfg["data"]["seq_len"])
                         * int(cfg["train"]["micro_batch_size"])
                         * int(cfg["parallel"]["world_size"]))
            if screen_tps % base_unit:
                raise ValueError(f"selected tokens/update is incompatible with {path}")
            cfg["train"]["grad_accum"] = screen_tps // base_unit
            cfg["train"]["expected_iters"] = int(cfg["train"]["budget_tokens"]) // screen_tps
            if cfg['train']['expected_iters'] < int(policy.get('minimum_screen_updates', 1)):
                raise ValueError(f'insufficient optimizer updates for {path}')
            cfg['train']['log_every'] = min(10, max(1, cfg['train']['expected_iters'] // 100))
            cfg['train']['save_every'] = max(20, min(1000, cfg['train']['expected_iters'] // 4))
            cfg["train"]["eval_every"] = max(
                20, min(500, max(1, int(cfg["train"]["expected_iters"]) // 4))
            )
            cfg["optim"]["warmup_steps"] = max(
                5,
                min(2_000, max(1, int(cfg["train"]["expected_iters"]) // 100)),
            )
            lr = math.exp(intercept + exponent * math.log(params)) * screen_lr_factor
            cfg["optim"]["lr"] = float(f"{lr:.10g}")
            cfg["optim"]["lr_resolution"] = {
                "method": "log_linear_power_law_plus_batch_pilot",
                "pilot_selection_sha256": selection_sha,
                "batch_selection_sha256": batch_selection_sha,
                "non_embedding_params": params,
                "tokens_per_step": screen_tps,
                "batch_lr_factor": screen_lr_factor,
            }
            text = immutable_started_config(path, cfg)
            pending.append((path, text))
            rows.append({"group": group, "config": str(path.relative_to(root)),
                         "preset": preset_name, "non_embedding_params": params,
                         "tokens_per_step": screen_tps,
                         "learning_rate": cfg["optim"]["lr"],
                         "config_sha256": hashlib.sha256(text.encode()).hexdigest()})
    for path, text in pending:
        atomic_text(path, text)
    return {
        "schema_version": 1,
        "status": ("PASS_PROVISIONAL" if batch.get("status") == "PASS_PROVISIONAL"
                   else "PASS"),
        "evidence_mode": ("loss_only" if loss_only
                          else "uncontended"),
        "resolver_tool_sha256": sha256(Path(__file__)),
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "selection": str(selection_path), "selection_sha256": selection_sha,
        "batch_selection": str(batch_selection_path),
        "batch_selection_sha256": batch_selection_sha,
        "selected_tokens_per_step": selected_tps,
        "screen_tokens_per_step": screen_tps,
        "budget_policy_sha256": sha256(policy_path) if policy_path.is_file() else None,
        "research_scope_sha256": scope_sha256(root),
        "selected_batch_learning_rate": selected_batch_lr,
        "batch_selection_policy": batch.get('selection_policy'),
        "batch_selection_claim": batch.get('selection_claim'),
        "batch_lr_factor": batch_lr_factor,
        "fit": {"log_intercept": intercept, "exponent": exponent,
                "formula": "lr = exp(log_intercept) * non_embedding_params ** exponent"},
        "anchors": [{"non_embedding_params": params, "learning_rate": lr} for params, lr in points],
        "configs_updated": len(rows), "configs": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--batch-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = resolve(args.root.resolve(), args.selection.resolve(), args.batch_selection.resolve())
    atomic_text(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "configs_updated": result["configs_updated"],
                      "exponent": result["fit"]["exponent"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
