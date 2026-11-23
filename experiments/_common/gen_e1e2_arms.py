"""E1/E2 实验臂配置生成器：model preset 追加 + experiments 配置/环境文件批量生成。

E1（24 臂）：L(N) 5 规模 × D=1B + L(D) 固定约 300M 的 5 个数据点
+ isoflop 3 条等算力线 × 4-5 点（14 臂）
E2（10 个唯一架构 × 3 seeds = 30 臂）：深度×宽度、MLP 比例、组件消融与
tied/untied；共同基线跨问题复用，不重复训练同 seed。
统一口径：vocab 65536、tied（untied 臂除外）、max_seq_len 4096、AdamW + cosine；
正式后端与数据 release 均 fail-closed。
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
MODEL_CFG = REPO / "model" / "configs" / "model_configs.yaml"
PRETRAIN_CFG_DIR = REPO / "pretrain" / "configs"
sys.path.insert(0, str(REPO / 'pretrain/scripts'))
from analyze_e2_results import flops_per_token

BASE_PRESET = {
    "vocab_size": 65536, "max_seq_len": 4096, "window_size": 128, "attention_every": 6,
    "rope_every": 1, "rope_base": 5000000.0, "rope_percent": 0.334,
    "qk_norm": True, "attention_output_gate": True, "norm_eps": 1e-6,
    "tie_word_embeddings": True,
}

E1_PRESETS = {
    "e1-50m": {"num_layers": 8, "hidden_size": 512, "intermediate_size": 1408, "num_heads": 8, "num_kv_heads": 4},
    "e1-100m": {"num_layers": 12, "hidden_size": 640, "intermediate_size": 1760, "num_heads": 10, "num_kv_heads": 5},
    "e1-200m": {"num_layers": 20, "hidden_size": 768, "intermediate_size": 2112, "num_heads": 12, "num_kv_heads": 6},
    "e1-300m": {"num_layers": 24, "hidden_size": 896, "intermediate_size": 2464, "num_heads": 14, "num_kv_heads": 7},
    "e1-800m": {"num_layers": 32, "hidden_size": 1408, "intermediate_size": 3872, "num_heads": 22, "num_kv_heads": 11},
}
E2_PRESETS = {
    # head_dim 72/80 keeps the depth contrast on efficient FlashAttention shapes;
    # the former 24-head L16 point had head_dim=60 and was not tensor-core aligned.
    "e2-l16-h1440": {"num_layers": 16, "hidden_size": 1440, "intermediate_size": 3960, "num_heads": 20, "num_kv_heads": 10},
    "e2-l48-h800": {"num_layers": 48, "hidden_size": 800, "intermediate_size": 2200, "num_heads": 10, "num_kv_heads": 10},
    "e2-ffn2x": {"num_layers": 39, "hidden_size": 1024, "intermediate_size": 2048, "num_heads": 16, "num_kv_heads": 8},
    "e2-ffn4x": {"num_layers": 24, "hidden_size": 1024, "intermediate_size": 4096, "num_heads": 16, "num_kv_heads": 8},
    "e2-nogate": {"num_layers": 32, "hidden_size": 1024, "intermediate_size": 2816, "num_heads": 16, "num_kv_heads": 8},
    "e2-noqkn": {"num_layers": 32, "hidden_size": 1024, "intermediate_size": 2816, "num_heads": 16, "num_kv_heads": 8},
    "e2-fullrope": {"num_layers": 32, "hidden_size": 1024, "intermediate_size": 2816, "num_heads": 16, "num_kv_heads": 8},
}


def non_emb_params(p: dict) -> float:
    h, L, ffn = p["hidden_size"], p["num_layers"], p["intermediate_size"]
    heads, kv_heads = p["num_heads"], p["num_kv_heads"]
    head_dim, kv_width = h // heads, kv_heads * (h // heads)
    full = sum((index + 1) % int(p.get("attention_every", BASE_PRESET["attention_every"])) == 0
               for index in range(L))
    gate = p.get("attention_output_gate", BASE_PRESET["attention_output_gate"])
    gated = L - full if gate else 0
    return (L * (h * (h + 2 * kv_width) + h * h) + gated * h * h
            + L * 3 * h * ffn + L * h + 2 * L * head_dim + h)


def arm_yaml(run_name: str, preset: str, budget_tokens: int, seq: int = 4096,
             lr: float = 2.0e-4, extra_model: dict | None = None,
             seed: int = 20261001, micro_batch_size: int = 4,
             grad_accum: int = 16,
             activation_checkpointing: bool = False,
             save_every: int = 1_000) -> str:
    tps = seq * micro_batch_size * grad_accum  # single-GPU formal experiment
    iters = budget_tokens // tps
    # Keep warmup near 1% even for short pilot arms. A 100-step floor would
    # consume 20-40% of the 1B/2B-token arms at the larger batch candidates.
    warmup = max(5, min(2_000, max(1, iters // 100)))
    eval_every = max(20, min(500, max(1, iters // 4)))
    em = ""
    if extra_model:
        em = "\n".join(f"    {k}: {v}" for k, v in extra_model.items())
    return f"""# 自动生成（gen_e1e2_arms.py）——{run_name}
run_name: {run_name}
backend: torchtitan-v0.3.0
seed: {seed}

model:
  preset: {preset}
  configs: ${{YUFORMER_MODEL_CONFIGS}}{chr(10) + em + chr(10) if em else ''}

out_dir: ${{YUFORMER_PRETRAIN_OUT}}

data:
  require_release: true
  require_eod: true
  release_manifest: ${{YUFORMER_DATA_RELEASE}}
  tokenizer_gate: ${{YUFORMER_TOKENIZER_GATE}}
  token_stream_audit: ${{YUFORMER_TOKEN_STREAM_AUDIT}}
  stages:
    - ${{YUFORMER_TOKEN_STREAMS}}/${{YUFORMER_ARM_DATA}}.tokens.bin
  meta:
    - ${{YUFORMER_TOKEN_STREAMS}}/${{YUFORMER_ARM_DATA}}.meta.json
  val_dir: ${{YUFORMER_TOKEN_STREAMS}}
  seq_len: {seq}
  epochs: 1
  val_samples_per_domain: 64
  val_max_tokens_per_domain: 20_000_000

optim:
  optimizer: adamw
  lr: {lr}
  weight_decay: 0.1
  warmup_steps: {warmup}
  lr_schedule: cosine
  min_lr_ratio: 0.1
  adam_betas: [0.9, 0.95]
  muon_momentum: 0.95
  muon_nesterov: true
  muon_ns_steps: 5
  clip_grad: 1.0

train:
  budget_tokens: {budget_tokens}
  micro_batch_size: {micro_batch_size}
  grad_accum: {grad_accum}
  expected_iters: {iters}
  activation_checkpointing: {str(activation_checkpointing).lower()}
  cuda_graphs: false
  bf16: true
  eval_every: {eval_every}
  log_every: {min(10, max(1, iters // 100))}
  save_every: {max(20, min(save_every, max(1, iters // 4)))}
  keep_last_n: 2
  keep_every: 50000

parallel:
  world_size: 1
  sharding: full_shard
"""


def env_file(cfg_rel: str, port: int) -> str:
    repo = REPO
    return f"""YUFORMER_PYTHON=/opt/yuformer-torchtitan/bin/python
YUFORMER_TORCHTITAN_ROOT=/opt/torchtitan-v0.3.0
YUFORMER_CONFIG={cfg_rel}
GPUS_PER_NODE=1
MASTER_PORT={port}
YUFORMER_MODEL_CONFIGS={repo}/model/configs/model_configs.yaml
YUFORMER_PRETRAIN_OUT=__OUT__
YUFORMER_TOKEN_STREAMS={repo}/data/token_streams/m3_20261020_yf64k
YUFORMER_ARM_DATA=mix_m3_e1_100b_stage2
YUFORMER_DATA_RELEASE={repo}/data/processed/releases/mix_m3_e1_100b-20261019.release.json
YUFORMER_TOKENIZER_GATE={repo}/data/TOKENIZER_READY.json
YUFORMER_TOKEN_STREAM_AUDIT={repo}/data/processed/audits/formal_token_stream_release_mix_m3_e1_100b_m3_20261020.json
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--groups', nargs='+', choices=('e1_pilot', 'batch_pilot', 'e1', 'e2'),
                        default=['e1_pilot', 'batch_pilot', 'e1', 'e2'])
    parser.add_argument('--skip-presets', action='store_true')
    args = parser.parse_args()
    selected_groups = set(args.groups)
    policy = yaml.safe_load((PRETRAIN_CFG_DIR / 'research_budget.yaml').read_text())
    divisor = int(policy['e1_budget_divisor'])
    fixed_budget = int(policy['screen_budget_tokens'])
    if divisor != 10 or fixed_budget != 1_000_000_000:
        raise ValueError('unsupported research budget contract')
    doc = yaml.safe_load(MODEL_CFG.read_text())
    presets = doc["presets"]
    added = 0
    for group, spec in (("e1", E1_PRESETS), ("e2", E2_PRESETS)):
        for name, p in spec.items():
            full = dict(BASE_PRESET)
            full.update(p)
            if name == "e2-nogate":
                full["attention_output_gate"] = False
            if name == "e2-noqkn":
                full["qk_norm"] = False
            if name == "e2-fullrope":
                full["rope_percent"] = 1.0
            full["notes"] = f"E1/E2 实验臂（~{non_emb_params(full)/1e6:.0f}M non-emb）"
            if presets.get(name) != full:
                presets[name] = full
                added += 1
    # Retired typo-bearing preset; generated formal configs use e2-l48-h800.
    presets.pop("e2-l48-h848", None)
    if not args.skip_presets:
        MODEL_CFG.write_text(yaml.safe_dump(doc, sort_keys=True, allow_unicode=True))

    arms: list[tuple[str, str, str, int, dict | None, float, int]] = []
    pilot_base_lr = {
        "50m": ("e1-50m", 6.0e-4),
        "200m": ("e1-200m", 3.0e-4),
        "800m": ("e1-800m", 1.5e-4),
    }
    for scale, (preset, center_lr) in pilot_base_lr.items():
        for label, multiplier in (
            ("vlow", 0.25), ("low", 0.5), ("mid", 1.0),
            ("high", 2.0), ("vhigh", 4.0),
        ):
            arms.append((
                f"e1_lr_{scale}_{label}", preset, "e1_pilot", 500_000_000,
                None, center_lr * multiplier, 16,
            ))
    # Batch/LR joint pilot. The initial centers are regenerated from the
    # selected 200M LR before launch; these values keep the static contracts
    # valid and encode the same square-root batch scaling rule.
    base_tpu = 4096 * 4 * 16
    largest_batch_tpu = 4096 * 4 * 256
    matched_batch_budget = 2_000_000_000 // largest_batch_tpu * largest_batch_tpu
    for label, grad_accum in (("1m", 64), ("2m", 128), ("4m", 256)):
        tokens_per_step = 4096 * 4 * grad_accum
        center = 3.0e-4 * (tokens_per_step / base_tpu) ** 0.5
        for lr_level, multiplier in (("low", 0.5), ("mid", 1.0), ("high", 2.0)):
            arms.append((f"batch_{label}_{lr_level}", "e1-200m", "batch_pilot",
                         matched_batch_budget, None, center * multiplier, grad_accum))
    fixed_data_budget = fixed_budget
    arms.append(("e1_ln_50m", "e1-50m", "e1", fixed_data_budget, None, 2.0e-4, 16))
    arms.append(("e1_ln_100m", "e1-100m", "e1", fixed_data_budget, None, 2.0e-4, 16))
    arms.append(("e1_ln_200m", "e1-200m", "e1", fixed_data_budget, None, 2.0e-4, 16))
    arms.append(("e1_ln_400m", "pretrain-0.5b", "e1", fixed_data_budget, None, 2.0e-4, 16))
    arms.append(("e1_ln_800m", "e1-800m", "e1", fixed_data_budget, None, 2.0e-4, 16))
    # Independent L(D) section.  Keeping N fixed is necessary to identify the
    # data exponent without relying only on correlated isoFLOP points.  The
    # original data axis is reduced tenfold while retaining its dynamic range.
    for label, budget in (
        ("100m", 100_000_000), ("300m", 300_000_000), ("1b", 1_000_000_000),
        ("3b", 3_000_000_000), ("9p85b", 9_846_128_640),
    ):
        arms.append((f"e1_ld_300m_{label}", "e1-300m", "e1", budget, None, 2.0e-4, 16))
    p_by_scale = {"50m": "e1-50m", "100m": "e1-100m", "200m": "e1-200m", "400m": "pretrain-0.5b", "800m": "e1-800m"}
    compute_contracts = {}
    for ci, C in enumerate(policy['iso_flops_targets']):
        for scale in p_by_scale:
            per_token = flops_per_token(presets[p_by_scale[scale]], 4096)
            D = int(C / per_token)
            if D < 500_000_000 // divisor or D > 100_000_000_000 // divisor:
                continue
            arms.append((f"e1_iso_c{ci}_{scale}", p_by_scale[scale], "e1", D, None, 2.0e-4, 16))
            compute_contracts[f'e1_iso_c{ci}_{scale}'] = {
                'target_algorithmic_flops': int(C), 'algorithmic_flops_per_token': per_token,
                'includes_vocabulary_projection_and_attention': True,
                'includes_activation_recomputation': False,
            }
    e2_architectures = (
        ("e2_l16", "e2-l16-h1440"), ("e2_l32", "pretrain-0.5b"),
        ("e2_l48", "e2-l48-h800"), ("e2_ffn2x", "e2-ffn2x"),
        ("e2_ffn4x", "e2-ffn4x"),
        ("e2_nogate", "e2-nogate"), ("e2_noqkn", "e2-noqkn"),
        ("e2_fullrope", "e2-fullrope"), ("e2_swaoff", "e2-swaoff"),
        ("e2_untied", "pretrain-0.5b-untied"),
    )
    for name, preset in e2_architectures:
        for seed_index in range(3):
            run_name = name if seed_index == 0 else f"{name}_s{seed_index + 1}"
            arms.append((run_name, preset, "e2", fixed_budget,
                         {"_seed": 20261001 + seed_index}, 2.0e-4, 16))

    # This directory is generator-owned. Remove retired contracts (not run
    # evidence) so the formal roster cannot silently retain an obsolete arm.
    for group in ('e1', 'e2'):
        directory = PRETRAIN_CFG_DIR / group
        wanted = {name for name, _, g, *_ in arms if g == group}
        if group in selected_groups and directory.is_dir():
            for path in (*directory.glob('*.yaml'), *directory.glob('*.env')):
                if path.stem not in wanted:
                    path.unlink()

    port = 29600
    counts = {"e1_pilot": 0, "batch_pilot": 0, "e1": 0, "e2": 0}
    for run_name, preset, group, budget, extra, lr, grad_accum in arms:
        if group not in selected_groups:
            continue
        d = PRETRAIN_CFG_DIR / group
        d.mkdir(parents=True, exist_ok=True)
        seed = int((extra or {}).get("_seed", 20261001))
        model_extra = {key: value for key, value in (extra or {}).items() if key != "_seed"}
        # The 800M dense arm exceeds the residual-memory envelope of a shared
        # H800 at micro-batch 4.  Preserve the exact 262,144 tokens/update
        # contract with micro-batch 1 + 4x accumulation and trade recompute for
        # a measured peak of ~17.3 GiB via full activation checkpointing.
        large_arm = preset == "e1-800m"
        micro_batch_size = 1 if large_arm else 4
        effective_grad_accum = grad_accum * 4 if large_arm else grad_accum
        # Pilot runs are short enough that the former 10,000-step interval
        # produced no recoverable mid-run checkpoint at all.  A 250-step
        # interval bounds lost pilot work while keeping DCP overhead modest.
        save_every = 250 if group in {"e1_pilot", "batch_pilot"} else 1_000
        (d / f"{run_name}.yaml").write_text(
            arm_yaml(run_name, preset, budget, lr=lr,
                     extra_model=model_extra or None, seed=seed,
                     micro_batch_size=micro_batch_size,
                     grad_accum=effective_grad_accum,
                     activation_checkpointing=large_arm,
                     save_every=save_every)
        )
        if run_name in compute_contracts:
            path = d / f'{run_name}.yaml'
            config = yaml.safe_load(path.read_text())
            config['research_compute'] = compute_contracts[run_name]
            path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        env = env_file(f"configs/{group}/{run_name}.yaml", port).replace(
            "__OUT__", str(REPO / "data" / "experiments" / group / run_name)
        )
        (d / f"{run_name}.env").write_text(env)
        counts[group] += 1
        port += 1
    print(
        f"presets added: {added}; arms: lr_pilot={counts['e1_pilot']} "
        f"batch_pilot={counts['batch_pilot']} "
        f"e1={counts['e1']} e2={counts['e2']} (total {sum(counts.values())})"
    )


if __name__ == "__main__":
    sys.exit(main())
