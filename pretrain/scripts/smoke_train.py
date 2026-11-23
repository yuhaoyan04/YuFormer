#!/usr/bin/env python3
"""端到端冒烟（AGENTS.md §5）：tiny 预设（4 层）+ 合成 1MB token 流 + 1 GPU 100 步。

验证项：
  1. loss 下降（fp32 100 步，确定性数据）
  2. 断点续训一致：A=新训 60 步 → B=同目录续训到 100 步；C=新训 100 步；
     断言 B 的 61..100 步 loss 与 C 逐步一致（fp32，atol=1e-5）
  3. marker 协议：RESUME_ITER 设错值时续训必须拒绝（exit 4）
  4. Muon 冒烟：随机 tiny 模型 30 步 muon+bf16，loss 有限且末段均值 < 首段均值
  5. checkpoint 内容完整：model/optim/cursor/rng + latest.json

用法：python scripts/smoke_train.py [--work-dir /tmp/yuformer_smoke] [--gpu 0]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PRETRAIN_SRC = REPO_ROOT / "pretrain" / "src"
MODEL_SRC = REPO_ROOT / "model" / "src"
MODEL_CONFIGS = REPO_ROOT / "model" / "configs" / "model_configs.yaml"
sys.path.insert(0, str(PRETRAIN_SRC))
sys.path.insert(0, str(MODEL_SRC))

SEQ_LEN = 128
MICRO_BS = 4
GRAD_ACCUM = 2
WORLD = 1
TOKENS_PER_STEP = SEQ_LEN * MICRO_BS * GRAD_ACCUM * WORLD  # 1024
STEPS = 100
SEED = 1234


def make_token_stream(path: Path, name: str, n_tokens: int, seed: int, kind: str = "chain") -> None:
    from yuformer_pretrain.data import _edge_sha256

    rng = np.random.default_rng(seed)
    if kind == "chain":
        # 可学习合成数据：90% 确定性链 a'=(7a+13)%512，10% 随机 token；
        # 期望 loss 从 ~ln(32768)≈10.4 降至 ~1 以下（真随机噪声则学不动）
        arr = np.empty(n_tokens, dtype=np.uint16)
        arr[0] = rng.integers(0, 512)
        noise = rng.random(n_tokens) < 0.1
        for i in range(1, n_tokens):
            a = int(arr[i - 1])
            arr[i] = rng.integers(0, 32768) if noise[i] else (7 * (a % 512) + 13) % 512
    else:
        arr = rng.integers(0, 32768, size=n_tokens, dtype=np.uint16)
    bin_path = path / f"{name}.tokens.bin"
    arr.tofile(bin_path)
    import hashlib as hl

    h = hl.sha256()
    with open(bin_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    meta = {
        "name": name,
        "total_tokens": int(n_tokens),
        "bytes": int(bin_path.stat().st_size),
        "sha256": h.hexdigest(),
        "edge_sha256": _edge_sha256(bin_path),
        "tokenizer_sha256": "synthetic-smoke",
        "tokenizer": "synthetic",
        "source": f"synthetic:{name}",
        "n_docs": 0,
    }
    (path / f"{name}.meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


def write_config(path: Path, work_dir: Path, data_dir: Path, run_name: str,
                 optimizer: str, bf16: bool) -> Path:
    cfg = {
        "run_name": run_name,
        "seed": SEED,
        "model": {"preset": "tiny", "configs": str(MODEL_CONFIGS)},
        "out_dir": str(work_dir / run_name),
        "data": {
            "stages": [str(data_dir / "train.tokens.bin")],
            "meta": [str(data_dir / "train.meta.json")],
            "val_dir": str(data_dir / "val"),
            "seq_len": SEQ_LEN,
            "epochs": 10,
            "val_samples_per_domain": 8,
            "val_max_tokens_per_domain": 1_000_000,
        },
        "optim": {
            "optimizer": optimizer,
            "lr": 2e-4,
            "weight_decay": 0.1,
            "warmup_steps": 10,
            "lr_schedule": "constant",
            "min_lr_ratio": 0.1,
            "adam_betas": [0.9, 0.95],
            "muon_momentum": 0.95,
            "muon_nesterov": True,
            "muon_ns_steps": 5,
            "clip_grad": 1.0,
        },
        "train": {
            "budget_tokens": TOKENS_PER_STEP * STEPS,
            "micro_batch_size": MICRO_BS,
            "grad_accum": GRAD_ACCUM,
            "expected_iters": STEPS,
            "activation_checkpointing": False,
            "bf16": bf16,
            "eval_every": 25,
            "log_every": 5,
            "save_every": 30,
        },
        "parallel": {"world_size": WORLD, "sharding": "no_shard"},
    }
    p = path / f"{run_name}.yaml"
    import yaml

    p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return p


def run_trainer(cfg_path: Path, gpu: int, extra: list[str] | None = None,
                env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONPATH": f"{PRETRAIN_SRC}:{MODEL_SRC}:{env.get('PYTHONPATH', '')}",
        "YUFORMER_TRACKER_ROOT": str(REPO_ROOT / "experiments" / "_common"),
    })
    if env_extra:
        env.update(env_extra)
    cmd = [sys.executable, "-m", "yuformer_pretrain.trainer", "--config", str(cfg_path)]
    if extra:
        cmd += extra
    return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)


def read_losses(run_dir: Path, run_name: str) -> dict[int, float]:
    metrics = run_dir / run_name / "results" / run_name / "metrics.jsonl"
    out = {}
    for line in metrics.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if "loss" in r:
                out[int(r["step"])] = float(r["loss"])
    return out


def main() -> None:
    global STEPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", default="/tmp/yuformer_smoke")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    work = Path(args.work_dir)
    if work.exists():
        shutil.rmtree(work)
    data_dir = work / "data"
    val_dir = data_dir / "val"
    val_dir.mkdir(parents=True)

    # 1MB 合成训练流（524288 tokens）+ 两个 val 域流
    make_token_stream(data_dir, "train", 1 << 19, seed=SEED)
    make_token_stream(val_dir, "web", 1 << 16, seed=SEED + 1)
    make_token_stream(val_dir, "math", 1 << 16, seed=SEED + 2)

    results: dict[str, object] = {}
    failures: list[str] = []

    # 阶段 A：新训 60 步（save_every=30 → ckpt 30/60）
    cfg_a = write_config(work, work, data_dir, "runA", "adamw", False)
    t0 = time.monotonic()
    pa = run_trainer(cfg_a, args.gpu, extra=["--max-iters", "60"])
    if pa.returncode != 0:
        print(pa.stdout[-3000:], pa.stderr[-3000:])
        raise SystemExit("runA failed")
    # 阶段 B：同目录续训 60→100
    pb = run_trainer(cfg_a, args.gpu)
    if pb.returncode != 0:
        print(pb.stdout[-3000:], pb.stderr[-3000:])
        raise SystemExit("runB (resume) failed")
    # 阶段 C：新训 100 步（对照）
    cfg_c = write_config(work, work, data_dir, "runC", "adamw", False)
    pc = run_trainer(cfg_c, args.gpu, extra=["--no-resume"])
    if pc.returncode != 0:
        print(pc.stdout[-3000:], pc.stderr[-3000:])
        raise SystemExit("runC failed")
    t_train = time.monotonic() - t0

    losses_b = read_losses(work, "runA")  # A/B 同 run_name 同目录
    losses_c = read_losses(work, "runC")
    common = sorted(set(losses_b) & set(losses_c) & set(range(61, 101)))
    max_diff = max(abs(losses_b[s] - losses_c[s]) for s in common)
    results["resume_max_loss_diff_steps61_100"] = max_diff
    results["resume_consistent"] = bool(max_diff < 1e-5)
    if not results["resume_consistent"]:
        failures.append(f"resume mismatch: max_diff={max_diff}")
    drop = losses_c[100] - losses_c[1]
    results["loss_first@1"] = losses_c[1]
    results["loss_final@100"] = losses_c[100]
    results["loss_drop"] = drop
    results["loss_decreased"] = bool(drop < -0.05)
    if not results["loss_decreased"]:
        failures.append(f"loss not decreased: {losses_c[5]} -> {losses_c[100]}")

    # marker 协议：RESUME_ITER 错误值必须拒绝（runA 目录已 at iter 100）
    p_bad = run_trainer(cfg_a, args.gpu, env_extra={"RESUME_ITER": "99"})
    results["marker_rejects_wrong_resume_iter"] = p_bad.returncode == 4
    if not results["marker_rejects_wrong_resume_iter"]:
        failures.append(f"marker protocol: returncode={p_bad.returncode}, expect 4")

    # Muon 冒烟：预算 40 步，首段截断 30 步存 ckpt，再续训 30→40（验证 named-OSD 加载）
    STEPS = 40
    cfg_m = write_config(work, work, data_dir, "runM", "muon", True)
    STEPS = 100
    pm = run_trainer(cfg_m, args.gpu, extra=["--max-iters", "30"])
    if pm.returncode != 0:
        print(pm.stdout[-3000:], pm.stderr[-3000:])
        raise SystemExit("runM (muon) failed")
    pm2 = run_trainer(cfg_m, args.gpu)
    if pm2.returncode != 0:
        print(pm2.stdout[-3000:], pm2.stderr[-3000:])
        raise SystemExit("runM resume failed")
    losses_m = read_losses(work, "runM")
    results["muon_resume_ok"] = any(s > 30 for s in losses_m)
    vals_m = [losses_m[s] for s in sorted(losses_m)]
    finite = all(np.isfinite(v) for v in vals_m)
    head = sum(vals_m[:5]) / min(5, len(vals_m))
    tail = sum(vals_m[-5:]) / min(5, len(vals_m))
    results["muon_finite"] = bool(finite)
    results["muon_loss_head"] = head
    results["muon_loss_tail"] = tail
    results["muon_decreased"] = bool(finite and tail < head)
    if not results["muon_decreased"]:
        failures.append(f"muon not decreased: head={head} tail={tail}")

    # checkpoint 完整性 + MFU 读数
    ck = work / "runA" / "checkpoints"
    marker = json.loads((ck / "latest.json").read_text())
    ck_dir = ck / marker["dir"]
    results["checkpoint_files"] = sorted(p.name for p in ck_dir.iterdir())
    for f in ("model.pt", "optim.pt", "cursor.json", "rng.pt"):
        if not (ck_dir / f).is_file():
            failures.append(f"checkpoint missing {f}")
    manifest_m = json.loads((work / "runM" / "results" / "runM" / "manifest.json").read_text())
    results["mfu_muon_bf16"] = manifest_m.get("mfu")
    results["achieved_tflops_muon"] = manifest_m.get("achieved_tflops")
    results["wallclock_fp32_100steps_s"] = round(t_train, 1)

    print(json.dumps(results, indent=2, sort_keys=True))
    if failures:
        print("SMOKE FAIL:\n  " + "\n  ".join(failures), file=sys.stderr)
        raise SystemExit(1)
    print("SMOKE OK")
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
