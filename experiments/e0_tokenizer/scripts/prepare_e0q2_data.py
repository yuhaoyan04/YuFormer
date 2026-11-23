"""E0-Q2 数据准备：mix_order 顺序表的 10B(64k-tokens 口径) 子集 × 4 tokenizer 物化。

固定算力对比口径（实验设计.md E0-Q2）：每臂 10B tokens（=同 FLOPs 同步数），
32k 臂涵盖文本更多、128k 臂更少——这正是"固定算力选词表"的工业决策视角。
选取 = mix_order stage1 前缀（position 序），各 tokenizer 独立截断到 10B。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # E-65：原 off-by-one 指向 experiments/
PRETRAIN = REPO / "pretrain"
DATA = REPO / "data"
OUT = DATA / "processed"
STREAMS = DATA / "token_streams"
TOKS = {
    "yf-32k-bpe": REPO / "tokenizer" / "tokenizers" / "yf-32k-bpe",
    "yf-64k-bpe": REPO / "tokenizer" / "tokenizers" / "yf-64k-bpe",
    "yf-128k-bpe": REPO / "tokenizer" / "tokenizers" / "yf-128k-bpe",
    "yf-64k-unigram": REPO / "tokenizer" / "tokenizers" / "yf-64k-unigram",
}
MIX = "mix_m3_e1_100b_fast"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget-tokens", type=int, default=10_000_000_000)
    args = ap.parse_args(argv)
    order = OUT / "mix_order" / f"{MIX}_stage1.parquet"
    if not order.is_file():
        raise SystemExit(f"mix order missing: {order}（等 S6 物化）")
    STREAMS.mkdir(parents=True, exist_ok=True)
    env = {"PYTHONPATH": str(PRETRAIN / "src"), "PATH": "/usr/bin:/bin", "HOME": "/root", "HF_HUB_OFFLINE": "1"}
    for name, tok_dir in TOKS.items():
        tok_json = tok_dir / "tokenizer.json"
        if not tok_json.is_file():
            spm = tok_dir / "yf_unigram.model"
            if spm.is_file() and name == "yf-64k-unigram":
                pass  # sentencepiece 模型：materialize_tokens 需 HF 格式——转换步骤见 train_unigram_spm 后处理
            print(f"[skip] {name}: tokenizer.json 不存在（未训完/未转换）", file=sys.stderr)
            continue
        meta = STREAMS / f"e0q2_{name}.meta.json"
        if meta.is_file():
            print(f"[skip] {name}: already done")
            continue
        print(f"[run] {name}", flush=True)
        r = subprocess.run(
            [sys.executable, str(PRETRAIN / "scripts" / "materialize_tokens.py"),
             "--order", str(order), "--s3-root", str(OUT / "s3"),
             "--tokenizer", str(tok_json), "--out-dir", str(STREAMS),
             "--name", f"e0q2_{name}", "--max-tokens", str(args.budget_tokens)],
            cwd=PRETRAIN, env=env,
        )
        if r.returncode != 0:
            print(f"[FAIL] {name}", file=sys.stderr)
            return 1
    print("E0Q2_DATA_READY")
    return 0


if __name__ == "__main__":
    sys.exit(main())
