"""E0 图表：图 E0-1 分域压缩率分组柱状图（含 baseline）；图 E0-2 词表大小三联图。

数据源：results/e0_full_metrics.json（eval_e0_full.py 产物）。风格统一：无网格线、
值标签、PNG 200dpi。输出 results/figures/E0-{1,2}.png。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIGURES = RESULTS / "figures"
DOMAINS = ["web", "code", "math", "academic", "books", "encyclopedia", "chinese", "qa"]
OURS = ["yf-32k-bpe", "yf-64k-bpe", "yf-128k-bpe"]
BASELINES = ["gpt2-50k", "qwen2.5-151k"]
ORDER = OURS + BASELINES
COLORS = {"yf-32k-bpe": "#8ecae6", "yf-64k-bpe": "#219ebc", "yf-128k-bpe": "#023047",
          "gpt2-50k": "#ffb703", "qwen2.5-151k": "#fb8500"}


def load():
    d = json.loads((RESULTS / "e0_full_metrics.json").read_text())
    return {t["tokenizer"]: t for t in d["tokenizers"]}


def fig_e0_1(toks):
    n_groups = len(DOMAINS)
    width = 0.16
    fig, ax = plt.subplots(figsize=(11, 4.5))
    xs = range(n_groups)
    for i, name in enumerate(ORDER):
        vals = [toks[name]["per_domain"].get(dm, {}).get("bytes_per_token") for dm in DOMAINS]
        pos = [x + (i - 2) * width for x in xs]
        bars = ax.bar(pos, [v or 0 for v in vals], width, label=f"{name} ({toks[name]['vocab_size']:,})",
                      color=COLORS[name])
        for p, v in zip(bars, vals):
            if v:
                ax.text(p.get_x() + p.get_width() / 2, v + 0.05, f"{v:.1f}", ha="center", fontsize=6.5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(DOMAINS)
    ax.set_ylabel("bytes / token（越高=压缩率越好）")
    ax.set_title("图 E0-1 分域压缩率：YuFormer BBPE 三档 vs 工业基线（tokenizer_corpus 17 源抽样）")
    ax.legend(fontsize=8, ncol=5)
    ax.set_ylim(0, 5.6)
    fig.tight_layout()
    out = FIGURES / "E0-1_domain_compression.png"
    fig.savefig(out, dpi=200)
    print(out)


def fig_e0_2(toks):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    vocabs = [toks[n]["vocab_size"] / 1024 for n in OURS]
    labels = [f"{int(v)}k" for v in vocabs]
    overall = [toks[n]["overall_bytes_per_token"] for n in OURS]
    qwen = toks["qwen2.5-151k"]["overall_bytes_per_token"]

    ax = axes[0]
    ax.plot(vocabs, overall, "o-", color="#219ebc")
    ax.axhline(qwen, ls="--", color="#fb8500", lw=1, label=f"Qwen2.5-151k = {qwen:.2f}")
    for x, y in zip(vocabs, overall):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("词表大小（k）")
    ax.set_ylabel("bytes/token")
    ax.set_title("(a) 压缩率 vs 词表")
    ax.legend(fontsize=8)

    ax = axes[1]
    hidden = 1024
    emb_params = [toks[n]["vocab_size"] * hidden / 1e6 for n in OURS]
    total_params = 473
    ratios = [p / total_params * 100 for p in emb_params]
    bars = ax.bar(labels, ratios, color=["#8ecae6", "#219ebc", "#023047"])
    for b, r in zip(bars, ratios):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.3, f"{r:.1f}%", ha="center", fontsize=9)
    ax.set_ylabel("embedding 占总参数比（tied，/473M）")
    ax.set_title("(b) embedding 参数占比 vs 词表")

    ax = axes[2]
    fw_rate = [toks[n]["per_source"]["fineweb"]["throughput_mb_s"] for n in OURS]
    gh_rate = [toks[n]["per_source"]["github-code"]["throughput_mb_s"] for n in OURS]
    x = range(len(OURS))
    ax.bar([i - 0.18 for i in x], fw_rate, 0.36, label="fineweb", color="#219ebc")
    ax.bar([i + 0.18 for i in x], gh_rate, 0.36, label="github-code", color="#ffb703")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("编码吞吐（MB/s，单线程）")
    ax.set_title("(c) 编码吞吐 vs 词表")
    ax.legend(fontsize=8)

    fig.suptitle("图 E0-2 词表大小消融三联图：压缩率 / 参数经济学 / 吞吐（0.5B 档 64k 为甜点）", y=1.02)
    fig.tight_layout()
    out = FIGURES / "E0-2_vocab_ablation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(out)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    toks = load()
    missing = [n for n in ORDER if n not in toks]
    if missing:
        raise SystemExit(f"missing tokenizers in results: {missing}")
    fig_e0_1(toks)
    fig_e0_2(toks)


if __name__ == "__main__":
    main()
