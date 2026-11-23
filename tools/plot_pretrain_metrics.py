#!/usr/bin/env python3
"""Render exported real measurements to a standalone, CPU-only SVG."""
import csv
import argparse
from pathlib import Path


def main(preview_png=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(__file__).resolve().parents[1]
    with (root/'report/metrics/pretrain_validation.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    domains = ['academic', 'books', 'chinese', 'code', 'encyclopedia',
               'math', 'multilingual', 'qa', 'web']
    x = [float(row['training_labels'])/1e9 for row in rows]
    plt.rcParams.update({'svg.fonttype': 'none', 'svg.hashsalt': 'YuFormer', 'font.size': 10})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3), constrained_layout=True)
    axes[0].plot(x, [float(row['aggregate']) for row in rows], marker='.', color='#176b9d')
    axes[0].set_title('Equal-domain validation loss')
    axes[0].set_ylabel('Cross entropy')
    normalized = [row for row in rows if int(row['step']) >= 2000]
    if normalized:
        for domain in domains:
            axes[1].plot([float(row['training_labels'])/1e9 for row in normalized],
                [float(row[domain])/float(normalized[0][domain]) for row in normalized], label=domain)
    axes[1].set_title('Each domain / its step-2000 loss')
    axes[1].set_ylabel('Relative loss')
    axes[1].legend(fontsize=7, ncol=2)
    recent = rows[-7:]
    if recent:
        for domain in domains:
            axes[2].plot([float(row['training_labels'])/1e9 for row in recent],
                [float(row[domain])/float(recent[0][domain]) for row in recent], marker='.')
    axes[2].set_title('Last 7 points / first shown loss')
    axes[2].set_ylabel('Relative domain loss (local scale)')
    for ax in axes:
        ax.set_xlabel('Consumed training labels (B)')
        ax.grid(alpha=.2)
    fig.suptitle('Fixed held-out validation; 64 samples/domain; point estimates, no confidence intervals', fontsize=11)
    output = root/'report/figures/pretrain_validation.svg'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, metadata={'Date': None})
    if preview_png is not None:
        fig.savefig(preview_png, dpi=120)
    plt.close(fig)
    print(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview-png', type=Path)
    args = parser.parse_args()
    main(args.preview_png)
