#!/usr/bin/env python3
"""Export completed matched sustained pilots; never plot partial runs as final."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DOMAINS = ('web', 'code', 'math', 'chinese', 'academic', 'books', 'encyclopedia', 'qa', 'multilingual')


def main():
    runs = ROOT/'data/experiments/flex_sustained_20261114'
    cases, sources = {}, {}
    for name in ('matched_baseline_b8', 'larger_lr1e4_b8', 'larger_lr2e4_b8'):
        path = runs/name/'result.json'
        if not path.is_file():
            continue
        result = json.loads(path.read_bytes())
        if result['status'] != 'PASS' or result['hyperparameters']['trained_tokens'] != 838860800:
            raise ValueError('quality figures require a complete successful 200-update pilot')
        cases[name] = result
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    baseline = cases['matched_baseline_b8']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    labels = {'matched_baseline_b8': '0.530B, peak LR 2e-4',
              'larger_lr1e4_b8': '1.517B, peak LR 1e-4', 'larger_lr2e4_b8': '1.517B, peak LR 2e-4'}
    colors = {'matched_baseline_b8': '#8296a5', 'larger_lr1e4_b8': '#187f72', 'larger_lr2e4_b8': '#c77729'}
    reductions = {}
    for name, result in cases.items():
        points = result['series']['validation_metrics/loss']
        axes[0].plot([p['step'] for p in points], [p['value'] for p in points],
            marker='o', color=colors[name], label=labels[name])
        final = points[-1]['value']
        axes[0].annotate(f'{final:.3f}', (points[-1]['step'], final),
            xytext=(5, -12 if name == 'larger_lr1e4_b8' else 5), textcoords='offset points',
            color=colors[name], fontsize=9)
        if name != 'matched_baseline_b8':
            reductions[name] = []
            for domain in DOMAINS:
                tag = f'validation_metrics/domain/{domain}/loss'
                before, after = baseline['metrics'][tag]['last'], result['metrics'][tag]['last']
                reductions[name].append(100*(before-after)/before)
    axes[0].set(xlabel='Recorded optimizer step (pilot horizon: 200)', ylabel='Aggregate validation loss',
                xlim=(0, 220), title='Learning progress at equal trained-token budgets')
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].margins(y=.1)
    x, width = np.arange(len(DOMAINS)), .7/max(len(reductions), 1)
    for i, (name, values) in enumerate(reductions.items()):
        offset = (i-(len(reductions)-1)/2)*width
        axes[1].bar(x+offset, values, width=width, color=colors[name], label=labels[name])
    axes[1].axhline(0, color='#536473', lw=.8)
    axes[1].set_xticks(x, DOMAINS, rotation=40, ha='right')
    axes[1].set(ylabel='Validation loss reduction vs 0.530B (%)',
                title='Final pilot validation across all nine domains')
    axes[1].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Hybrid SWA: completed sustained model/LR controls', fontsize=13)
    fig.supxlabel('Same tokenizer, data, seed, 8 GPUs and 838,860,800 trained tokens per run.\n'
                  'These are early optimization pilots; full 1T quality and production qualification are separate.', fontsize=8)
    output = ROOT/'report/figures/flex_sustained_quality_20261114'
    for ext in ('png', 'svg'):
        fig.savefig(output.with_suffix('.'+ext), dpi=170)
    output.with_suffix('.data.json').write_text(json.dumps({'source_sha256': sources,
        'completed_cases': list(cases), 'domain_loss_reduction_percent': reductions,
        'production_qualified': False}, sort_keys=True, indent=2)+'\n')
    print(json.dumps({'completed_cases': list(cases), 'figure': str(output.with_suffix('.png'))}))


if __name__ == '__main__':
    main()
