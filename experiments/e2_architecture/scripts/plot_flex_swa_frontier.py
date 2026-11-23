#!/usr/bin/env python3
"""Export paired completed eight-GPU whole-model SWA backend measurements."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def main():
    paths = [ROOT/'data/health/model-size-sweep.json', ROOT/'data/health/model-flex-frontier.json']
    old, new = [json.loads(p.read_bytes()) for p in paths]
    previous = {r['parameters']: r for r in old['cases'].values()
                if r['evaluation']['status'] == 'PASS_SCREENING'}
    pairs = []
    for row in new['cases'].values():
        if (row['evaluation']['status'] != 'PASS_SCREENING' or row.get('requires_smaller_batch_comparison')
                or row['parameters'] not in previous):
            continue
        before = previous[row['parameters']]
        if before['variant']['microbatch'] != row['variant']['microbatch']:
            continue
        pairs.append({'parameters': row['parameters'],
            'old_variant': before['variant']['name'], 'new_variant': row['variant']['name'],
            'old_tps': before['evaluation']['p10_tps_per_device'],
            'new_tps': row['evaluation']['p10_tps_per_device'],
            'old_days': before['evaluation']['projected_1t_days_duty_09'],
            'new_days': row['evaluation']['projected_1t_days_duty_09']})
    pairs.sort(key=lambda r: r['parameters'])
    if not pairs:
        raise ValueError('no completed same-size/batch comparison pairs')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout='constrained')
    x, width = np.arange(len(pairs)), .36
    labels = [f"{r['parameters']/1e9:.3f}B" for r in pairs]
    colors = ('#8296a5', '#187f72')
    for ax, fields, ylabel in ((axes[0], ('old_tps', 'new_tps'), 'p10 tokens / second / GPU'),
                               (axes[1], ('old_days', 'new_days'), 'Projected wall time (days)')):
        for index, field in enumerate(fields):
            values = [r[field] for r in pairs]
            bars = ax.bar(x+(index-.5)*width, values, width=width, color=colors[index],
                label=('CuDNN SWA', 'Triton block-sparse SWA')[index])
            ax.bar_label(bars, labels=[f'{v/1000:.1f}k' if field.endswith('tps') else f'{v:.1f}' for v in values],
                         padding=3, fontsize=9)
        ax.set_xticks(x, labels)
        ax.set_xlabel('Actual parameter count; same model shape per pair')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.margins(y=.2)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].axhline(30, color='#b78c13', ls='--', lw=1, label='30-day target')
    axes[1].axhline(33, color='#b14235', ls=':', lw=1, label='33-day launch ceiling')
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle('Whole-model throughput after structured sliding-window attention', fontsize=13)
    fig.supxlabel('8 x H800 | HSDP 2 x 4 | BF16 | 4K | 4,194,304 tokens/update | 24-update screening\n'
                  'Global attention unchanged; 1T-token forecast includes 10% wall-time reserve. Sustained quality/recovery gates pending.', fontsize=8)
    output = ROOT/'report/figures/flex_swa_frontier_20261114'
    output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg'):
        fig.savefig(output.with_suffix('.'+extension), dpi=170)
    payload = {'purpose': 'COMPLETED_WHOLE_MODEL_RESEARCH_SCREENING_NOT_PRODUCTION_QUALIFICATION',
        'sources_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'pairs': pairs, 'production_qualified': False}
    output.with_suffix('.data.json').write_text(json.dumps(payload, sort_keys=True, indent=2)+'\n')
    print(json.dumps(payload, sort_keys=True))


if __name__ == '__main__':
    main()
