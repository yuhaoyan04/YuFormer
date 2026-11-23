#!/usr/bin/env python3
"""Plot empirical eight-GPU throughput forecasts with measurement provenance."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter

from analyze_model_size_sweep import analyze, ROOT


def main():
    report = analyze()
    rows = [dict(row, horizon=24) for row in report['measurements']
            if row['p10_tps_per_device'] and row['projected_1t_days_duty_09']]
    for name in ('model-selected-sustained-canary.json', 'model-budget-sustained-canary.json',
                 'model-budget-lr-controls.json', 'model-power-width-frontier.json'):
        path = ROOT/'data/health'/name
        if not path.exists():
            continue
        state = json.loads(path.read_bytes())
        for key, case in state.get('cases', {}).items():
            value = case['evaluation']
            if value['status'] != 'PASS_SCREENING':
                continue
            rows.append({'variant': key, 'parameters': case['parameters'],
                'p10_tps_per_device': value['p10_tps_per_device'],
                'projected_1t_days_duty_09': value['projected_1t_days_duty_09'],
                'requires_smaller_batch_comparison': case.get('requires_smaller_batch_comparison', False),
                'horizon': 24 if name == 'model-power-width-frontier.json' else 200,
                'result_sha256': case['result_sha256'], 'finished_at': case['finished_at'],
                'production_qualified': False})
    folder = ROOT/'report/figures'
    folder.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.1), gridspec_kw={'width_ratios': [1.35, 1]})
    for axis in axes:
        axis.axhspan(30, 33, color='#f6d89a', alpha=.35, zorder=0)
        axis.axhline(30, color='#25764a', linestyle='--', linewidth=1.2, label='30-day target')
        axis.axhline(33, color='#b55a18', linestyle=':', linewidth=1.5, label='33-day launch ceiling')
        for label, color, marker, size, predicate in (
            ('24-step screening', '#71869a', 'o', 27, lambda row: row['horizon'] == 24 and not row['requires_smaller_batch_comparison']),
            ('200-step sustained', '#155a8a', 'D', 47, lambda row: row['horizon'] == 200),
            ('Allocator pressure: excluded', '#bd3333', 'x', 65, lambda row: row['requires_smaller_batch_comparison'])):
            selected = [row for row in rows if predicate(row)]
            axis.scatter([r['parameters']/1e9 for r in selected],
                         [r['projected_1t_days_duty_09'] for r in selected],
                         color=color, marker=marker, s=size, label=label, alpha=.85, zorder=3)
        axis.set_xlabel('Total parameters (billions)')
        axis.grid(alpha=.18)
    axes[0].set_yscale('log')
    axes[0].set_ylim(19, 190)
    axes[0].set_xlim(.4, 5.2)
    axes[0].set_yticks([20, 30, 45, 60, 90, 120, 180])
    axes[0].yaxis.set_major_formatter(ScalarFormatter())
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_ylabel('Projected 1T-token wall time (days)')
    axes[0].set_title('Full measured range', fontsize=11)
    axes[1].set_xlim(.95, 1.45)
    axes[1].set_ylim(27.5, 38)
    axes[1].set_title('Near-budget architecture comparisons', fontsize=11)
    labels = [('baseline_053b_b8', '0.53B baseline', (9, -12)),
              ('size_15b_b8', '1.52B', (8, 5)), ('size_2b_b4', '1.97B', (8, 5)),
              ('size_3b_b4', '2.98B', (8, 5)), ('size_4b_lowmem_control_b2', '4.01B / micro2', (-15, -17)),
              ('size_5b_b2', '4.96B', (-40, -12))]
    for key, label, offset in labels:
        row = next((r for r in rows if r['variant'] == key), None)
        if row:
            axes[0].annotate(label, (row['parameters']/1e9, row['projected_1t_days_duty_09']),
                             xytext=offset, textcoords='offset points', fontsize=8)
    sustained = next((r for r in rows if r['variant'] == 'budget_sustained_b8'), None)
    if sustained:
        axes[1].annotate(f"1.20B sustained: {sustained['projected_1t_days_duty_09']:.2f}d\nquality/LR review pending",
            (sustained['parameters']/1e9, sustained['projected_1t_days_duty_09']),
            xytext=(10, 32), textcoords='offset points', fontsize=8,
            arrowprops={'arrowstyle': '-', 'color': '#155a8a'})
    axes[1].text(.98, 28.1, 'Same parameter count does not imply\nthe same depth, kernels, or throughput.', fontsize=8, color='#384a58')
    figure.suptitle('Eight H800 | BF16 AdamW | 4K | 1T tokens | p10 throughput | 90% planned duty', fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=8)
    figure.subplots_adjust(top=.86, bottom=.21, wspace=.25)
    for suffix in ('svg', 'png'):
        figure.savefig(folder/f'model_size_throughput_20261114.{suffix}', dpi=160)
    plt.close(figure)
    data = {'purpose': 'EMPIRICAL_MODEL_SCALE_COMPARISON_NOT_PRODUCTION_QUALIFICATION',
            'rows': rows, 'target_days': 30, 'hard_ceiling_days': 33,
            'plot_tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (folder/'model_size_throughput_20261114.data.json').write_text(json.dumps(data, sort_keys=True, indent=2)+'\n')
    print(json.dumps({'points': len(rows), 'figure': str(folder/'model_size_throughput_20261114.png')}))


if __name__ == '__main__':
    main()
