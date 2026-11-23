#!/usr/bin/env python3
"""Export identity-audited E1 endpoints and diagnostic plots without training GPUs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic(value: dict) -> dict:
    return {k: v for k, v in value.items() if k != 'observed_at'}


def atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_bytes(data)
    temporary.replace(path)


def render(analysis_path: Path, audit_path: Path, output: Path) -> dict:
    analysis = json.loads(analysis_path.read_text())
    audit = json.loads(audit_path.read_text())
    seed_level = analysis.get('fit_diagnostics', {}).get('seed_level')
    seed_count = (int(seed_level.get('seed_count_per_design_point', 0))
                  if isinstance(seed_level, dict) else 0)
    seed_aware = seed_count >= 2
    measurement_label = (f'{seed_count}-seed mean final held-out loss' if seed_aware
                         else 'Measured final held-out loss')
    observation_label = (f'{seed_count}-seed means' if seed_aware
                         else 'single-seed screening')
    trigger = hashlib.sha256(json.dumps({
        'analysis': semantic(analysis), 'identity': semantic(audit),
        'renderer_sha256': digest(Path(__file__)),
    }, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if (previous.get('input_sha256') == trigger
                and all((output / n).is_file() and digest(output / n) == sha
                        for n, sha in previous.get('artifacts', {}).items())
                and len(previous.get('artifacts', {})) == 5):
            return previous
    accepted = {r['name'] for r in audit.get('rows', [])
                if r.get('group') == 'e1' and r.get('status') == 'PASS'}
    rows, excluded = [], []
    seen = set()
    for row in analysis.get('runs', []):
        name = row.get('run_name')
        finite = all(isinstance(row.get(k), (int, float))
                     and math.isfinite(row[k]) and row[k] > 0
                     for k in ('n_params', 'tokens', 'validation_loss'))
        if name in seen:
            raise ValueError(f'duplicate E1 endpoint: {name}')
        seen.add(name)
        if row.get('status') == 'PASS' and name in accepted and finite:
            rows.append(row)
        else:
            excluded.append(name)
    expected = int(analysis.get('expected_results', 24))
    residuals = analysis.get('fit_diagnostics', {}).get('residuals', [])
    fit_gate = analysis.get('fit_quality_gate', {}).get('status', 'PENDING')
    fit_available = (len(rows) == expected == 24 and not excluded
                     and analysis.get('status') in {'PASS', 'PASS_PROVISIONAL', 'FAIL'}
                     and len(residuals) == expected)
    ready = (fit_available and analysis['status'] == 'PASS'
             and fit_gate == 'PASS' and audit.get('status') == 'PASS')
    status = 'ANALYSIS_AND_IDENTITY_PASS' if ready else 'EXPLORATORY_PROGRESS'
    frontier = analysis.get('compute_optimal_frontier', [])
    frontier_available = (ready and len(frontier) == 3
                          and {row.get('family') for row in frontier} == {'c0', 'c1', 'c2'}
                          and all(row.get('status') == 'PASS' for row in frontier))
    claim_limit = (
        f'Each point is the arithmetic mean across {seed_count} seeds. Parameter intervals use '
        'joint seed-index resampling across design points to preserve paired-seed covariance. '
        + ('Compute-optimal frontier points are analytic minima of the fitted law under C=6ND. '
           if frontier_available else
           ('The predeclared fit-quality gate failed; frontier is withheld and residuals are diagnostic only. '
            if fit_gate == 'FAIL' else
            'Compute-optimal frontier is withheld because fit, identity, or all-family interval checks are incomplete. ')) +
        'The 1T prediction extrapolates '
        'beyond observed budgets; lines join measured means, including nonmonotonic results.'
        if seed_aware else
        'Single-seed screening; cross-point bootstrap does not measure seed variance. '
        'Lines join all eligible observations, including nonmonotonic results.'
    )

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    figure.suptitle(
        f'E1 scaling | {len(rows)}/{expected} audited endpoints | {status} | fit gate {fit_gate} | {observation_label}',
        fontsize=13)
    for axis, prefix, key, title, xlabel, fit_key in (
        (axes[0, 0], 'e1_ln_', 'n_params', 'Fixed-data model scaling',
         'Non-embedding parameters', 'L_N_fixed_D'),
        (axes[0, 1], 'e1_ld_', 'tokens', 'Fixed-model data scaling',
         'Actually trained tokens', 'L_D_fixed_300M'),
    ):
        points = sorted((r for r in rows if r['run_name'].startswith(prefix)), key=lambda r: r[key])
        if points:
            axis.scatter([r[key] for r in points], [r['validation_loss'] for r in points],
                         label=measurement_label, color='#2463a0', zorder=3)
        fit = analysis.get('controlled_axis_fits', {}).get(fit_key, {})
        if fit_available and fit.get('parameters') and len(points) >= 3:
            p = fit['parameters']
            x = np.geomspace(points[0][key], points[-1][key], 160)
            y = p['floor'] + np.exp(p['log_coefficient'] - p['exponent'] * np.log(x))
            axis.plot(x, y, label='Controlled-axis fit', color='#d3781c')
        axis.set(xscale='log', title=title, xlabel=xlabel, ylabel='Validation cross-entropy')
        if points: axis.legend(fontsize=8)
    axis = axes[1, 0]
    for family, color in zip(('c0', 'c1', 'c2'), ('#2463a0', '#d3781c', '#3c8c59')):
        points = sorted((r for r in rows if r['run_name'].startswith(f'e1_iso_{family}_')),
                        key=lambda r: r['n_params'])
        if points:
            axis.plot([r['n_params'] for r in points], [r['validation_loss'] for r in points],
                      marker='o', label=family, color=color)
    axis.set(xscale='log', title=f'IsoFLOP families ({observation_label})',
             xlabel='Non-embedding parameters', ylabel='Validation cross-entropy')
    if axis.lines: axis.legend(fontsize=8)
    axis = axes[1, 1]
    if fit_available and residuals:
        axis.scatter([r['observed'] for r in residuals], [r['residual'] for r in residuals],
                     color='#2463a0')
        axis.axhline(0, color='gray', linewidth=1)
        axis.set(xlabel='Observed held-out loss', ylabel='Predicted minus observed')
    else:
        axis.text(.5, .5, 'Joint fit diagnostics unavailable: endpoints or identities incomplete',
                  ha='center', va='center', transform=axis.transAxes, wrap=True)
        axis.set(xlabel='Observed held-out loss', ylabel='Fit residual')
    axis.set_title('Joint L(N,D) residuals' + (' — diagnostic only' if fit_gate != 'PASS' else ''))
    for axis in axes.flat:
        axis.grid(alpha=.2)
    paths = []
    try:
        for extension in ('png', 'svg'):
            path = output / f'scaling_curves.{extension}'
            temporary = output / f'.scaling_curves.{os.getpid()}.{extension}'
            figure.savefig(temporary, format=extension, dpi=180)
            temporary.replace(path)
            paths.append(path)
    finally:
        plt.close(figure)

    frontier_figure, frontier_axis = plt.subplots(figsize=(7.5, 5.2), layout='constrained')
    frontier = analysis.get('compute_optimal_frontier', [])
    frontier_rows = ([row for row in frontier if row.get('status') == 'PASS']
                     if frontier_available else [])
    if frontier_rows:
        frontier_rows = sorted(frontier_rows, key=lambda row: row['flops'])
        xs = np.asarray([row['flops'] for row in frontier_rows], dtype=float)
        ys = np.asarray([row['tokens_per_param'] for row in frontier_rows], dtype=float)
        lower = np.asarray([row['ci95']['tokens_per_param'][0] for row in frontier_rows], dtype=float)
        upper = np.asarray([row['ci95']['tokens_per_param'][1] for row in frontier_rows], dtype=float)
        frontier_axis.plot(xs, ys, marker='o', color='#2463a0',
                           label='Fitted compute-optimal D*/N*')
        frontier_axis.fill_between(xs, lower, upper, color='#2463a0', alpha=.18,
                                   label='Paired-seed bootstrap 95% interval')
    else:
        frontier_axis.text(.5, .5, 'Compute-optimal frontier pending a stable seed-aware fit',
                           ha='center', va='center', transform=frontier_axis.transAxes, wrap=True)
    frontier_axis.axhline(20.0, color='#d3781c', linestyle='--',
                          label='Chinchilla 2022 reference ≈20 tokens/parameter')
    frontier_axis.set(xscale='log', yscale='log', xlabel='Compute budget C=6ND (FLOPs)',
                      ylabel='Optimal training tokens / non-embedding parameter',
                      title='Compute-optimal frontier from fitted L(N,D)')
    frontier_axis.grid(alpha=.2)
    frontier_axis.legend(fontsize=8)
    try:
        for extension in ('png', 'svg'):
            path = output / f'compute_optimal_frontier.{extension}'
            temporary = output / f'.compute_optimal_frontier.{os.getpid()}.{extension}'
            frontier_figure.savefig(temporary, format=extension, dpi=180)
            temporary.replace(path)
            paths.append(path)
    finally:
        plt.close(frontier_figure)
    text = io.StringIO()
    fields = ['run_name', 'n_params', 'tokens', 'validation_loss', 'identity_sha256']
    writer = csv.DictWriter(text, fieldnames=fields)
    writer.writeheader()
    for row in sorted(rows, key=lambda r: r['run_name']):
        writer.writerow({key: row.get(key) for key in fields})
    table = output / 'endpoints.csv'
    atomic(table, text.getvalue().encode())
    paths.append(table)
    manifest = {'schema_version': 1, 'status': status, 'audited_endpoints': len(rows),
                'expected_endpoints': expected, 'excluded': excluded,
                'fit_status': analysis.get('status'), 'fit_quality_gate': fit_gate,
                'fit_nrmse': analysis.get('fit_diagnostics', {}).get('normalized_rmse'),
                'input_sha256': trigger,
                'compute_optimal_frontier_status': (
                    'PASS' if frontier_available else
                    'WITHHELD_FIT_GATE_FAIL' if fit_gate == 'FAIL' else 'PENDING'),
                'compute_optimal_frontier_families': sorted(
                    row.get('family') for row in frontier_rows),
                'parameter_basis': 'non-embedding parameters',
                'claim_limit': claim_limit,
                'artifacts': {p.name: digest(p) for p in paths}}
    atomic(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode())
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--identity-audit', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    result = render(args.analysis, args.identity_audit, args.output_dir)
    print(json.dumps({'status': result['status'], 'audited_endpoints': result['audited_endpoints']}))
