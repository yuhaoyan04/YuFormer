import csv
import json
from pathlib import Path

import pytest

from render_e1_scaling import render


def write(path, value):
    path.write_text(json.dumps(value))


def endpoint(name, loss=3.0):
    return {'run_name': name, 'status': 'PASS', 'n_params': 10_000_000,
            'tokens': 1_000_000_000, 'validation_loss': loss, 'identity_sha256': name}


def residuals(rows):
    return [{'run_name': row['run_name'], 'observed': row['validation_loss'],
             'predicted': row['validation_loss'], 'residual': 0.0} for row in rows]


def test_partial_export_requires_identity_and_keeps_nonmonotonic_points(tmp_path):
    analysis, audit, out = tmp_path / 'fit.json', tmp_path / 'audit.json', tmp_path / 'curves'
    rows = [endpoint('e1_iso_c0_50m', 3.2), endpoint('e1_iso_c0_100m', 3.5),
            endpoint('e1_ln_50m'), endpoint('e1_ln_100m', float('nan'))]
    write(analysis, {'status': 'BLOCKED', 'expected_results': 24, 'runs': rows})
    write(audit, {'status': 'BLOCKED', 'rows': [
        {'group': 'e1', 'name': r['run_name'], 'status': 'PASS' if i != 2 else 'FAIL'}
        for i, r in enumerate(rows)]})
    report = render(analysis, audit, out)
    assert report['status'] == 'EXPLORATORY_PROGRESS'
    assert report['audited_endpoints'] == 2
    assert set(report['excluded']) == {'e1_ln_50m', 'e1_ln_100m'}
    with (out / 'endpoints.csv').open() as stream:
        values = list(csv.DictReader(stream))
    assert [float(row['validation_loss']) for row in values] == [3.5, 3.2]
    assert all((out / filename).stat().st_size > 100 for filename in report['artifacts'])
    before = (out / 'scaling_curves.png').stat().st_mtime_ns
    write(analysis, {**json.loads(analysis.read_text()), 'observed_at': 'refreshed'})
    assert render(analysis, audit, out)['input_sha256'] == report['input_sha256']
    assert (out / 'scaling_curves.png').stat().st_mtime_ns == before
    (out / 'endpoints.csv').write_text('corrupted')
    render(analysis, audit, out)
    assert (out / 'endpoints.csv').read_text().startswith('run_name,')


def test_fit_pass_without_complete_identity_is_still_exploratory(tmp_path):
    analysis, audit, out = tmp_path / 'fit.json', tmp_path / 'audit.json', tmp_path / 'curves'
    rows = [endpoint(f'e1_iso_c0_{i}') for i in range(24)]
    write(analysis, {'status': 'PASS', 'expected_results': 24, 'runs': rows,
                     'fit_diagnostics': {'residuals': residuals(rows)},
                     'fit_quality_gate': {'status': 'PASS'}})
    identity = {'status': 'BLOCKED', 'rows': [
        {'group': 'e1', 'name': r['run_name'], 'status': 'PASS'} for r in rows]}
    write(audit, identity)
    assert render(analysis, audit, out)['status'] == 'EXPLORATORY_PROGRESS'
    write(audit, {**identity, 'status': 'PASS'})
    assert render(analysis, audit, out)['status'] == 'ANALYSIS_AND_IDENTITY_PASS'
    write(analysis, {**json.loads(analysis.read_text()), 'status': 'FAIL'})
    assert render(analysis, audit, out)['status'] == 'EXPLORATORY_PROGRESS'


def test_seed_aware_export_describes_seed_mean_and_paired_bootstrap(tmp_path):
    analysis, audit, out = tmp_path / 'fit.json', tmp_path / 'audit.json', tmp_path / 'curves'
    rows = [endpoint(f'e1_iso_c0_{i}') for i in range(24)]
    write(analysis, {
        'status': 'PASS', 'expected_results': 24, 'runs': rows,
        'fit_quality_gate': {'status': 'PASS'},
        'fit_diagnostics': {'seed_level': {'seed_count_per_design_point': 3}, 'residuals': residuals(rows)},
        'compute_optimal_frontier': [
            {'family': family, 'status': 'PASS', 'flops': flops,
             'tokens_per_param': ratio,
             'ci95': {'tokens_per_param': [ratio * .8, ratio * 1.2]}}
            for family, flops, ratio in (('c0', 1e18, 12.0), ('c1', 2e18, 18.0),
                                         ('c2', 4e18, 24.0))
        ],
    })
    write(audit, {'status': 'PASS', 'rows': [
        {'group': 'e1', 'name': row['run_name'], 'status': 'PASS'} for row in rows]})

    report = render(analysis, audit, out)

    assert report['status'] == 'ANALYSIS_AND_IDENTITY_PASS'
    assert 'arithmetic mean across 3 seeds' in report['claim_limit']
    assert 'preserve paired-seed covariance' in report['claim_limit']
    assert 'analytic minima of the fitted law under C=6ND' in report['claim_limit']
    assert 'compute_optimal_frontier.png' in report['artifacts']
    assert report['compute_optimal_frontier_status'] == 'PASS'
    assert report['compute_optimal_frontier_families'] == ['c0', 'c1', 'c2']
    assert '1T prediction extrapolates' in report['claim_limit']


def test_failed_seed_fit_does_not_present_compute_optimal_frontier_as_qualified(tmp_path):
    analysis, audit, out = tmp_path / 'fit.json', tmp_path / 'audit.json', tmp_path / 'curves'
    rows = [endpoint(f'e1_iso_c0_{i}') for i in range(24)]
    write(analysis, {
        'status': 'FAIL', 'expected_results': 24, 'runs': rows,
        'fit_quality_gate': {'status': 'FAIL'},
        'fit_diagnostics': {'seed_level': {'seed_count_per_design_point': 3}, 'residuals': residuals(rows)},
        'compute_optimal_frontier': [{'family': 'c0', 'status': 'PASS', 'flops': 1e18,
                                      'tokens_per_param': 20.0,
                                      'ci95': {'tokens_per_param': [10.0, 30.0]}}],
    })
    write(audit, {'status': 'PASS', 'rows': [
        {'group': 'e1', 'name': row['run_name'], 'status': 'PASS'} for row in rows]})

    report = render(analysis, audit, out)

    assert report['status'] == 'EXPLORATORY_PROGRESS'
    assert 'frontier is withheld' in report['claim_limit']


def test_partial_compute_optimal_frontier_is_not_rendered_as_complete(tmp_path):
    analysis, audit, out = tmp_path / 'fit.json', tmp_path / 'audit.json', tmp_path / 'curves'
    rows = [endpoint(f'e1_iso_c0_{i}') for i in range(24)]
    write(analysis, {
        'status': 'PASS', 'expected_results': 24, 'runs': rows,
        'fit_quality_gate': {'status': 'PASS'},
        'fit_diagnostics': {'seed_level': {'seed_count_per_design_point': 3}, 'residuals': residuals(rows)},
        'compute_optimal_frontier': [{'family': 'c0', 'status': 'PASS', 'flops': 1e18,
                                      'tokens_per_param': 20.0,
                                      'ci95': {'tokens_per_param': [10.0, 30.0]}}],
    })
    write(audit, {'status': 'PASS', 'rows': [
        {'group': 'e1', 'name': row['run_name'], 'status': 'PASS'} for row in rows]})

    report = render(analysis, audit, out)

    assert report['status'] == 'ANALYSIS_AND_IDENTITY_PASS'
    assert 'frontier is withheld' in report['claim_limit']
    assert report['compute_optimal_frontier_status'] == 'PENDING'
    assert report['compute_optimal_frontier_families'] == []


def test_duplicate_endpoints_are_rejected(tmp_path):
    analysis, audit = tmp_path / 'fit.json', tmp_path / 'audit.json'
    row = endpoint('e1_ln_50m')
    write(analysis, {'runs': [row, row]})
    write(audit, {'rows': []})
    with pytest.raises(ValueError, match='duplicate E1 endpoint'):
        render(analysis, audit, tmp_path / 'curves')
