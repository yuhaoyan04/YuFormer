"""Compare source-bound development evaluations while preserving failed endpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
WINDOW_KEYS = ('domains', 'seq_len', 'world_size', 'local_batch_size', 'global_batch_size',
               'source_sample_offset', 'samples_per_domain', 'global_batches_per_domain',
               'target_interval', 'validation_metadata_sha256', 'uncertainty')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stat_identity(path):
    value = Path(path).stat()
    return dict(bytes=value.st_size, device=value.st_dev, inode=value.st_ino, mtime_ns=value.st_mtime_ns)


def load_evaluation(root, plan_path, case):
    plan_path = root/plan_path
    plan = json.loads(plan_path.read_bytes())
    protocol = json.loads((root/plan['protocol']).read_bytes())
    row = plan['cases'][case]
    path = root/row['report']
    report = json.loads(path.read_bytes())
    if (report.get('status') != 'PASS_READONLY_DISJOINT_VALIDATION'
            or report.get('case') != case
            or report.get('plan_sha256') != digest(plan_path)
            or report.get('protocol_sha256') != digest(root/plan['protocol'])
            or report.get('training_source_sha256') != protocol['training_source_sha256']
            or report.get('checkpoint_metadata_sha256') != row['checkpoint_metadata_sha256']
            or report.get('source_inputs') != plan['source_inputs']
            or report.get('training_steps_executed') != 0
            or report.get('checkpoint_modified') is not False):
        raise ValueError('development report does not close its actual execution plan: '+case)
    for name, expected in plan['source_inputs'].items():
        if digest(root/name) != expected:
            raise ValueError('development evidence source changed: '+name)
    checkpoint = root/row['checkpoint']
    actual = {p.name:stat_identity(p) for p in sorted(checkpoint.iterdir()) if p.is_file()}
    if actual != row['checkpoint_stat'] or digest(checkpoint/'.metadata') != row['checkpoint_metadata_sha256']:
        raise ValueError('development checkpoint changed: '+case)
    if any(stat_identity(root/v['file']) != v['stat'] for v in plan['validation_streams'].values()):
        raise ValueError('fully audited validation bytes changed')
    primary = json.loads((root/row['primary_result']).read_bytes())
    hyper = primary['hyperparameters']
    if (primary.get('status') != 'PASS' or hyper.get('trained_steps') != 200
            or hyper.get('trained_tokens') != 838860800 or hyper.get('world_size') != 8):
        raise ValueError('development checkpoint did not consume the matched actual training budget')
    if set(report['domains']) != set(protocol['domains']) or len(protocol['domains']) != 9:
        raise ValueError('development evaluation needs all nine domains')
    for domain in protocol['domains']:
        values = report['domains'][domain]
        batches = np.asarray(values['batch_mean_losses'], dtype=np.float64)
        original = primary['metrics'][f'validation_metrics/domain/{domain}/loss']['last']
        scalars = [original, values['mean_loss'], values['original_primary_loss'], values['primary_reproduced_loss']]
        if (batches.shape != (protocol['global_batches_per_domain'],)
                or not np.isfinite(batches).all() or not np.isfinite(scalars).all()
                or values['samples'] != protocol['samples_per_domain']
                or values['target_tokens'] != protocol['samples_per_domain']*protocol['seq_len']
                or abs(values['mean_loss']-float(batches.mean())) > 1e-12
                or values['original_primary_loss'] != original
                or abs(values['primary_reproduced_loss']-original) > 1e-5):
            raise ValueError('development batches or primary reproduction do not close: '+case+'/'+domain)
    aggregate = float(np.mean([report['domains'][d]['mean_loss'] for d in protocol['domains']]))
    if not np.isfinite(report['aggregate_mean_loss']) or abs(aggregate-report['aggregate_mean_loss']) > 1e-12:
        raise ValueError('development aggregate does not match all nine domains')
    inputs = {str(plan_path.relative_to(root)):digest(plan_path), plan['protocol']:digest(root/plan['protocol']),
              row['report']:digest(path), row['primary_result']:digest(root/row['primary_result']),
              **plan['source_inputs']}
    return protocol, report, inputs


def compare(candidate_protocol, candidate, reference_protocol, reference, rng):
    if any(candidate_protocol[k] != reference_protocol[k] for k in WINDOW_KEYS):
        raise ValueError('development models were evaluated on different windows/protocols')
    if candidate_protocol['training_source_sha256'] != reference_protocol['training_source_sha256']:
        raise ValueError('development models use different training source')
    domains = {}
    for domain in candidate_protocol['domains']:
        a = np.asarray(candidate['domains'][domain]['batch_mean_losses'], dtype=np.float64)
        b = np.asarray(reference['domains'][domain]['batch_mean_losses'], dtype=np.float64)
        expected = candidate_protocol['global_batches_per_domain']
        if a.shape != (expected,) or b.shape != (expected,) or not np.isfinite([a, b]).all():
            raise ValueError('all paired finite development batches are required')
        delta = a-b
        samples = rng.integers(0, len(delta), size=(candidate_protocol['uncertainty']['resamples'], len(delta)))
        low, high = np.quantile(delta[samples].mean(axis=1), [.025, .975])
        domains[domain] = dict(candidate_mean=float(a.mean()), reference_mean=float(b.mean()),
            delta=float(delta.mean()), paired_bootstrap_95_ci=[float(low), float(high)],
            mean_strictly_improves=bool(delta.mean() < 0))
    return domains


def review(root, candidate_plan, case, reference_plan):
    protocol, candidate, inputs = load_evaluation(root, candidate_plan, case)
    rng = np.random.default_rng(protocol['uncertainty']['seed'])
    comparisons = {}
    primary = {}
    aggregate_passed = True
    for other in ('matched_baseline_b8', 'larger_lr2e4_b8'):
        other_protocol, reference, other_inputs = load_evaluation(root, reference_plan, other)
        comparisons[other] = compare(protocol, candidate, other_protocol, reference, rng)
        for key, value in other_inputs.items():
            if key in inputs and inputs[key] != value:
                raise ValueError('inconsistent evidence identities: '+key)
            inputs[key] = value
        aggregate_passed &= candidate['aggregate_mean_loss'] < reference['aggregate_mean_loss']
        if other == 'matched_baseline_b8':
            for domain in protocol['domains']:
                a = candidate['domains'][domain]['original_primary_loss']
                b = reference['domains'][domain]['original_primary_loss']
                primary[domain] = dict(candidate_loss=a, baseline_loss=b, delta=a-b, strictly_improves=a < b)
    primary_passed = all(v['strictly_improves'] for v in primary.values())
    expanded_passed = all(v['mean_strictly_improves'] for rows in comparisons.values() for v in rows.values())
    passed = primary_passed and expanded_passed and aggregate_passed
    inputs[str(Path(__file__).relative_to(root))] = digest(__file__)
    return dict(status='PASS_PRIMARY_AND_EXPANDED_ALL_NINE' if passed else 'FAIL_PRIMARY_OR_EXPANDED_ALL_NINE',
                case=case, primary_all_nine_passed=primary_passed, expanded_all_nine_passed=expanded_passed,
                aggregate_passed=bool(aggregate_passed), primary_comparison=primary, comparisons=comparisons,
                source_inputs=inputs, candidate_aggregate_mean_loss=candidate['aggregate_mean_loss'],
                training_source_sha256=protocol['training_source_sha256'],
                interpretation='Development comparison; confidence intervals are descriptive; all prior failures retained.',
                production_qualified=False, formal_pretrain_started=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-plan', type=Path, required=True)
    parser.add_argument('--reference-plan', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review(ROOT, args.candidate_plan, args.case, args.reference_plan)
    with args.output.open('x') as output:
        json.dump(result, output, sort_keys=True, indent=2)
        output.write('\n')
    print(json.dumps({key:result[key] for key in ('status','case','primary_all_nine_passed',
                    'expanded_all_nine_passed','aggregate_passed','candidate_aggregate_mean_loss')}))
