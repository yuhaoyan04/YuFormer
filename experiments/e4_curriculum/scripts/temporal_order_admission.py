"""Bind actual all-phase temporal recounts to the published production order."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from training_data_origin import read_bound

PHASES = ('warmup', 'stable', 'anneal')
RECOUNT = 'data/health/production-stage-order-recount.json'
REVIEW = 'data/health/production-stage-order-admission.json'
AUDITOR = 'experiments/e4_curriculum/scripts/audit_training_block_permutation.py'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind_recount_inputs(root, evidence, data_recipe_sha256):
    inputs = {}
    for dependency, expected in evidence.items():
        if dependency == 'experiments/e4_curriculum/results/frozen_1t_recipe.json':
            if expected != data_recipe_sha256:
                raise ValueError('temporal recount belongs to another original physical data recipe')
            # Training ADOPT replaces the current freeze. The physical data
            # proof must keep binding its immutable original bytes, otherwise
            # every valid training adoption makes the proof circular/stale.
            archive = Path(root)/'data/processed/recipe_origins'/expected/'objects'/expected
            read_bound(archive, expected)
            inputs[str(archive.relative_to(root))] = expected
        else:
            read_bound(Path(root)/dependency, expected)
            inputs[dependency] = expected
    return inputs


def validate_phase(report, row, summary, master):
    """Reject a claimed PASS if actual windows, population, or order disagree."""
    samples, seq, block = row['stage_sample_limit'], master['seq_len'], master['block_samples']
    offset = row['source_sample_offset']
    expected_interval = [1+offset*seq, 1+(offset+samples)*seq]
    if (report.get('status') != 'PASS_DIAGNOSTIC'
            or report.get('source_sample_offset') != offset
            or report.get('target_interval') != expected_interval
            or report.get('source_window', {}).get('selection_report') != row['source_window_selection']
            or report.get('block_samples') != block or report.get('seq_len') != seq
            or report.get('stage_sample_limit') != samples
            or report.get('trained_target_tokens') != samples*seq
            or report.get('block_count') != samples//block
            or report.get('permutation_sha256') != row['permutation']['sha256']
            or report.get('base_permutation_sha256') != row['base_permutation_sha256']
            or report.get('every_causal_block_exactly_once') is not True
            or report.get('whole_domain_token_counts_unchanged') is not True
            or report.get('windows_are_whole_blocks') is not True
            or summary.get('source_sample_offset') != offset
            or summary.get('temporal_mixing_pass') is not True
            or summary.get('whole_consumed_counts_match_admitted_window') is not True
            or summary.get('permutation_matches_production_window_array') is not True):
        raise ValueError('temporal admission does not match the actual production window/order')
    counts = report['whole_domain_token_counts']
    domains = list(report['permuted_window_weights'])
    if (set(domains) != set(counts) or len(domains) != 9
            or sum(counts.values()) != samples*seq or any(v < 0 for v in counts.values())):
        raise ValueError('temporal admitted domain population does not close exactly')
    # The actual auditor uses calibration.DOMAINS; JSON sort order is not that order.
    from calibrate_training_tokens import DOMAINS
    if set(domains) != set(DOMAINS):
        raise ValueError('temporal review must cover exactly the nine training domains')
    if any(len(report['permuted_window_weights'][domain]) != 10 for domain in DOMAINS):
        raise ValueError('temporal admission is missing actual window weights')
    if any(not math.isclose(sum(report['permuted_window_weights'][domain][i] for domain in DOMAINS),
                            1., rel_tol=0, abs_tol=1e-12) for i in range(10)):
        raise ValueError('each actual temporal window must contain the entire domain population')
    edges = report['window_block_boundaries']
    if (len(edges) != 11 or edges[0] != 0 or edges[-1] != samples//block
            or any(a >= b for a, b in zip(edges[:-1], edges[1:]))):
        raise ValueError('temporal windows must cover all causal blocks without empty windows')
    mixing = report['temporal_mixing']
    if (mixing.get('pass') is not True or mixing.get('sigma_multiplier') != 6
            or mixing.get('global_domain_tolerances_unchanged') is not True
            or len(mixing.get('allowed_deviation', [])) != 10
            or len(mixing.get('deviation', [])) != 10):
        raise ValueError('actual temporal statistical gate has not passed')
    for domain_index, domain in enumerate(DOMAINS):
        weights = report['permuted_window_weights'][domain]
        if len(weights) != 10:
            raise ValueError('temporal admission is missing actual window weights')
        mean = counts[domain]/(samples*seq)
        for index, weight in enumerate(weights):
            allowed = mixing['allowed_deviation'][index][domain_index]
            actual = abs(weight-mean)
            declared = mixing['deviation'][index][domain_index]
            if (not all(math.isfinite(x) for x in (weight, allowed, actual, declared))
                    or not 0 <= weight <= 1 or allowed < 0 or actual > allowed
                    or not math.isclose(actual, declared, rel_tol=1e-9, abs_tol=1e-12)):
                raise ValueError('claimed temporal PASS disagrees with actual window weights')
    return {'source_sample_offset': offset, 'target_interval': expected_interval,
            'whole_domain_token_counts': counts, 'temporal_mixing_pass': True}


def inspect_order(root, master_path):
    root, master_path = Path(root).resolve(), Path(master_path).resolve()
    master_raw, master_sha = read_bound(master_path)
    master = json.loads(master_raw)
    if master.get('scope') != 'PRODUCTION' or master.get('schema_version') != 3 or len(master['stages']) != 3:
        raise ValueError('temporal admission requires the actual three-stage production window manifest')
    recount_raw, recount_sha = read_bound(root/RECOUNT)
    recount = json.loads(recount_raw)
    if (recount.get('status') != 'PASS_EXACT_COVERAGE_AWAITING_TEMPORAL_REVIEW'
            or recount.get('audit_tool_sha256') != digest(root/AUDITOR)
            or [r.get('phase') for r in recount['phases']] != list(PHASES)):
        raise ValueError('all three current-source independent temporal recounts are required')
    inputs = {str(master_path.relative_to(root)): master_sha, RECOUNT: recount_sha,
              AUDITOR: digest(root/AUDITOR)}
    rows = []
    for phase, row, summary in zip(PHASES, master['stages'], recount['phases']):
        name = f'data/health/{phase}-training-block-permutation.json'
        if summary.get('report') != name:
            raise ValueError('temporal recount points to a different phase report')
        raw, sha = read_bound(root/name, summary.get('report_sha256'))
        report = json.loads(raw)
        if report.get('phase') != phase or report.get('audit_tool_sha256') != inputs[AUDITOR]:
            raise ValueError('temporal phase source identity is stale')
        generator = 'experiments/e4_curriculum/scripts/training_block_permutation.py'
        meta_name = f'data/token_streams/pretrain-1t/{phase}.meta.json'
        if (report.get('permutation_tool_sha256') != digest(root/generator)
                or report.get('evidence_sha256', {}).get(meta_name) != row['stream_meta_sha256']):
            raise ValueError('temporal recount source generator or actual stream metadata differs from production')
        inputs[generator] = digest(root/generator)
        inputs.update(bind_recount_inputs(root, report['evidence_sha256'], master['data_frozen_recipe_sha256']))
        inputs[name] = sha
        rows.append({'phase': phase, 'report_sha256': sha, **validate_phase(report, row, summary, master)})
    return inputs, rows


def require_temporal_admission(root, master_path):
    raw, sha = read_bound(Path(root)/REVIEW)
    report = json.loads(raw)
    inputs, rows = inspect_order(root, master_path)
    if (report.get('status') != 'PASS_PRODUCTION_TEMPORAL_ADMISSION'
            or report.get('review_tool_sha256') != digest(__file__)
            or report.get('inputs') != inputs or report.get('phases') != rows):
        raise ValueError('production temporal admission is missing, stale, or belongs to another order')
    inputs[REVIEW] = sha
    inputs[str(Path(__file__).relative_to(Path(root)))] = digest(__file__)
    return inputs, {'report': REVIEW, 'sha256': sha}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--master', type=Path, required=True)
    args = parser.parse_args()
    inputs, rows = inspect_order(args.root, args.master)
    report = {'schema_version': 1, 'status': 'PASS_PRODUCTION_TEMPORAL_ADMISSION',
              'review_kind': 'EXPLICIT_STATISTICAL_AND_EXACT_WINDOW_REVALIDATION',
              'production_qualified': False, 'review_tool_sha256': digest(__file__),
              'inputs': inputs, 'phases': rows}
    target = args.root/REVIEW
    temporary = target.with_name(target.name+f'.tmp-{os.getpid()}')
    with temporary.open('w') as handle:
        handle.write(json.dumps(report, sort_keys=True, indent=2)+'\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    require_temporal_admission(args.root, args.master)
    print(json.dumps({'status': report['status'], 'phases': rows}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
