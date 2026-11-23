import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from calibrate_training_tokens import DOMAINS
from temporal_order_admission import validate_phase
from temporal_order_admission import bind_recount_inputs


@pytest.fixture
def actual_window():
    samples, seq, block, offset = 80, 4096, 8, 16
    counts = {d: 0 for d in DOMAINS}
    counts['web'] = samples*seq
    weights = {d: [int(d == 'web')]*10 for d in DOMAINS}
    row = {'stage_sample_limit': samples, 'source_sample_offset': offset,
           'source_window_selection': {'file': 'actual-window.json', 'sha256': 'a'*64},
           'permutation': {'sha256': 'b'*64}, 'base_permutation_sha256': 'c'*64}
    master = {'seq_len': seq, 'block_samples': block}
    summary = dict(source_sample_offset=offset, temporal_mixing_pass=True,
                   whole_consumed_counts_match_admitted_window=True,
                   permutation_matches_production_window_array=True)
    report = dict(status='PASS_DIAGNOSTIC', source_sample_offset=offset,
        target_interval=[1+offset*seq, 1+(offset+samples)*seq],
        source_window={'selection_report': copy.deepcopy(row['source_window_selection'])},
        block_samples=block, seq_len=seq, stage_sample_limit=samples,
        trained_target_tokens=samples*seq, block_count=10, permutation_sha256='b'*64,
        base_permutation_sha256='c'*64, every_causal_block_exactly_once=True,
        whole_domain_token_counts_unchanged=True, windows_are_whole_blocks=True,
        whole_domain_token_counts=counts, permuted_window_weights=weights,
        window_block_boundaries=list(range(11)), temporal_mixing=dict(
            pass_=True, sigma_multiplier=6, global_domain_tolerances_unchanged=True,
            allowed_deviation=[[.0001]*9 for _ in range(10)],
            deviation=[[0]*9 for _ in range(10)]))
    report['temporal_mixing']['pass'] = report['temporal_mixing'].pop('pass_')
    return report, row, summary, master


def test_exact_actual_absolute_window_order_can_be_admitted(actual_window):
    result = validate_phase(*actual_window)
    assert result['source_sample_offset'] == 16
    assert result['target_interval'] == [65537, 393217]


def test_training_freeze_replacement_keeps_the_exact_archived_physical_origin(tmp_path):
    import hashlib
    original = b'original physically selected recipe'
    sha = hashlib.sha256(original).hexdigest()
    archive = tmp_path/'data/processed/recipe_origins'/sha/'objects'/sha
    archive.parent.mkdir(parents=True)
    archive.write_bytes(original)
    current = tmp_path/'experiments/e4_curriculum/results/frozen_1t_recipe.json'
    current.parent.mkdir(parents=True)
    current.write_bytes(b'new legitimate training-only freeze')
    evidence = {'experiments/e4_curriculum/results/frozen_1t_recipe.json': sha}
    assert bind_recount_inputs(tmp_path, evidence, sha) == {str(archive.relative_to(tmp_path)): sha}
    with pytest.raises(ValueError, match='another original physical'):
        bind_recount_inputs(tmp_path, evidence, '0'*64)
    archive.write_bytes(b'corrupted old data origin')
    with pytest.raises(ValueError, match='byte identity'):
        bind_recount_inputs(tmp_path, evidence, sha)


@pytest.mark.parametrize('fault', ['wrong_offset', 'prefix_interval', 'wrong_permutation',
    'wrong_window_proof', 'wrong_population', 'missing_domain', 'empty_window',
    'false_statistical_pass', 'false_reported_deviation', 'nan_band', 'relaxed_global_gate'])
def test_claimed_pass_cannot_hide_wrong_consumption_or_temporal_bias(actual_window, fault):
    report, row, summary, master = actual_window
    if fault == 'wrong_offset': report['source_sample_offset'] = 0
    elif fault == 'prefix_interval': report['target_interval'] = [1, 80*4096+1]
    elif fault == 'wrong_permutation': report['permutation_sha256'] = 'd'*64
    elif fault == 'wrong_window_proof': report['source_window']['selection_report']['sha256'] = 'e'*64
    elif fault == 'wrong_population': report['whole_domain_token_counts']['web'] -= 1
    elif fault == 'missing_domain': report['permuted_window_weights'].pop('math')
    elif fault == 'empty_window': report['window_block_boundaries'][1] = 0
    elif fault == 'false_statistical_pass': report['permuted_window_weights']['web'][0] = .5
    elif fault == 'false_reported_deviation': report['temporal_mixing']['deviation'][0][0] = .1
    elif fault == 'nan_band': report['temporal_mixing']['allowed_deviation'][0][0] = float('nan')
    else: report['temporal_mixing']['global_domain_tolerances_unchanged'] = False
    with pytest.raises(ValueError):
        validate_phase(report, row, summary, master)
