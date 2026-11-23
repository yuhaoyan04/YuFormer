from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import training_source_windows as windows


def fixture_files(tmp_path, positions=None, lengths=None):
    lengths = [3, 1, 4, 2, 6, 8] if lengths is None else lengths
    domains = ['web', 'code', 'web', 'math', 'code', 'books']
    ledger, order = tmp_path/'lengths.u32', tmp_path/'order.parquet'
    np.array(lengths, dtype='<u4').tofile(ledger)
    pq.write_table(pa.table({'position': list(range(6)) if positions is None else positions,
                             'domain': domains}), order)
    return order, ledger, lengths, domains


def test_every_candidate_matches_brute_force_token_domain_intervals(tmp_path):
    order, ledger, lengths, domains = fixture_files(tmp_path)
    result = windows.count_source_windows(order, ledger, seq_len=2, stage_samples=4, block_samples=2)
    tokens = [domain for domain, length in zip(domains, lengths) for _ in range(length)]
    assert result['total_source_tokens'] == len(tokens)
    assert [r['source_sample_offset'] for r in result['candidates']] == [0, 2, 4, 6]
    for row in result['candidates']:
        start, stop = row['target_interval']
        assert row['domain_tokens'] == {d: tokens[start:stop].count(d) for d in windows.DOMAINS}
        assert stop-start == 8


def test_zero_headroom_still_checks_the_one_exact_window(tmp_path):
    order, ledger, _, _ = fixture_files(tmp_path)
    result = windows.count_source_windows(order, ledger, seq_len=1, stage_samples=23, block_samples=1)
    assert [r['source_sample_offset'] for r in result['candidates']] == [0]
    assert result['candidates'][0]['target_interval'] == [1, 24]


@pytest.mark.parametrize('samples,block', [(4, 3), (0, 2), (26, 2)])
def test_invalid_budget_alignment_or_capacity_is_rejected(tmp_path, samples, block):
    order, ledger, _, _ = fixture_files(tmp_path)
    with pytest.raises(ValueError):
        windows.count_source_windows(order, ledger, seq_len=2, stage_samples=samples, block_samples=block)


def test_noncanonical_source_positions_cannot_silently_mislabel_lengths(tmp_path):
    order, ledger, _, _ = fixture_files(tmp_path, positions=[1, 0, 2, 3, 4, 5])
    with pytest.raises(ValueError, match='position-ordered'):
        windows.count_source_windows(order, ledger, seq_len=2, stage_samples=4, block_samples=2)


def test_zero_document_length_is_not_admitted(tmp_path):
    order, ledger, _, _ = fixture_files(tmp_path, lengths=[3, 0, 4, 2, 6, 8])
    with pytest.raises(ValueError, match='positive EOD'):
        windows.count_source_windows(order, ledger, seq_len=2, stage_samples=4, block_samples=2)


def test_failed_prefix_uses_best_valid_window_without_relaxing_the_gate():
    target = dict.fromkeys(windows.DOMAINS, 1/9)
    wrong = dict.fromkeys(windows.DOMAINS, 100)
    wrong.update(web=103, code=97)
    good = dict.fromkeys(windows.DOMAINS, 100)
    result = windows.choose_source_window({'candidates': [
        {'source_sample_offset': 0, 'domain_tokens': wrong},
        {'source_sample_offset': 8, 'domain_tokens': good}]}, target)
    assert result['candidates'][0]['mixture']['status'] == 'FAIL'
    assert result['selected']['source_sample_offset'] == 8
    assert result['selected']['mixture']['absolute_tolerance'] == .002
    assert result['supervised_prefix_unchanged'] is False


def test_a_valid_original_prefix_is_preserved():
    target = dict.fromkeys(windows.DOMAINS, 1/9)
    counts = dict.fromkeys(windows.DOMAINS, 100)
    result = windows.choose_source_window({'candidates': [
        {'source_sample_offset': 0, 'domain_tokens': counts},
        {'source_sample_offset': 8, 'domain_tokens': counts}]}, target)
    assert result['selected']['source_sample_offset'] == 0
    assert result['supervised_prefix_unchanged'] is True


def test_no_valid_candidate_remains_blocked():
    result = windows.choose_source_window({'candidates': [
        {'source_sample_offset': 0, 'domain_tokens': {'web': 900}}]}, dict.fromkeys(windows.DOMAINS, 1/9))
    assert result['status'] == 'BLOCKED_NO_COMPLIANT_WINDOW'
    assert result['selected'] is None
