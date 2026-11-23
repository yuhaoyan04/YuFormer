from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import training_block_permutation as blocks


def test_independent_order_preserves_every_causal_target_and_stage():
    observed, expected = [], []
    for phase, samples in [('warmup', 128), ('stable', 512), ('anneal', 256)]:
        permutation = blocks.build_permutation(samples, 8, seed='e134-test', phase=phase)
        tokens = np.arange(samples*16+1)
        targets = [tokens[blocks.physical_sample(i, permutation, 8)*16+1:
                          blocks.physical_sample(i, permutation, 8)*16+17] for i in range(samples)]
        np.testing.assert_array_equal(np.sort(np.concatenate(targets)), tokens[1:])
        observed.extend([(phase, blocks.physical_sample(i, permutation, 8)) for i in range(samples)])
        expected.extend([(phase, i) for i in range(samples)])
        for cursor in range(0, samples+1, 8):
            resumed = [blocks.physical_sample(i, permutation, 8) for i in range(cursor, samples)]
            assert resumed == [sample for p, sample in observed if p == phase][cursor:]
    assert sorted(observed) == sorted(expected)


def test_order_binds_seed_phase_and_explicit_block_dimensions():
    first = blocks.build_permutation(512, 8, seed='a', phase='warmup')
    np.testing.assert_array_equal(first, blocks.build_permutation(512, 8, seed='a', phase='warmup'))
    assert not np.array_equal(first, blocks.build_permutation(512, 8, seed='b', phase='warmup'))
    assert not np.array_equal(first, blocks.build_permutation(512, 8, seed='a', phase='stable'))
    with pytest.raises(ValueError, match='divisible'):
        blocks.build_permutation(511, 8, seed='a', phase='warmup')


@pytest.mark.parametrize('permutation', [np.array([0, 0], dtype='<u4'),
                                       np.array([0, 2], dtype='<u4'),
                                       np.array([0, 1], dtype='<u8'),
                                       np.array([0], dtype='<u4')])
def test_incomplete_duplicate_out_of_range_or_wrong_dtype_is_rejected(permutation):
    with pytest.raises(ValueError, match='exactly once'):
        blocks.validate_permutation(permutation, 16, 8)


def test_long_document_token_accounting_matches_naive_expansion():
    lengths = np.array([1, 2, 37, 3, 28, 9], dtype=np.int64)
    domains = np.array([0, 1, 2, 0, 1, 2])
    ends, starts = np.cumsum(lengths), np.cumsum(lengths)-lengths
    counts, delta = np.zeros((10, 3), dtype=np.int64), np.zeros((11, 3), dtype=np.int64)
    for domain in range(3):
        mask = domains == domain
        blocks.add_document_intervals(counts, delta, domain, starts[mask], ends[mask], block_tokens=8)
    counts += np.cumsum(delta, axis=0)[:-1]
    expanded = np.repeat(domains, lengths).reshape(10, 8)
    expected = np.array([[sum(row == d) for d in range(3)] for row in expanded])
    np.testing.assert_array_equal(counts, expected)


def test_independent_blocks_remove_monotonic_domain_curriculum():
    counts = np.zeros((10000, 2), dtype=np.int64)
    counts[:, 0] = np.linspace(9000, 1000, 10000).astype(np.int64)
    counts[:, 1] = 10000-counts[:, 0]
    permutation = blocks.build_permutation(640000, 64, seed='e134-test', phase='warmup')
    before = counts.reshape(10, 1000, 2).sum(axis=1)
    after = counts[permutation].reshape(10, 1000, 2).sum(axis=1)
    assert np.max(np.abs(before/before.sum(axis=1, keepdims=True)-.5)) > .35
    assert np.max(np.abs(after/after.sum(axis=1, keepdims=True)-.5)) < .025
    np.testing.assert_array_equal(after.sum(axis=0), counts.sum(axis=0))
