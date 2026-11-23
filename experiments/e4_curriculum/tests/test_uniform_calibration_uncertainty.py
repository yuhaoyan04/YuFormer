from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_uniform_calibration import ratio_statistics


def test_constant_expansion_has_zero_ratio_variance():
    row = ratio_statistics([10, 20, 30], [20, 40, 60], 100)
    assert row['ratio'] == 2
    assert row['variance'] == 0


def test_ratio_uncertainty_accounts_for_document_lengths_and_finite_population():
    sample = ratio_statistics([10, 20, 30], [10, 30, 30], 100)
    census = ratio_statistics([10, 20, 30], [10, 30, 30], 3)
    assert sample['ratio'] == pytest.approx(70/60)
    assert sample['variance'] > 0
    assert sample['approximate_95pct_interval'][0] < 70/60 < sample['approximate_95pct_interval'][1]
    assert census['variance'] == 0


@pytest.mark.parametrize('ref,actual,n', [([0, 1], [1, 1], 2), ([1, 1], [1, 0], 2),
    ([1, 1], [1, np.nan], 2), ([1, 1], [1, 1], 1), ([1], [1], 5), ([1, 2], [1], 5)])
def test_invalid_samples_are_rejected(ref, actual, n):
    with pytest.raises(ValueError, match='invalid'):
        ratio_statistics(ref, actual, n)
