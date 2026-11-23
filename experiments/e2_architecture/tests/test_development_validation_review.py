import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('development_review', Path(__file__).parents[1]/'scripts/development_validation_review.py')
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def evidence():
    domains = ['academic', 'books', 'chinese', 'code', 'encyclopedia', 'math', 'multilingual', 'qa', 'web']
    protocol = dict(domains=domains, seq_len=4096, world_size=8, local_batch_size=8, global_batch_size=64,
                    source_sample_offset=64, samples_per_domain=2816, global_batches_per_domain=44,
                    target_interval=[262145,11796481], validation_metadata_sha256={'all9':'same'},
                    uncertainty=dict(seed=20261114, resamples=10000), training_source_sha256='same')
    reference = dict(domains={d:dict(batch_mean_losses=[7.0+i*.001 for i in range(44)]) for d in domains})
    candidate = copy.deepcopy(reference)
    for values in candidate['domains'].values():
        values['batch_mean_losses'] = [x-.01 for x in values['batch_mean_losses']]
    return protocol, candidate, reference


def test_one_positive_chinese_delta_is_not_hidden_by_other_domains():
    protocol, candidate, reference = evidence()
    candidate['domains']['chinese']['batch_mean_losses'] = [x+.000001 for x in reference['domains']['chinese']['batch_mean_losses']]
    result = review.compare(protocol, candidate, protocol, reference, np.random.default_rng(1))
    assert not result['chinese']['mean_strictly_improves']
    assert result['chinese']['paired_bootstrap_95_ci'][0] > 0
    assert all(result[d]['mean_strictly_improves'] for d in protocol['domains'] if d != 'chinese')


def test_equal_length_but_shifted_window_is_rejected():
    protocol, candidate, reference = evidence()
    other = copy.deepcopy(protocol)
    other['target_interval'] = [262146,11796482]
    with pytest.raises(ValueError, match='different windows'):
        review.compare(protocol, candidate, other, reference, np.random.default_rng(1))


@pytest.mark.parametrize('replacement', [[6.0]*43, [6.0]*43+[float('nan')]])
def test_missing_or_nonfinite_paired_batch_is_rejected(replacement):
    protocol, candidate, reference = evidence()
    candidate['domains']['chinese']['batch_mean_losses'] = replacement
    with pytest.raises(ValueError, match='paired finite'):
        review.compare(protocol, candidate, protocol, reference, np.random.default_rng(1))
