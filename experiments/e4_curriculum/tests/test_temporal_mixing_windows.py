from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_training_block_permutation import temporal_mixing_check


def weights(counts,edges):
    values=np.array([counts[a:b].sum(axis=0) for a,b in zip(edges[:-1],edges[1:])])
    return values/values.sum(axis=1,keepdims=True)


def test_global_balance_cannot_hide_a_strong_temporal_domain_shift():
    counts=np.zeros((20000,2),dtype=np.int64);counts[:10000,0]=100;counts[10000:,1]=100
    edges=list(range(0,20001,2000))
    assert temporal_mixing_check(counts,weights(counts,edges),edges)['pass'] is False
    shuffled=counts[np.random.default_rng(7).permutation(len(counts))]
    result=temporal_mixing_check(counts,weights(shuffled,edges),edges)
    assert result['pass'] is True
    assert result['global_domain_tolerances_unchanged'] is True


def test_constant_block_mixtures_have_zero_variance_and_remain_valid():
    counts=np.tile(np.array([[7,3]]),(1000,1));edges=list(range(0,1001,100))
    assert temporal_mixing_check(counts,weights(counts,edges),edges)['pass'] is True


def test_empty_temporal_windows_are_never_silently_accepted():
    counts=np.tile(np.array([[7,3]]),(4,1))
    with pytest.raises(ValueError):
        temporal_mixing_check(counts,np.zeros((2,2)),[0,0,4])
