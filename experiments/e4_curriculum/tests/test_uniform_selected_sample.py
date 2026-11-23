import importlib.util
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import uniform_selected_sample as sampling


def population_file(tmp_path, n=32):
    domains = sampling.calibration.DOMAINS
    rows = [{'doc_id': (rank*len(domains)+index).to_bytes(16, 'big'),
             'domain': domain, 'n_tokens': 1}
            for rank in range(n) for index, domain in enumerate(domains)]
    path = tmp_path / 'population.parquet'
    pq.write_table(pa.Table.from_pylist(rows), path)
    population = {d: {'selected_docs': n, 'selected_tokens': n} for d in domains}
    return path, population


def select(parent, population, destination, target=10, batch_size=17, **kwargs):
    return sampling.sample_order(parent, target, destination, population=population,
        expected_parent_sha256=sampling.calibration.digest(parent), seed='regression-133',
        phase='warmup', batch_size=batch_size, **kwargs)


def test_independent_sampling_is_deterministic_across_read_batches(tmp_path):
    parent, population = population_file(tmp_path)
    first, second = tmp_path / 'first.parquet', tmp_path / 'second.parquet'
    a = select(parent, population, first)
    b = select(parent, population, second, batch_size=31)
    assert first.read_bytes() == second.read_bytes()
    assert a == b
    table = pq.read_table(first)
    assert table['position'].to_pylist() == list(range(90))
    assert set(a['reference_tokens_by_domain'].values()) == {10}


def test_quality_tie_truncated_prefix_bias_is_removed(tmp_path):
    parent, population = population_file(tmp_path, n=10000)
    sample = tmp_path / 'sample.parquet'
    select(parent, population, sample, target=500, batch_size=4096)
    table = pq.read_table(sample).to_pylist()
    web_ranks = [int.from_bytes(row['doc_id'], 'big') // 9 for row in table if row['domain'] == 'web']
    # In this selected population, source B survived only at ranks 0..999.
    # A prefix of 500 would contain 100% B; its full-population share is 10%.
    fraction_b = sum(rank < 1000 for rank in web_ranks) / len(web_ranks)
    assert .06 < fraction_b < .14
    assert max(web_ranks) > 9500


@pytest.mark.parametrize('field', ['selected_docs', 'selected_tokens'])
def test_manifest_population_mismatch_fails_closed(tmp_path, field):
    parent, population = population_file(tmp_path)
    population['web'][field] += 1
    with pytest.raises(ValueError, match='counts differ'):
        select(parent, population, tmp_path / 'sample.parquet')
    assert not (tmp_path / 'sample.parquet').exists()


def test_wrong_sha_missing_domain_and_capacity_are_rejected(tmp_path):
    parent, population = population_file(tmp_path)
    with pytest.raises(ValueError, match='SHA-256'):
        sampling.sample_order(parent, 10, tmp_path / 'out', population=population,
            expected_parent_sha256='0'*64, seed='test', phase='warmup')
    with pytest.raises(ValueError, match='insufficient'):
        select(parent, population, tmp_path / 'out', target=33)
    del population['web']
    with pytest.raises(ValueError, match='domain roster'):
        select(parent, population, tmp_path / 'out')


def test_hash_and_parquet_read_use_same_open_file_on_path_rollback(tmp_path, monkeypatch):
    parent, population = population_file(tmp_path)
    alternate, saved = tmp_path / 'alternate', tmp_path / 'saved'
    pq.write_table(pa.table({'doc_id': [b'wrong'], 'domain': ['web'], 'n_tokens': [1]}), alternate)
    original = sampling.pq.ParquetFile
    def rollback(handle, *args, **kwargs):
        assert hasattr(handle, 'fileno')
        parent.rename(saved)
        alternate.rename(parent)
        try:
            return original(handle, *args, **kwargs)
        finally:
            parent.rename(alternate)
            saved.rename(parent)
    monkeypatch.setattr(sampling.pq, 'ParquetFile', rollback)
    select(parent, population, tmp_path / 'out')


def test_rank_draws_bind_seed_phase_domain_and_have_no_duplicates():
    values = [sampling.draw_ranks(10000, 200, seed=seed, phase=phase, domain=domain)
              for seed, phase, domain in [('a', 'warmup', 'web'), ('b', 'warmup', 'web'),
                                         ('a', 'stable', 'web'), ('a', 'warmup', 'code')]]
    assert all(len(np.unique(ranks)) == 200 and set(priorities) == set(range(200))
               for ranks, priorities in values)
    assert all(not np.array_equal(values[0][0], value[0]) for value in values[1:])
