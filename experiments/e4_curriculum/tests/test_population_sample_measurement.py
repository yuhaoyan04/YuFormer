from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from measure_uniform_sample_from_population import selected_positions


def test_uniform_sample_matches_its_exact_population_document_positions(tmp_path):
    parent = pa.table({'doc_id': [f'{i:032x}' for i in range(100)],
                       'domain': ['web']*100, 'n_tokens': list(range(1, 101))})
    path = tmp_path/'parent.parquet'
    pq.write_table(parent, path, row_group_size=7)
    expected = [0, 11, 43, 99]
    sample = parent.take(pa.array(expected))
    with path.open('rb') as handle:
        positions, rows = selected_positions(handle, sample)
    np.testing.assert_array_equal(positions, expected)
    assert rows == 100


@pytest.mark.parametrize('field,value', [('domain', 'wrong'), ('n_tokens', 999),
                                       ('doc_id', f'{999:032x}')])
def test_foreign_ids_or_mismatched_domain_reference_are_rejected(tmp_path, field, value):
    row = {'doc_id': f'{10:032x}', 'domain': 'web', 'n_tokens': 10}
    parent = pa.Table.from_pylist([row])
    path = tmp_path/'parent.parquet'
    pq.write_table(parent, path)
    row[field] = value
    with path.open('rb') as handle, pytest.raises(ValueError):
        selected_positions(handle, pa.Table.from_pylist([row]))
