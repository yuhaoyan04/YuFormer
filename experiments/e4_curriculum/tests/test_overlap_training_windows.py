from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from overlap_training_windows import consumption_cache, resolve_windows


def test_shifted_full_overlap_uses_exact_same_record_identity_as_actual_admission(tmp_path):
    path, identity = consumption_cache(tmp_path, 'stable', 'order:ledger:195312640:4096', 2015232)
    assert path.name == 'stable.window-2015232.consumed.records'
    assert identity == 'order:ledger:195312640:4096:source_sample_offset=2015232'
    assert path != consumption_cache(tmp_path, 'stable', 'order:ledger:195312640:4096', 0)[0]


def test_legacy_zero_window_preserves_actual_original_prefix_cache(tmp_path):
    path, identity = consumption_cache(tmp_path, 'warmup', 'unchanged', 0)
    assert path == tmp_path/'warmup.consumed.records' and identity == 'unchanged'
    value = resolve_windows(tmp_path, {'data':{}}, {}, 'not-used-in-legacy-prefix-mode')
    assert value['source_sample_offsets'] == [0,0,0]


@pytest.mark.parametrize('offset', [True,-1,.5])
def test_implicit_or_invalid_source_offsets_are_never_coerced(tmp_path, offset):
    with pytest.raises(ValueError):
        consumption_cache(tmp_path, 'stable', 'identity', offset)


def test_sequential_mode_cannot_hide_a_production_order(tmp_path):
    with pytest.raises(ValueError, match='cannot hide'):
        resolve_windows(tmp_path, {'data':{'block_permutation_manifest':'manifest.json'}}, {}, 'unused')


def test_production_window_audit_requires_a_real_explicit_environment(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_windows(tmp_path, {'data':{'sampling_policy':'block_permutation'}}, {}, 'unused')
