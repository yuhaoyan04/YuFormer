import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(os.environ.get('YUFORMER_ROOT', Path(__file__).resolve().parents[2]))
SOURCE = Path(os.environ.get('YUFORMER_TEST_MATERIALIZE_SOURCE',
                            ROOT / 'data-process/src/yuformer_data/scale/materialize.py'))
sys.path.insert(0, str(ROOT / 'data-process/src'))
SPEC = importlib.util.spec_from_file_location('yuformer_data.scale.materialize_fd_test', SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_parts(directory, count=64):
    paths = []
    for index in range(count):
        path = directory / f'source{index % 3}__{index:04d}.npz'
        np.savez(path, ids=np.frombuffer(index.to_bytes(16, 'big'), dtype='V16'),
                 comp=np.array([index/100], dtype=np.float32),
                 nt=np.array([index+1], dtype=np.uint32),
                 up=np.array([index/10], dtype=np.float64))
        paths.append(path)
    return sorted(paths)


def test_merge_is_identical_and_leaves_no_candidate_handles(tmp_path):
    parts = make_parts(tmp_path)
    expected = {key: [] for key in ('ids', 'comp', 'nt', 'up')}
    source_keys = sorted({p.name.split('__')[0] for p in parts})
    source_indices = []
    for path in parts:
        with np.load(path) as archive:
            for key in expected:
                expected[key].append(archive[key])
            source_indices.append(source_keys.index(path.name.split('__')[0]))
    before = len(list(Path('/proc/self/fd').iterdir()))
    for _ in range(3):
        ids, comp, nt, up, indices, keys = MODULE.load_candidate_parts(parts, len(parts))
        for key, actual in zip(expected, (ids, comp, nt, up)):
            np.testing.assert_array_equal(actual, np.concatenate(expected[key]))
        np.testing.assert_array_equal(indices, np.asarray(source_indices, dtype=np.uint16))
        assert keys == source_keys
        assert len(list(Path('/proc/self/fd').iterdir())) <= before


def test_many_parts_pass_under_32_fd_limit_and_old_open_all_fails(tmp_path):
    make_parts(tmp_path, 96)
    script = '''
import importlib.util, pathlib, resource, sys
import numpy as np
source, root, directory = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(root / 'data-process/src'))
spec = importlib.util.spec_from_file_location('yuformer_data.scale.fd_subprocess', source)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
parts = sorted(directory.glob('*.npz'))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
opened = []
try:
    for path in parts:
        opened.append(np.load(path))
except OSError as error:
    assert error.errno == 24, error
else:
    raise AssertionError('old all-open pattern unexpectedly fit 96 files into 32 FDs')
finally:
    for archive in opened: archive.close()
result = module.load_candidate_parts(parts, len(parts))
assert len(result[0]) == 96
assert len(list(pathlib.Path('/proc/self/fd').iterdir())) < 16
print('OLD_EMFILE_NEW_PASS')
'''
    result = subprocess.run([sys.executable, '-c', script, str(SOURCE), str(ROOT), str(tmp_path)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert 'OLD_EMFILE_NEW_PASS' in result.stdout


@pytest.mark.parametrize('fault', ['missing_column', 'wrong_dtype', 'short_column', 'extra_rows', 'missing_rows'])
def test_invalid_parts_fail_closed_and_release_handles(tmp_path, fault):
    paths = make_parts(tmp_path, 4)
    rows = 4
    if fault == 'extra_rows':
        rows = 3
    elif fault == 'missing_rows':
        rows = 5
    else:
        with np.load(paths[0]) as archive:
            values = {key: archive[key] for key in archive.files}
        if fault == 'missing_column':
            del values['nt']
        elif fault == 'wrong_dtype':
            values['nt'] = values['nt'].astype(np.int64)
        else:
            values['nt'] = np.empty(0, dtype=np.uint32)
        np.savez(paths[0], **values)
    before = len(list(Path('/proc/self/fd').iterdir()))
    with pytest.raises((ValueError, KeyError)):
        MODULE.load_candidate_parts(paths, rows)
    assert len(list(Path('/proc/self/fd').iterdir())) <= before


def test_empty_committed_population(tmp_path):
    arrays = MODULE.load_candidate_parts([], 0)
    assert all(len(array) == 0 for array in arrays)
    with pytest.raises(ValueError, match='cover'):
        MODULE.load_candidate_parts([], 1)
