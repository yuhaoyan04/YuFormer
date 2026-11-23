import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import publish_training_block_manifest as publisher


@pytest.fixture
def prepared(tmp_path):
    path=tmp_path/'stable.blocks.u32'
    np.array([2,0,3,1],dtype='<u4').tofile(path)
    return tmp_path,{'phase':'stable','permutation_population_samples':16,
        'permutation':{'file':path.name,'bytes':16,'dtype':'<u4',
                       'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}


def test_window_projection_preserves_original_and_encodes_absolute_physical_blocks(prepared):
    directory,plan=prepared;original=(directory/plan['permutation']['file']).read_bytes()
    result=publisher.shift_prepared_permutation(directory,plan,source_sample_offset=4,block_samples=4)
    assert (directory/plan['permutation']['file']).read_bytes()==original
    assert np.fromfile(directory/result['file'],dtype='<u4').tolist()==[3,1,4,2]
    assert result['sha256']==hashlib.sha256((directory/result['file']).read_bytes()).hexdigest()
    assert publisher.shift_prepared_permutation(directory,plan,source_sample_offset=4,block_samples=4)==result


def test_zero_window_reuses_the_exact_original_artifact(prepared):
    directory,plan=prepared
    assert publisher.shift_prepared_permutation(directory,plan,source_sample_offset=0,block_samples=4)==plan['permutation']


@pytest.mark.parametrize('offset',[-4,2,True,2**32*4])
def test_invalid_alignment_or_unsigned_overflow_cannot_publish_indices(prepared,offset):
    directory,plan=prepared
    with pytest.raises(ValueError):
        publisher.shift_prepared_permutation(directory,plan,source_sample_offset=offset,block_samples=4)


def test_duplicate_base_blocks_are_rejected_even_with_a_matching_hash(prepared):
    directory,plan=prepared;path=directory/plan['permutation']['file']
    np.array([0,1,1,3],dtype='<u4').tofile(path)
    plan['permutation']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='bijection'):
        publisher.shift_prepared_permutation(directory,plan,source_sample_offset=4,block_samples=4)


def test_a_different_existing_window_artifact_is_never_overwritten(prepared):
    directory,plan=prepared;destination=directory/'stable.window-4.blocks.u32'
    destination.write_bytes(b'wrong immutable contents')
    with pytest.raises(ValueError,match='overwrite'):
        publisher.shift_prepared_permutation(directory,plan,source_sample_offset=4,block_samples=4)
    assert destination.read_bytes()==b'wrong immutable contents'
