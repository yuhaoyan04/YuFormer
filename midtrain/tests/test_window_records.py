import copy

import numpy as np
import pytest

from yuformer_midtrain.window_records import WindowRecords, WindowCursor


def records():
    source = np.arange(1024, dtype='<u4')
    # Distinct S+1 records deliberately leave gaps in the underlying source.
    index = np.array([(0, i*11, int(i % 8 in (1, 6))) for i in range(32)], dtype='<u8')
    return WindowRecords([source], index, seq_len=8, identity='a'*64)


def test_records_preserve_exact_input_target_shift_across_document_gaps():
    data = records()
    for i in range(32):
        x, y = data.sample(i)
        assert x.tolist() == list(range(i*11, i*11+8))
        assert y.tolist() == list(range(i*11+1, i*11+9))


def test_all_eight_ranks_and_resume_consume_each_window_once():
    data = records()
    ranks = [WindowCursor(data, world_size=8, rank=i, local_batch_size=1) for i in range(8)]
    seen = []
    for _ in range(2):
        seen.extend(int(cursor.next_batch()[0][0, 0])//11 for cursor in ranks)
    state = ranks[0].state_dict()
    # Resharding changes ownership, not the global next sample.
    resumed = [WindowCursor(data, world_size=4, rank=i, local_batch_size=2) for i in range(4)]
    for cursor in resumed:
        cursor.load_state_dict(state)
    for _ in range(2):
        for cursor in resumed:
            seen.extend((cursor.next_batch()[0][:, 0]//11).tolist())
    assert seen == list(range(32))
    with pytest.raises(StopIteration):
        resumed[0].next_batch()


@pytest.mark.parametrize('fault', ['manifest', 'length', 'budget', 'unaligned', 'boolean', 'past_end'])
def test_resume_rejects_other_stage_and_invalid_cursor(fault):
    cursor = WindowCursor(records(), world_size=8, rank=0, local_batch_size=1)
    state = copy.deepcopy(cursor.state_dict())
    if fault == 'manifest': state['window_manifest_sha256'] = 'b'*64
    elif fault == 'length': state['seq_len'] = 16
    elif fault == 'budget': state['sample_budget'] = 16
    elif fault == 'unaligned': state['consumed_samples'] = 1
    elif fault == 'boolean': state['consumed_samples'] = False
    else: state['consumed_samples'] = 40
    with pytest.raises(ValueError):
        cursor.load_state_dict(state)


@pytest.mark.parametrize('fault', ['replay', 'source', 'overflow'])
def test_invalid_window_index_cannot_enter_training(fault):
    data = records()
    index = data.index.copy()
    if fault == 'replay': index[0, 2] = 1
    elif fault == 'source': index[0, 0] = 1
    else: index[0, 1] = np.iinfo(np.uint64).max
    with pytest.raises(ValueError):
        WindowRecords(data.sources, index, seq_len=8, identity='a'*64)
