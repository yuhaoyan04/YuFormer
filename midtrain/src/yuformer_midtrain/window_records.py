"""Exact source windows and a global sample cursor for context extension.

Index rows are (source_id, token_offset, short_replay). Each row names S+1
source tokens independently: flattening these records into a stride-S stream
would silently move every subsequent document boundary.
"""
from __future__ import annotations

import numpy as np


class WindowRecords:
    def __init__(self, sources, index, *, seq_len, identity):
        if type(seq_len) is not int or seq_len <= 0:
            raise ValueError('window sequence length must be positive')
        if (not isinstance(identity, str) or len(identity) != 64
                or any(c not in '0123456789abcdef' for c in identity)):
            raise ValueError('window records need their immutable manifest SHA-256')
        self.sources = tuple(np.asarray(value) for value in sources)
        if not self.sources or any(a.ndim != 1 or a.dtype.kind != 'u' for a in self.sources):
            raise ValueError('window sources must be one-dimensional unsigned token arrays')
        self.index = np.asarray(index)
        if (self.index.ndim != 2 or self.index.shape[1] != 3
                or self.index.dtype != np.dtype('<u8') or not len(self.index)
                or len(self.index) % 8):
            raise ValueError('window index must contain whole groups of eight uint64 triples')
        source_ids, starts, replay = self.index.T
        if np.any(source_ids >= len(self.sources)) or np.any(replay > 1):
            raise ValueError('window index has an unknown source or replay kind')
        if not np.all(replay.reshape(-1, 8).sum(axis=1) == 2):
            raise ValueError('every eight windows must contain exactly 25% short replay')
        for i, array in enumerate(self.sources):
            positions = starts[source_ids == i]
            # Subtract before comparing, so an overflowing uint64 offset cannot
            # wrap around and pass the capacity check.
            if len(array) < seq_len + 1 or np.any(positions > len(array) - seq_len - 1):
                raise ValueError('window extends beyond its source token capacity')
        self.seq_len, self.identity = seq_len, identity

    def sample(self, ordinal):
        if type(ordinal) is not int or not 0 <= ordinal < len(self.index):
            raise IndexError('window record budget exhausted')
        source, offset, _ = map(int, self.index[ordinal])
        window = np.asarray(self.sources[source][offset:offset+self.seq_len+1], dtype=np.int64)
        return window[:-1], window[1:]


class WindowCursor:
    """Checkpoint global consumed samples, never a topology-dependent rank offset."""

    def __init__(self, records, *, world_size, rank, local_batch_size, samples=None):
        if (any(type(v) is not int or v <= 0 for v in (world_size, local_batch_size))
                or type(rank) is not int or not 0 <= rank < world_size):
            raise ValueError('invalid data parallel geometry')
        self.records = records
        self.world_size, self.rank, self.local_batch_size = world_size, rank, local_batch_size
        self.global_batch = world_size * local_batch_size
        self.capacity = len(records.index) if samples is None else samples
        if (type(self.capacity) is not int or not 0 < self.capacity <= len(records.index)
                or self.capacity % self.global_batch or self.global_batch % 8):
            raise ValueError('sample budget must close whole global batches and replay groups')
        self.consumed = 0

    def next_batch(self):
        if self.consumed + self.global_batch > self.capacity:
            raise StopIteration('window record budget exhausted')
        first = self.consumed + self.rank * self.local_batch_size
        rows = [self.records.sample(first+i) for i in range(self.local_batch_size)]
        self.consumed += self.global_batch
        return np.stack([row[0] for row in rows]), np.stack([row[1] for row in rows])

    def state_dict(self):
        return {'consumed_samples': self.consumed, 'seq_len': self.records.seq_len,
                'window_manifest_sha256': self.records.identity, 'sample_budget': self.capacity}

    def load_state_dict(self, state):
        if not state:
            return
        for key in ('seq_len', 'window_manifest_sha256', 'sample_budget'):
            if type(state.get(key)) is not type(self.state_dict()[key]) or state[key] != self.state_dict()[key]:
                raise ValueError('cannot restore a different long-stage dataset, length or budget')
        consumed = state.get('consumed_samples')
        if (type(consumed) is not int or not 0 <= consumed <= self.capacity
                or consumed % self.global_batch):
            raise ValueError('checkpoint cursor does not close the resumed global batch')
        self.consumed = consumed
