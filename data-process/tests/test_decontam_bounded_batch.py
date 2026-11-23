from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from yuformer_data.scale import decontam as dct


class RecordingIndex:
    names = ["bench"]

    def __init__(self) -> None:
        self.query_sizes: list[int] = []

    def hit_codes_batch(self, shingles: np.ndarray) -> np.ndarray:
        self.query_sizes.append(len(shingles))
        return np.full(len(shingles), -1, dtype=np.int16)


def test_full_scan_index_queries_are_memory_bounded(monkeypatch):
    monkeypatch.setattr(dct, "_MAX_QUERY_SHINGLES", 100)
    index = RecordingIndex()
    texts = [" ".join(f"word{i}_{j}" for j in range(80)) for i in range(5)]
    verdicts, truncated, short = dct._decide_batch(texts, index, 8, None)
    assert verdicts == [(False, "", False)] * len(texts)
    assert truncated == 0
    assert short == 0
    # One document may exceed the budget, but multiple documents are never
    # accumulated past it.
    assert index.query_sizes == [73] * 5
