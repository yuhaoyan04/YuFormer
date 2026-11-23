from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "materialize_tokenizer_eval_corpus.py"
SPEC = importlib.util.spec_from_file_location("materialize_tokenizer_eval_corpus", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_select_prefix_reaches_budget() -> None:
    assert MODULE.select_prefix(["a", "b", "c"], np.asarray([3, 5, 7]), 8) == 2
    assert MODULE.select_prefix(["a", "b"], np.asarray([3, 5]), 99) == 2


def test_split_utf8_is_lossless_and_bounded() -> None:
    text = "中文🙂abc" * 100
    parts = MODULE.split_utf8(text, 17)
    assert "".join(value for value, _ in parts) == text
    assert sum(size for _, size in parts) == len(text.encode("utf-8"))
    assert all(size <= 17 for _, size in parts)


def test_lookup_sorted_doc_ids_restores_query_order() -> None:
    raw = [bytes.fromhex(f"{value:032x}") for value in (1, 3, 8, 10)]
    index = np.frombuffer(b"".join(raw), dtype="V16")
    query_raw = [bytes.fromhex(f"{value:032x}") for value in (10, 1, 9)]
    query = np.frombuffer(b"".join(query_raw), dtype="V16")
    positions, hits = MODULE.lookup_sorted_doc_ids(index, query, chunk_rows=2)
    assert positions.tolist() == [3, 0, 3]
    assert hits.tolist() == [True, True, False]
