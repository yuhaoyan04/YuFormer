import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sample_tokenizer_corpus.py"
SPEC = importlib.util.spec_from_file_location("tokenizer_corpus_sampler", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_select_hash_prefix_uses_whole_documents_and_domain_prefixes():
    batch = pa.RecordBatch.from_pydict(
        {
            "position": pa.array(range(6), type=pa.int64()),
            "doc_id": pa.array([f"{i:032x}" for i in range(6)]),
            "domain": pa.array(["web", "code", "web", "code", "web", "code"]),
            "n_tokens": pa.array([4, 3, 5, 8, 7, 2], type=pa.uint32()),
        }
    )
    ids, tokens, scanned = MODULE.select_hash_prefix([batch], {"web": 8, "code": 10})
    assert scanned == 6
    assert ids["web"] == [f"{i:032x}" for i in (0, 2)]
    assert tokens["web"] == [4, 5]
    assert ids["code"] == [f"{i:032x}" for i in (1, 3)]
    assert tokens["code"] == [3, 8]


def test_select_hash_prefix_rejects_non_contiguous_positions():
    batch = pa.RecordBatch.from_pydict(
        {
            "position": pa.array([0, 2], type=pa.int64()),
            "doc_id": pa.array(["0" * 32, "1" * 32]),
            "domain": pa.array(["web", "web"]),
            "n_tokens": pa.array([4, 4], type=pa.uint32()),
        }
    )
    with pytest.raises(ValueError, match="position"):
        MODULE.select_hash_prefix([batch], {"web": 4})


def test_parallel_fetch_preserves_deterministic_write_order(tmp_path, monkeypatch):
    doc_ids = [f"{i:032x}" for i in range(6)]
    index_ids = np.asarray([bytes.fromhex(value) for value in doc_ids], dtype="V16")
    index_fidx = np.asarray([1, 0, 1, 0, 2, 2], dtype=np.uint32)
    index_ridx = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.uint32)
    files = [tmp_path / f"part-{i}.parquet" for i in range(3)]

    monkeypatch.setattr(
        MODULE, "build_doc_index",
        lambda _root: (index_ids, index_fidx, index_ridx, files),
    )

    class FakeFetcher:
        def __init__(self, _files):
            pass

        def fetch(self, fi, rows):
            return [f"text-{fi}-{int(row)}" for row in rows]

    monkeypatch.setattr(MODULE, "ShardTextFetcher", FakeFetcher)
    selected_ids = {"code": [doc_ids[1], doc_ids[3]], "web": [doc_ids[0], doc_ids[2], doc_ids[4], doc_ids[5]]}
    selected_tokens = {"code": [3, 5], "web": [2, 4, 6, 7]}
    sequential = tmp_path / "sequential"
    parallel = tmp_path / "parallel"
    sequential.mkdir()
    parallel.mkdir()
    stats_one = MODULE.write_corpus(
        selected_ids=selected_ids, selected_tokens=selected_tokens,
        s3_root=tmp_path, staging=sequential, fetch_workers=1,
    )
    stats_many = MODULE.write_corpus(
        selected_ids=selected_ids, selected_tokens=selected_tokens,
        s3_root=tmp_path, staging=parallel, fetch_workers=3,
    )
    assert stats_one == stats_many
    for domain in selected_ids:
        left = pq.read_table(sequential / f"{domain}.parquet").to_pydict()
        right = pq.read_table(parallel / f"{domain}.parquet").to_pydict()
        assert left == right


def test_publish_uses_file_moves_and_manifest_last(tmp_path, monkeypatch):
    staging = tmp_path / ".corpus.tmp-1"
    destination = tmp_path / "corpus"
    staging.mkdir()
    (staging / "web.parquet").write_bytes(b"web")
    (staging / "code.parquet").write_bytes(b"code")
    (staging / "manifest.json").write_text('{"status":"PASS"}')
    real_replace = MODULE.os.replace
    moves = []

    def checked_replace(source, target):
        source = Path(source)
        assert source.is_file(), "directory rename is not NFS-safe"
        moves.append(source.name)
        return real_replace(source, target)

    monkeypatch.setattr(MODULE.os, "replace", checked_replace)
    MODULE.publish_corpus_directory(staging, destination)
    assert moves[-1] == "manifest.json"
    assert sorted(p.name for p in destination.iterdir()) == [
        "code.parquet", "manifest.json", "web.parquet"
    ]
    assert not staging.exists()
