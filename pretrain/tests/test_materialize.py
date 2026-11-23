"""materialize_tokens.py 契约测试：s3 正文 + order 表 → 定序 token 流。

用仓库自带 yf-32k-bpe tokenizer 与合成 s3/order parquet（doc_id 用 S6 同款
make_doc_id 公式），验证：顺序表 position 序 = 输出流序、meta 哈希可被
pretrain TokenStream 消费、val 域物化。
"""
import json
import hashlib
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "data-process" / "src"))
sys.path.insert(0, str(REPO_ROOT / "pretrain" / "scripts"))

from yuformer_data.scale.schema import CANONICAL_SCHEMA, make_doc_id  # noqa: E402

from materialize_tokens import (  # noqa: E402
    ShardTextFetcher,
    lookup_sorted_doc_ids,
    materialize_stream,
    merge_parts_by_position,
)

TOKENIZER = REPO_ROOT / "tokenizer" / "tokenizers" / "yf-32k-bpe" / "tokenizer.json"

TEXTS = {
    "alpha": "The quick brown fox jumps over the lazy dog. " * 8,
    "beta": "数学 是 研究 数量、结构、变化 的 学科。" * 8,
    "gamma": "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n" * 4,
    "delta": "Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 6,
}


def test_sorted_index_lookup_restores_hash_shuffled_query_order():
    index_ids = np.asarray(
        [bytes.fromhex(f"{value:032x}") for value in (1, 3, 7, 9)], dtype="V16"
    )
    query_ids = np.asarray(
        [bytes.fromhex(f"{value:032x}") for value in (9, 2, 1, 7)], dtype="V16"
    )
    positions, hits = lookup_sorted_doc_ids(
        index_ids, query_ids, index_chunk_rows=2, direct_query_threshold=0
    )
    assert positions.tolist() == [3, 1, 0, 2]
    assert hits.tolist() == [True, False, True, True]


def test_sorted_index_lookup_sparse_path_handles_missing_tail():
    index_ids = np.asarray(
        [bytes.fromhex(f"{value:032x}") for value in (1, 3, 7, 9)], dtype="V16"
    )
    query_ids = np.asarray(
        [bytes.fromhex(f"{value:032x}") for value in (9, 1, 12)], dtype="V16"
    )
    positions, hits = lookup_sorted_doc_ids(
        index_ids, query_ids, index_chunk_rows=1, direct_query_threshold=3
    )
    assert positions.tolist() == [3, 0, 4]
    assert hits.tolist() == [True, True, False]


def test_arrow_part_merge_restores_order_without_global_gather(tmp_path, monkeypatch):
    import pyarrow.compute as pc

    part0 = tmp_path / "part000000.npz"
    part1 = tmp_path / "part000001.npz"
    np.savez(
        part0,
        positions=np.asarray([1, 3], dtype=np.int64),
        offsets=np.asarray([0, 2, 3], dtype=np.int64),
        tokens=np.asarray([10, 11, 30], dtype="<u2"),
    )
    np.savez(
        part1,
        positions=np.asarray([0, 2], dtype=np.int64),
        offsets=np.asarray([0, 1, 4], dtype=np.int64),
        tokens=np.asarray([0, 20, 21, 22], dtype="<u2"),
    )
    output = tmp_path / "merged.bin"
    lengths = tmp_path / "merged.doc_lengths.u32"
    take_input_docs = []
    real_take = pc.take

    def bounded_take(values, indices, *args, **kwargs):
        take_input_docs.append(len(values))
        return real_take(values, indices, *args, **kwargs)

    monkeypatch.setattr(pc, "take", bounded_take)
    total = merge_parts_by_position(
        [part0, part1], output, expected_docs=4, tok_dtype="<u2", chunk_docs=2,
        doc_lengths_path=lengths,
    )
    assert total == 7
    assert np.fromfile(output, dtype="<u2").tolist() == [0, 10, 11, 20, 21, 22, 30]
    assert np.fromfile(lengths, dtype="<u4").tolist() == [1, 2, 3, 1]
    assert take_input_docs
    assert max(take_input_docs) <= 2


def test_external_part_merge_restores_order_with_bounded_mmaps(tmp_path):
    part0 = tmp_path / "part000000.npz"
    part1 = tmp_path / "part000001.npz"
    np.savez(part0, positions=np.asarray([1, 3], dtype=np.int64),
             offsets=np.asarray([0, 2, 3], dtype=np.int64),
             tokens=np.asarray([10, 11, 30], dtype="<u2"))
    np.savez(part1, positions=np.asarray([0, 2], dtype=np.int64),
             offsets=np.asarray([0, 1, 4], dtype=np.int64),
             tokens=np.asarray([0, 20, 21, 22], dtype="<u2"))
    output = tmp_path / "external.bin"
    lengths = tmp_path / "external.doc_lengths.u32"
    total = merge_parts_by_position(
        [part0, part1], output, expected_docs=4, tok_dtype="<u2",
        external_threshold_bytes=0,
        doc_lengths_path=lengths,
    )
    assert total == 7
    assert np.fromfile(output, dtype="<u2").tolist() == [0, 10, 11, 20, 21, 22, 30]
    assert np.fromfile(lengths, dtype="<u4").tolist() == [1, 2, 3, 1]
    assert not (tmp_path / f".{output.name}.merge-parts").exists()


def test_text_fetcher_takes_only_requested_rows_and_preserves_order(tmp_path):
    path = tmp_path / "rows.parquet"
    pq.write_table(
        pa.table({"text": [f"row-{i}" for i in range(10)]}),
        path,
        row_group_size=3,
    )
    fetcher = ShardTextFetcher([path])
    rows = np.asarray([8, 1, 6, 2], dtype=np.int64)
    assert fetcher.fetch(0, rows) == ["row-8", "row-1", "row-6", "row-2"]


def _write_s3(tmp_path):
    s3 = tmp_path / "s3" / "fineweb"
    s3.mkdir(parents=True)
    rows = {"doc_id": [], "text": [], "domain": [], "source": [], "lang": [],
            "upstream_score": [], "meta": []}
    docs = []
    for i, (name, text) in enumerate(TEXTS.items()):
        did = make_doc_id("fineweb", "shard-0000", i)
        docs.append((did, text))
        rows["doc_id"].append(did)
        rows["text"].append(text)
        rows["domain"].append("web")
        rows["source"].append("fineweb")
        rows["lang"].append("en")
        rows["upstream_score"].append(0.5)
        rows["meta"].append("{}")
    table = pa.table(rows, schema=CANONICAL_SCHEMA)
    pq.write_table(table, s3 / "shard-0000.parquet")
    return docs


def _write_order(tmp_path, docs, order):
    order_schema = pa.schema([
        pa.field("position", pa.int64(), nullable=False),
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("domain", pa.string(), nullable=False),
        pa.field("n_tokens", pa.uint32(), nullable=False),
        pa.field("upstream_score", pa.float64(), nullable=True),
    ])
    rows = {
        "position": list(range(len(order))),
        "doc_id": [docs[i][0] for i in order],
        "domain": ["web"] * len(order),
        "n_tokens": [0] * len(order),
        "upstream_score": [None] * len(order),
    }
    p = tmp_path / "mix_order" / "mix_test_stage2.parquet"
    p.parent.mkdir(exist_ok=True)
    pq.write_table(pa.table(rows, schema=order_schema), p)
    return p


def _expected_tokens(docs, order):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKENIZER))
    enc = tok.encode_batch([docs[i][1] for i in order], add_special_tokens=False)
    eod_id = tok.token_to_id("<|endoftext|>")
    return np.concatenate([
        np.concatenate([np.asarray(e.ids, dtype="<u2"), np.asarray([eod_id], dtype="<u2")])
        for e in enc
    ])


def test_materialize_preserves_order(tmp_path):
    docs = _write_s3(tmp_path)
    order = [2, 0, 3, 1]  # 非 s3 行序，验证按 position 表消费
    order_path = _write_order(tmp_path, docs, order)
    out = tmp_path / "streams"
    bin_path = materialize_stream(
        order_path, tmp_path / "s3", TOKENIZER, out, "mix_test", write_doc_lengths=True
    )
    got = np.fromfile(bin_path, dtype="<u2")
    want = _expected_tokens(docs, order)
    assert np.array_equal(got, want)

    meta = json.loads((out / "mix_test.meta.json").read_text())
    assert meta["schema_version"] == 3
    assert meta["total_tokens"] == len(want)
    assert meta["tokenizer_sha256"]
    assert meta["document_boundary"] == {
        "policy": "append_eod",
        "token": "<|endoftext|>",
        "token_id": 0,
        "count": len(order),
    }
    assert meta["selection"]["selected_reference_tokens"] == 0
    assert meta["selection"]["actual_tokenizer_text_tokens"] == len(want) - len(order)
    assert meta["selection"]["reference_tokens_are_selection_only"] is True
    lengths = np.fromfile(out / "mix_test.doc_lengths.u32", dtype="<u4")
    assert lengths.tolist() == [len(x) for x in np.split(want, np.flatnonzero(want == 0) + 1) if len(x)]
    assert meta["doc_lengths"]["sum_tokens"] == len(want)
    prep_files = list((out / ".prep_cache").glob("*.npz"))
    assert len(prep_files) == 1
    with np.load(prep_files[0]) as cached:
        assert len(cached["spos"]) == len(order)

    from yuformer_pretrain.data import TokenStream

    s = TokenStream.open(bin_path, out / "mix_test.meta.json", seq_len=63)
    assert s.samples == len(want) // 64


def test_materialize_missing_doc_fails(tmp_path):
    docs = _write_s3(tmp_path)
    ghost = make_doc_id("fineweb", "shard-9999", 0)
    order_path = _write_order(tmp_path, docs + [(ghost, "x")], [0, 4, 1])
    try:
        materialize_stream(order_path, tmp_path / "s3", TOKENIZER, tmp_path / "streams", "bad")
    except SystemExit as e:
        assert "不存在" in str(e)
    else:
        raise AssertionError("missing doc_id 应 fail-fast")


def test_explicit_workers_override_environment(tmp_path, monkeypatch):
    """The documented --workers path must not be shadowed by the environment."""
    monkeypatch.setenv("YUFORMER_MAT_WORKERS", "not-an-integer")
    docs = _write_s3(tmp_path)
    order = [1, 0]
    order_path = _write_order(tmp_path, docs, order)
    out = tmp_path / "streams"
    bin_path = materialize_stream(
        order_path,
        tmp_path / "s3",
        TOKENIZER,
        out,
        "workers",
        workers=1,
    )
    got = np.fromfile(bin_path, dtype="<u2")
    assert np.array_equal(got, _expected_tokens(docs, order))


def test_materialize_binds_order_to_snapshot_artifact(tmp_path):
    docs = _write_s3(tmp_path)
    order_path = _write_order(tmp_path, docs, [0, 1])
    payload = order_path.read_bytes()
    snapshot = {
        "snapshot_id": "yuformer-snapshot-test",
        "release_name": "test",
        "artifacts": [{
            "path": order_path.resolve().as_posix(),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }],
    }
    out = tmp_path / "streams"
    materialize_stream(
        order_path, tmp_path / "s3", TOKENIZER, out, "bound",
        data_snapshot=snapshot, workers=1,
    )
    meta = json.loads((out / "bound.meta.json").read_text())
    assert meta["data_snapshot_id"] == "yuformer-snapshot-test"
    assert meta["order_sha256"] == snapshot["artifacts"][0]["sha256"]

    snapshot["artifacts"][0]["sha256"] = "0" * 64
    try:
        materialize_stream(
            order_path, tmp_path / "s3", TOKENIZER, out, "tampered",
            data_snapshot=snapshot, workers=1,
        )
    except SystemExit as exc:
        assert "SHA-256" in str(exc)
    else:
        raise AssertionError("snapshot SHA mismatch 应 fail-fast")
