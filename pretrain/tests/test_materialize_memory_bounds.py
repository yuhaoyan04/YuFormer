import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import materialize_tokens as materializer


def test_part_budget_uses_selected_document_counts_in_scan_order():
    reference = np.array([100, 1, 2, 3, 4, 5])
    positions = np.array([5, 4, 0, 1, 3, 2])
    boundary = np.array([0, 2, 4, 6])
    chunks = materializer.token_bounded_chunks(reference, positions, boundary,
                                               max_tokens=20, max_docs=4)
    assert chunks == [(0, 0, 1), (1, 1, 2), (2, 2, 3)]
    assert [group for _, start, stop in chunks for group in range(start, stop)] == [0, 1, 2]
    # The oversized row group remains alone; its Encoding objects are batched.
    assert sum(reference[positions[2:4]]) > 20


def test_part_budget_bounds_small_documents_and_row_group_count():
    positions = np.arange(12)
    reference = np.ones(12, dtype=np.int64)
    boundary = np.arange(0, 13, 2)
    chunks = materializer.token_bounded_chunks(reference, positions, boundary,
                                               max_tokens=100, max_docs=4, max_groups=2)
    assert chunks == [(0, 0, 2), (1, 2, 4), (2, 4, 6)]


def test_worker_encoding_batches_preserve_exact_tokens_eod_and_positions(tmp_path, monkeypatch):
    import tokenizers
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    tokenizer_path = Path(__file__).resolve().parents[2] / "tokenizer/tokenizers/yf-32k-bpe/tokenizer.json"
    tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
    eod = tokenizer.token_to_id("<|endoftext|>")
    texts = ["alpha beta gamma", "x" * 70, "你好世界" * 4, "def f(x): return x"]
    positions = np.array([3, 1, 2, 0], dtype=np.int64)
    source = tmp_path / "source.parquet"
    doc_ids = [f"{i:032x}" for i in range(4)]
    pq.write_table(pa.table({"doc_id": doc_ids, "text": texts}), source, row_group_size=2)
    expected_ids = np.empty(4, dtype="V16")
    expected_ids[positions] = materializer.hex_to_v16(doc_ids)
    for key, value in {"sfi": np.zeros(4, dtype=np.int64), "srg": np.array([0, 0, 1, 1]),
                       "srow": np.arange(4), "spos": positions,
                       "boundary": np.array([0, 2, 4]), "expected_ids": expected_ids}.items():
        np.save(tmp_path / f"{key}.npy", value)
    (tmp_path / "files.json").write_text(json.dumps([str(source)]))
    (tmp_path / "file_identities.json").write_text(json.dumps([materializer.source_file_identity(source)]))
    (tmp_path / "tokenizer_sha256.txt").write_text(materializer.file_sha256(tokenizer_path))
    calls = []

    def encode_batch(batch, **kwargs):
        calls.append(list(batch))
        assert len(batch) <= 2
        assert sum(map(len, batch)) <= 40 or len(batch) == 1
        return tokenizer.encode_batch(batch, **kwargs)

    monkeypatch.setattr(tokenizers, "Tokenizer", SimpleNamespace(
        from_str=lambda _: SimpleNamespace(encode_batch=encode_batch)))
    monkeypatch.setattr(materializer, "ENCODE_BATCH_CHAR_LIMIT", 40)
    monkeypatch.setattr(materializer, "ENCODE_BATCH_DOCUMENT_LIMIT", 2)
    materializer._pass2a_chunk((0, 0, 2, str(tmp_path), str(tokenizer_path), str(tmp_path), "<u2", eod))
    expected = np.concatenate([np.asarray(tokenizer.encode(text, add_special_tokens=False).ids + [eod], dtype="<u2")
                               for _, text in sorted(zip(positions, texts))])
    with np.load(tmp_path / "part000000.npz") as part:
        np.testing.assert_array_equal(part["positions"], np.arange(4))
        np.testing.assert_array_equal(part["tokens"], expected)
        assert len(part["offsets"]) == 5
        assert int(part["offsets"][-1]) == len(expected)
    assert len(calls) >= 2


def test_worker_thread_environment_restores_parent_settings(monkeypatch):
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "99")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "true")
    with materializer.single_thread_worker_environment():
        assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
        assert os.environ["TOKENIZERS_PARALLELISM"] == "false"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "99"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "true"


def test_native_merge_grows_part_table_under_low_fd_limit(tmp_path):
    source = Path(materializer.__file__).with_name("merge_token_parts.c")
    binary = tmp_path / "merge"
    subprocess.run(["cc", "-O2", "-std=c11", str(source), "-o", str(binary)], check=True)
    count, lines = 1050, []
    for index in range(count):
        position = count - index - 1
        paths = [tmp_path / f"{index}.{kind}" for kind in ("positions", "offsets", "tokens")]
        for path, data in zip(paths, (np.array([position], dtype="<u8"),
                                     np.array([0, 2], dtype="<u8"),
                                     np.array([position + 10, 0], dtype="<u2")), strict=True):
            data.tofile(path)
        lines.append("\t".join(map(str, paths)) + "\t1\t2")
    manifest, output, lengths = [tmp_path / name for name in ("manifest.tsv", "tokens.bin", "lengths.u32")]
    manifest.write_text("\n".join(lines) + "\n")
    program = "import os,resource,sys; resource.setrlimit(resource.RLIMIT_NOFILE,(16,16)); os.execv(sys.argv[1],sys.argv[1:])"
    result = subprocess.run([sys.executable, "-c", program, str(binary), str(manifest), str(output),
                             str(count), "2", str(lengths)], capture_output=True, text=True,
                            env={**os.environ, "MALLOC_PERTURB_": "165"}, timeout=30)
    assert result.returncode == 0, result.stderr
    expected = np.column_stack([np.arange(count) + 10, np.zeros(count)]).astype("<u2").ravel()
    np.testing.assert_array_equal(np.fromfile(output, dtype="<u2"), expected)
    np.testing.assert_array_equal(np.fromfile(lengths, dtype="<u4"), np.full(count, 2))
