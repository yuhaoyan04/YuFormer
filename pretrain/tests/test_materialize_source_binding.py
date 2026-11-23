import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import materialize_tokens as materializer
from source_document_binding import verified_source_document_binding


IDS = [f"{i:032x}" for i in (1, 2)]
TOKENIZER = Path(__file__).resolve().parents[2] / "tokenizer/tokenizers/yf-32k-bpe/tokenizer.json"


def write_source(path, *, reverse=False, words=("one", "two")):
    previous_mtime = path.stat().st_mtime_ns if path.exists() else None
    path.parent.mkdir(parents=True, exist_ok=True)
    order = [1, 0] if reverse else [0, 1]
    pq.write_table(pa.table({"doc_id": [IDS[i] for i in order],
                            "text": [words[i] for i in order]}), path,
                   compression=None, use_dictionary=False, row_group_size=1)
    if previous_mtime is not None and path.stat().st_mtime_ns <= previous_mtime:
        os.utime(path, ns=(path.stat().st_atime_ns, previous_mtime + 1_000_000))


def write_order(path):
    pq.write_table(pa.table({"doc_id": IDS, "position": [0, 1], "n_tokens": [1, 1]}), path)


def test_index_rebuilds_after_same_size_row_rewrite_and_preserves_old_generation(tmp_path):
    source = tmp_path / "src/shard.parquet"
    write_source(source)
    materializer.build_doc_index(tmp_path)
    old_ids, _, old_rows, _ = materializer.build_doc_index(tmp_path)
    assert isinstance(old_rows, np.memmap)
    old_values = old_rows.copy()
    size = source.stat().st_size
    write_source(source, reverse=True)
    assert source.stat().st_size == size
    ids, files, rows, paths = materializer.build_doc_index(tmp_path)
    positions, hits = materializer.lookup_sorted_doc_ids(ids, materializer.hex_to_v16([IDS[0]]))
    assert hits.all()
    position = int(positions[0])
    assert materializer.ShardTextFetcher(paths).fetch(int(files[position]), rows[position:position + 1]) == ["one"]
    assert np.array_equal(old_rows, old_values)
    assert np.array_equal(old_ids, ids)
    assert len(list((tmp_path / ".doc_index_cache").glob("index-v2-*.json"))) == 2


@pytest.mark.parametrize("damage", ["missing", "truncated", "wrong_shape"])
def test_index_cache_requires_complete_matching_arrays(tmp_path, damage):
    source = tmp_path / "src/shard.parquet"
    write_source(source)
    materializer.build_doc_index(tmp_path)
    cache = tmp_path / ".doc_index_cache"
    manifest = json.loads(next(cache.glob("index-v2-*.json")).read_text())
    path = cache / f"ridx-{manifest['fingerprint']}.npy"
    if damage == "missing":
        path.unlink()
    elif damage == "truncated":
        path.write_bytes(b"broken")
    else:
        np.save(path, np.asarray([0], dtype=np.uint32))
    ids, files, rows, _ = materializer.build_doc_index(tmp_path)
    assert len(ids) == len(files) == len(rows) == 2
    assert rows.tolist() == [0, 1]


def test_index_manifest_is_not_published_after_interrupted_array_write(tmp_path, monkeypatch):
    write_source(tmp_path / "src/shard.parquet")
    replace = materializer.os.replace
    interrupted = [False]

    def fail_once(source, target):
        if Path(target).name.startswith("ridx-") and not interrupted[0]:
            interrupted[0] = True
            raise OSError("injected interrupted index publication")
        return replace(source, target)

    monkeypatch.setattr(materializer.os, "replace", fail_once)
    with pytest.raises(OSError, match="interrupted index"):
        materializer.build_doc_index(tmp_path)
    assert not list((tmp_path / ".doc_index_cache").glob("index-v2-*.json"))
    assert materializer.build_doc_index(tmp_path)[2].tolist() == [0, 1]


def test_index_rejects_ambiguous_duplicate_document_ids(tmp_path):
    source = tmp_path / "src/shard.parquet"
    source.parent.mkdir(parents=True)
    pq.write_table(pa.table({"doc_id": [IDS[0], IDS[0]], "text": ["one", "two"]}), source)
    with pytest.raises(ValueError, match="duplicate doc_id"):
        materializer.build_doc_index(tmp_path)
    assert not list((tmp_path / ".doc_index_cache").glob("index-v2-*.json"))


def test_source_change_during_index_build_prevents_commit(tmp_path, monkeypatch):
    source = tmp_path / "src/shard.parquet"
    write_source(source)
    save = materializer.np.save
    changed = [False]

    def rewrite_after_array_write(path, array, *args, **kwargs):
        result = save(path, array, *args, **kwargs)
        if not changed[0]:
            changed[0] = True
            write_source(source, reverse=True)
        return result

    monkeypatch.setattr(materializer.np, "save", rewrite_after_array_write)
    with pytest.raises(ValueError, match="source changed while building document index"):
        materializer.build_doc_index(tmp_path)
    assert not list((tmp_path / ".doc_index_cache").glob("index-v2-*.json"))


def prepare_worker(root):
    from tokenizers import Tokenizer
    source = root / "source.parquet"
    write_source(source)
    arrays = {"sfi": [0, 0], "srg": [0, 1], "srow": [0, 1],
              "spos": [0, 1], "boundary": [0, 1, 2]}
    for key, values in arrays.items():
        np.save(root / f"{key}.npy", np.asarray(values, dtype=np.int64))
    np.save(root / "expected_ids.npy", materializer.hex_to_v16(IDS))
    (root / "files.json").write_text(json.dumps([str(source)]))
    (root / "file_identities.json").write_text(json.dumps([materializer.source_file_identity(source)]))
    (root / "tokenizer_sha256.txt").write_text(materializer.file_sha256(TOKENIZER))
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    return source, tokenizer, (0, 0, 2, str(root), str(TOKENIZER), str(root), "<u2",
                               tokenizer.token_to_id("<|endoftext|>"))


def test_worker_rejects_wrong_cached_row_before_encoding(tmp_path):
    _, _, task = prepare_worker(tmp_path)
    np.save(tmp_path / "expected_ids.npy", materializer.hex_to_v16(list(reversed(IDS))))
    with pytest.raises(ValueError, match="doc_id differs from requested order"):
        materializer._pass2a_chunk(task)
    assert not (tmp_path / "part000000.npz").exists()


def test_worker_source_path_rollback_cannot_replace_text_for_the_same_ids(tmp_path, monkeypatch):
    source, tokenizer, task = prepare_worker(tmp_path)
    replacement = tmp_path / "replacement.parquet"
    write_source(replacement, words=("sun", "sky"))
    original = tmp_path / "original.parquet"
    parquet_file = pq.ParquetFile
    injected = []

    def replace_while_opening(path_or_handle, *args, **kwargs):
        name = (path_or_handle if isinstance(path_or_handle, (str, Path))
                else path_or_handle.name)
        if Path(name) != source or injected:
            return parquet_file(path_or_handle, *args, **kwargs)
        injected.append(True)
        os.replace(source, original)
        os.replace(replacement, source)
        try:
            return parquet_file(path_or_handle, *args, **kwargs)
        finally:
            os.replace(original, source)

    monkeypatch.setattr(pq, "ParquetFile", replace_while_opening)
    materializer._pass2a_chunk(task)
    eod = tokenizer.token_to_id("<|endoftext|>")
    expected = [token for text in ("one", "two")
                for token in tokenizer.encode(text, add_special_tokens=False).ids + [eod]]
    with np.load(tmp_path / "part000000.npz") as part:
        assert injected == [True]
        assert part["tokens"].tolist() == expected


def test_worker_rejects_replacement_between_path_check_and_open(tmp_path, monkeypatch):
    source, _, task = prepare_worker(tmp_path)
    replacement = tmp_path / "replacement.parquet"
    write_source(replacement, words=("sun", "sky"))
    original = tmp_path / "original.parquet"
    open_file = Path.open

    def open_replacement(path, *args, **kwargs):
        if path != source:
            return open_file(path, *args, **kwargs)
        os.replace(source, original)
        os.replace(replacement, source)
        try:
            return open_file(path, *args, **kwargs)
        finally:
            os.replace(original, source)

    monkeypatch.setattr(Path, "open", open_replacement)
    with pytest.raises(ValueError, match="opened source identity differs"):
        materializer._pass2a_chunk(task)
    assert not (tmp_path / "part000000.npz").exists()


def test_worker_rejects_source_change_during_encoding(tmp_path, monkeypatch):
    import tokenizers
    source, tokenizer, task = prepare_worker(tmp_path)

    def encode(texts, **kwargs):
        result = tokenizer.encode_batch(texts, **kwargs)
        stat = source.stat()
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        return result

    monkeypatch.setattr(tokenizers, "Tokenizer", SimpleNamespace(
        from_str=lambda _: SimpleNamespace(encode_batch=encode)))
    with pytest.raises(ValueError, match="source identity changed during encoding"):
        materializer._pass2a_chunk(task)
    assert not (tmp_path / "part000000.npz").exists()


def test_changed_source_cannot_reuse_unpublished_token_parts(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    source = tmp_path / "s3/src/shard.parquet"
    write_source(source)
    order = tmp_path / "order.parquet"
    write_order(order)
    out = tmp_path / "streams"
    merge = materializer.merge_parts_by_position
    monkeypatch.setattr(materializer, "merge_parts_by_position", lambda *_a, **_k: (_ for _ in ()).throw(
        RuntimeError("injected after token parts")))
    with pytest.raises(RuntimeError, match="after token parts"):
        materializer.materialize_stream(order, tmp_path / "s3", TOKENIZER, out, "train")
    old = set(out.glob("train.parts-*"))
    assert len(old) == 1
    size = source.stat().st_size
    write_source(source, words=("sun", "sky"))
    assert source.stat().st_size == size
    monkeypatch.setattr(materializer, "merge_parts_by_position", merge)
    output = materializer.materialize_stream(order, tmp_path / "s3", TOKENIZER, out, "train")
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    eod = tokenizer.token_to_id("<|endoftext|>")
    expected = [token for text in ("sun", "sky") for token in tokenizer.encode(text, add_special_tokens=False).ids + [eod]]
    assert np.fromfile(output, dtype="<u2").tolist() == expected
    assert old <= set(out.glob("train.parts-*"))
    assert verified_source_document_binding(json.loads((out / "train.meta.json").read_text()))


def test_source_change_before_stream_publication_leaves_no_committed_stream(tmp_path, monkeypatch):
    source = tmp_path / "s3/src/shard.parquet"
    write_source(source)
    order = tmp_path / "order.parquet"
    write_order(order)
    merge = materializer.merge_parts_by_position

    def change_after_merge(*args, **kwargs):
        result = merge(*args, **kwargs)
        stat = source.stat()
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        return result

    monkeypatch.setattr(materializer, "merge_parts_by_position", change_after_merge)
    out = tmp_path / "streams"
    with pytest.raises(ValueError, match="source changed before token-stream publication"):
        materializer.materialize_stream(order, tmp_path / "s3", TOKENIZER, out, "train")
    assert not (out / "train.tokens.bin").exists()
    assert not (out / "train.meta.json").exists()


def changed_tokenizer_bytes(tokenizer):
    data = json.loads(TOKENIZER.read_bytes())
    first = tokenizer.id_to_token(tokenizer.encode("one", add_special_tokens=False).ids[0])
    second = tokenizer.id_to_token(tokenizer.encode("two", add_special_tokens=False).ids[0])
    vocab = data["model"]["vocab"]
    vocab[first], vocab[second] = vocab[second], vocab[first]
    return json.dumps(data, ensure_ascii=False).encode()


def test_atomic_order_rollback_consumes_only_the_verified_file(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    write_source(tmp_path / "s3/src/shard.parquet")
    order = tmp_path / "order.parquet"
    write_order(order)
    replacement = tmp_path / "replacement.parquet"
    pq.write_table(pa.table({"doc_id": list(reversed(IDS)), "position": [0, 1],
                             "n_tokens": [1, 1]}), replacement)
    backup = tmp_path / "verified-original.parquet"
    identity = materializer.source_file_identity(order)
    digest = materializer.file_sha256(order)
    snapshot = {"snapshot_id": "verified", "release_name": "verified",
                "artifacts": [{"path": str(order), "sha256": digest, **identity}]}
    read_table = pq.read_table
    injected = []

    def rollback_during_read(path_or_handle, *args, **kwargs):
        name = (path_or_handle if isinstance(path_or_handle, (str, Path))
                else path_or_handle.name)
        if Path(name) != order:
            return read_table(path_or_handle, *args, **kwargs)
        injected.append(True)
        os.replace(order, backup)
        os.replace(replacement, order)
        try:
            return read_table(path_or_handle, *args, **kwargs)
        finally:
            os.replace(backup, order)

    monkeypatch.setattr(pq, "read_table", rollback_during_read)
    output = materializer.materialize_stream(order, tmp_path / "s3", TOKENIZER,
                                             tmp_path / "streams", "train",
                                             data_snapshot=snapshot)
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    eod = tokenizer.token_to_id("<|endoftext|>")
    expected = [token for text in ("one", "two")
                for token in tokenizer.encode(text, add_special_tokens=False).ids + [eod]]
    assert injected == [True]
    assert materializer.source_file_identity(order) == identity
    assert materializer.file_sha256(order) == digest
    assert np.fromfile(output, dtype="<u2").tolist() == expected


def test_order_mutation_during_verified_read_is_rejected_before_index_build(tmp_path, monkeypatch):
    write_source(tmp_path / "s3/src/shard.parquet")
    order = tmp_path / "order.parquet"
    write_order(order)
    read = pq.read_table

    def touch_during_read(*args, **kwargs):
        result = read(*args, **kwargs)
        stat = order.stat()
        os.utime(order, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        return result

    monkeypatch.setattr(pq, "read_table", touch_during_read)
    with pytest.raises(ValueError, match="order changed while loading"):
        materializer.materialize_stream(order, tmp_path / "s3", TOKENIZER,
                                        tmp_path / "streams", "train")
    assert not (tmp_path / "s3/.doc_index_cache").exists()


def test_order_mutation_after_merge_prevents_publication_and_safe_parts_resume(tmp_path, monkeypatch):
    write_source(tmp_path / "s3/src/shard.parquet")
    order = tmp_path / "order.parquet"
    write_order(order)
    original = order.stat()
    merge = materializer.merge_parts_by_position

    def touch_after_merge(*args, **kwargs):
        result = merge(*args, **kwargs)
        os.utime(order, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000))
        return result

    monkeypatch.setattr(materializer, "merge_parts_by_position", touch_after_merge)
    out = tmp_path / "streams"
    with pytest.raises(ValueError, match="order changed before token-stream publication"):
        materializer.materialize_stream(order, tmp_path / "s3", TOKENIZER, out, "train")
    assert not (out / "train.tokens.bin").exists()
    assert not (out / "train.meta.json").exists()
    os.utime(order, ns=(original.st_atime_ns, original.st_mtime_ns))
    monkeypatch.setattr(materializer, "merge_parts_by_position", merge)
    materializer.materialize_stream(order, tmp_path / "s3", TOKENIZER, out, "train")
    assert verified_source_document_binding(json.loads((out / "train.meta.json").read_text()))


def test_worker_rejects_tokenizer_replacement_without_poisoning_parts(tmp_path):
    _, tokenizer, task = prepare_worker(tmp_path)
    path = tmp_path / "changed-tokenizer.json"
    path.write_bytes(changed_tokenizer_bytes(tokenizer))
    changed = list(task)
    changed[4] = str(path)
    with pytest.raises(ValueError, match="tokenizer bytes differ from the bound"):
        materializer._pass2a_chunk(tuple(changed))
    assert not (tmp_path / "part000000.npz").exists()


def test_verified_tokenizer_bytes_survive_path_replacement_during_parsing(tmp_path, monkeypatch):
    import tokenizers
    cls = tokenizers.Tokenizer
    original = TOKENIZER.read_bytes()
    reference = cls.from_str(original.decode())
    path = tmp_path / "tokenizer.json"
    path.write_bytes(original)

    def parse(serialized):
        path.write_bytes(changed_tokenizer_bytes(reference))
        return cls.from_str(serialized)

    monkeypatch.setattr(tokenizers, "Tokenizer", SimpleNamespace(from_str=parse))
    tokenizer, digest = materializer.load_bound_tokenizer(path, materializer.file_sha256(TOKENIZER))
    assert digest == materializer.file_sha256(TOKENIZER)
    assert tokenizer.encode("one").ids == reference.encode("one").ids
    assert materializer.file_sha256(path) != digest


def test_tokenizer_change_before_stream_commit_is_rejected_and_rollback_reuses_safe_parts(tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    source = tmp_path / "s3/src/shard.parquet"
    write_source(source)
    order = tmp_path / "order.parquet"
    write_order(order)
    tokenizer_path = tmp_path / "tokenizer.json"
    original = TOKENIZER.read_bytes()
    tokenizer_path.write_bytes(original)
    tokenizer = Tokenizer.from_str(original.decode())
    merge = materializer.merge_parts_by_position

    def change_after_merge(*args, **kwargs):
        result = merge(*args, **kwargs)
        tokenizer_path.write_bytes(changed_tokenizer_bytes(tokenizer))
        return result

    monkeypatch.setattr(materializer, "merge_parts_by_position", change_after_merge)
    out = tmp_path / "streams"
    with pytest.raises(ValueError, match="tokenizer changed before token-stream publication"):
        materializer.materialize_stream(order, tmp_path / "s3", tokenizer_path, out, "train")
    assert not (out / "train.tokens.bin").exists()
    assert not (out / "train.meta.json").exists()
    tokenizer_path.write_bytes(original)
    monkeypatch.setattr(materializer, "merge_parts_by_position", merge)
    output = materializer.materialize_stream(order, tmp_path / "s3", tokenizer_path, out, "train")
    eod = tokenizer.token_to_id("<|endoftext|>")
    expected = [token for text in ("one", "two") for token in tokenizer.encode(text, add_special_tokens=False).ids + [eod]]
    assert np.fromfile(output, dtype="<u2").tolist() == expected
    assert verified_source_document_binding(json.loads((out / "train.meta.json").read_text()))


def test_parallel_index_readers_preserve_file_row_mapping(tmp_path):
    results = []
    for label, workers in (("serial", 1), ("parallel", 3)):
        root = tmp_path / label
        root.mkdir()
        for file_index in range(4):
            ids = [f"{2 * file_index + row:032x}" for row in (2, 1)]
            pq.write_table(pa.table({"doc_id": ids, "text": ["one", "two"]}),
                           root / f"{file_index:02d}.parquet", row_group_size=1)
        results.append(materializer.build_doc_index(root, workers=workers)[:3])
    assert all(np.array_equal(left, right) for left, right in zip(*results))
    assert results[0][1].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert results[0][2].tolist() == [1, 0] * 4


def test_index_reader_concurrency_is_bounded_and_output_stays_ordered(monkeypatch):
    import threading
    import time
    active = 0
    peak = 0
    lock = threading.Lock()

    def read(path):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        index = int(path.name)
        time.sleep(0.005 * (1 + index % 3))
        with lock:
            active -= 1
        return index

    monkeypatch.setattr(materializer, "_read_doc_index_file", read)
    rows = list(materializer._iter_doc_index_files([Path(str(i)) for i in range(32)], 128))
    assert rows == [(i, i) for i in range(32)]
    assert 1 < peak <= 8


@pytest.mark.parametrize("binding", [None, {}, {"schema_version": True},
    {"schema_version": 1, "expected_doc_ids_verified": False, "roster_fingerprint": "a" * 64},
    {"schema_version": 1, "expected_doc_ids_verified": True, "roster_fingerprint": "bad"}])
def test_source_binding_rejects_missing_or_malformed_evidence(binding):
    assert not verified_source_document_binding({"source_document_binding": binding})


@pytest.mark.parametrize("field,value", [("order_bytes_verified", None),
    ("source_handles_verified", None), ("source_handles_verified", False),
    ("order_bytes_verified", False), ("tokenizer_bytes_verified", False),
    ("expected_doc_ids_verified", False), ("schema_version", 2)])
def test_source_binding_requires_current_order_tokenizer_and_row_evidence(field, value):
    binding = {"schema_version": 4, "expected_doc_ids_verified": True,
               "tokenizer_bytes_verified": True, "order_bytes_verified": True, "source_handles_verified": True,
               "roster_fingerprint": "a" * 64}
    if value is None:
        binding.pop(field)
    else:
        binding[field] = value
    assert not verified_source_document_binding({"source_document_binding": binding})
