"""数据游标确定性（宪法 4）：断点续训消费序列与原跑完全一致。"""
import hashlib
import json

import numpy as np
import pytest

from yuformer_pretrain.data import (
    DataCursor,
    PrefetchLoader,
    TokenStream,
    concat_streams,
    distributed_sample_index,
    load_data_release,
    load_data_snapshot,
)

SEQ = 31  # 小 seq_len 加速测试；span = 32


def _make_stream(tmp_path, name, n_tokens, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 60000, size=n_tokens, dtype=np.uint16)
    bin_path = tmp_path / f"{name}.tokens.bin"
    arr.tofile(bin_path)
    sha = hashlib.sha256(arr.tobytes()).hexdigest()
    from yuformer_pretrain.data import _edge_sha256

    meta = {
        "name": name,
        "total_tokens": int(n_tokens),
        "bytes": int(bin_path.stat().st_size),
        "sha256": sha,
        "edge_sha256": _edge_sha256(bin_path),
        "tokenizer_sha256": "test",
        "source": f"test:{name}",
    }
    meta_path = tmp_path / f"{name}.meta.json"
    meta_path.write_text(json.dumps(meta))
    return TokenStream.open(bin_path, meta_path, SEQ)


def test_stream_open_guard(tmp_path):
    s = _make_stream(tmp_path, "a", 1000)
    assert s.samples == 1000 // (SEQ + 1)
    (tmp_path / "a.meta.json").write_text("{}")  # edge hash 不匹配
    with pytest.raises(Exception):
        TokenStream.open(tmp_path / "a.tokens.bin", tmp_path / "a.meta.json", SEQ)


def test_formal_stream_requires_eod_and_matching_dataset_release(tmp_path):
    _make_stream(tmp_path, "formal", 1000)
    bin_path = tmp_path / "formal.tokens.bin"
    meta_path = tmp_path / "formal.meta.json"
    with pytest.raises(Exception, match="append_eod"):
        TokenStream.open(bin_path, meta_path, SEQ, require_eod=True)

    meta = json.loads(meta_path.read_text())
    meta.update(
        {
            "n_docs": 10,
            "data_snapshot_id": "yuformer-snapshot-a",
            "document_boundary": {
                "policy": "append_eod",
                "token": "<|endoftext|>",
                "token_id": 0,
                "count": 10,
            },
        }
    )
    meta_path.write_text(json.dumps(meta))
    stream = TokenStream.open(
        bin_path,
        meta_path,
        SEQ,
        require_eod=True,
        expected_snapshot_id="yuformer-snapshot-a",
    )
    assert stream.meta["document_boundary"]["count"] == 10
    with pytest.raises(Exception, match="data_snapshot_id"):
        TokenStream.open(
            bin_path,
            meta_path,
            SEQ,
            require_eod=True,
            expected_snapshot_id="yuformer-snapshot-b",
        )


def test_concat_lut():
    streams = []
    for n in (320, 640):
        # 不触碰磁盘，直接构造
        s = TokenStream.__new__(TokenStream)
        s.samples = n // (SEQ + 1)
        streams.append(s)
    total, lut = concat_streams(streams)
    assert total == 10 + 20
    assert lut[9].tolist() == [0, 9]
    assert lut[10].tolist() == [1, 0]
    assert lut[29].tolist() == [1, 19]


def test_cursor_state_roundtrip():
    c = DataCursor(100, 2)
    c.advance(30)
    c2 = DataCursor(100, 2)
    c2.load_state_dict(c.state_dict())
    assert c2.consumed == 30
    c2.advance(170)  # 200 capacity
    with pytest.raises(Exception, match="预算不足"):
        c2.advance(1)


def _prefetch_collect(streams, start, count, workers=2, in_flight=4):
    total, _ = concat_streams(streams)
    cur = DataCursor(total, 1)
    cur.consumed = start
    cur.capacity = 1 << 62
    out = []
    with PrefetchLoader(streams, cur, num_workers=workers, in_flight=in_flight) as ld:
        for _ in range(count):
            out.append(ld.next_sample())
    return out


def test_prefetch_order_and_resume_equivalence(tmp_path):
    streams = [
        _make_stream(tmp_path, "s1", 3200, seed=1),
        _make_stream(tmp_path, "s2", 6400, seed=2),
    ]
    full = _prefetch_collect(streams, 0, 90, workers=2)

    head = _prefetch_collect(streams, 0, 40, workers=2)
    tail = _prefetch_collect(streams, 40, 50, workers=3)  # 换 worker 数也不变
    assert len(full[40]) == SEQ + 1
    for a, b in zip(full, head + tail):
        assert np.array_equal(a, b)


def test_prefetch_mid_window_abandon(tmp_path):
    streams = [_make_stream(tmp_path, "s1", 3200, seed=1)]
    head = _prefetch_collect(streams, 0, 7, workers=2, in_flight=8)  # 窗口内中途放弃
    tail = _prefetch_collect(streams, 7, 20, workers=2, in_flight=8)
    full = _prefetch_collect(streams, 0, 27, workers=1)
    for a, b in zip(full, head + tail):
        assert np.array_equal(a, b)


def test_prefetch_cross_stage_and_epoch(tmp_path):
    streams = [
        _make_stream(tmp_path, "s1", 320, seed=1),   # 10 样本
        _make_stream(tmp_path, "s2", 640, seed=2),   # 20 样本
    ]
    total, _ = concat_streams(streams)
    assert total == 30
    seqs = _prefetch_collect(streams, 25, 10)  # 跨 stage 边界 + 回绕（capacity 需够）
    assert len(seqs) == 10
    first = _prefetch_collect(streams, 0, 1)[0]
    assert np.array_equal(seqs[5], first)  # 全局序号 30 ≡ 0（回绕到 stage0 样本 0）


def test_distributed_sample_index_is_disjoint_and_resume_exact():
    spr, resume = 3, 120
    for world in (1, 2, 4):
        per_rank = [
            [
                distributed_sample_index(
                    i,
                    resume_consumed=resume,
                    samples_per_rank=spr,
                    rank=rank,
                    world_size=world,
                )
                for i in range(2 * spr)
            ]
            for rank in range(world)
        ]
        assert sorted(x for xs in per_rank for x in xs) == list(
            range(resume, resume + 2 * spr * world)
        )
        assert all(set(per_rank[a]).isdisjoint(per_rank[b])
                   for a in range(world) for b in range(a + 1, world))
    # 第一个本地样本从 checkpoint 的下一全局 step 开始，rank 子块位置保持不变。
    assert [xs[0] for xs in per_rank] == [120, 123, 126, 129]
    assert distributed_sample_index(
        0, resume_consumed=144, samples_per_rank=spr, rank=2, world_size=world
    ) == 150


def test_prefetch_uses_distributed_physical_index(tmp_path):
    stream = _make_stream(tmp_path, "s1", (SEQ + 1) * 64, seed=7)
    cur = DataCursor(stream.samples, 1)
    cur.capacity = 1 << 62
    mapping = lambda i: distributed_sample_index(  # noqa: E731 - 测试映射直观性
        i, resume_consumed=8, samples_per_rank=2, rank=1, world_size=3
    )
    with PrefetchLoader([stream], cur, num_workers=2, sample_index_fn=mapping) as loader:
        got = [loader.next_sample() for _ in range(4)]
    # rank1: step0 取 10,11；step1 取 16,17。
    for arr, physical in zip(got, [10, 11, 16, 17]):
        assert np.array_equal(arr, stream.sample(physical))


def test_open_val_streams_vocab_and_tokenizer_guard(tmp_path):
    from yuformer_pretrain.data import _edge_sha256, open_val_streams

    def mk(name, vocab, tokenizer_sha="tok-A"):
        rng = np.random.default_rng(0)
        arr = rng.integers(0, min(max(vocab, 2), 65536), size=(SEQ + 1) * 3, dtype=np.uint16)
        bin_path = tmp_path / f"{name}.tokens.bin"
        arr.tofile(bin_path)
        meta = {
            "name": name,
            "total_tokens": int(arr.size),
            "bytes": int(bin_path.stat().st_size),
            "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
            "edge_sha256": _edge_sha256(bin_path),
            "vocab_size": vocab,
            "tokenizer_sha256": tokenizer_sha,
        }
        (tmp_path / f"{name}.meta.json").write_text(json.dumps(meta))

    mk("val_a", 65536)
    mk("val_b", 131072)  # E-68 实证：128k 流落进 val_dir 曾炸掉全部 65536 词表臂
    mk("val_c", 0, tokenizer_sha="tok-B")
    mk("val_legacy", 0, tokenizer_sha=None)
    mk("train_stage", 65536)  # 同目录训练流绝不能进入 held-out validation
    streams, skipped = open_val_streams(
        tmp_path, SEQ, model_vocab=65536, tokenizer_sha256="tok-A"
    )
    assert [s.bin_path.name for s in streams] == ["val_a.tokens.bin"]
    assert skipped == ["val_b.tokens.bin", "val_c.tokens.bin", "val_legacy.tokens.bin"]
    streams32, skipped32 = open_val_streams(
        tmp_path, SEQ, model_vocab=32768, tokenizer_sha256="tok-A"
    )
    assert streams32 == []
    assert skipped32 == ["val_a.tokens.bin", "val_b.tokens.bin", "val_c.tokens.bin",
                         "val_legacy.tokens.bin"]


def test_data_release_requires_pass_and_dataset_id(tmp_path):
    manifest = tmp_path / "release.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "status": "PASS",
        "dataset_id": "yuformer-data-abc",
        "snapshot_id": "yuformer-snapshot-abc",
        "spec_sha256": "spec",
        "release_name": "release",
    }))
    identity = load_data_release(manifest, required=True)
    assert identity["dataset_id"] == "yuformer-data-abc"
    assert identity["snapshot_id"] == "yuformer-snapshot-abc"

    manifest.write_text(json.dumps({
        "schema_version": 1,
        "status": "BLOCKED",
        "dataset_id": None,
        "failed_gates": ["B1"],
    }))
    with pytest.raises(Exception, match="未通过门禁"):
        load_data_release(manifest, required=True)


def test_data_release_optional_only_for_compatibility_runs():
    assert load_data_release(None, required=False) is None
    with pytest.raises(Exception, match="release_manifest"):
        load_data_release(None, required=True)


def test_blocked_candidate_snapshot_can_drive_token_materialization(tmp_path):
    manifest = tmp_path / "candidate.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "status": "BLOCKED",
        "dataset_id": None,
        "snapshot_id": "yuformer-snapshot-candidate",
        "release_name": "candidate",
    }))
    snapshot = load_data_snapshot(manifest)
    assert snapshot["snapshot_id"] == "yuformer-snapshot-candidate"
    assert snapshot["status"] == "BLOCKED"
