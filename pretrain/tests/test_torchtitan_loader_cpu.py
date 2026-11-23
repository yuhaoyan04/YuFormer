"""Exercise the real stream reader on CPU without installing TorchTitan.

Only its abstract base and exhaustion exception are substituted; sampling,
memmaps, batching, state save/load, and tensor construction use adapter code.
"""
import importlib.util
import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def reader(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "backends/torchtitan/yuformer_ext"
    package = types.ModuleType("_loader_cpu_audit")
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    abstract = types.ModuleType("torchtitan.components.dataloader")

    class BaseDataLoader:
        @dataclass
        class Config:
            pass

    class Exhausted(Exception):
        pass

    abstract.BaseDataLoader = BaseDataLoader
    abstract.DataloaderExhaustedError = Exhausted
    monkeypatch.setitem(sys.modules, abstract.__name__, abstract)
    name = package.__name__ + ".data"
    spec = importlib.util.spec_from_file_location(name, root / "data.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    yield module.PretokenizedDataLoader, Exhausted
    monkeypatch.delitem(sys.modules, package.__name__ + ".stream_math", raising=False)


def test_three_stage_accumulation_and_resume_preserve_every_target(reader, tmp_path):
    loader_class, exhausted = reader
    gate = {"status": "PASS", "dataset_id": "d", "snapshot_id": "s",
            "tokenizer_sha256": "tok", "vocab_size": 10000}
    (tmp_path / "gate.json").write_text(json.dumps(gate))
    (tmp_path / "release.json").write_text(json.dumps(gate))
    limits, stages, metas = [8, 16, 8], [], []
    expected = []
    for stage, limit in enumerate(limits):
        # Include headroom and prove that stage_sample_limits exclude it.
        tokens = np.arange((limit + 3) * 8 + 1, dtype="<u2") + stage * 1000
        binary, meta = tmp_path / f"{stage}.bin", tmp_path / f"{stage}.json"
        tokens.tofile(binary)
        meta.write_text(json.dumps({
            "dtype": "<u2", "total_tokens": len(tokens), "n_docs": 1,
            "tokenizer_sha256": "tok", "vocab_size": 10000, "data_snapshot_id": "s",
            "document_boundary": {"policy": "append_eod", "count": 1},
        }))
        stages.append(str(binary))
        metas.append(str(meta))
        expected.extend(tokens[1:limit * 8 + 1].tolist())
    cfg = loader_class.Config(
        stages=stages, meta=metas, stage_sample_limits=limits,
        tokenizer_gate=str(tmp_path / "gate.json"),
        release_manifest=str(tmp_path / "release.json"), emit_stage_id=True,
    )

    def readers(world=2, batch=2):
        return [loader_class(cfg, dp_world_size=world, dp_rank=rank, tokenizer=None,
                             seq_len=8, local_batch_size=batch) for rank in range(world)]

    original = readers()
    iterators = [iter(loader) for loader in original]
    observed, checkpoints = [], []
    for microstep in range(8):
        batches = [next(iterator) for iterator in iterators]
        for inputs, targets in batches:
            observed.extend(targets.flatten().tolist())
            expected_stage = 0 if microstep < 2 else (1 if microstep < 6 else 2)
            assert inputs["_yuformer_stage_id"].tolist() == [expected_stage] * 2
        # Two microbatches per optimizer update. Save at every legal boundary.
        if microstep % 2 == 1:
            checkpoints.append((len(observed), original[0].state_dict()))
    assert observed == expected
    for offset, state in checkpoints:
        restored = readers(world=1, batch=4)
        restored[0].load_state_dict(state)
        remaining = []
        resumed_iterator = iter(restored[0])
        for _ in range((sum(limits) - state["consumed_samples"]) // 4):
            _, targets = next(resumed_iterator)
            remaining.extend(targets.flatten().tolist())
        # Include exact phase boundaries and the terminal cursor.
        assert remaining == expected[offset:]
        assert restored[0].state_dict()["consumed_samples"] == sum(limits)
        with pytest.raises(exhausted):
            next(iter(restored[0]))
    with pytest.raises(exhausted):
        next(iterators[0])


@pytest.mark.parametrize("bind_identity", [False, True])
def test_stream_path_rollback_cannot_change_the_attested_mapping(reader, tmp_path, monkeypatch, bind_identity):
    loader_class, _ = reader
    module = sys.modules[loader_class.__module__]
    gate = {"status": "PASS", "dataset_id": "d", "snapshot_id": "s",
            "tokenizer_sha256": "tok", "vocab_size": 10000}
    for name in ("gate.json", "release.json"):
        (tmp_path / name).write_text(json.dumps(gate))
    binary, replacement = tmp_path / "train.bin", tmp_path / "replacement.bin"
    np.arange(9, dtype="<u2").tofile(binary)
    (np.arange(9, dtype="<u2") + 100).tofile(replacement)
    meta = tmp_path / "meta.json"
    value = {"dtype": "<u2", "total_tokens": 9, "n_docs": 1,
        "tokenizer_sha256": "tok", "vocab_size": 10000, "data_snapshot_id": "s",
        "edge_sha256": module._edge_sha256(binary),
        "document_boundary": {"policy": "append_eod", "count": 1}}
    if bind_identity:
        stat = binary.stat()
        value["stream_stat_identity"] = {"bytes": stat.st_size, "device": stat.st_dev,
                                         "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns}
    meta.write_text(json.dumps(value))
    original = tmp_path / "original.bin"
    memmap = np.memmap
    injected = []

    def replace_during_mapping(path_or_handle, *args, **kwargs):
        injected.append(True)
        os.replace(binary, original)
        os.replace(replacement, binary)
        try:
            return memmap(path_or_handle, *args, **kwargs)
        finally:
            os.replace(original, binary)

    monkeypatch.setattr(np, "memmap", replace_during_mapping)
    config = loader_class.Config(stages=[str(binary)], meta=[str(meta)],
        tokenizer_gate=str(tmp_path / "gate.json"), release_manifest=str(tmp_path / "release.json"))
    loader = loader_class(config, dp_world_size=1, dp_rank=0, tokenizer=None,
                          seq_len=4, local_batch_size=1)
    _, labels = next(iter(loader))
    assert injected == [True]
    assert labels.tolist() == [[1, 2, 3, 4]]


def test_published_stream_identity_rejects_interior_change_with_unchanged_edges(reader, tmp_path):
    loader_class, _ = reader
    module = sys.modules[loader_class.__module__]
    gate = {"status": "PASS", "dataset_id": "d", "snapshot_id": "s",
            "tokenizer_sha256": "tok", "vocab_size": 10000}
    for name in ("gate.json", "release.json"):
        (tmp_path / name).write_text(json.dumps(gate))
    binary, replacement = tmp_path / "train.bin", tmp_path / "replacement.bin"
    tokens = np.arange(1_600_001, dtype="<u2") % 10000
    tokens.tofile(binary)
    edge = module._edge_sha256(binary)
    stat = binary.stat()
    tokens[800_000] = (tokens[800_000] + 1) % 10000
    tokens.tofile(replacement)
    assert module._edge_sha256(replacement) == edge
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"dtype": "<u2", "total_tokens": len(tokens), "n_docs": 1,
        "tokenizer_sha256": "tok", "vocab_size": 10000, "data_snapshot_id": "s",
        "edge_sha256": edge, "document_boundary": {"policy": "append_eod", "count": 1},
        "stream_stat_identity": {"bytes": stat.st_size, "device": stat.st_dev,
                                 "inode": stat.st_ino, "mtime_ns": stat.st_mtime_ns}}))
    os.replace(replacement, binary)
    config = loader_class.Config(stages=[str(binary)], meta=[str(meta)],
        tokenizer_gate=str(tmp_path / "gate.json"), release_manifest=str(tmp_path / "release.json"))
    with pytest.raises(ValueError, match="stream published identity mismatch"):
        loader_class(config, dp_world_size=1, dp_rank=0, tokenizer=None,
                     seq_len=4, local_batch_size=1)
