"""Run in the pinned TorchTitan environment; legacy environments skip this file."""
import json
import queue
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

if tuple(int(part) for part in torch.__version__.split(".")[:2]) < (2, 14):
    pytest.skip("requires the pinned TorchTitan environment", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pretrain/backends/torchtitan"))
sys.path.insert(0, str(ROOT / "model/src"))

from yuformer_ext.checkpoint import YuFormerCheckpointManager
from yuformer_ext.data import PretokenizedDataLoader
from yuformer_ext.metrics import isolated_validation_metrics
from yuformer_ext.model import YuFormerTitanModel
from yuformer_ext.optim import adamw_config
from yuformer_ext.runtime import capture_rng, restore_rng
from yuformer_model.layers import RMSNorm


def test_meta_adapter_initialization_and_tying(tmp_path):
    config_path = tmp_path / "models.yaml"
    config_path.write_text(yaml.safe_dump({"presets": {"tiny": {
        "vocab_size": 128, "hidden_size": 32, "num_layers": 2,
        "intermediate_size": 64, "num_heads": 4, "num_kv_heads": 2,
        "max_seq_len": 16, "qk_norm": True, "tie_word_embeddings": True,
    }}}))
    with torch.device("meta"):
        model = YuFormerTitanModel(YuFormerTitanModel.Config(
            preset="tiny", config_path=str(config_path)))
    model.to_empty(device="cpu")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(float("nan"))
        model.init_states()
    assert model.core.embed_tokens.weight is model.core.lm_head.weight
    for module in model.modules():
        if isinstance(module, RMSNorm):
            torch.testing.assert_close(module.weight, torch.ones_like(module.weight))
    loss = model(torch.ones(2, 8, dtype=torch.long)).float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_python_numpy_and_torch_rng_restore():
    snapshot = capture_rng()
    expected = (random.random(), np.random.rand(), torch.rand(5))
    restore_rng(snapshot)
    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(5), expected[2], rtol=0, atol=0)


def test_norms_have_no_weight_decay_and_nonfinite_update_is_refused():
    model = torch.nn.Sequential()
    model.add_module("norm", RMSNorm(8))
    model.add_module("projection", torch.nn.Linear(8, 4, bias=False))
    config = adamw_config({"lr": 0.001, "weight_decay": 0.1})
    config.implementation = "for-loop"
    optimizer = config.build(model_parts=[model])
    groups = next(iter(optimizer)).param_groups
    assert [group["weight_decay"] for group in groups] == [0, 0.1]
    model(torch.ones(2, 8)).sum().backward()
    with torch.no_grad():
        model.projection.weight.grad[0, 0] = float("nan")
    before = model.projection.weight.clone()
    with pytest.raises(RuntimeError, match="non-finite gradients"):
        optimizer.step()
    torch.testing.assert_close(model.projection.weight, before)


def test_milestone_checkpoints_survive_recent_retention(tmp_path):
    manager = object.__new__(YuFormerCheckpointManager)
    manager.enable = False
    manager.folder = str(tmp_path)
    manager.keep_every, manager.keep_latest_k = 10, 2
    manager.purge_queue = queue.Queue()
    manager._should_purge = lambda: True
    for step in (5, 10, 15, 20, 25, 30):
        path = tmp_path / f"step-{step}"
        path.mkdir()
        (path / ".metadata").touch()
    manager._purge_stale_checkpoints()
    removed = []
    while not manager.purge_queue.empty():
        removed.append(Path(manager.purge_queue.get()).name)
    assert removed == ["step-5", "step-15"]


def test_validation_does_not_discard_unlogged_training_tokens():
    metrics = SimpleNamespace(ntokens_since_last_log=123, data_loading_times=[0.5],
                              time_last_log=0.0)
    validator = SimpleNamespace(metrics_processor=metrics)

    @isolated_validation_metrics
    def validate(self):
        assert self.metrics_processor.ntokens_since_last_log == 0
        self.metrics_processor.ntokens_since_last_log = 900
        self.metrics_processor.time_last_log = 10000

    validate(validator)
    assert metrics.ntokens_since_last_log == 123
    assert metrics.data_loading_times == [0.5]
    assert 0 <= metrics.time_last_log < 1


def test_data_cursor_reshards_without_dropping_or_repeating_targets(tmp_path):
    gate = {"status": "PASS", "dataset_id": "d", "snapshot_id": "s",
            "tokenizer_sha256": "t", "vocab_size": 128}
    release = {"status": "PASS", "dataset_id": "d", "snapshot_id": "s"}
    (tmp_path / "gate.json").write_text(json.dumps(gate))
    (tmp_path / "release.json").write_text(json.dumps(release))
    binary = tmp_path / "tokens.bin"
    np.arange(97, dtype="<u2").tofile(binary)
    meta = {"total_tokens": 97, "dtype": "<u2", "n_docs": 1,
            "data_snapshot_id": "s", "tokenizer_sha256": "t", "vocab_size": 128,
            "document_boundary": {"policy": "append_eod", "count": 1}}
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    config = PretokenizedDataLoader.Config(
        stages=[str(binary)], meta=[str(tmp_path / "meta.json")],
        tokenizer_gate=str(tmp_path / "gate.json"),
        release_manifest=str(tmp_path / "release.json"))
    original = config.build(dp_world_size=1, dp_rank=0, tokenizer=None,
                            seq_len=8, local_batch_size=2)
    iterator = iter(original)
    first = next(iterator)[1]
    snapshot = original.state_dict()
    left = config.build(dp_world_size=2, dp_rank=0, tokenizer=None,
                        seq_len=8, local_batch_size=1)
    right = config.build(dp_world_size=2, dp_rank=1, tokenizer=None,
                         seq_len=8, local_batch_size=1)
    left.load_state_dict(snapshot)
    right.load_state_dict(snapshot)
    resumed = torch.cat([next(iter(left))[1], next(iter(right))[1]])
    torch.testing.assert_close(resumed, next(iterator)[1])
    torch.testing.assert_close(torch.cat([first.flatten(), resumed.flatten()]),
                               torch.arange(1, 33))
