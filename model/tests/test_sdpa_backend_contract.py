from contextlib import contextmanager
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yuformer_model import attention
from yuformer_model.config import ModelConfig


def test_cudnn_selection_is_strict_and_receives_contiguous_qkv(monkeypatch):
    events = []

    @contextmanager
    def strict(backend):
        events.append(backend)
        yield

    api = ModuleType("torch.nn.attention")
    api.SDPBackend = SimpleNamespace(CUDNN_ATTENTION="cudnn_only")
    api.sdpa_kernel = strict
    monkeypatch.setitem(sys.modules, "torch.nn.attention", api)
    monkeypatch.setattr(attention, "_SDPA_SUPPORTS_GQA", True)

    def sdpa(q, k, v, **kwargs):
        assert events == ["cudnn_only"]
        assert all(value.is_contiguous() for value in (q, k, v))
        assert q.shape[1] == 4 and k.shape[1] == v.shape[1] == 2
        assert kwargs["enable_gqa"] is True
        return torch.zeros_like(q)

    monkeypatch.setattr(attention.F, "scaled_dot_product_attention", sdpa)
    config = ModelConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_layers=1, num_heads=4, num_kv_heads=2,
                         window_size=8, attention_every=2, qk_norm=True,
                         attention_output_gate=True)
    module = attention.GQAAttention(config, 0)
    module.sdpa_backend = "cudnn"
    output = module(torch.randn(2, 16, 32))
    assert output.shape == (2, 16, 32)
    assert events == ["cudnn_only"]
