from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from yuformer_model import ModelConfig, create_model, YuFormerModel
from yuformer_model.attention import precompute_rope_freqs, apply_rope


def test_config():
    config = ModelConfig(num_layers=6, hidden_size=256, num_heads=4, num_kv_heads=2)
    assert config.head_dim == 64
    assert config.is_full_attn(0) == True
    assert config.is_full_attn(1) == False
    assert config.is_swa(1) == True
    assert config.use_rope(0) == True
    assert config.use_rope(1) == False
    print(f"test_config: ok (head_dim={config.head_dim})")


def test_preset():
    config = ModelConfig.from_preset("tiny")
    assert config.num_layers == 4
    assert config.hidden_size == 256
    assert config.vocab_size == 8000
    print(f"test_preset: ok (layers={config.num_layers}, hidden={config.hidden_size})")


def test_forward_tiny():
    config = ModelConfig.from_preset("tiny")
    model = create_model(config)
    model.eval()

    tokens = torch.randint(0, config.vocab_size, (2, 128))
    with torch.no_grad():
        logits = model(tokens)

    assert logits.shape == (2, 128, config.vocab_size)
    assert not torch.isnan(logits).any()
    print(f"test_forward_tiny: ok (logits={logits.shape}, params={model.num_parameters():,})")


def test_forward_small():
    config = ModelConfig.from_preset("small")
    model = create_model(config)
    model.eval()

    tokens = torch.randint(0, config.vocab_size, (2, 256))
    with torch.no_grad():
        logits = model(tokens)

    assert logits.shape == (2, 256, config.vocab_size)
    assert not torch.isnan(logits).any()
    print(f"test_forward_small: ok (logits={logits.shape}, params={model.num_parameters():,})")


def test_swa_full_alternation():
    config = ModelConfig(num_layers=8, hidden_size=256, num_heads=4, num_kv_heads=2,
                        attention_every=4, rope_every=2)
    model = create_model(config)

    full_layers = [i for i in range(8) if config.is_full_attn(i)]
    swa_layers = [i for i in range(8) if config.is_swa(i)]
    rope_layers = [i for i in range(8) if config.use_rope(i)]
    nope_layers = [i for i in range(8) if not config.use_rope(i)]

    assert full_layers == [0, 4], f"expected [0,4], got {full_layers}"
    assert len(swa_layers) == 6, f"expected 6 SWA, got {len(swa_layers)}"
    assert rope_layers == [0, 2, 4, 6], f"expected [0,2,4,6], got {rope_layers}"
    assert len(nope_layers) == 4, f"expected 4 NoPE, got {len(nope_layers)}"
    print(f"test_swa_full_alternation: ok (full={full_layers}, swa={len(swa_layers)}, "
          f"rope={rope_layers}, nope={len(nope_layers)})")


def test_backward():
    config = ModelConfig.from_preset("tiny")
    model = create_model(config)
    model.train()

    tokens = torch.randint(0, config.vocab_size, (2, 64))
    logits = model(tokens)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, config.vocab_size),
        tokens.view(-1),
    )
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"{name} has NaN gradient"
    print(f"test_backward: ok (loss={loss.item():.4f})")


def test_different_seq_lens():
    config = ModelConfig.from_preset("tiny")
    model = create_model(config)
    model.eval()

    for seq_len in [32, 64, 128, 256, 512]:
        tokens = torch.randint(0, config.vocab_size, (1, seq_len))
        with torch.no_grad():
            logits = model(tokens)
        assert logits.shape == (1, seq_len, config.vocab_size)
    print("test_different_seq_lens: ok (32, 64, 128, 256, 512)")


def test_rope():
    head_dim = 64
    freqs = precompute_rope_freqs(head_dim, 128, rope_base=10000.0, rope_percent=0.5)
    cos = freqs[..., 0]
    sin = freqs[..., 1]

    x = torch.randn(1, 1, 128, head_dim)
    rotated = apply_rope(x, cos, sin)
    assert rotated.shape == x.shape
    assert not torch.equal(rotated, x)
    print("test_rope: ok")


def test_tied_embeddings():
    config_tied = ModelConfig(num_layers=2, hidden_size=64, num_heads=4, num_kv_heads=2,
                             vocab_size=100, tie_word_embeddings=True)
    model_tied = create_model(config_tied)
    assert model_tied.lm_head is model_tied.embed_tokens

    config_untied = ModelConfig(num_layers=2, hidden_size=64, num_heads=4, num_kv_heads=2,
                                vocab_size=100, tie_word_embeddings=False)
    model_untied = create_model(config_untied)
    assert model_untied.lm_head is not model_untied.embed_tokens
    print("test_tied_embeddings: ok")


def test_param_count():
    config = ModelConfig.from_preset("tiny")
    model = create_model(config)
    total = model.num_parameters()
    non_emb = model.num_parameters(non_embedding=True)
    assert total > 0
    assert non_emb > 0
    assert total > non_emb
    print(f"test_param_count: ok (total={total:,}, non_embedding={non_emb:,})")


if __name__ == "__main__":
    test_config()
    test_preset()
    test_rope()
    test_forward_tiny()
    test_forward_small()
    test_swa_full_alternation()
    test_backward()
    test_different_seq_lens()
    test_tied_embeddings()
    test_param_count()
    print("\nAll model tests passed!")
