#!/usr/bin/env python3
"""End-to-end smoke test: create model, forward, backward, generate one token."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from yuformer_model import ModelConfig, create_model


def main():
    print("=" * 60)
    print("  YuFormer Model Smoke Test")
    print("=" * 60)

    config = ModelConfig.from_preset("small")
    print(f"\nConfig: small")
    print(f"  layers={config.num_layers}, hidden={config.hidden_size}, "
          f"heads={config.num_heads}/{config.num_kv_heads}")
    print(f"  attention_every={config.attention_every}, rope_every={config.rope_every}")
    print(f"  full_attn_layers={config.num_full_attn_layers}, "
          f"swa_layers={config.num_swa_layers}")
    print(f"  window_size={config.window_size}, rope_base={config.rope_base}")

    model = create_model(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    total_params = model.num_parameters()
    print(f"\nTotal parameters: {total_params:,}")

    print(f"\n--- Forward pass (seq=512) ---")
    tokens = torch.randint(0, config.vocab_size, (4, 512), device=device)
    with torch.no_grad():
        logits = model(tokens)
    print(f"  input: {tokens.shape}")
    print(f"  output: {logits.shape}")
    assert logits.shape == (4, 512, config.vocab_size)

    print(f"\n--- Backward pass ---")
    model.train()
    tokens = torch.randint(0, config.vocab_size, (2, 256), device=device)
    logits = model(tokens)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, config.vocab_size),
        tokens[:, 1:].reshape(-1),
    )
    loss.backward()
    print(f"  loss: {loss.item():.4f}")

    print(f"\n--- Generation (greedy, 10 tokens) ---")
    model.eval()
    input_ids = torch.tensor([[1]], device=device)
    with torch.no_grad():
        for _ in range(10):
            logits = model(input_ids)
            next_token = logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)
    print(f"  generated tokens: {input_ids[0].tolist()}")

    print(f"\n{'=' * 60}")
    print(f"  All smoke tests passed!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
