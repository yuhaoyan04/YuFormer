from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn.functional as F

from yuformer_model import ModelConfig, YuFormerModel, create_model
from yuformer_model.attention import GQAAttention, apply_rope, precompute_rope_freqs


def _tiny_config(**overrides) -> ModelConfig:
    kwargs = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=256,
        window_size=16,
        attention_every=1,
        rope_every=1,
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


def analytical_param_count(config: ModelConfig) -> int:
    d = config.hidden_size
    hd = config.head_dim
    inter = config.intermediate_size
    emb = config.vocab_size * d * (1 if config.tie_word_embeddings else 2)

    def layer(is_full: bool) -> int:
        qkv = d * (d + 2 * config.num_kv_heads * hd)
        if config.attention_output_gate and not is_full:
            qkv += d * d
        o = d * d
        mlp = 3 * d * inter
        norms = 2 * d + (2 * hd if config.qk_norm else 0)
        return qkv + o + mlp + norms

    total = emb + config.num_full_attn_layers * layer(True) + config.num_swa_layers * layer(False)
    return total + d


def test_meta_materialization_reinitializes_every_normalization_weight():
    from yuformer_model.layers import RMSNorm

    cfg = _tiny_config(qk_norm=True, attention_every=2, attention_output_gate=True)
    with torch.device("meta"):
        model = YuFormerModel(cfg)
    model.to_empty(device="cpu")
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, RMSNorm):
                module.weight.fill_(float("nan"))
        model._init_parameters()
        model.rope_cos, model.rope_sin = precompute_rope_freqs(
            cfg.head_dim, cfg.max_seq_len, cfg.rope_base, cfg.rope_percent
        )
    for module in model.modules():
        if isinstance(module, RMSNorm):
            torch.testing.assert_close(module.weight, torch.ones_like(module.weight))
    assert torch.isfinite(model(torch.randint(1, cfg.vocab_size, (2, 16)))).all()


def test_eod_embedding_is_trainable_when_tokenizer_aliases_pad_and_eos():
    model = YuFormerModel(_tiny_config(pad_token_id=0, eos_token_id=0, tie_word_embeddings=True))
    assert model.embed_tokens.padding_idx is None
    assert model.embed_tokens.weight[0].abs().sum() > 0
    model.embed_tokens(torch.tensor([0])).sum().backward()
    torch.testing.assert_close(model.embed_tokens.weight.grad[0], torch.ones(64))
    inputs = torch.tensor([[0, 0, 3, 4, 0, 5]])
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(inputs).reshape(-1, 128), inputs.reshape(-1))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


# ---------- config ----------

def test_config():
    config = _tiny_config(attention_every=2, rope_every=2)
    assert config.head_dim == 32
    assert config.is_full_attn(0) is False   # 1-indexed: layer 1 % 2 != 0
    assert config.is_full_attn(1) is True    # layer 2 % 2 == 0
    assert config.is_swa(0) is True
    assert config.use_rope(0) is False      # 1-indexed: layer 1 % 2 != 0 → NoPE
    assert config.use_rope(1) is True       # layer 2 % 2 == 0 → RoPE
    print(f"test_config: ok (head_dim={config.head_dim})")


def test_swa_rope_alternation():
    config = ModelConfig(num_layers=8, hidden_size=256, num_heads=4, num_kv_heads=2,
                         attention_every=4, rope_every=2)
    full_layers = [i for i in range(8) if config.is_full_attn(i)]
    swa_layers = [i for i in range(8) if config.is_swa(i)]
    rope_layers = [i for i in range(8) if config.use_rope(i)]
    nope_layers = [i for i in range(8) if not config.use_rope(i)]

    assert [i + 1 for i in full_layers] == [4, 8], f"expected layers [4,8], got {[i + 1 for i in full_layers]}"
    assert len(swa_layers) == 6
    assert [i + 1 for i in rope_layers] == [2, 4, 6, 8]
    assert len(nope_layers) == 4
    print("test_swa_rope_alternation: ok (full=[4,8], swa=6, rope=[2,4,6,8], nope=4)")


def test_config_layer_distribution_1indexed():
    config = ModelConfig(num_layers=32, hidden_size=256, num_heads=4, num_kv_heads=2, attention_every=6)
    full = [i for i in range(32) if config.is_full_attn(i)]
    assert [i + 1 for i in full] == [6, 12, 18, 24, 30]
    assert config.num_full_attn_layers == 5
    assert config.num_swa_layers == 27

    config = ModelConfig(num_layers=24, hidden_size=256, num_heads=4, num_kv_heads=2, attention_every=6)
    full = [i + 1 for i in range(24) if config.is_full_attn(i)]
    assert full == [6, 12, 18, 24]
    assert config.num_swa_layers == 20
    print("test_config_layer_distribution_1indexed: ok (ZGCM 32L: 5 full + 27 swa; 24L: 4 full + 20 swa)")


def test_config_validation():
    for bad in (
        dict(vocab_size=100),
        dict(num_heads=3),
        dict(num_kv_heads=3),
        dict(hidden_size=100, num_heads=8),
        dict(attention_every=0),
        dict(rope_percent=0.0),
    ):
        try:
            _tiny_config(**bad)
        except (ValueError, AssertionError):
            continue
        raise AssertionError(f"expected validation error for {bad}")
    try:
        ModelConfig(vocab_size=128, hidden_size=30, num_heads=2, num_kv_heads=1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected odd head_dim rejection")
    print("test_config_validation: ok")


def test_preset():
    config = ModelConfig.from_preset("tiny")
    assert config.num_layers == 4
    assert config.hidden_size == 256
    assert config.vocab_size == 32768

    half = ModelConfig.from_preset("pretrain-0.5b")
    assert half.num_layers == 32 and half.hidden_size == 1024
    assert half.intermediate_size == 2816
    assert half.num_heads == 16 and half.num_kv_heads == 8
    assert half.vocab_size == 65536 and half.max_seq_len == 4096
    assert half.attention_every == 6 and half.window_size == 128
    assert half.rope_base == 5_000_000.0 and abs(half.rope_percent - 0.334) < 1e-9
    assert half.qk_norm is True and half.attention_output_gate is True
    assert half.tie_word_embeddings is True   # 0.5B 档锁定 tied；消融对照 pretrain-0.5b-untied
    assert half.num_full_attn_layers == 5 and half.num_swa_layers == 27

    cs336_path = Path(__file__).resolve().parents[1] / "configs" / "pretrain_0.5b_cs336.yaml"
    cs336 = ModelConfig.from_preset("pretrain-0.5b-cs336", config_path=cs336_path)
    assert cs336.num_layers == 16 and cs336.hidden_size == 1536
    assert cs336.intermediate_size == 4096 and cs336.intermediate_size * 3 == cs336.hidden_size * 8
    assert cs336.num_heads == 24 and cs336.num_kv_heads == 6 and cs336.head_dim == 64
    assert cs336.hidden_size == cs336.num_heads * cs336.head_dim
    assert cs336.hidden_size / cs336.num_layers == 96
    assert cs336.tie_word_embeddings is True

    untied = ModelConfig.from_preset("pretrain-0.5b-untied")
    assert untied.tie_word_embeddings is False
    assert untied.num_layers == half.num_layers and untied.hidden_size == half.hidden_size
    print("test_preset: ok (tiny + pretrain-0.5b + pretrain-0.5b-untied)")


# ---------- forward / backward ----------

def test_forward_tiny():
    config = ModelConfig.from_preset("tiny")
    model = create_model(config)
    model.eval()

    tokens = torch.randint(0, config.vocab_size, (2, 128))
    with torch.no_grad():
        logits = model(tokens)

    assert logits.shape == (2, 128, config.vocab_size)
    assert logits.dtype == torch.float32
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


def test_max_seq_len_guard():
    config = _tiny_config(max_seq_len=64)
    model = create_model(config)
    model.eval()
    try:
        model(torch.randint(0, config.vocab_size, (1, 65)))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for seq_len > max_seq_len")
    print("test_max_seq_len_guard: ok")


def test_backward():
    config = ModelConfig.from_preset("tiny")
    model = create_model(config)
    model.train()

    tokens = torch.randint(0, config.vocab_size, (2, 64))
    logits = model(tokens)
    loss = F.cross_entropy(
        logits.view(-1, config.vocab_size),
        tokens.view(-1),
    )
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"{name} has NaN gradient"
    print(f"test_backward: ok (loss={loss.item():.4f})")


def test_tied_embeddings():
    config_tied = _tiny_config(num_layers=1, tie_word_embeddings=True)
    model_tied = create_model(config_tied)
    assert model_tied.lm_head is model_tied.embed_tokens

    config_untied = _tiny_config(num_layers=1, tie_word_embeddings=False)
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


# ---------- mask contract ----------

def test_full_layer_causal_with_padding_mask():
    torch.manual_seed(0)
    config = _tiny_config(num_layers=1, attention_every=1)
    model = create_model(config)
    model.eval()

    seq_len = 32
    tokens = torch.randint(0, config.vocab_size, (1, seq_len))
    padding_mask = torch.ones(1, seq_len, dtype=torch.bool)
    padding_mask[:, -4:] = False

    with torch.no_grad():
        logits = model(tokens, attention_mask=padding_mask)

    # causality: changing a future real token must not affect earlier positions
    future = seq_len - 5
    tokens2 = tokens.clone()
    tokens2[:, future] = (tokens2[:, future] + 7) % config.vocab_size
    with torch.no_grad():
        logits2 = model(tokens2, attention_mask=padding_mask)
    assert torch.allclose(logits[:, :future], logits2[:, :future], atol=1e-5), \
        "full layer lost causality when a padding mask was passed"

    # padding invariance: changing padded tokens must not affect real positions
    tokens3 = tokens.clone()
    tokens3[:, -4:] = (tokens3[:, -4:] + 3) % config.vocab_size
    with torch.no_grad():
        logits3 = model(tokens3, attention_mask=padding_mask)
    assert torch.allclose(logits[:, :future + 1], logits3[:, :future + 1], atol=1e-5), \
        "outputs at real positions depend on padded tokens"
    print("test_full_layer_causal_with_padding_mask: ok")


def test_swa_mask_no_nan_with_padding():
    torch.manual_seed(0)
    config = _tiny_config(num_layers=2, attention_every=6, window_size=8)
    assert config.num_swa_layers == 2
    model = create_model(config)
    model.eval()

    tokens = torch.randint(0, config.vocab_size, (2, 64))
    padding_mask = torch.ones(2, 64, dtype=torch.bool)
    padding_mask[1, -10:] = False

    with torch.no_grad():
        logits = model(tokens, attention_mask=padding_mask)
    assert not torch.isnan(logits).any()
    print("test_swa_mask_no_nan_with_padding: ok")


def test_swa_equals_full_when_window_covers_seq():
    torch.manual_seed(0)
    seq_len = 32
    cfg_full = _tiny_config(attention_every=1, window_size=seq_len)
    cfg_swa = _tiny_config(attention_every=6, window_size=seq_len)
    attn_full = GQAAttention(cfg_full, 0)
    attn_swa = GQAAttention(cfg_swa, 0)
    attn_swa.load_state_dict(attn_full.state_dict())

    cos, sin = precompute_rope_freqs(cfg_full.head_dim, seq_len, cfg_full.rope_base, cfg_full.rope_percent)
    x = torch.randn(1, seq_len, cfg_full.hidden_size)

    with torch.no_grad():
        out_full = attn_full(x, cos, sin)
        out_swa = attn_swa(x, cos, sin)
    assert torch.allclose(out_full, out_swa, atol=1e-5), \
        f"SWA with window>=seq_len diverges from full causal: {(out_full - out_swa).abs().max()}"
    print("test_swa_equals_full_when_window_covers_seq: ok")


# ---------- RoPE ----------

def test_rope():
    head_dim = 64
    cos, sin = precompute_rope_freqs(head_dim, 128, rope_base=10000.0, rope_percent=0.5)
    assert cos.shape == (128, 16) and sin.shape == (128, 16)

    x = torch.randn(1, 1, 128, head_dim)
    rotated = apply_rope(x, cos, sin)
    assert rotated.shape == x.shape
    assert not torch.equal(rotated, x)
    print("test_rope: ok")


def test_rope_rotary_dim_truncation():
    cos, _ = precompute_rope_freqs(128, 64, rope_base=10000.0, rope_percent=0.334)
    assert cos.size(-1) == 21, f"int(128*0.334)=42 → 21 pairs, got {cos.size(-1)}"
    print("test_rope_rotary_dim_truncation: ok (rot_dim=42)")


def test_rope_relative_position_invariance():
    torch.manual_seed(0)
    head_dim = 64
    cos, sin = precompute_rope_freqs(head_dim, 256, rope_base=10000.0, rope_percent=0.5)

    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)
    m, n, delta = 10, 4, 37

    q_m = apply_rope(q, cos, sin, torch.tensor([[m]]))
    k_n = apply_rope(k, cos, sin, torch.tensor([[n]]))
    q_m2 = apply_rope(q, cos, sin, torch.tensor([[m + delta]]))
    k_n2 = apply_rope(k, cos, sin, torch.tensor([[n + delta]]))

    dot1 = (q_m * k_n).sum().item()
    dot2 = (q_m2 * k_n2).sum().item()
    assert abs(dot1 - dot2) < 1e-4, f"RoPE inner product depends on absolute position: {dot1} vs {dot2}"
    print("test_rope_relative_position_invariance: ok")


def test_position_ids_wiring():
    torch.manual_seed(0)
    head_dim = 64
    seq_len = 16
    cos, sin = precompute_rope_freqs(head_dim, 128, rope_base=10000.0, rope_percent=0.5)
    x = torch.randn(1, 1, seq_len, head_dim)

    offset = 40
    pos_shifted = torch.arange(seq_len).unsqueeze(0) + offset
    via_ids = apply_rope(x, cos, sin, position_ids=pos_shifted)
    via_slice = apply_rope(x, cos[offset:offset + seq_len], sin[offset:offset + seq_len])
    assert torch.allclose(via_ids, via_slice, atol=1e-6), "position_ids gather does not match offset slice"

    config = _tiny_config(num_layers=1, attention_every=1)
    model = create_model(config)
    model.eval()
    tokens = torch.randint(0, config.vocab_size, (1, seq_len))
    with torch.no_grad():
        logits_default = model(tokens)
        # uniform offset: RoPE is relative, so a constant shift must not change output
        logits_offset = model(tokens, position_ids=pos_shifted)
        # reversed order changes every relative distance → output must change
        logits_reversed = model(tokens, position_ids=torch.flip(pos_shifted, dims=[1]))
    assert torch.allclose(logits_default, logits_offset, atol=1e-5), \
        "uniform position offset changed output (RoPE must be relative)"
    assert not torch.allclose(logits_default, logits_reversed, atol=1e-6), \
        "position_ids has no effect on model output"
    print("test_position_ids_wiring: ok")


# ---------- gate / qk-norm ----------

def test_attention_output_gate_params_and_effect():
    torch.manual_seed(0)
    config = _tiny_config(attention_every=6, window_size=8, attention_output_gate=True, qk_norm=False)
    attn_swa = GQAAttention(config, 0)          # 1-indexed layer 1 → SWA, gated
    attn_full = GQAAttention(config, 5)         # 1-indexed layer 6 → full, no gate

    d = config.hidden_size
    kv_w = config.num_kv_heads * config.head_dim
    assert attn_swa.has_gate and attn_full.has_gate is False
    assert attn_swa.qkv_proj.out_features == 2 * d + 2 * kv_w   # [q|g|k|v]
    assert attn_full.qkv_proj.out_features == d + 2 * kv_w      # [q|k|v]
    assert not any("gate" in n for n, _ in attn_full.named_parameters())

    # gate ≈ 0 (zero gate weights → pre-activation 0 → sigmoid = 0.5) halves the output
    ungated_cfg = _tiny_config(attention_every=6, window_size=8, attention_output_gate=False, qk_norm=False)
    attn_ref = GQAAttention(ungated_cfg, 0)
    with torch.no_grad():
        attn_ref.qkv_proj.weight.copy_(
            torch.cat([attn_swa.qkv_proj.weight[:d], attn_swa.qkv_proj.weight[2 * d:]], dim=0)
        )
        attn_ref.o_proj.weight.copy_(attn_swa.o_proj.weight)
        attn_swa.qkv_proj.weight[d:2 * d] = 0.0  # gate slice

    cos, sin = precompute_rope_freqs(config.head_dim, 32, config.rope_base, config.rope_percent)
    x = torch.randn(1, 32, d)
    with torch.no_grad():
        out_gated = attn_swa(x, cos, sin)
        out_ref = attn_ref(x, cos, sin)
    assert torch.allclose(out_gated, 0.5 * out_ref, atol=1e-5), \
        "gate with sigmoid≈0.5 does not scale output by 0.5"

    # strongly negative gate pre-activation → sigmoid ≈ 0 → output suppressed
    # (all-positive inputs make -50 * sum(x) unambiguously negative for every token)
    x_pos = x.abs() + 0.5
    with torch.no_grad():
        attn_swa.qkv_proj.weight[d:2 * d] = -50.0
        out_suppressed = attn_swa(x_pos, cos, sin)
    assert out_suppressed.abs().max() < 1e-3, \
        f"gate≈0 does not suppress output: max={out_suppressed.abs().max()}"
    print("test_attention_output_gate_params_and_effect: ok")


# ---------- flash 后端（E-52，可选：按能力 guard，无可用 flash-attn 则跳过） ----------

def test_flash_backend_guard():
    """E-52 guard：包在但版本不足（如 2.0.x 无 window_size）时，config init 与 forward
    都 fail-fast，报错指向 attn_impl=sdpa / 升级 flash-attn；能力满足时正常放行。"""
    pytest = __import__("pytest")
    from yuformer_model import attention as attn_mod

    assert attn_mod._parse_major_minor("2.0.4") == (2, 0)
    assert attn_mod._parse_major_minor("2.2.0a0") == (2, 2)
    assert attn_mod._parse_major_minor("2.3.0") == (2, 3)
    assert attn_mod._parse_major_minor("2.6.3+cu123") == (2, 6)
    assert attn_mod._parse_major_minor("garbage") is None

    saved = (attn_mod._HAS_FLASH_ATTN, attn_mod._FLASH_SUPPORTS_WINDOW, attn_mod._FLASH_ATTN_USABLE)
    try:
        # 场景 1：包未安装
        attn_mod._HAS_FLASH_ATTN = False
        attn_mod._FLASH_SUPPORTS_WINDOW = False
        attn_mod._FLASH_ATTN_USABLE = False
        err = attn_mod.flash_backend_error()
        assert err is not None and "未安装" in err and "sdpa" in err
        with pytest.raises(ValueError, match="attn_impl=flash"):
            _tiny_config(attn_impl="flash")

        # 场景 2：包在但版本不足（本机 flash_attn 2.0.4 的真实情形）
        attn_mod._HAS_FLASH_ATTN = True
        attn_mod._FLASH_SUPPORTS_WINDOW = False
        attn_mod._FLASH_ATTN_USABLE = False
        err = attn_mod.flash_backend_error()
        assert err is not None and "window_size" in err and "sdpa" in err
        with pytest.raises(ValueError, match="attn_impl=flash"):
            _tiny_config(attn_impl="flash")

        # forward 路径同样 fail-fast（绕过 config 校验直抵 flash 分支）
        cfg = _tiny_config(attn_impl="sdpa")
        attn = GQAAttention(cfg, 0)
        cfg.attn_impl = "flash"
        cos, sin = precompute_rope_freqs(cfg.head_dim, 32, cfg.rope_base, cfg.rope_percent)
        x = torch.randn(1, 16, cfg.hidden_size)
        with pytest.raises(RuntimeError, match="attn_impl=flash"):
            attn(x, cos, sin)

        # 场景 3：能力满足 → config 放行
        attn_mod._HAS_FLASH_ATTN = True
        attn_mod._FLASH_SUPPORTS_WINDOW = True
        attn_mod._FLASH_ATTN_USABLE = True
        assert attn_mod.flash_backend_error() is None
        cfg_flash = _tiny_config(attn_impl="flash")
        assert cfg_flash.attn_impl == "flash"
    finally:
        (attn_mod._HAS_FLASH_ATTN, attn_mod._FLASH_SUPPORTS_WINDOW,
         attn_mod._FLASH_ATTN_USABLE) = saved
    print("test_flash_backend_guard: ok")


def test_flash_backend_parity():
    pytest = __import__("pytest")
    from yuformer_model import attention as attn_mod
    if not torch.cuda.is_available() or not attn_mod._FLASH_ATTN_USABLE:
        pytest.skip(f"flash 后端不可用（{attn_mod.flash_backend_error()}），跳过 parity")

    torch.manual_seed(0)
    seq_len = 64
    cos_sin_len = 128
    for window, every in [(16, 6), (0, 1)]:  # SWA 层 + 全注意力层
        cfg = _tiny_config(attention_every=every, window_size=window or 16, qk_norm=True)
        attn = GQAAttention(cfg, 0)  # layer 1：every=6 → SWA；every=1 → full
        cos, sin = precompute_rope_freqs(cfg.head_dim, cos_sin_len, cfg.rope_base, cfg.rope_percent)
        x = torch.randn(2, seq_len, cfg.hidden_size)

        device = "cuda"
        attn = attn.to(device)
        cos, sin, x = cos.to(device), sin.to(device), x.to(device)
        attn_cfg_flash = _tiny_config(attention_every=every, window_size=window or 16,
                                      qk_norm=True, attn_impl="flash")
        attn_flash = GQAAttention(attn_cfg_flash, 0)
        attn_flash.load_state_dict(attn.state_dict())
        attn_flash = attn_flash.to(device)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out_ref = attn(x, cos, sin)
            out_flash = attn_flash(x, cos, sin)
        err = (out_ref.float() - out_flash.float()).abs()
        scale = out_ref.float().abs().mean()
        assert err.max() < 3e-2 and err.mean() < 5e-3, \
            f"flash parity fail (window={window}, every={every}): max={err.max()}, mean={err.mean()}, scale={scale}"
    print("test_flash_backend_parity: ok")


def test_qk_norm_bounds():
    torch.manual_seed(0)
    config = _tiny_config(attention_every=1, qk_norm=True)
    attn = GQAAttention(config, 0)

    x = torch.randn(4, 2, 16, config.head_dim) * 1000.0
    normed = attn.q_norm(x)
    rms = normed.float().pow(2).mean(dim=-1)
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4), f"q_norm RMS deviates from 1: {rms.mean()}"
    per_head_norm = normed.float().norm(dim=-1)
    assert torch.allclose(per_head_norm, math.sqrt(config.head_dim) * torch.ones_like(per_head_norm), rtol=1e-3)

    with torch.no_grad():
        logits_in = torch.randn(2, 32, config.hidden_size) * 1000.0
        out = attn(logits_in, *precompute_rope_freqs(config.head_dim, 32, config.rope_base, config.rope_percent))
    assert torch.isfinite(out.float()).all(), "QK-normed attention blew up on large inputs"
    print("test_qk_norm_bounds: ok")


# ---------- GQA ----------

def test_gqa_equals_mha():
    torch.manual_seed(0)
    seq_len = 32
    config = _tiny_config(num_kv_heads=2)  # kv == heads → MHA
    attn = GQAAttention(config, 0)
    attn.eval()

    cos, sin = precompute_rope_freqs(config.head_dim, seq_len, config.rope_base, config.rope_percent)
    x = torch.randn(1, seq_len, config.hidden_size)
    with torch.no_grad():
        out_module = attn(x, cos, sin)

    with torch.no_grad():
        qkv = attn.qkv_proj(x)
        q, k, v = torch.split(qkv, [config.num_heads * config.head_dim] * 3, dim=-1)
        q = q.view(1, seq_len, config.num_heads, config.head_dim).transpose(1, 2)
        k = k.view(1, seq_len, config.num_heads, config.head_dim).transpose(1, 2)
        v = v.view(1, seq_len, config.num_heads, config.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out_mha = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out_mha = attn.o_proj(out_mha.transpose(1, 2).reshape(1, seq_len, -1))

    assert torch.allclose(out_module, out_mha, atol=1e-5), \
        f"GQA module with kv==heads diverges from MHA: {(out_module - out_mha).abs().max()}"
    print("test_gqa_equals_mha: ok")


# ---------- parameter accounting ----------

def test_preset_param_counts():
    for name in ("tiny", "small", "base", "medium", "pretrain-0.5b", "small-full-attn", "small-all-rope"):
        config = ModelConfig.from_preset(name)
        model = create_model(config)
        expected = analytical_param_count(config)
        actual = model.num_parameters()
        assert abs(actual - expected) / expected < 0.02, \
            f"preset {name}: actual {actual:,} vs analytical {expected:,}"

    config = ModelConfig.from_preset("large")
    with torch.device("meta"):
        model = create_model(config)
    expected = analytical_param_count(config)
    actual = model.num_parameters()
    assert abs(actual - expected) / expected < 0.02, \
        f"preset large: actual {actual:,} vs analytical {expected:,}"
    assert actual == 7_197_700_096
    candidate_path = Path(__file__).resolve().parents[1] / "configs" / "pretrain_0.5b_cs336.yaml"
    candidate = ModelConfig.from_preset("pretrain-0.5b-cs336", config_path=candidate_path)
    with torch.device("meta"):
        candidate_model = create_model(candidate)
    candidate_expected = analytical_param_count(candidate)
    candidate_actual = candidate_model.num_parameters()
    assert candidate_actual == candidate_expected == 530_107_904
    print("test_preset_param_counts: ok (all presets within 2% of analytical formula)")


# ---------- dtype smoke ----------

def test_bf16_forward_backward():
    config = ModelConfig.from_preset("tiny")
    model = create_model(config).to(torch.bfloat16)
    model.train()

    tokens = torch.randint(0, config.vocab_size, (2, 64))
    logits = model(tokens)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()

    loss = F.cross_entropy(logits.view(-1, config.vocab_size), tokens.view(-1))
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"{name} has no gradient"
            assert torch.isfinite(param.grad.float()).all(), f"{name} has non-finite gradient"
    print(f"test_bf16_forward_backward: ok (loss={loss.item():.4f})")


if __name__ == "__main__":
    test_config()
    test_config_layer_distribution_1indexed()
    test_config_validation()
    test_preset()
    test_rope()
    test_rope_rotary_dim_truncation()
    test_rope_relative_position_invariance()
    test_position_ids_wiring()
    test_forward_tiny()
    test_forward_small()
    test_swa_rope_alternation()
    test_full_layer_causal_with_padding_mask()
    test_swa_mask_no_nan_with_padding()
    test_swa_equals_full_when_window_covers_seq()
    test_attention_output_gate_params_and_effect()
    test_qk_norm_bounds()
    test_gqa_equals_mha()
    test_max_seq_len_guard()
    test_different_seq_lens()
    test_backward()
    test_tied_embeddings()
    test_param_count()
    test_preset_param_counts()
    test_bf16_forward_backward()
    print("\nAll model tests passed!")
