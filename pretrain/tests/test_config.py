import pytest

from yuformer_pretrain.config import ConfigError, load_config


def _write(tmp_path, budget, expected, world=1, mbs=2, accum=2, seq_len=64, text=None):
    import yaml

    cfg = {
        "run_name": "t",
        "seed": 1,
        "model": {"preset": "tiny", "configs": "/x/model_configs.yaml"},
        "out_dir": str(tmp_path / "out"),
        "data": {"stages": ["/x/a.tokens.bin"], "meta": ["/x/a.meta.json"], "seq_len": seq_len},
        "optim": {"optimizer": "adamw", "lr": 1e-4},
        "train": {
            "budget_tokens": budget,
            "micro_batch_size": mbs,
            "grad_accum": accum,
            "expected_iters": expected,
        },
        "parallel": {"world_size": world},
    }
    if text is not None:
        cfg = text
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_tokens_per_step_and_derived_iters(tmp_path):
    p = _write(tmp_path, budget=64 * 2 * 2 * 10, expected=10)
    cfg = load_config(p)
    assert cfg.tokens_per_step == 64 * 2 * 2
    assert cfg.derived_iters == 10


def test_expected_iters_guard(tmp_path):
    p = _write(tmp_path, budget=64 * 2 * 2 * 10, expected=11)
    with pytest.raises(ConfigError, match="身份守卫"):
        load_config(p)
    cfg = load_config(p, allow_iter_override=True)
    assert cfg.derived_iters == 10


def test_missing_env_fails_fast(tmp_path, monkeypatch):
    monkeypatch.delenv("YUFORMER_TEST_STREAM", raising=False)
    p = _write(tmp_path, budget=2560, expected=10)
    text = p.read_text().replace("/x/a.tokens.bin", "${YUFORMER_TEST_STREAM}")
    p.write_text(text)
    with pytest.raises(ConfigError, match="YUFORMER_TEST_STREAM"):
        load_config(p)


def test_env_resolves(tmp_path, monkeypatch):
    monkeypatch.setenv("YUFORMER_TEST_STREAM", "/data/streams/a.tokens.bin")
    p = _write(tmp_path, budget=2560, expected=10)
    p.write_text(p.read_text().replace("/x/a.tokens.bin", "${YUFORMER_TEST_STREAM}"))
    cfg = load_config(p)
    assert cfg.data.stages == ["/data/streams/a.tokens.bin"]


def test_optimizer_choice_guard(tmp_path):
    p = _write(tmp_path, budget=2560, expected=10)
    p.write_text(p.read_text().replace("adamw", "sgd"))
    with pytest.raises(ConfigError, match="optimizer"):
        load_config(p)


def test_invalidated_experiment_cannot_launch(tmp_path):
    p = _write(tmp_path, budget=2560, expected=10)
    text = p.read_text()
    p.write_text("experiment_status: invalidated\n" + text)
    with pytest.raises(ConfigError, match="禁止启动"):
        load_config(p)


def test_backend_is_explicit_and_validated(tmp_path):
    p = _write(tmp_path, budget=2560, expected=10)
    assert load_config(p).backend == "pytorch-fsdp1-compat"
    p.write_text("backend: torchtitan-v0.3.0\n" + p.read_text())
    assert load_config(p).backend == "torchtitan-v0.3.0"
    p.write_text(p.read_text().replace("torchtitan-v0.3.0", "unknown"))
    with pytest.raises(ConfigError, match="unsupported training backend"):
        load_config(p)
