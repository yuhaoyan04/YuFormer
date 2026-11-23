import json

import pytest

from yuformer_pretrain.run_identity import (
    build_run_identity,
    ensure_run_identity,
    source_tree_sha256,
    write_attempt_manifest,
)


def _identity(tmp_path, *, config=None):
    repo = tmp_path / "repo"
    source = repo / "src"
    source.mkdir(parents=True, exist_ok=True)
    source_file = source / "model.py"
    if not source_file.exists():
        source_file.write_text("VALUE = 1\n")
    model_config = repo / "model.yaml"
    if not model_config.exists():
        model_config.write_text("hidden: 16\n")
    return build_run_identity(
        run_name="test",
        config=config or {"seed": 1},
        model_config_path=model_config,
        tokenizer_sha256="tok-sha",
        data=[{"path": "tokens.bin", "sha256": "data-sha"}],
        dataset_release={"dataset_id": "data-test"},
        source_paths=[source],
        repo_root=repo,
        backend="test-backend",
    )


def test_identity_is_stable_and_detects_source_change(tmp_path):
    first = _identity(tmp_path)
    second = _identity(tmp_path)
    assert first == second
    (tmp_path / "repo" / "src" / "model.py").write_text("VALUE = 2\n")
    third = _identity(tmp_path)
    assert third["run_id"] != first["run_id"]


def test_identity_refuses_mutation_and_legacy_adoption(tmp_path):
    identity = _identity(tmp_path)
    out = tmp_path / "out"
    ensure_run_identity(out, identity)
    assert json.loads((out / "run_identity.json").read_text()) == identity
    ensure_run_identity(out, identity)
    with pytest.raises(RuntimeError, match="changed fields"):
        ensure_run_identity(out, _identity(tmp_path, config={"seed": 2}))

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "metrics.jsonl").write_text("")
    with pytest.raises(RuntimeError, match="legacy out_dir"):
        ensure_run_identity(legacy, identity)


def test_identity_allows_writer_lease_control_entries(tmp_path):
    identity = _identity(tmp_path)
    out = tmp_path / "leased"
    (out / ".run.lease" / "generation-00000000000000000001").mkdir(parents=True)
    (out / "run.lock").write_text("{}")
    ensure_run_identity(out, identity)
    assert json.loads((out / "run_identity.json").read_text()) == identity


def test_attempt_manifest_is_separate_from_identity(tmp_path):
    identity = _identity(tmp_path)
    out = tmp_path / "out"
    ensure_run_identity(out, identity)
    attempt = write_attempt_manifest(out, identity, world_size=4, argv=["train"])
    payload = json.loads(attempt.read_text())
    assert payload["run_id"] == identity["run_id"]
    assert payload["world_size"] == 4
    assert json.loads((out / "run_identity.json").read_text()) == identity


def test_source_hash_includes_untracked_style_files(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "new.py"
    source.write_text("x = 1\n")
    before = source_tree_sha256([root], root)
    source.write_text("x = 2\n")
    assert source_tree_sha256([root], root) != before
