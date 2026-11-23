import importlib.util
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "backends/torchtitan/validate_contract.py"
SPEC = importlib.util.spec_from_file_location("validate_contract", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def _fixture(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / "model/src").mkdir(parents=True)
    (repo / "model/configs").mkdir(parents=True)
    (repo / "pretrain/backends/torchtitan").mkdir(parents=True)
    origin = repo/'experiments/e4_curriculum/scripts/training_data_origin.py'
    origin.parent.mkdir(parents=True)
    shutil.copyfile(SCRIPT.parents[3]/'experiments/e4_curriculum/scripts/training_data_origin.py', origin)
    (repo / "model/src/model.py").write_text("x = 1\n")
    (repo / "model/configs/model.yaml").write_text(yaml.safe_dump({
        "presets": {"test": {"vocab_size": 64, "max_seq_len": 16}}
    }))
    (repo / "pretrain/backends/torchtitan/adapter.py").write_text("x = 1\n")
    tokenizer = repo / "tokenizer.json"
    tokenizer.write_text("{}\n")
    tokenizer_sha256 = MODULE._sha256(tokenizer)
    gate = {
        "status": "PASS", "dataset_id": "data-a", "snapshot_id": "snap-a",
        "tokenizer_id": "tok-a", "tokenizer_path": "tokenizer.json",
        "tokenizer_sha256": tokenizer_sha256, "vocab_size": 64,
    }
    release = {"status": "PASS", "dataset_id": "data-a", "snapshot_id": "snap-a"}
    (repo / "gate.json").write_text(json.dumps(gate))
    (repo / "release.json").write_text(json.dumps(release))
    tokens = np.arange(65, dtype=np.uint16) % 64
    binary = repo / "stage.tokens.bin"
    tokens.tofile(binary)
    meta = {
        "bytes": binary.stat().st_size, "total_tokens": tokens.size, "dtype": "<u2",
        "edge_sha256": MODULE._edge_sha256(binary), "tokenizer_sha256": tokenizer_sha256,
        "vocab_size": 64, "data_snapshot_id": "snap-a", "n_docs": 1,
        "document_boundary": {"policy": "append_eod", "count": 1, "token_id": 0},
    }
    meta["sha256"] = MODULE._sha256(binary)
    (repo / "stage.meta.json").write_text(json.dumps(meta))
    for domain in ("web", "code", "math", "chinese", "academic", "books",
                   "encyclopedia", "qa", "multilingual"):
        val_binary = repo / f"val_{domain}.tokens.bin"
        tokens.tofile(val_binary)
        (repo / f"val_{domain}.meta.json").write_text(json.dumps(meta))
    audit = {
        "status": "PASS", "roster_exact": True, "dataset_id": "data-a",
        "snapshot_id": "snap-a", "tokenizer_sha256": tokenizer_sha256,
        "streams": [{
            "status": "PASS", "actual_sha256": meta["sha256"],
            "actual_bytes": binary.stat().st_size,
            "artifacts": {"stream": str(binary)},
        }, *[
            {"name": f"val_{domain}", "status": "PASS",
             "actual_sha256": meta["sha256"], "actual_bytes": binary.stat().st_size,
             "artifacts": {"stream": str(repo / f"val_{domain}.tokens.bin")}}
            for domain in ("web", "code", "math", "chinese", "academic", "books",
                           "encyclopedia", "qa", "multilingual")
        ]],
    }
    (repo / "stream-audit.json").write_text(json.dumps(audit))
    cfg = {
        "run_name": "test", "backend": "torchtitan-v0.3.0", "seed": 7,
        "out_dir": str(repo / "out"),
        "model": {"preset": "test", "configs": str(repo / "model/configs/model.yaml")},
        "data": {"seq_len": 16, "tokenizer_gate": str(repo / "gate.json"),
                 "release_manifest": str(repo / "release.json"),
                 "token_stream_audit": str(repo / "stream-audit.json"),
                 "stages": [str(binary)], "meta": [str(repo / "stage.meta.json")],
                 "val_dir": str(repo),
                 "val_samples_per_domain": 4},
        "train": {"budget_tokens": 64, "micro_batch_size": 1, "grad_accum": 1,
                  "expected_iters": 4, "save_every": 2, "eval_every": 2, "log_every": 1},
        "optim": {"optimizer": "adamw", "lr": 0.001, "clip_grad": 1.0,
                  "warmup_steps": 0, "lr_schedule": "cosine"},
        "parallel": {"world_size": 1, "sharding": "full_shard"},
    }
    config = repo / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))
    return repo, config


def test_validator_publishes_stable_identity(tmp_path):
    repo, config = _fixture(tmp_path)
    first = MODULE.validate(repo, config, publish_identity=True)
    second = MODULE.validate(repo, config, publish_identity=True)
    assert first == second
    assert json.loads((repo / "out/run_identity.json").read_text()) == first["identity"]
    assert (repo / "out/resolved_config.yaml").read_text() == config.read_text()


def test_explicit_qualification_contract_remains_launchable(tmp_path):
    repo, config = _fixture(tmp_path)
    cfg = yaml.safe_load(config.read_text())
    cfg["experiment_status"] = "qualification"
    config.write_text(yaml.safe_dump(cfg))
    assert MODULE.validate(repo, config, publish_identity=False)["status"] == "PASS"


def test_full_production_rejects_a_sequential_contract(tmp_path):
    repo, config = _fixture(tmp_path)
    with pytest.raises(ValueError, match='explicit independent block-order'):
        MODULE.validate(repo, config, publish_identity=True, require_production_order=True)
    assert not (repo/'out/run_identity.json').exists()


def test_validator_binds_block_bytes_and_rejects_probe_for_full_production(tmp_path):
    repo, config = _fixture(tmp_path)
    cfg = yaml.safe_load(config.read_text())
    array = repo/'blocks.u32'
    np.array([3, 1, 0, 2], dtype='<u4').tofile(array)
    manifest = {'schema_version': 2, 'scope': 'PRODUCTION', 'status': 'PASS',
        'purpose': 'TRAINING_CAUSAL_BLOCK_PERMUTATION',
        'algorithm': 'blake2b-128-independent-causal-block-sort-v1',
        'seed': 'separate-from-selection', 'seq_len': 16, 'block_samples': 1,
        'release_manifest_sha256': MODULE._sha256(repo/'release.json'),
        'dataset_id': 'data-a', 'snapshot_id': 'snap-a',
        'stages': [{'stage_index': 0, 'stage_sample_limit': 4, 'permutation_population_samples': 4,
            'stream_meta_sha256': MODULE._sha256(repo/'stage.meta.json'),
            'permutation': {'file': array.name, 'sha256': MODULE._sha256(array), 'bytes': 16, 'dtype': '<u4'}}]}
    master = repo/'order.json'
    master.write_text(json.dumps(manifest))
    cfg['data'].update(sampling_policy='block_permutation', stage_sample_limits=[4],
                       block_permutation_manifest=str(master))
    config.write_text(yaml.safe_dump(cfg))
    validated = MODULE.validate(repo, config, publish_identity=False, require_production_order=True)
    assert validated['identity']['training_sample_order']['manifest_sha256'] == MODULE._sha256(master)
    assert validated['identity']['training_sample_order']['artifacts'][0]['sha256'] == MODULE._sha256(array)
    probe = dict(manifest, scope='PREFIX_PROBE',
                 parent_manifest={'file': master.name, 'sha256': MODULE._sha256(master)})
    probe['stages'] = [dict(manifest['stages'][0], parent_stage_index=0)]
    child = repo/'probe.json'
    child.write_text(json.dumps(probe))
    cfg['data']['block_permutation_manifest'] = str(child)
    config.write_text(yaml.safe_dump(cfg))
    assert MODULE.validate(repo, config, publish_identity=False)['status'] == 'PASS'
    with pytest.raises(ValueError, match='cannot use a prefix-probe'):
        MODULE.validate(repo, config, publish_identity=True, require_production_order=True)
    np.array([0, 0, 0, 0], dtype='<u4').tofile(array)
    with pytest.raises(ValueError, match='SHA-256'):
        MODULE.validate(repo, config, publish_identity=False)


@pytest.mark.parametrize("section,key,value", [
    (None, "resume", "/wrong/checkpoint"),
    (None, "init_weights_from", "/weights/never-loaded"),
    ("data", "sampling_policy", "round_robin"),
    ("data", "curriculum", {"enabled": True}),
    ("optim", "warmup_tokens", 1000),
    ("train", "gradient_accumulation", 8),
])
def test_ignored_training_options_fail_before_any_identity_is_published(tmp_path, section, key, value):
    repo, config = _fixture(tmp_path)
    cfg = yaml.safe_load(config.read_text())
    target = cfg if section is None else cfg[section]
    target[key] = value
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="unsupported (config fields|sampling policy)"):
        MODULE.validate(repo, config, publish_identity=True)
    assert not (repo / "out").exists()


def test_duplicate_yaml_warmup_is_rejected(tmp_path):
    repo, config = _fixture(tmp_path)
    text = config.read_text().replace("warmup_steps: 0", "warmup_steps: 0\n  warmup_steps: 100")
    config.write_text(text)
    with pytest.raises(ValueError, match="duplicate YAML key"):
        MODULE.validate(repo, config, publish_identity=False)


def test_deterministic_runtime_requires_an_actual_boolean(tmp_path):
    repo, config = _fixture(tmp_path)
    cfg = yaml.safe_load(config.read_text())
    cfg['train']['deterministic'] = 'false'
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='deterministic must be a boolean'):
        MODULE.validate(repo, config, publish_identity=True)
    assert not (repo/'out').exists()
    cfg['train']['deterministic'] = True
    config.write_text(yaml.safe_dump(cfg))
    assert MODULE.validate(repo, config, publish_identity=False)['status'] == 'PASS'


def test_reshard_policy_is_explicit_and_typo_cannot_be_ignored(tmp_path):
    repo, config = _fixture(tmp_path)
    cfg = yaml.safe_load(config.read_text())
    cfg['parallel']['fsdp_reshard_after_forward'] = 'nevre'
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='fsdp_reshard_after_forward'):
        MODULE.validate(repo, config, publish_identity=True)
    assert not (repo/'out').exists()
    cfg['parallel']['fsdp_reshard_after_forward'] = 'never'
    config.write_text(yaml.safe_dump(cfg))
    assert MODULE.validate(repo, config, publish_identity=False)['status'] == 'PASS'


def test_duplicate_model_context_setting_is_rejected(tmp_path):
    repo, config = _fixture(tmp_path)
    model = repo / "model/configs/model.yaml"
    model.write_text("presets:\n  test:\n    vocab_size: 64\n    max_seq_len: 16\n    max_seq_len: 4096\n")
    with pytest.raises(ValueError, match="duplicate YAML key"):
        MODULE.validate(repo, config, publish_identity=False)


def test_wsd_rejects_data_and_lr_phase_disagreement(tmp_path):
    repo, config = _fixture(tmp_path)
    cfg = yaml.safe_load(config.read_text())
    cfg["data"]["stages"] *= 3
    cfg["data"]["meta"] *= 3
    cfg["data"]["stage_sample_limits"] = [1, 2, 1]
    cfg["optim"].update(lr_schedule="wsd", warmup_steps=2, decay_steps=1)
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="data boundaries must match LR"):
        MODULE.validate(repo, config, publish_identity=False)
    cfg["optim"]["warmup_steps"] = 1
    config.write_text(yaml.safe_dump(cfg))
    assert MODULE.validate(repo, config, publish_identity=False)["status"] == "PASS"


@pytest.mark.parametrize("section,key,value", [
    ("optim", "warmup_steps", 1.5),
    ("data", "stage_sample_limits", [4.9]),
    ("train", "bf16", "false"),
    ("train", "keep_every", -1),
])
def test_ambiguous_training_values_fail(tmp_path, section, key, value):
    repo, config = _fixture(tmp_path)
    cfg = yaml.safe_load(config.read_text())
    cfg[section][key] = value
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        MODULE.validate(repo, config, publish_identity=False)


def test_model_eos_must_match_stream_eod(tmp_path):
    repo, config = _fixture(tmp_path)
    path = repo / 'model/configs/model.yaml'
    doc = yaml.safe_load(path.read_text())
    doc['presets']['test']['eos_token_id'] = 2
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match='model EOS and stream EOD ids differ'):
        MODULE.validate(repo, config, publish_identity=False)


def test_validator_rejects_stream_mutation(tmp_path):
    repo, config = _fixture(tmp_path)
    MODULE.validate(repo, config, publish_identity=True)
    with (repo / "stage.tokens.bin").open("r+b") as fh:
        fh.write(b"xx")
    with pytest.raises(ValueError, match="edge hash"):
        MODULE.validate(repo, config, publish_identity=False)


def test_validator_rejects_tokenizer_artifact_mutation(tmp_path):
    repo, config = _fixture(tmp_path)
    (repo / "tokenizer.json").write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match="tokenizer artifact hash mismatch"):
        MODULE.validate(repo, config, publish_identity=False)


def test_identity_binds_selected_model_config_file(tmp_path):
    repo, config = _fixture(tmp_path)
    first = MODULE.validate(repo, config, publish_identity=False)
    with (repo / "model/configs/model.yaml").open("a") as handle:
        handle.write("x: 2\n")
    second = MODULE.validate(repo, config, publish_identity=False)
    assert first["identity"]["source_sha256"] == second["identity"]["source_sha256"]
    assert first["identity"]["model_configs_sha256"] != second["identity"]["model_configs_sha256"]


def test_validator_rejects_unbound_full_audit(tmp_path):
    repo, config = _fixture(tmp_path)
    audit_path = repo / "stream-audit.json"
    audit = json.loads(audit_path.read_text())
    audit["streams"][0]["actual_sha256"] = "f" * 64
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="full audit hash"):
        MODULE.validate(repo, config, publish_identity=False)


def test_validator_rejects_stream_changed_after_v2_full_audit(tmp_path):
    repo, config = _fixture(tmp_path)
    binary = repo / "stage.tokens.bin"
    stat = binary.stat()
    audit_path = repo / "stream-audit.json"
    audit = json.loads(audit_path.read_text())
    audit["schema_version"] = 2
    audit["streams"][0].update({
        "actual_device": stat.st_dev,
        "actual_inode": stat.st_ino,
        "actual_mtime_ns": stat.st_mtime_ns,
    })
    for row in audit["streams"][1:]:
        val_stat = Path(row["artifacts"]["stream"]).stat()
        row.update(actual_device=val_stat.st_dev, actual_inode=val_stat.st_ino,
                   actual_mtime_ns=val_stat.st_mtime_ns)
    audit_path.write_text(json.dumps(audit))
    MODULE.validate(repo, config, publish_identity=False)
    os.utime(binary, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match="changed after full audit"):
        MODULE.validate(repo, config, publish_identity=False)


def test_production_validator_binds_audited_document_length_ledger(tmp_path):
    repo, config = _fixture(tmp_path)
    binary = repo / "stage.tokens.bin"
    meta_path = repo / "stage.meta.json"
    audit_path = repo / "stream-audit.json"
    lengths = repo / "stage.doc_lengths.u32"
    np.asarray([65], dtype="<u4").tofile(lengths)
    meta = json.loads(meta_path.read_text())
    meta.update({
        "schema_version": 3,
        "doc_lengths": {"file": lengths.name, "dtype": "<u4", "rows": 1,
                        "bytes": lengths.stat().st_size, "sha256": MODULE._sha256(lengths),
                        "edge_sha256": MODULE._edge_sha256(lengths), "sum_tokens": 65,
                        "includes_appended_eod": True, "position_order": True},
    })
    meta_path.write_text(json.dumps(meta))
    stream_stat, length_stat = binary.stat(), lengths.stat()
    audit = json.loads(audit_path.read_text())
    stream_tool = repo / "experiments/e4_curriculum/scripts/audit_1t_token_streams.py"
    shared_tool = repo / "pretrain/scripts/audit_token_stream_release.py"
    release_tool = repo / "experiments/e4_curriculum/scripts/build_1t_order_release.py"
    for path in (stream_tool, shared_tool, release_tool):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {path.name}\n", encoding="utf-8")
    audit.update({"schema_version": 2, "audit": "pretrain_1t_token_streams",
                  "train_doc_lengths_audited": True,
                  "audit_tool_sha256": MODULE._sha256(stream_tool),
                  "shared_auditor_sha256": MODULE._sha256(shared_tool)})
    audit["streams"][0].update({
        "actual_device": stream_stat.st_dev, "actual_inode": stream_stat.st_ino,
        "actual_mtime_ns": stream_stat.st_mtime_ns,
        "actual_doc_lengths_bytes": length_stat.st_size,
        "actual_doc_lengths_sha256": MODULE._sha256(lengths),
        "actual_doc_lengths_device": length_stat.st_dev,
        "actual_doc_lengths_inode": length_stat.st_ino,
        "actual_doc_lengths_mtime_ns": length_stat.st_mtime_ns,
        "artifacts": {"stream": str(binary), "doc_lengths": str(lengths)},
    })
    for row in audit["streams"][1:]:
        val_stat = Path(row["artifacts"]["stream"]).stat()
        row.update(actual_device=val_stat.st_dev, actual_inode=val_stat.st_ino,
                   actual_mtime_ns=val_stat.st_mtime_ns)
    audit_path.write_text(json.dumps(audit))
    release_path = repo / "release.json"
    release = json.loads(release_path.read_text())
    release["builder_tool_sha256"] = MODULE._sha256(release_tool)
    freeze_path = repo / "experiments/e4_curriculum/results/frozen_1t_recipe.json"
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    freeze_path.write_text('{"status":"PASS"}\n', encoding="utf-8")
    release["frozen_recipe_sha256"] = MODULE._sha256(freeze_path)
    release_path.write_text(json.dumps(release))
    MODULE.validate(repo, config, publish_identity=False)
    os.utime(lengths, ns=(length_stat.st_atime_ns, length_stat.st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match="ledger changed after full audit"):
        MODULE.validate(repo, config, publish_identity=False)


def test_validator_rejects_missing_seed(tmp_path):
    repo, config = _fixture(tmp_path)
    cfg = yaml.safe_load(config.read_text())
    del cfg["seed"]
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="integer seed"):
        MODULE.validate(repo, config, publish_identity=False)


@pytest.mark.parametrize("section,key,value,message", [
    ("train", "cuda_graphs", True, "CUDA graphs"),
    ("parallel", "sharding", "no_shard", "supports full_shard and hybrid_shard"),
    ("optim", "adam_betas", [0.9, 1.0], "Adam betas"),
    ("optim", "adam_eps", 0, "epsilon"),
    ("optim", "weight_decay", float("nan"), "weight decay"),
])
def test_validator_refuses_unsupported_or_invalid_optimizer_runtime(tmp_path, section, key, value, message):
    repo, config = _fixture(tmp_path)
    cfg = yaml.safe_load(config.read_text())
    cfg[section][key] = value
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match=message):
        MODULE.validate(repo, config, publish_identity=False)


def test_validator_accepts_valid_hybrid_shard_topology(tmp_path):
    repo, config = _fixture(tmp_path)
    cfg = yaml.safe_load(config.read_text())
    cfg["parallel"]["world_size"] = 4
    cfg["train"]["budget_tokens"] = 64
    cfg["train"]["expected_iters"] = 1
    cfg["parallel"].update({
        "sharding": "hybrid_shard", "replicate_degree": 2, "shard_degree": 2,
    })
    config.write_text(yaml.safe_dump(cfg))
    result = MODULE.validate(repo, config, publish_identity=False)
    assert result["status"] == "PASS"


def test_validator_refuses_hybrid_shard_wrong_degree_product(tmp_path):
    repo, config = _fixture(tmp_path)
    cfg = yaml.safe_load(config.read_text())
    cfg["parallel"].update({
        "sharding": "hybrid_shard", "replicate_degree": 2, "shard_degree": 3,
    })
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="must equal world_size"):
        MODULE.validate(repo, config, publish_identity=False)


def test_validator_rejects_aggregate_only_validation_alignment(tmp_path):
    repo, config = _fixture(tmp_path)
    cfg = yaml.safe_load(config.read_text())
    # Nine two-sample domains have 18 aggregate samples, divisible by the
    # six-sample global batch, but no domain can form a domain-pure batch.
    audit_path = repo / "stream-audit.json"
    audit = json.loads(audit_path.read_text())
    audit_path.write_text(json.dumps(audit))
    cfg["data"]["val_samples_per_domain"] = 2
    cfg["parallel"]["world_size"] = 2
    cfg["train"]["micro_batch_size"] = 3
    cfg["train"]["budget_tokens"] = 96
    cfg["train"]["expected_iters"] = 1
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="each domain"):
        MODULE.validate(repo, config, publish_identity=False)


def test_validator_rejects_wrong_nine_domain_roster(tmp_path):
    repo, config = _fixture(tmp_path)
    audit_path = repo / "stream-audit.json"
    audit = json.loads(audit_path.read_text())
    audit["streams"][-1]["name"] = "val_wrong"
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="domain roster is not exact"):
        MODULE.validate(repo, config, publish_identity=False)


def test_validator_accepts_exact_stage_sample_limits(tmp_path):
    repo, config = _fixture(tmp_path)
    cfg = yaml.safe_load(config.read_text())
    cfg["data"]["stage_sample_limits"] = [4]
    config.write_text(yaml.safe_dump(cfg))
    result = MODULE.validate(repo, config, publish_identity=False)
    assert result["identity"]["streams"][0]["sample_limit"] == 4


def test_validator_rejects_stage_limit_above_capacity(tmp_path):
    repo, config = _fixture(tmp_path)
    cfg = yaml.safe_load(config.read_text())
    cfg["data"]["stage_sample_limits"] = [5]
    cfg["train"].update({"budget_tokens": 80, "expected_iters": 5})
    config.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="exceeds stream capacity"):
        MODULE.validate(repo, config, publish_identity=False)
