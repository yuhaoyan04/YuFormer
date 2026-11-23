import json
from pathlib import Path
import shutil

import pytest
import yaml

import run_production_recovery as recovery


@pytest.fixture
def root(tmp_path):
    source = Path(__file__).resolve().parents[2]
    for relative in ("pretrain/configs/pretrain_base.yaml", "pretrain/configs/pretrain_base.env",
                     "pretrain/configs/qualification/q3_recovery_reference.yaml",
                     "pretrain/configs/qualification/q3_recovery_reference.env",
                     "pretrain/scripts/run_production_recovery.py",
                     "pretrain/scripts/production_probe_contracts.py",
                     "pretrain/scripts/derive_block_prefix_probe.py"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    path = tmp_path / "pretrain/configs/pretrain_base.yaml"
    base = yaml.safe_load(path.read_text())
    base["parallel"] = {"world_size": 8, "sharding": "hybrid_shard", "replicate_degree": 2, "shard_degree": 4}
    base["attention"] = {"sdpa_backend": "cudnn"}
    base['data'].update(sampling_policy='block_permutation',
        block_permutation_manifest=str(tmp_path/'order.json'))
    (tmp_path/'order.json').write_text(json.dumps({
        'schema_version': 2, 'scope': 'PRODUCTION', 'status': 'PASS',
        'purpose': 'TRAINING_CAUSAL_BLOCK_PERMUTATION',
        'algorithm': 'blake2b-128-independent-causal-block-sort-v1', 'block_samples': 8,
        'stages': [{'stage_index': i, 'stage_sample_limit': n,
                    'permutation_population_samples': n, 'permutation': {'file': f'{i}.u32'}}
                   for i, n in enumerate(base['data']['stage_sample_limits'])]}))
    path.write_text(yaml.safe_dump(base))
    health = tmp_path / "data/health"
    health.mkdir(parents=True)
    (health / "torchtitan-env-freeze.txt").write_text("torch==2.14.0+cu126\n")
    (health / "torchtitan-qualification.json").write_text('{"status":"PASS"}\n')
    return tmp_path


def publish_contracts(root):
    documents, production = recovery.contracts(root, publish_probe=True)
    for path, text in documents.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return documents, production


@pytest.mark.parametrize("accumulation", [1, 2, 16])
def test_recovery_uses_actual_production_graph_and_matching_fault_contracts(root, accumulation):
    path = root / "pretrain/configs/pretrain_base.yaml"
    base = yaml.safe_load(path.read_text())
    base["train"]["grad_accum"] = accumulation
    path.write_text(yaml.safe_dump(base))
    documents, production = publish_contracts(root)
    configs = [yaml.safe_load(text) for path, text in documents.items() if path.suffix == ".yaml"]
    assert len(configs) == 4
    assert production["parallel"]["replicate_degree"] == 2
    assert production["attention"] == {"sdpa_backend": "cudnn"}
    assert production["micro_batch_size"] == 8
    assert production["compile"]["enable"] is True
    assert production["gradient_accumulation"] == accumulation
    for config in configs:
        for key in ("model", "data", "train", "optim", "parallel", "compile", "attention", "seed"):
            assert config[key] == configs[0][key]
        assert config["train"]["grad_accum"] == accumulation
        assert config["train"]["budget_tokens"] == 24 * 8 * 8 * accumulation * 4096
        assert config["data"]["val_samples_per_domain"] == 64
        assert config['seed'] == base['seed']
        assert config['optim']['lr'] == base['optim']['lr']
        assert config['optim']['lr_schedule'] == 'wsd'
        assert config['optim']['warmup_steps'] == config['optim']['decay_steps'] == 8
        assert config['data']['sampling_policy'] == 'block_permutation'
        assert config['data']['stage_sample_limits'] == [8*8*8*accumulation]*3
        assert config['data']['stages'] == base['data']['stages']


@pytest.mark.parametrize("drift", ["topology", "accumulation", "runtime", "script", "missing_stage"])
def test_production_recovery_rejects_stale_or_incomplete_evidence(root, drift):
    publish_contracts(root)
    report = {"status": "PASS", "qualification_identity": recovery.production_recovery_identity(root),
              "stages": {stage: {"status": "PASS"} for stage in recovery.STAGES}}
    path = root / recovery.REPORT_RELATIVE
    path.write_text(json.dumps(report))
    assert recovery.validate_report(root)["status"] == "PASS"
    if drift == "topology":
        path = root / "pretrain/configs/pretrain_base.yaml"
        base = yaml.safe_load(path.read_text())
        base["parallel"].update(replicate_degree=4, shard_degree=2)
        path.write_text(yaml.safe_dump(base))
    elif drift == "accumulation":
        path = root / "pretrain/configs/pretrain_base.yaml"
        base = yaml.safe_load(path.read_text())
        base["train"]["grad_accum"] = 8
        path.write_text(yaml.safe_dump(base))
    elif drift == "runtime":
        (root / "data/health/torchtitan-env-freeze.txt").write_text("changed runtime\n")
    elif drift == "script":
        with (root / "pretrain/scripts/run_production_recovery.py").open("a") as handle:
            handle.write("\n# changed fault injector\n")
    else:
        report["stages"].pop("saving")
        path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        recovery.validate_report(root)
