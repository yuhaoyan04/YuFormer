import importlib.util
import json
import shutil
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import yaml
import torch
from yuformer_model.config import ModelConfig
from yuformer_model.model import YuFormerModel


ROOT = Path(__file__).resolve().parents[2]


def _module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


E1 = _module("analyze_e1_scaling", "pretrain/scripts/analyze_e1_scaling.py")
SEEDREP = _module("analyze_e1_seedrep", "pretrain/scripts/analyze_e1_seedrep.py")
IDENTITY_AUDIT = sys.modules["audit_experiment_identities"]
PREPARE_SEEDREP = _module("prepare_e1_seedrep", "pretrain/scripts/prepare_e1_seedrep.py")
E2 = _module("analyze_e2_results", "pretrain/scripts/analyze_e2_results.py")
LR = _module("resolve_experiment_lrs", "pretrain/scripts/resolve_experiment_lrs.py")
FREEZE = _module("freeze_1t_recipe", "experiments/e4_curriculum/scripts/freeze_1t_recipe.py")


def test_paired_seed_bootstrap_uses_one_shared_seed_index_draw():
    class FixedRng:
        calls = 0

        def integers(self, low, high, size):
            self.calls += 1
            assert (low, high, size) == (0, 3, 3)
            return np.asarray([0, 0, 2])

    rng = FixedRng()
    means = SEEDREP.paired_seed_bootstrap_mean([
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([10.0, 20.0, 30.0]),
    ], rng)

    assert rng.calls == 1
    np.testing.assert_allclose(means, [5.0 / 3.0, 50.0 / 3.0])


def test_compute_optimal_point_rejects_float_underflow_parameters():
    theta = np.asarray([0.5, -30.0, 0.01, 50.0, 0.01])
    assert SEEDREP.compute_optimal_point(theta, 1e18) is None


def test_seedrep_metrics_reject_validation_not_from_final_training_step():
    name = "e1_ln_50m"
    cfg = yaml.safe_load((ROOT / f"pretrain/configs/e1/{name}.yaml").read_text())
    steps = int(cfg["train"]["expected_iters"])
    tps = (int(cfg["data"]["seq_len"]) * int(cfg["train"]["micro_batch_size"])
           * int(cfg["train"]["grad_accum"]) * int(cfg["parallel"]["world_size"]))
    metrics = {"validation_metrics/loss": {"last": 3.0, "last_step": steps - 1}}
    metrics.update({f"validation_metrics/domain/{domain}/loss": {
        "last": 3.1, "last_step": steps} for domain in SEEDREP.DOMAINS})
    result = {
        "run_name": name, "identity_sha256": "fixture",
        "trained_steps": steps, "trained_tokens": steps * tps,
        "configured_budget_shortfall_tokens": int(cfg["train"]["budget_tokens"]) - steps * tps,
        "hyperparameters": {"budget_tokens": int(cfg["train"]["budget_tokens"]),
                            "tokens_per_step": tps, "trained_steps": steps,
                            "trained_tokens": steps * tps},
        "metrics": metrics,
    }

    with pytest.raises(ValueError, match="not from the final training step"):
        SEEDREP.metrics_for(result, cfg, name)


def test_compute_optimal_point_satisfies_closed_form_and_compute_constraint():
    theta = np.asarray([0.5, np.log(10.0), 0.5, np.log(20.0), 0.25])
    flops = 6.0 * 10_000.0
    point = SEEDREP.compute_optimal_point(theta, flops)
    expected_n = ((10.0 * 0.5) / (20.0 * 0.25)
                  * (flops / 6.0) ** 0.25) ** (1.0 / 0.75)

    assert point is not None
    assert np.isclose(point["n_params"], expected_n)
    assert np.isclose(6.0 * point["n_params"] * point["tokens"], flops)
    assert np.isclose(point["tokens_per_param"], point["tokens"] / point["n_params"])


def test_e1_identity_audit_rejects_roster_substitution_even_when_count_stays_24(monkeypatch):
    expected = {path.stem for path in (ROOT / "pretrain/configs/e1").glob("*.yaml")}
    assert len(expected) == 24
    SEEDREP.validate_e1_roster(expected)

    monkeypatch.setattr(IDENTITY_AUDIT, "source_sha256", lambda _root: "fixture")
    changed = set(expected)
    changed.remove("e1_iso_c2_800m")
    changed.add("e1_iso_c2_600m")
    assert len(changed) == 24
    with pytest.raises(ValueError, match="E1 design roster mismatch"):
        IDENTITY_AUDIT.audit(ROOT, {"e1": changed})


def test_seedrep_blocks_bootstrap_policy_implementation_mismatch(tmp_path):
    policy_path = tmp_path / "pretrain/configs/research_budget.yaml"
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text(
        "e1_seed_replication:\n"
        "  seed_bootstrap: resample_three_seed_losses_with_replacement_within_each_fixed_design_point\n",
        encoding="utf-8")

    report = SEEDREP._analyze(tmp_path)

    assert report["status"] == "BLOCKED"
    assert report["reason"] == "seed_bootstrap_policy_mismatch"
    assert report["expected_method"] == SEEDREP.SEED_BOOTSTRAP_POLICY_VALUE


def test_seedrep_blocks_stale_frozen_research_policy_hash(tmp_path):
    policy_path = tmp_path / "pretrain/configs/research_budget.yaml"
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text(
        "e1_seed_replication:\n"
        f"  seed_bootstrap: {SEEDREP.SEED_BOOTSTRAP_POLICY_VALUE}\n"
        "  seed_bootstrap_samples: 2000\n  maximum_additional_tokens: 100\n",
        encoding="utf-8")
    manifest_path = tmp_path / "data/health/e1_seedrep-contracts.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({
        "status": "PASS", "runs": [{}] * 48,
        "inputs": {"pretrain/configs/research_budget.yaml": "intentionally-stale"},
    }), encoding="utf-8")

    report = SEEDREP._analyze(tmp_path)

    assert report["status"] == "BLOCKED"
    assert report["reason"] == "stale_replication_input:pretrain/configs/research_budget.yaml"


def test_seedrep_prepare_refuses_to_silently_rebuild_after_budget_policy_change(tmp_path):
    policy_path = tmp_path / "pretrain/configs/research_budget.yaml"
    policy_path.parent.mkdir(parents=True)
    policy_path.write_text("profile: changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="explicit contract amendment"):
        PREPARE_SEEDREP.require_frozen_budget_binding(tmp_path, {
            "status": "PASS",
            "inputs": {"pretrain/configs/research_budget.yaml": "old-sha256"},
        })


def test_non_embedding_parameter_accounting_is_identical_across_research_stages():
    presets = yaml.safe_load((ROOT / "model/configs/model_configs.yaml").read_text(
        encoding="utf-8"))["presets"]
    for name in ("e1-50m", "e1-100m", "e1-200m", "e1-300m", "e1-800m"):
        values = {
            E1.non_embedding_params(presets[name]),
            E2.non_embedding_params(presets[name]),
            LR.non_embedding_params(presets[name]),
            FREEZE.non_embedding_params(presets[name]),
        }
        assert len(values) == 1, f"parameter accounting drift for {name}: {values}"
        with torch.device("meta"):
            model = YuFormerModel(ModelConfig(**presets[name]))
        measured = sum(parameter.numel() for key, parameter in model.named_parameters()
                       if not key.startswith(("embed_tokens.", "lm_head.")))
        assert values == {measured}


def test_e2_requires_exact_nine_domain_results_and_selects_minimum(tmp_path):
    scaling = {
        "status": "PASS",
        "parameters": {
            "floor": {"estimate": 1.0}, "log_a": {"estimate": 0.0},
            "alpha": {"estimate": 0.01}, "log_b": {"estimate": 0.0},
            "beta": {"estimate": 0.01},
        },
    }
    for index, run_name in enumerate(sorted(E2.EXPECTED)):
        run_dir = tmp_path / run_name
        run_dir.mkdir()
        metrics = {"validation_metrics/loss": {"last": 2.0 + index / 100},
                   "throughput(tps)": {"last": 1000.0}}
        metrics.update({f"validation_metrics/domain/{domain}/loss": {"last": 2.1 + index / 100}
                        for domain in E2.DOMAINS})
        cfg = yaml.safe_load((ROOT / f"pretrain/configs/e2/{run_name}.yaml").read_text())
        tokens_per_step = (cfg["data"]["seq_len"]
                           * cfg["train"]["micro_batch_size"]
                           * cfg["train"]["grad_accum"]
                           * cfg["parallel"]["world_size"])
        trained_tokens = cfg["train"]["expected_iters"] * tokens_per_step
        (run_dir / "result.json").write_text(json.dumps({
            "status": "PASS", "run_name": run_name, "identity_sha256": run_name,
            "hyperparameters": {
                "model_preset": cfg["model"]["preset"],
                "budget_tokens": cfg["train"]["budget_tokens"],
                "tokens_per_step": tokens_per_step,
                "trained_steps": cfg["train"]["expected_iters"],
                "trained_tokens": trained_tokens,
            },
            "metrics": metrics,
        }), encoding="utf-8")
    report = E2.analyze(tmp_path, ROOT, scaling)
    assert report["status"] == "PASS"
    assert report["observed_results"] == 30
    assert report["selection"]["run_name"] == sorted(E2.ARCHITECTURES)[0]
    assert len(report["selection"]["seed_runs"]) == 3

    for path in tmp_path.glob("*/result.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        value["execution_context"] = {"loss_only": True}
        value["metrics"].pop("throughput(tps)")
        path.write_text(json.dumps(value), encoding="utf-8")
    provisional_scaling = {**scaling, "status": "PASS_PROVISIONAL"}
    provisional = E2.analyze(tmp_path, ROOT, provisional_scaling)
    assert provisional["status"] == "PASS_PROVISIONAL"

    adjustments = {row["run_name"]: row["parameter_loss_adjustment"]
                   for row in report["ranking"]}
    for path in tmp_path.glob("*/result.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        architecture = E2.architecture_name(value["run_name"])
        value["metrics"]["validation_metrics/loss"]["last"] = 2.0 + adjustments[architecture]
        path.write_text(json.dumps(value), encoding="utf-8")
    tied = E2.analyze(tmp_path, ROOT, scaling)
    assert len(tied["statistically_indistinguishable"]) == len(E2.ARCHITECTURES)
    selected = next(row for row in tied["ranking"]
                    if row["run_name"] == tied["selection"]["run_name"])
    assert selected["model_flops_per_token"] == min(
        row["model_flops_per_token"] for row in tied["ranking"]
    )

    broken = tmp_path / sorted(E2.EXPECTED)[-1] / "result.json"
    value = json.loads(broken.read_text(encoding="utf-8"))
    del value["metrics"]["validation_metrics/domain/web/loss"]
    broken.write_text(json.dumps(value), encoding="utf-8")
    assert E2.analyze(tmp_path, ROOT, scaling)["status"] == "BLOCKED"


def test_e1_scaling_fit_predicts_synthetic_law():
    n = np.repeat(np.asarray([5e7, 1e8, 2e8, 4e8, 8e8]), 4)
    d = np.tile(np.asarray([2e9, 5e9, 1e10, 2e10]), 5)
    truth = np.asarray([1.4, 12.0, 0.22, 7.0, 0.18])
    loss = E1.predict(truth, n, d)
    fitted, rmse = E1.fit_law(n, d, loss)
    assert rmse < 1e-6
    assert np.max(np.abs(E1.predict(fitted, n, d) - loss)) < 2e-6


def test_seedrep_identity_audit_covers_base_and_replication_rosters(tmp_path, monkeypatch):
    (tmp_path / "pretrain/configs/e1").mkdir(parents=True)
    (tmp_path / "pretrain/configs/research_budget.yaml").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "pretrain/configs/research_budget.yaml").write_text(
        "e1_seed_replication:\n  seed_bootstrap_samples: 2000\n"
        "  seed_bootstrap: resample_seed_indices_with_replacement_jointly_across_all_fixed_design_points\n"
        "  maximum_additional_tokens: 4800\n", encoding="utf-8")
    health = tmp_path / "data/health"
    health.mkdir(parents=True)
    run_rows = []
    base_names = {path.stem for path in (ROOT / "pretrain/configs/e1").glob("*.yaml")}
    assert len(base_names) == 24
    for index, source_name in enumerate(sorted(base_names)):
        for suffix, seed in ((2, 20261002), (3, 20261003)):
            name = f"e1_seedrep_{source_name}_s{suffix}"
            config = tmp_path / f"rep-{index}-{suffix}.yaml"
            env = tmp_path / f"rep-{index}-{suffix}.env"
            config.write_text("placeholder: true\n", encoding="utf-8")
            env.write_text("placeholder=true\n", encoding="utf-8")
            run_rows.append({
                "name": name, "source_run": source_name, "seed": seed,
                "config": str(config.relative_to(tmp_path)),
                "env": str(env.relative_to(tmp_path)),
                "config_sha256": SEEDREP.sha256(config), "env_sha256": SEEDREP.sha256(env),
                "budget_tokens": 100,
            })
    (health / "e1_seedrep-contracts.json").write_text(json.dumps({
        "status": "PASS", "runs": run_rows, "inputs": {},
        "maximum_additional_tokens": 4800,
        "paired_seed_design": True, "design_points": 24,
        "existing_seed": 20261001,
        "additional_seeds": [20261002, 20261003],
        "total_seed_count_per_point": 3,
    }), encoding="utf-8")
    for name in base_names:
        (tmp_path / "pretrain/configs/e1" / f"{name}.yaml").write_text(
            "train:\n  budget_tokens: 100\n", encoding="utf-8")
    observed = {}

    def reject_identity_audit(root, groups):
        observed.update(groups)
        return {"status": "BLOCKED", "total": 72}

    monkeypatch.setattr(SEEDREP, "audit", reject_identity_audit)
    report = SEEDREP._analyze(tmp_path)

    assert report["status"] == "BLOCKED"
    assert report["reason"] == "e1_and_replication_identity_audit"
    assert observed["e1"] == base_names
    assert observed["e1_seedrep"] == {row["name"] for row in run_rows}

    manifest_path = health / "e1_seedrep-contracts.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["paired_seed_design"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    paired_report = SEEDREP._analyze(tmp_path)
    assert paired_report["status"] == "BLOCKED"
    assert paired_report["reason"] == "seed_replication_design_audit"

    manifest["paired_seed_design"] = True
    original_seed = manifest["runs"][0]["seed"]
    manifest["runs"][0]["seed"] = 20261003
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    roster_report = SEEDREP._analyze(tmp_path)
    assert roster_report["status"] == "BLOCKED"
    assert roster_report["reason"] == "seed_replication_design_audit"

    manifest["runs"][0]["seed"] = original_seed
    manifest["maximum_additional_tokens"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    budget_report = SEEDREP._analyze(tmp_path)
    assert budget_report["status"] == "BLOCKED"
    assert budget_report["reason"] == "seed_replication_budget_audit"
    assert budget_report["contract_additional_tokens"] == 4800
    assert budget_report["configured_additional_tokens"] == 4800

    budget_source = sorted(base_names)[0]
    (tmp_path / "pretrain/configs/e1" / f"{budget_source}.yaml").write_text(
        "train:\n  budget_tokens: 101\n", encoding="utf-8")
    for row in manifest["runs"]:
        if row["source_run"] == budget_source:
            row["budget_tokens"] = 101
    manifest["maximum_additional_tokens"] = 4802
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    policy_report = SEEDREP._analyze(tmp_path)
    assert policy_report["status"] == "BLOCKED"
    assert policy_report["reason"] == "seed_replication_budget_audit"
    assert policy_report["contract_additional_tokens"] == 4802
    assert policy_report["configured_additional_tokens"] == 4802
    assert policy_report["policy_max_additional_tokens"] == 4800


def test_seedrep_config_must_match_its_paired_source_point():
    source = {"run_name": "e1_ln_50m", "seed": 20261001,
              "data": {"seq_len": 4096}, "train": {"budget_tokens": 1_000_000}}
    run_name = "e1_seedrep_e1_ln_50m_s2"
    contract = {"name": run_name, "source_run": "e1_ln_50m", "seed": 20261002,
                "budget_tokens": 1_000_000}
    replicate = {**source, "run_name": run_name, "seed": 20261002}

    SEEDREP.validate_replication_config("e1_ln_50m", source, contract, replicate,
                                        20261002, run_name)
    changed = {**replicate, "data": {"seq_len": 8192}}
    with pytest.raises(ValueError, match="matched source design point"):
        SEEDREP.validate_replication_config("e1_ln_50m", source, contract, changed,
                                            20261002, run_name)
    wrong_budget = {**contract, "budget_tokens": 2_000_000}
    with pytest.raises(ValueError, match="matched source design point"):
        SEEDREP.validate_replication_config("e1_ln_50m", source, wrong_budget, replicate,
                                            20261002, run_name)


def test_e1_analysis_dispatches_to_seedrep_when_all_48_results_exist(tmp_path, monkeypatch):
    health = tmp_path / "data/health"
    health.mkdir(parents=True)
    jobs = tmp_path / "runs"
    envs = tmp_path / "envs"
    jobs.mkdir()
    envs.mkdir()
    rows = []
    for index in range(48):
        output = jobs / f"run-{index}"
        output.mkdir()
        (output / "result.json").write_text("{}", encoding="utf-8")
        env = envs / f"run-{index}.env"
        env.write_text(f"YUFORMER_PRETRAIN_OUT={output}\n", encoding="utf-8")
        rows.append({"env": str(env.relative_to(tmp_path))})
    (health / "e1_seedrep-contracts.json").write_text(json.dumps({
        "status": "PASS", "runs": rows,
    }), encoding="utf-8")
    called = {}
    seedrep = types.ModuleType("analyze_e1_seedrep")

    def fake_analyze(root, bootstrap_samples):
        called.update(root=root, bootstrap_samples=bootstrap_samples)
        return {"status": "PASS", "study": "seed-aware"}

    seedrep.analyze = fake_analyze
    monkeypatch.setitem(sys.modules, "analyze_e1_seedrep", seedrep)

    report = E1.analyze(tmp_path, tmp_path / "unused", bootstrap_samples=123)

    assert report == {"status": "PASS", "study": "seed-aware"}
    assert called == {"root": tmp_path, "bootstrap_samples": 123}


def test_seedrep_end_to_end_synthetic_fit_reports_all_audits(tmp_path, monkeypatch):
    configs_dir = tmp_path / "pretrain/configs/e1"
    rep_configs_dir = tmp_path / "pretrain/configs/e1_seedrep"
    health = tmp_path / "data/health"
    results_dir = tmp_path / "experiments/e1_scaling/results"
    for directory in (configs_dir, rep_configs_dir, health, results_dir,
                      tmp_path / "model/configs"):
        directory.mkdir(parents=True, exist_ok=True)

    source_model_configs = ROOT / "model/configs/model_configs.yaml"
    (tmp_path / "model/configs/model_configs.yaml").write_text(
        source_model_configs.read_text(encoding="utf-8"), encoding="utf-8")
    presets = yaml.safe_load(source_model_configs.read_text(encoding="utf-8"))["presets"]
    source_configs = {
        path.stem: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in (ROOT / "pretrain/configs/e1").glob("*.yaml")
    }
    rows, additional_tokens, actual_additional_tokens = [], 0, 0
    theta = np.asarray([1.5, 2.0, 0.2, 2.5, 0.15])

    def make_result(name, cfg, loss):
        tps = (int(cfg["data"]["seq_len"]) * int(cfg["train"]["micro_batch_size"])
               * int(cfg["train"]["grad_accum"]) * int(cfg["parallel"]["world_size"]))
        steps = int(cfg["train"]["expected_iters"])
        metrics = {"validation_metrics/loss": {"last": float(loss), "last_step": steps}}
        metrics.update({f"validation_metrics/domain/{domain}/loss": {
            "last": float(loss + 0.1), "last_step": steps}
                        for domain in SEEDREP.DOMAINS})
        return {
            "status": "PASS", "run_name": name, "identity_sha256": f"identity:{name}",
            "trained_steps": steps, "trained_tokens": steps * tps,
            "configured_budget_shortfall_tokens": (
                int(cfg["train"]["budget_tokens"]) - steps * tps),
            "hyperparameters": {
                "budget_tokens": int(cfg["train"]["budget_tokens"]),
                "tokens_per_step": tps, "trained_steps": steps,
                "trained_tokens": steps * tps,
            },
            "metrics": metrics,
        }

    for source_name, source_cfg in source_configs.items():
        base_config_path = configs_dir / f"{source_name}.yaml"
        base_config_path.write_text(yaml.safe_dump(source_cfg, sort_keys=False), encoding="utf-8")
        base_out = tmp_path / "data/experiments/e1" / source_name
        base_out.mkdir(parents=True)
        base_env = configs_dir / f"{source_name}.env"
        base_env.write_text(f"YUFORMER_PRETRAIN_OUT={base_out}\n", encoding="utf-8")
        tps = (int(source_cfg["data"]["seq_len"])
               * int(source_cfg["train"]["micro_batch_size"])
               * int(source_cfg["train"]["grad_accum"])
               * int(source_cfg["parallel"]["world_size"]))
        tokens = int(source_cfg["train"]["expected_iters"]) * tps
        n_params = SEEDREP.non_embedding_params(presets[source_cfg["model"]["preset"]])
        mean_loss = float(E1.predict(theta, np.asarray([n_params]), np.asarray([tokens]))[0])
        (base_out / "result.json").write_text(json.dumps(
            make_result(source_name, source_cfg, mean_loss - 0.001)), encoding="utf-8")

        for suffix, seed, offset in ((2, 20261002, 0.0), (3, 20261003, 0.001)):
            run_name = f"e1_seedrep_{source_name}_s{suffix}"
            rep_cfg = dict(source_cfg)
            rep_cfg.update(run_name=run_name, seed=seed)
            config_path = rep_configs_dir / f"{run_name}.yaml"
            config_path.write_text(yaml.safe_dump(rep_cfg, sort_keys=False), encoding="utf-8")
            output = tmp_path / "data/experiments/e1_seedrep" / run_name
            output.mkdir(parents=True)
            env_path = rep_configs_dir / f"{run_name}.env"
            env_path.write_text(f"YUFORMER_PRETRAIN_OUT={output}\n", encoding="utf-8")
            (output / "result.json").write_text(json.dumps(
                make_result(run_name, rep_cfg, mean_loss + offset)), encoding="utf-8")
            budget = int(rep_cfg["train"]["budget_tokens"])
            additional_tokens += budget
            actual_additional_tokens += int(rep_cfg["train"]["expected_iters"]) * tps
            rows.append({
                "name": run_name, "source_run": source_name, "seed": seed,
                "config": str(config_path.relative_to(tmp_path)),
                "env": str(env_path.relative_to(tmp_path)), "budget_tokens": budget,
                "config_sha256": SEEDREP.sha256(config_path),
                "env_sha256": SEEDREP.sha256(env_path),
            })

    policy_dir = tmp_path / "pretrain/configs"
    (policy_dir / "research_budget.yaml").write_text(
        f"e1_seed_replication:\n  seed_bootstrap_samples: 500\n"
        f"  seed_bootstrap: {SEEDREP.SEED_BOOTSTRAP_POLICY_VALUE}\n"
        f"  maximum_additional_tokens: {additional_tokens}\n", encoding="utf-8")
    manifest = {
        "status": "PASS", "runs": rows, "inputs": {},
        "source_identity_evidence": {}, "maximum_additional_tokens": additional_tokens,
        "design_points": 24, "existing_seed": 20261001,
        "additional_seeds": [20261002, 20261003], "total_seed_count_per_point": 3,
        "paired_seed_design": True,
    }
    (health / "e1_seedrep-contracts.json").write_text(json.dumps(manifest), encoding="utf-8")
    ln_names = sorted(name for name in source_configs if name.startswith("e1_ln_"))
    screening = {
        "fit_quality_gate": {"checks": {"throughput_log_r_squared": True}},
        "throughput_fit": {"status": "PASS"},
        "independent_performance": {"status": "PASS"},
        "performance_evidence_status": "PASS", "production_policy": {"selection": "fixture"},
        "production_candidates": [
            {"source_run": name, "feasible_within_45_days": True, "predicted_days_4gpu": 1.0}
            for name in ln_names
        ],
    }
    (results_dir / "scaling_fit_screening_20261107.json").write_text(
        json.dumps(screening), encoding="utf-8")

    def pass_identity_audit(root, groups):
        assert len(groups["e1"]) == 24 and len(groups["e1_seedrep"]) == 48
        return {"status": "PASS", "passed": 72, "total": 72, "roster_exact": True,
                "evidence_snapshot_count": 0, "evidence_snapshot": {}}

    monkeypatch.setattr(SEEDREP, "audit", pass_identity_audit)
    report = SEEDREP._analyze(tmp_path, bootstrap_samples=500)

    assert report["status"] == "PASS"
    assert report["observed_results"] == 24
    assert report["replication_result_count"] == 72
    assert report["identity_audit"]["passed"] == 72
    assert report["budget_audit"]["policy_max_additional_tokens"] == additional_tokens
    assert report["budget_audit"]["configured_additional_tokens"] == additional_tokens
    assert report["budget_audit"]["actual_additional_trained_tokens"] == actual_additional_tokens
    assert report["budget_audit"]["configured_budget_shortfall_tokens"] == (
        additional_tokens - actual_additional_tokens)
    assert all(row["tokens"] + row["configured_budget_shortfall_tokens"]
               == row["configured_budget_tokens"] for row in report["runs"])
    assert report["seed_design_audit"]["paired_seed_design"] is True
    assert report["fit_diagnostics"]["seed_level"]["bootstrap"] == (
        "joint_seed_index_resampling_across_design_points")
    assert report["fit_diagnostics"]["seed_level"]["policy_value"] == (
        SEEDREP.SEED_BOOTSTRAP_POLICY_VALUE)
    assert report["analysis_method_amendments"] == []
    assert report["seed_bootstrap_rng_seed"] == SEEDREP.SEED_BOOTSTRAP_RNG_SEED
    assert [row["family"] for row in report["compute_optimal_frontier"]] == ["c0", "c1", "c2"]
    assert all(row["status"] == "PASS" and row["valid_bootstrap_fraction"] == 1.0
               for row in report["compute_optimal_frontier"])
    assert all(row["budget_definition"] == "C=6*N*D with N=non_embedding_parameters"
               for row in report["compute_optimal_frontier"])
    assert report["fit_quality_gate"]["status"] == "PASS"


def test_e1_analysis_includes_controlled_axes_and_production_recommendation(tmp_path, monkeypatch):
    analysis_root = tmp_path / "repository"
    shutil.copytree(ROOT / "pretrain/configs/e1", analysis_root / "pretrain/configs/e1")
    (analysis_root / "model/configs").mkdir(parents=True)
    shutil.copyfile(ROOT / "model/configs/model_configs.yaml", analysis_root / "model/configs/model_configs.yaml")
    configs = {path.stem: yaml.safe_load(path.read_text(encoding="utf-8"))
               for path in (ROOT / "pretrain/configs/e1").glob("*.yaml")}
    presets = yaml.safe_load((ROOT / "model/configs/model_configs.yaml").read_text(
        encoding="utf-8"))["presets"]
    truth = np.asarray([1.3, 12.0, 0.22, 7.0, 0.18])
    for name, cfg in configs.items():
        n = E1.non_embedding_params(presets[cfg["model"]["preset"]])
        tokens_per_step = (int(cfg["data"]["seq_len"])
                           * int(cfg["train"]["micro_batch_size"])
                           * int(cfg["train"]["grad_accum"])
                           * int(cfg["parallel"]["world_size"]))
        d = int(cfg["train"]["expected_iters"]) * tokens_per_step
        loss = float(E1.predict(truth, np.asarray([n]), np.asarray([d]))[0])
        run = tmp_path / name
        run.mkdir()
        (run / "result.json").write_text(json.dumps({
            "status": "PASS", "run_name": name, "identity_sha256": name,
            "hyperparameters": {
                "model_preset": cfg["model"]["preset"],
                "budget_tokens": int(cfg["train"]["budget_tokens"]),
                "tokens_per_step": tokens_per_step,
                "trained_steps": int(cfg["train"]["expected_iters"]),
                "trained_tokens": d,
            },
            "metrics": {"validation_metrics/loss": {"last": loss}},
            "series": {"throughput(tps)": [{"step": 1, "value": 1e11 / n},
                                             {"step": 2, "value": 1e11 / n}]},
        }), encoding="utf-8")
    report = E1.analyze(analysis_root, tmp_path, bootstrap_samples=100)
    assert report["status"] == "PASS"
    assert report["observed_results"] == 24
    assert report["controlled_axis_fits"]["L_N_fixed_D"]["points"] == 5
    assert report["controlled_axis_fits"]["L_D_fixed_300M"]["points"] == 5
    assert report["production_recommendation"]["source_run"].startswith("e1_ln_")
    assert len(report["fit_diagnostics"]["residuals"]) == 24
    assert len(report["fit_diagnostics"]["leave_one_out"]) == 24
    assert report["fit_diagnostics"]["normalized_rmse"] < 0.01
    assert report["fit_quality_gate"]["status"] == "PASS"
    assert report["throughput_fit"]["log_r_squared"] > 0.99
    assert len(report["production_recommendation"]["predicted_loss_at_1t_ci95"]) == 2
    assert len(report["production_recommendation"]["predicted_days_4gpu_ci95"]) == 2

    for path in tmp_path.glob("*/result.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        value["execution_context"] = {"loss_only": True}
        value["series"].pop("throughput(tps)")
        path.write_text(json.dumps(value), encoding="utf-8")
    # The synthetic loss-only case must not consume live production probe files.
    monkeypatch.setattr(E1, "qualified_performance", lambda *args: {
        "status": "BLOCKED", "runs": [], "issues": ["fixture has no independent probes"],
        "snapshot": {}})
    provisional = E1.analyze(analysis_root, tmp_path, bootstrap_samples=100)
    assert provisional["status"] == "PASS_PROVISIONAL"
    assert provisional["throughput_fit"]["status"] == "EXCLUDED_CONTENDED_RESOURCE"
    assert "predicted_days_4gpu" not in provisional["production_recommendation"]
    rows = [{"n_params": E1.non_embedding_params(presets[configs[f"e1_ln_{size}"]["model"]["preset"]]),
             "median_tps_per_device": 1e11 / E1.non_embedding_params(
                 presets[configs[f"e1_ln_{size}"]["model"]["preset"]])}
            for size in ("50m", "100m", "200m", "400m", "800m")]
    monkeypatch.setattr(E1, "qualified_performance", lambda *args: {
        "status": "PASS", "runs": rows, "snapshot": {"evidence.json": "sha"}})
    qualified = E1.analyze(analysis_root, tmp_path, bootstrap_samples=100)
    assert qualified["status"] == "PASS"
    assert qualified["independent_performance"]["status"] == "PASS"
    assert all(row["loss_only"] for row in qualified["runs"])
    assert "predicted_days_4gpu" in qualified["production_recommendation"]
