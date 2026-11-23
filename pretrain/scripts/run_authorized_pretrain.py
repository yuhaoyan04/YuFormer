#!/usr/bin/env python3
"""Execute the explicitly authorized eight-GPU production dependency chain."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "pretrain/src"), str(ROOT / "pretrain/scripts")]
from yuformer_pretrain.lease import RunLease
from run_production_canary import evaluate_result, validate_frozen_bindings
from user_pretrain_authorization import load_authorization, source_drift_only
from gpu_resource_policy import allowed_gpu_indices
from artifact_archive import move_artifact
from training_tokenizer_calibration import load_training_ratios

PYTHON = Path("/opt/yuformer-torchtitan/bin/python")
STATE = ROOT / "data/health/authorized-pretrain-pipeline.json"
LOGS = ROOT / "report/logs/authorized-pretrain"


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


class Pipeline:
    def __init__(self, probe_configs=None, *, reuse_prepared_production=False):
        self.state = {"status": "RUNNING", "pid": os.getpid(), "stages": {}}
        self.probe_configs = [ROOT / path for path in probe_configs] if probe_configs else None
        self.reuse_prepared_production = reuse_prepared_production
        self.lease = RunLease(ROOT / "data/health/authorized-pretrain-pipeline",
                              "authorized-eight-gpu-pretrain", stale_seconds=600).acquire()
        LOGS.mkdir(parents=True, exist_ok=True)

    def publish(self):
        self.lease.assert_owner()
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic(STATE, self.state)

    def command(self, stage, argv, *, env=None, accepted=(0,)):
        self.state["stage"] = stage
        log = LOGS / f"{stage}.log"
        row = {"status": "RUNNING", "argv": list(map(str, argv)), "log": str(log),
               "started_at": datetime.now(timezone.utc).isoformat()}
        self.state["stages"][stage] = row
        self.publish()
        with log.open("a") as handle:
            child = subprocess.Popen(list(map(str, argv)), cwd=ROOT, env=env,
                                     stdout=handle, stderr=subprocess.STDOUT)
            row["pid"] = child.pid
            self.publish()
            code = child.wait()
        row.update(returncode=code, status="PASS" if code in accepted else "FAIL")
        self.publish()
        if code not in accepted:
            raise RuntimeError(f"{stage} failed with rc={code}; see {log}")

    def require(self, relative, status):
        value = read(ROOT / relative)
        if value.get("status") != status:
            raise RuntimeError(f"{relative}: expected {status}, observed {value.get('status')}")
        return value

    def script(self, stage, relative, *args, pinned=False, accepted=(0,)):
        self.command(stage, [PYTHON if pinned else sys.executable, ROOT / relative, *args],
                     accepted=accepted)

    def wait_for_gpus(self):
        if allowed_gpu_indices(ROOT / "data/health") != set(range(8)):
            raise RuntimeError("pipeline requires authorization for all eight physical GPUs")
        deadline = time.monotonic() + 12 * 3600
        while True:
            inventory = subprocess.run([
                "nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=30, check=True)
            apps = subprocess.run([
                "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=30, check=True)
            rows = [tuple(int(v.strip()) for v in line.split(","))
                    for line in inventory.stdout.splitlines() if line.strip()]
            if ({row[0] for row in rows} == set(range(8))
                    and all(row[1] <= 1024 for row in rows) and not apps.stdout.strip()):
                return
            self.state.update(stage="waiting_for_eight_idle_gpus", gpu_inventory=rows)
            self.publish()
            if time.monotonic() >= deadline:
                raise RuntimeError("eight exclusive GPUs did not become available within 12 hours")
            time.sleep(30)

    def environment(self):
        # Another preparation process may already hold prepare_env's lock.
        # A successful lock handoff is not proof that the environment is ready.
        deadline = time.monotonic() + 12 * 3600
        while True:
            self.command("prepare_environment", ["bash", ROOT / "pretrain/backends/torchtitan/prepare_env.sh"])
            preparation = read(ROOT / "data/health/torchtitan-env-prepare.json")
            if PYTHON.is_file() and preparation.get("status") == "PASS":
                break
            if time.monotonic() > deadline:
                raise RuntimeError("locked runtime preparation exceeded 12 hours")
            time.sleep(30)
        self.script("userspace", "pretrain/backends/torchtitan/validate_userspace.py",
                    "--environment", PYTHON.parent.parent, "--source", "/opt/torchtitan-v0.3.0",
                    "--freeze", ROOT / "data/health/torchtitan-env-freeze.txt",
                    "--out", ROOT / "data/health/torchtitan-userspace.json", pinned=True)
        env = dict(os.environ, YUFORMER_TORCHTITAN_ROOT="/opt/torchtitan-v0.3.0")
        self.command("preflight", [PYTHON, ROOT / "pretrain/backends/torchtitan/preflight.py",
                                  "--out", ROOT / "data/health/torchtitan-preflight.json"], env=env)

    def qualify(self, label):
        self.wait_for_gpus()
        self.script(label, "pretrain/scripts/run_torchtitan_qualification.py",
                    "--gpus", "0,1,2,3,4,5,6,7", pinned=True)
        report = self.require("data/health/torchtitan-qualification.json", "PASS")
        if report.get("stages", {}).get("reshard_4to8", {}).get("status") != "PASS":
            raise RuntimeError("eight-GPU reshard qualification is incomplete")

    def probes(self, authorization):
        validator = module("pipeline_contract", "pretrain/backends/torchtitan/validate_contract.py")
        freezer = module("pipeline_freezer", "experiments/e4_curriculum/scripts/freeze_1t_recipe.py")
        outcomes = []
        configs = self.probe_configs or sorted((ROOT / "pretrain/configs/eight_gpu_probe").glob("*.yaml"))
        for config in configs:
            values = dict(line.split("=", 1) for line in config.with_suffix(".env").read_text().splitlines()
                          if line and not line.startswith("#"))
            previous = {key: os.environ.get(key) for key in values}
            os.environ.update(values)
            try:
                validated = validator.validate(ROOT, config, publish_identity=False)
            finally:
                for key, old in previous.items():
                    if old is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old
            run = Path(validated["out_dir"])
            result_path = run / "result.json"
            result = read(result_path) if result_path.is_file() else {}
            reusable = (result.get("status") == "PASS"
                        and result.get("identity_sha256") == validated["identity"]["identity_sha256"])
            if not reusable:
                self.wait_for_gpus()
                if run.exists():
                    target = run.parent / "_stale" / f"{run.name}-{time.time_ns()}"
                    with RunLease(run, "archive-stale-eight-gpu-probe"):
                        target.mkdir(parents=True, exist_ok=False)
                        for child in list(run.iterdir()):
                            if child.name != ".run.lease":
                                move_artifact(child, target / child.name)
                self.command(config.stem, ["bash", ROOT / "pretrain/backends/torchtitan/launch.sh",
                                          config.with_suffix(".env")])
                self.script(config.stem + "_collect", "pretrain/scripts/collect_torchtitan_result.py",
                            run, pinned=True)
                result = read(result_path)
            cfg = yaml.safe_load(config.read_text())
            evaluation = evaluate_result(result, {
                "tokens_per_step": authorization["training"]["tokens_per_step"],
                "steps": cfg["train"]["expected_iters"], "world_size": 8,
                "planned_training_duty_cycle": authorization["training"]["planned_training_duty_cycle"],
                "maximum_projected_days": authorization["training"]["launch_runtime_ceiling_days"],
                "target_completion_days": 30})
            outcomes.append({"config": str(config.relative_to(ROOT)), "evaluation": evaluation,
                             "result": str(result_path.relative_to(ROOT))})
        atomic(ROOT / "data/health/eight-gpu-topology-selection.json", {"candidates": outcomes})
        passing = [row for row in outcomes if row["evaluation"]["status"] == "PASS"]
        if not passing:
            raise RuntimeError("no eight-GPU topology meets the 33-day wall-time ceiling; optimization required")
        best = min(passing, key=lambda r: r["evaluation"].get("projected_total_training_days", r["evaluation"]["projected_1t_days_from_p10"]))
        cfg = yaml.safe_load((ROOT / best["config"]).read_text())
        self.wait_for_gpus()
        parity_env = dict(os.environ, CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7")
        parity_env["PYTHONPATH"] = str(ROOT / "pretrain/backends/torchtitan") + ":" + str(ROOT / "model/src")
        parity_report = ROOT / "data/health/selected-eight-gpu-gradient-parity.json"
        self.command("selected_topology_gradient_parity", [
            PYTHON, "-m", "torch.distributed.run", "--nproc_per_node", "8",
            "--master_port", "29881", ROOT / "pretrain/backends/torchtitan/distributed_parity.py",
            "--replicate-degree", str(cfg["parallel"]["replicate_degree"]),
            "--output", parity_report], env=parity_env)
        self.require("data/health/selected-eight-gpu-gradient-parity.json", "PASS")
        base_path = ROOT / "pretrain/configs/pretrain_base.yaml"
        base = yaml.safe_load(base_path.read_text())
        base["parallel"] = cfg["parallel"]
        base["attention"] = cfg.get("attention", {"sdpa_backend": "auto"})
        base_path.write_text(yaml.safe_dump(base, sort_keys=False, allow_unicode=True))
        source_model = ROOT / authorization["architecture"]["config"]
        authorized = yaml.safe_load(source_model.read_text())["presets"][authorization["architecture"]["preset"]]
        actual = yaml.safe_load((ROOT / "model/configs/pretrain_final_model_configs.yaml").read_text())["presets"]["pretrain-final"]
        if authorized != actual:
            raise RuntimeError("probe architecture differs from the authorized architecture")
        atomic(ROOT / "data/health/full-pretrain-optimization-canary.json", {
            "status": "PASS", "world_size": 8, "source_sha256": freezer.source_sha256(ROOT),
            "model_configs_sha256": digest(source_model), "tokens_per_step": 4194304,
            "micro_batch_size": cfg["train"]["micro_batch_size"], "compile": cfg["compile"],
            "attention": base["attention"],
            "selected_topology": cfg["parallel"], "evidence": best,
            "gradient_parity_report": str(parity_report.relative_to(ROOT)),
            "gradient_parity_sha256": digest(parity_report),
            "result_sha256": digest(ROOT / best["result"])})

    def audit_identity(self):
        self.script("experiment_identity", "pretrain/scripts/audit_experiment_identities.py", accepted=(0, 2))
        identity = read(ROOT / "report/experiment_identity_audit_latest.json")
        if identity.get("status") != "PASS" and not source_drift_only(identity):
            raise RuntimeError("identity audit contains failures outside the explicit source-drift waiver")

    def frozen_recipe_current(self):
        try:
            frozen = read(ROOT / "experiments/e4_curriculum/results/frozen_1t_recipe.json")
            if frozen.get("status") != "PASS":
                return False
            validate_frozen_bindings(ROOT, frozen)
            if frozen.get('data_origin', {}).get('kind') == 'TRAINING_ONLY_WITH_EXACT_DATA_REUSE':
                origin = module('pipeline_training_data_origin', 'experiments/e4_curriculum/scripts/training_data_origin.py')
                origin.resolve_data_origin(ROOT, ROOT/'data/processed/releases/pretrain-1t-frozen.release.json')
                identity = module('pipeline_training_qualification_identity', 'pretrain/scripts/qualification_identity.py')
                return frozen.get('training_adoption', {}).get('qualification_identity') == identity.qualification_identity(ROOT)['sha256']
            optimization = read(ROOT / "data/health/full-pretrain-optimization-canary.json")
            freezer = module("pipeline_resume_freezer", "experiments/e4_curriculum/scripts/freeze_1t_recipe.py")
            return (optimization.get("status") == "PASS"
                    and optimization.get("source_sha256") == freezer.source_sha256(ROOT))
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def prepare_training_calibration(self):
        try:
            load_training_ratios(ROOT, {})
            self.state["training_tokenizer_calibration_reused"] = True
            return
        except (OSError, ValueError, KeyError, TypeError):
            pass
        release_path = ROOT / "data/processed/releases/pretrain-1t-frozen.release.json"
        if not release_path.is_file():
            raise RuntimeError("training calibration requires a selected three-phase draft order release before freezing")
        release = read(release_path)
        if release.get("status") != "PASS" or set(release.get("stages", {})) != {"warmup", "stable", "anneal"}:
            raise RuntimeError("training calibration requires a PASS three-phase selected order release")
        work = ROOT / "data/processed/tokenizer_calibration" / f"selected-phases-{digest(release_path)[:16]}-100m"
        self.script("training_tokenizer_calibration", "experiments/e4_curriculum/scripts/calibrate_training_tokens.py",
                    "--work-dir", work, "--tokenizer",
                    ROOT / "tokenizer/tokenizers/m3_20261020/yf-64k-bpe/tokenizer.json",
                    "--workers", "128", "--reference-tokens-per-domain", "100000000", pinned=True)
        load_training_ratios(ROOT, {})
        self.state["training_tokenizer_calibration_reused"] = False

    def has_production_checkpoint(self):
        path = ROOT / "pretrain/configs/pretrain_base.env"
        if not path.is_file():
            return False
        values = dict(line.split("=", 1) for line in path.read_text().splitlines()
                      if line.strip() and not line.lstrip().startswith("#"))
        out = values.get("YUFORMER_PRETRAIN_OUT")
        return bool(out and any((ROOT / out).glob("checkpoint/step-*/.metadata")))

    def production_assets_current(self):
        if not (ROOT / "data/processed/audits/pretrain_1t_overlap.json").is_file():
            return False
        try:
            validator = module("pipeline_resume_validator", "pretrain/scripts/validate_production_contract.py")
            validator.validate_frozen_evidence(ROOT)
            return True
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def production_canary_current(self):
        try:
            report = read(ROOT / "data/health/production-throughput-canary.json")
            if report.get("status") != "PASS":
                return False
            canary = module("pipeline_resume_canary", "pretrain/scripts/run_production_canary.py")
            contract = canary.build_contract(ROOT)
            contract["gpu_indices"] = sorted(allowed_gpu_indices(ROOT / "data/health"))[:8]
            if any(report.get(key) != value for key, value in contract.items()):
                return False
            result_path = ROOT / canary.RUN_DIR.relative_to(canary.ROOT) / "result.json"
            return (report.get("result_sha256") == digest(result_path)
                    and report.get("evaluation", {}).get("status") == "PASS"
                    and canary.evaluate_result(read(result_path), contract).get("status") == "PASS")
        except (OSError, ValueError, KeyError, TypeError):
            return False

    def run(self):
        authorization = load_authorization(ROOT)
        if authorization["training"].get("world_size") != 8:
            raise RuntimeError("this coordinator requires explicit eight-GPU authorization")
        adoption_path = ROOT/'data/health/pretrain-model-order-adoption.json'
        if (adoption_path.is_file() and read(adoption_path).get('status') not in
                {'READY_FOR_PRETRAIN', 'FORMAL_PRETRAIN_RUNNING', 'FORMAL_PRETRAIN_COMPLETE'}):
            raise RuntimeError('model/data-order adoption is incomplete; generic refreeze and data rebuilding are forbidden while preprocessing continues independently')
        frozen_path = ROOT/'experiments/e4_curriculum/results/frozen_1t_recipe.json'
        if (frozen_path.is_file() and read(frozen_path).get('data_origin', {}).get('kind')
                == 'TRAINING_ONLY_WITH_EXACT_DATA_REUSE'):
            self.reuse_prepared_production = True
            self.state['adopted_recipe_automatically_forces_reuse_only'] = True
        if getattr(self, 'reuse_prepared_production', False):
            if (not self.frozen_recipe_current() or not self.production_canary_current()
                    or not self.production_assets_current()):
                raise RuntimeError('prepared-production launch requires all current recipe, canary and data evidence; rebuilding is forbidden in this mode')
            self.state.update(frozen_recipe_reused=True, production_canary_reused=True,
                              production_assets_reused=True, reuse_prepared_production=True)
            self.finish_and_launch()
            return
        self.environment()
        self.qualify("qualification_before_probes")
        recipe_current = self.frozen_recipe_current()
        if not recipe_current and self.has_production_checkpoint():
            raise RuntimeError("existing production checkpoint has changed recipe bindings; refusing to refreeze or rebuild data")
        self.state["frozen_recipe_reused"] = recipe_current
        if not recipe_current:
            self.probes(authorization)
            self.audit_identity()
            self.prepare_training_calibration()
            self.script("freeze", "experiments/e4_curriculum/scripts/freeze_1t_recipe.py")
            self.require("experiments/e4_curriculum/results/frozen_1t_recipe.json", "PASS")
        self.qualify("qualification_after_freeze")
        self.wait_for_gpus()
        self.script("production_recovery", "pretrain/scripts/run_production_recovery.py", pinned=True)
        self.require("data/health/production-recovery-qualification.json", "PASS")
        self.wait_for_gpus()
        canary_current = self.production_canary_current()
        self.state["production_canary_reused"] = canary_current
        if not canary_current:
            self.script("production_canary", "pretrain/scripts/run_production_canary.py", pinned=True)
        self.require("data/health/production-throughput-canary.json", "PASS")
        assets_current = self.production_assets_current()
        if not assets_current and self.has_production_checkpoint():
            raise RuntimeError("existing production checkpoint has stale data evidence; refusing to rebuild training data")
        self.state["production_assets_reused"] = assets_current
        if not assets_current:
            self.script("plan", "experiments/e4_curriculum/scripts/audit_1t_plan.py", "--output",
                        ROOT / "experiments/e4_curriculum/results/e4_1t_plan_static_audit.json")
            self.script("orders", "experiments/e4_curriculum/scripts/materialize_1t_orders.py", "--workers", "128", pinned=True)
            self.script("release", "experiments/e4_curriculum/scripts/build_1t_order_release.py", pinned=True)
            self.script("tokens", "experiments/e4_curriculum/scripts/tokenize_1t_orders.py", "--workers", "128", pinned=True)
            self.script("stream_audit", "experiments/e4_curriculum/scripts/audit_1t_token_streams.py", pinned=True)
            self.script("overlap_and_mixture", "experiments/e4_curriculum/scripts/audit_1t_overlap.py", pinned=True)
            self.script("plan_after_streams", "experiments/e4_curriculum/scripts/audit_1t_plan.py", "--output",
                        ROOT / "experiments/e4_curriculum/results/e4_1t_plan_static_audit.json")
        self.finish_and_launch()

    def finish_and_launch(self):
        self.wait_for_gpus()
        self.script("final_preflight", "pretrain/backends/torchtitan/preflight.py", "--out",
                    ROOT / "data/health/torchtitan-preflight.json", pinned=True)
        self.script("production_contract", "pretrain/scripts/validate_production_contract.py", pinned=True)
        self.require("data/health/pretrain-production-contract.json", "PASS")
        self.script("stack", "pretrain/scripts/audit_pretrain_stack.py", "--output",
                    ROOT / "data/health/pretrain-stack-audit.json")
        self.require("data/health/pretrain-stack-audit.json", "READY")
        self.wait_for_gpus()
        self.command("full_pretrain", ["bash", ROOT / "pretrain/backends/torchtitan/launch.sh",
                                      ROOT / "pretrain/configs/pretrain_base.env"])
        self.state.update(status="COMPLETE")
        self.publish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-config", type=Path, action="append",
                        help="measure only the explicitly selected candidates (repeatable)")
    parser.add_argument('--reuse-prepared-production', action='store_true',
                        help='Launch only current fully prepared production; never refreeze or rebuild data')
    args = parser.parse_args()
    pipeline = Pipeline(args.probe_config, reuse_prepared_production=args.reuse_prepared_production)
    try:
        pipeline.run()
    except Exception as exc:
        pipeline.state.update(status="BLOCKED", error=str(exc), traceback=traceback.format_exc())
        pipeline.publish()
        raise
    finally:
        pipeline.lease.release()


if __name__ == "__main__":
    main()
