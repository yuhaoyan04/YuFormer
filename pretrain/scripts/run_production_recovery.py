#!/usr/bin/env python3
"""Interrupt and restore the actual production model, microbatch, and topology."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pretrain/scripts"))
import run_torchtitan_qualification as q
from production_probe_contracts import build_probe

STEMS = ("q7_reference", "q7_training", "q7_saving", "q7_loading")
STAGES = ("reference", "training", "saving", "loading")
CONFIG_RELATIVE = "data/health/production-recovery-configs"
REPORT_RELATIVE = "data/health/production-recovery-qualification.json"


def contracts(root: Path, *, publish_probe=False):
    base = yaml.safe_load((root / "pretrain/configs/pretrain_base.yaml").read_text())
    production_env = q.parse_env(root / "pretrain/configs/pretrain_base.env")
    world = int(base["parallel"]["world_size"])
    if world != 8:
        raise ValueError("production recovery qualification requires eight GPUs")
    micro = int(base["train"]["micro_batch_size"])
    accumulation = int(base["train"]["grad_accum"])
    template, probe_identity = build_probe(root, base, production_env, role='recovery',
                                          publish=publish_probe)
    template["data"]["val_samples_per_domain"] = world * micro
    template["train"].update({
        "micro_batch_size": micro, "grad_accum": accumulation, "expected_iters": 24,
        "budget_tokens": 24 * world * micro * accumulation * template["data"]["seq_len"],
        "activation_checkpointing": base["train"]["activation_checkpointing"],
        "cuda_graphs": base["train"]["cuda_graphs"], "bf16": base["train"]["bf16"],
        "save_every": 4, "eval_every": 8, "log_every": 1, "keep_last_n": 3,
    })
    documents = {}
    for index, stem in enumerate(STEMS):
        config = json.loads(json.dumps(template))
        config["run_name"] = stem
        config_path = root / CONFIG_RELATIVE / f"{stem}.yaml"
        env = dict(production_env, GPUS_PER_NODE="8", MASTER_PORT=str(29885 + index),
                   CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
                   YUFORMER_CONFIG=str(config_path),
                   YUFORMER_PRETRAIN_OUT=str(root / "data/experiments/production_recovery" / stem),
                   YUFORMER_MODEL_CONFIGS=production_env["YUFORMER_MODEL_CONFIGS"])
        documents[config_path] = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        documents[config_path.with_suffix(".env")] = "".join(f"{key}={value}\n" for key, value in env.items())
    return documents, {"parallel": base["parallel"], "attention": template["attention"],
                       "compile": template["compile"], "micro_batch_size": micro,
                       "gradient_accumulation": accumulation, "model": base["model"],
                       "seq_len": template["data"]["seq_len"], "steps": 24,
                       "probe": probe_identity}


def production_recovery_identity(root: Path):
    documents, production = contracts(root)
    for path, expected in documents.items():
        if not path.is_file() or path.read_text() != expected:
            raise ValueError(f"production recovery contract differs from production recipe: {path}")
    relatives = ["pretrain/scripts/run_production_recovery.py",
                 "pretrain/scripts/production_probe_contracts.py",
                 "pretrain/scripts/derive_block_prefix_probe.py",
                 "pretrain/configs/pretrain_base.env", "data/health/torchtitan-qualification.json",
                 "data/health/torchtitan-env-freeze.txt", *[str(p.relative_to(root)) for p in documents]]
    rows = []
    for relative in sorted(relatives):
        path = root / relative
        rows.append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    rows.append({"current_qualification_inputs": q.qualification_identity(root)["sha256"]})
    probe_path = Path(production['probe']['probe_order_manifest'])
    if not probe_path.is_file() or hashlib.sha256(probe_path.read_bytes()).hexdigest() != production['probe']['probe_order_manifest_sha256']:
        raise ValueError('production recovery prefix probe is absent or changed')
    rows.append({'production_probe': production['probe']})
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return {"schema_version": 1, "sha256": digest, "files": rows}


def validate_report(root: Path):
    report = q.read_json_result(root / REPORT_RELATIVE)
    if (report.get("status") != "PASS"
            or not q.report_matches(report, production_recovery_identity(root))
            or any(report.get("stages", {}).get(stage, {}).get("status") != "PASS" for stage in STAGES)):
        raise ValueError("production topology recovery is absent, stale, or incomplete")
    return report


def main():
    q.RUNS = ROOT / "data/experiments/production_recovery"
    q.CONFIGS = ROOT / CONFIG_RELATIVE
    q.REPORT = ROOT / REPORT_RELATIVE
    q._REQUIRED_STAGES = STAGES
    lease = q.acquire_runner_lease()
    q._RUNNER_LEASE = lease
    try:
        qualification = q.read_json_result(ROOT / "data/health/torchtitan-qualification.json")
        if qualification.get("status") != "PASS" or not q.report_matches(qualification, q.qualification_identity(ROOT)):
            raise ValueError("current base GPU qualification must pass before production recovery")
        documents, production = contracts(ROOT, publish_probe=True)
        q.CONFIGS.mkdir(parents=True, exist_ok=True)
        for path, text in documents.items():
            temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
            temporary.write_text(text)
            os.replace(temporary, path)
        identity = production_recovery_identity(ROOT)
        report = q.load_report()
        if report and not q.report_matches(report, identity):
            archive = q.RUNS / "_stale" / str(time.time_ns())
            archive.mkdir(parents=True)
            for path in q.RUNS.iterdir():
                if path.name not in {".leases", ".run.lease", "run.lock", "_stale"}:
                    q.move_artifact(path, archive / path.name)
            os.replace(q.REPORT, archive / q.REPORT.name)
            report = {}
        q._QUALIFICATION_IDENTITY = identity
        report["production_contract"] = production
        if report.get("stages", {}).get("reference", {}).get("status") != "PASS":
            rc, evidence = q.launch(STEMS[0], list(range(8)), actual_world=8)
            collected, result = q.collect(STEMS[0]) if rc == 0 else (1, {})
            passed = rc == 0 and collected == 0 and result.get("status") == "PASS"
            q.save_stage(report, "reference", "PASS" if passed else "FAIL", **evidence)
            if not passed:
                return 1
        for stage, name in zip(STAGES[1:], STEMS[1:], strict=True):
            if report.get("stages", {}).get(stage, {}).get("status") == "PASS":
                continue
            marker = q.RUNS / name / "fault_injected.json"
            fault = q.read_json_result(marker)
            if stage == "loading" and not fault.get("seed_checkpoint_step"):
                _, seeded = q.launch(name, list(range(8)), actual_world=8, inject_mode="checkpoint")
                if seeded.get("killed_at_step") is None:
                    q.save_stage(report, stage, "FAIL", reason="load fault checkpoint was not created", **seeded)
                    return 1
                fault["seed_checkpoint_step"] = seeded["killed_at_step"]
                q.atomic_json(marker, fault)
            if not fault.get("injection"):
                _, injected = q.launch(name, list(range(8)), actual_world=8, inject_mode=stage)
                if injected.get("killed_at_step") is None or (
                        stage == "saving" and not q.saving_interrupt_observed(injected)):
                    q.save_stage(report, stage, "FAIL", reason="requested fault was not observed", **injected)
                    return 1
                fault["injection"] = injected
                q.atomic_json(marker, fault)
            baseline = fault.get("seed_checkpoint_step", fault["injection"]["killed_at_step"])
            rc, evidence = q.launch(name, list(range(8)), actual_world=8)
            collected, result = q.collect(name) if rc == 0 else (1, {})
            cursor = q.verify_cursor(name, source_world=8, resumed_world=8, checkpoint_step=baseline,
                                     local_batch_size=production["micro_batch_size"],
                                     grad_accum=production["gradient_accumulation"])
            parity = q.compare_loss(STEMS[0], name, after_step=baseline, through_step=24)
            passed = (rc == 0 and collected == 0 and result.get("status") == "PASS"
                      and cursor["status"] == "PASS" and parity["status"] == "PASS")
            q.save_stage(report, stage, "PASS" if passed else "FAIL", cursor=cursor,
                         loss_parity=parity, fault_evidence=fault, **evidence)
            if not passed:
                return 1
        validate_report(ROOT)
        return 0
    finally:
        lease.release()


if __name__ == "__main__":
    raise SystemExit(main())
