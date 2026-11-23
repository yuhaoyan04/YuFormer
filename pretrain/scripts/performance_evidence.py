"""Independently bind short throughput measurements to current E1 contracts."""
import hashlib
import importlib.util
import json
import math
import statistics
from pathlib import Path

import yaml

from prepare_performance_canaries import SIZES
from qualification_identity import qualification_identity, report_matches
from audit_experiment_identities import parse_env, expand_config
import batch_pilot_contract


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_source_sha(root: Path) -> str:
    script = Path(__file__).resolve().parents[1] / "backends/torchtitan/validate_contract.py"
    spec = importlib.util.spec_from_file_location("performance_contract_validator", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._source_sha256(root)


def qualified_performance(root: Path, configs: dict, presets: dict, count_params) -> dict:
    rows, issues, snapshot = [], [], {}
    for tool in (Path(__file__).resolve(), Path(__file__).with_name("prepare_performance_canaries.py").resolve()):
        snapshot[str(tool.relative_to(root)) if tool.is_relative_to(root) else str(tool)] = sha256(tool)
    source_sha = current_source_sha(root)
    try:
        path = root / "data/health/torchtitan-qualification.json"
        report = json.loads(path.read_text())
        if report.get("status") != "PASS" or not report_matches(report, qualification_identity(root)):
            raise ValueError("current backend qualification is not PASS")
        snapshot[str(path.relative_to(root))] = sha256(path)
    except (OSError, ValueError) as exc:
        return {"status": "BLOCKED", "issues": [str(exc)], "runs": [], "snapshot": {}}
    for size in SIZES:
        try:
            name = f"e1_perf_{size}"
            run = root / f"data/experiments/e1_performance/{name}"
            paths = {key: run / filename for key, filename in {
                "result": "result.json", "identity": "run_identity.json",
                "attempt": "current_attempt.json", "completion": "training_complete.json",
                "context": "execution_context.json", "config": "resolved_config.yaml",
            }.items()}
            objects = {key: json.loads(path.read_text()) for key, path in paths.items() if key != "config"}
            result, identity = objects["result"], objects["identity"]
            attempt, completion, context = (objects[key] for key in ("attempt", "completion", "context"))
            if (result.get("status") != "PASS" or completion.get("status") != "PASS"
                    or attempt.get("status") != "PASS" or context.get("loss_only") is not False
                    or context.get("performance_probe") is not True
                    or context.get("idle_admission") is not True
                    or context.get("contention_detected") is not False
                    or int(context.get("contention_checks", 0)) < 2):
                raise ValueError("throughput evidence was not committed with idle admission")
            if not attempt.get("attempt_id") or any(
                    item.get("attempt_id") != attempt["attempt_id"] for item in (result, completion, context)):
                raise ValueError("performance attempt binding mismatch")
            if any(item.get("identity_sha256") != identity.get("identity_sha256")
                   for item in (result, attempt, completion)) or identity.get("source_sha256") != source_sha:
                raise ValueError("performance code/identity drift")
            if identity.get("resolved_config_sha256") != sha256(paths["config"]):
                raise ValueError("resolved performance config drift")
            contract = root / f"pretrain/configs/e1_performance/{name}.yaml"
            environment = contract.with_suffix(".env")
            rendered = expand_config(contract, parse_env(environment))
            if hashlib.sha256(rendered.encode()).hexdigest() != identity.get("resolved_config_sha256"):
                raise ValueError("performance evidence differs from the current generated contract")
            cfg = yaml.safe_load(paths["config"].read_text())
            expected = configs[f"e1_ln_{size}"]
            if (cfg["model"]["preset"] != expected["model"]["preset"]
                    or cfg["data"]["seq_len"] != expected["data"]["seq_len"]
                    or cfg["parallel"] != expected["parallel"]
                    or cfg["optim"]["lr"] != expected["optim"]["lr"]
                    or any(cfg["train"][key] != expected["train"][key]
                           for key in ("micro_batch_size", "grad_accum", "activation_checkpointing", "bf16"))):
                raise ValueError("performance shape/batch differs from selected E1 contract")
            model_path = Path(cfg["model"]["configs"])
            if identity.get("model_configs_sha256") != sha256(model_path):
                raise ValueError("performance model config drift")
            steps = int(cfg["train"]["expected_iters"])
            points = result.get("series", {}).get("throughput(tps)", [])
            steady = [float(point["value"]) for point in points if steps // 4 < int(point["step"]) < steps]
            if (steps < 32 or len(steady) < 16
                    or any(not math.isfinite(value) or value <= 0 for value in steady)):
                raise ValueError("insufficient finite steady-state throughput observations")
            rows.append({"run_name": name, "n_params": count_params(presets[cfg["model"]["preset"]]),
                         "median_tps_per_device": statistics.median(steady),
                         "steady_points": len(steady), "identity_sha256": identity["identity_sha256"]})
            for path in (*paths.values(), model_path, contract, environment):
                snapshot[str(path.relative_to(root)) if path.is_relative_to(root) else str(path)] = sha256(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            issues.append(f"{size}: {exc}")
    return {"status": "PASS" if len(rows) == 5 and not issues else "BLOCKED",
            "runs": rows, "issues": issues, "snapshot": snapshot}


def promote_bound_loss_analysis(root: Path, group: str, analysis: dict) -> dict:
    """Qualify loss decisions without relabelling contended performance metrics."""
    if analysis.get("status") != "PASS_PROVISIONAL" or group not in {"e2", "e3", 'batch_pilot'}:
        return analysis
    try:
        path = root / "data/health/torchtitan-qualification.json"
        qualification = json.loads(path.read_text())
        audit = json.loads((root / "report/experiment_identity_audit_latest.json").read_text())
        groups = {group, batch_pilot_contract.GROUP} if group == 'batch_pilot' else {group}
        rows = [row for row in audit.get("rows", []) if row.get("group") in groups]
        expected = (len(batch_pilot_contract.expected_names(root)) if group == 'batch_pilot'
                    else {'e2': 30, 'e3': 8}[group])
        if group == 'batch_pilot' and ({row.get('name') for row in rows} != batch_pilot_contract.expected_names(root)
                                      or not batch_pilot_contract.selection_is_current(root, analysis)):
            return analysis
        if (qualification.get("status") != "PASS"
                or not report_matches(qualification, qualification_identity(root))
                or audit.get("source_sha256") != current_source_sha(root)
                or len(rows) != expected or any(row.get("status") != "PASS" for row in rows)):
            return analysis
        prefixes = tuple(f'data/experiments/{name}/' for name in groups)
        bindings = {key: value for key, value in (audit.get("evidence_snapshot") or {}).items()
                    if key.startswith(prefixes)}
        if len(bindings) < expected * 4 or any(sha256(root / key) != value for key, value in bindings.items()):
            return analysis
        runs = analysis.get("runs", analysis.get("ranking", []))
        if group == 'batch_pilot':
            runs = [row for curve in analysis.get('curves', {}).values() for row in curve]
        identities = {row["run_name"]: row["identity_sha256"] for row in runs}
        if len(identities) != expected or any(not value for value in identities.values()):
            return analysis
        analysis = {**analysis, "status": "PASS", "loss_evidence_qualification": {
            "qualification_sha256": sha256(path), "group_run_identities": identities,
            "qualification_tool_sha256": sha256(Path(__file__)),
            "scope": "loss-based selection only; shared throughput/MFU/wall_time remain excluded",
        }}
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return analysis
