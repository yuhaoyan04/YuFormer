#!/usr/bin/env python3
"""Validate every generated E1 pilot, E1, E2, and conditional E3 contract."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import yaml
from research_budget import screen_budget
from analyze_e2_results import flops_per_token
from lr_pilot_contract import extensions, EXTENSION_GROUP
import batch_pilot_contract as batch_contract
from night_batch_search import extended_config
from research_scope import group_enabled


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = ROOT / "pretrain/backends/torchtitan/validate_contract.py"
GROUPS = ("e1_pilot", "batch_pilot", "e1", "e1_seedrep", "e2", "e3", 'e2_confirm', 'e3_confirm', EXTENSION_GROUP, batch_contract.GROUP)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        try:
            previous_text = path.read_text(encoding="utf-8")
            if previous_text == content:
                return
            previous = json.loads(previous_text)
            current = json.loads(content)
            if isinstance(previous, dict) and isinstance(current, dict):
                previous.pop("observed_at", None)
                current.pop("observed_at", None)
                if previous == current:
                    return
        except (OSError, ValueError, AttributeError):
            pass
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        if not key or not key.replace("_", "").isalnum():
            raise ValueError(f"{path}:{line_number}: invalid environment key")
        if any(token in value for token in ("$(", "`", ";", "&&", "||")):
            raise ValueError(f"{path}:{line_number}: shell expressions are not supported")
        values[key] = value.strip().strip("'\"")
    return values


@contextmanager
def scoped_environment(values: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_validator() -> Any:
    spec = importlib.util.spec_from_file_location("yuformer_validate_contract", VALIDATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {VALIDATOR}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_roster(root: Path) -> dict[str, int]:
    generation = root / "experiments/e3_data_mix/results/config_generation.json"
    try:
        e3_ready = json.loads(generation.read_text(encoding="utf-8")).get("status") == "PASS"
    except (OSError, ValueError):
        e3_ready = False
    expected = {"e1_pilot": 15, "batch_pilot": 9, "e1": 24, "e2": 30,
                "e3": 8 if e3_ready else 0}
    seedrep = root / "data/health/e1_seedrep-contracts.json"
    expected["e1_seedrep"] = len(json.loads(seedrep.read_text()).get("runs", [])) if seedrep.is_file() else 0
    expected[EXTENSION_GROUP] = len(extensions(root))
    expected[batch_contract.GROUP] = len(batch_contract.extensions(root))
    for stage in ('e2', 'e3'):
        path = root / f'data/health/{stage}_confirm-contracts.json'
        expected[f'{stage}_confirm'] = len(json.loads(path.read_text()).get('runs', [])) if path.is_file() else 0
    return {group: count if group_enabled(root, group) else 0 for group, count in expected.items()}


def expected_name_roster(root: Path) -> dict[str, set[str]]:
    pilots = {f"e1_lr_{size}_{level}" for size in ("50m", "200m", "800m")
              for level in ("vlow", "low", "mid", "high", "vhigh")}
    e1 = {f"e1_ln_{size}" for size in ("50m", "100m", "200m", "400m", "800m")}
    e1 |= {f"e1_ld_300m_{size}" for size in ("100m", "300m", "1b", "3b", "9p85b")}
    e1 |= {f"e1_iso_c{line}_{size}"
           for line, sizes in ((0, ("50m", "100m", "200m", "400m", "800m")),
                               (1, ("50m", "100m", "200m", "400m", "800m")),
                               (2, ("100m", "200m", "400m", "800m")))
           for size in sizes}
    architectures = (
        "e2_l16", "e2_l32", "e2_l48", "e2_ffn2x", "e2_ffn4x",
        "e2_nogate", "e2_noqkn", "e2_fullrope", "e2_swaoff", "e2_untied",
    )
    e2 = {name if seed == 1 else f"{name}_s{seed}"
          for name in architectures for seed in (1, 2, 3)}
    e3: set[str] = set()
    if expected_roster(root)["e3"]:
        matrix = yaml.safe_load(
            (root / "experiments/e3_data_mix/configs/e3_matrix.yaml").read_text(encoding="utf-8")
        )
        e3 = set(matrix["arms"])
    batches = {f"batch_{batch}_{level}" for batch in ("1m", "2m", "4m")
               for level in ("low", "mid", "high")}
    names = {"e1_pilot": pilots, "batch_pilot": batches, "e1": e1, "e2": e2, "e3": e3}
    names[EXTENSION_GROUP] = {row['run_name'] for row in extensions(root)}
    names[batch_contract.GROUP] = {row['run_name'] for row in batch_contract.extensions(root)}
    for stage in ('e2', 'e3'):
        manifest = root / f'data/health/{stage}_confirm-contracts.json'
        value = json.loads(manifest.read_text()) if manifest.is_file() else {}
        names[f'{stage}_confirm'] = {row['name'] for row in value.get('runs', [])}
    seedrep_manifest = root / 'data/health/e1_seedrep-contracts.json'
    seedrep = json.loads(seedrep_manifest.read_text()) if seedrep_manifest.is_file() else {}
    names['e1_seedrep'] = {row['name'] for row in seedrep.get('runs', [])}
    return {group: roster if group_enabled(root, group) else set() for group, roster in names.items()}


def validate_all(root: Path) -> dict[str, Any]:
    validator = load_validator()
    model_presets = yaml.safe_load(
        (root / "model/configs/model_configs.yaml").read_text(encoding="utf-8")
    )["presets"]
    rows: list[dict[str, Any]] = []
    for group in GROUPS:
        if not group_enabled(root, group):
            continue
        for env_path in sorted((root / "pretrain/configs" / group).glob("*.env")):
            row: dict[str, Any] = {
                "group": group,
                "env": str(env_path.relative_to(root)),
            }
            try:
                values = parse_env(env_path)
                raw_config = values["YUFORMER_CONFIG"]
                config_path = Path(raw_config)
                if not config_path.is_absolute():
                    config_path = root / "pretrain" / config_path
                row["config"] = str(config_path.relative_to(root))
                with scoped_environment(values):
                    result = validator.validate(root, config_path, publish_identity=False)
                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                if group == EXTENSION_GROUP:
                    declared = next(row for row in extensions(root) if row['run_name'] == config['run_name'])
                    baseline_path = root / f"pretrain/configs/e1_pilot/e1_lr_{declared['size']}_vhigh.yaml"
                    baseline = yaml.safe_load(baseline_path.read_text())
                    baseline['run_name'] = declared['run_name']
                    baseline['optim']['lr'] = declared['learning_rate']
                    if config != baseline:
                        raise ValueError('supplemental LR pilot differs from its matched baseline')
                if group == batch_contract.GROUP:
                    declared = next(row for row in batch_contract.extensions(root) if row['run_name'] == config['run_name'])
                    baseline = yaml.safe_load((root / f"pretrain/configs/batch_pilot/batch_{declared['batch']}_mid.yaml").read_text())
                    if config != extended_config(baseline, declared):
                        raise ValueError('supplemental batch pilot differs from its matched baseline')
                if group == 'e1' and config['run_name'].startswith('e1_iso_'):
                    compute = config['research_compute']
                    per_token = flops_per_token(model_presets[config['model']['preset']], int(config['data']['seq_len']))
                    if compute['algorithmic_flops_per_token'] != per_token:
                        raise ValueError('isoFLOP model compute accounting differs')
                    tpu = config['data']['seq_len'] * config['train']['micro_batch_size'] * config['train']['grad_accum']
                    trained = config['train']['expected_iters'] * tpu
                    gap = compute['target_algorithmic_flops'] - trained * per_token
                    if gap < 0 or gap >= tpu * per_token:
                        raise ValueError('isoFLOP arm is not matched within one optimizer update')
                if group == "e2":
                    policy = yaml.safe_load((root / 'pretrain/configs/research_budget.yaml').read_text())
                    if int(config["train"]["budget_tokens"]) != int(policy['screen_budget_tokens']):
                        raise ValueError("formal E2 budget differs from research budget policy")
                    preset = model_presets[config["model"]["preset"]]
                    head_dim = int(preset["hidden_size"]) // int(preset["num_heads"])
                    if head_dim % 8:
                        raise ValueError(f"E2 head_dim must be a multiple of 8, got {head_dim}")
                if group in {"batch_pilot", batch_contract.GROUP}:
                    largest_tps = 4096 * 4 * 256
                    matched_budget = 2_000_000_000 // largest_tps * largest_tps
                    if int(config["train"]["budget_tokens"]) != matched_budget:
                        raise ValueError(
                            "formal batch pilot arms must share the optimizer-step-aligned ~2B budget"
                        )
                if group.endswith('_confirm'):
                    policy = yaml.safe_load((root / 'pretrain/configs/research_budget.yaml').read_text())['confirmation']
                    if int(config['train']['budget_tokens']) != int(policy['budget_tokens']):
                        raise ValueError('confirmation budget differs from the research policy')
                    if int(config['seed']) not in policy['seeds']:
                        raise ValueError('confirmation seed differs from the research policy')
                if group == 'e1_seedrep':
                    manifest = json.loads((root / 'data/health/e1_seedrep-contracts.json').read_text())
                    contract = next(item for item in manifest['runs'] if item['name'] == config['run_name'])
                    source = yaml.safe_load((root / f"pretrain/configs/e1/{contract['source_run']}.yaml").read_text())
                    expected_config = dict(source)
                    expected_config.update(run_name=config['run_name'], seed=int(contract['seed']))
                    if config != expected_config:
                        raise ValueError('E1 replication differs from its matched source design point')
                    if int(config['train']['budget_tokens']) != int(contract['budget_tokens']):
                        raise ValueError('E1 replication token budget differs from the source point')
                if group == "e3":
                    matrix = yaml.safe_load(
                        (root / "experiments/e3_data_mix/configs/e3_matrix.yaml").read_text(
                            encoding="utf-8"
                        )
                    )
                    tps = (int(config["data"]["seq_len"])
                           * int(config["train"]["micro_batch_size"])
                           * int(config["train"]["grad_accum"])
                           * int(config["parallel"]["world_size"]))
                    expected_budget = screen_budget(root, int(matrix["budget_tokens_per_arm"])) // tps * tps
                    if int(config["train"]["budget_tokens"]) != expected_budget:
                        raise ValueError("formal E3 arm does not use the aligned matrix budget")
                row.update({"status": "PASS", "run_name": result["identity"]["run_name"],
                            'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
                            'env_sha256': hashlib.sha256(env_path.read_bytes()).hexdigest()})
            except Exception as exc:
                row.update({"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
            rows.append(row)
    passed = sum(row["status"] == "PASS" for row in rows)
    expected_names = expected_name_roster(root)
    expected = {group: len(names) for group, names in expected_names.items()}
    observed_names = {group: {Path(row["env"]).stem for row in rows if row["group"] == group}
                      for group in GROUPS}
    observed = {group: len(names) for group, names in observed_names.items()}
    roster_exact = observed_names == expected_names
    return {
        "schema_version": 1,
        "status": "PASS" if roster_exact and passed == len(rows) else "FAIL",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "expected_by_group": expected,
        "observed_by_group": observed,
        "missing_by_group": {group: sorted(expected_names[group] - observed_names[group])
                             for group in GROUPS},
        "unexpected_by_group": {group: sorted(observed_names[group] - expected_names[group])
                                for group in GROUPS},
        "roster_exact": roster_exact,
        "passed": passed,
        "total": len(rows),
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path("report/formal_contract_validation_latest.json"))
    parser.add_argument("--tsv", type=Path, default=Path("report/formal_contract_validation_latest.tsv"))
    args = parser.parse_args()
    root = args.root.resolve()
    result = validate_all(root)
    result["validation_tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = args.output if args.output.is_absolute() else root / args.output
    tsv = args.tsv if args.tsv.is_absolute() else root / args.tsv
    atomic_write(output, json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    lines = ["status\tgroup\tenv\tconfig\terror"]
    for row in result["rows"]:
        lines.append("\t".join(str(row.get(key, "")).replace("\t", " ").replace("\n", " ")
                               for key in ("status", "group", "env", "config", "error")))
    atomic_write(tsv, "\n".join(lines) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "passed", "total", "roster_exact")}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
