#!/usr/bin/env python3
"""Generate five short, idle-GPU throughput contracts from resolved E1 arms."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

SIZES = ("50m", "100m", "200m", "400m", "800m")


def training_contract(cfg: dict) -> dict:
    """Exclude only the resolver's provenance annotation, never optimizer settings."""
    value = copy.deepcopy(cfg)
    value.get("optim", {}).pop("lr_resolution", None)
    return value


def write_if_changed(path: Path, text: str) -> None:
    if not path.is_file() or path.read_text() != text:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text)
        temporary.replace(path)


def prepare(root: Path, steps: int = 32) -> dict:
    if steps < 32:
        raise ValueError("performance canaries need at least 32 updates")
    resolution = root / "experiments/e1_scaling/results/lr_resolution.json"
    if json.loads(resolution.read_text()).get("status") != "PASS":
        raise ValueError("LR/batch resolution must PASS before performance contracts")
    generated, pending_writes = [], []
    for index, size in enumerate(SIZES):
        source = root / f"pretrain/configs/e1/e1_ln_{size}.yaml"
        cfg = yaml.safe_load(source.read_text())
        name = f"e1_perf_{size}"
        cfg["run_name"] = name
        cfg["experiment_status"] = "performance_qualification"
        tps = (cfg["data"]["seq_len"] * cfg["train"]["micro_batch_size"]
               * cfg["train"]["grad_accum"] * cfg["parallel"]["world_size"])
        if cfg["parallel"]["world_size"] != 1:
            raise ValueError("E1 throughput canaries must retain the single-GPU topology")
        cfg["optim"].update(warmup_steps=2, lr_schedule="cosine")
        cfg["optim"].pop("decay_steps", None)
        cfg["train"].update(expected_iters=steps, budget_tokens=steps * tps,
                            log_every=1, save_every=steps, eval_every=steps,
                            keep_last_n=2, keep_every=0)
        target = root / f"pretrain/configs/e1_performance/{name}.yaml"
        config_text = yaml.safe_dump(cfg, sort_keys=False)
        source_env = root / f"pretrain/configs/e1/e1_ln_{size}.env"
        values = dict(line.split("=", 1) for line in source_env.read_text().splitlines()
                      if line.strip() and not line.lstrip().startswith("#"))
        values.update(YUFORMER_CONFIG=f"configs/e1_performance/{name}.yaml",
                      YUFORMER_PRETRAIN_OUT=str(root / f"data/experiments/e1_performance/{name}"),
                      MASTER_PORT=str(29820 + index))
        env_path = target.with_suffix(".env")
        env_text = "".join(f"{key}={value}\n" for key, value in values.items())
        identity = root / f"data/experiments/e1_performance/{name}/run_identity.json"
        if identity.exists():
            # Preserve the exact executed bytes and their original provenance.
            # A changed report reference alone cannot invalidate a measurement.
            old_text = target.read_text()
            if (training_contract(yaml.safe_load(old_text)) != training_contract(cfg)
                    or env_path.read_text() != env_text):
                raise ValueError(f"refusing to change started performance contract: {name}")
            config_text = old_text
        pending_writes.extend(((target, config_text), (env_path, env_text)))
        generated.append({"name": name, "source_config": str(source.relative_to(root)),
                          "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                          "config": str(target.relative_to(root)),
                          "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
                          "tokens_per_update": tps})
    # Validate every arm before changing any file.
    for path, text in pending_writes:
        write_if_changed(path, text)
    result = {"schema_version": 1, "status": "PASS", "steps": steps,
              "admission_policy": "idle_GPU_only; contended throughput is ineligible",
              "lr_resolution_sha256": hashlib.sha256(resolution.read_bytes()).hexdigest(),
              "generated": generated}
    write_if_changed(root / "experiments/e1_scaling/results/performance_contracts.json",
                     json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(prepare(args.root.resolve()), sort_keys=True))
