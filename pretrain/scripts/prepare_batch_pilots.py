#!/usr/bin/env python3
"""Bind the batch/LR joint pilot grid to the selected 200M base LR."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lr_pilot_contract import selection_is_current
from night_research import batch_anchor_is_current


ROOT = Path(__file__).resolve().parents[2]
NAME = re.compile(r"^batch_(1m|2m|4m)_(low|mid|high)$")
LEVEL_MULTIPLIER = {"low": 0.5, "mid": 1.0, "high": 2.0}
BASE_TOKENS_PER_STEP = 4096 * 4 * 16


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def prepare(root: Path, selection_path: Path) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not (selection_is_current(root, selection) or batch_anchor_is_current(root, selection)):
        raise ValueError("current declared and bracketed LR selection must PASS")
    base_lr = selection["selections"]["200m"].get("learning_rate")
    if not isinstance(base_lr, (int, float)) or not math.isfinite(float(base_lr)) or base_lr <= 0:
        raise ValueError("selected 200M learning rate is invalid")
    config_dir = root / "pretrain/configs/batch_pilot"
    expected = {f"batch_{batch}_{level}" for batch in ("1m", "2m", "4m")
                for level in LEVEL_MULTIPLIER}
    paths = {path.stem: path for path in config_dir.glob("*.yaml")}
    if set(paths) != expected:
        raise ValueError("batch pilot config roster is not exact")
    rows, pending_writes = [], []
    for name in sorted(expected):
        match = NAME.fullmatch(name)
        assert match is not None
        path = paths[name]
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        tps = (int(config["data"]["seq_len"])
               * int(config["train"]["micro_batch_size"])
               * int(config["train"]["grad_accum"])
               * int(config["parallel"]["world_size"]))
        center = float(base_lr) * math.sqrt(tps / BASE_TOKENS_PER_STEP)
        lr = center * LEVEL_MULTIPLIER[match.group(2)]
        config["optim"]["lr"] = float(f"{lr:.10g}")
        config["optim"]["batch_pilot_resolution"] = {
            "method": "sqrt_tokens_per_step",
            "base_lr": float(base_lr),
            "base_tokens_per_step": BASE_TOKENS_PER_STEP,
            "tokens_per_step": tps,
            "level_multiplier": LEVEL_MULTIPLIER[match.group(2)],
            "lr_selection_sha256": sha256(selection_path),
        }
        content = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        env_path = path.with_suffix('.env')
        if env_path.is_file() and path.read_text() != content:
            values = dict(line.split('=', 1) for line in env_path.read_text().splitlines()
                          if line and not line.startswith('#'))
            run = Path(values['YUFORMER_PRETRAIN_OUT'])
            if (run / 'run_identity.json').exists() or (run / 'current_attempt.json').exists():
                raise ValueError(f'refusing to change a batch contract that already started: {name}')
        pending_writes.append((path, content))
        rows.append({"run_name": name, "config": str(path.relative_to(root)),
                     "config_sha256": hashlib.sha256(content.encode()).hexdigest(), "tokens_per_step": tps,
                     "learning_rate": config["optim"]["lr"]})
    for path, content in pending_writes:
        if path.read_text() != content:
            atomic_text(path, content)
    return {
        "schema_version": 1, "status": "PASS",
        "preparer_tool_sha256": sha256(Path(__file__)),
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "lr_selection": str(selection_path.relative_to(root)),
        "lr_selection_sha256": sha256(selection_path), "base_200m_lr": float(base_lr),
        "configs": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--selection", type=Path,
                        default=ROOT / "experiments/e1_scaling/results/lr_selection.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments/e1_scaling/results/batch_pilot_resolution.json")
    args = parser.parse_args()
    result = prepare(args.root.resolve(), args.selection.resolve())
    atomic_text(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "configs": len(result["configs"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
