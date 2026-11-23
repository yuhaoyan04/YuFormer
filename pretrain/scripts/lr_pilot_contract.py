"""Declared supplemental LR pilots shared by scheduling and evidence gates."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

SIZES = ("50m", "200m", "800m")
LEVELS = ("vlow", "low", "mid", "high", "vhigh")
EXTENSION_GROUP = "e1_pilot_extension"
CONTRACT = "pretrain/configs/lr_pilot_search.json"


def extensions(root: Path) -> list[dict]:
    path = root / CONTRACT
    if not path.is_file():
        return []
    value = json.loads(path.read_text())
    if value.get("schema_version") != 1 or not isinstance(value.get("extensions"), list):
        raise ValueError("invalid LR search contract")
    rows = value["extensions"]
    names, rates = set(), set()
    for row in rows:
        size, level, lr = row.get("size"), row.get("level"), row.get("learning_rate")
        name = row.get("run_name")
        if (size not in SIZES or not isinstance(level, str)
                or not re.fullmatch(r"ext[0-9]+", level)
                or name != f"e1_lr_{size}_{level}"
                or type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0
                or row.get("group") != EXTENSION_GROUP
                or name in names or (size, lr) in rates):
            raise ValueError("invalid or duplicate supplemental LR pilot")
        names.add(name)
        rates.add((size, lr))
    return rows


def contract_sha256(root: Path) -> str | None:
    path = root / CONTRACT
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def expected_names(root: Path) -> set[str]:
    return ({f"e1_lr_{size}_{level}" for size in SIZES for level in LEVELS}
            | {row["run_name"] for row in extensions(root)})


def selection_is_current(root: Path, selection: dict) -> bool:
    rows = extensions(root)
    expected = expected_names(root)
    if (selection.get("status") != "PASS"
            or selection.get("expected_results") != len(expected)
            or selection.get("observed_results") != len(expected)
            or selection.get("boundary_selections")
            or set(selection.get("selections", {})) != set(SIZES)):
        return False
    if rows or (root / CONTRACT).is_file():
        return (selection.get("search_contract_sha256") == contract_sha256(root)
                and set(selection.get("expected_run_names", [])) == expected)
    return True
