from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_manifest(
    records: list[dict[str, Any]],
    *,
    stage: str,
    output: str | None = None,
) -> dict[str, Any]:
    counts: dict[str, int] = {"keep": 0, "review": 0, "drop": 0}
    total = 0
    for record in records:
        total += 1
        decision = str(record.get("decision", "keep"))
        counts[decision] = counts.get(decision, 0) + 1
    manifest: dict[str, Any] = {"stage": stage, "records": total, "decisions": counts}
    if output is not None:
        manifest["output"] = output
    return manifest


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(dict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
