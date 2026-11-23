#!/usr/bin/env python3
"""Reinvoke S6 and certify that every published artifact remains byte-identical."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(paths: list[Path]) -> dict[str, dict[str, int | str]]:
    return {
        path.as_posix(): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in paths
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--mix-config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()

    out_root = args.out_root.resolve()
    cfg = yaml.safe_load(args.mix_config.read_text(encoding="utf-8"))
    mix = str(cfg["mix_name"])
    manifest_path = out_root / "mix_order" / f"{mix}.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = [
        out_root / "mix_members" / f"{mix}.parquet",
        *(out_root / "mix_order" / name for name in manifest.get("order_tables", [])),
        manifest_path,
        out_root / "stage_s6_global_summary.json",
    ]
    if manifest.get("goldrule_research"):
        paths.append(out_root / str(manifest["goldrule_research"]))
    missing = [path.as_posix() for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"S6 published artifacts missing: {missing}")

    before = snapshot(paths)
    command = [
        sys.executable,
        "-m",
        "yuformer_data.scale.driver",
        "--stage",
        "s6",
        "--mix-config",
        str(args.mix_config.resolve()),
        "--workers",
        str(args.workers),
        "--out-root",
        str(out_root),
    ]
    env = dict(os.environ)
    src = Path(__file__).resolve().parents[1] / "src"
    env["PYTHONPATH"] = str(src) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["YUFORMER_ALLOW_PAUSED_S6"] = "1"
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    after = snapshot(paths)
    unchanged = before == after
    output = completed.stdout or ""
    skipped = "already materialized" in output
    report = {
        "audit": "s6_idempotency",
        "schema_version": 1,
        "mix_name": mix,
        "status": "PASS" if completed.returncode == 0 and unchanged and skipped else "FAIL",
        "driver": {
            "command": command,
            "returncode": completed.returncode,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "reported_idempotent_skip": skipped,
            "output_tail": output[-4000:],
        },
        "artifacts_before": before,
        "artifacts_after": after,
        "artifacts_byte_identical": unchanged,
    }
    atomic_json(args.out, report)
    print(json.dumps({
        "status": report["status"],
        "returncode": completed.returncode,
        "reported_idempotent_skip": skipped,
        "artifacts_byte_identical": unchanged,
        "out": args.out.as_posix(),
    }, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
