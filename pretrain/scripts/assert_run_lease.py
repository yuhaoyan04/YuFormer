#!/usr/bin/env python3
"""Assert that the calling process still belongs to the current run lease."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pretrain/src"))

from yuformer_pretrain.config import ConfigError  # noqa: E402
from yuformer_pretrain.lease import RunLease  # noqa: E402


def assert_from_environment(out_dir: Path) -> None:
    token = os.environ.get("YUFORMER_RUN_LEASE_TOKEN", "")
    generation_text = os.environ.get("YUFORMER_RUN_LEASE_GENERATION", "")
    if not token or not generation_text:
        raise ConfigError("run lease token/generation is absent")
    generation = Path(generation_text).resolve()
    expected_root = (out_dir.resolve() / ".run.lease")
    if generation.parent != expected_root or not generation.name.startswith("generation-"):
        raise ConfigError("run lease generation is outside the output lease root")
    lease = RunLease(out_dir, "assert-only", heartbeat_seconds=3600)
    lease.token = token
    lease.path = generation
    lease.assert_owner()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        assert_from_environment(args.out_dir)
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 74
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
