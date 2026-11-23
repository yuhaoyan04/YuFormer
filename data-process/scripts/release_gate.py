#!/usr/bin/env python3
"""Evaluate a data release spec and emit its signed gate manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "data-process" / "src"))

from yuformer_data.release import (  # noqa: E402
    evaluate_release,
    load_release_spec,
    write_release_result,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--hash-blocked-artifacts", action="store_true")
    args = parser.parse_args()
    result = evaluate_release(
        load_release_spec(args.spec),
        repo_root=REPO_ROOT,
        hash_artifacts_when_blocked=args.hash_blocked_artifacts,
    )
    write_release_result(args.out, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "dataset_id": result["dataset_id"],
                "snapshot_id": result["snapshot_id"],
                "failed_gates": result["failed_gates"],
                "missing_artifacts": result["missing_artifacts"],
                "manifest": args.out.as_posix(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
