#!/usr/bin/env python3
"""Audit grouped RLVR rollout JSONL before an optimizer update."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yuformer_rl.diagnostics import audit_rollouts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--max-response-tokens", type=int, required=True)
    parser.add_argument("--variance-epsilon", type=float, default=1e-12)
    args = parser.parse_args()
    records = []
    parse_errors = []
    with args.input.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                parse_errors.append({"row": line_number, "errors": [f"invalid JSON: {exc}"]})
    result = audit_rollouts(
        records,
        expected_group_size=args.group_size,
        max_response_tokens=args.max_response_tokens,
        variance_epsilon=args.variance_epsilon,
    )
    if parse_errors:
        result["status"] = "FAIL"
        result["errors"] = parse_errors + result["errors"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": result["status"], "out": str(args.out)}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
