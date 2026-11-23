#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from yuformer_eval.audit import audit_result_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a YuFormer evaluation result manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = audit_result_manifest(json.loads(args.manifest.read_text()), root=args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
