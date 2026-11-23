#!/usr/bin/env python3
"""Recover damaged S5 summaries from certified compact outputs without text IO."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from yuformer_data.scale.driver import _acquire_stage_lease
from yuformer_data.scale.length_bucket import (
    load_context_lengths,
    recover_s5_summary_from_certified_output,
    resolve_tokenizer_path,
)
from yuformer_data.scale.profiles import DEFAULT_PROFILES_PATH, load_profiles


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("../data/processed"))
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES_PATH)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--lineage-audit", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--audit-out", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads(args.lineage_audit.read_text(encoding="utf-8"))
    if not audit.get("passed") or int(audit.get("failed") or 0):
        raise SystemExit("lineage audit is not a passing dataset-level certificate")
    token_audit_sha256 = str(audit.get("token_audit_sha256") or "")
    if len(token_audit_sha256) != 64:
        raise SystemExit("lineage audit has no valid token_audit_sha256")

    profiles = load_profiles(args.profiles)
    tokenizer = resolve_tokenizer_path(args.tokenizer)
    contexts = load_context_lengths()
    damaged: list[tuple[str, str, str]] = []
    for path in args.out_root.joinpath("s5").rglob("*.summary.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if not summary.get("errors") and not summary.get("shard_fatal"):
            continue
        source = str(summary.get("source") or "")
        relpath = str(summary.get("relpath") or "")
        if source not in profiles or not relpath:
            raise SystemExit(f"damaged summary lacks a known identity: {path}")
        damaged.append((source, relpath, str(profiles[source]["domain"])))

    lease = _acquire_stage_lease(args.out_root, "s5")
    if lease is None:
        raise SystemExit("another S5 writer owns the shared stage lease")
    started = time.monotonic()
    recovered: list[dict] = []
    failures: list[dict[str, str]] = []
    per_source: Counter[str] = Counter()
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(
                    recover_s5_summary_from_certified_output,
                    args.out_root,
                    source,
                    relpath,
                    domain,
                    contexts,
                    tokenizer,
                    audit_sha256=token_audit_sha256,
                ): (source, relpath)
                for source, relpath, domain in damaged
            }
            for index, future in enumerate(as_completed(futures), 1):
                source, relpath = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failures.append(
                        {"source": source, "relpath": relpath, "error": f"{type(exc).__name__}: {exc}"}
                    )
                else:
                    recovered.append(result)
                    per_source[source] += 1
                if index % 10 == 0 or index == len(futures):
                    print(
                        f"progress {index}/{len(futures)} recovered={len(recovered)} "
                        f"failed={len(failures)}",
                        flush=True,
                    )
    finally:
        lease.release()

    payload = {
        "schema_version": 1,
        "stage": "s5_summary_recovery",
        "status": "PASS" if not failures else "FAIL",
        "damaged": len(damaged),
        "recovered": len(recovered),
        "failed": len(failures),
        "failures": failures,
        "per_source": dict(sorted(per_source.items())),
        "s3_text_read": False,
        "ordered_keep_set_verified": True,
        "lineage_audit": {
            "path": str(args.lineage_audit),
            "sha256": _sha256(args.lineage_audit),
            "token_audit_sha256": token_audit_sha256,
        },
        "tokenizer_sha256": _sha256(tokenizer),
        "elapsed_sec": round(time.monotonic() - started, 3),
    }
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.audit_out.with_name(args.audit_out.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    tmp.replace(args.audit_out)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

