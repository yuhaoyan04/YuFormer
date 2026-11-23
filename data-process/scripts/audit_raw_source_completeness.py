#!/usr/bin/env python3
"""Verify the selected raw-file roster against immutable upstream metadata.

This audit complements recovered provenance: provenance proves the identity of
files that entered S0, while this audit proves that the downloader did not stop
early or omit an upstream file from the deterministic selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import blake2b
import fnmatch
import json
import os
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
import yaml


def _matches(path: str, patterns: list[str]) -> bool:
    return not patterns or any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _selected_inventory(config: dict[str, Any], files: list[tuple[str, int]]) -> list[tuple[str, int]]:
    include = list(config.get("include_patterns") or [])
    exclude = list(config.get("exclude_patterns") or [])
    filtered = [
        (path, int(size or 0))
        for path, size in files
        if _matches(path, include) and not (exclude and _matches(path, exclude))
    ]
    max_size_gb = float(config.get("max_size_gb") or 0)
    if max_size_gb <= 0:
        return sorted(filtered)

    source = str(config["name"])

    def order_key(item: tuple[str, int]) -> bytes:
        digest = blake2b(digest_size=16, person=source[:16].encode().ljust(16, b"\0"))
        digest.update(item[0].encode())
        return digest.digest()

    limit = max_size_gb * 1e9
    selected: list[tuple[str, int]] = []
    cumulative = 0
    for item in sorted(filtered, key=order_key):
        if cumulative >= limit:
            continue
        selected.append(item)
        cumulative += item[1]
    return sorted(selected)


def _s0_inventory(s0_root: Path, source: str) -> list[tuple[str, int]]:
    rows: list[tuple[str, int]] = []
    for path in sorted((s0_root / source).rglob("*.summary.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("stage") != "s0" or value.get("source") != source:
            raise ValueError(f"invalid S0 identity: {path}")
        rows.append((str(value["relpath"]), int(value["input_bytes"])))
    if not rows:
        raise ValueError(f"no S0 summaries: {source}")
    return sorted(rows)


def compare_inventories(
    expected: list[tuple[str, int]], observed: list[tuple[str, int]]
) -> dict[str, Any]:
    expected_map = dict(expected)
    observed_map = dict(observed)
    missing = sorted(set(expected_map) - set(observed_map))
    unexpected = sorted(set(observed_map) - set(expected_map))
    size_mismatches = [
        {"path": path, "expected_bytes": expected_map[path], "observed_bytes": observed_map[path]}
        for path in sorted(set(expected_map) & set(observed_map))
        if expected_map[path] != observed_map[path]
    ]
    return {
        "status": "PASS" if not missing and not unexpected and not size_mismatches else "FAIL",
        "expected_files": len(expected),
        "observed_files": len(observed),
        "expected_bytes": sum(size for _, size in expected),
        "observed_bytes": sum(size for _, size in observed),
        "missing_files": missing,
        "unexpected_files": unexpected,
        "size_mismatches": size_mismatches,
    }


def _remote_inventory(api: HfApi, config: dict[str, Any]) -> tuple[str, list[tuple[str, int]]]:
    if config.get("repo_id") == "direct-urls":
        files = [(url.rstrip("/").rsplit("/", 1)[-1], 0) for url in config.get("urls") or []]
        return "configured-direct-urls", files
    revision = str(config.get("revision") or "")
    info = api.repo_info(
        str(config["repo_id"]), repo_type="dataset", revision=revision, files_metadata=True
    )
    if str(info.sha) != revision:
        raise ValueError(f"resolved revision {info.sha} != configured {revision}")
    files = [(sibling.rfilename, int(sibling.size or 0)) for sibling in (info.siblings or [])]
    return revision, _selected_inventory(config, files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--s0-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source_cfg = yaml.safe_load(args.sources.read_text(encoding="utf-8")) or {}
    configs = {row["name"]: row for row in source_cfg.get("sources") or []}
    scope_cfg = yaml.safe_load(args.scope.read_text(encoding="utf-8")) or {}
    audit_scope = scope_cfg.get("audit_scope") or {}
    included = sorted(audit_scope.get("included_sources") or [])
    excluded_reasons = dict(audit_scope.get("excluded_sources") or {})
    audited = sorted(set(included) | (set(excluded_reasons) & set(configs)))
    api = HfApi()
    results: dict[str, Any] = {}
    for source in audited:
        try:
            config = configs[source]
            revision, expected = _remote_inventory(api, config)
            observed = _s0_inventory(args.s0_root, source)
            # Direct URL sizes are not advertised in config.  File presence is
            # checked here; recovered provenance separately verifies SHA-256.
            if config.get("repo_id") == "direct-urls":
                observed_sizes = dict(observed)
                expected = [(path, observed_sizes.get(path, -1)) for path, _ in expected]
            result = compare_inventories(expected, observed)
            result["immutable_revision"] = revision
        except Exception as exc:
            result = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
        result["release_scope"] = "included" if source in included else "excluded"
        if source in excluded_reasons:
            result["exclusion_reason"] = excluded_reasons[source]
        results[source] = result
        print(f"{source}: {result['status']} ({result['release_scope']})", flush=True)

    included_failures = [source for source in included if results.get(source, {}).get("status") != "PASS"]
    report = {
        "schema_version": 1,
        "audit": "raw_source_completeness",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not included_failures else "FAIL",
        "included_sources": included,
        "included_failures": included_failures,
        "sources": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.out)
    print(json.dumps({"status": report["status"], "included_failures": included_failures}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
