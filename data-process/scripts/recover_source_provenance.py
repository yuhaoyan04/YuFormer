#!/usr/bin/env python3
"""Recover immutable raw-source provenance from the exact S0 input roster.

Hugging Face leaves a sidecar for every downloaded file containing the resolved
repository commit and content etag.  Combining those sidecars with S0's exact
``relpath``/``input_bytes`` summaries reconstructs the selected-file inventory
that the original download report omitted.  Direct-URL sources are verified
against configured SHA-256 values instead.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

import yaml


HEX_RE = re.compile(r"^[0-9a-f]{40,64}$")


def sha256_file(path: Path, chunk_bytes: int = 32 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def load_s0_roster(s0_root: Path, source: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((s0_root / source).rglob("*.summary.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("source") != source or row.get("stage") != "s0":
            raise ValueError(f"invalid S0 summary identity: {path}")
        records.append(
            {
                "relpath": str(row["relpath"]),
                "input_bytes": int(row["input_bytes"]),
                # Row-level parse rejection belongs to the S0 conservation
                # ledger; it does not weaken the identity of the raw file.
                "s0_error_rows": int(row.get("error_rows") or 0),
                "summary": path.as_posix(),
            }
        )
    if not records:
        raise ValueError(f"no S0 summaries for source {source}")
    if len({row["relpath"] for row in records}) != len(records):
        raise ValueError(f"duplicate S0 relpath for source {source}")
    return records


def recover_hf_record(raw_root: Path, source: str, expected_revision: str,
                      row: dict[str, Any]) -> dict[str, Any]:
    relpath = row["relpath"]
    raw_path = raw_root / source / relpath
    metadata_path = (
        raw_root / source / ".cache" / "huggingface" / "download"
        / f"{relpath}.metadata"
    )
    if not raw_path.is_file():
        raise ValueError(f"raw input missing: {raw_path}")
    if raw_path.stat().st_size != row["input_bytes"]:
        raise ValueError(f"raw size differs from S0 summary: {raw_path}")
    if not metadata_path.is_file():
        raise ValueError(f"Hugging Face metadata missing: {metadata_path}")
    lines = metadata_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2 or not HEX_RE.fullmatch(lines[0]) or not HEX_RE.fullmatch(lines[1]):
        raise ValueError(f"invalid Hugging Face metadata: {metadata_path}")
    if lines[0] != expected_revision:
        raise ValueError(
            f"resolved commit {lines[0]} != configured revision {expected_revision}: {metadata_path}"
        )
    return {
        "relpath": relpath,
        "bytes": row["input_bytes"],
        "resolved_commit": lines[0],
        "content_etag": lines[1],
        "s0_error_rows": row["s0_error_rows"],
        "s0_summary": row["summary"],
    }


def recover_direct_record(raw_root: Path, source: str, expected: dict[str, str],
                          row: dict[str, Any]) -> dict[str, Any]:
    relpath = row["relpath"]
    raw_path = raw_root / source / relpath
    expected_sha = expected.get(Path(relpath).name)
    if not isinstance(expected_sha, str) or not HEX_RE.fullmatch(expected_sha):
        raise ValueError(f"direct URL has no configured SHA-256: {source}/{relpath}")
    if not raw_path.is_file() or raw_path.stat().st_size != row["input_bytes"]:
        raise ValueError(f"direct raw input missing or size mismatch: {raw_path}")
    actual_sha = sha256_file(raw_path)
    if actual_sha != expected_sha:
        raise ValueError(f"direct raw SHA-256 mismatch: {raw_path}")
    return {
        "relpath": relpath,
        "bytes": row["input_bytes"],
        "sha256": actual_sha,
        "s0_error_rows": row["s0_error_rows"],
        "s0_summary": row["summary"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--s0-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    source_config = yaml.safe_load(args.sources.read_text(encoding="utf-8")) or {}
    configs = {row["name"]: row for row in source_config.get("sources", [])}
    scope = yaml.safe_load(args.scope.read_text(encoding="utf-8")) or {}
    included = sorted((scope.get("audit_scope") or {}).get("included_sources") or [])
    output: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for source in included:
        try:
            config = configs[source]
            roster = load_s0_roster(args.s0_root, source)
            if config.get("repo_id") == "direct-urls":
                expected = config.get("url_sha256") or {}
                with ThreadPoolExecutor(max_workers=min(args.workers, len(roster))) as pool:
                    records = list(pool.map(
                        lambda row: recover_direct_record(
                            args.raw_root, source, expected, row
                        ),
                        roster,
                    ))
                mechanism = "configured-direct-url-sha256"
                revision = hashlib.sha256(
                    json.dumps(expected, sort_keys=True, separators=(",", ":")).encode("utf-8")
                ).hexdigest()
            else:
                revision = str(config.get("revision") or "")
                if not HEX_RE.fullmatch(revision):
                    raise ValueError(f"missing immutable configured revision for {source}")
                with ThreadPoolExecutor(max_workers=min(args.workers, len(roster))) as pool:
                    records = list(pool.map(
                        lambda row: recover_hf_record(
                            args.raw_root, source, revision, row
                        ),
                        roster,
                    ))
                mechanism = "huggingface-cache-sidecar-plus-s0-roster"
            records.sort(key=lambda row: row["relpath"])
            inventory_bytes = b"".join(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
                for row in records
            )
            inventory_path = args.out_dir / f"{source}.inventory.jsonl"
            atomic_write(inventory_path, inventory_bytes)
            output[source] = {
                "status": "PASS",
                "mechanism": mechanism,
                "immutable_revision": revision,
                "files": len(records),
                "bytes": sum(int(row["bytes"]) for row in records),
                "inventory_path": inventory_path.as_posix(),
                "inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
            }
            print(f"{source}: PASS ({len(records)} files)", flush=True)
        except Exception as exc:
            failures[source] = f"{type(exc).__name__}: {exc}"
            output[source] = {"status": "FAIL", "error": failures[source]}
            print(f"{source}: FAIL: {exc}", flush=True)

    report = {
        "schema_version": 1,
        "audit": "recovered_source_provenance",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failures else "FAIL",
        "included_sources": included,
        "sources": output,
        "failures": failures,
    }
    atomic_write(
        args.out,
        (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps({"status": report["status"], "out": str(args.out), "failures": failures}, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
