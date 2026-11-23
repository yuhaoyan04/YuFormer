#!/usr/bin/env python3
"""Issue the final pretrain data-gate certificate from independently audited inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_bytes: int = 64 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_pass(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise ValueError(f"{label} status is not PASS: {payload.get('status')}")
    return payload


def build_certificate(
    *,
    release_path: Path,
    token_stream_audit_path: Path,
    corpus_manifest_path: Path,
    corpus_audit_path: Path,
) -> dict[str, Any]:
    release = load_pass(release_path, "data release")
    stream = load_pass(token_stream_audit_path, "token stream audit")
    corpus = load_pass(corpus_manifest_path, "tokenizer corpus manifest")
    corpus_audit = load_pass(corpus_audit_path, "tokenizer corpus audit")
    dataset_id = release.get("dataset_id")
    snapshot_id = release.get("snapshot_id")
    if not dataset_id or not snapshot_id:
        raise ValueError("data release lacks dataset_id or snapshot_id")
    identities = {
        "release": (dataset_id, snapshot_id),
        "token_stream_audit": (stream.get("dataset_id"), stream.get("snapshot_id")),
        "tokenizer_corpus": (corpus.get("dataset_id"), corpus.get("snapshot_id")),
        "tokenizer_corpus_audit": (
            corpus_audit.get("dataset_id"), corpus_audit.get("snapshot_id")
        ),
    }
    if any(value != (dataset_id, snapshot_id) for value in identities.values()):
        raise ValueError(f"data identity mismatch: {identities}")
    if stream.get("roster_exact") is not True or any(
        row.get("status") != "PASS" for row in stream.get("streams", [])
    ):
        raise ValueError("token stream audit is not roster-complete")
    if corpus_audit.get("validation_intersection") != 0:
        raise ValueError("tokenizer corpus intersects validation IDs")

    evidence_paths = [
        release_path,
        token_stream_audit_path,
        corpus_manifest_path,
        corpus_audit_path,
    ]
    evidence = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in evidence_paths
    ]
    identity = {
        "dataset_id": dataset_id,
        "snapshot_id": snapshot_id,
        "token_streams": sorted(row["name"] for row in stream["streams"]),
        "tokenizer_corpus_name": corpus.get("name"),
        "tokenizer_corpus_totals": corpus.get("totals"),
        "evidence": evidence,
    }
    return {
        "schema_version": 1,
        "status": "PASS",
        "gate": "pretrain_data_ready",
        "gate_id": f"yuformer-pretrain-data-{canonical_sha256(identity)[:20]}",
        "issued_at": datetime.now(timezone.utc).isoformat(),
        **identity,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--token-stream-audit", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--corpus-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    certificate = build_certificate(
        release_path=args.release,
        token_stream_audit_path=args.token_stream_audit,
        corpus_manifest_path=args.corpus_manifest,
        corpus_audit_path=args.corpus_audit,
    )
    atomic_json(args.out, certificate)
    print(json.dumps({
        "status": certificate["status"],
        "gate_id": certificate["gate_id"],
        "out": str(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
