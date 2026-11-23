#!/usr/bin/env python3
"""Independently close a formal S6 token-stream materialization.

The materializer writes hashes into each meta file.  This audit deliberately
re-reads every published byte and derives expected document/token totals from
the immutable S6 order inputs, so a partial copy or stale meta cannot pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def sha256_file(path: Path, chunk_bytes: int = 64 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def edge_sha256(path: Path, edge_bytes: int = 1 << 20) -> str:
    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(edge_bytes))
        if size > edge_bytes:
            handle.seek(max(0, size - edge_bytes))
            digest.update(handle.read(edge_bytes))
    return digest.hexdigest()


def snapshot_artifact(snapshot: dict[str, Any], path: Path) -> dict[str, Any]:
    resolved = path.resolve().as_posix()
    matches = [
        row for row in snapshot.get("artifacts", [])
        if isinstance(row, dict)
        and isinstance(row.get("path"), str)
        and (resolved == row["path"] or resolved.endswith("/" + row["path"]))
    ]
    if len(matches) != 1:
        raise ValueError(f"snapshot 中未唯一声明输入: {path}")
    return matches[0]


def selected_val_contract(path: Path, max_tokens: int) -> tuple[int, int]:
    values = pq.read_table(path, columns=["n_tokens"]).column("n_tokens").to_numpy()
    if len(values) == 0:
        return 0, 0
    keep = len(values)
    if max_tokens:
        keep = int(np.searchsorted(np.cumsum(values, dtype=np.int64), max_tokens, side="right")) or 1
    return keep, int(values[:keep].sum(dtype=np.int64))


def audit_one(
    *,
    stream_dir: Path,
    name: str,
    order_path: Path,
    expected_docs: int,
    expected_reference_tokens: int,
    snapshot: dict[str, Any],
    tokenizer_sha256: str,
    tokenizer_gate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    binary = stream_dir / f"{name}.tokens.bin"
    meta_path = stream_dir / f"{name}.meta.json"
    errors: list[str] = []
    if not binary.is_file() or not meta_path.is_file():
        return {
            "status": "FAIL",
            "name": name,
            "errors": [f"missing stream or meta: {binary}, {meta_path}"],
        }
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        dtype = np.dtype(meta.get("dtype", "<u2"))
    except Exception as exc:
        return {"status": "FAIL", "name": name, "errors": [f"invalid meta: {exc}"]}
    if dtype.str not in ("<u2", "<u4"):
        errors.append(f"unsupported dtype {dtype.str}")
    binary_stat = binary.stat()
    actual_bytes = binary_stat.st_size
    actual_sha = sha256_file(binary)
    actual_edge = edge_sha256(binary)
    artifact = snapshot_artifact(snapshot, order_path)
    actual_total = actual_bytes // dtype.itemsize if dtype.itemsize else -1
    selection = meta.get("selection") or {}
    is_v2 = int(meta.get("schema_version", 1)) >= 2
    lengths_evidence: dict[str, Any] = {}
    checks = {
        "stream_bytes_match_meta": actual_bytes == int(meta.get("bytes", -1)),
        "stream_bytes_aligned": actual_bytes % dtype.itemsize == 0,
        "total_tokens_match_bytes": int(meta.get("total_tokens", -1)) == actual_total,
        "n_docs_exact": int(meta.get("n_docs", -1)) == expected_docs,
        "full_sha256_exact": meta.get("sha256") == actual_sha,
        "edge_sha256_exact": meta.get("edge_sha256") == actual_edge,
        "tokenizer_sha256_exact": meta.get("tokenizer_sha256") == tokenizer_sha256,
        "snapshot_id_exact": meta.get("data_snapshot_id") == snapshot.get("snapshot_id"),
        "release_name_exact": meta.get("data_release_name") == snapshot.get("release_name"),
        "order_sha256_exact": meta.get("order_sha256") == artifact.get("sha256"),
        "order_bytes_exact": int(meta.get("order_bytes", -1)) == int(artifact.get("bytes", -2)),
        "eod_policy": (meta.get("document_boundary") or {}).get("policy") == "append_eod",
        "eod_count_exact": int((meta.get("document_boundary") or {}).get("count", -1)) == expected_docs,
    }
    if is_v2:
        checks.update({
            "selection_docs_exact": int(selection.get("selected_docs", -1)) == expected_docs,
            "selection_reference_tokens_exact": int(selection.get("selected_reference_tokens", -1)) == expected_reference_tokens,
            "selection_actual_tokens_exact": int(selection.get("actual_tokenizer_text_tokens", -1)) == actual_total - expected_docs,
            "selection_text_tokens_alias_exact": int(selection.get("selected_text_tokens", -1)) == actual_total - expected_docs,
            "reference_tokens_selection_only": selection.get("reference_tokens_are_selection_only") is True,
        })
    else:
        # Legacy streams were built with the same tokenizer that produced the
        # S6 n_tokens column.  Keep their historical closure auditable.
        expected_total = expected_reference_tokens + expected_docs
        checks.update({
            "legacy_stream_bytes_match_tokens": actual_bytes == expected_total * dtype.itemsize,
            "legacy_total_tokens_exact": int(meta.get("total_tokens", -1)) == expected_total,
        })
    if int(meta.get("schema_version", 1)) >= 3:
        lengths_record = meta.get("doc_lengths") or {}
        lengths_path = stream_dir / str(lengths_record.get("file", ""))
        try:
            lengths_stat = lengths_path.stat()
            lengths_bytes = lengths_stat.st_size
            lengths = np.memmap(lengths_path, dtype="<u4", mode="r")
            lengths_sum = int(lengths.sum(dtype=np.uint64))
            lengths_sha = sha256_file(lengths_path)
            lengths_edge = edge_sha256(lengths_path)
        except (OSError, ValueError):
            lengths_bytes, lengths_sum, lengths_sha, lengths_edge = -1, -1, "", ""
            lengths_stat = None
        if lengths_stat is not None:
            lengths_evidence = {
                "actual_doc_lengths_bytes": lengths_bytes,
                "actual_doc_lengths_sha256": lengths_sha,
                "actual_doc_lengths_edge_sha256": lengths_edge,
                "actual_doc_lengths_mtime_ns": lengths_stat.st_mtime_ns,
                "actual_doc_lengths_inode": lengths_stat.st_ino,
                "actual_doc_lengths_device": lengths_stat.st_dev,
            }
        checks.update({
            "doc_lengths_filename_exact": lengths_path.name == f"{name}.doc_lengths.u32",
            "doc_lengths_dtype_exact": lengths_record.get("dtype") == "<u4",
            "doc_lengths_rows_exact": int(lengths_record.get("rows", -1)) == expected_docs,
            "doc_lengths_bytes_exact": (
                lengths_bytes == expected_docs * 4
                and int(lengths_record.get("bytes", -1)) == lengths_bytes
            ),
            "doc_lengths_sum_exact": (
                lengths_sum == actual_total
                and int(lengths_record.get("sum_tokens", -1)) == actual_total
            ),
            "doc_lengths_sha256_exact": lengths_record.get("sha256") == lengths_sha,
            "doc_lengths_edge_sha256_exact": lengths_record.get("edge_sha256") == lengths_edge,
            "doc_lengths_include_eod": lengths_record.get("includes_appended_eod") is True,
            "doc_lengths_position_order": lengths_record.get("position_order") is True,
        })
    if tokenizer_gate is not None:
        checks.update({
            "tokenizer_id_exact": meta.get("tokenizer_id") == tokenizer_gate.get("tokenizer_id"),
            "tokenizer_gate_id_exact": meta.get("tokenizer_gate_id") == tokenizer_gate.get("gate_id"),
            "tokenizer_gate_sha256_exact": meta.get("tokenizer_gate_sha256") == tokenizer_gate.get("_gate_sha256"),
        })
    errors.extend(sorted(key for key, value in checks.items() if not value))
    return {
        "status": "PASS" if not errors else "FAIL",
        "name": name,
        "expected_docs": expected_docs,
        "expected_reference_tokens": expected_reference_tokens,
        "actual_total_tokens_with_eod": actual_total,
        "actual_bytes": actual_bytes,
        "actual_mtime_ns": binary_stat.st_mtime_ns,
        "actual_inode": binary_stat.st_ino,
        "actual_device": binary_stat.st_dev,
        "actual_sha256": actual_sha,
        "actual_edge_sha256": actual_edge,
        **lengths_evidence,
        "checks": checks,
        "errors": errors,
        "artifacts": {
            "stream": str(binary), "meta": str(meta_path), "order": str(order_path),
            **({"doc_lengths": str(stream_dir / meta["doc_lengths"]["file"])}
               if int(meta.get("schema_version", 1)) >= 3 and meta.get("doc_lengths") else {}),
        },
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
    parser.add_argument("--stream-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--order", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--val-max-tokens", type=int, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--s6-summary", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-gate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tokenizer_file = args.tokenizer / "tokenizer.json" if args.tokenizer.is_dir() else args.tokenizer
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    summary = json.loads(args.s6_summary.read_text(encoding="utf-8"))
    stage2 = (summary.get("curriculum") or {}).get("stage2") or {}
    tokenizer_sha = sha256_file(tokenizer_file)
    tokenizer_gate = None
    if args.tokenizer_gate:
        tokenizer_gate = json.loads(args.tokenizer_gate.read_text(encoding="utf-8"))
        tokenizer_gate["_gate_sha256"] = sha256_file(args.tokenizer_gate)
        if tokenizer_gate.get("status") != "PASS" or tokenizer_gate.get("tokenizer_sha256") != tokenizer_sha:
            raise SystemExit("tokenizer gate is not PASS or does not bind the tokenizer")
    streams = [
        audit_one(
            stream_dir=args.stream_dir,
            name=args.name,
            order_path=args.order,
            expected_docs=int(stage2["docs"]),
            expected_reference_tokens=int(stage2["tokens"]),
            snapshot=snapshot,
            tokenizer_sha256=tokenizer_sha,
            tokenizer_gate=tokenizer_gate,
        )
    ]
    for order_path in sorted(args.val_dir.glob("*.parquet")):
        docs, text_tokens = selected_val_contract(order_path, args.val_max_tokens)
        streams.append(
            audit_one(
                stream_dir=args.stream_dir,
                name=f"val_{order_path.stem}",
                order_path=order_path,
                expected_docs=docs,
                expected_reference_tokens=text_tokens,
                snapshot=snapshot,
                tokenizer_sha256=tokenizer_sha,
                tokenizer_gate=tokenizer_gate,
            )
        )
    expected_names = {args.name} | {f"val_{p.stem}" for p in args.val_dir.glob("*.parquet")}
    published_names = {p.name.removesuffix(".meta.json") for p in args.stream_dir.glob("*.meta.json")}
    roster_exact = published_names == expected_names
    status = "PASS" if roster_exact and all(row["status"] == "PASS" for row in streams) else "FAIL"
    payload = {
        "schema_version": 2,
        "audit": "formal_token_stream_release",
        "status": status,
        "dataset_id": snapshot.get("dataset_id"),
        "snapshot_id": snapshot.get("snapshot_id"),
        "release_name": snapshot.get("release_name"),
        "tokenizer_sha256": tokenizer_sha,
        "val_max_tokens_per_domain": args.val_max_tokens,
        "roster_exact": roster_exact,
        "expected_streams": sorted(expected_names),
        "published_streams": sorted(published_names),
        "streams": streams,
    }
    atomic_json(args.out, payload)
    print(json.dumps({"status": status, "out": str(args.out), "streams": len(streams)}, indent=2))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
