#!/usr/bin/env python3
"""Build and verify the E4 tokenizer/materializer cross-check certificate.

The review sample supplies 27 released mix identities.  This tool writes a
minimal order parquet for the real pretrain materializer, then independently
fetches the same S3 texts and compares every emitted token ID including EOD.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from build_mix_review_samples import fetch_s3_rows, sha256_file


def _tokenizer_file(path: Path) -> Path:
    value = path.resolve()
    return value / "tokenizer.json" if value.is_dir() else value


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--review-json", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--s3-root", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--order", type=Path, required=True)
    ap.add_argument("--stream", type=Path)
    ap.add_argument("--meta", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()

    tokenizer_path = _tokenizer_file(args.tokenizer)
    for path in (args.review_json, args.manifest, args.s3_root, tokenizer_path):
        if not path.exists():
            raise SystemExit(f"missing required input: {path}")
    report = json.loads(args.review_json.read_text(encoding="utf-8"))
    identity = report.get("identity") or {}
    actual_manifest_sha = sha256_file(args.manifest)
    actual_tokenizer_sha = sha256_file(tokenizer_path)
    if identity.get("manifest_sha256") != actual_manifest_sha:
        raise SystemExit("review sample is stale: manifest SHA-256 mismatch")
    if identity.get("tokenizer_sha256") != actual_tokenizer_sha:
        raise SystemExit("review sample is stale: tokenizer SHA-256 mismatch")
    samples = report.get("samples") or []
    if not samples:
        raise SystemExit("review sample contains no documents")
    if not all(bool(row.get("token_count_match")) for row in samples):
        raise SystemExit("review sample token-count gate is not PASS")

    order = pa.table(
        {
            "position": pa.array(range(len(samples)), type=pa.int64()),
            "doc_id": pa.array([row["doc_id"] for row in samples], type=pa.string()),
            "n_tokens": pa.array([int(row["n_tokens"]) for row in samples], type=pa.uint32()),
        }
    )
    args.order.parent.mkdir(parents=True, exist_ok=True)
    order_tmp = args.order.with_name(f".{args.order.name}.tmp-{os.getpid()}")
    pq.write_table(order, order_tmp, compression="zstd")
    os.replace(order_tmp, args.order)
    if args.prepare_only:
        print(json.dumps({"status": "PREPARED", "order": str(args.order), "documents": len(samples)}))
        return 0

    if args.stream is None or args.meta is None or args.out is None:
        raise SystemExit("--stream, --meta and --out are required unless --prepare-only is set")
    for path in (args.stream, args.meta):
        if not path.is_file():
            raise SystemExit(f"missing materializer output: {path}")

    records = [{"doc_id": row["doc_id"]} for row in samples]
    fetch_s3_rows(args.s3_root.resolve(), records)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    eod_token = "<|endoftext|>"
    eod_id = tokenizer.token_to_id(eod_token)
    if eod_id is None:
        raise SystemExit(f"tokenizer has no {eod_token!r}")
    encoded = tokenizer.encode_batch(
        [row["text"] for row in records], add_special_tokens=False
    )
    expected = np.concatenate(
        [
            np.concatenate(
                [np.asarray(item.ids, dtype=np.uint32), np.asarray([eod_id], dtype=np.uint32)]
            )
            for item in encoded
        ]
    )
    meta = json.loads(args.meta.read_text(encoding="utf-8"))
    dtype = np.dtype(str(meta["dtype"]))
    actual = np.fromfile(args.stream, dtype=dtype).astype(np.uint32, copy=False)
    exact = bool(np.array_equal(expected, actual))
    meta_closed = (
        int(meta.get("n_docs") or 0) == len(samples)
        and int(meta.get("total_tokens") or 0) == len(actual)
        and int((meta.get("document_boundary") or {}).get("count") or 0) == len(samples)
        and meta.get("tokenizer_sha256") == actual_tokenizer_sha
        and meta.get("sha256") == sha256_file(args.stream)
    )
    status = "PASS" if exact and meta_closed else "FAIL"
    payload = {
        "schema_version": 1,
        "audit": "E4_tokenizer_materializer_crosscheck",
        "status": status,
        "mix_name": report.get("mix_name"),
        "documents": len(samples),
        "sample_contract": "9 domains x exact selected-mix p25/p50/p75 quality quantiles",
        "checks": {
            "s5_s6_n_tokens_vs_retokenize": f"{len(samples)}/{len(samples)}",
            "expected_total_tokens_including_eod": int(len(expected)),
            "actual_total_tokens": int(len(actual)),
            "append_eod_count": int((meta.get("document_boundary") or {}).get("count") or 0),
            "materialized_stream_exact_token_ids": exact,
            "materializer_meta_closed": meta_closed,
            "max_token_id": int(actual.max()) if len(actual) else None,
            "vocab_size": int(meta.get("vocab_size") or 0),
        },
        "identity": {
            "mix_manifest_sha256": actual_manifest_sha,
            "human_review_json_sha256": sha256_file(args.review_json),
            "sample_order_sha256": sha256_file(args.order),
            "token_stream_sha256": sha256_file(args.stream),
            "tokenizer_sha256": actual_tokenizer_sha,
        },
        "artifacts": {
            "review_json": str(args.review_json),
            "order": str(args.order),
            "stream": str(args.stream),
            "stream_meta": str(args.meta),
        },
    }
    _atomic_json(args.out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
