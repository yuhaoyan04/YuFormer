#!/usr/bin/env python3
"""Build a release-bound, held-out raw-text corpus for tokenizer evaluation.

The validation rosters are immutable doc-id lists without text.  This command
resolves a deterministic prefix of every domain against the S3 text index and
publishes equal UTF-8 byte budgets.  The manifest is the commit marker: a
corpus without a PASS manifest is never a formal E0 input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def sha256_file(path: Path, chunk_bytes: int = 64 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def hex_to_v16(values: list[str]) -> np.ndarray:
    return np.frombuffer(b"".join(bytes.fromhex(value) for value in values), dtype="V16").reshape(-1)


def lookup_sorted_doc_ids(
    index_ids: np.ndarray,
    query_ids: np.ndarray,
    *,
    chunk_rows: int = 16_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve sorted V16 IDs with one sequential pass over the large mmap."""
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    order = np.argsort(query_ids, kind="stable")
    queries = query_ids[order]
    sorted_positions = np.full(len(queries), len(index_ids), dtype=np.int64)
    sorted_hits = np.zeros(len(queries), dtype=bool)
    cursor = 0
    for start in range(0, len(index_ids), chunk_rows):
        stop = min(start + chunk_rows, len(index_ids))
        # A view keeps searchsorted's sparse mmap access pattern, which turns
        # one production lookup into tens of minutes of 4-KiB NFS faults.
        # Copying forces one storage-local sequential read per chunk.
        chunk = np.array(index_ids[start:stop], copy=True)
        query_stop = int(np.searchsorted(queries, chunk[-1], side="right"))
        if query_stop <= cursor:
            continue
        local = queries[cursor:query_stop]
        positions = np.searchsorted(chunk, local)
        valid = positions < len(chunk)
        hits = valid & (chunk[np.minimum(positions, len(chunk) - 1)] == local)
        sorted_positions[cursor:query_stop] = start + positions
        sorted_hits[cursor:query_stop] = hits
        cursor = query_stop
        if cursor == len(queries):
            break
    positions = np.empty_like(sorted_positions)
    hits = np.empty_like(sorted_hits)
    positions[order] = sorted_positions
    hits[order] = sorted_hits
    return positions, hits


def select_prefix(doc_ids: list[str], token_counts: np.ndarray, target_tokens: int) -> int:
    if not doc_ids or len(doc_ids) != len(token_counts):
        raise ValueError("validation roster is empty or misaligned")
    if target_tokens <= 0:
        return len(doc_ids)
    cumulative = np.cumsum(token_counts, dtype=np.int64)
    return min(len(doc_ids), int(np.searchsorted(cumulative, target_tokens, side="left")) + 1)


def split_utf8(text: str, max_segment_bytes: int) -> list[tuple[str, int]]:
    """Split at valid UTF-8 boundaries without dropping or duplicating bytes."""
    if max_segment_bytes < 4:
        raise ValueError("max_segment_bytes must be at least 4")
    raw = text.encode("utf-8")
    out: list[tuple[str, int]] = []
    start = 0
    while start < len(raw):
        stop = min(start + max_segment_bytes, len(raw))
        while stop > start:
            try:
                value = raw[start:stop].decode("utf-8")
                break
            except UnicodeDecodeError:
                stop -= 1
        if stop == start:
            raise ValueError("unable to find a UTF-8 boundary")
        out.append((value, stop - start))
        start = stop
    return out


def release_artifact(release: dict[str, Any], path: Path) -> dict[str, Any]:
    resolved = path.resolve().as_posix()
    matches = [
        item for item in release.get("artifacts", [])
        if isinstance(item, dict)
        and isinstance(item.get("path"), str)
        and (resolved == item["path"] or resolved.endswith("/" + item["path"]))
    ]
    if len(matches) != 1:
        raise ValueError(f"release does not uniquely bind {path}")
    item = matches[0]
    if int(item.get("bytes", -1)) != path.stat().st_size or item.get("sha256") != sha256_file(path):
        raise ValueError(f"release artifact mismatch: {path}")
    return item


def index_files_and_fingerprint(s3_root: Path) -> tuple[list[Path], str]:
    files = sorted(s3_root.rglob("*.parquet"))
    payload = json.dumps(
        [(str(path.relative_to(s3_root)), path.stat().st_size) for path in files],
        separators=(",", ":"),
    ).encode()
    return files, hashlib.sha256(payload).hexdigest()[:16]


def row_group_bounds(path: Path) -> np.ndarray:
    metadata = pq.ParquetFile(path).metadata
    return np.concatenate([
        np.asarray([0], dtype=np.int64),
        np.cumsum([metadata.row_group(i).num_rows for i in range(metadata.num_row_groups)], dtype=np.int64),
    ])


def fetch_texts(
    files: list[Path],
    file_ids: np.ndarray,
    row_ids: np.ndarray,
    expected_doc_ids: list[str],
) -> list[str]:
    texts: list[str | None] = [None] * len(expected_doc_ids)
    by_file = np.argsort(file_ids, kind="stable")
    sorted_files = file_ids[by_file]
    cuts = np.concatenate([
        np.asarray([0], dtype=np.int64),
        np.flatnonzero(np.diff(sorted_files) != 0).astype(np.int64) + 1,
        np.asarray([len(sorted_files)], dtype=np.int64),
    ])
    for cut in range(len(cuts) - 1):
        selected = by_file[int(cuts[cut]):int(cuts[cut + 1])]
        file_id = int(file_ids[selected[0]])
        parquet = pq.ParquetFile(files[file_id])
        bounds = row_group_bounds(files[file_id])
        groups = np.searchsorted(bounds, row_ids[selected], side="right") - 1
        for group in np.unique(groups):
            local_selected = selected[groups == group]
            local_rows = np.asarray(row_ids[local_selected] - bounds[int(group)], dtype=np.int64)
            table = parquet.read_row_group(int(group), columns=["doc_id", "text"])
            subset = table.take(pa.array(local_rows, type=pa.int64()))
            got_ids = subset.column("doc_id").to_pylist()
            got_texts = subset.column("text").to_pylist()
            for position, got_id, text in zip(local_selected, got_ids, got_texts):
                expected = expected_doc_ids[int(position)]
                if got_id != expected:
                    raise ValueError(f"doc_id index mismatch: {got_id} != {expected}")
                texts[int(position)] = text or ""
    if any(text is None for text in texts):
        raise ValueError("not every selected validation document was fetched")
    return [str(text) for text in texts]


def publish_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copyfile(source, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, destination)
    except OSError:
        shutil.copyfile(temporary, destination)
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--s3-root", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--selection-reference-tokens", type=int, default=8_000_000)
    parser.add_argument("--bytes-per-domain", type=int, default=16 << 20)
    parser.add_argument("--max-segment-bytes", type=int, default=64 << 10)
    parser.add_argument("--index-chunk-rows", type=int, default=16_000_000)
    args = parser.parse_args()

    release = json.loads(args.release.read_text(encoding="utf-8"))
    if release.get("status") != "PASS":
        raise SystemExit("formal held-out corpus requires a PASS data release")
    roster_paths = sorted(path for path in args.val_dir.glob("*.parquet") if not path.stem.startswith("_"))
    if not roster_paths:
        raise SystemExit("no validation rosters found")

    records: list[dict[str, Any]] = []
    roster_artifacts: list[dict[str, Any]] = []
    domain_selection: dict[str, dict[str, int]] = {}
    for path in roster_paths:
        artifact = release_artifact(release, path)
        table = pq.read_table(path, columns=["doc_id", "domain", "n_tokens"])
        doc_ids = table.column("doc_id").to_pylist()
        domains = set(table.column("domain").to_pylist())
        if domains != {path.stem}:
            raise ValueError(f"domain mismatch in {path}: {domains}")
        # PyArrow 12 ChunkedArray.to_numpy does not expose Array's
        # zero_copy_only keyword.  combine_chunks gives one primitive Array
        # and keeps the conversion compatible across the supported versions.
        token_counts = table.column("n_tokens").combine_chunks().to_numpy(zero_copy_only=False)
        keep = select_prefix(doc_ids, token_counts, args.selection_reference_tokens)
        start = len(records)
        records.extend(
            {"domain": path.stem, "doc_id": doc_id, "reference_tokens": int(tokens)}
            for doc_id, tokens in zip(doc_ids[:keep], token_counts[:keep])
        )
        domain_selection[path.stem] = {
            "record_start": start,
            "record_stop": len(records),
            "selected_docs": keep,
            "selected_reference_tokens": int(token_counts[:keep].sum(dtype=np.int64)),
        }
        roster_artifacts.append({"path": str(path), "bytes": artifact["bytes"], "sha256": artifact["sha256"]})

    cache = args.s3_root / ".doc_index_cache"
    files, fingerprint = index_files_and_fingerprint(args.s3_root)
    recorded_fingerprint = (cache / "fingerprint.txt").read_text().strip()
    if fingerprint != recorded_fingerprint:
        raise SystemExit(f"S3 index cache is stale: {recorded_fingerprint} != {fingerprint}")
    ids = np.load(cache / "ids.npy", mmap_mode="r")
    fidx = np.load(cache / "fidx.npy", mmap_mode="r")
    ridx = np.load(cache / "ridx.npy", mmap_mode="r")
    if not (len(ids) == len(fidx) == len(ridx)):
        raise ValueError("S3 index arrays are misaligned")

    expected_ids = [record["doc_id"] for record in records]
    positions, hits = lookup_sorted_doc_ids(ids, hex_to_v16(expected_ids), chunk_rows=args.index_chunk_rows)
    if not bool(np.all(hits)):
        raise ValueError(f"{int((~hits).sum())} selected validation IDs are absent from S3")
    texts = fetch_texts(files, fidx[positions], ridx[positions], expected_ids)

    rows: dict[str, list[Any]] = {
        "domain": [], "doc_id": [], "segment_index": [], "text": [], "utf8_bytes": []
    }
    domain_stats: dict[str, dict[str, int]] = {}
    for domain, selection in domain_selection.items():
        used_bytes = 0
        used_docs = 0
        segments = 0
        for global_index in range(selection["record_start"], selection["record_stop"]):
            if used_bytes >= args.bytes_per_domain:
                break
            text = texts[global_index]
            doc_used = False
            for segment_index, (segment, n_bytes) in enumerate(split_utf8(text, args.max_segment_bytes)):
                remaining = args.bytes_per_domain - used_bytes
                if remaining <= 0:
                    break
                if n_bytes > remaining:
                    truncated = segment.encode("utf-8")[:remaining]
                    while truncated:
                        try:
                            segment = truncated.decode("utf-8")
                            break
                        except UnicodeDecodeError:
                            truncated = truncated[:-1]
                    n_bytes = len(truncated)
                if not segment or n_bytes == 0:
                    continue
                rows["domain"].append(domain)
                rows["doc_id"].append(expected_ids[global_index])
                rows["segment_index"].append(segment_index)
                rows["text"].append(segment)
                rows["utf8_bytes"].append(n_bytes)
                used_bytes += n_bytes
                segments += 1
                doc_used = True
            used_docs += int(doc_used)
        if used_bytes < args.bytes_per_domain - 3:
            raise ValueError(f"insufficient raw text for {domain}: {used_bytes} < {args.bytes_per_domain}")
        domain_stats[domain] = {
            **selection,
            "materialized_docs": used_docs,
            "segments": segments,
            "utf8_bytes": used_bytes,
        }

    table = pa.table({
        "domain": pa.array(rows["domain"], type=pa.string()),
        "doc_id": pa.array(rows["doc_id"], type=pa.string()),
        "segment_index": pa.array(rows["segment_index"], type=pa.uint32()),
        "text": pa.array(rows["text"], type=pa.large_string()),
        "utf8_bytes": pa.array(rows["utf8_bytes"], type=pa.uint32()),
    })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    staging = args.out_dir / f".heldout_text.parquet.build-{os.getpid()}"
    pq.write_table(table, staging, compression="zstd", row_group_size=1024)
    output = args.out_dir / "heldout_text.parquet"
    publish_file(staging, output)
    staging.unlink(missing_ok=True)

    release_sha = sha256_file(args.release)
    contract = {
        "schema_version": 1,
        "dataset_id": release.get("dataset_id"),
        "snapshot_id": release.get("snapshot_id"),
        "release": {"path": str(args.release), "bytes": args.release.stat().st_size, "sha256": release_sha},
        "validation_rosters": roster_artifacts,
        "selection_reference_tokens": args.selection_reference_tokens,
        "bytes_per_domain": args.bytes_per_domain,
        "max_segment_bytes": args.max_segment_bytes,
        "s3_index_fingerprint": fingerprint,
        "builder_sha256": sha256_file(Path(__file__)),
    }
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = {
        **contract,
        "name": "tokenizer_e0_heldout_v1",
        "status": "PASS",
        "eval_corpus_id": f"yuformer-tokenizer-eval-{identity[:20]}",
        "selection": "prefix of each immutable hash-ordered S6 validation roster",
        "domain_aggregation": "equal UTF-8 byte budget per domain",
        "domains": domain_stats,
        "totals": {
            "domains": len(domain_stats),
            "source_docs": sum(item["materialized_docs"] for item in domain_stats.values()),
            "segments": table.num_rows,
            "utf8_bytes": sum(item["utf8_bytes"] for item in domain_stats.values()),
        },
        "artifact": {"path": str(output), "bytes": output.stat().st_size, "sha256": sha256_file(output)},
    }
    manifest_staging = args.out_dir / f".manifest.json.build-{os.getpid()}"
    manifest_staging.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    publish_file(manifest_staging, args.out_dir / "manifest.json")
    manifest_staging.unlink(missing_ok=True)
    print(json.dumps({"status": "PASS", "eval_corpus_id": manifest["eval_corpus_id"], "totals": manifest["totals"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
