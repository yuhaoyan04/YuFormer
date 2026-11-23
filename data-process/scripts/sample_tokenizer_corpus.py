#!/usr/bin/env python3
"""Build an immutable tokenizer-training corpus from one accepted S6 train order.

The old sampler read arbitrary S3 shards with an M2-era source table. That could
admit validation rows, excluded sources, or documents outside the released mix.
This implementation samples only IDs present in the snapshot-bound S6 train
order. Within each domain it takes a deterministic prefix of the global doc-id
hash order, then restores storage locality only while fetching text.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import mmap
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "pretrain" / "scripts"))
from materialize_tokens import (  # noqa: E402
    ShardTextFetcher,
    build_doc_index,
    hex_to_v16,
    lookup_sorted_doc_ids,
)


def sha256_file(path: Path, chunk_bytes: int = 64 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_artifact(snapshot: dict[str, Any], path: Path) -> dict[str, Any]:
    resolved = path.resolve().as_posix()
    matches = [
        row
        for row in snapshot.get("artifacts", [])
        if isinstance(row, dict)
        and isinstance(row.get("path"), str)
        and (resolved == row["path"] or resolved.endswith("/" + row["path"]))
    ]
    if len(matches) != 1:
        raise ValueError(f"snapshot 中未唯一声明 train order: {path}")
    record = matches[0]
    if path.stat().st_size != int(record.get("bytes", -1)):
        raise ValueError(f"train order 字节数与 snapshot 不一致: {path}")
    actual = sha256_file(path)
    if actual != record.get("sha256"):
        raise ValueError(f"train order SHA-256 与 snapshot 不一致: {path}")
    return record


def load_recipe(path: Path, total_reference_tokens: int) -> tuple[str, dict[str, int], dict[str, float]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    name = str(payload.get("name") or path.stem)
    weights = {str(k): float(v) for k, v in (payload.get("domain_weights") or {}).items()}
    if not weights or any(v <= 0 for v in weights.values()):
        raise ValueError("domain_weights 必须为非空正数映射")
    total_weight = sum(weights.values())
    if not np.isclose(total_weight, 1.0, rtol=0, atol=1e-9):
        raise ValueError(f"domain_weights 必须精确归一（当前 {total_weight}）")
    targets = {domain: int(total_reference_tokens * weight) for domain, weight in weights.items()}
    largest = sorted(weights, key=lambda key: (-weights[key], key))[0]
    targets[largest] += total_reference_tokens - sum(targets.values())
    return name, targets, weights


def select_hash_prefix(
    batches: Iterable[pa.RecordBatch], targets: dict[str, int]
) -> tuple[dict[str, list[str]], dict[str, list[int]], int]:
    """Select whole documents up to each per-domain reference-token target."""
    ids: dict[str, list[str]] = {domain: [] for domain in targets}
    token_counts: dict[str, list[int]] = {domain: [] for domain in targets}
    totals = {domain: 0 for domain in targets}
    scanned = 0
    expected_position = 0
    for batch in batches:
        names = batch.schema.names
        if "position" in names:
            positions = batch.column(names.index("position")).to_numpy(zero_copy_only=False)
            expected = np.arange(expected_position, expected_position + len(positions))
            if len(positions) and not np.array_equal(positions, expected):
                raise ValueError("train order 的 position 不是从 0 连续递增")
        expected_position += batch.num_rows
        scanned += batch.num_rows
        batch_domains = np.asarray(batch.column(names.index("domain")).to_pylist(), dtype=object)
        batch_tokens = batch.column(names.index("n_tokens")).to_numpy(zero_copy_only=False).astype(
            np.int64, copy=False
        )
        batch_ids = batch.column(names.index("doc_id")).to_pylist()
        for domain, target in targets.items():
            if totals[domain] >= target:
                continue
            domain_idx = np.flatnonzero(batch_domains == domain)
            if not len(domain_idx):
                continue
            cumulative = np.cumsum(batch_tokens[domain_idx], dtype=np.int64)
            remaining = target - totals[domain]
            keep = min(len(domain_idx), int(np.searchsorted(cumulative, remaining, side="left")) + 1)
            chosen = domain_idx[:keep]
            ids[domain].extend(batch_ids[int(i)] for i in chosen)
            values = [int(batch_tokens[int(i)]) for i in chosen]
            token_counts[domain].extend(values)
            totals[domain] += sum(values)
        if all(totals[d] >= targets[d] for d in targets):
            break
    missing = {domain: targets[domain] - totals[domain] for domain in targets if totals[domain] < targets[domain]}
    if missing:
        raise ValueError(f"train order 容量不足以满足 tokenizer corpus 配额: {missing}")
    return ids, token_counts, scanned


def write_corpus(
    *,
    selected_ids: dict[str, list[str]],
    selected_tokens: dict[str, list[int]],
    s3_root: Path,
    staging: Path,
    fetch_workers: int = 1,
) -> dict[str, dict[str, int]]:
    flat_ids: list[str] = []
    flat_domains: list[str] = []
    flat_tokens: list[int] = []
    for domain in sorted(selected_ids):
        flat_ids.extend(selected_ids[domain])
        flat_domains.extend([domain] * len(selected_ids[domain]))
        flat_tokens.extend(selected_tokens[domain])

    index_ids, index_fidx, index_ridx, files = build_doc_index(s3_root)
    positions, hits = lookup_sorted_doc_ids(index_ids, hex_to_v16(flat_ids))
    if not bool(np.all(hits)):
        raise ValueError(f"{int((~hits).sum())}/{len(hits)} 个 tokenizer 样本 ID 在 S3 中不存在")
    file_idx = np.asarray(index_fidx[positions], dtype=np.int64)
    row_idx = np.asarray(index_ridx[positions], dtype=np.int64)
    # The three production memmaps total ~155 GiB.  Sequential lookup faults
    # most pages into this process's RSS even though only the compact selected
    # file/row vectors are needed during text extraction.  Explicitly release
    # those clean pages before opening parquet writers so corpus generation can
    # coexist with other jobs on a shared node.
    for mapped in (index_ids, index_fidx, index_ridx):
        backing = getattr(mapped, "_mmap", None)
        if backing is not None and hasattr(backing, "madvise"):
            try:
                backing.madvise(mmap.MADV_DONTNEED)
            except (OSError, ValueError):
                pass
    del index_ids, index_fidx, index_ridx, positions, hits
    storage_order = np.lexsort((row_idx, file_idx))
    if fetch_workers < 1:
        raise ValueError("fetch_workers must be >= 1")
    schema = pa.schema(
        [
            pa.field("doc_id", pa.string(), nullable=False),
            pa.field("domain", pa.string(), nullable=False),
            pa.field("n_tokens_reference", pa.uint32(), nullable=False),
            pa.field("text", pa.string(), nullable=False),
        ]
    )
    writers: dict[str, pq.ParquetWriter] = {}
    stats = {domain: {"docs": 0, "reference_tokens": 0, "utf8_bytes": 0} for domain in selected_ids}
    try:
        cuts = np.concatenate(
            [
                np.asarray([0], dtype=np.int64),
                np.flatnonzero(np.diff(file_idx[storage_order]) != 0).astype(np.int64) + 1,
                np.asarray([len(storage_order)], dtype=np.int64),
            ]
        )
        n_cuts = len(cuts) - 1

        def fetch_cut(cut: int) -> tuple[np.ndarray, list[str]]:
            a, b = int(cuts[cut]), int(cuts[cut + 1])
            chosen = np.asarray(storage_order[a:b], dtype=np.int64)
            fi = int(file_idx[chosen[0]])
            # One fetcher per file task keeps its row-group cache thread-local.
            texts = ShardTextFetcher(files).fetch(fi, row_idx[chosen])
            return chosen, texts

        # Read/decompress independent parquet files concurrently, while the
        # main thread consumes futures in deterministic file order and remains
        # the sole ParquetWriter owner.  The 2x window bounds completed-text
        # memory even when an early large file is slower than later files.
        window = min(n_cuts, 2 * fetch_workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=fetch_workers) as pool:
            pending = {cut: pool.submit(fetch_cut, cut) for cut in range(window)}
            next_submit = window
            for cut in range(n_cuts):
                chosen, texts = pending.pop(cut).result()
                if next_submit < n_cuts:
                    pending[next_submit] = pool.submit(fetch_cut, next_submit)
                    next_submit += 1
                by_domain: dict[str, list[int]] = defaultdict(list)
                for local, original in enumerate(chosen):
                    by_domain[flat_domains[int(original)]].append(local)
                for domain, local_indices in by_domain.items():
                    original_indices = [int(chosen[local]) for local in local_indices]
                    domain_texts = [texts[local] for local in local_indices]
                    if domain not in writers:
                        writers[domain] = pq.ParquetWriter(
                            staging / f"{domain}.parquet", schema, compression="zstd"
                        )
                    table = pa.table(
                        {
                            "doc_id": [flat_ids[i] for i in original_indices],
                            "domain": [domain] * len(original_indices),
                            "n_tokens_reference": pa.array(
                                [flat_tokens[i] for i in original_indices], type=pa.uint32()
                            ),
                            "text": domain_texts,
                        },
                        schema=schema,
                    )
                    writers[domain].write_table(table, row_group_size=4096)
                    stats[domain]["docs"] += len(original_indices)
                    stats[domain]["reference_tokens"] += sum(flat_tokens[i] for i in original_indices)
                    stats[domain]["utf8_bytes"] += sum(len(text.encode("utf-8")) for text in domain_texts)
                if cut % 200 == 0:
                    print(f"  fetch {cut}/{n_cuts} S3 files ({fetch_workers} workers)", flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    return stats


def publish_corpus_directory(staging: Path, destination: Path) -> None:
    """Publish files on NFS without renaming a directory.

    The shared filesystem permits file-level ``os.replace`` but rejects a
    rename of a directory created under root-squash. Consumers treat
    ``manifest.json`` as the commit marker, so data files move first and the
    manifest moves last. A failed publish rolls moved files back to staging.
    """
    manifest = staging / "manifest.json"
    if not manifest.is_file():
        raise ValueError(f"staging corpus lacks manifest: {staging}")
    if destination.exists():
        if (destination / "manifest.json").is_file():
            raise FileExistsError(f"refusing to overwrite committed corpus: {destination}")
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    moved: list[Path] = []
    try:
        for source in sorted(p for p in staging.iterdir() if p.name != "manifest.json"):
            if not source.is_file():
                raise ValueError(f"unexpected non-file in corpus staging: {source}")
            target = destination / source.name
            os.replace(source, target)
            moved.append(target)
        os.replace(manifest, destination / manifest.name)
        staging.rmdir()
    except BaseException:
        for target in reversed(moved):
            if target.exists():
                try:
                    os.replace(target, staging / target.name)
                except OSError:
                    pass
        if destination.exists() and not any(destination.iterdir()):
            destination.rmdir()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=Path, required=True)
    parser.add_argument("--s3-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--total-reference-tokens", type=int, default=6_500_000_000)
    parser.add_argument("--workers", type=int, default=16,
                        help="bounded parallel S3 parquet readers (default: 16)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.total_reference_tokens <= 0:
        raise SystemExit("--total-reference-tokens 必须为正数")
    if args.workers <= 0:
        raise SystemExit("--workers 必须为正数")
    if args.out_root.exists() and any(args.out_root.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"输出目录非空，拒绝混写: {args.out_root}")
        shutil.rmtree(args.out_root)

    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    artifact = snapshot_artifact(snapshot, args.order)
    recipe_name, targets, weights = load_recipe(args.recipe, args.total_reference_tokens)
    parquet = pq.ParquetFile(args.order)
    batches = parquet.iter_batches(
        batch_size=1_000_000, columns=["position", "doc_id", "domain", "n_tokens"]
    )
    selected_ids, selected_tokens, scanned = select_hash_prefix(batches, targets)
    print(
        f"selected {sum(map(len, selected_ids.values())):,} docs after scanning {scanned:,} order rows",
        flush=True,
    )

    staging = args.out_root.with_name(f".{args.out_root.name}.tmp-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        stats = write_corpus(
            selected_ids=selected_ids,
            selected_tokens=selected_tokens,
            s3_root=args.s3_root,
            staging=staging,
            fetch_workers=args.workers,
        )
        files: list[dict[str, Any]] = []
        for path in sorted(staging.glob("*.parquet")):
            files.append(
                {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
        # Governance evidence can be signed while this long extraction is in
        # flight. Re-read the release immediately before committing, but only
        # accept a refreshed release when the immutable data snapshot and the
        # declared order artifact are unchanged.
        final_snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
        final_artifact = snapshot_artifact(final_snapshot, args.order)
        if (
            final_snapshot.get("snapshot_id") != snapshot.get("snapshot_id")
            or final_artifact.get("bytes") != artifact.get("bytes")
            or final_artifact.get("sha256") != artifact.get("sha256")
        ):
            raise RuntimeError("release data snapshot changed during corpus extraction")
        manifest = {
            "schema_version": 1,
            "status": "CANDIDATE" if final_snapshot.get("status") != "PASS" else "PASS",
            "name": recipe_name,
            "sampling": "per-domain prefix of immutable doc_id-hash train order; whole documents",
            "storage_order": "S3 file,row for extraction locality; BPE frequency counts are order-invariant",
            "extraction": {
                "reader_workers": args.workers,
                "write_order": "deterministic S3 file order; one main-thread parquet writer per domain",
            },
            "snapshot_id": final_snapshot.get("snapshot_id"),
            "dataset_id": final_snapshot.get("dataset_id"),
            "release_status_at_build": final_snapshot.get("status"),
            "order": {
                "path": str(args.order),
                "bytes": int(artifact["bytes"]),
                "sha256": artifact["sha256"],
                "rows_scanned": scanned,
            },
            "recipe": {
                "path": str(args.recipe),
                "sha256": sha256_file(args.recipe),
                "total_reference_tokens": args.total_reference_tokens,
                "domain_weights": weights,
                "target_reference_tokens": targets,
            },
            "domains": stats,
            "totals": {
                key: sum(row[key] for row in stats.values())
                for key in ("docs", "reference_tokens", "utf8_bytes")
            },
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        publish_corpus_directory(staging, args.out_root)
    except BaseException:
        # Preserve a fully built, hashed staging generation when publication
        # itself fails; operators can retry the cheap publish step without
        # repeating multi-hour extraction. Incomplete generations have no
        # manifest commit marker and are safe to discard.
        if not (staging / "manifest.json").is_file():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": manifest["status"], "out": str(args.out_root), **manifest["totals"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
