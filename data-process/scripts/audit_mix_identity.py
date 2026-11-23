#!/usr/bin/env python3
"""Exact D4 audit using external 256-way partitions with bounded memory."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


HEX_LUT = np.full(256, 255, dtype=np.uint8)
for _index, _char in enumerate(b"0123456789abcdef"):
    HEX_LUT[_char] = _index


def _fixed_hex32_view(array) -> np.ndarray:
    if array.null_count:
        raise ValueError("doc_id contains null values")
    buffers = array.buffers()
    offset_dtype = "<i8" if pa.types.is_large_string(array.type) else "<i4"
    offsets = np.frombuffer(buffers[1], dtype=offset_dtype)
    start_index = array.offset
    selected = offsets[start_index : start_index + len(array) + 1]
    if len(array) and not np.all(np.diff(selected) == 32):
        raise ValueError("doc_id must contain exactly 32 ASCII hex characters")
    start = int(selected[0]) if len(selected) else 0
    values = np.frombuffer(buffers[2], dtype="S32", count=len(array), offset=start)
    if len(values):
        chars = values.view(np.uint8).reshape(-1, 32)
        if np.any(HEX_LUT[chars] == 255):
            raise ValueError("doc_id contains non-lowercase-hex characters")
    return values


def _partition_parquet(
    path: Path,
    out_dir: Path,
    prefix: str,
    *,
    validate_position: bool,
    batch_size: int,
) -> tuple[int, list[str]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    handles = [(out_dir / f"{prefix}-{bucket:02x}.bin").open("ab") for bucket in range(256)]
    rows = 0
    flags: list[str] = []
    expected_position = 0
    columns = ["doc_id", "position"] if validate_position else ["doc_id"]
    try:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            ids = _fixed_hex32_view(batch.column(0))
            if validate_position:
                positions = batch.column(1).to_numpy(zero_copy_only=False)
                expected = np.arange(
                    expected_position,
                    expected_position + len(positions),
                    dtype=np.int64,
                )
                if not np.array_equal(positions, expected):
                    flags.append(
                        f"{path.name}: position is not contiguous at row {expected_position}"
                    )
                expected_position += len(positions)
            if len(ids):
                chars = ids.view(np.uint8).reshape(-1, 32)
                buckets = (
                    HEX_LUT[chars[:, 0]].astype(np.uint16) * 16
                    + HEX_LUT[chars[:, 1]].astype(np.uint16)
                ).astype(np.uint8)
                order = np.argsort(buckets, kind="stable")
                sorted_buckets = buckets[order]
                boundaries = np.flatnonzero(np.diff(sorted_buckets)) + 1
                starts = np.concatenate(([0], boundaries))
                ends = np.concatenate((boundaries, [len(order)]))
                for start, end in zip(starts, ends):
                    handles[int(sorted_buckets[start])].write(ids[order[start:end]].tobytes())
            rows += len(ids)
    finally:
        for handle in handles:
            handle.close()
    return rows, flags


def audit_mix_identity(
    members: Path,
    order_tables: list[Path],
    work_root: Path,
    *,
    batch_size: int = 1_000_000,
) -> dict:
    flags: list[str] = []
    with tempfile.TemporaryDirectory(prefix="yuformer-d4-", dir=work_root) as raw_tmp:
        tmp = Path(raw_tmp)
        member_rows, member_flags = _partition_parquet(
            members, tmp, "members", validate_position=False, batch_size=batch_size
        )
        flags.extend(member_flags)
        order_rows = 0
        for index, path in enumerate(order_tables):
            rows, order_flags = _partition_parquet(
                path,
                tmp,
                f"order-{index}",
                validate_position=True,
                batch_size=batch_size,
            )
            order_rows += rows
            flags.extend(order_flags)

        duplicate_members = 0
        duplicate_order = 0
        mismatched_buckets: list[str] = []
        for bucket in range(256):
            member_ids = np.fromfile(tmp / f"members-{bucket:02x}.bin", dtype="S32")
            order_parts = [
                np.fromfile(tmp / f"order-{index}-{bucket:02x}.bin", dtype="S32")
                for index in range(len(order_tables))
            ]
            order_ids = (
                np.concatenate(order_parts) if order_parts else np.empty(0, dtype="S32")
            )
            member_ids.sort()
            order_ids.sort()
            duplicate_members += int(np.count_nonzero(member_ids[1:] == member_ids[:-1]))
            duplicate_order += int(np.count_nonzero(order_ids[1:] == order_ids[:-1]))
            if not np.array_equal(member_ids, order_ids):
                mismatched_buckets.append(f"{bucket:02x}")

    if member_rows != order_rows:
        flags.append(f"row count mismatch: members={member_rows}, orders={order_rows}")
    if duplicate_members:
        flags.append(f"members contains {duplicate_members} duplicate doc_ids")
    if duplicate_order:
        flags.append(f"order tables contain {duplicate_order} duplicate doc_ids")
    if mismatched_buckets:
        flags.append(f"member/order ID multiset mismatch in buckets {mismatched_buckets[:20]}")
    return {
        "schema_version": 1,
        "status": "PASS" if not flags else "FAIL",
        "gate": "D4",
        "members": str(members),
        "order_tables": [str(path) for path in order_tables],
        "member_rows": member_rows,
        "order_rows": order_rows,
        "duplicate_member_ids": duplicate_members,
        "duplicate_order_ids": duplicate_order,
        "mismatched_buckets": mismatched_buckets,
        "flags": flags,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", required=True, type=Path)
    parser.add_argument("--order", required=True, type=Path, action="append")
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=1_000_000)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.work_root.mkdir(parents=True, exist_ok=True)
    result = audit_mix_identity(
        args.members.resolve(),
        [path.resolve() for path in args.order],
        args.work_root.resolve(),
        batch_size=args.batch_size,
    )
    destination = args.out.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, destination)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
