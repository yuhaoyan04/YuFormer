from __future__ import annotations

import gzip
import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def open_text(path: str | Path, mode: str = "rt"):
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".gz":
        return gzip.open(p, mode, encoding="utf-8")
    if suffix == ".zst":
        try:
            import zstandard
        except ImportError:
            raise RuntimeError("Reading .zst requires optional dependency 'zstandard'")
        raw = p.open("rb")
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        return io.TextIOWrapper(reader, encoding="utf-8")
    return p.open(mode, encoding="utf-8")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON at {path}:{line_number}")
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield obj


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n"
            )
            count += 1
    return count
