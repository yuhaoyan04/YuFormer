from __future__ import annotations

import hashlib

import pyarrow as pa

CANONICAL_COLUMNS: tuple[str, ...] = (
    "doc_id",
    "text",
    "domain",
    "source",
    "lang",
    "upstream_score",
    "meta",
)

CANONICAL_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("text", pa.string(), nullable=False),
        pa.field("domain", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
        pa.field("lang", pa.string(), nullable=False),
        pa.field("upstream_score", pa.float64(), nullable=True),
        pa.field("meta", pa.string(), nullable=False),
    ]
)

S1_EXTRA_COLUMNS: tuple[str, ...] = ("decision", "reasons", "metrics")

S1_SCHEMA: pa.Schema = CANONICAL_SCHEMA.append(
    pa.field("decision", pa.string(), nullable=False)
).append(pa.field("reasons", pa.list_(pa.string()), nullable=False)).append(
    pa.field("metrics", pa.string(), nullable=False)
)

AUDIT_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("reasons", pa.list_(pa.string()), nullable=False),
        pa.field("text_head", pa.string(), nullable=False),
        pa.field("source", pa.string(), nullable=False),
    ]
)

META_MAX_CHARS = 4096


def make_doc_id(source: str, relpath: str, row_idx: int) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(source.encode("utf-8"))
    h.update(b"\x00")
    h.update(relpath.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(row_idx).encode("ascii"))
    return h.hexdigest()
