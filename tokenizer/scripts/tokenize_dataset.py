#!/usr/bin/env python3
"""Tokenize datasets for training.

Reads parquet/jsonl data, applies tokenizer, outputs JSONL with token_ids.
Supports streaming for large files.

Usage:
  PYTHONPATH=src python scripts/tokenize_dataset.py \
    --tokenizer tokenizers/yf-64k-bpe \
    --input data/fineweb-edu/ --output data-tokenized/fineweb-edu/ \
    --text-field text --max-files 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pyarrow.parquet as pq
from yuformer_tokenizer.trainer import load_tokenizer


def iter_parquet_records(path: Path, text_field: str):
    table = pq.read_table(str(path))
    cols = table.column_names
    field = text_field if text_field in cols else next(
        (c for c in cols if c.lower() in ("text", "content", "body")), None
    )
    if not field:
        return
    col_names = [field, "id", "source", "source_id", "url", "score", "quality_mean", "quality_score"]
    present = [c for c in col_names if c in cols]
    for batch in table.select(present).to_batches(1000):
        for row in batch.to_pylist():
            text = row.get(field)
            if isinstance(text, str) and text:
                yield row, text


def iter_jsonl_records(path: Path):
    import gzip
    import io

    suffix = path.suffix.lower()
    if suffix == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8")
    elif suffix == ".zst":
        import zstandard
        reader = zstandard.ZstdDecompressor().stream_reader(path.open("rb"))
        handle = io.TextIOWrapper(reader, encoding="utf-8")
    else:
        handle = path.open("r", encoding="utf-8")

    try:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for field in ("text", "content", "body", "raw_content"):
                text = obj.get(field)
                if isinstance(text, str) and text:
                    yield obj, text
                    break
    finally:
        handle.close()


def tokenize_file(tokenizer, input_path: Path, output_path: Path, text_field: str) -> int:
    suffix = input_path.suffix.lower()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    total_tokens = 0

    with output_path.open("w", encoding="utf-8") as out:
        if suffix == ".parquet":
            for record, text in iter_parquet_records(input_path, text_field):
                encoding = tokenizer.encode(text)
                out.write(json.dumps({
                    "source_id": str(record.get("id") or record.get("source_id") or f"{input_path.stem}:{count}"),
                    "tokens": encoding.ids,
                    "n_tokens": len(encoding.ids),
                    "source": str(record.get("source") or input_path.stem),
                }, ensure_ascii=False) + "\n")
                total_tokens += len(encoding.ids)
                count += 1
        elif suffix in (".jsonl", ".json", ".gz", ".zst"):
            for record, text in iter_jsonl_records(input_path):
                encoding = tokenizer.encode(text)
                out.write(json.dumps({
                    "source_id": str(record.get("id") or record.get("source_id") or f"{input_path.stem}:{count}"),
                    "tokens": encoding.ids,
                    "n_tokens": len(encoding.ids),
                    "source": str(record.get("source") or input_path.stem),
                }, ensure_ascii=False) + "\n")
                total_tokens += len(encoding.ids)
                count += 1

    print(f"  {input_path.name}: {count} records, {total_tokens} tokens")
    return count


def main():
    parser = argparse.ArgumentParser(description="Tokenize datasets")
    parser.add_argument("--tokenizer", required=True, help="tokenizer dir or json path")
    parser.add_argument("--input", required=True, help="input dir or file")
    parser.add_argument("--output", required=True, help="output dir")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-files", type=int, default=0, help="max files (0=all)")
    parser.add_argument("--max-tokens", type=int, default=0, help="stop after N total tokens (0=all)")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(
            f for f in input_path.rglob("*")
            if f.is_file() and f.suffix.lower() in (".parquet", ".jsonl", ".json", ".gz", ".zst")
        )

    print(f"Tokenizing {len(files)} files with {tokenizer.get_vocab_size()} vocab tokenizer")

    total_count = 0
    total_tokens = 0
    for i, f in enumerate(files):
        if args.max_files and i >= args.max_files:
            break
        if args.max_tokens and total_tokens >= args.max_tokens:
            print(f"  reached max_tokens={args.max_tokens}, stopping")
            break

        out_name = f.stem + ".tokens.jsonl"
        out_path = output_dir / out_name
        count = tokenize_file(tokenizer, f, out_path, args.text_field)
        total_count += count
        total_tokens += sum(
            len(json.loads(line)["tokens"])
            for line in out_path.open()
            if line.strip()
        )

    print(f"\nDone! {total_count} records, {total_tokens} tokens")
    meta = {"tokenizer": args.tokenizer, "records": total_count, "tokens": total_tokens}
    (output_dir / "_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
