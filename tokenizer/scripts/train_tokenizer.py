#!/usr/bin/env python3
"""Train a tokenizer from corpus files.

Supports streaming from parquet, jsonl (.gz/.zst), and plain text.
Reads text field from each file type automatically.

Usage:
  PYTHONPATH=src python scripts/train_tokenizer.py \
    --corpus data/fineweb-edu/ data/proof-pile-2/ \
    --vocab-size 65536 --model bpe --output tokenizers/yf-64k-bpe

  # Use a named config from tokenizer_configs.yaml
  PYTHONPATH=src python scripts/train_tokenizer.py --config yf-32k-bpe \
    --corpus data/fineweb-edu/ --output tokenizers/yf-32k-bpe
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pyarrow.parquet as pq
from yuformer_tokenizer.trainer import train_from_iterator, save_tokenizer
from yuformer_tokenizer.special_tokens import DEFAULT_SPECIAL_TOKENS, MINIMAL_SPECIAL_TOKENS

import yaml


def iter_parquet_text(path: Path, text_field: str = "text"):
    table = pq.read_table(str(path))
    if text_field in table.column_names:
        for val in table.column(text_field):
            if val and val.as_py():
                yield val.as_py()
    else:
        for col_name in table.column_names:
            if col_name.lower() in ("text", "content", "body", "raw_content"):
                for val in table.column(col_name):
                    if val and val.as_py():
                        yield val.as_py()
                return


def iter_jsonl_text(path: Path):
    import gzip
    import json
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
                val = obj.get(field)
                if isinstance(val, str) and val:
                    yield val
                    break
    finally:
        handle.close()


def iter_corpus_text(corpus_paths: list[str], text_field: str = "text"):
    for path_str in corpus_paths:
        p = Path(path_str)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in (".parquet", ".jsonl", ".json", ".gz", ".zst"):
                    yield from _iter_single_file(f, text_field)
        elif p.is_file():
            yield from _iter_single_file(p, text_field)


def _iter_single_file(p: Path, text_field: str):
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        yield from iter_parquet_text(p, text_field)
    elif suffix in (".jsonl", ".json"):
        yield from iter_jsonl_text(p)
    elif suffix in (".gz", ".zst"):
        yield from iter_jsonl_text(p)
    elif suffix == ".txt":
        yield from (line.strip() for line in p.open("r", encoding="utf-8") if line.strip())


def load_config(config_path: str | Path, config_name: str | None = None) -> dict:
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if config_name:
        for t in data.get("tokenizers", []):
            if t["name"] == config_name:
                return t
        raise ValueError(f"config '{config_name}' not found in {config_path}")
    return {}


def main():
    parser = argparse.ArgumentParser(description="Train YuFormer tokenizer")
    parser.add_argument("--corpus", nargs="+", required=True, help="corpus paths (dirs or files)")
    parser.add_argument("--vocab-size", type=int, default=65536)
    parser.add_argument("--model", default="bpe", choices=["bpe", "wordpiece", "unigram"])
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--config", default=None, help="named config from tokenizer_configs.yaml")
    parser.add_argument("--text-field", default="text", help="text field name in parquet")
    parser.add_argument("--max-samples", type=int, default=0, help="max training samples (0=all)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent.parent
    config_path = script_dir / "configs" / "tokenizer_configs.yaml"

    cfg = {}
    if args.config:
        cfg = load_config(config_path, args.config)
        args.vocab_size = cfg.get("vocab_size", args.vocab_size)
        args.model = cfg.get("model", args.model)

    special_tokens = cfg.get("special_tokens")
    if special_tokens is None:
        special_tokens = DEFAULT_SPECIAL_TOKENS

    print(f"Training tokenizer:")
    print(f"  model: {args.model}")
    print(f"  vocab_size: {args.vocab_size}")
    print(f"  special_tokens: {len(special_tokens)} tokens")
    print(f"  corpus: {args.corpus}")
    print(f"  output: {args.output}")

    def text_gen():
        count = 0
        for text in iter_corpus_text(args.corpus, args.text_field):
            count += 1
            if args.max_samples and count >= args.max_samples:
                break
            yield text
        print(f"  corpus samples: {count}")

    tokenizer = train_from_iterator(
        text_gen(),
        model_type=args.model,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
    )

    save_tokenizer(tokenizer, args.output)
    actual_vocab = tokenizer.get_vocab_size()
    print(f"\nDone! vocab_size={actual_vocab}, saved to {args.output}")


if __name__ == "__main__":
    main()
