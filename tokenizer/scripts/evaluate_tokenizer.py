#!/usr/bin/env python3
"""Evaluate a tokenizer on eval corpus.

Metrics: bytes/token, chars/token, vocab utilization, token frequency.

Usage:
  PYTHONPATH=src python scripts/evaluate_tokenizer.py \
    --tokenizer tokenizers/yf-64k-bpe --eval-corpus data/wikipedia-en/ \
    --output reports/tokenizer-yf-64k-bpe.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from yuformer_tokenizer.trainer import load_tokenizer
from yuformer_tokenizer.evaluator import evaluate_tokenizer, save_report

from train_tokenizer import iter_corpus_text


def main():
    parser = argparse.ArgumentParser(description="Evaluate tokenizer")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--eval-corpus", nargs="+", required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--output", default=None, help="output report JSON path")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    print(f"Tokenizer: {args.tokenizer} (vocab={tokenizer.get_vocab_size()})")
    print(f"Eval corpus: {args.eval_corpus}")
    print(f"Max samples: {args.max_samples}")

    text_iter = iter_corpus_text(args.eval_corpus, args.text_field)
    report = evaluate_tokenizer(tokenizer, text_iter, max_samples=args.max_samples)

    print(f"\nResults:")
    for k, v in report.items():
        if k != "top_10_tokens":
            print(f"  {k}: {v}")

    if args.output:
        save_report(report, args.output)
        print(f"\nReport saved: {args.output}")


if __name__ == "__main__":
    main()
