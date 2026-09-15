#!/usr/bin/env python3
"""Compare multiple tokenizers on the same eval corpus.

Produces a side-by-side comparison table and saves a report.

Usage:
  PYTHONPATH=src python scripts/compare_tokenizers.py \
    --tokenizers tokenizers/yf-32k-bpe tokenizers/yf-64k-bpe tokenizers/yf-128k-bpe \
    --eval-corpus data/wikipedia-en/ data/proof-pile-2/ \
    --output reports/tokenizer-comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from yuformer_tokenizer.trainer import load_tokenizer
from yuformer_tokenizer.evaluator import evaluate_tokenizer, save_report

from train_tokenizer import iter_corpus_text


def main():
    parser = argparse.ArgumentParser(description="Compare tokenizers")
    parser.add_argument("--tokenizers", nargs="+", required=True)
    parser.add_argument("--eval-corpus", nargs="+", required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--max-samples", type=int, default=10000)
    parser.add_argument("--output", default="reports/tokenizer-comparison.json")
    args = parser.parse_args()

    print(f"Comparing {len(args.tokenizers)} tokenizers on {len(args.eval_corpus)} corpus paths")
    print(f"Max samples per tokenizer: {args.max_samples}\n")

    comparison = {}
    for tk_path in args.tokenizers:
        name = Path(tk_path).name
        print(f"--- {name} ---")
        tokenizer = load_tokenizer(tk_path)
        text_iter = iter_corpus_text(args.eval_corpus, args.text_field)
        report = evaluate_tokenizer(tokenizer, text_iter, max_samples=args.max_samples)
        comparison[name] = report
        print(f"  vocab={report['vocab_size']}  bytes/token={report['bytes_per_token']}  "
              f"util={report['vocab_utilization']}  tokens/sample={report['tokens_per_sample']}\n")

    print(f"\n{'='*80}")
    print(f"{'Tokenizer':<30s} {'Vocab':>8s} {'B/Tok':>8s} {'C/Tok':>8s} {'Util':>8s} {'Tok/Smp':>8s}")
    print(f"{'-'*80}")
    for name, r in comparison.items():
        print(f"{name:<30s} {r['vocab_size']:>8d} {r['bytes_per_token']:>8.2f} "
              f"{r['chars_per_token']:>8.2f} {r['vocab_utilization']:>8.4f} "
              f"{r['tokens_per_sample']:>8.1f}")
    print(f"{'='*80}")

    save_report(comparison, args.output)
    print(f"\nReport saved: {args.output}")


if __name__ == "__main__":
    main()
