"""自训 Unigram 64k（sentencepiece 官方 C++ 实现，同 yf-BPE 语料）。

背景（E0-Q1）：HF tokenizers 的 UnigramTrainer 在生产语料两次 Rust 级崩溃
（TryFromInt/Internal）被迫取消；本脚本用 sentencepiece 官方库补齐算法对照——
否则"BBPE vs Unigram"只有工业 tokenizer 代表（T5/Gemma），缺少同语料受控对比。
产物：tokenizers/yf-64k-unigram/（sentencepiece .model + 转换的 HF tokenizer.json
供 eval_e0_full.py 统一评测）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import sentencepiece as spm


def iter_corpus(corpus_dir: Path):
    import pyarrow.parquet as pq

    for f in sorted(corpus_dir.glob("*.parquet")):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=1024, columns=["text"]):
            for t in batch.column(0).to_pylist():
                if t and len(t) > 50:
                    yield t


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dir", default="../data/tokenizer_corpus")
    ap.add_argument("--out-dir", default="tokenizers/yf-64k-unigram")
    ap.add_argument("--vocab-size", type=int, default=65536)
    ap.add_argument("--byte-fallback", action="store_true", help="加 byte fallback（无 OOV；T5 无此机制）")
    args = ap.parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = out_dir / "yf_unigram"
    corpus_dir = Path(args.corpus_dir)

    tmp_list = out_dir / "_corpus_files.txt"
    files = sorted(str(p.absolute()) for p in corpus_dir.glob("*.parquet"))
    tmp_list.write_text("\n".join(files))

    t0 = time.time()
    spm.SentencePieceTrainer.train(
        input=",".join(files),
        model_prefix=str(model_prefix),
        vocab_size=args.vocab_size,
        model_type="unigram",
        character_coverage=0.9995,
        input_sentence_size=4_000_000,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
        byte_fallback=args.byte_fallback,
        unk_id=0,
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        num_threads=32,
    )
    dt = time.time() - t0
    model_path = out_dir / "yf_unigram.model"
    meta = {
        "stage": "e0_unigram_train",
        "vocab_size": args.vocab_size,
        "model_type": "unigram",
        "byte_fallback": args.byte_fallback,
        "corpus_files": len(files),
        "train_sec": round(dt, 1),
        "model": model_path.name,
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, sort_keys=True, indent=1))
    print(json.dumps(meta, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
