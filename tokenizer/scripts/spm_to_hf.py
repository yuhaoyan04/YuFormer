"""sentencepiece Unigram .model → HF tokenizers tokenizer.json 转换（E0-Q2 消费格式）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sentencepiece as spm


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spm", default="tokenizers/yf-64k-unigram/yf_unigram.model")
    ap.add_argument("--out", default="tokenizers/yf-64k-unigram/tokenizer.json")
    args = ap.parse_args(argv)
    sp = spm.SentencePieceProcessor(model_file=args.spm)
    pieces = []
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        score = sp.get_score(i)
        if piece.startswith("<") and piece.endswith(">") and i < 3:
            pieces.append({"type": "Control" if i in (0, 1, 2) else "Special", "content": piece, "id": i})
            continue
        pieces.append({"type": "Token", "content": piece, "id": i, "score": score})
    vocab = {"version": "1.0", "truncation": None, "padding": None, "added_tokens": []}
    vocab["model"] = {
        "type": "Unigram",
        "unk_id": 0,
        "byte_fallback": False,
        "vocab": [[sp.id_to_piece(i), sp.get_score(i)] for i in range(sp.get_piece_size())],
    }
    vocab["normalizer"] = {"type": "Sequence", "normalizers": [{"type": "Prepend", "prepend": " "}, {"type": "NFKC"}]}
    vocab["pre_tokenizer"] = {"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": False}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(vocab, ensure_ascii=False))
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(out))
    enc = tok.encode("The history of astronomy is one of the oldest natural sciences.", add_special_tokens=False)
    dec = tok.decode(enc.ids)
    assert "astronomy" in dec, f"roundtrip broken: {dec!r}"
    meta = {"stage": "spm_to_hf", "vocab_size": sp.get_piece_size(), "out": str(out), "roundtrip": "ok"}
    print(json.dumps(meta, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
