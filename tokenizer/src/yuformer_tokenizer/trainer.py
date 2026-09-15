from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import (
    ByteLevel,
    Metaspace,
    Sequence as PreTokenizerSequence,
)
from tokenizers.decoders import ByteLevel as ByteLevelDecoder, Metaspace as MetaspaceDecoder
from tokenizers.processors import TemplateProcessing

from .special_tokens import DEFAULT_SPECIAL_TOKENS, EOS_TOKEN, PAD_TOKEN


def _build_model(model_type: str, vocab_size: int, special_tokens: list[str]):
    if model_type == "bpe":
        model = BPE(unk_token="<|unk|>")
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            initial_alphabet=ByteLevel.alphabet(),
            min_frequency=2,
            show_progress=True,
        )
        return model, trainer
    if model_type == "wordpiece":
        model = WordPiece(unk_token="<|unk|>")
        trainer = WordPieceTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            show_progress=True,
        )
        return model, trainer
    if model_type == "unigram":
        model = Unigram()
        trainer = UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            unk_token="<|unk|>",
            show_progress=True,
        )
        return model, trainer
    raise ValueError(f"unknown model type: {model_type}")


def _build_pretokenizer(model_type: str):
    if model_type == "bpe":
        return ByteLevel(add_prefix_space=False, use_regex=True)
    if model_type in ("wordpiece", "unigram"):
        return Metaspace(replacement="\u2581", prepend_scheme="always")
    raise ValueError(f"unknown model type: {model_type}")


def _build_decoder(model_type: str):
    if model_type == "bpe":
        return ByteLevelDecoder()
    if model_type in ("wordpiece", "unigram"):
        return MetaspaceDecoder(replacement="\u2581", prepend_scheme="always")
    raise ValueError(f"unknown model type: {model_type}")


def create_tokenizer(
    model_type: str = "bpe",
    vocab_size: int = 65536,
    special_tokens: list[str] | None = None,
) -> tuple[Tokenizer, object]:
    if special_tokens is None:
        special_tokens = DEFAULT_SPECIAL_TOKENS

    model, trainer = _build_model(model_type, vocab_size, special_tokens)
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = _build_pretokenizer(model_type)
    tokenizer.decoder = _build_decoder(model_type)

    eos_id = special_tokens.index(EOS_TOKEN) if EOS_TOKEN in special_tokens else 0
    pad_id = special_tokens.index(PAD_TOKEN) if PAD_TOKEN in special_tokens else 0
    tokenizer.post_processor = TemplateProcessing(
        single="$A",
        special_tokens=[(EOS_TOKEN, eos_id)],
        pair="$A $B:0",
    )
    return tokenizer, trainer


def train_from_iterator(
    text_iterator: Iterator[str],
    *,
    model_type: str = "bpe",
    vocab_size: int = 65536,
    special_tokens: list[str] | None = None,
) -> Tokenizer:
    tokenizer, trainer = create_tokenizer(model_type, vocab_size, special_tokens)
    tokenizer.train_from_iterator(text_iterator, trainer=trainer)
    return tokenizer


def save_tokenizer(tokenizer: Tokenizer, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out / "tokenizer.json"))
    config = {
        "model_type": tokenizer.model.__class__.__name__,
        "vocab_size": tokenizer.get_vocab_size(),
    }
    (out / "tokenizer_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_tokenizer(path: str | Path) -> Tokenizer:
    p = Path(path)
    if p.is_dir():
        p = p / "tokenizer.json"
    return Tokenizer.from_file(str(p))
