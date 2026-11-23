from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer


def evaluate_tokenizer(
    tokenizer: Tokenizer,
    text_iterator: Iterator[str],
    *,
    max_samples: int = 10000,
) -> dict:
    total_bytes = 0
    total_tokens = 0
    total_chars = 0
    unk_count = 0
    roundtrip_ok = 0
    token_freq: Counter = Counter()
    sample_count = 0
    empty_count = 0

    unk_id = tokenizer.token_to_id("<|unk|>")

    for text in text_iterator:
        if sample_count >= max_samples:
            break
        if not text or not text.strip():
            empty_count += 1
            continue
        sample_count += 1
        total_bytes += len(text.encode("utf-8"))
        total_chars += len(text)
        encoding = tokenizer.encode(text)
        total_tokens += len(encoding.ids)
        token_freq.update(encoding.ids)
        if unk_id is not None:
            unk_count += encoding.ids.count(unk_id)
        if tokenizer.decode(encoding.ids) == text:
            roundtrip_ok += 1

    if total_tokens == 0:
        return {"error": "no tokens produced", "samples": sample_count}

    vocab_size = tokenizer.get_vocab_size()
    used_tokens = len(token_freq)

    result = {
        "samples": sample_count,
        "total_bytes": total_bytes,
        "total_chars": total_chars,
        "total_tokens": total_tokens,
        "bytes_per_token": round(total_bytes / total_tokens, 2),
        "chars_per_token": round(total_chars / total_tokens, 2),
        "tokens_per_sample": round(total_tokens / sample_count, 1) if sample_count else 0,
        "vocab_size": vocab_size,
        "vocab_utilization": round(used_tokens / vocab_size, 4),
        "used_tokens": used_tokens,
        "unk_tokens": unk_count,
        "unk_rate": round(unk_count / total_tokens, 6),
        "roundtrip_rate": round(roundtrip_ok / sample_count, 4) if sample_count else 0,
        "empty_samples": empty_count,
        "top_10_tokens": [
            {"id": tid, "token": tokenizer.decode([tid]), "count": cnt}
            for tid, cnt in token_freq.most_common(10)
        ],
    }
    return result


def evaluate_by_language(
    tokenizer: Tokenizer,
    texts_zh: Iterator[str],
    texts_en: Iterator[str],
    texts_code: Iterator[str],
    *,
    max_samples: int = 5000,
) -> dict:
    results = {}
    for name, iterator in [("chinese", texts_zh), ("english", texts_en), ("code", texts_code)]:
        results[name] = evaluate_tokenizer(tokenizer, iterator, max_samples=max_samples)
    return results


def save_report(report: dict, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
