from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from yuformer_tokenizer.special_tokens import DEFAULT_SPECIAL_TOKENS, THINK_START, THINK_END
from yuformer_tokenizer.trainer import create_tokenizer, train_from_iterator, save_tokenizer, load_tokenizer
from yuformer_tokenizer.evaluator import evaluate_tokenizer
import tempfile
import json


def test_train_bpe():
    corpus = [
        "def add(a, b):\n    return a + b\n",
        "The quick brown fox jumps over the lazy dog. " * 10,
        "We compute 2+2=4. Therefore the final answer is 4. " * 5,
        "这是一段中文测试文本，用来验证分词器对中文的支持。 " * 5,
    ] * 100

    tokenizer = train_from_iterator(iter(corpus), model_type="bpe", vocab_size=2000)
    vocab = tokenizer.get_vocab_size()
    assert vocab > 100, f"vocab too small: {vocab}"

    text = "def hello(): return 'world'"
    encoding = tokenizer.encode(text)
    assert len(encoding.ids) > 0
    decoded = tokenizer.decode(encoding.ids)
    assert "def" in decoded or "hello" in decoded
    print(f"test_train_bpe: ok (vocab={vocab}, tokens={len(encoding.ids)})")
    return tokenizer


def test_special_tokens():
    tokenizer = train_from_iterator(
        iter(["hello world " * 50, "test text " * 50]),
        model_type="bpe",
        vocab_size=500,
        special_tokens=DEFAULT_SPECIAL_TOKENS,
    )
    vocab = tokenizer.get_vocab(with_added_tokens=True)
    assert THINK_START in vocab, f"{THINK_START} not in vocab"
    assert THINK_END in vocab, f"{THINK_END} not in vocab"
    print(f"test_special_tokens: ok ({THINK_START} id={vocab[THINK_START]})")


def test_save_load():
    tokenizer = train_from_iterator(
        iter(["save and load test " * 50]),
        model_type="bpe",
        vocab_size=500,
    )
    with tempfile.TemporaryDirectory() as tmp:
        save_tokenizer(tokenizer, tmp)
        assert (Path(tmp) / "tokenizer.json").exists()
        loaded = load_tokenizer(tmp)
        text = "save and load test"
        ids1 = tokenizer.encode(text).ids
        ids2 = loaded.encode(text).ids
        assert ids1 == ids2, f"ids mismatch: {ids1} vs {ids2}"
    print("test_save_load: ok")


def test_evaluate():
    tokenizer = train_from_iterator(
        iter(["evaluation test text " * 50, "another document " * 50]),
        model_type="bpe",
        vocab_size=500,
    )
    report = evaluate_tokenizer(
        tokenizer,
        iter(["evaluation test text " * 5, "another document here " * 5]),
        max_samples=10,
    )
    assert "bytes_per_token" in report
    assert report["total_tokens"] > 0
    assert report["vocab_utilization"] > 0
    print(f"test_evaluate: ok (b/tok={report['bytes_per_token']}, util={report['vocab_utilization']})")


def test_chinese():
    corpus = ["这是一段中文测试文本。 " * 20, "另一个中文文档用于训练。 " * 20] * 50
    tokenizer = train_from_iterator(iter(corpus), model_type="bpe", vocab_size=1000)
    text = "这是一段中文测试文本"
    encoding = tokenizer.encode(text)
    decoded = tokenizer.decode(encoding.ids)
    assert len(encoding.ids) > 0
    print(f"test_chinese: ok (tokens={len(encoding.ids)}, decoded={decoded[:20]})")


def test_compare_models():
    corpus = ["def f(x): return x+1\n", "hello world " * 10, "math is fun " * 10] * 100

    results = {}
    for model_type in ["bpe", "wordpiece", "unigram"]:
        tokenizer = train_from_iterator(iter(corpus), model_type=model_type, vocab_size=1000)
        text = "def f(x): return x+1"
        ids = tokenizer.encode(text).ids
        results[model_type] = len(ids)

    print(f"test_compare_models: ok (bpe={results['bpe']}, wp={results['wordpiece']}, uni={results['unigram']})")


if __name__ == "__main__":
    test_train_bpe()
    test_special_tokens()
    test_save_load()
    test_evaluate()
    test_chinese()
    test_compare_models()
    print("\nAll tokenizer tests passed!")
