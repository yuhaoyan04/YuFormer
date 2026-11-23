from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from yuformer_tokenizer.special_tokens import (
    DEFAULT_SPECIAL_TOKENS,
    THINK_END,
    THINK_START,
    TOOL_CALL_END,
    TOOL_CALL_START,
)
from yuformer_tokenizer.trainer import create_tokenizer, train_from_iterator, save_tokenizer, load_tokenizer
from yuformer_tokenizer.evaluator import evaluate_tokenizer
import tempfile
import json
import os


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
    assert tokenizer.encode(TOOL_CALL_START).ids == [vocab[TOOL_CALL_START]]
    assert tokenizer.encode(TOOL_CALL_END).ids == [vocab[TOOL_CALL_END]]
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
        config = json.loads((Path(tmp) / "tokenizer_config.json").read_text())
        assert config["chat_template_sha256"]
        assert "add_generation_prompt" in config["chat_template"]
        assert json.loads((Path(tmp) / "special_tokens_map.json").read_text())["eos_token"] == "<|endoftext|>"
        loaded = load_tokenizer(tmp)
        text = "save and load test"
        ids1 = tokenizer.encode(text).ids
        ids2 = loaded.encode(text).ids
        assert ids1 == ids2, f"ids mismatch: {ids1} vs {ids2}"
    print("test_save_load: ok")


def test_formal_cli_writes_atomic_training_identity(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    parquet = corpus / "web.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq

    texts = [" ".join(f"term{i}_{j}" for j in range(80)) for i in range(200)]
    pq.write_table(pa.table({"text": texts}), parquet)
    digest = hashlib.sha256(parquet.read_bytes()).hexdigest()
    (corpus / "manifest.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "snapshot_id": "snapshot-test",
                "dataset_id": "dataset-test",
                "totals": {"docs": len(texts)},
                "files": [{"path": parquet.name, "bytes": parquet.stat().st_size, "sha256": digest}],
            }
        )
    )
    output = tmp_path / "tokenizer"
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_tokenizer.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--corpus",
            str(corpus),
            "--vocab-size",
            "400",
            "--model",
            "bpe",
            "--output",
            str(output),
            "--formal",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    meta = json.loads((output / "train_meta.json").read_text())
    assert meta["status"] == "PASS"
    assert meta["corpus_artifacts_verified"] is True
    assert meta["complete_corpus"] is True
    assert meta["run_identity_sha256"]
    assert len(meta["training_contract"]["sources"]) == 3
    assert meta["iterator_batch_size"] == 1024
    assert meta["corpus"][0]["snapshot_id"] == "snapshot-test"
    assert len(meta["special_tokens"]) == len(DEFAULT_SPECIAL_TOKENS)
    assert meta["artifacts"]["tokenizer.json"]["sha256"] == hashlib.sha256(
        (output / "tokenizer.json").read_bytes()
    ).hexdigest()


def test_publish_uses_file_moves_and_training_meta_last(tmp_path, monkeypatch):
    import train_tokenizer

    staging = tmp_path / ".candidate.tmp-1"
    output = tmp_path / "candidate"
    staging.mkdir()
    (staging / "tokenizer.json").write_text("tokenizer")
    (staging / "tokenizer_config.json").write_text("config")
    (staging / "train_meta.json").write_text("meta")
    real_replace = os.replace
    moves = []

    def checked_replace(source, target):
        source = Path(source)
        assert source.is_file(), "directory rename is not NFS-safe"
        moves.append(source.name)
        return real_replace(source, target)

    monkeypatch.setattr(train_tokenizer.os, "replace", checked_replace)
    train_tokenizer.publish_tokenizer_directory(staging, output)
    assert moves[-1] == "train_meta.json"
    assert sorted(p.name for p in output.iterdir()) == [
        "tokenizer.json", "tokenizer_config.json", "train_meta.json"
    ]
    assert not staging.exists()


def test_special_tokens_no_crosstalk():
    tokenizer = train_from_iterator(
        iter(["the user said hello " * 30, "system and tool calls " * 30]),
        model_type="bpe",
        vocab_size=1000,
        special_tokens=DEFAULT_SPECIAL_TOKENS,
    )
    vocab = tokenizer.get_vocab()
    assert "user" not in [t for t in DEFAULT_SPECIAL_TOKENS]
    encoding = tokenizer.encode("the user said")
    assert encoding.tokens[0] != "user", f"plain 'user' matched as special token: {encoding.tokens}"
    # chat template boundaries must hit special ids
    e = tokenizer.encode("<|im_start|>user\nhi<|im_end|>")
    assert e.ids[0] == vocab["<|im_start|>"] and e.ids[-1] == vocab["<|im_end|>"]
    print(f"test_special_tokens_no_crosstalk: ok ({encoding.tokens})")


def test_unicode_no_oov_roundtrip():
    tokenizer = train_from_iterator(
        iter(["中文测试文本与自然语言处理 " * 30, "emoji 🚀🔥 and text " * 30]),
        model_type="bpe",
        vocab_size=2000,
        special_tokens=DEFAULT_SPECIAL_TOKENS,
    )
    samples = [
        "Hello, World! 123",
        "中文测试：自然语言处理。",
        "emoji 🚀🔥😀",
        "rare chars: 𝔄𝔅 𠮷 ﷺ ẞ",
    ]
    unk_id = tokenizer.token_to_id("<|unk|>")
    for s in samples:
        encoding = tokenizer.encode(s)
        assert encoding.ids.count(unk_id) == 0, f"unk in {s!r}"
        assert tokenizer.decode(encoding.ids) == s, f"roundtrip failed: {s!r}"
    print("test_unicode_no_oov_roundtrip: ok")


def test_config_resolution():
    from train_tokenizer import load_config
    config_path = Path(__file__).resolve().parent.parent / "configs" / "tokenizer_configs.yaml"
    for name, model, vs in [
        ("yf-32k-bpe", "bpe", 32768),
        ("yf-64k-bpe", "bpe", 65536),
        ("yf-128k-bpe", "bpe", 131072),
        ("yf-64k-unigram", "unigram", 65536),
    ]:
        cfg = load_config(config_path, name)
        assert cfg["model"] == model and cfg["vocab_size"] == vs, (name, cfg)
    print("test_config_resolution: ok")


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
    assert "unk_rate" in report and "roundtrip_rate" in report
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
    test_special_tokens_no_crosstalk()
    test_unicode_no_oov_roundtrip()
    test_config_resolution()
    test_save_load()
    test_evaluate()
    test_chinese()
    test_compare_models()
    print("\nAll tokenizer tests passed!")
