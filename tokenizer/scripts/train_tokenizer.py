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
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pyarrow.parquet as pq
import pyarrow
import tokenizers
from yuformer_tokenizer.trainer import train_from_iterator, save_tokenizer
from yuformer_tokenizer.special_tokens import DEFAULT_SPECIAL_TOKENS, MINIMAL_SPECIAL_TOKENS

import yaml


def sha256_file(path: Path, chunk_bytes: int = 64 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def corpus_identity(corpus_paths: list[str], *, formal: bool) -> list[dict]:
    records = []
    for raw in corpus_paths:
        path = Path(raw)
        manifest = path / "manifest.json" if path.is_dir() else None
        if manifest is None or not manifest.is_file():
            if formal:
                raise SystemExit(f"formal tokenizer 语料必须带 manifest.json: {path}")
            records.append({"path": str(path), "manifest": None, "status": "UNBOUND"})
            continue
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        for artifact in payload.get("files", []):
            file_path = path / artifact["path"]
            if not file_path.is_file():
                raise SystemExit(f"tokenizer corpus 缺少 manifest 工件: {file_path}")
            if file_path.stat().st_size != int(artifact.get("bytes", -1)):
                raise SystemExit(f"tokenizer corpus 字节数不符: {file_path}")
            if sha256_file(file_path) != artifact.get("sha256"):
                raise SystemExit(f"tokenizer corpus SHA-256 不符: {file_path}")
        records.append(
            {
                "path": str(path),
                "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest),
                "snapshot_id": payload.get("snapshot_id"),
                "dataset_id": payload.get("dataset_id"),
                "status": payload.get("status"),
                "totals": payload.get("totals"),
            }
        )
    return records


def publish_tokenizer_directory(staging: Path, destination: Path) -> None:
    """Publish files on root-squash NFS with train_meta.json as commit marker."""
    marker = staging / "train_meta.json"
    if not marker.is_file():
        raise ValueError(f"tokenizer staging lacks commit marker: {staging}")
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise FileExistsError(f"tokenizer destination is not empty: {destination}")
    moved: list[Path] = []
    try:
        for source in sorted(p for p in staging.iterdir() if p.name != marker.name):
            if not source.is_file():
                raise ValueError(f"unexpected non-file in tokenizer staging: {source}")
            target = destination / source.name
            os.replace(source, target)
            moved.append(target)
        os.replace(marker, destination / marker.name)
        staging.rmdir()
    except BaseException:
        for target in reversed(moved):
            if target.exists():
                os.replace(target, staging / target.name)
        try:
            destination.rmdir()
        except OSError:
            pass
        raise


def iter_parquet_text(path: Path, text_field: str = "text"):
    # 2026-09-25: pq.read_table 整文件入内存——26GB 样本文件直接 cgroup OOM（SIGKILL 实录）。
    # 改为 iter_batches 流式读取（内存恒定为 batch 大小）。
    pf = pq.ParquetFile(str(path))
    names = pf.schema_arrow.names
    col = text_field if text_field in names else next(
        (c for c in names if c.lower() in ("text", "content", "body", "raw_content")), None
    )
    if col is None:
        return
    for batch in pf.iter_batches(batch_size=4096, columns=[col]):
        for val in batch.column(0):
            if val and val.as_py():
                yield val.as_py()


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
                return {**(data.get("defaults") or {}), **t}
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
    parser.add_argument("--iterator-batch-size", type=int, default=1024,
                        help="documents per Python-to-Rust iterator batch")
    parser.add_argument("--formal", action="store_true",
                        help="require and fully hash-verify corpus manifest artifacts")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"输出路径已存在，拒绝覆盖 tokenizer 工件: {output}")
    corpus_records = corpus_identity(args.corpus, formal=args.formal)

    script_dir = Path(__file__).resolve().parent.parent
    config_path = script_dir / "configs" / "tokenizer_configs.yaml"

    cfg = {}
    if args.config:
        cfg = load_config(config_path, args.config)
        args.vocab_size = cfg.get("vocab_size", args.vocab_size)
        args.model = cfg.get("model", args.model)
        args.iterator_batch_size = cfg.get("iterator_batch_size", args.iterator_batch_size)
    if args.iterator_batch_size <= 0:
        raise SystemExit("--iterator-batch-size 必须为正数")

    special_tokens = cfg.get("special_tokens")
    if special_tokens is None:
        special_tokens = DEFAULT_SPECIAL_TOKENS

    source_paths = [
        Path(__file__).resolve(),
        script_dir / "src" / "yuformer_tokenizer" / "trainer.py",
        script_dir / "src" / "yuformer_tokenizer" / "special_tokens.py",
    ]
    source_identity = [file_identity(path) for path in source_paths]
    config_identity = file_identity(config_path)
    training_contract = {
        "model": args.model,
        "vocab_size": args.vocab_size,
        "special_tokens": special_tokens,
        "text_field": args.text_field,
        "max_samples": args.max_samples,
        "iterator_batch_size": args.iterator_batch_size,
        "formal": args.formal,
        "config_name": args.config,
        "config": config_identity,
        "sources": source_identity,
        "corpus": corpus_records,
        "environment": {
            "python": platform.python_version(),
            "tokenizers": tokenizers.__version__,
            "pyarrow": pyarrow.__version__,
            "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS"),
            "malloc_arena_max": os.environ.get("MALLOC_ARENA_MAX"),
        },
    }
    run_identity_sha256 = canonical_sha256(training_contract)

    print(f"Training tokenizer:")
    print(f"  model: {args.model}")
    print(f"  vocab_size: {args.vocab_size}")
    print(f"  special_tokens: {len(special_tokens)} tokens")
    print(f"  corpus: {args.corpus}")
    print(f"  output: {args.output}")

    observed = {"samples": 0, "utf8_bytes": 0}

    def text_gen():
        batch: list[str] = []
        for text in iter_corpus_text(args.corpus, args.text_field):
            if args.max_samples and observed["samples"] >= args.max_samples:
                break
            observed["samples"] += 1
            observed["utf8_bytes"] += len(text.encode("utf-8"))
            if observed["samples"] % 100_000 == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    f"  corpus progress: {observed['samples']:,} docs, "
                    f"{observed['utf8_bytes'] / (1 << 30):.2f} GiB UTF-8, "
                    f"{observed['samples'] / elapsed:,.0f} docs/s",
                    flush=True,
                )
            batch.append(text)
            if len(batch) >= args.iterator_batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
        print(f"  corpus samples: {observed['samples']}")

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.monotonic()
    tokenizer = train_from_iterator(
        text_gen(),
        model_type=args.model,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
    )

    actual_vocab = tokenizer.get_vocab_size()
    if actual_vocab != args.vocab_size:
        raise SystemExit(
            f"FATAL: trained vocab_size={actual_vocab} != requested {args.vocab_size}. "
            "ModelConfig requires an exact match (vocab_size anchors tied embeddings). "
            "Corpus too small to reach target — provide more/broader corpus."
        )
    missing_special = [token for token in special_tokens if tokenizer.token_to_id(token) is None]
    if missing_special:
        raise SystemExit(f"FATAL: missing special tokens: {missing_special}")

    staging = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    shutil.rmtree(staging, ignore_errors=True)
    try:
        save_tokenizer(tokenizer, staging)
        tokenizer_path = staging / "tokenizer.json"
        if [file_identity(path) for path in source_paths] != source_identity:
            raise RuntimeError("tokenizer training source changed while the run was active")
        if file_identity(config_path) != config_identity:
            raise RuntimeError("tokenizer config changed while the run was active")
        upstream_pass = bool(corpus_records) and all(row.get("status") == "PASS" for row in corpus_records)
        complete_corpus = args.max_samples == 0
        train_meta = {
            "schema_version": 1,
            "status": "PASS" if args.formal and upstream_pass and complete_corpus else "CANDIDATE",
            "corpus_artifacts_verified": args.formal,
            "complete_corpus": complete_corpus,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": round(time.monotonic() - started, 3),
            "model": args.model,
            "requested_vocab_size": args.vocab_size,
            "actual_vocab_size": actual_vocab,
            "config_name": args.config,
            "config_path": str(config_path),
            "config_sha256": config_identity["sha256"],
            "run_identity_sha256": run_identity_sha256,
            "training_contract": training_contract,
            "special_tokens": [
                {"token": token, "id": tokenizer.token_to_id(token)} for token in special_tokens
            ],
            "corpus": corpus_records,
            "observed_corpus": observed,
            "max_samples": args.max_samples,
            "iterator_batch_size": args.iterator_batch_size,
            "text_field": args.text_field,
            "artifacts": {
                "tokenizer.json": {
                    "bytes": tokenizer_path.stat().st_size,
                    "sha256": sha256_file(tokenizer_path),
                },
                "tokenizer_config.json": {
                    "bytes": (staging / "tokenizer_config.json").stat().st_size,
                    "sha256": sha256_file(staging / "tokenizer_config.json"),
                },
                "special_tokens_map.json": {
                    "bytes": (staging / "special_tokens_map.json").stat().st_size,
                    "sha256": sha256_file(staging / "special_tokens_map.json"),
                },
            },
            "environment": {
                "python": platform.python_version(),
                "tokenizers": tokenizers.__version__,
                "pyarrow": pyarrow.__version__,
            },
        }
        (staging / "train_meta.json").write_text(
            json.dumps(train_meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        publish_tokenizer_directory(staging, output)
    except BaseException:
        # Preserve a fully trained and hashed generation if NFS publication
        # fails. Incomplete generations have no train_meta commit marker.
        if not (staging / "train_meta.json").is_file():
            shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"\nDone! vocab_size={actual_vocab}, saved to {args.output}")


if __name__ == "__main__":
    main()
