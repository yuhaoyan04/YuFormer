#!/usr/bin/env python3
"""Evaluate formal tokenizer candidates on one release-bound held-out corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer


CONTROL_TOKENS = (
    "<|endoftext|>", "<|unk|>", "<|im_start|>", "<|im_end|>",
    "<|think_start|>", "<|think_end|>",
    "<|tool_call_start|>", "<|tool_call_end|>",
)
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_NONSPACE = re.compile(r"\S+")


def sha256_file(path: Path, chunk_bytes: int = 64 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def word_units(text: str) -> int:
    """Diagnostic multilingual fertility denominator, not a linguistic tokenizer."""
    cjk = len(_CJK.findall(text))
    without_cjk = _CJK.sub(" ", text)
    return cjk + len(_NONSPACE.findall(without_cjk))


def batches(values: list[str], size: int):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def verify_corpus(manifest_path: Path) -> tuple[dict[str, Any], Path, dict[str, list[str]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "PASS":
        raise ValueError("held-out corpus manifest is not PASS")
    artifact = manifest.get("artifact") or {}
    corpus_path = Path(artifact.get("path", ""))
    if not corpus_path.is_absolute():
        # Formal manifests are normally written from the repository root.  A
        # relative path remains relative to the caller's working directory.
        corpus_path = Path.cwd() / corpus_path
    if not corpus_path.is_file():
        raise ValueError(f"held-out corpus is missing: {corpus_path}")
    if corpus_path.stat().st_size != int(artifact.get("bytes", -1)):
        raise ValueError("held-out corpus byte size differs from manifest")
    if sha256_file(corpus_path) != artifact.get("sha256"):
        raise ValueError("held-out corpus hash differs from manifest")
    table = pq.read_table(corpus_path, columns=["domain", "text", "utf8_bytes"])
    domains: dict[str, list[str]] = {}
    observed_bytes: dict[str, int] = {}
    for domain, text, declared_bytes in zip(
        table.column("domain").to_pylist(),
        table.column("text").to_pylist(),
        table.column("utf8_bytes").to_pylist(),
    ):
        actual = len(text.encode("utf-8"))
        if actual != declared_bytes:
            raise ValueError(f"UTF-8 byte count mismatch in {domain}")
        domains.setdefault(domain, []).append(text)
        observed_bytes[domain] = observed_bytes.get(domain, 0) + actual
    expected_domains = manifest.get("domains") or {}
    if set(domains) != set(expected_domains):
        raise ValueError("held-out domain roster differs from manifest")
    for domain, texts in domains.items():
        expected = int(expected_domains[domain]["utf8_bytes"])
        if observed_bytes[domain] != expected or not texts:
            raise ValueError(f"held-out domain totals differ for {domain}")
    return manifest, corpus_path, domains


def verify_candidate(name: str, tokenizer_dir: Path, audit_path: Path) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    if audit.get("status") != "PASS":
        raise ValueError(f"candidate audit is not PASS: {name}")
    expected = ((audit.get("identity") or {}).get("tokenizer_sha256"))
    actual = sha256_file(tokenizer_path)
    if actual != expected:
        raise ValueError(f"candidate hash differs from audit: {name}")
    return {
        "name": name,
        "kind": "candidate",
        "path": tokenizer_path,
        "sha256": actual,
        "tokenizer_id": audit.get("tokenizer_id"),
        "artifact_audit": str(audit_path),
    }


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.int64), q))


def evaluate_one(
    spec: dict[str, Any],
    domains: dict[str, list[str]],
    *,
    batch_size: int,
    throughput_repeats: int,
    hidden_size: int,
    non_embedding_parameters: int,
) -> dict[str, Any]:
    tokenizer = Tokenizer.from_file(str(spec["path"]))
    vocab_size = tokenizer.get_vocab_size()
    unk_id = tokenizer.token_to_id("<|unk|>")
    per_domain: dict[str, dict[str, Any]] = {}
    all_texts: list[str] = []
    for domain in sorted(domains):
        texts = domains[domain]
        all_texts.extend(texts)
        token_lengths: list[int] = []
        tokens = unknown = roundtrip_failures = units = chars = raw_bytes = 0
        control_literal_segments = 0
        for text_batch in batches(texts, batch_size):
            encodings = tokenizer.encode_batch(text_batch, add_special_tokens=False)
            for text, encoding in zip(text_batch, encodings):
                length = len(encoding.ids)
                token_lengths.append(length)
                tokens += length
                raw_bytes += len(text.encode("utf-8"))
                chars += len(text)
                units += word_units(text)
                if unk_id is not None:
                    unknown += sum(token_id == unk_id for token_id in encoding.ids)
                decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
                roundtrip_failures += int(decoded != text)
                control_literal_segments += int(any(control in text for control in CONTROL_TOKENS))
        per_domain[domain] = {
            "segments": len(texts),
            "utf8_bytes": raw_bytes,
            "characters": chars,
            "word_units": units,
            "tokens": tokens,
            "utf8_bytes_per_token": raw_bytes / tokens,
            "tokens_per_kib": tokens * 1024 / raw_bytes,
            "characters_per_token": chars / tokens,
            "tokens_per_word_unit": tokens / max(units, 1),
            "tokens_per_segment_p50": percentile(token_lengths, 50),
            "tokens_per_segment_p95": percentile(token_lengths, 95),
            "unknown_tokens": unknown,
            "roundtrip_failures": roundtrip_failures,
            "control_literal_segments": control_literal_segments,
        }

    warmup = all_texts[: min(len(all_texts), max(batch_size, 64))]
    tokenizer.encode_batch(warmup, add_special_tokens=False)
    timings: list[float] = []
    benchmark_tokens: list[int] = []
    total_bytes = sum(len(text.encode("utf-8")) for text in all_texts)
    for _ in range(throughput_repeats):
        count = 0
        started = time.perf_counter()
        for text_batch in batches(all_texts, batch_size):
            count += sum(len(item.ids) for item in tokenizer.encode_batch(text_batch, add_special_tokens=False))
        timings.append(time.perf_counter() - started)
        benchmark_tokens.append(count)
    if len(set(benchmark_tokens)) != 1:
        raise ValueError(f"non-deterministic token count for {spec['name']}")

    domain_values = list(per_domain.values())
    overall_tokens = sum(item["tokens"] for item in domain_values)
    overall_bytes = sum(item["utf8_bytes"] for item in domain_values)
    exact_roundtrip = sum(item["roundtrip_failures"] for item in domain_values) == 0
    checks = {
        "all_domains_present": len(per_domain) == 9,
        "no_unknown_tokens": sum(item["unknown_tokens"] for item in domain_values) == 0,
        "exact_roundtrip": exact_roundtrip,
        # External baselines may intentionally apply Unicode normalization.
        # Record that behavior, but only YuFormer candidates are release-gated
        # on byte-exact reversibility.
        "roundtrip_policy_satisfied": exact_roundtrip or spec["kind"] == "baseline",
        "benchmark_token_count_exact": benchmark_tokens[0] == overall_tokens,
        "positive_compression": overall_tokens > 0 and overall_bytes > 0,
    }
    gate_keys = (
        "all_domains_present", "no_unknown_tokens", "roundtrip_policy_satisfied",
        "benchmark_token_count_exact", "positive_compression",
    )
    total_parameters = non_embedding_parameters + vocab_size * hidden_size
    bytes_per_token = overall_bytes / overall_tokens
    return {
        **{key: value for key, value in spec.items() if key != "path"},
        "path": str(spec["path"]),
        "vocab_size": vocab_size,
        "status": "PASS" if all(checks[key] for key in gate_keys) else "FAIL",
        "checks": checks,
        "per_domain": per_domain,
        "aggregate": {
            "utf8_bytes": overall_bytes,
            "tokens": overall_tokens,
            "utf8_bytes_per_token_micro": overall_bytes / overall_tokens,
            "utf8_bytes_per_token_domain_macro": statistics.fmean(
                item["utf8_bytes_per_token"] for item in domain_values
            ),
            "tokens_per_kib_micro": overall_tokens * 1024 / overall_bytes,
            "throughput_mb_s_median": total_bytes / statistics.median(timings) / 1e6,
            "throughput_mb_s_repeats": [total_bytes / seconds / 1e6 for seconds in timings],
            "roundtrip_failures": sum(item["roundtrip_failures"] for item in domain_values),
            "unknown_tokens": sum(item["unknown_tokens"] for item in domain_values),
        },
        "parameter_economics": {
            "hidden_size": hidden_size,
            "non_embedding_parameters": non_embedding_parameters,
            "tied_embedding_parameters": vocab_size * hidden_size,
            "tied_embedding_parameters_millions": vocab_size * hidden_size / 1e6,
            "total_parameters": total_parameters,
            "output_projection_flops_per_token": 2 * vocab_size * hidden_size,
            "dense_training_flops_per_token_proxy": 6 * total_parameters,
            "dense_training_flops_per_utf8_byte_proxy": 6 * total_parameters / bytes_per_token,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", default=["yf-32k-bpe", "yf-64k-bpe", "yf-128k-bpe"])
    parser.add_argument("--baselines", nargs="*", type=Path, default=[])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--throughput-repeats", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument(
        "--non-embedding-parameters", type=int, default=405_869_568,
        help="Exact pretrain-0.5b non-embedding parameter count",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.throughput_repeats < 1:
        raise SystemExit("batch size and throughput repeats must be positive")

    corpus_manifest, corpus_path, domains = verify_corpus(args.corpus_manifest)
    specs = [
        verify_candidate(
            name,
            args.candidate_root / name,
            args.audit_root / f"{name}.audit.json",
        )
        for name in args.candidates
    ]
    for baseline in args.baselines:
        specs.append({
            "name": baseline.stem.replace(".tokenizer", ""),
            "kind": "baseline",
            "path": baseline,
            "sha256": sha256_file(baseline),
            "tokenizer_id": None,
            "artifact_audit": None,
        })

    results = []
    for spec in specs:
        print(f"[evaluate] {spec['name']}", flush=True)
        result = evaluate_one(
            spec, domains,
            batch_size=args.batch_size,
            throughput_repeats=args.throughput_repeats,
            hidden_size=args.hidden_size,
            non_embedding_parameters=args.non_embedding_parameters,
        )
        results.append(result)
        metric = result["aggregate"]["utf8_bytes_per_token_domain_macro"]
        print(f"[done] {spec['name']} status={result['status']} bytes/token={metric:.6f}", flush=True)

    payload = {
        "schema_version": 1,
        "experiment": "E0_tokenizer_heldout",
        "status": "PASS" if all(item["status"] == "PASS" for item in results) else "FAIL",
        "corpus": {
            "eval_corpus_id": corpus_manifest.get("eval_corpus_id"),
            "manifest": str(args.corpus_manifest),
            "manifest_sha256": sha256_file(args.corpus_manifest),
            "artifact": str(corpus_path),
            "artifact_sha256": sha256_file(corpus_path),
        },
        "environment": {
            "rayon_num_threads": os.environ.get("RAYON_NUM_THREADS", ""),
            "batch_size": args.batch_size,
            "throughput_repeats": args.throughput_repeats,
        },
        "metric_notes": {
            "primary_compression": "equal-domain macro UTF-8 bytes/token; higher is better",
            "fertility": "tokens/word-unit is diagnostic because word units are language-dependent",
            "throughput": "median wall throughput after warmup; same process and corpus",
            "bpb": "reserved for model cross-entropy evaluation; tokenizer-only compression is not labeled BPB",
            "roundtrip": "byte-exact for YuFormer candidates; external baseline normalization is reported but does not gate candidates",
        },
        "evaluator_sha256": sha256_file(Path(__file__)),
        "tokenizers": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.out)
    print(json.dumps({"status": payload["status"], "out": str(args.out)}, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
