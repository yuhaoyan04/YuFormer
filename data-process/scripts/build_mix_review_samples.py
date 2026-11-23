#!/usr/bin/env python3
"""Build deterministic, reviewer-ready samples from a published S6 mix.

The sample contract is deliberately tied to the released mix rather than to an
arbitrary S3 shard: for every domain, select the document at each requested
quality-score quantile, resolve its S3 text through the immutable doc index,
and independently re-tokenize it with ``add_special_tokens=False``.

Outputs:
  * JSON: machine-readable identities, checks, and blank human verdict fields.
  * Markdown: compact redacted excerpts for a reviewer to mark PASS/REJECT.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
IDCARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
MOJIBAKE_RE = re.compile(r"Ã[\x80-\xbf©®°±²³µ¶·¸¹º¼½¾¿]|â€|ï¼|Ã©")
BOILER_RE = re.compile(
    r"cookie|javascript|sign in|log ?in|subscribe|newsletter|all rights reserved|"
    r"privacy policy|terms of (service|use)|click here|related (articles|posts)|"
    r"share (this|on)|follow us|advertisement|©",
    re.I,
)
TAG_RE = re.compile(r"<[^<>\n]{1,200}>")
SCRIPT_STYLE_RE = re.compile(r"(?is)<\s*(script|style)[\s>]")
URL_RE = re.compile(r"https?://", re.I)
CHAR_RUN_RE = re.compile(r"(.)\1{49,}", re.S)
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_quantiles(raw: str) -> list[float]:
    values = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not values or any(not 0.0 <= x <= 1.0 for x in values):
        raise ValueError("quantiles must be a non-empty comma list in [0, 1]")
    if len(set(values)) != len(values):
        raise ValueError("quantiles must be unique")
    return values


def quantile_row_indices(
    scores: np.ndarray,
    domain_counts: dict[str, int],
    quantiles: list[float],
) -> list[dict[str, Any]]:
    """Return exact nearest-rank score quantiles for contiguous domain blocks.

    S6 writes ``mix_members`` in ``(domain, doc_id)`` order.  This function is
    separated from parquet IO so the rank and tie behavior is unit-testable.
    Equal-score ties resolve to the first row, hence to the smallest doc_id.
    """
    if sum(domain_counts.values()) != len(scores):
        raise ValueError(
            f"manifest/member row mismatch: {sum(domain_counts.values())} != {len(scores)}"
        )
    selected: list[dict[str, Any]] = []
    start = 0
    for domain in sorted(domain_counts):
        count = int(domain_counts[domain])
        if count <= 0:
            raise ValueError(f"domain {domain} has no selected documents")
        block = np.asarray(scores[start : start + count], dtype=np.float32)
        ranks = [int(math.floor(q * (count - 1) + 0.5)) for q in quantiles]
        # One partition computes every requested exact order statistic in O(n).
        partitioned = np.partition(block.copy(), ranks)
        for q, rank in zip(quantiles, ranks):
            target = np.float32(partitioned[rank])
            ties = np.flatnonzero(block == target)
            if not len(ties):  # Defensive; float32 partition must preserve values.
                raise RuntimeError(f"quantile score disappeared for {domain} q={q}")
            selected.append(
                {
                    "domain": domain,
                    "quality_quantile": q,
                    "domain_rank": rank,
                    "global_row": start + int(ties[0]),
                    "score": float(target),
                    "tie_count": int(len(ties)),
                }
            )
        start += count
    return selected


def read_selected_member_rows(
    members_path: Path, selected: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    wanted = {int(r["global_row"]): r for r in selected}
    cursor = 0
    for batch in pq.ParquetFile(members_path).iter_batches(
        batch_size=262_144,
        columns=["doc_id", "domain", "composite", "n_tokens"],
    ):
        stop = cursor + batch.num_rows
        local = [i - cursor for i in sorted(wanted) if cursor <= i < stop]
        if local:
            rows = batch.take(pa.array(local, type=pa.int64())).to_pylist()
            for offset, row in zip(local, rows):
                meta = wanted[cursor + offset]
                if row["domain"] != meta["domain"]:
                    raise RuntimeError(
                        "mix_members is not in the declared (domain, doc_id) layout: "
                        f"row {cursor + offset} expected {meta['domain']}, got {row['domain']}"
                    )
                if np.float32(row["composite"]) != np.float32(meta["score"]):
                    raise RuntimeError(f"score changed while reading row {cursor + offset}")
                meta.update(row)
        cursor = stop
    missing = [r["global_row"] for r in selected if "doc_id" not in r]
    if missing:
        raise RuntimeError(f"failed to read selected member rows: {missing}")
    return selected


def s3_fingerprint(s3_root: Path, files: list[Path]) -> str:
    payload = [(str(path.relative_to(s3_root)), path.stat().st_size) for path in files]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()[:16]


def fetch_s3_rows(s3_root: Path, records: list[dict[str, Any]]) -> None:
    files = sorted(s3_root.rglob("*.parquet"))
    cache = s3_root / ".doc_index_cache"
    expected_fp = s3_fingerprint(s3_root, files)
    actual_fp = (cache / "fingerprint.txt").read_text().strip()
    if actual_fp != expected_fp:
        raise RuntimeError(f"stale S3 doc index: cache={actual_fp}, current={expected_fp}")

    ids = np.load(cache / "ids.npy", mmap_mode="r")
    fidx = np.load(cache / "fidx.npy", mmap_mode="r")
    ridx = np.load(cache / "ridx.npy", mmap_mode="r")
    query = np.frombuffer(
        b"".join(bytes.fromhex(str(r["doc_id"])) for r in records), dtype="V16"
    )
    positions = np.searchsorted(ids, query)
    valid = positions < len(ids)
    clipped = np.minimum(positions, len(ids) - 1)
    valid &= np.asarray(ids[clipped]) == query
    if not bool(valid.all()):
        missing = [records[i]["doc_id"] for i in np.flatnonzero(~valid)]
        raise RuntimeError(f"member ids missing from S3 index: {missing}")

    locations: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for sample_i, pos in enumerate(clipped):
        locations[int(fidx[pos])].append((sample_i, int(ridx[pos])))

    for file_i, requests in sorted(locations.items()):
        pf = pq.ParquetFile(files[file_i])
        bounds = np.concatenate(
            [[0], np.cumsum([pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)])]
        )
        by_rg: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for sample_i, row_i in requests:
            rg = int(np.searchsorted(bounds, row_i, side="right") - 1)
            by_rg[rg].append((sample_i, row_i))
        available = set(pf.schema_arrow.names)
        columns = [x for x in ("doc_id", "text", "source", "lang", "domain") if x in available]
        for rg, group in sorted(by_rg.items()):
            table = pf.read_row_group(rg, columns=columns)
            lo = int(bounds[rg])
            for sample_i, row_i in group:
                row = table.slice(row_i - lo, 1).to_pylist()[0]
                if row["doc_id"] != records[sample_i]["doc_id"]:
                    raise RuntimeError(
                        f"S3 index identity mismatch: {records[sample_i]['doc_id']} != {row['doc_id']}"
                    )
                records[sample_i].update(
                    {
                        "text": row.get("text") or "",
                        "source": row.get("source"),
                        "lang": row.get("lang"),
                        "s3_domain": row.get("domain"),
                        "s3_path": str(files[file_i].relative_to(s3_root)),
                        "s3_row": row_i,
                    }
                )


def text_checks(text: str) -> dict[str, Any]:
    n = max(len(text), 1)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    words = text[:20_000].split()
    duplicate_8grams = 0
    n_8grams = max(0, len(words) - 7)
    if n_8grams:
        seen: set[str] = set()
        for i in range(n_8grams):
            gram = " ".join(words[i : i + 8])
            duplicate_8grams += gram in seen
            seen.add(gram)
    tags = TAG_RE.findall(text)
    return {
        "replacement_chars": text.count("\ufffd"),
        "mojibake_hits": len(MOJIBAKE_RE.findall(text)),
        "control_chars": len(CTRL_RE.findall(text)),
        "boilerplate_hits": len(BOILER_RE.findall(text)),
        "html_tag_ratio": sum(map(len, tags)) / n,
        "html_script_style": bool(SCRIPT_STYLE_RE.search(text)),
        "url_per_1k_chars": len(URL_RE.findall(text)) / (n / 1000),
        "char_run_50": bool(CHAR_RUN_RE.search(text)),
        "line_duplicate_ratio": 1 - len(set(lines)) / len(lines) if len(lines) >= 4 else 0.0,
        "eightgram_duplicate_ratio": duplicate_8grams / n_8grams if n_8grams else 0.0,
        "pii_email_hits": len(EMAIL_RE.findall(text)),
        "pii_phone_hits": len(PHONE_RE.findall(text)),
        "pii_idcard_hits": len(IDCARD_RE.findall(text)),
    }


def automatic_flags(checks: dict[str, Any]) -> list[str]:
    tests = {
        "replacement_chars": checks["replacement_chars"] > 0,
        "mojibake": checks["mojibake_hits"] > 2,
        "control_chars": checks["control_chars"] > 0,
        "boilerplate": checks["boilerplate_hits"] >= 3,
        "html": checks["html_tag_ratio"] > 0.05 or checks["html_script_style"],
        "url_dense": checks["url_per_1k_chars"] > 0.5,
        "char_run": checks["char_run_50"],
        "line_duplicate": checks["line_duplicate_ratio"] > 0.55,
        "eightgram_duplicate": checks["eightgram_duplicate_ratio"] > 0.5,
        "pii": any(checks[k] for k in ("pii_email_hits", "pii_phone_hits", "pii_idcard_hits")),
    }
    return [name for name, hit in tests.items() if hit]


def redacted_excerpt(text: str, chars: int) -> str:
    """Return a deterministic head/middle/tail review view.

    Prefix-only views systematically hide notebook code behind metadata and
    source files behind license headers.  ``chars`` is the number of source
    characters shown; omission markers are additional display text.
    """
    if chars <= 0:
        return ""
    if len(text) <= chars:
        value = text
    else:
        head_n = (chars + 2) // 3
        middle_n = (chars + 1) // 3
        tail_n = chars - head_n - middle_n
        middle_start = (len(text) - middle_n) // 2
        tail_start = len(text) - tail_n
        windows = (
            (0, head_n),
            (middle_start, middle_start + middle_n),
            (tail_start, len(text)),
        )
        pieces: list[str] = []
        previous_end = 0
        for start, end in windows:
            if start > previous_end:
                pieces.append(f"\n\n[... {start - previous_end} source characters omitted ...]\n\n")
            pieces.append(text[start:end])
            previous_end = end
        value = "".join(pieces)
    value = EMAIL_RE.sub("[EMAIL_REDACTED]", value)
    value = PHONE_RE.sub("[PHONE_REDACTED]", value)
    value = IDCARD_RE.sub("[ID_REDACTED]", value)
    return value.replace("\r", "").strip()


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# Human review: {report['mix_name']}",
        "",
        f"- Generated: `{report['generated_date']}`",
        f"- Mix manifest SHA-256: `{report['identity']['manifest_sha256']}`",
        f"- Members SHA-256: `{report['identity']['members_sha256']}`",
        f"- Tokenizer SHA-256: `{report['identity']['tokenizer_sha256']}`",
        f"- Machine gate: **{report['machine_verdict']}** ({report['n_samples']} samples)",
        f"- Human gate: **{report.get('human_review_status', 'PENDING')}**",
        "",
        "Review each excerpt for coherence, domain fit, useful content, and visible data-quality issues. "
        "Mark `PASS` or `REJECT`; add a short reason for every rejection. PII-like strings are redacted. "
        "These are review-only excerpts: the released mix and token materializer use each full S3 text.",
        "",
        "| # | Domain | Mix quality quantile | Source | Lang | Tokens | Auto flags | Human verdict | Reason |",
        "|---:|---|---:|---|---|---:|---|---|---|",
    ]
    for i, item in enumerate(report["samples"], 1):
        flags = ", ".join(item["automatic_flags"]) or "none"
        verdict = item.get("human_verdict") or ""
        reason = str(item.get("human_reason") or "").replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {i} | {item['domain']} | p{int(item['quality_quantile'] * 100):02d} | "
            f"{item.get('source') or ''} | {item.get('lang') or ''} | {item['n_tokens']} | "
            f"{flags} | {verdict} | {reason} |"
        )
    for i, item in enumerate(report["samples"], 1):
        excerpt = item["excerpt"].replace("```", "` ` `")
        verdict = item.get("human_verdict") or ""
        pass_mark = "x" if verdict == "PASS" else " "
        reject_mark = "x" if verdict == "REJECT" else " "
        lines.extend(
            [
                "",
                f"## {i}. {item['domain']} p{int(item['quality_quantile'] * 100):02d}",
                "",
                f"`doc_id={item['doc_id']}` · `source={item.get('source')}` · "
                f"`score={item['composite']:.6f}` · `n_tokens={item['n_tokens']}` · "
                f"`retokenized={item['retokenized_n_tokens']}` · "
                f"`review_source_chars={item['excerpt_source_chars']}/{item['text_chars']}` · "
                f"`excerpt_truncated={str(item['excerpt_truncated']).lower()}`",
                "",
                "```text",
                excerpt,
                "```",
                *(
                    [
                        "",
                        f"_Review display truncated; {item['text_chars'] - item['excerpt_source_chars']} "
                        "source characters are omitted here. Training uses the full text._",
                    ]
                    if item["excerpt_truncated"]
                    else []
                ),
                "",
                f"- Verdict: `[{pass_mark}] PASS  [{reject_mark}] REJECT`",
                f"- Reason: {item.get('human_reason') or ''}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mix", default="mix_m3_e1_100b")
    ap.add_argument("--processed-root", type=Path, default=repo / "data" / "processed")
    ap.add_argument("--tokenizer", type=Path, default=repo / "tokenizer/tokenizers/yf-64k-bpe/tokenizer.json")
    ap.add_argument("--quantiles", default="0.25,0.50,0.75")
    ap.add_argument("--excerpt-chars", type=int, default=1200)
    ap.add_argument("--date", default="20261019")
    ap.add_argument("--out-json", type=Path)
    ap.add_argument("--out-md", type=Path)
    args = ap.parse_args(argv)

    processed = args.processed_root.resolve()
    tokenizer_path = args.tokenizer.resolve()
    if tokenizer_path.is_dir():
        tokenizer_path = tokenizer_path / "tokenizer.json"
    members = processed / "mix_members" / f"{args.mix}.parquet"
    manifest_path = processed / "mix_order" / f"{args.mix}.manifest.json"
    s3_root = processed / "s3"
    for path in (members, manifest_path, s3_root, tokenizer_path):
        if not path.exists():
            raise SystemExit(f"missing required input: {path}")
    out_json = args.out_json or processed / "audits" / f"human_review_{args.mix}_{args.date}.json"
    out_md = args.out_md or processed / "audits" / f"human_review_{args.mix}_{args.date}.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = {d: int(v["selected_docs"]) for d, v in manifest["selected"].items()}
    parquet = pq.ParquetFile(members)
    if parquet.metadata.num_rows != sum(counts.values()):
        raise SystemExit("manifest selected_docs does not match mix_members rows")
    print(f"pass1: read {parquet.metadata.num_rows:,} member scores", flush=True)
    score_chunks = [
        np.asarray(batch.column(0).to_numpy(zero_copy_only=False), dtype=np.float32)
        for batch in parquet.iter_batches(batch_size=4_194_304, columns=["composite"])
    ]
    scores = np.concatenate(score_chunks)
    selected = quantile_row_indices(scores, counts, parse_quantiles(args.quantiles))
    del score_chunks, scores

    print(f"pass2: resolve {len(selected)} member identities", flush=True)
    read_selected_member_rows(members, selected)
    print("pass3: fetch S3 text through immutable doc index", flush=True)
    fetch_s3_rows(s3_root, selected)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    encodings = tokenizer.encode_batch([r["text"] for r in selected], add_special_tokens=False)
    mismatches: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    for record, encoding in zip(selected, encodings):
        checks = text_checks(record["text"])
        actual = len(encoding.ids)
        excerpt = redacted_excerpt(record["text"], args.excerpt_chars)
        excerpt_source_chars = min(len(record["text"]), args.excerpt_chars)
        if actual != int(record["n_tokens"]):
            mismatches.append(
                {"doc_id": record["doc_id"], "expected": record["n_tokens"], "actual": actual}
            )
        samples.append(
            {
                "doc_id": record["doc_id"],
                "domain": record["domain"],
                "source": record.get("source"),
                "lang": record.get("lang"),
                "quality_quantile": record["quality_quantile"],
                "domain_rank": record["domain_rank"],
                "global_row": record["global_row"],
                "composite": float(record["composite"]),
                "score_tie_count": record["tie_count"],
                "n_tokens": int(record["n_tokens"]),
                "retokenized_n_tokens": actual,
                "token_count_match": actual == int(record["n_tokens"]),
                "s3_path": record["s3_path"],
                "s3_row": record["s3_row"],
                "automatic_checks": checks,
                "automatic_flags": automatic_flags(checks),
                "text_chars": len(record["text"]),
                "text_lines": record["text"].count("\n") + 1,
                "excerpt_source_chars": excerpt_source_chars,
                "excerpt_chars": len(excerpt),
                "excerpt_truncated": len(record["text"]) > args.excerpt_chars,
                "excerpt": excerpt,
                "human_verdict": None,
                "human_reason": None,
            }
        )

    report = {
        "schema_version": 1,
        "audit": "formal_mix_human_review_and_token_count_crosscheck",
        "generated_date": args.date,
        "mix_name": args.mix,
        "n_samples": len(samples),
        "sampling": {
            "unit": "released S6 mix member",
            "strata": "domain x exact nearest-rank composite quantile",
            "quantiles": parse_quantiles(args.quantiles),
            "tie_break": "smallest doc_id (mix_members is sorted by domain, doc_id)",
            "excerpt_chars": args.excerpt_chars,
        },
        "identity": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "members": str(members),
            "members_sha256": sha256_file(members),
            "upstream_snapshot_sha256": manifest.get("upstream_snapshot_sha256"),
            "tokenizer": str(tokenizer_path),
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "add_special_tokens": False,
        },
        "token_count_crosscheck": {
            "checked": len(samples),
            "matched": len(samples) - len(mismatches),
            "mismatches": mismatches,
            "status": "PASS" if not mismatches else "FAIL",
        },
        "machine_verdict": "PASS" if not mismatches else "FAIL",
        "human_review_status": "PENDING",
        "samples": samples,
    }
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(out_md, report)
    print(
        json.dumps(
            {
                "machine_verdict": report["machine_verdict"],
                "samples": len(samples),
                "token_count_matched": len(samples) - len(mismatches),
                "json": str(out_json),
                "markdown": str(out_md),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if not mismatches else 1


if __name__ == "__main__":
    sys.exit(main())
