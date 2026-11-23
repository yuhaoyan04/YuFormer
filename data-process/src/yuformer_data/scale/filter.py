from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from . import detectors
from .ingest import BatchWriter, shard_output_paths, write_summary
from .schema import AUDIT_SCHEMA, S1_SCHEMA

AUDIT_TEXT_HEAD_CHARS = 200
LANG_COUNTER_MAX = 100


def s1_output_paths(out_root: Path, source: str, relpath: str) -> tuple[Path, Path, Path]:
    base = out_root / "s1" / source / relpath
    audit_base = out_root / "s1_audit" / source / relpath
    return (
        base.with_name(base.name + ".parquet"),
        audit_base.with_name(audit_base.name + ".parquet"),
        base.with_name(base.name + ".summary.json"),
    )


def process_shard_s1(
    s0_path: Path,
    out_root: Path,
    source: str,
    relpath: str,
    domain: str,
) -> dict[str, Any]:
    out_path, audit_path, summary_path = s1_output_paths(out_root, source, relpath)
    if summary_path.is_file() and out_path.is_file() and audit_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["skipped"] = True
        return summary

    started = time.monotonic()
    decisions: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    langs: Counter[str] = Counter()
    in_rows = out_rows = audit_rows = 0
    skipped_steps: set[str] = set()
    keep_writer = BatchWriter(out_path, S1_SCHEMA)
    audit_writer = BatchWriter(audit_path, AUDIT_SCHEMA)
    try:
        pf = pq.ParquetFile(s0_path)
        for batch in pf.iter_batches(batch_size=4096):
            cols = batch.to_pylist()
            for row in cols:
                in_rows += 1
                text = detectors.clean_for_domain(row["text"], domain)
                verdict, lang = detectors.evaluate(text, domain, row["meta"], row["lang"] or "")
                if verdict.metrics.get("lang_skipped"):
                    skipped_steps.add("lang")
                decision = verdict.decision
                decisions[decision] += 1
                langs[lang] += 1
                for reason in verdict.reasons:
                    reasons[reason] += 1
                if decision == detectors.DECISION_DROP:
                    audit_writer.add(
                        {
                            "doc_id": row["doc_id"],
                            "reasons": verdict.reasons,
                            "text_head": text[:AUDIT_TEXT_HEAD_CHARS],
                            "source": source,
                        }
                    )
                    audit_rows += 1
                    continue
                keep_writer.add(
                    {
                        "doc_id": row["doc_id"],
                        "text": text,
                        "domain": row["domain"],
                        "source": row["source"],
                        "lang": lang,
                        "upstream_score": row["upstream_score"],
                        "meta": row["meta"],
                        "decision": decision,
                        "reasons": verdict.reasons,
                        "metrics": json.dumps(
                            {k: v for k, v in sorted(verdict.metrics.items())},
                            sort_keys=True,
                        ),
                    }
                )
                out_rows += 1
        out_bytes = keep_writer.close()
        audit_bytes = audit_writer.close()
    except Exception as exc:
        keep_writer.abort()
        audit_writer.abort()
        elapsed = time.monotonic() - started
        summary = {
            "stage": "s1",
            "source": source,
            "relpath": relpath,
            "in_rows": in_rows,
            "out_rows": out_rows,
            "audit_rows": audit_rows,
            "output_bytes": 0,
            "audit_bytes": 0,
            "decisions": dict(decisions),
            "reasons": dict(reasons),
            "skipped_steps": sorted(skipped_steps),
            "elapsed_sec": round(elapsed, 3),
            "errors": {f"shard_error:{type(exc).__name__}": 1},
            "skipped": False,
        }
        write_summary(summary_path, summary)
        return summary

    elapsed = time.monotonic() - started
    summary = {
        "stage": "s1",
        "source": source,
        "relpath": relpath,
        "in_rows": in_rows,
        "out_rows": out_rows,
        "audit_rows": audit_rows,
        "output_bytes": out_bytes,
        "audit_bytes": audit_bytes,
        "decisions": dict(decisions),
        "reasons": dict(reasons),
        "langs": dict(langs.most_common(LANG_COUNTER_MAX)),
        "skipped_steps": sorted(skipped_steps),
        "elapsed_sec": round(elapsed, 3),
        "rows_per_sec": round(in_rows / max(elapsed, 1e-9), 1),
        "errors": {},
        "skipped": False,
    }
    write_summary(summary_path, summary)
    return summary
