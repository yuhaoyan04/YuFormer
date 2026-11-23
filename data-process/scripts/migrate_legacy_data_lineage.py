#!/usr/bin/env python3
"""One-time, fail-closed migration for 2026-09 decontam/S4/S5 artifacts.

The production files predate per-shard input/config/content contracts.  This
tool adds those contracts only after structural checks, and explicitly refuses
to promote sources whose decisions changed.  It never rewrites parquet data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "data-process" / "src"))

from yuformer_data.scale import decontam as dct  # noqa: E402
from yuformer_data.scale import length_bucket as lb  # noqa: E402
from yuformer_data.scale.dedup_minhash import s3_output_paths  # noqa: E402
from yuformer_data.scale.profiles import load_profiles  # noqa: E402
from yuformer_data.scale import score as s4  # noqa: E402
from yuformer_data.scale.rule_score import rule_scores_batch  # noqa: E402


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1),
        encoding="utf-8",
    )
    os.replace(tmp, path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evenly_spaced(items: list[Any], n: int) -> list[Any]:
    if len(items) <= n:
        return items
    return [items[(i * len(items)) // n] for i in range(n)]


def _audit_one_token_shard(
    out_root: Path,
    source: str,
    relpath: str,
    tokenizer: Any,
    rows_per_shard: int,
) -> dict[str, Any]:
    s5_path, _ = lb.s5_output_paths(out_root, source, relpath)
    s3_path, _ = s3_output_paths(out_root, source, relpath)
    first = next(
        iter(pq.ParquetFile(s5_path).iter_batches(
            batch_size=rows_per_shard, columns=["doc_id", "n_tokens"]
        )),
        None,
    )
    if first is None or first.num_rows == 0:
        return {"source": source, "relpath": relpath, "checked": 0, "mismatches": []}
    sample = first.to_pydict()
    expected = {doc: int(nt) for doc, nt in zip(sample["doc_id"], sample["n_tokens"])}
    found: dict[str, str] = {}
    scanned = 0
    for batch in pq.ParquetFile(s3_path).iter_batches(
        # Academic/books shards contain multi-MB documents. A large row batch
        # would turn this tiny audit into multi-GB RSS despite sampling 8 rows.
        batch_size=256, columns=["doc_id", "text"]
    ):
        d = batch.to_pydict()
        scanned += len(d["doc_id"])
        for doc_id, text in zip(d["doc_id"], d["text"]):
            if doc_id in expected:
                found[doc_id] = text if isinstance(text, str) else ""
        if len(found) == len(expected):
            break
        if scanned >= 1_000_000:
            break
    missing = sorted(set(expected) - set(found))
    if missing:
        raise ValueError(
            f"sample ids not found within first {scanned} S3 rows: {source}/{relpath}"
        )
    ordered = list(expected)
    actual = [
        len(enc.ids)
        for enc in tokenizer.encode_batch([found[doc] for doc in ordered], add_special_tokens=False)
    ]
    mismatches = [
        {"doc_id": doc, "stored": expected[doc], "current": got}
        for doc, got in zip(ordered, actual)
        if expected[doc] != got
    ]
    return {
        "source": source,
        "relpath": relpath,
        "checked": len(ordered),
        "s3_rows_scanned": scanned,
        "s5_fingerprint": dct.fingerprint(s5_path),
        "mismatches": mismatches,
    }


def run_token_audit(
    out_root: Path,
    profiles: dict[str, dict[str, Any]],
    tokenizer_path: Path,
    shards_per_source: int,
    rows_per_shard: int,
) -> Path:
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    grouped: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for task in lb.s5_tasks(out_root, profiles):
        grouped[task[0]].append(task)
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for source in sorted(grouped):
        for _source, relpath, _domain in _evenly_spaced(
            grouped[source], shards_per_source
        ):
            try:
                results.append(
                    _audit_one_token_shard(
                        out_root, source, relpath, tokenizer, rows_per_shard
                    )
                )
            except Exception as exc:
                errors.append(f"{source}/{relpath}: {type(exc).__name__}: {exc}")
        source_rows = sum(
            int(row["checked"]) for row in results if row["source"] == source
        )
        print(f"token audit: {source} checked_rows={source_rows}", flush=True)
    mismatch_count = sum(len(row["mismatches"]) for row in results)
    per_source: dict[str, Counter[str]] = defaultdict(Counter)
    for row in results:
        per_source[row["source"]]["shards"] += 1
        per_source[row["source"]]["rows"] += int(row["checked"])
        per_source[row["source"]]["mismatches"] += len(row["mismatches"])
    payload = {
        "stage": "legacy_s5_token_audit",
        "schema_version": 1,
        "method": "evenly_spaced_shards_first_retained_rows_exact_retokenization",
        "tokenizer_sha256": _sha256(tokenizer_path),
        "shards_per_source": shards_per_source,
        "rows_per_shard": rows_per_shard,
        "checked_rows": sum(int(row["checked"]) for row in results),
        "mismatches": mismatch_count,
        "errors": errors,
        "passed": not errors and mismatch_count == 0,
        "per_source": {k: dict(v) for k, v in sorted(per_source.items())},
        "samples": results,
    }
    return _write_json(out_root / "audits" / "s5_legacy_token_audit.json", payload)


def run_decontam_migration(
    out_root: Path, cfg: dict[str, Any], workers: int
) -> Path:
    tasks = [
        task for task in dct.shard_tasks(out_root, cfg, None, 0)
        if not ((cfg.get("source_min_distinct_hits") or {}).get(task[1]) or {})
    ]

    def migrate(task: tuple[Path, str, str, str]) -> dict[str, Any]:
        _root, source, relpath, raw_input = task
        out_path, summary_path = dct.decontam_output_paths(out_root, source, relpath)
        try:
            result = dct.migrate_legacy_summary(
                Path(raw_input), out_path, summary_path, cfg, source
            )
            return {"source": source, "relpath": relpath, "ok": True,
                    "skipped": bool(result.get("skipped"))}
        except Exception as exc:
            return {"source": source, "relpath": relpath, "ok": False,
                    "error": f"{type(exc).__name__}: {exc}"}

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(migrate, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if i % 500 == 0:
                print(f"decontam migration: {i}/{len(tasks)}", flush=True)
    failed = [row for row in rows if not row["ok"]]
    payload = {
        "stage": "legacy_decontam_lineage_migration",
        "schema_version": 1,
        "tasks": len(tasks),
        "passed": not failed,
        "failed": len(failed),
        "errors": failed[:100],
        "per_source": dict(Counter(row["source"] for row in rows if row["ok"])),
    }
    return _write_json(out_root / "audits" / "decontam_lineage_migration.json", payload)


def _s4_tasks(out_root: Path) -> list[tuple[Path, str, str]]:
    root = out_root / "s1"
    tasks: list[tuple[Path, str, str]] = []
    for path in sorted(root.rglob("*.parquet")):
        rel = path.relative_to(root)
        relpath = Path(*rel.parts[1:]).as_posix()
        tasks.append((path, rel.parts[0], relpath[: -len(".parquet")]))
    return tasks


def _audit_one_s4_shard(
    out_root: Path,
    task: tuple[Path, str, str],
    cfg: dict[str, Any],
    goldrule: Any,
    rows_per_shard: int,
) -> dict[str, Any]:
    s1_path, source, relpath = task
    scored_path, _pairs_path, _summary_path = s4.scored_output_paths(
        out_root, source, relpath
    )
    columns = ["doc_id", "text", "source", "upstream_score"]
    raw = next(
        iter(pq.ParquetFile(s1_path).iter_batches(
            batch_size=rows_per_shard, columns=columns
        )),
        None,
    )
    stored = next(
        iter(pq.ParquetFile(scored_path).iter_batches(
            batch_size=rows_per_shard,
            columns=["doc_id", "hybrid_v2", "llm_distill", "rule_score"],
        )),
        None,
    )
    if raw is None or raw.num_rows == 0:
        return {"source": source, "relpath": relpath, "checked": 0, "mismatches": []}
    if stored is None or stored.num_rows != raw.num_rows:
        raise ValueError(f"S1/S4 sample row closure failed: {source}/{relpath}")
    rd = raw.to_pydict()
    sd = stored.to_pydict()
    if rd["doc_id"] != sd["doc_id"]:
        raise ValueError(f"S1/S4 sample order mismatch: {source}/{relpath}")
    texts = [text if isinstance(text, str) else "" for text in rd["text"]]
    max_chars = int((cfg.get("inference") or {}).get("max_score_chars", 20000))
    rule = rule_scores_batch(texts, min(max_chars, 2048))
    llm = s4.goldrule_scores_batch(goldrule, texts)
    upstream_norm, _valid = s4._upstream_norm_array(
        rd["source"], rd["upstream_score"], cfg
    )
    expected = s4.hybrid_v2_vector(llm, upstream_norm, rule, cfg).astype(np.float32)
    effective = s4.effective_hybrid_v2(
        np.asarray(sd["hybrid_v2"], dtype=np.float32),
        rd["upstream_score"],
        cfg,
        "legacy_zero_upstream_v1",
    )
    stored_llm = np.asarray(sd["llm_distill"], dtype=np.float32)
    stored_rule = np.asarray(sd["rule_score"], dtype=np.float32)
    mismatches: list[dict[str, Any]] = []
    for i, doc_id in enumerate(rd["doc_id"]):
        diffs = {
            "llm_distill": abs(float(stored_llm[i]) - float(llm[i])),
            "rule_score": abs(float(stored_rule[i]) - float(rule[i])),
            "effective_hybrid_v2": abs(float(effective[i]) - float(expected[i])),
        }
        if max(diffs.values()) > 2e-6:
            mismatches.append({"doc_id": doc_id, "abs_diff": diffs})
    return {
        "source": source,
        "relpath": relpath,
        "checked": raw.num_rows,
        "missing_upstream": sum(v is None for v in rd["upstream_score"]),
        "mismatches": mismatches,
    }


def run_s4_audit(
    out_root: Path,
    cfg: dict[str, Any],
    shards_per_source: int,
    rows_per_shard: int,
) -> Path:
    goldrule = s4.load_goldrule_model(cfg)
    if goldrule is None:
        raise SystemExit("S4 audit requires the pinned goldrule model")
    grouped: dict[str, list[tuple[Path, str, str]]] = defaultdict(list)
    for task in _s4_tasks(out_root):
        grouped[task[1]].append(task)
    results: list[dict[str, Any]] = []
    errors: list[str] = []
    for source in sorted(grouped):
        for task in _evenly_spaced(grouped[source], shards_per_source):
            try:
                results.append(
                    _audit_one_s4_shard(
                        out_root, task, cfg, goldrule, rows_per_shard
                    )
                )
            except Exception as exc:
                errors.append(
                    f"{source}/{task[2]}: {type(exc).__name__}: {exc}"
                )
        print(
            f"S4 audit: {source} checked_rows="
            f"{sum(r['checked'] for r in results if r['source'] == source)}",
            flush=True,
        )
    mismatches = sum(len(row["mismatches"]) for row in results)
    payload = {
        "stage": "legacy_s4_hybrid_v2_exact_audit",
        "schema_version": 1,
        "method": "evenly_spaced_shards_first_rows_exact_component_recompute_with_v1_to_v2_transform",
        "selection_semantic_sha256": s4.selection_semantic_sha256(cfg, "hybrid_v2"),
        "goldrule_sha256": _sha256(Path(str(cfg["goldrule_model"]))),
        "shards_per_source": shards_per_source,
        "rows_per_shard": rows_per_shard,
        "checked_rows": sum(int(row["checked"]) for row in results),
        "missing_upstream_rows": sum(
            int(row.get("missing_upstream") or 0) for row in results
        ),
        "mismatches": mismatches,
        "errors": errors,
        "passed": not errors and mismatches == 0,
        "samples": results,
    }
    return _write_json(
        out_root / "audits" / "s4_hybrid_v2_legacy_audit.json", payload
    )


def run_s4_certification(
    out_root: Path, cfg: dict[str, Any], workers: int
) -> Path:
    audit_path = out_root / "audits" / "s4_hybrid_v2_legacy_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        not audit.get("passed")
        or audit.get("selection_semantic_sha256")
        != s4.selection_semantic_sha256(cfg, "hybrid_v2")
    ):
        raise SystemExit("S4 exact audit missing, failed, or bound to stale semantics")
    audit_sha = _sha256(audit_path)
    tasks = _s4_tasks(out_root)

    def certify(task: tuple[Path, str, str]) -> dict[str, Any]:
        s1_path, source, relpath = task
        try:
            result = s4.certify_legacy_selection_shard(
                s1_path, out_root, source, relpath, cfg, audit_sha
            )
            return {
                "source": source,
                "relpath": relpath,
                "ok": True,
                "provenance": (result.get("selection_contracts") or {})
                .get("hybrid_v2", {})
                .get("provenance"),
            }
        except Exception as exc:
            return {
                "source": source,
                "relpath": relpath,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(certify, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if i % 500 == 0:
                print(f"S4 certification: {i}/{len(tasks)}", flush=True)
    failed = [row for row in rows if not row["ok"]]
    payload = {
        "stage": "legacy_s4_selection_certification",
        "schema_version": 1,
        "exact_audit_sha256": audit_sha,
        "tasks": len(tasks),
        "passed": not failed,
        "failed": len(failed),
        "errors": failed[:100],
        "per_source": dict(Counter(row["source"] for row in rows if row["ok"])),
        "provenance": dict(Counter(row.get("provenance") for row in rows if row["ok"])),
    }
    return _write_json(out_root / "audits" / "s4_lineage_migration.json", payload)


def run_s5_certification(
    out_root: Path,
    profiles: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
    tokenizer_path: Path,
    workers: int,
    promote_unchanged: bool,
) -> Path:
    audit_path = out_root / "audits" / "s5_legacy_token_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("passed") or audit.get("tokenizer_sha256") != _sha256(tokenizer_path):
        raise SystemExit("S5 legacy token audit missing, failed, or bound to another tokenizer")
    audit_sha = _sha256(audit_path)
    ctx = lb.load_context_lengths()
    tasks = lb.s5_tasks(out_root, profiles)
    changed_sources = set((cfg.get("source_min_distinct_hits") or {}).keys())

    def certify(task: tuple[str, str, str]) -> dict[str, Any]:
        source, relpath, domain = task
        promote = promote_unchanged and domain not in {"web", "code"} and source not in changed_sources
        try:
            if promote:
                in_path, _ = s3_output_paths(out_root, source, relpath)
                dec_path, dec_summary_path = dct.decontam_output_paths(out_root, source, relpath)
                dec_summary = json.loads(dec_summary_path.read_text(encoding="utf-8"))
                reason = dct.validate_published_summary(
                    in_path, dec_path, dec_summary, cfg, source
                )
                if reason:
                    raise ValueError(f"decontam dependency is stale: {reason}")
            result = lb.certify_legacy_s5_output(
                out_root, source, relpath, domain, ctx, tokenizer_path,
                promote_processing_contract=promote,
                audit_sha256=audit_sha,
            )
            return {"source": source, "relpath": relpath, "ok": True,
                    "promoted": promote, "skipped": bool(result.get("skipped"))}
        except Exception as exc:
            return {"source": source, "relpath": relpath, "ok": False,
                    "promoted": promote, "error": f"{type(exc).__name__}: {exc}"}

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(certify, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if i % 500 == 0:
                print(f"S5 certification: {i}/{len(tasks)}", flush=True)
    failed = [row for row in rows if not row["ok"]]
    payload = {
        "stage": "legacy_s5_structural_certification",
        "schema_version": 1,
        "token_audit_sha256": audit_sha,
        "tasks": len(tasks),
        "passed": not failed,
        "failed": len(failed),
        "promoted": sum(bool(row.get("promoted")) for row in rows if row["ok"]),
        "cache_only": sum(not bool(row.get("promoted")) for row in rows if row["ok"]),
        "errors": failed[:100],
        "per_source": dict(Counter(row["source"] for row in rows if row["ok"])),
    }
    return _write_json(out_root / "audits" / "s5_lineage_migration.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["token-audit", "decontam", "s4-audit", "s4-certify", "s5"],
        required=True,
    )
    parser.add_argument("--out-root", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--profiles", type=Path, default=None)
    parser.add_argument("--decontam-config", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--scorer", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--shards-per-source", type=int, default=16)
    parser.add_argument("--rows-per-shard", type=int, default=8)
    parser.add_argument("--promote-unchanged", action="store_true")
    args = parser.parse_args()
    profiles = load_profiles(args.profiles)
    cfg = dct.load_decontam_config(args.decontam_config)
    if args.stage == "token-audit":
        tokenizer_path = lb.resolve_tokenizer_path(args.tokenizer)
        path = run_token_audit(
            args.out_root, profiles, tokenizer_path,
            args.shards_per_source, args.rows_per_shard,
        )
    elif args.stage == "decontam":
        path = run_decontam_migration(args.out_root, cfg, args.workers)
    elif args.stage == "s4-audit":
        path = run_s4_audit(
            args.out_root,
            s4.load_scorer_config(args.scorer),
            args.shards_per_source,
            args.rows_per_shard,
        )
    elif args.stage == "s4-certify":
        path = run_s4_certification(
            args.out_root, s4.load_scorer_config(args.scorer), args.workers
        )
    else:
        tokenizer_path = lb.resolve_tokenizer_path(args.tokenizer)
        path = run_s5_certification(
            args.out_root, profiles, cfg, tokenizer_path,
            args.workers, args.promote_unchanged,
        )
    print(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return 0 if payload.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
