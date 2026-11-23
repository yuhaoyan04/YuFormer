from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from yuformer_data.scale import length_bucket as lb
from yuformer_data.scale.dedup_minhash import S3_DECISIONS_SCHEMA
from yuformer_data.scale.schema import make_doc_id

CTX = [4096, 16384, 65536]


def _doc_ids(n: int, tag: str) -> list[str]:
    return [make_doc_id(tag, "shard0", i) for i in range(n)]


def _write_s3_shard(root: Path, source: str, rel: str, rows: list[dict]) -> None:
    table = pa.table(
        {
            "doc_id": [r["doc_id"] for r in rows],
            "text": [r.get("text", "some words here") for r in rows],
            "s2_decision": [r.get("s2", "keep") for r in rows],
        }
    )
    path = root / "s3" / source / (rel + ".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def _write_decontam(root: Path, source: str, rel: str, drops: set[str]) -> None:
    from yuformer_data.scale.decontam import DECONTAM_SCHEMA

    table = pa.table(
        {
            "doc_id": sorted(drops),
            "decision": ["drop_contaminated"] * len(drops),
            "hit_benchmark": ["mmlu"] * len(drops),
        },
        schema=DECONTAM_SCHEMA,
    )
    path = root / "decontam" / source / (rel + ".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    lb.write_summary(
        root / "decontam" / source / (rel + ".summary.json"),
        {"stage": "decontam", "in_rows": len(drops), "errors": {}},
    )


def _write_s3_decisions(root: Path, domain: str, drops: list[str]) -> None:
    path = root / "s3_decisions" / f"{domain}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "doc_id": drops,
            "decision": ["drop_near_dup"] * len(drops),
            "dup_of": ["00" * 16] * len(drops),
        },
        schema=S3_DECISIONS_SCHEMA,
    )
    pq.write_table(table, path)


def test_bucket_and_eligibility():
    assert lb.eligibility_flags(4096, CTX) == ["B4K", "B16K", "B64K"]
    assert lb.eligibility_flags(4097, CTX) == ["B16K", "B64K"]
    assert lb.eligibility_flags(16384, CTX) == ["B16K", "B64K"]
    assert lb.eligibility_flags(65536, CTX) == ["B64K"]
    assert lb.eligibility_flags(65537, CTX) == []


def test_s5_global_contract_tracks_control_files(tmp_path: Path):
    root = tmp_path / "out"
    (root / "s3_decisions").mkdir(parents=True)
    (root / "s3_v2").mkdir(parents=True)
    controls = [
        root / "stage_decontam_global_summary.json",
        *(root / "s3_decisions" / f"{d}.summary.json" for d in ("code", "web")),
        *(root / "s3_v2" / f"{d}_minhash_precision_audit.json" for d in ("code", "web")),
    ]
    for path in controls:
        path.write_text("{}")
    tokenizer = tmp_path / "tokenizer.json"
    cleaning = tmp_path / "cleaning.yaml"
    tokenizer.write_text("tokenizer-v1")
    cleaning.write_text("context_lengths: [4096]")
    before = lb.s5_global_input_contract(root, tokenizer, cleaning)
    controls[0].write_text('{"generation": 2}')
    after = lb.s5_global_input_contract(root, tokenizer, cleaning)
    assert before["schema_version"] == 2
    assert before != after


def test_process_shard_s5_decision_join_and_idempotency(tmp_path: Path):
    root = tmp_path / "out"
    source, rel, domain = "dolma-books", "shard0", "books"
    ids = _doc_ids(6, source)
    rows = [
        {"doc_id": ids[0], "text": "hello world"},
        {"doc_id": ids[1], "text": "hello world"},  # s2 drop
        {"doc_id": ids[2], "text": "hello world"},  # s3 drop
        {"doc_id": ids[3], "text": "hello world"},  # decontam drop
        {"doc_id": ids[4], "text": "word " * 6000},  # ~7.5k tokens → B16K
        {"doc_id": ids[5], "text": "word " * 150000},  # ~190k tokens → gt_64K
    ]
    rows[1]["s2"] = "drop_exact_dup"
    _write_s3_shard(root, source, rel, rows)
    _write_s3_decisions(root, domain, [ids[2]])
    _write_decontam(root, source, rel, {ids[3]})

    summary = lb.process_shard_s5(root, source, rel, domain, CTX)
    assert summary["in_rows"] == 6
    assert summary["decisions"]["drop_s2_exact_dup"] == 1
    assert summary["decisions"]["drop_s3_near_dup"] == 1
    assert summary["decisions"]["drop_decontam"] == 1
    assert summary["decisions"]["keep"] == 3
    assert summary["buckets"]["B4K"] == 1
    assert summary["buckets"]["B16K"] == 1
    assert summary["buckets"]["gt_64K"] == 1
    assert sum(summary["bucket_tokens"].values()) == summary["total_tokens"]
    assert len(summary["content_sha256"]) == 64
    assert summary["token_count_cache_contract"]["schema_version"] == 2
    assert summary["token_count_cache_contract"]["provenance"].startswith(
        "current_pipeline_tokenization"
    )

    out_path, out_summary = lb.s5_output_paths(root, source, rel)
    table = pq.read_table(out_path)
    assert table.schema.equals(lb.S5_SCHEMA)
    by_id = {r["doc_id"]: r for r in table.to_pylist()}
    assert set(by_id) == {ids[0], ids[4], ids[5]}
    # n_tokens 与 yf-64k-bpe encode 精确一致
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(lb.resolve_tokenizer_path()))
    assert by_id[ids[0]]["n_tokens"] == len(tok.encode("hello world").ids)
    assert by_id[ids[0]]["mix_eligibility"] == ["B4K", "B16K", "B64K"]
    assert by_id[ids[4]]["mix_eligibility"] == ["B16K", "B64K"]
    assert by_id[ids[5]]["mix_eligibility"] == []

    # 幂等：二次运行 skip
    again = lb.process_shard_s5(root, source, rel, domain, CTX)
    assert again["skipped"] is True

    # Any decontam contract change invalidates S5 even when the drop parquet has
    # the same row count/size. This closes the old targeted-rerun stale-cache bug.
    dec_summary = root / "decontam" / source / (rel + ".summary.json")
    changed = json.loads(dec_summary.read_text())
    changed["config_sha256"] = "new-policy"
    dec_summary.write_text(json.dumps(changed, sort_keys=True))
    refreshed = lb.process_shard_s5(root, source, rel, domain, CTX)
    assert refreshed["skipped"] is False
    assert refreshed["processing_contract"]["decontam_summary_sha256"]


def test_process_shard_s5_missing_decontam_is_error(tmp_path: Path):
    root = tmp_path / "out"
    source, rel, domain = "dolma-books", "shard0", "books"
    _write_s3_shard(root, source, rel, [{"doc_id": d} for d in _doc_ids(2, source)])
    _write_s3_decisions(root, domain, [])
    summary = lb.process_shard_s5(root, source, rel, domain, CTX)
    assert summary["errors"]
    assert "decontam shard missing" in summary.get("error_message", "")


def test_certified_legacy_token_cache_reuses_counts_and_tokenizes_rescues(tmp_path: Path):
    root = tmp_path / "out"
    source, rel, domain = "dolma-books", "shard0", "books"
    ids = _doc_ids(3, source)
    _write_s3_shard(
        root,
        source,
        rel,
        [
            {"doc_id": ids[0], "text": "first retained text"},
            {"doc_id": ids[1], "text": "previously contaminated text"},
            {"doc_id": ids[2], "text": "third retained text"},
        ],
    )
    _write_s3_decisions(root, domain, [])
    _write_decontam(root, source, rel, {ids[1]})
    first = lb.process_shard_s5(root, source, rel, domain, CTX)
    assert first["out_rows"] == 2

    certified = lb.certify_legacy_s5_output(
        root,
        source,
        rel,
        domain,
        CTX,
        lb.resolve_tokenizer_path(),
        audit_sha256="a" * 64,
    )
    assert certified["token_count_cache_contract"]["structural_validation_passed"]

    # Policy correction rescues one record. Existing records must use the
    # certified cache, while the rescued record is tokenized with the current model.
    _write_decontam(root, source, rel, set())
    refreshed = lb.process_shard_s5(root, source, rel, domain, CTX)
    assert refreshed["out_rows"] == 3
    assert refreshed["token_cache_reuse"] == {
        "enabled": True,
        "reused_rows": 2,
        "newly_tokenized_rows": 1,
        "exact_audited_rows": 2,
    }


def test_recover_damaged_s5_summary_without_retokenizing(tmp_path: Path):
    root = tmp_path / "out"
    source, rel, domain = "dolma-books", "shard0", "books"
    ids = _doc_ids(4, source)
    rows = [
        {"doc_id": ids[0], "text": "first"},
        {"doc_id": ids[1], "text": "exact duplicate", "s2": "drop_exact_dup"},
        {"doc_id": ids[2], "text": "near duplicate"},
        {"doc_id": ids[3], "text": "retained"},
    ]
    _write_s3_shard(root, source, rel, rows)
    _write_s3_decisions(root, domain, [ids[2]])
    _write_decontam(root, source, rel, set())
    good = lb.process_shard_s5(root, source, rel, domain, CTX)
    _, summary_path = lb.s5_output_paths(root, source, rel)
    summary_path.write_text(json.dumps({
        "stage": "s5", "source": source, "relpath": rel, "domain": domain,
        "errors": {"shard_error:OSError": 1}, "error_message": "error closing file",
    }))

    recovered = lb.recover_s5_summary_from_certified_output(
        root,
        source,
        rel,
        domain,
        CTX,
        lb.resolve_tokenizer_path(),
        audit_sha256="c" * 64,
    )
    assert recovered["errors"] == {}
    assert recovered["decisions"] == good["decisions"]
    assert recovered["content_sha256"] == good["content_sha256"]
    assert recovered["lineage_recovery"]["ordered_keep_set_verified"] is True
    assert recovered["lineage_recovery"]["s3_text_read"] is False
    assert recovered["token_cache_reuse"]["newly_tokenized_rows"] == 0


def test_restrictive_ndv2_filters_certified_s5_without_s3_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "out"
    source, rel, domain = "fineweb", "shard0", "web"
    ids = _doc_ids(3, source)
    _write_s3_shard(
        root,
        source,
        rel,
        [{"doc_id": doc_id, "text": f"document {i}"} for i, doc_id in enumerate(ids)],
    )
    _write_s3_decisions(root, domain, [])
    _write_decontam(root, source, rel, {ids[2]})
    first = lb.process_shard_s5(root, source, rel, domain, CTX)
    assert first["out_rows"] == 2
    certified = lb.certify_legacy_s5_output(
        root,
        source,
        rel,
        domain,
        CTX,
        lb.resolve_tokenizer_path(),
        audit_sha256="b" * 64,
    )
    assert certified["token_count_cache_contract"]["structural_validation_passed"]

    # The new lookup adds one drop from the old keep table and overlaps the old
    # decontam drop once.  No record can be rescued from an empty old lookup.
    lookup = root / "s3_lookup" / domain
    lookup.mkdir(parents=True)
    encoded = np.sort(lb._ids_to_void([ids[1], ids[2]]))
    (lookup / "lookup.ids.bin").write_bytes(encoded.tobytes())
    (lookup / "lookup.codes.bin").write_bytes(b"\x01\x01")
    monkeypatch.setattr(
        lb,
        "_encode_counts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("restrictive delta must not tokenize")
        ),
    )
    refreshed = lb.process_shard_s5(root, source, rel, domain, CTX)
    assert refreshed["decisions"] == {
        "drop_s2_exact_dup": 0,
        "drop_s3_near_dup": 2,
        "drop_decontam": 0,
        "keep": 1,
    }
    assert refreshed["incremental_filter"]["s3_text_read"] is False
    assert refreshed["token_cache_reuse"]["newly_tokenized_rows"] == 0
    out_path, _ = lb.s5_output_paths(root, source, rel)
    assert pq.read_table(out_path, columns=["doc_id"]).column(0).to_pylist() == [ids[0]]


def test_fullscan_decontam_filters_certified_s5_without_s3_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "out"
    source, rel, domain = "dolma-books", "shard0", "books"
    ids = _doc_ids(3, source)
    _write_s3_shard(
        root,
        source,
        rel,
        [{"doc_id": doc_id, "text": f"document {i}"} for i, doc_id in enumerate(ids)],
    )
    _write_s3_decisions(root, domain, [])
    _write_decontam(root, source, rel, set())
    first = lb.process_shard_s5(root, source, rel, domain, CTX)
    assert first["out_rows"] == 3
    certified = lb.certify_legacy_s5_output(
        root,
        source,
        rel,
        domain,
        CTX,
        lb.resolve_tokenizer_path(),
        audit_sha256="c" * 64,
    )
    assert certified["token_count_cache_contract"]["structural_validation_passed"]

    _write_decontam(root, source, rel, {ids[1]})
    summary_path = root / "decontam" / source / (rel + ".summary.json")
    decontam_summary = json.loads(summary_path.read_text())
    decontam_summary["scan_policy"] = {
        "schema_version": 1,
        "scope": "full_document",
        "max_shingles_per_doc": None,
        "transition": {
            "from_scope": "prefix",
            "from_max_shingles_per_doc": 50000,
            "monotonic_effect": "only_add_drops",
        },
    }
    summary_path.write_text(json.dumps(decontam_summary, sort_keys=True))
    monkeypatch.setattr(
        lb,
        "_encode_counts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fullscan decontam delta must not tokenize")
        ),
    )
    refreshed = lb.process_shard_s5(root, source, rel, domain, CTX)
    assert refreshed["decisions"] == {
        "drop_s2_exact_dup": 0,
        "drop_s3_near_dup": 0,
        "drop_decontam": 1,
        "keep": 2,
    }
    assert refreshed["incremental_filter"] == {
        "schema_version": 1,
        "method": "prefix_to_full_decontam_only_adds_drops",
        "old_rows": 3,
        "newly_dropped_rows": 1,
        "s3_text_read": False,
        "proof": "full-document shingle set is a superset of the former prefix",
    }
    assert refreshed["token_count_cache_contract"]["output_fingerprint"] == lb.fingerprint(
        lb.s5_output_paths(root, source, rel)[0]
    )
    assert pq.read_table(
        lb.s5_output_paths(root, source, rel)[0], columns=["doc_id"]
    ).column(0).to_pylist() == [ids[0], ids[2]]


def test_s5_run_gate_decontam_coverage(tmp_path: Path):
    root = tmp_path / "out"
    source, rel = "dolma-books", "shard0"
    _write_s3_shard(root, source, rel, [{"doc_id": d} for d in _doc_ids(2, source)])
    profiles = {source: {"domain": "books"}}
    with pytest.raises(SystemExit) as exc:
        lb.run_s5(root, profiles, workers=1, sources=[source],
                  guard=(lambda: None, __import__("threading").Event()))
    assert "decontam 未覆盖" in str(exc.value)


def test_s5_revokes_global_publication_when_upstream_changes_mid_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "out"
    source, rel = "dolma-books", "shard0"
    _write_s3_shard(root, source, rel, [{"doc_id": d} for d in _doc_ids(1, source)])
    (root / "stage_decontam_global_summary.json").write_text(json.dumps({
        "stage": "decontam",
        "shard_errors": 0,
        "coverage": {"expected": 1, "complete": 1, "missing_summary": 0, "missing_output": 0},
        "totals": {"truncated_docs": 0},
        "publication_contract": {
            "scan_policy": {"scope": "full_document", "max_shingles_per_doc": None}
        },
    }))
    global_path = root / "stage_s5_global_summary.json"
    global_path.write_text(json.dumps({"input_contract": {"generation": 1}}))
    monkeypatch.setattr(lb, "check_decontam_coverage", lambda *_a, **_kw: [])
    monkeypatch.setattr(lb, "check_ndv2_done", lambda *_a, **_kw: [])
    monkeypatch.setattr(lb, "run_s5_scan", lambda *_a, **_kw: [])
    monkeypatch.setattr(
        lb, "collect_published_summaries", lambda *_a, **_kw: ([], {"expected": 1, "complete": 1})
    )
    contracts = iter(({"generation": 1}, {"generation": 2}))
    monkeypatch.setattr(lb, "s5_global_input_contract", lambda *_a, **_kw: next(contracts))
    with pytest.raises(SystemExit, match="upstream changed"):
        lb.run_s5(
            root,
            {source: {"domain": "books"}},
            workers=1,
            sources=[source],
            guard=(lambda: None, __import__("threading").Event()),
        )
    assert not global_path.exists()


def test_s5_does_not_publish_a_generation_with_shard_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "out"
    source, rel = "dolma-books", "shard0"
    _write_s3_shard(root, source, rel, [{"doc_id": d} for d in _doc_ids(1, source)])
    (root / "stage_decontam_global_summary.json").write_text(json.dumps({
        "stage": "decontam",
        "shard_errors": 0,
        "coverage": {"expected": 1, "complete": 1, "missing_summary": 0, "missing_output": 0},
        "totals": {"truncated_docs": 0},
        "publication_contract": {
            "scan_policy": {"scope": "full_document", "max_shingles_per_doc": None}
        },
    }))
    global_path = root / "stage_s5_global_summary.json"
    global_path.write_text(json.dumps({"input_contract": {"generation": 1}}))
    monkeypatch.setattr(lb, "check_decontam_coverage", lambda *_a, **_kw: [])
    monkeypatch.setattr(lb, "check_ndv2_done", lambda *_a, **_kw: [])
    monkeypatch.setattr(lb, "run_s5_scan", lambda *_a, **_kw: [])
    failed = {
        "stage": "s5", "source": source, "relpath": rel, "domain": "books",
        "in_rows": 1, "out_rows": 0, "errors": {"shard_error:OSError": 1},
    }
    monkeypatch.setattr(
        lb,
        "collect_published_summaries",
        lambda *_a, **_kw: (
            [failed],
            {"expected": 1, "complete": 1, "missing_summary": 0, "missing_output": 0},
        ),
    )
    monkeypatch.setattr(
        lb, "s5_global_input_contract", lambda *_a, **_kw: {"generation": 1}
    )
    with pytest.raises(SystemExit, match="publication rejected"):
        lb.run_s5(
            root,
            {source: {"domain": "books"}},
            workers=1,
            sources=[source],
            guard=(lambda: None, __import__("threading").Event()),
        )
    assert not global_path.exists()
    diagnostic = root / "audits" / "stage_s5_last_failed_summary.json"
    assert json.loads(diagnostic.read_text())["shard_errors"] == 1
