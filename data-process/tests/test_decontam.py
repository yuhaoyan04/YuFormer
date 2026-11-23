from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from yuformer_data.scale import decontam as dct

N = 8

# 合成基准文本（各基准一段"题面+选项+答案"式文本）
FAKE_BENCH_TEXTS: dict[str, list[str]] = {
    "mmlu": [
        "Which planet has the largest ring system in the solar system?\n"
        "Mercury\nVenus\nSaturn\nMars\nThe answer is Saturn."
    ],
    "gsm8k": [
        "Natalia sold clips to 48 friends in April, and then she sold half as many "
        "clips in May. How many clips did she sell altogether in April and May?\n"
        "In May she sold 48/2 = 24 clips. Total 48+24 = 72."
    ],
    "math": [
        "What is the sum of all positive integers n such that n^2 - 19n + 99 is a perfect square?\n"
        "Solution: write n^2 - 19n + 99 = k^2 and complete the square."
    ],
    "humaneval": [
        "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n    return a + b\n"
        "assert add(2, 3) == 5"
    ],
    "ceval": [
        "下列哪一项属于货币政策工具？\n公开市场操作\n财政补贴\n税收优惠\n行政管制\n答案是公开市场操作。"
    ],
}


def _cfg(tmp_path: Path, **over) -> dict:
    cfg = {
        "ngram_n": N,
        "max_shingles_per_doc": 50000,
        "input_stage": "s3",
        "index_dir": str(tmp_path / "index"),
        "memory_cap_gb": 4.0,
        "benchmarks": [{"name": name, "repo": "fake/repo", "paths": ["x.parquet"],
                        "fields": ["text"]} for name in FAKE_BENCH_TEXTS],
    }
    cfg.update(over)
    return cfg


def _build_index(tmp_path: Path, cfg: dict | None = None) -> tuple[dict, dct.BenchmarkIndex]:
    cfg = cfg or _cfg(tmp_path)
    payload = dct.write_index(cfg, Path(cfg["index_dir"]), FAKE_BENCH_TEXTS)
    return payload, dct.BenchmarkIndex(Path(cfg["index_dir"]))


def _write_shard(out_root: Path, source: str, relpath: str, docs: list[tuple[str, str]]) -> Path:
    root = out_root / "s3" / source
    path = root / (relpath + ".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"doc_id": [d for d, _ in docs], "text": [t for _, t in docs]}),
        path,
        compression="zstd",
    )
    return path


def _read_decisions(out_root: Path, source: str, relpath: str) -> dict[str, dict]:
    path, _ = dct.decontam_output_paths(out_root, source, relpath)
    rows = pq.read_table(path).to_pylist()
    return {r["doc_id"]: r for r in rows}


def _overlap_doc(bench_text: str, n_words: int = 14) -> str:
    """取基准文本前 n_words 个 token 作为独立文档（共享 n_words-N+1 个 8-gram）。"""
    return " ".join(dct._tokenize(bench_text)[:n_words])


def test_index_from_injected_fake_data(tmp_path):
    payload, index = _build_index(tmp_path)
    assert payload["skipped"] is False
    assert index.n == payload["n_unique_shingles"]
    assert index.n > 0
    total = sum(v["unique_shingles"] for v in payload["per_benchmark"].values())
    assert total >= index.n  # 跨基准去重后不大于分基准之和
    names = [index.names[int(c)] for c in index.codes]
    for name in FAKE_BENCH_TEXTS:
        assert name in names


def test_fast_normalize_byte_identical():
    # ASCII 快路径必须与 dedup_common.normalize_for_hash 逐字节一致（否则索引/扫描两侧失配）
    from yuformer_data.scale.dedup_common import normalize_for_hash

    corpus = [
        "Hello   World\n\n\tThis is a TEST.\n",
        "line1\nline2  \nline3\r\nline4",
        "trailing space   ",
        "mixed CASE with numbers 123 and punct !@#$%^&*()",
        "\x00control\x01 chars\x7f here",
        "The quick brown fox  jumps over  the lazy dog. " * 40,
    ]
    for text in corpus:
        assert dct._fast_normalize(text) == normalize_for_hash(text), repr(text[:50])
    # CJK 走慢路径也应一致
    zh = "下列哪一项属于货币政策工具？\n公开市场操作  财政补贴\n答案是公开市场操作。"
    assert dct._fast_normalize(zh) == normalize_for_hash(zh)


def test_vectorized_shingle_hash_matches_polynomial_definition():
    words = np.array([1, 7, 13, 2, 99, 5, 8, 21, 34, 55, 89], dtype="<u8")
    got = dct.shingle_hashes_u64(words, N)
    expected = []
    for start in range(len(words) - N + 1):
        value = 0
        for word in words[start : start + N]:
            value = (value * dct._ROLL_PRIME + int(word)) % (2**64)
        expected.append(value)
    assert got.tolist() == expected


def test_index_idempotent_config_echo(tmp_path):
    cfg = _cfg(tmp_path)
    p1 = dct.write_index(cfg, Path(cfg["index_dir"]), FAKE_BENCH_TEXTS)
    p2 = dct.build_benchmark_index(cfg, Path(cfg["index_dir"]))
    assert p2["skipped"] is True
    assert p2["n_unique_shingles"] == p1["n_unique_shingles"]
    # Scan-only caps must not invalidate identical benchmark index bytes.
    changed_scan = {**cfg, "max_shingles_per_doc": 123}
    assert dct._config_echo(changed_scan) == dct._config_echo(cfg)
    assert dct._scan_config_echo(changed_scan, "src") != dct._scan_config_echo(cfg, "src")


def test_known_overlap_dropped_with_benchmark_name(tmp_path):
    _, index = _build_index(tmp_path)
    cfg = _cfg(tmp_path)
    out_root = tmp_path / "out"
    docs = [
        ("d1", _overlap_doc(FAKE_BENCH_TEXTS["gsm8k"][0])),
        ("d2", _overlap_doc(FAKE_BENCH_TEXTS["ceval"][0])),
        ("d3", "totally unrelated document about gardening tools and watering cans"),
    ]
    _write_shard(out_root, "src1", "a/b", docs)
    summary = dct.process_shard_decontam(
        out_root / "s3" / "src1" / "a" / "b.parquet", out_root, "src1", "a/b", index, cfg
    )
    decisions = _read_decisions(out_root, "src1", "a/b")
    assert decisions["d1"]["decision"] == dct.DECISION_DROP
    assert decisions["d1"]["hit_benchmark"] == "gsm8k"
    assert decisions["d2"]["decision"] == dct.DECISION_DROP
    assert decisions["d2"]["hit_benchmark"] == "ceval"
    assert decisions["d3"]["decision"] == dct.DECISION_KEEP
    assert decisions["d3"]["hit_benchmark"] == ""
    assert summary["hit_benchmarks"] == {"gsm8k": 1, "ceval": 1}
    assert summary["errors"] == {}


def test_benchmark_specific_distinct_hit_threshold(tmp_path):
    cfg = _cfg(tmp_path)
    for bench in cfg["benchmarks"]:
        if bench["name"] == "ceval":
            bench["min_distinct_hits"] = 2
    _, index = _build_index(tmp_path, cfg)
    tokens = dct._tokenize(FAKE_BENCH_TEXTS["ceval"][0])
    out_root = tmp_path / "out"
    _write_shard(
        out_root,
        "src1",
        "a",
        [("one", "".join(tokens[:N])), ("two", "".join(tokens[:N + 1]))],
    )
    dct.process_shard_decontam(
        out_root / "s3" / "src1" / "a.parquet", out_root, "src1", "a", index, cfg
    )
    decisions = _read_decisions(out_root, "src1", "a")
    assert decisions["one"]["decision"] == dct.DECISION_KEEP
    assert decisions["two"]["decision"] == dct.DECISION_DROP
    assert decisions["two"]["hit_benchmark"] == "ceval"


def test_source_specific_threshold_does_not_weaken_other_sources(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["source_min_distinct_hits"] = {"zh-source": {"ceval": 2}}
    _, index = _build_index(tmp_path, cfg)
    tokens = dct._tokenize(FAKE_BENCH_TEXTS["ceval"][0])
    text = "".join(tokens[:N])  # exactly one distinct C-Eval shingle
    out_root = tmp_path / "out"
    for source in ("zh-source", "other-source"):
        _write_shard(out_root, source, "a", [(source, text)])
        dct.process_shard_decontam(
            out_root / "s3" / source / "a.parquet",
            out_root, source, "a", index, cfg,
        )
    zh = _read_decisions(out_root, "zh-source", "a")
    other = _read_decisions(out_root, "other-source", "a")
    assert zh["zh-source"]["decision"] == dct.DECISION_KEEP
    assert other["other-source"]["decision"] == dct.DECISION_DROP


def test_legacy_threshold_increase_recomputes_only_selected_benchmark(tmp_path):
    base_cfg = _cfg(tmp_path)
    _, index = _build_index(tmp_path, base_cfg)
    tokens = dct._tokenize(FAKE_BENCH_TEXTS["ceval"][0])
    docs = [
        ("ceval-one", "".join(tokens[:N])),
        ("ceval-two", "".join(tokens[: N + 1])),
        ("mmlu", _overlap_doc(FAKE_BENCH_TEXTS["mmlu"][0])),
        ("clean", "unrelated garden tools and watering cans"),
    ]
    out_root = tmp_path / "out"
    in_path = _write_shard(out_root, "zh-source", "a", docs)
    first = dct.process_shard_decontam(
        in_path, out_root, "zh-source", "a", index, base_cfg
    )
    assert first["decisions"][dct.DECISION_DROP] == 3

    # Reproduce the unversioned production summary while keeping the exact old
    # row-aligned decision table as the incremental input.
    out_path, summary_path = dct.decontam_output_paths(out_root, "zh-source", "a")
    legacy = dict(first)
    for key in (
        "input_fingerprint",
        "config_sha256",
        "index_sha256",
        "effective_min_distinct_hits",
    ):
        legacy.pop(key, None)
    summary_path.write_text(json.dumps(legacy), encoding="utf-8")
    assert in_path.stat().st_mtime_ns <= out_path.stat().st_mtime_ns

    changed_cfg = dict(base_cfg)
    changed_cfg["source_min_distinct_hits"] = {"zh-source": {"ceval": 2}}
    second = dct.process_shard_decontam(
        in_path, out_root, "zh-source", "a", index, changed_cfg
    )
    decisions = _read_decisions(out_root, "zh-source", "a")
    assert decisions["ceval-one"]["decision"] == dct.DECISION_KEEP
    assert decisions["ceval-two"]["decision"] == dct.DECISION_DROP
    assert decisions["mmlu"]["hit_benchmark"] == "mmlu"
    assert decisions["clean"]["decision"] == dct.DECISION_KEEP
    assert second["incremental_rescan"]["recomputed_rows"] == 2
    assert second["incremental_rescan"]["changed_benchmarks"] == ["ceval"]
    assert dct.validate_published_summary(
        in_path, out_path, second, changed_cfg, "zh-source"
    ) is None


def test_seven_gram_overlap_not_dropped(tmp_path):
    # 边界：仅共享 7-gram（<8 词）不命中
    _, index = _build_index(tmp_path)
    cfg = _cfg(tmp_path)
    out_root = tmp_path / "out"
    bench_words = FAKE_BENCH_TEXTS["mmlu"][0].split()
    doc = " ".join(bench_words[:7])  # 恰好 7 个词 ⇒ 0 个 8-gram
    assert len(doc.split()) == 7
    _write_shard(out_root, "src1", "a", [("d1", doc)])
    dct.process_shard_decontam(
        out_root / "s3" / "src1" / "a.parquet", out_root, "src1", "a", index, cfg
    )
    decisions = _read_decisions(out_root, "src1", "a")
    assert decisions["d1"]["decision"] == dct.DECISION_KEEP


def test_normalization_equivalence(tmp_path):
    # 大小写 / 多空格 / 换行差异不影响命中（NFKC+casefold+空白折叠归一）
    _, index = _build_index(tmp_path)
    cfg = _cfg(tmp_path)
    out_root = tmp_path / "out"
    snippet = " ".join(FAKE_BENCH_TEXTS["math"][0].split()[10:24])
    loud = "  " + snippet.upper().replace(" ", "    ") + "\n\n"
    _write_shard(out_root, "src1", "a", [("d1", loud), ("d2", "nothing to see here")])
    dct.process_shard_decontam(
        out_root / "s3" / "src1" / "a.parquet", out_root, "src1", "a", index, cfg
    )
    decisions = _read_decisions(out_root, "src1", "a")
    assert decisions["d1"]["decision"] == dct.DECISION_DROP
    assert decisions["d1"]["hit_benchmark"] == "math"


def test_shard_idempotent_byte_identical(tmp_path):
    _, index = _build_index(tmp_path)
    cfg = _cfg(tmp_path)
    out_root = tmp_path / "out"
    docs = [("d1", _overlap_doc(FAKE_BENCH_TEXTS["humaneval"][0])), ("d2", "clean doc")]
    path = _write_shard(out_root, "src1", "a", docs)
    s1 = dct.process_shard_decontam(path, out_root, "src1", "a", index, cfg)
    out_path, summary_path = dct.decontam_output_paths(out_root, "src1", "a")
    bytes1 = out_path.read_bytes()
    s2 = dct.process_shard_decontam(path, out_root, "src1", "a", index, cfg)
    assert s2["skipped"] is True
    assert out_path.read_bytes() == bytes1
    assert s1["decisions"] == s2["decisions"]


def test_shard_idempotency_invalidates_on_config_or_input_change(tmp_path):
    _, index = _build_index(tmp_path)
    cfg = _cfg(tmp_path)
    out_root = tmp_path / "out"
    path = _write_shard(out_root, "src1", "a", [("d1", "clean doc")])
    first = dct.process_shard_decontam(path, out_root, "src1", "a", index, cfg)
    assert first["skipped"] is False

    changed_cfg = {**cfg, "max_shingles_per_doc": 1234}
    second = dct.process_shard_decontam(
        path, out_root, "src1", "a", index, changed_cfg
    )
    assert second["skipped"] is False
    assert second["config_sha256"] != first["config_sha256"]

    _write_shard(out_root, "src1", "a", [("d1", "changed clean document")])
    third = dct.process_shard_decontam(
        path, out_root, "src1", "a", index, changed_cfg
    )
    assert third["skipped"] is False
    assert third["input_fingerprint"] != second["input_fingerprint"]


def test_migrate_legacy_summary_binds_unchanged_source_lineage(tmp_path):
    cfg = _cfg(tmp_path)
    _, index = _build_index(tmp_path, cfg)
    out_root = tmp_path / "out"
    in_path = _write_shard(out_root, "src1", "a", [("d1", "clean doc")])
    fresh = dct.process_shard_decontam(in_path, out_root, "src1", "a", index, cfg)
    out_path, summary_path = dct.decontam_output_paths(out_root, "src1", "a")
    legacy = dict(fresh)
    for key in ("input_fingerprint", "config_sha256", "index_sha256", "effective_min_distinct_hits"):
        legacy.pop(key, None)
    summary_path.write_text(json.dumps(legacy), encoding="utf-8")

    migrated = dct.migrate_legacy_summary(
        in_path, out_path, summary_path, cfg, "src1"
    )
    assert migrated["lineage_migration"]["method"] == "legacy_structural_binding"
    assert dct.validate_published_summary(in_path, out_path, migrated, cfg, "src1") is None


def test_migrate_legacy_summary_refuses_changed_source_policy(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["source_min_distinct_hits"] = {"src1": {"ceval": 2}}
    _, index = _build_index(tmp_path, cfg)
    out_root = tmp_path / "out"
    in_path = _write_shard(out_root, "src1", "a", [("d1", "clean doc")])
    dct.process_shard_decontam(in_path, out_root, "src1", "a", index, cfg)
    out_path, summary_path = dct.decontam_output_paths(out_root, "src1", "a")
    with pytest.raises(ValueError, match="must be rescanned"):
        dct.migrate_legacy_summary(in_path, out_path, summary_path, cfg, "src1")


def test_validate_accepts_proven_s3_decision_only_refresh(tmp_path):
    from yuformer_data.scale.dedup_common import fingerprint

    cfg = _cfg(tmp_path)
    _, index = _build_index(tmp_path, cfg)
    out_root = tmp_path / "out"
    in_path = _write_shard(out_root, "src1", "a", [("d1", "clean doc")])
    summary = dct.process_shard_decontam(in_path, out_root, "src1", "a", index, cfg)
    old_fp = fingerprint(in_path)
    table = pq.read_table(in_path).append_column("s3_decision", pa.array(["keep"]))
    time.sleep(0.001)
    pq.write_table(table, in_path)
    new_fp = fingerprint(in_path)
    s3_summary = in_path.with_name("a.summary.json")
    s3_summary.write_text(
        json.dumps(
            {
                "ndv2_refresh": {
                    "transform_contract": {
                        "method": "stream_replace_s3_decision_v2",
                        "input_fingerprint": old_fp,
                        "output_fingerprint": new_fp,
                        "preserved_columns": ["doc_id", "text"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    out_path, _ = dct.decontam_output_paths(out_root, "src1", "a")
    assert dct.validate_published_summary(in_path, out_path, summary, cfg, "src1") is None


def test_truncation_semantics(tmp_path):
    # cap=50：命中 8-gram 位于截断前 → drop；位于截断后（文档尾部）→ 判定性 miss，keep
    cfg = _cfg(tmp_path, max_shingles_per_doc=50)
    _, index = _build_index(tmp_path, cfg)
    out_root = tmp_path / "out"
    tail_words = " ".join(FAKE_BENCH_TEXTS["gsm8k"][0].split()[:12])
    filler = " ".join(f"fillerword{i:05d}" for i in range(300))  # 300-8+1=293 词组 > 50
    head_hit = tail_words + " " + filler           # 命中在前 50 词组内
    tail_hit = filler + " " + tail_words           # 命中词组索引 ~293+，被截断
    _write_shard(out_root, "src1", "a", [("head", head_hit), ("tail", tail_hit)])
    summary = dct.process_shard_decontam(
        out_root / "s3" / "src1" / "a.parquet", out_root, "src1", "a", index, cfg
    )
    decisions = _read_decisions(out_root, "src1", "a")
    assert decisions["head"]["decision"] == dct.DECISION_DROP
    assert decisions["tail"]["decision"] == dct.DECISION_KEEP  # 判定性截断：尾部不扫
    assert summary["truncated_docs"] == 2


def test_full_scan_detects_overlap_beyond_legacy_prefix_cap(tmp_path):
    cfg = _cfg(tmp_path, max_shingles_per_doc=None)
    _, index = _build_index(tmp_path, cfg)
    out_root = tmp_path / "out"
    benchmark = " ".join(FAKE_BENCH_TEXTS["gsm8k"][0].split()[:12])
    filler = " ".join(f"fillerword{i:05d}" for i in range(300))
    _write_shard(out_root, "src1", "a", [("tail", filler + " " + benchmark)])
    summary = dct.process_shard_decontam(
        out_root / "s3" / "src1" / "a.parquet", out_root, "src1", "a", index, cfg
    )
    decisions = _read_decisions(out_root, "src1", "a")
    assert decisions["tail"]["decision"] == dct.DECISION_DROP
    assert decisions["tail"]["hit_benchmark"] == "gsm8k"
    assert summary["truncated_docs"] == 0


def test_prefix_to_full_scan_recomputes_only_previously_truncated_docs(tmp_path):
    old_cfg = _cfg(tmp_path, max_shingles_per_doc=50)
    _, index = _build_index(tmp_path, old_cfg)
    out_root = tmp_path / "out"
    benchmark = " ".join(FAKE_BENCH_TEXTS["gsm8k"][0].split()[:12])
    filler = " ".join(f"fillerword{i:05d}" for i in range(300))
    docs = [
        ("short", "a short unrelated document"),
        ("head", benchmark + " " + filler),
        ("tail", filler + " " + benchmark),
    ]
    in_path = _write_shard(out_root, "src1", "a", docs)
    first = dct.process_shard_decontam(in_path, out_root, "src1", "a", index, old_cfg)
    assert first["decisions"] == {"keep": 2, "drop_contaminated": 1}

    full_cfg = dict(old_cfg)
    full_cfg["max_shingles_per_doc"] = None
    full_cfg["monotonic_from_max_shingles_per_doc"] = 50
    second = dct.process_shard_decontam(in_path, out_root, "src1", "a", index, full_cfg)
    decisions = _read_decisions(out_root, "src1", "a")
    assert decisions["short"]["decision"] == dct.DECISION_KEEP
    assert decisions["head"]["decision"] == dct.DECISION_DROP
    assert decisions["tail"]["decision"] == dct.DECISION_DROP
    assert second["truncated_docs"] == 0
    assert second["incremental_rescan"]["method"] == "legacy_prefix_to_full_document"
    assert second["incremental_rescan"]["recomputed_rows"] == 2
    assert second["incremental_rescan"]["previous_decisions"] == first["decisions"]
    assert second["incremental_rescan"]["newly_dropped"] == 1
    assert second["scan_policy"]["transition"]["monotonic_effect"] == "only_add_drops"


def test_fast_ascii_cap_probe_matches_legacy_shingle_truncation():
    docs = [
        "short words only",
        "punctuation, tabs\tand\nnewlines! " * 40,
        " ".join(f"word{i}" for i in range(200)),
    ]
    for cap in (5, 50, 150):
        for text in docs:
            _hashes, truncated = dct.doc_shingle_hashes(text, N, cap)
            assert dct.exceeds_shingle_cap(text, N, cap) == truncated

    # Exact character lower bound also holds for mixed CJK/Latin tokenization.
    assert not dct.exceeds_shingle_cap("中英 mixed", N, 50)


def test_fullscan_transition_audit_proves_monotonic_drop_delta():
    def summary(dropped, truncated, per_source, full=False):
        policy = (
            {"scan_policy": {"scope": "full_document", "max_shingles_per_doc": None}}
            if full else {}
        )
        return {
            "stage": "decontam",
            "shard_errors": 0,
            "coverage": {"expected": 2, "complete": 2},
            "totals": {"in_rows": 100, "out_rows": 100, "truncated_docs": truncated},
            "decisions": {dct.DECISION_DROP: dropped, dct.DECISION_KEEP: 100 - dropped},
            "per_source": per_source,
            "publication_contract": {"index_sha256": "index", **policy},
        }

    before = summary(7, 3, {"src": {"hit.a": 7}})
    after = summary(9, 0, {"src": {"hit.a": 8, "hit.b": 1}}, full=True)
    result = dct.audit_fullscan_transition(before, after)
    assert result["status"] == "PASS"
    assert result["newly_dropped"] == 2
    assert result["per_source"]["src"]["newly_dropped"] == 2


def test_decontam_config_accepts_null_cap(tmp_path):
    path = tmp_path / "decontam.yaml"
    path.write_text(
        "ngram_n: 8\nmax_shingles_per_doc: null\nindex_dir: index\n"
        "benchmarks:\n  - name: fake\n    repo: fake/repo\n"
        "    paths: [x.parquet]\n    fields: [text]\n",
        encoding="utf-8",
    )
    cfg = dct.load_decontam_config(path)
    assert cfg["max_shingles_per_doc"] is None


def test_failed_shard_no_output(tmp_path):
    # E-16：失败分片只落 error summary，parquet 空产物不落盘
    _, index = _build_index(tmp_path)
    cfg = _cfg(tmp_path)
    out_root = tmp_path / "out"
    path = _write_shard(out_root, "src1", "a", [("d1", "x")])
    path.unlink()
    summary = dct.process_shard_decontam(path, out_root, "src1", "a", index, cfg)
    assert summary["errors"]
    out_path, _ = dct.decontam_output_paths(out_root, "src1", "a")
    assert not out_path.exists()


def test_short_docs_counted(tmp_path):
    _, index = _build_index(tmp_path)
    cfg = _cfg(tmp_path)
    out_root = tmp_path / "out"
    _write_shard(out_root, "src1", "a", [("d1", "tiny"), ("d2", "word " * 20)])
    summary = dct.process_shard_decontam(
        out_root / "s3" / "src1" / "a.parquet", out_root, "src1", "a", index, cfg
    )
    assert summary["short_docs"] == 1
    assert summary["decisions"] == {"keep": 2}


def test_determinism_across_processes(tmp_path):
    # 词缓存清空/跨进程不影响输出：同输入两次独立进程扫描字节一致
    _, index = _build_index(tmp_path)
    cfg = _cfg(tmp_path)
    out_root = tmp_path / "out"
    docs = [
        ("d1", _overlap_doc(FAKE_BENCH_TEXTS["mmlu"][0])),
        ("d2", "clean gardening document " * 30),
    ]
    path = _write_shard(out_root, "src1", "a", docs)
    dct.process_shard_decontam(path, out_root, "src1", "a", index, cfg)
    out_path, summary_path = dct.decontam_output_paths(out_root, "src1", "a")
    bytes1 = out_path.read_bytes()
    dct._WORD_CACHE.clear()
    out_path.unlink()
    summary_path.unlink()
    dct.process_shard_decontam(path, out_root, "src1", "a", index, cfg)
    assert out_path.read_bytes() == bytes1


def test_global_summary_aggregates(tmp_path):
    _, index = _build_index(tmp_path)
    cfg = _cfg(tmp_path)
    out_root = tmp_path / "out"
    _write_shard(out_root, "src1", "a", [("d1", _overlap_doc(FAKE_BENCH_TEXTS["math"][0]))])
    _write_shard(out_root, "src1", "b", [("d2", "clean")])
    for rel in ("a", "b"):
        dct.process_shard_decontam(
            out_root / "s3" / "src1" / f"{rel}.parquet",
            out_root, "src1", rel, index, cfg,
        )
    summaries = [
        json.loads(p.read_text())
        for p in sorted((out_root / "decontam" / "src1").glob("*.summary.json"))
    ]
    path = dct.write_global_summary(out_root, summaries)
    payload = json.loads(path.read_text())
    assert payload["shard_errors"] == 0
    assert payload["hit_benchmarks"] == {"math": 1}
    assert payload["decisions"] == {"drop_contaminated": 1, "keep": 1}
    assert payload["per_source"]["src1"]["in_rows"] == 2


def test_global_summary_rebuild_uses_full_published_roster(tmp_path):
    _, index = _build_index(tmp_path)
    cfg = _cfg(tmp_path)
    out_root = tmp_path / "out"
    for source in ("src1", "src2"):
        _write_shard(out_root, source, "a", [(source, "clean text")])
        dct.process_shard_decontam(
            out_root / "s3" / source / "a.parquet",
            out_root, source, "a", index, cfg,
        )
    published, coverage = dct.collect_published_summaries(out_root, cfg)
    payload = json.loads(
        dct.write_global_summary(out_root, published, coverage=coverage).read_text()
    )
    assert payload["coverage"] == {
        "expected": 2, "complete": 2, "missing_summary": 0, "missing_output": 0,
    }
    assert set(payload["per_source"]) == {"src1", "src2"}


# ---------------------------------------------------------------- 真实基准下载（不可用时自动 skip）

def _hf_reachable() -> bool:
    try:
        req = urllib.request.Request(
            "https://hf-mirror.com/api/datasets/openai/gsm8k",
            headers={"User-Agent": "yuformer-decontam-test"},
        )
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        return False


REAL_BENCH = pytest.mark.skipif(not _hf_reachable(), reason="hf-mirror 不可达，跳过真实下载测试")


@REAL_BENCH
def test_real_benchmark_download_index(tmp_path):
    # 端到端真实下载（gsm8k + humaneval，小基准）验证 fields/下载/索引链路；
    # mmlu/math/ceval 的可用性 2026-09-25 已人工验证（HTTP 206 + schema 实读），构建时逐路径容错。
    cfg = _cfg(
        tmp_path,
        benchmarks=[
            {"name": "gsm8k", "repo": "openai/gsm8k",
             "paths": ["main/test-00000-of-00001.parquet"],
             "fields": ["question", "answer"], "clean": ["strip_calculator_annotations"]},
            {"name": "humaneval", "repo": "openai/openai_humaneval",
             "paths": ["openai_humaneval/test-00000-of-00001.parquet"],
             "fields": ["prompt", "canonical_solution", "test"]},
        ],
    )
    payload = dct.build_benchmark_index(cfg, Path(cfg["index_dir"]))
    assert payload.get("errors", {}) == {}
    per = payload["per_benchmark"]
    assert per["gsm8k"]["items"] == 1319
    assert per["humaneval"]["items"] == 164
    index = dct.BenchmarkIndex(Path(cfg["index_dir"]))
    # 真实题面命中：复制 gsm8k 第一题正文必 drop 且基准名正确
    import os

    from huggingface_hub import hf_hub_download

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    local = hf_hub_download(
        repo_id="openai/gsm8k", repo_type="dataset", revision="main",
        filename="main/test-00000-of-00001.parquet",
        local_dir=str(Path(cfg["index_dir"]) / "repo_cache"),
    )
    first_q = pq.read_table(local).column("question").to_pylist()[0]
    out_root = tmp_path / "out"
    _write_shard(out_root, "src1", "a", [("d1", first_q)])
    dct.process_shard_decontam(
        out_root / "s3" / "src1" / "a.parquet", out_root, "src1", "a", index, cfg
    )
    decisions = _read_decisions(out_root, "src1", "a")
    assert decisions["d1"]["decision"] == dct.DECISION_DROP
    assert decisions["d1"]["hit_benchmark"] == "gsm8k"


# ---------------------------------------------------------------- 吞吐实测（合成语料，单核）

def test_throughput_single_core(tmp_path):
    _, index = _build_index(tmp_path)
    cfg = _cfg(tmp_path)
    out_root = tmp_path / "out"
    rng = np.random.default_rng(0)
    vocab = np.array([f"w{i:05d}" for i in range(20000)])
    docs = []
    for i in range(20000):
        words = vocab[rng.integers(0, len(vocab), size=300)]
        docs.append((f"t{i:06d}", " ".join(words.tolist())))
    path = _write_shard(out_root, "src1", "big", docs)
    started = time.monotonic()
    summary = dct.process_shard_decontam(path, out_root, "src1", "big", index, cfg)
    elapsed = time.monotonic() - started
    rate = summary["in_rows"] / max(elapsed, 1e-9)
    print(f"\n[throughput] 20k docs x ~300 words: {rate:,.0f} docs/s/core "
          f"({rate * 300 / 1e6:.1f}M shingles/s/core)")
    assert rate > 500  # 机器高负载（GPU 推理+fasttext 训练占 ~130 核）+ 并发测试下的保守下限；
    # 正常单跑实测 6.9k-8.5k docs/s/核（见 README 实测表）


# ---------------------------------------------------------------- E-22 回归：词缓存 clear 时序 bug

def test_word_cache_eviction_no_keyerror(monkeypatch):
    """复现生产事故：缓存超 CAP 触发 clear 后，本批老词（不在 missing）KeyError。"""
    monkeypatch.setattr(dct, "_WORD_CACHE_CAP", 10)
    dct._WORD_CACHE.clear()
    words_a = [f"w{i}" for i in range(10)]
    h_a = dct.word_hashes(words_a)
    assert len(h_a) == 10
    assert all(w in dct._WORD_CACHE for w in words_a)
    # 本批含老词 + 新词，填充后超 CAP(10) → clear + 全批重算，不得 KeyError
    words_b = [f"w{i}" for i in range(8)] + ["new1", "new2", "new3"]
    h_b = dct.word_hashes(words_b)
    assert len(h_b) == 11
    # 哈希必须与无缓存语义一致：FNV-1a-64 只依赖词本身
    for w, h in zip(words_b, h_b.tolist()):
        ref = dct._fnv1a64_batch([w])[0]
        assert h == ref
    # 缓存语义：clear 后缓存恰好 = 本批词
    assert set(dct._WORD_CACHE) == set(words_b)
    # 全命中批（missing 为空）超 CAP 也必须正常重算而非 KeyError
    words_c = words_b + ["new4"]
    h_c = dct.word_hashes(words_c)
    assert len(h_c) == 12
    dct._WORD_CACHE.clear()
