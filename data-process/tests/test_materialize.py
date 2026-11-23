from __future__ import annotations

import hashlib
import json
import sys
import threading
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from yuformer_data.scale import length_bucket as lb
from yuformer_data.scale import materialize as mz
from yuformer_data.training import allocate_mix_quotas, assign_context_bucket

PROFILES = {
    "src-books": {"domain": "books", "extensions": [".parquet"]},
    "src-math": {"domain": "math", "extensions": [".parquet"]},
}


def _guard_factory(stage: str) -> tuple:
    return (lambda: None, threading.Event())


def _doc_ids(n: int, tag: str) -> list[str]:
    return [hashlib.blake2b(f"{tag}-{i}".encode(), digest_size=16).hexdigest() for i in range(n)]


def _mix_cfg(dir_path: Path, **over) -> Path:
    cfg = {
        "mix_name": "mix_t",
        "total_tokens": 1000,
        "domains": {
            "books": {"weight": 0.6, "top_k": 0.5},
            "math": {"weight": 0.4, "top_k": 1.0},
        },
        "curriculum": {
            "enabled": True,
            "method": "upstream_score_legacy",
            "stage1_fraction": 0.25,
            "experimental": True,
        },
        "val": {"tokens_per_domain": 250, "max_fraction": 0.5},
    }
    cfg.update(over)
    dir_path.mkdir(parents=True, exist_ok=True)
    path = dir_path / "mix.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _summ(root: Path, rel: str, payload: dict) -> None:
    p = root / (rel + ".summary.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def build_pool(root: Path) -> dict:
    """合成两域打分池（books 10 docs ×100 tok，math 8 docs ×50 tok）。

    返回 {domain: {"doc_ids","nt","comp","up","source"}} 供断言。"""
    spec = {
        "books": {"source": "src-books", "nt": 100, "n": 10,
                  "comp": [0.05 * (i + 1) for i in range(10)],
                  "up": [float(i) for i in range(10)]},
        "math": {"source": "src-math", "nt": 50, "n": 8,
                 "comp": [0.1 * (i + 1) for i in range(8)],
                 "up": [None] * 8},
    }
    info: dict[str, dict] = {}
    for dom, sp in spec.items():
        src, rel = sp["source"], "shard0"
        ids = _doc_ids(sp["n"], dom)
        info[dom] = {"doc_ids": ids, "nt": [sp["nt"]] * sp["n"],
                     "comp": sp["comp"], "up": sp["up"], "source": src}
        # s3 输入分片（gate 覆盖检查的对象；内容不参与 S6 计算）
        s3_path = root / "s3" / src / (rel + ".parquet")
        s3_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"x": [1]}), s3_path)
        _summ(root, f"s3/{src}/{rel}", {"stage": "s3", "errors": {}})
        # s4_scored（composite 分来源 + 全轨分数列——goldrule 研究子集消费）
        s4_path = root / "s4_scored" / src / (rel + ".parquet")
        s4_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "doc_id": ids,
                    "domain": [dom] * len(ids),
                    "self_score": np.array(sp["comp"], dtype=np.float32),
                    "composite": np.array(sp["comp"], dtype=np.float32),
                    "composite_v2": np.array(sp["comp"], dtype=np.float32),
                    "upstream_score": pa.array(sp["up"], type=pa.float64()),
                    "upstream_norm": pa.array(sp["up"], type=pa.float32()),
                    "fasttext_score": np.array(sp["comp"], dtype=np.float32),
                    "bert_score": pa.array(
                        sp["comp"] if dom == "books" else [None] * len(ids),
                        type=pa.float32(),
                    ),
                    "rule_composite": np.array(sp["comp"], dtype=np.float32),
                    "rule_score": np.array(sp["comp"], dtype=np.float32),
                    "hybrid_composite": np.array(sp["comp"], dtype=np.float32),
                    "llm_distill": np.array(sp["comp"], dtype=np.float32),
                    "hybrid_v2": np.array(sp["comp"], dtype=np.float32),
                }
            ),
            s4_path,
        )
        _summ(root, f"s4_scored/{src}/{rel}", {"stage": "s4_score", "errors": {}})
        # s5 计长产物
        s5_path = root / "s5" / src / (rel + ".parquet")
        s5_path.parent.mkdir(parents=True, exist_ok=True)
        ctx = lb.load_context_lengths()
        pq.write_table(
            pa.table(
                {
                    "doc_id": ids,
                    "n_tokens": np.array([sp["nt"]] * len(ids), dtype=np.uint32),
                    "length_bucket": [assign_context_bucket(sp["nt"], ctx) for _ in ids],
                    "mix_eligibility": [lb.eligibility_flags(sp["nt"], ctx) for _ in ids],
                },
                schema=lb.S5_SCHEMA,
            ),
            s5_path,
        )
        _summ(
            root, f"s5/{src}/{rel}",
            {"stage": "s5", "domain": dom, "in_rows": len(ids), "out_rows": len(ids),
             "total_tokens": sp["nt"] * len(ids),
             "decisions": {"keep": len(ids)}, "errors": {}},
        )
        # decontam 决策表（全 keep）
        d_path = root / "decontam" / src / (rel + ".parquet")
        d_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {"doc_id": ids,
                 "decision": ["keep"] * len(ids),
                 "hit_benchmark": [""] * len(ids)}
            ),
            d_path,
        )
        _summ(root, f"decontam/{src}/{rel}", {"stage": "decontam", "errors": {}})
        # 分位表（s4 quantile build 产物：域内 composite_v2 全量排序）
        tdir = root / "s4_quantiles" / "tables"
        tdir.mkdir(parents=True, exist_ok=True)
        np.save(tdir / f"{dom}.npy", np.sort(np.array(sp["comp"], dtype=np.float32)))
    tables_meta = {
        dom: {"npy": f"{dom}.npy", "n": len(spec[dom]["comp"])}
        for dom in spec
    }
    total_rows = sum(sp["n"] for sp in spec.values())
    (root / "s4_quantiles" / "tables" / "_build.summary.json").write_text(
        json.dumps({
            "score_field": "composite",
            "pair_shards": len(spec),
            "total_rows": total_rows,
            "domains": tables_meta,
        }, sort_keys=True), encoding="utf-8"
    )
    # 全局 summaries（shard_errors=0 是硬门条件）
    (root / "stage_s4_global_summary.json").write_text(
        json.dumps({
            "shard_errors": 0,
            "score_shards": len(spec),
            "quantile_shards": len(spec),
            "total_rows": total_rows,
            "quantile_rows": total_rows,
        }), encoding="utf-8"
    )
    (root / "stage_decontam_global_summary.json").write_text(
        json.dumps({
            "stage": "decontam",
            "shard_errors": 0,
            "coverage": {
                "expected": len(spec), "complete": len(spec),
                "missing_summary": 0, "missing_output": 0,
            },
            "totals": {
                "in_rows": total_rows, "out_rows": total_rows,
                "truncated_docs": 0,
            },
            "publication_contract": {
                "scan_policy": {
                    "scope": "full_document", "max_shingles_per_doc": None,
                }
            },
        }, sort_keys=True),
        encoding="utf-8",
    )
    (root / "stage_ndv2_global_summary.json").write_text(
        '{"shard_errors": 0, "per_domain": {}}', encoding="utf-8"
    )
    per_dom = {}
    for dom, sp in spec.items():
        per_dom[dom] = {
            "out_rows": sp["n"],
            "total_tokens": sp["nt"] * sp["n"],
            "decisions.drop_decontam": 0,
            "decisions.drop_s2_exact_dup": 0,
            "decisions.drop_s3_near_dup": 0,
        }
    s5_payload = {"shard_errors": 0, "per_domain": per_dom}
    s5_payload["input_contract"] = lb.s5_global_input_contract(
        root, lb.resolve_tokenizer_path()
    )
    (root / "stage_s5_global_summary.json").write_text(
        json.dumps(s5_payload, sort_keys=True), encoding="utf-8"
    )
    return info


# ---------------------------------------------------------------- 最大余数法

def test_largest_remainder_quotas_numeric():
    q = allocate_mix_quotas({"x": 1.0, "y": 1.0, "z": 1.0}, 100)
    assert q == {"x": 34, "y": 33, "z": 33}
    q = allocate_mix_quotas({"a": 0.25, "b": 0.25, "c": 0.5}, 10)
    assert q == {"a": 3, "b": 2, "c": 5}
    # M2 权重：配额和恰等于总量；+1 余数分配与纯 Python 参考实现一致
    weights = {"web": 0.729, "code": 0.125, "math": 0.042, "chinese": 0.052,
               "academic": 0.031, "books": 0.008, "encyclopedia": 0.004,
               "qa": 0.004, "multilingual": 0.005}
    total = 1_440_000_000_000
    q = allocate_mix_quotas(weights, total)
    assert sum(q.values()) == total
    assert abs(q["web"] - 0.729 * total) <= 2
    wsum = sum(weights.values())
    exact = {d: total * w / wsum for d, w in weights.items()}
    floors = {d: int(np.floor(v)) for d, v in exact.items()}
    remaining = total - sum(floors.values())
    expect_plus = set(
        sorted(weights, key=lambda d: (-(exact[d] - floors[d]), d))[:remaining]
    )
    got_plus = {d for d in weights if q[d] == floors[d] + 1}
    assert got_plus == expect_plus


def test_formal_source_roster_is_authoritative_and_in_identity(tmp_path: Path):
    base_path = _mix_cfg(tmp_path / "base")
    scoped_path = _mix_cfg(tmp_path / "scoped", included_sources=["src-books"])
    base = mz.load_mix_config(base_path)
    scoped = mz.load_mix_config(scoped_path)

    assert mz.config_echo(base) != mz.config_echo(scoped)
    assert mz._resolve_release_sources(scoped, PROFILES, None) == ["src-books"]
    assert mz._resolve_release_sources(scoped, PROFILES, ["src-books"]) == ["src-books"]
    with pytest.raises(ValueError, match="exactly match"):
        mz._resolve_release_sources(scoped, PROFILES, ["src-math"])


def test_source_scoped_pool_ledger_excludes_other_sources(tmp_path: Path):
    root = tmp_path / "out"
    build_pool(root)
    tasks = mz.s5_tasks(root, PROFILES, sources=["src-books"])
    pool = mz._pool_stats_for_tasks(root, tasks)

    assert pool == {
        "books": {
            "docs": 10,
            "tokens": 1000,
            "decisions.keep": 10,
        }
    }


def test_source_scoped_s6_publishes_only_configured_roster(tmp_path: Path):
    root = tmp_path / "out"
    build_pool(root)
    cfg_path = _mix_cfg(
        tmp_path,
        included_sources=["src-books"],
        total_tokens=400,
        domains={"books": {"weight": 1.0, "top_k": 1.0}},
        val={"tokens_per_domain": 100, "max_fraction": 0.1},
    )
    summary = _run_s6(root, cfg_path)
    manifest = json.loads(
        (root / "mix_order" / "mix_t.manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["included_sources"] == ["src-books"]
    assert summary["included_sources"] == ["src-books"]
    assert set(summary["per_domain"]) == {"books"}
    assert summary["per_domain"]["books"]["pool_docs"] == 10
    assert summary["per_domain"]["books"]["pool_tokens"] == 1000


# ---------------------------------------------------------------- 硬门 fail-fast

def test_gate_fail_fast_each_gate(tmp_path: Path):
    root = tmp_path / "out"
    build_pool(root)
    cfg = mz.load_mix_config(_mix_cfg(tmp_path))

    # 全齐 → 无错误
    assert mz.gate_errors(root, PROFILES, cfg) == {}

    # ① s4_scored 缺分片
    o1 = tmp_path / "o1"
    build_pool(o1)
    (o1 / "s4_scored" / "src-books" / "shard0.parquet").unlink()
    errors = mz.gate_errors(o1, PROFILES, cfg)
    assert any("①" in k for k in errors), errors

    # ① 分位表 meta 缺领域
    o2 = tmp_path / "o2"
    build_pool(o2)
    meta = json.loads((o2 / "s4_quantiles" / "tables" / "_build.summary.json").read_text())
    del meta["domains"]["math"]
    (o2 / "s4_quantiles" / "tables" / "_build.summary.json").write_text(
        json.dumps(meta), encoding="utf-8"
    )
    errors = mz.gate_errors(o2, PROFILES, cfg)
    assert any("①" in k and "math" in "".join(v) for k, v in errors.items())

    # ② decontam error shard
    o3 = tmp_path / "o3"
    build_pool(o3)
    (o3 / "stage_decontam_global_summary.json").write_text('{"shard_errors": 3}')
    errors = mz.gate_errors(o3, PROFILES, cfg)
    assert any("②" in k for k in errors), errors

    # ⑤ s5 缺分片
    o4 = tmp_path / "o4"
    build_pool(o4)
    (o4 / "s5" / "src-math" / "shard0.parquet").unlink()
    errors = mz.gate_errors(o4, PROFILES, cfg)
    assert any("⑤" in k for k in errors), errors

    # ⑤ 全局 summary 存在但绑定旧上游，同样必须拒绝。
    o4b = tmp_path / "o4b"
    build_pool(o4b)
    stale = json.loads((o4b / "stage_s5_global_summary.json").read_text())
    stale["input_contract"]["tokenizer_sha256"] = "stale"
    (o4b / "stage_s5_global_summary.json").write_text(json.dumps(stale))
    errors = mz.gate_errors(o4b, PROFILES, cfg)
    assert any("⑤" in k and "input_contract" in "".join(v) for k, v in errors.items())

    # ③ ndv2 E-17 占位（web 域）拒收；policy 刷新后 ③ 通过
    o5 = tmp_path / "o5"
    (o5 / "s3" / "src-web").mkdir(parents=True)
    pq.write_table(pa.table({"x": [1]}), o5 / "s3" / "src-web" / "shard0.parquet")
    (o5 / "s3_decisions").mkdir(parents=True)
    (o5 / "s3_decisions" / "web.summary.json").write_text(
        json.dumps({"policy": "near_dup_clustering_skipped"}), encoding="utf-8"
    )
    profiles_web = {"src-web": {"domain": "web", "extensions": [".parquet"]}}
    errors = mz.gate_errors(o5, profiles_web, cfg)
    assert any("③" in k and "E-17" in "".join(v) for k, v in errors.items())
    edge_fp = [512, 12345]
    (o5 / "s3_decisions" / "web.summary.json").write_text(
        json.dumps({
            "policy": "ndv2_minhash128_bbit4",
            "edges_fingerprint": edge_fp,
            "decision_contract": {"method": "edge_dominance_external"},
            "dropped": 1,
            "errors": {},
        }), encoding="utf-8"
    )
    (o5 / "s3_v2").mkdir(parents=True)
    (o5 / "s3_v2" / "web_minhash_precision_audit.json").write_text(
        json.dumps({
            "passed": True,
            "errors": {},
            "audit_contract": {"edges_fingerprint": edge_fp},
        }), encoding="utf-8"
    )
    lookup = o5 / "s3_lookup" / "web"
    lookup.mkdir(parents=True)
    (lookup / "lookup.ids.bin").write_bytes(b"0" * 16)
    (lookup / "lookup.codes.bin").write_bytes(b"\x01")
    (o5 / "stage_ndv2_global_summary.json").write_text(
        json.dumps({
            "shard_errors": 0,
            "task_errors": 0,
            "per_domain": {"web": {"dropped": 1}},
        }),
        encoding="utf-8",
    )
    errors = mz.gate_errors(o5, profiles_web, cfg)
    assert not any("③" in k for k in errors), errors


def test_assert_gates_report_all(tmp_path: Path):
    root = tmp_path / "out"
    build_pool(root)
    cfg = mz.load_mix_config(_mix_cfg(tmp_path))
    (root / "stage_decontam_global_summary.json").write_text('{"shard_errors": 1}')
    (root / "s5" / "src-books" / "shard0.parquet").unlink()
    with pytest.raises(SystemExit) as exc:
        mz.assert_gates(mz.gate_errors(root, PROFILES, cfg))
    text = str(exc.value)
    assert "②" in text and "⑤" in text


def test_curriculum_requires_explicit_experimental_legacy_mode(tmp_path: Path):
    with pytest.raises(ValueError, match="must be marked experimental"):
        mz.load_mix_config(
            _mix_cfg(
                tmp_path,
                curriculum={
                    "enabled": True,
                    "method": "upstream_score_legacy",
                    "stage1_fraction": 0.25,
                },
            )
        )

    cfg = mz.load_mix_config(
        _mix_cfg(
            tmp_path,
            curriculum={"enabled": False, "method": "random", "stage1_fraction": 0.0},
        )
    )
    assert cfg["curriculum"]["method"] == "random"


def test_goldrule_is_disabled_unless_explicitly_configured(tmp_path: Path):
    cfg = mz.load_mix_config(_mix_cfg(tmp_path))
    assert cfg["goldrule"] == {
        "per_domain_quota": 0,
        "sample_rate": 0.0,
        "min_total_rows": 0,
    }


def test_goldrule_hash_sampling_uses_full_uint64_threshold():
    threshold = int(0.1 * (1 << 64))
    assert mz._goldrule_hash_selected(threshold - 1, 0.1)
    assert not mz._goldrule_hash_selected(threshold, 0.1)
    assert not mz._goldrule_hash_selected(0, 0.0)
    assert mz._goldrule_hash_selected((1 << 64) - 1, 1.0)


def test_goldrule_acceptance_floor_requires_an_active_sampler(tmp_path: Path):
    with pytest.raises(ValueError, match="requires positive"):
        mz.load_mix_config(
            _mix_cfg(
                tmp_path,
                goldrule={"per_domain_quota": 0, "sample_rate": 0.1, "min_total_rows": 10},
            )
        )


def test_goldrule_research_exports_full_tracks_and_coverage(tmp_path: Path):
    root = tmp_path / "out"
    build_pool(root)
    cfg_path = _mix_cfg(
        tmp_path,
        goldrule={"per_domain_quota": 100, "sample_rate": 1.0, "min_total_rows": 1},
    )
    summary = _run_s6(root, cfg_path)
    audit = summary["goldrule_research"]
    table = pq.read_table(root / audit["path"])
    assert table.num_rows >= 1
    assert set(mz.GOLDRULE_TRACK_COLUMNS) <= set(table.column_names)
    assert {"source", "sample_hash"} <= set(table.column_names)
    assert table.schema.field("upstream_score").type == pa.float64()
    for name in mz.GOLDRULE_REQUIRED_FINITE:
        assert table[name].null_count == 0
        assert audit["track_coverage"][name]["fraction"] == 1.0
    # The fixture deliberately has no retired BERT or upstream score for math;
    # nullable legacy-track coverage must be measured rather than hidden.
    assert audit["track_coverage"]["bert_score"]["fraction"] < 1.0
    assert audit["track_coverage"]["upstream_norm"]["fraction"] < 1.0
    assert audit["hybrid_v2_semantics"].startswith("effective current_v2")
    manifest = json.loads(
        (root / "mix_order" / "mix_t.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["goldrule_research"] == audit["path"]
    assert manifest["goldrule_research_sha256"] == audit["sha256"]
    assert audit["bytes"] == (root / audit["path"]).stat().st_size

    # The manifest cannot make a generation skippable when C4 was deleted.
    (root / audit["path"]).unlink()
    rebuilt = _run_s6(root, cfg_path)
    assert (root / rebuilt["goldrule_research"]["path"]).is_file()


def test_collect_shard_resumes_only_after_commit_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "out"
    build_pool(root)
    work = tmp_path / "work"
    contract = {
        "schema_version": mz.S6_COLLECT_SCHEMA_VERSION,
        "config_echo": "cfg",
        "upstream_snapshot_sha256": "upstream",
        "task_identity_sha256": "tasks",
        "task_count": 1,
    }
    assert not mz._prepare_collect_work_root(work, contract)
    ctx = {
        "out_root": str(root),
        "work_dir": str(work),
        "score_field": "composite_v2",
        "thresholds": {"books": 0.0},
        "val_ids": {},
        "goldrule_quota": 100,
        "goldrule_rate": 1.0,
        "hybrid_v2_scorer_cfg": mz.load_scorer_config(),
    }
    task = ("src-books", "shard0", "books")
    real_parquet_file = mz.pq.ParquetFile
    s4_opens = 0

    def counted_parquet_file(path, *args, **kwargs):
        nonlocal s4_opens
        if "s4_scored" in str(path):
            s4_opens += 1
        return real_parquet_file(path, *args, **kwargs)

    monkeypatch.setattr(mz.pq, "ParquetFile", counted_parquet_file)
    first = mz._collect_worker((task, ctx))
    assert not first.get("errors")
    assert s4_opens == 1, "C4 tracks must piggyback on the selection scan"
    done = mz._collect_done_path(work, *task)
    assert done.is_file()

    monkeypatch.setattr(
        mz.pq, "ParquetFile", lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("reread"))
    )
    second = mz._collect_worker((task, ctx))
    assert second["skipped"] is True
    assert second["candidates"] == first["candidates"]

    sentinel = work / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    assert mz._prepare_collect_work_root(work, contract)
    assert sentinel.is_file()
    changed = {**contract, "upstream_snapshot_sha256": "new"}
    assert not mz._prepare_collect_work_root(work, changed)
    assert not sentinel.exists()


# ---------------------------------------------------------------- 端到端 + 确定性 + val 互斥

def _run_s6(root: Path, cfg_path: Path) -> dict:
    path = mz.run_s6(root, PROFILES, cfg_path, workers=2, guard_factory=_guard_factory)
    return json.loads(Path(path).read_text("utf-8"))


def test_domain_expansion_is_bounded_before_table_schema_cast(tmp_path, monkeypatch):
    monkeypatch.setattr(mz, "DOMAIN_BUILD_MAX_BYTES", 16)
    domains = ["academic", "中文"]
    codes = np.asarray([0, 1, 0, 1, 0], dtype=np.int64)
    column = mz._domain_array(codes, domains)
    assert column.type == pa.string()
    assert column.num_chunks == 3
    assert all(chunk.buffers()[2].size <= 16 for chunk in column.chunks)
    table = pa.table({"domain": column}, schema=pa.schema([pa.field("domain", pa.string(), nullable=False)]))
    path = tmp_path / "domains.parquet"
    pq.write_table(table, path, row_group_size=2)
    assert pq.read_table(path).column("domain").to_pylist() == [domains[i] for i in codes]
    assert mz._domain_array(np.asarray([], dtype=np.int64), domains).to_pylist() == []


def test_validation_contract_is_shared_across_curriculum_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(mz, "active_domains", lambda *_args: {
        "books": ["src-books"], "math": ["src-math"],
    })
    monkeypatch.setattr(mz, "s5_tasks", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(mz, "_pool_stats_for_tasks", lambda *_args: {
        "books": {"tokens": 10_000}, "math": {"tokens": 10_000},
    })
    monkeypatch.setattr(mz, "_s5_input_snapshot", lambda *_args: {"sha256": "fixed-input"})
    common = {"val": {"tokens_per_domain": 100, "max_fraction": 0.01}}
    first = mz._prepare_val_contract(
        tmp_path, PROFILES, {**common, "seed": "warmup-order-seed"}, None
    )[4]
    second = mz._prepare_val_contract(
        tmp_path, PROFILES, {**common, "seed": "stable-order-seed"}, None
    )[4]
    assert first == second


def test_validation_publication_stays_identical_across_stage_seeds(tmp_path):
    root = tmp_path / "out"
    build_pool(root)
    cfg = mz.load_mix_config(_mix_cfg(tmp_path))
    baseline = None
    for seed in ("yuformer", "warmup-order-seed", "stable-order-seed", "anneal-order-seed"):
        mz.ensure_val_sets(
            root, PROFILES, {**cfg, "seed": seed}, workers=2,
            guard_factory=_guard_factory,
        )
        artifacts = {path.name: path.read_bytes() for path in (root / "val_sets").iterdir()
                     if path.suffix in (".json", ".bin", ".parquet")}
        data_inodes = {path.name: path.stat().st_ino
                       for path in (root / "val_sets").iterdir()
                       if path.suffix in (".bin", ".parquet")}
        assert json.loads(artifacts["_val_global_summary.json"])["seed"] == "yuformer"
        if baseline is None:
            baseline = artifacts, data_inodes
        else:
            assert (artifacts, data_inodes) == baseline


def test_s6_end_to_end(tmp_path: Path):
    root = tmp_path / "out"
    info = build_pool(root)
    cfg_path = _mix_cfg(tmp_path)
    summary = _run_s6(root, cfg_path)

    mix_name = "mix_t"
    members_path = root / "mix_members" / f"{mix_name}.parquet"
    stage1_path = root / "mix_order" / f"{mix_name}_stage1.parquet"
    stage2_path = root / "mix_order" / f"{mix_name}_stage2.parquet"
    manifest_path = root / "mix_order" / f"{mix_name}.manifest.json"
    for p in (members_path, stage1_path, stage2_path, manifest_path,
              root / "stage_s6_global_summary.json",
              root / "val_sets" / "books.parquet", root / "val_sets" / "math.parquet",
              root / "val_sets" / "_val_global_summary.json"):
        assert p.is_file(), p

    members = pq.read_table(members_path).to_pylist()
    val_ids = set(pq.read_table(root / "val_sets" / "books.parquet").column("doc_id").to_pylist())
    val_ids |= set(pq.read_table(root / "val_sets" / "math.parquet").column("doc_id").to_pylist())
    member_ids = {r["doc_id"] for r in members}
    # val 互斥：val 与 train 无交集
    assert not (val_ids & member_ids)

    # books：top_k=0.5 → 候选为 composite 最高的 5 篇（阈值来自分位表），
    # 去掉 val 后在真实 top-k 候选中按质量序取不超过 quota=600 的完整文档前缀。
    books_pool = {d: c for d, c in zip(info["books"]["doc_ids"], info["books"]["comp"])}
    books_sel = [r for r in members if r["domain"] == "books"]
    books_val = val_ids & set(info["books"]["doc_ids"])
    avail = {d: c for d, c in books_pool.items() if d not in books_val}
    sel_ids = {r["doc_id"] for r in books_sel}
    rest = [c for d, c in avail.items() if d not in sel_ids]
    sel_comp = sorted((r["composite"] for r in books_sel), reverse=True)
    if rest and sel_comp:
        assert min(sel_comp) >= max(rest) - 1e-6, "选取必须是可用池的质量 top"
    assert summary["per_domain"]["books"]["selected_tokens"] == sum(r["n_tokens"] for r in books_sel)
    assert summary["per_domain"]["books"]["selected_tokens"] <= 600
    assert summary["per_domain"]["books"]["quality_truncated_tokens"] >= 0

    # 课程两段：token/篇数守恒 + stage1 内 books upstream 升序（edu 低→高）
    s1 = pq.read_table(stage1_path).to_pylist()
    s2 = pq.read_table(stage2_path).to_pylist()
    total_tok = sum(r["n_tokens"] for r in members)
    assert sum(r["n_tokens"] for r in s1) + sum(r["n_tokens"] for r in s2) == total_tok
    assert len(s1) + len(s2) == len(members)
    s1_books_up = [r["upstream_score"] for r in s1 if r["domain"] == "books"]
    assert s1_books_up == sorted(s1_books_up)
    # stage1 ≈ 25% tokens
    s1_tok = sum(r["n_tokens"] for r in s1)
    assert 0.20 * total_tok <= s1_tok <= 0.30 * total_tok

    # manifest echo 与全局 summary 一致；summary 的 per-source 统计非空
    manifest = json.loads(manifest_path.read_text("utf-8"))
    assert manifest["config_echo"] == summary["config_echo"]
    assert manifest["upstream_snapshot_sha256"] == summary["upstream_snapshot_sha256"]
    assert manifest["upstream_snapshot"]["control_sha256"]
    assert summary["per_domain"]["books"]["selected_tokens_by_source"]["src-books"] == \
        summary["per_domain"]["books"]["selected_tokens"]

    # 二次运行：config echo 命中 → 幂等 skip，产物字节不变
    before = members_path.read_bytes()
    _run_s6(root, cfg_path)
    assert members_path.read_bytes() == before

    # 上游 S5 任一分片 summary 改变都级联失效 val 与 mix，不能只看 mix config 静默复用。
    old_upstream = manifest["upstream_snapshot_sha256"]
    old_val_input = json.loads(
        (root / "val_sets" / "books.summary.json").read_text()
    )["input_snapshot_sha256"]
    s5_summary = root / "s5" / "src-books" / "shard0.summary.json"
    changed = json.loads(s5_summary.read_text())
    changed["audit_revision"] = 2
    s5_summary.write_text(json.dumps(changed, sort_keys=True), encoding="utf-8")
    _run_s6(root, cfg_path)
    refreshed_manifest = json.loads(manifest_path.read_text("utf-8"))
    refreshed_val = json.loads((root / "val_sets" / "books.summary.json").read_text())
    assert refreshed_manifest["upstream_snapshot_sha256"] != old_upstream
    assert refreshed_val["input_snapshot_sha256"] != old_val_input

    # Switching the same mix to the production random baseline must remove the
    # legacy stage1 artifact and exclude it from the manifest.
    cfg_path = _mix_cfg(
        tmp_path,
        curriculum={"enabled": False, "method": "random", "stage1_fraction": 0.0},
    )
    _run_s6(root, cfg_path)
    random_manifest = json.loads(manifest_path.read_text("utf-8"))
    assert not stage1_path.exists()
    assert random_manifest["order_tables"] == ["mix_t_stage2.parquet"]
    random_order = pq.read_table(stage2_path)
    assert random_order.num_rows == len(pq.read_table(members_path))


def test_topk_selection_deterministic(tmp_path: Path):
    roots = []
    cfgs = []
    for i in range(2):
        d = tmp_path / f"case{i}"
        root = d / "out"
        build_pool(root)
        roots.append(root)
        cfgs.append(_mix_cfg(d))
    _run_s6(roots[0], cfgs[0])
    _run_s6(roots[1], cfgs[1])
    for rel in (
        "mix_members/mix_t.parquet",
        "mix_order/mix_t_stage1.parquet",
        "mix_order/mix_t_stage2.parquet",
        "val_sets/books.parquet",
        "val_sets/math.parquet",
        "stage_s6_global_summary.json",
        "val_sets/_val_global_summary.json",
    ):
        a = (roots[0] / rel).read_bytes()
        b = (roots[1] / rel).read_bytes()
        assert a == b, f"non-deterministic: {rel}"


def test_deepen_to_quota_expands_only_deficient_quality_band(tmp_path: Path):
    root = tmp_path / "out"
    build_pool(root)
    cfg_path = _mix_cfg(
        tmp_path,
        quota_fill_policy="deepen_to_quota",
        # The synthetic math pool loses half its tokens to validation, so this
        # tolerance keeps the fixture focused on the books deep-fill behavior.
        max_total_shortfall_tokens=250,
    )
    summary = _run_s6(root, cfg_path)
    manifest = json.loads(
        (root / "mix_order" / "mix_t.manifest.json").read_text("utf-8")
    )

    # books' requested top-50% band is only 500 tokens for a 600-token quota.
    # Deep fill scans the domain and selects the best 600-token prefix after val.
    assert manifest["requested_top_k"]["books"] == 0.5
    assert manifest["effective_top_k"]["books"] == 1.0
    assert manifest["effective_top_k"]["math"] == 1.0
    assert summary["per_domain"]["books"]["selected_tokens"] == 600
    assert summary["per_domain"]["books"]["quota_shortfall_tokens"] == 0
    assert summary["total_quota_shortfall_tokens"] <= 250


def _short_high_quality_pool(tmp_path, **overrides):
    root = tmp_path / "out"
    info = build_pool(root)
    path = root / "s5/src-books/shard0.parquet"
    table = pq.read_table(path)
    lengths = [150] * 5 + [50] * 5
    table = table.set_column(table.schema.get_field_index("n_tokens"), "n_tokens",
                             pa.array(lengths, type=pa.uint32()))
    pq.write_table(table, path)
    config = _mix_cfg(
        tmp_path, included_sources=["src-books"], total_tokens=400,
        domains={"books": {"weight": 1.0, "top_k": 0.5}},
        val={"tokens_per_domain": 100, "max_fraction": 0.1},
        quota_fill_policy=overrides.get("quota_fill_policy", "deepen_to_quota"),
        max_total_shortfall_tokens=overrides.get("max_total_shortfall_tokens", 150),
    )
    return root, config, info["books"], lengths


def test_measured_deep_fill_handles_shorter_high_quality_documents_and_resume(tmp_path, monkeypatch):
    root, config, books, lengths = _short_high_quality_pool(tmp_path)
    scans = []
    scan = mz._scan

    def observed_scan(stage, tasks, *args, **kwargs):
        result = scan(stage, tasks, *args, **kwargs)
        if stage in ("s6_collect", "s6_deepfill"):
            scans.append((stage, result))
        return result

    monkeypatch.setattr(mz, "_scan", observed_scan)
    check = mz._assert_quota_capacity
    stop = [True]

    def interrupt_after_selection(*args, **kwargs):
        check(*args, **kwargs)
        if args[-1] == "selected_quality_prefix" and stop[0]:
            stop[0] = False
            raise RuntimeError("injected after measured deep fill")

    monkeypatch.setattr(mz, "_assert_quota_capacity", interrupt_after_selection)
    with pytest.raises(RuntimeError, match="injected after measured deep fill"):
        _run_s6(root, config)
    summary = _run_s6(root, config)
    manifest = json.loads((root / "mix_order/mix_t.manifest.json").read_text())
    assert manifest["requested_top_k"]["books"] == 0.5
    assert manifest["effective_top_k"]["books"] == 1.0
    assert [stage for stage, _ in scans] == ["s6_collect", "s6_deepfill"] * 2
    assert scans[0][1][0]["candidate_tokens"] < 400 <= 0.5 * 1000
    assert scans[1][1][0]["candidate_tokens"] >= 400
    assert all(row.get("skipped") for _, rows in scans[2:] for row in rows)
    val_ids = set(pq.read_table(root / "val_sets/books.parquet").column("doc_id").to_pylist())
    tokens = dict(zip(books["doc_ids"], lengths))
    scores = dict(zip(books["doc_ids"], books["comp"]))
    chosen = []
    total = 0
    for doc in sorted(set(tokens) - val_ids, key=lambda doc: (-scores[doc], doc)):
        if total + tokens[doc] > 400:
            break
        chosen.append(doc)
        total += tokens[doc]
    actual = pq.read_table(root / "mix_order/mix_t_stage2.parquet").column("doc_id").to_pylist()
    assert actual == sorted(chosen)
    assert summary["per_domain"]["books"]["selected_tokens"] == total


def test_strict_quality_capacity_fails_before_sorting_and_records_domain(tmp_path, monkeypatch):
    root, config, _, _ = _short_high_quality_pool(
        tmp_path, quota_fill_policy="strict_top_k", max_total_shortfall_tokens=0)
    monkeypatch.setattr(mz, "sort_key_quality", lambda *_args: pytest.fail("sorted an under-capacity pool"))
    with pytest.raises(SystemExit, match="candidate_capacity"):
        _run_s6(root, config)
    failure = json.loads((root / "audits/s6_mix_t_quota_failure.json").read_text())
    assert failure["phase"] == "candidate_capacity"
    assert failure["per_domain"]["books"]["shortfall_tokens"] > 0
    assert not (root / "mix_order/mix_t.manifest.json").exists()


def test_complete_document_prefix_shortfall_fails_before_table_building(tmp_path, monkeypatch):
    root = tmp_path / "out"
    build_pool(root)
    config = _mix_cfg(
        tmp_path, included_sources=["src-books"], total_tokens=450,
        domains={"books": {"weight": 1.0, "top_k": 1.0}},
        val={"tokens_per_domain": 100, "max_fraction": 0.1},
        quota_fill_policy="deepen_to_quota", max_total_shortfall_tokens=0,
    )
    monkeypatch.setattr(mz, "_doc_id_chunked", lambda *_args: pytest.fail("built table with invalid quotas"))
    with pytest.raises(SystemExit, match="selected_quality_prefix"):
        _run_s6(root, config)
    failure = json.loads((root / "audits/s6_mix_t_quota_failure.json").read_text())
    assert failure["phase"] == "selected_quality_prefix"
    assert failure["total_shortfall_tokens"] == 50
    assert not (root / "mix_members/mix_t.parquet").exists()


def test_topk_document_band_is_not_recapped_by_token_fraction():
    # Two long documents can both belong to a top document-quality band even
    # when their 600 tokens exceed a nominal top_k*pool_tokens estimate of 500.
    # The complete prefix exactly fitting the quota must retain both.
    nt = np.array([300, 300], dtype=np.uint32)
    assert mz.candidate_prefix_count(nt, 600) == 2


def test_deepen_to_quota_fails_closed_above_shortfall_tolerance(tmp_path: Path):
    root = tmp_path / "out"
    build_pool(root)
    cfg_path = _mix_cfg(
        tmp_path,
        quota_fill_policy="deepen_to_quota",
        max_total_shortfall_tokens=0,
    )
    with pytest.raises(SystemExit, match="quota shortfall"):
        _run_s6(root, cfg_path)
    assert not (root / "mix_order" / "mix_t.manifest.json").exists()
    assert not (root / "mix_members" / "mix_t.parquet").exists()
    assert not (root / "mix_order" / "mix_t_stage2.parquet").exists()


def test_s6_snapshot_ignores_unselected_quantile_track(tmp_path: Path):
    root = tmp_path / "out"
    build_pool(root)
    tasks = mz.s5_tasks(root, PROFILES)
    before = mz._s6_upstream_snapshot(root, tasks, "composite_v2")

    tables = root / "s4_quantiles" / "tables"
    np.save(tables / "irrelevant.rule.npy", np.array([0.1, 0.9], dtype=np.float32))
    (tables / "_build.rule.summary.json").write_text(
        json.dumps({"score_field": "rule_composite", "domains": {}}),
        encoding="utf-8",
    )
    after_irrelevant = mz._s6_upstream_snapshot(root, tasks, "composite_v2")
    assert after_irrelevant["snapshot_sha256"] == before["snapshot_sha256"]

    active = tables / "books.npy"
    values = np.load(active)
    np.save(active, values + np.float32(0.001))
    after_active = mz._s6_upstream_snapshot(root, tasks, "composite_v2")
    assert after_active["snapshot_sha256"] != before["snapshot_sha256"]


def test_s6_refuses_publish_when_upstream_changes_during_collect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "out"
    build_pool(root)
    cfg_path = _mix_cfg(tmp_path)
    real_snapshot = mz._s6_upstream_snapshot
    calls = 0

    def drifting_snapshot(*args, **kwargs):
        nonlocal calls
        calls += 1
        value = real_snapshot(*args, **kwargs)
        if calls == 2:
            value = dict(value)
            value["snapshot_sha256"] = "changed-during-s6"
        return value

    monkeypatch.setattr(mz, "_s6_upstream_snapshot", drifting_snapshot)
    with pytest.raises(SystemExit, match="upstream changed during materialization"):
        mz.run_s6(root, PROFILES, cfg_path, workers=2, guard_factory=_guard_factory)
    assert not (root / "mix_order" / "mix_t.manifest.json").exists()
    assert not (root / "mix_members" / "mix_t.parquet").exists()
    assert not (root / "mix_order" / "mix_t_stage2.parquet").exists()


def test_s6_revokes_commit_before_rewriting_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "out"
    build_pool(root)
    cfg_path = _mix_cfg(tmp_path)
    _run_s6(root, cfg_path)
    manifest = root / "mix_order" / "mix_t.manifest.json"
    assert manifest.is_file()

    summary_path = root / "s5" / "src-books" / "shard0.summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["generation"] = 2
    summary_path.write_text(json.dumps(summary, sort_keys=True), encoding="utf-8")

    real_write = mz._write_val_domain
    observed = []

    def guarded_write(*args, **kwargs):
        observed.append(not manifest.exists())
        return real_write(*args, **kwargs)

    monkeypatch.setattr(mz, "_write_val_domain", guarded_write)
    _run_s6(root, cfg_path)
    assert observed and all(observed)
    assert manifest.is_file()


def test_val_disjoint_and_quota(tmp_path: Path):
    root = tmp_path / "out"
    info = build_pool(root)
    cfg_path = _mix_cfg(tmp_path)
    _run_s6(root, cfg_path)
    member_ids = set(
        pq.read_table(root / "mix_members" / "mix_t.parquet").column("doc_id").to_pylist()
    )
    for dom in ("books", "math"):
        val_ids = set(
            pq.read_table(root / "val_sets" / f"{dom}.parquet").column("doc_id").to_pylist()
        )
        assert val_ids <= set(info[dom]["doc_ids"])
        assert not (val_ids & member_ids)
        summ = json.loads((root / "val_sets" / f"{dom}.summary.json").read_text("utf-8"))
        # 前缀填充：quota ≤ val_tokens ≤ quota + 单篇上限（100/50 tok）
        assert summ["quota_tokens"] <= summ["val_tokens"] <= summ["quota_tokens"] + 100


def test_s6_via_driver_main(tmp_path: Path):
    root = tmp_path / "out"
    build_pool(root)
    cfg_path = _mix_cfg(tmp_path)
    profiles_path = tmp_path / "profiles.yaml"
    profiles_path.write_text(
        yaml.safe_dump({"profiles": PROFILES}), encoding="utf-8"
    )
    from yuformer_data.scale import driver

    rc = driver.main(
        ["--stage", "s6", "--mix-config", str(cfg_path), "--out-root", str(root),
         "--profiles", str(profiles_path), "--workers", "2"]
    )
    assert rc == 0
    assert (root / "mix_members" / "mix_t.parquet").is_file()
    assert (root / "stage_s6_global_summary.json").is_file()
