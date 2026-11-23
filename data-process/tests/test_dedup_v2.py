"""ndv2（web/code 近重去重 v2）测试：全文验证正确性 / scipy 聚类等价性 / survivor 规则 /
幂等 / E-12 内存模式审查。真实数据验证见 README（code 域单字节桶端到端）。"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from yuformer_data.scale import dedup_common as dc
from yuformer_data.scale import dedup_v2 as dv2
from yuformer_data.scale.dedup_common import (
    UnionFind,
    jaccard_sets,
    load_dedup_config,
    unique_shingle_set,
)
from yuformer_data.scale.dedup_minhash import (
    PAIRS_SCHEMA,
    S3_DECISIONS_SCHEMA,
    s3_output_paths,
)


def _cfg() -> dict:
    cfg = load_dedup_config()
    cfg["s3_v2"]["verification_method"] = "fulltext"
    cfg["s3_v2"]["a_chunk_docs"] = 50
    cfg["s3_v2"]["chunk_text_cap_gb"] = 0.5
    cfg["s3_v2"]["b_cache_cap_gb"] = 0.05
    cfg["s3_v2"]["pairs_cap_gb"] = 1.0
    cfg["s3_v2"]["component_mem_cap_gb"] = 2.0
    cfg["s3_v2"]["edges_cap_gb"] = 1.0
    return cfg


def _words(seed: int, n: int) -> list[str]:
    rng = np.random.default_rng(seed)
    return [f"w{int(rng.integers(0, 5000))}" for _ in range(n)]


def _join(ws: list[str]) -> str:
    return " ".join(ws)


# ---------------------------------------------------------------- 全文验证正确性

def test_fulltext_verify_accepts_j085_tail_modified_pair():
    """J≈0.85 全文重复对必须判重；修改发生在 2048 字符之后（前缀验证的盲区）。"""
    cfg = _cfg()
    base = _words(7, 4000)
    head = _join(base)
    assert len(head) > 2048
    # 尾部 15% 词替换 → 全文 shingle J ≈ 0.85（前缀与原文完全一致）
    tail = base[-600:]
    rng = np.random.default_rng(11)
    mod = tail.copy()
    for i in range(0, len(mod), 7):
        mod[i] = f"x{int(rng.integers(0, 10**6))}"
    doc_b = _join(base[:-600] + mod)
    sa = unique_shingle_set(head, 5, 100_000)
    sb = unique_shingle_set(doc_b, 5, 100_000)
    j = jaccard_sets(sa, sb)
    assert 0.8 <= j < 0.92, f"构造偏离目标: J={j:.3f}"
    assert dv2.verify_pair_fulltext(head, doc_b, cfg) is True


def test_fulltext_verify_rejects_prefix_only_similar_pair():
    """仅前缀相似（共享 2048+ 字符 boilerplate，正文完全不同）必须拒绝——E-17 病理对。"""
    cfg = _cfg()
    boiler = _join(_words(3, 800))  # ~5KB 共享前缀
    assert len(boiler) > 2048
    body_a = _join(_words(101, 2000))
    body_b = _join(_words(202, 2000))
    doc_a = boiler + " " + body_a
    doc_b = boiler + " " + body_b
    # 旧前缀验证（2048 字符）会放行：
    prefix_j = jaccard_sets(
        unique_shingle_set(doc_a[:2048], 5, 100_000),
        unique_shingle_set(doc_b[:2048], 5, 100_000),
    )
    assert prefix_j >= 0.8, "构造偏离目标：前缀 J 应 ≥0.8（模拟 E-17 放行对）"
    assert dv2.verify_pair_fulltext(doc_a, doc_b, cfg) is False


def test_shingle_cap_is_deterministic_uniform_sample():
    """shingle 上限 = 排序后取前 cap（确定性均匀采样，哈希值均匀 ⇒ 等概率抽样）。"""
    text = _join(_words(5, 200_000))  # ~200k shingles > cap
    s1 = unique_shingle_set(text, 5, 100_000)
    s2 = unique_shingle_set(text, 5, 100_000)
    assert len(s1) == 100_000
    assert np.array_equal(s1, s2)
    assert np.array_equal(s1, np.sort(s1))


def test_fast_shingle_set_matches_blake2b_jaccard():
    """向量化 shingle 哈希（verify 用）与 blake2b 版（S3 签名用）的 Jaccard 必须一致：
    哈希函数是实现细节，Jaccard 定义在 5-gram 词组集合上。含位置偏移敏感性回归
    （相同词在两篇正文不同偏移处必须哈希相同——词尾零填充 token 的上下文无关性）。"""
    from yuformer_data.scale.dedup_v2 import fast_shingle_set

    def j_fast(a: str, b: str) -> float:
        return jaccard_sets(fast_shingle_set(a, 5, 100_000), fast_shingle_set(b, 5, 100_000))

    def j_blake(a: str, b: str) -> float:
        return jaccard_sets(unique_shingle_set(a, 5, 100_000), unique_shingle_set(b, 5, 100_000))

    rng = np.random.default_rng(9)
    words = _words(9, 4000)
    base = _join(words)
    for step in (12, 7, 3):
        mod = words.copy()
        for i in range(0, len(mod), step):
            mod[i] = f"q{int(rng.integers(0, 10**6))}"
        variant = _join(mod)
        jf, jb = j_fast(base, variant), j_blake(base, variant)
        assert abs(jf - jb) < 0.01, f"step={step}: fast={jf:.4f} blake={jb:.4f}"
    # 位置偏移：同一词序列前置不同数量的词（token 必须是词内容纯函数）
    shifted = _join(_words(31, 37)) + " " + base
    jf, jb = j_fast(shifted, base), j_blake(shifted, base)
    assert abs(jf - jb) < 0.01
    assert jf > 0.9  # 共享尾部大部长文，J 必须高（上下文泄漏 bug 会把它打塌）


# ---------------------------------------------------------------- scipy 聚类等价性

@pytest.mark.parametrize("seed", [1, 2, 3])
def test_scipy_components_equivalent_to_python_union_find(seed: int):
    rng = np.random.default_rng(seed)
    n = 200
    edges = np.column_stack(
        [rng.integers(0, n, 400), rng.integers(0, n, 400)]
    ).astype(np.int32)
    uf = UnionFind()
    for i in range(n):
        uf.find(i.to_bytes(16, "big"))
    for a, b in edges.tolist():
        if a != b:
            uf.union(a.to_bytes(16, "big"), b.to_bytes(16, "big"))
    py_comps = set()
    for comp in uf.components():
        py_comps.add(tuple(sorted(int.from_bytes(x, "big") for x in comp)))
    labels = dv2._components_scipy(n, edges)
    sp_comps = set()
    for c in np.unique(labels):
        sp_comps.add(tuple(sorted(np.flatnonzero(labels == c).tolist())))
    assert sp_comps == py_comps


# ---------------------------------------------------------------- 合成 docstore / 布局 fixture

WEB_SOURCES = ["dclm-baseline", "fineweb", "fineweb-edu"]


def _write_parts(tmp_path: Path, docs: list[tuple[str, str, str]]) -> Path:
    """docs: (doc_id_hex, source, text) → s3_cand_parts/<domain>/test/part.parquet"""
    part_dir = tmp_path / "s3_cand_parts" / "web" / "testsrc"
    part_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {"doc_id": [d[0] for d in docs], "source": [d[1] for d in docs], "text": [d[2] for d in docs]}
    )
    pq.write_table(table, part_dir / "part-00000.parquet")
    return part_dir / "part-00000.parquet"


def _build_store(tmp_path: Path, docs: list[tuple[str, str, str]]):
    part = _write_parts(tmp_path, docs)
    dc.build_docstore(
        [part], tmp_path / "s3_store" / "web", run_rows=200, cap_gb=1.0,
        sources=sorted({d[1] for d in docs}),
    )
    dv2.build_flat_ctext(tmp_path, "web", 100_000, workers=2)
    return dc.DocStore(tmp_path / "s3_store" / "web")


def _doc(seed: int) -> str:
    return f"{seed:032x}"[-32:]


def _write_pairs_bydoc(tmp_path: Path, pairs: list[tuple[str, str]]) -> None:
    """按 a 首字节写 s3_pairs_bydoc/web/<byte>/chunk-*.parquet"""
    by_byte: dict[int, list[tuple[str, str]]] = {}
    for a, b in pairs:
        by_byte.setdefault(int(a[:2], 16), []).append((a, b))
    for byte, rows in by_byte.items():
        d = tmp_path / "s3_pairs_bydoc" / "web" / f"{byte:02x}"
        d.mkdir(parents=True, exist_ok=True)
        arr = np.frombuffer(
            b"".join(bytes.fromhex(a) + bytes.fromhex(b) for a, b in rows), dtype=np.uint64
        ).reshape(-1, 4)
        pq.write_table(
            pa.table(
                {"a0": arr[:, 0], "a1": arr[:, 1], "b0": arr[:, 2], "b1": arr[:, 3]},
                schema=PAIRS_SCHEMA,
            ),
            d / "chunk-00000.parquet",
        )


def _near_dup_pair(seed: int) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    base = _words(seed, 3000)
    a = ( _doc(seed), "fineweb", _join(base) )
    mod = base[-300:]
    rng = np.random.default_rng(seed + 1)
    for i in range(0, len(mod), 9):
        mod[i] = f"z{int(rng.integers(0, 10**6))}"
    b = (_doc(seed + 10_000), "dclm-baseline", _join(base[:-300] + mod))
    return a, b


# ---------------------------------------------------------------- verify 任务端到端（合成）

def test_verify_bucket_task_end_to_end_and_idempotent(tmp_path):
    cfg = _cfg()
    docs: list[tuple[str, str, str]] = []
    pairs: list[tuple[str, str]] = []
    expect_edges = 0
    for seed in (11, 22, 33):
        a, b = _near_dup_pair(seed)
        docs += [a, b]
        pairs.append((a[0], b[0]))
        expect_edges += 1
    # 非重复对（应被拒绝）
    c = (_doc(44), "fineweb", _join(_words(44, 3000)))
    d = (_doc(55), "fineweb-edu", _join(_words(55, 3000)))
    docs += [c, d]
    pairs.append((c[0], d[0]))
    _build_store(tmp_path, docs)
    _write_pairs_bydoc(tmp_path, pairs)

    s1 = dv2.process_ndv2_verify(tmp_path, "web", int(pairs[0][0][:2], 16), cfg, WEB_SOURCES)
    assert not s1.get("errors"), s1
    assert s1["counters"]["verified_passed"] == expect_edges
    # 幂等：重跑 skip
    s2 = dv2.process_ndv2_verify(tmp_path, "web", int(pairs[0][0][:2], 16), cfg, WEB_SOURCES)
    assert s2.get("skipped") is True
    # 失败不产 part（E-16）：破坏 pairs 目录后换个新桶无 chunk → 空边成功路径
    byte2 = 0xAB
    s3 = dv2.process_ndv2_verify(tmp_path, "web", byte2, cfg, WEB_SOURCES)
    assert s3["counters"]["pair_rows"] == 0
    assert s3["counters"]["verified_passed"] == 0


def test_verify_error_writes_no_part(tmp_path):
    cfg = _cfg()
    a, b = _near_dup_pair(1)
    _write_pairs_bydoc(tmp_path, [(a[0], b[0])])
    # docstore 缺失 → 任务失败：只落 error summary，不落 part（E-16）
    s = dv2.process_ndv2_verify(tmp_path, "web", 0, cfg, WEB_SOURCES)
    assert s.get("errors")
    assert (tmp_path / "s3_v2" / "verified" / "web" / "00.error.json").is_file()
    assert not (tmp_path / "s3_v2" / "verified" / "web" / "00.parquet").exists()
    # 修复后重跑可成功（error summary 不阻止重试）
    _build_store(tmp_path, [a, b])
    s2 = dv2.process_ndv2_verify(tmp_path, "web", 0, cfg, WEB_SOURCES)
    assert not s2.get("errors")
    assert s2["counters"]["verified_passed"] == 1


def test_packed_minhash_verifier_accepts_similar_and_rejects_dissimilar(tmp_path):
    cfg = _cfg()
    cfg["s3_v2"].update({
        "verification_method": "minhash_signature",
        "signature_bits": 4,
        "minhash_jaccard_threshold": 0.8,
        "signature_chunk_pairs": 2,
    })
    ids = np.array([bytes.fromhex(_doc(i)) for i in (1, 2, 3, 4)], dtype="V16")
    ctext = tmp_path / "s3_v2" / "ctext" / "web" / "00"
    ctext.mkdir(parents=True)
    ids.tofile(ctext / "ids.bin")
    csig = tmp_path / "s3_csig" / "web" / "_idx"
    csig.mkdir(parents=True)
    ids.tofile(csig / "00.ids.bin")
    signatures = np.empty((4, 128), dtype=np.uint32)
    signatures[0] = np.arange(128, dtype=np.uint32)
    signatures[1] = signatures[0]  # identical -> accept
    signatures[2] = np.arange(128, dtype=np.uint32) + 3
    signatures[3] = np.arange(128, dtype=np.uint32) + 11  # no low-nibble matches
    signatures.tofile(csig / "00.sig.bin")
    _write_pairs_bydoc(tmp_path, [(_doc(1), _doc(2)), (_doc(3), _doc(4))])

    packed = dv2.build_packed_csig(tmp_path, "web", workers=1)
    assert packed["rows"] == 4
    summary = dv2.process_ndv2_verify(tmp_path, "web", 0, cfg, WEB_SOURCES)
    assert summary["counters"]["signature_checked"] == 2
    assert summary["counters"]["verified_passed"] == 1
    edges = pq.read_table(tmp_path / "s3_v2" / "verified" / "web" / "00.parquet")
    assert edges.num_rows == 1


def test_precision_audit_gates_sampled_edges_with_exact_fulltext(tmp_path):
    cfg = _cfg()
    cfg["s3_v2"].update(
        {
            "precision_audit_sample_size": 2,
            "precision_audit_min": 0.5,
            "precision_audit_wilson_min": 0.0,
            "precision_audit_text_cap_gb": 1.0,
        }
    )
    a, b = _near_dup_pair(61)
    c = (_doc(62), "fineweb", _join(_words(62, 3000)))
    d = (_doc(63), "dclm-baseline", _join(_words(63, 3000)))
    _build_store(tmp_path, [a, b, c, d])
    vdir = tmp_path / "s3_v2" / "verified" / "web"
    vdir.mkdir(parents=True)
    raw = np.frombuffer(
        bytes.fromhex(a[0]) + bytes.fromhex(b[0]) + bytes.fromhex(c[0]) + bytes.fromhex(d[0]),
        dtype=np.uint64,
    ).reshape(-1, 4)
    pq.write_table(
        pa.table(
            {"a0": raw[:, 0], "a1": raw[:, 1], "b0": raw[:, 2], "b1": raw[:, 3]},
            schema=PAIRS_SCHEMA,
        ),
        vdir / "00.parquet",
    )
    summary = dv2.run_ndv2_precision_audit(tmp_path, "web", cfg)
    assert summary["passed"] is True
    assert summary["precision"] == 0.5
    assert summary["exact_jaccard_ge_threshold"] == 1
    assert dv2.run_ndv2_precision_audit(tmp_path, "web", cfg)["skipped"] is True

    cfg["s3_v2"]["precision_audit_min"] = 0.75
    failed = dv2.run_ndv2_precision_audit(tmp_path, "web", cfg)
    assert failed["passed"] is False
    assert failed["errors"] == {"precision_gate_failed": 1}


# ---------------------------------------------------------------- survivor 规则 / decide 端到端

def test_cluster_decide_survivor_rule_backup_and_idempotency(tmp_path):
    cfg = _cfg()
    cfg["s3_v2"]["clustering_method"] = "connected_components"
    cfg["source_priority"] = ["fineweb-edu", "fineweb", "dclm-baseline"]
    docs: list[tuple[str, str, str]] = []
    edges: list[tuple[str, str]] = []
    # 簇1：edu + fineweb + dclm 三节链 → survivor 必须 edu（源优先级最高）
    a1, b1 = _near_dup_pair(101)
    c1 = (_doc(202), "fineweb-edu", _join(_words(999, 2500)))  # 不同文本，仅参与聚类结构
    docs += [a1, b1, c1]
    edges += [(a1[0], b1[0]), (b1[0], c1[0])]
    # 簇2：两个未知优先级源（testsrc 不在 priority 表）→ doc_id 字典序决胜
    e1 = (_doc(0x10), "fineweb", _join(_words(301, 2000)))
    e2 = (_doc(0x20), "fineweb", _join(_words(302, 2000)))
    e3 = (_doc(0x18), "fineweb", _join(_words(303, 2000)))
    docs += [e1, e2, e3]
    edges += [(e1[0], e2[0]), (e2[0], e3[0])]
    store = _build_store(tmp_path, docs)
    # 注意：_build_store 的 sources.csv 按 sorted(sources) 排，cfg priority 以源名为键——名字对上即可

    # 写两个桶的 verified edges
    vdir = tmp_path / "s3_v2" / "verified" / "web"
    by_byte: dict[int, list[tuple[str, str]]] = {}
    for a, b in edges:
        by_byte.setdefault(int(a[:2], 16), []).append((a, b))
    for byte, rows in by_byte.items():
        d = vdir / f"{byte:02x}"
        d.mkdir(parents=True, exist_ok=True)
        arr = np.frombuffer(
            b"".join(bytes.fromhex(a) + bytes.fromhex(b) for a, b in rows), dtype=np.uint64
        ).reshape(-1, 4)
        pq.write_table(
            pa.table(
                {"a0": arr[:, 0], "a1": arr[:, 1], "b0": arr[:, 2], "b1": arr[:, 3]},
                schema=PAIRS_SCHEMA,
            ),
            d / "part.parquet",
        )

    # E-17 占位（全 keep）先落盘 → decide 应备份
    dec_path = tmp_path / "s3_decisions" / "web.parquet"
    dec_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({"doc_id": [], "decision": [], "dup_of": []}, schema=S3_DECISIONS_SCHEMA),
        dec_path,
    )

    s1 = dv2.run_ndv2_cluster_decide(tmp_path, "web", cfg, WEB_SOURCES)
    assert not s1.get("errors"), s1
    assert s1["clusters"] == 2
    assert s1["dropped"] == 4  # 簇1 留 edu(c1) drop a1,b1；簇2 留 e1(0x10 最小) drop e2,e3
    assert s1["policy"] == dv2.POLICY
    assert (tmp_path / "s3_decisions" / "web.e17_backup.parquet").is_file()

    table = pq.read_table(dec_path)
    drops = {r["doc_id"]: r["dup_of"] for r in table.to_pylist()}
    edu_surv = f"{202:032x}"[-32:]
    assert drops[a1[0]] == edu_surv
    assert drops[b1[0]] == edu_surv
    e1_id, e2_id, e3_id = f"{0x10:032x}"[-32:], f"{0x20:032x}"[-32:], f"{0x18:032x}"[-32:]
    assert e1_id not in drops  # doc_id 最小者幸存
    assert drops[e2_id] == e1_id
    assert drops[e3_id] == e1_id
    assert s1["per_source_drops"] == {"fineweb": 3, "dclm-baseline": 1}  # a1,e2,e3 / b1

    # drops lookup 与 drops_for 兼容
    lookup = dc.load_drops_lookup(tmp_path / "s3_lookup" / "web")
    ids = np.array([bytes.fromhex(x) for x in drops], dtype="V16")
    codes = dc.drops_for(ids, lookup)
    assert (codes == 1).all()

    # 幂等：重跑 skip
    s2 = dv2.run_ndv2_cluster_decide(tmp_path, "web", cfg, WEB_SOURCES)
    assert s2.get("skipped") is True


def test_external_edge_dominance_does_not_chain_unrelated_endpoints(tmp_path):
    cfg = _cfg()
    cfg["s3_v2"].update(
        {
            "clustering_method": "edge_dominance_external",
            "decision_partitions": 16,
            "decision_partition_cap_gb": 1.0,
        }
    )
    cfg["source_priority"] = ["fineweb-edu", "fineweb", "dclm-baseline"]
    # a--b--c: both a and c beat b directly.  a and c have no verified edge and
    # therefore must both survive; connected components would incorrectly keep only a.
    a = (_doc(0x30), "fineweb-edu", "a")
    b = (_doc(0x20), "dclm-baseline", "b")
    c = (_doc(0x10), "fineweb", "c")
    # Same-source 0x10--0x20--0x18: only 0x20 loses a direct edge. 0x18 survives.
    d = (_doc(0x110), "fineweb", "d")
    e = (_doc(0x120), "fineweb", "e")
    f = (_doc(0x118), "fineweb", "f")
    docs = [a, b, c, d, e, f]
    edges = [(a[0], b[0]), (b[0], c[0]), (d[0], e[0]), (e[0], f[0])]
    _build_store(tmp_path, docs)

    vdir = tmp_path / "s3_v2" / "verified" / "web"
    vdir.mkdir(parents=True)
    arr = np.frombuffer(
        b"".join(bytes.fromhex(x) + bytes.fromhex(y) for x, y in edges),
        dtype=np.uint64,
    ).reshape(-1, 4)
    pq.write_table(
        pa.table(
            {"a0": arr[:, 0], "a1": arr[:, 1], "b0": arr[:, 2], "b1": arr[:, 3]},
            schema=PAIRS_SCHEMA,
        ),
        vdir / "00.parquet",
    )

    summary = dv2.run_ndv2_cluster_decide(tmp_path, "web", cfg, WEB_SOURCES)
    assert not summary.get("errors"), summary
    assert summary["dropped"] == 2
    assert summary["clusters"] is None
    assert summary["decision_contract"]["transitive_chaining"] is False
    table = pq.read_table(tmp_path / "s3_decisions" / "web.parquet")
    drops = {row["doc_id"]: row["dup_of"] for row in table.to_pylist()}
    assert drops == {b[0]: a[0], e[0]: d[0]}
    assert c[0] not in drops and f[0] not in drops

    again = dv2.run_ndv2_cluster_decide(tmp_path, "web", cfg, WEB_SOURCES)
    assert again.get("skipped") is True


def test_dry_run_writes_nothing(tmp_path):
    cfg = _cfg()
    a, b = _near_dup_pair(77)
    _build_store(tmp_path, [a, b])
    vdir = tmp_path / "s3_v2" / "verified" / "web" / f"{int(a[0][:2], 16):02x}"
    vdir.mkdir(parents=True)
    arr = np.frombuffer(bytes.fromhex(a[0]) + bytes.fromhex(b[0]), dtype=np.uint64).reshape(-1, 4)
    pq.write_table(
        pa.table({"a0": arr[:, 0], "a1": arr[:, 1], "b0": arr[:, 2], "b1": arr[:, 3]}, schema=PAIRS_SCHEMA),
        vdir / "part.parquet",
    )
    s = dv2.run_ndv2_cluster_decide(tmp_path, "web", cfg, WEB_SOURCES, dry_run=True)
    assert s["dry_run"] is True and s["dropped"] == 1
    assert not (tmp_path / "s3_decisions" / "web.parquet").exists()
    assert not (tmp_path / "s3_decisions" / "web.summary.json").exists()


# ---------------------------------------------------------------- ndv2_refresh

def _s3_shard(tmp_path: Path, rows: list[dict]) -> Path:
    out_path, summary_path = s3_output_paths(tmp_path, "fineweb", "x/y")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    table = pa.table({k: [r[k] for r in rows] for k in keys})
    pq.write_table(table, out_path)
    summary_path.write_text(json.dumps({"stage": "s3_mat", "errors": {}}), "utf-8")
    return out_path


def test_flat_ctext_texts_for_ids_exact_bytes(tmp_path):
    """回归锚定：texts_for_ids 必须逐字节精确，命中篇之间有 filler（非命中篇）时
    尤其关键——np.split 表达不相交区间时末篇会吞掉前隙（piece 虚增 ~150×，实测
    触发 4GB 内存护栏）；start/end 边界错误会让每篇吃掉相邻非命中字节。"""
    filler = [(_doc(0xD000 + i), "fineweb", f"filler-{i} " * 50) for i in range(1, 20)]
    docs = [
        (_doc(0xC001), "fineweb", "alpha " * 100),
        *filler[:5],
        (_doc(0xC002), "fineweb", "bravo"),
        *filler[5:12],
        (_doc(0xC003), "fineweb-edu", "charlie " * 500),
        *filler[12:],
        (_doc(0xC004), "dclm-baseline", "delta"),
    ]
    _build_store(tmp_path, docs)
    fct = dv2.FlatCText(dv2.flat_ctext_dir(tmp_path, "web"))
    target_ids = {_doc(0xC001), _doc(0xC002), _doc(0xC003), _doc(0xC004)}
    targets = [d for d in docs if d[0] in target_ids]
    assert len(targets) == 4
    ids = np.array([bytes.fromhex(d[0]) for d in targets], dtype="V16")
    for probe in (ids, ids[::-1]):  # 逆序 + 批内乱序两个维度
        texts = fct.texts_for_ids(probe)
        assert len(texts) == 4
        for d in targets:
            assert texts[bytes.fromhex(d[0])] == d[2], f"{d[0]}: 字节不精确"
    assert len(fct.texts_for_ids(np.array([bytes.fromhex(_doc(0xC099))], dtype="V16"))) == 0


def test_ndv2_refresh_updates_decision_and_idempotent(tmp_path):
    cfg = _cfg()
    a, b = _near_dup_pair(88)
    store = _build_store(tmp_path, [a, b])
    # 先制造 drops lookup：直接写一份（decide 已在上面测过，这里聚焦 refresh）
    drop_id = bytes.fromhex(b[0])
    dv2._write_lookup_vec(tmp_path / "s3_lookup" / "web", np.array([drop_id], dtype="V16"), [1])
    rows = [
        {"doc_id": a[0], "text": a[2], "s3_decision": "keep"},
        {"doc_id": b[0], "text": b[2], "s3_decision": "keep"},
    ]
    out_path = _s3_shard(tmp_path, rows)

    s1 = dv2.process_shard_ndv2_refresh(tmp_path, "fineweb", "x/y", "web", cfg)
    assert not s1.get("errors"), s1
    assert s1["stage"] == "s3_mat"  # refresh must preserve the S3 provenance ledger
    assert s1["ndv2_refresh"]["decisions"] == {"keep": 1, "drop_near_dup": 1}
    contract = s1["ndv2_refresh"]["transform_contract"]
    assert contract["method"] == "stream_replace_s3_decision_v2"
    assert contract["input_fingerprint"] != contract["output_fingerprint"]
    assert contract["preserved_columns"] == ["doc_id", "text"]
    assert len(contract["doc_id_sha256"]) == 64
    table = pq.read_table(out_path)
    dec = dict(zip(table.column("doc_id").to_pylist(), table.column("s3_decision").to_pylist()))
    assert dec[a[0]] == "keep" and dec[b[0]] == "drop_near_dup"

    s2 = dv2.process_shard_ndv2_refresh(tmp_path, "fineweb", "x/y", "web", cfg)
    assert s2.get("skipped") is True


def test_ndv2_refresh_failure_keeps_original(tmp_path):
    cfg = _cfg()
    rows = [{"doc_id": _doc(1), "text": "t", "s3_decision": "keep"}]
    out_path = _s3_shard(tmp_path, rows)
    before = out_path.read_bytes()
    # lookup 缺失 → drops_for 空 lookup 也不会失败……用无 s3_decision 列的表触发失败
    bad_path, _ = s3_output_paths(tmp_path, "fineweb", "x/z")
    pq.write_table(pa.table({"doc_id": [_doc(2)], "text": ["t"]}), bad_path)
    s = dv2.process_shard_ndv2_refresh(tmp_path, "fineweb", "x/z", "web", cfg)
    assert s.get("errors")
    assert bad_path.read_bytes()  # 原产物仍在
    assert out_path.read_bytes() == before


def test_refresh_global_summary_reconstructs_published_roster(tmp_path):
    cfg = _cfg()
    rows = [{"doc_id": _doc(1), "text": "t", "s3_decision": "keep"}]
    _s3_shard(tmp_path, rows)
    dv2._write_lookup_vec(
        tmp_path / "s3_lookup" / "web", np.array([bytes.fromhex(_doc(1))], dtype="V16"), [1]
    )
    dv2.run_ndv2_refresh(
        tmp_path,
        {"fineweb": {"domain": "web"}},
        cfg,
        workers=1,
    )
    global_summary = json.loads(
        (tmp_path / "stage_ndv2_refresh_global_summary.json").read_text()
    )
    assert global_summary["tasks"] == 1
    assert global_summary["shard_errors"] == 0
    assert global_summary["decisions"] == {"drop_near_dup": 1}
    shard_summary = json.loads(
        (tmp_path / "s3" / "fineweb" / "x/y.summary.json").read_text()
    )
    assert shard_summary["stage"] == "s3_mat"


# ---------------------------------------------------------------- E-12 内存模式审查

def _rss_mb() -> int:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) // 1024
    raise AssertionError("VmRSS unavailable")


def _make_verify_layout(tmp_path: Path, n_pairs: int, seed: int = 5) -> int:
    """n_pairs 个近重复对（不同文本模板），全部落在首字节 0x00 桶。"""
    docs: list[tuple[str, str, str]] = []
    pairs: list[tuple[str, str]] = []
    for i in range(n_pairs):
        a, b = _near_dup_pair(seed * 1000 + i)
        docs += [a, b]
        pairs.append((a[0], b[0]))
    _build_store(tmp_path, docs)
    _write_pairs_bydoc(tmp_path, pairs)
    return 0


def _cap_glibc_arenas() -> None:
    """176 核容器 glibc 默认 MALLOC_ARENA_MAX=8×ncores≈1408 个 arena；多线程 numpy
    分配在各 arena 驻留，把 RSS 基线抬高数百 MB（本机实测 1456MB）。本测试的泄漏
    判据是"同任务两遍的 RSS 增量"，绝对常驻上限只是 sanity 天花板，不应被分配器
    策略左右。进程内 mallopt(M_ARENA_MAX) 与 MALLOC_ARENA_MAX=4 启动口径一致。"""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").mallopt(-8, 4)  # M_ARENA_MAX
    except Exception:
        pass


def test_memory_growth_sublinear_in_pairs():
    """E-12/E-5 审查：数据 ×10，(a) 峰值增量有界（块内计算即弃 + 缓存字节账 + 分批
    正文），(b) 任务完成后 RSS 回落——无常驻残留（泄漏 = 攒齐再处理的慢性形式）。

    注：不用"增长率比值"判据——共享机上 VmRSS 噪声（页缓存回写/分配器保留）使
    小基线比值在满载时不稳定。泄漏判据用"同任务跑两遍"：真泄漏随次数线性增长，
    分配器驻留（glibc arena/numpy 大对象）在第二遍即达平台期。"""
    _cap_glibc_arenas()
    cfg = _cfg()
    cfg["s3_v2"]["a_chunk_docs"] = 20
    baseline_rss = _rss_mb()

    import tempfile
    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
        _make_verify_layout(Path(t1), 3000, seed=50)
        _make_verify_layout(Path(t2), 3000, seed=60)
        rss = []
        for i, t in enumerate((t1, t2)):
            s = dv2.process_ndv2_verify(Path(t), "web", 0, _cfg(), WEB_SOURCES)
            assert not s.get("errors")
            del s
            gc.collect()
            rss.append(_rss_mb())
    growth_second_run = rss[1] - rss[0]
    assert growth_second_run < 50, (
        f"同规模任务第二遍 RSS 再增 {growth_second_run}MB——真泄漏（E-12），"
        "分配器驻留应在第一遍后达平台期"
    )
    assert rss[1] - baseline_rss < 512, (
        f"任务相对既有 pytest/shared-library 基线新增常驻 "
        f"{rss[1] - baseline_rss}MB（baseline={baseline_rss}MB, final={rss[1]}MB）"
    )
