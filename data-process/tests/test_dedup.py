from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from yuformer_data.scale import dedup_common as dc
from yuformer_data.scale.dedup_common import (
    MemoryCapExceeded,
    check_memory_cap,
    load_dedup_config,
    minhash_signature,
    shingle_hashes,
)
from yuformer_data.scale.dedup_exact import (
    S2_SCHEMA,
    build_s2_lookups,
    process_s2_reduce,
    run_s2,
    s2_decisions_dir,
    s2_lookup_dir,
    s2_output_paths,
)
from yuformer_data.scale.dedup_minhash import (
    S3_SCHEMA,
    plan_s3_band_tasks,
    run_s3,
    s3_decisions_path,
    s3_output_paths,
)
from yuformer_data.scale.schema import S1_SCHEMA

REPO_ROOT = Path(__file__).resolve().parents[2]
S1_ROOT = REPO_ROOT / "data" / "processed" / "s1"


def _profiles() -> dict:
    return {
        "fineweb-edu": {"domain": "web"},
        "fineweb": {"domain": "web"},
        "dclm-baseline": {"domain": "web"},
        "dolma-books": {"domain": "books"},
    }


def _cfg(**s3_over: float) -> dict:
    cfg = load_dedup_config()
    cfg["s2"]["num_buckets"] = 64
    cfg["s2"]["reduce_range"] = 64
    cfg["s3"]["lsh_buckets"] = 64
    cfg["s3"]["reduce_range"] = 64
    cfg["s3"]["verify_buckets"] = 8
    cfg["s3"]["docstore_run_rows"] = 500
    for k, v in s3_over.items():
        cfg["s3"][k] = v
    return cfg


def _s1_row(doc_id: str, text: str, source: str, domain: str = "web") -> dict:
    return {
        "doc_id": doc_id,
        "text": text,
        "domain": domain,
        "source": source,
        "lang": "en",
        "upstream_score": None,
        "meta": "{}",
        "decision": "keep",
        "reasons": [],
        "metrics": "{}",
    }


def _write_s1_shard(out_root: Path, source: str, relpath: str, rows: list[dict]) -> Path:
    path = out_root / "s1" / source / (relpath + ".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {name: [r.get(name) for r in rows] for name in S1_SCHEMA.names}
    pq.write_table(pa.table(cols, schema=S1_SCHEMA), path, compression="zstd")
    return path


def _write_s2_shard(out_root: Path, source: str, relpath: str, rows: list[dict]) -> Path:
    path = out_root / "s2" / source / (relpath + ".parquet")
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {name: [r.get(name) for r in rows] for name in S2_SCHEMA.names}
    pq.write_table(pa.table(cols, schema=S2_SCHEMA), path, compression="zstd")
    return path


def _read_s2(out_root: Path, source: str, relpath: str) -> list[dict]:
    return pq.read_table(s2_output_paths(out_root, source, relpath)[0]).to_pylist()


def _read_s3(out_root: Path, source: str, relpath: str) -> list[dict]:
    return pq.read_table(s3_output_paths(out_root, source, relpath)[0]).to_pylist()


def _doc_ids(prefix: str, n: int) -> list[str]:
    return [prefix + f"{i:031x}" for i in range(n)]


BASE_TEXT = (
    "The history of the steam engine stretches back into the first century AD; "
    "the aeolipile described by Hero of Alexandria is the earliest known example. "
    "In the following centuries the fundamentals were gradually developed by "
    "workers across many countries, and by the eighteenth century the steam "
    "engine had become the prime mover of the industrial revolution, powering "
    "mines, mills, locomotives and ships with ever greater efficiency and scale. "
    "James Watt's separate condenser dramatically improved fuel economy, while "
    "high pressure designs by Trevithick enabled compact engines for transport. "
    "The story of steam is a story of thermodynamics, materials and measurement. "
)


def _variants() -> tuple[str, str, str, str]:
    words = BASE_TEXT.split()
    n = len(words)

    def spaced(tokens: str, every: int) -> str:
        w = list(words)
        for i in range(0, n, every):
            w[i] = f"{tokens}{i}"
        return " ".join(w)

    near = spaced("replaced", 50)
    far = spaced("other", 2)
    far2 = spaced("alt", 2)
    j_near = dc.jaccard_sets(
        np.unique(shingle_hashes(BASE_TEXT, 5)), np.unique(shingle_hashes(near, 5))
    )
    j_far = dc.jaccard_sets(
        np.unique(shingle_hashes(BASE_TEXT, 5)), np.unique(shingle_hashes(far, 5))
    )
    j_far2 = dc.jaccard_sets(
        np.unique(shingle_hashes(far, 5)), np.unique(shingle_hashes(far2, 5))
    )
    assert 0.8 <= j_near <= 0.93, j_near
    assert j_far <= 0.55, j_far
    assert j_far2 <= 0.2, j_far2
    return BASE_TEXT, near, far, far2


# ---------------------------------------------------------------- S2

def test_docstore_run_ranged_reader_ground_truth(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from yuformer_data.scale.dedup_common import DocStore, build_docstore

    d = tmp_path / "store"
    part = tmp_path / "parts" / "p.parquet"
    part.parent.mkdir(parents=True, exist_ok=True)
    # 构造跨首字节（含 0x00 / 0xff）、多 source、超长短文本的已知数据
    rows = []
    expect = {}
    texts = {
        0x00: "alpha document " * 30,
        0x11: "beta " * 500,
        0x80: "gamma " * 400,
        0xFF: "delta " * 340,
    }
    srcs_in = {0x00: "fineweb", 0x11: "dclm-baseline", 0x80: "fineweb-edu", 0xFF: "fineweb"}
    rng = np.random.RandomState(7)
    for b, t in texts.items():
        for i in range(5):
            did = bytes([b]) + rng.bytes(15)
            rows.append(
                {"doc_id": did.hex(), "source": srcs_in[b], "text": f"{t}-{i}-{'x' * (b + 50)}"}
            )
            expect[did] = (srcs_in[b], rows[-1]["text"])
    # 另加一批随机 id 填充多个 run（docstore_run_rows 调小以切出多 run）
    for i in range(3000):
        did = rng.bytes(16)
        rows.append({"doc_id": did.hex(), "source": "fineweb", "text": f"filler {i} " * 20})
        expect[did] = ("fineweb", rows[-1]["text"])
    pq.write_table(pa.table({k: [r[k] for r in rows] for k in rows[0]}), part)

    cfg = _cfg()
    payload = build_docstore([part], d, 800, 0.5, ["dclm-baseline", "fineweb", "fineweb-edu"])
    assert payload["format"] == "runs2" and payload["docs"] == len(rows) and payload["runs"] > 1
    assert len(list((d / "_runs").glob("*.ids.bin"))) == payload["runs"]

    store = DocStore(d)
    assert not store.legacy
    # 逐桶对照 ground truth（vb=256 连续区间路径 + vb=8 掩码路径）
    for vb in (256, 8):
        seen = {}
        for k in range(vb):
            ids, srcs, txts = store.texts_for_bucket(k, vb, 10_000, 1.0)
            for i in range(len(ids)):
                did = ids[i].tobytes()
                exp_src, exp_text = expect[did]
                assert txts[i] == exp_text
                assert did not in seen
                seen[did] = True
        assert len(seen) == len(expect)
    # 空桶：vb>256 时高位桶（byte≥256）必为空（首字节只有 0-255），走掩码路径
    ids, srcs, txts = store.texts_for_bucket(300, 512, 10_000, 1.0)
    assert len(ids) == 0 and txts == []
    # texts_and_sources_for_ids：命中 + 未命中混合
    from yuformer_data.scale.dedup_common import doc_id_to_bin

    known = np.array([doc_id_to_bin(rows[0]["doc_id"]), doc_id_to_bin(rows[7]["doc_id"])], dtype="V16")
    missing = np.array([b"\xde" * 16], dtype="V16")
    q = np.concatenate([known, missing])
    tt, ss = store.texts_and_sources_for_ids(q, 10_000, 1.0)
    name_by_code = (d / "sources.csv").read_text("utf-8").split()
    assert set(tt) == {known[0].tobytes(), known[1].tobytes()}
    assert tt[known[0].tobytes()] == expect[known[0].tobytes()][1]
    assert name_by_code[ss[known[1].tobytes()]] == expect[known[1].tobytes()][0]
    # sources() 点查：missing 回落 0 码
    codes = store.sources(q)
    assert name_by_code[codes[1]] == expect[known[1].tobytes()][0]
    assert codes[2] == 0
    # 截断语义
    ids, srcs, txts = store.texts_for_bucket(0x11, 256, 100, 1.0)
    for t in txts:
        assert len(t) <= 100
    ids2, _, txts_full = store.texts_for_bucket(0x11, 256, 10_000, 1.0)
    assert [t[:100] for t in txts_full] == txts


def test_s2_exact_dedup_cross_shard_normalized_and_deterministic(tmp_path):
    out_root = tmp_path / "out"
    dup_a = "  Hello \n\tWorld " + "exact duplicate body " * 20
    dup_b = "hello world " + "exact duplicate body " * 20
    assert dc.normalize_for_hash(dup_a) == dc.normalize_for_hash(dup_b)

    ids = _doc_ids("a", 4)
    _write_s1_shard(
        out_root, "fineweb", "s0/part-0", [_s1_row(ids[0], dup_a, "fineweb")]
    )
    _write_s1_shard(
        out_root,
        "fineweb-edu",
        "s0/part-1",
        [_s1_row(ids[1], dup_b, "fineweb-edu"), _s1_row(ids[2], BASE_TEXT, "fineweb-edu")],
    )

    cfg = _cfg()
    run_s2(out_root, _profiles(), cfg, workers=2)

    rows = _read_s2(out_root, "fineweb", "s0/part-0") + _read_s2(
        out_root, "fineweb-edu", "s0/part-1"
    )
    by_id = {r["doc_id"]: r for r in rows}
    assert by_id[ids[0]]["s2_decision"] == "drop_exact_dup"
    assert by_id[ids[0]]["s2_decision"] == "drop_exact_dup"
    assert by_id[ids[1]]["s2_decision"] == "keep"
    # 规范化后等价 → 只留一篇，且 fineweb-edu 优先级高 → survivor 是 ids[1]
    assert by_id[ids[2]]["s2_decision"] == "keep"

    dec = [
        d
        for f in sorted(s2_decisions_dir(out_root, "web").glob("*.parquet"))
        for d in pq.read_table(f).to_pylist()
    ]
    dec_map = {d["doc_id"]: d["dup_of"] for d in dec}
    assert dec_map[ids[0]] == ids[1]

    first_bytes = {p: p.read_bytes() for p in sorted((out_root / "s2").rglob("*.parquet"))}
    summaries2 = run_s2(out_root, _profiles(), cfg, workers=2)
    assert all(s.get("skipped") for s in summaries2 if s.get("stage") == "s2_merge")
    assert {p: p.read_bytes() for p in sorted((out_root / "s2").rglob("*.parquet"))} == first_bytes


def test_s2_reduce_memory_cap(tmp_path):
    out_root = tmp_path / "out"
    ids = _doc_ids("b", 3)
    _write_s1_shard(
        out_root,
        "fineweb",
        "s0/part-0",
        [_s1_row(i, BASE_TEXT, "fineweb") for i in ids],
    )
    cfg = _cfg()
    # 直接构造 pass1 parts
    run_s2(out_root, _profiles(), cfg, workers=1)
    tiny = _cfg()
    tiny["s2"]["memory_cap_gb"] = 1e-9
    # expected_sources 与已写 summary 不一致 → 绕过 skip，强制重算触发上限
    summary = process_s2_reduce(out_root, "web", 0, 64, tiny, ["fineweb", "other"])
    assert summary["errors"]
    assert "MemoryCapExceeded" in json.dumps(summary)
    with pytest.raises(MemoryCapExceeded):
        check_memory_cap(1 << 30, 1e-9, "unit")


# ---------------------------------------------------------------- S3

def _s2_row_from_s1(row: dict) -> dict:
    out = dict(row)
    out["s2_decision"] = "keep"
    return out


def _setup_s3_tree(tmp_path: Path) -> tuple[Path, dict]:
    out_root = tmp_path / "out"
    base, near, far, far2 = _variants()
    ids = _doc_ids("c", 6)
    rows_a = [
        _s2_row_from_s1(_s1_row(ids[0], base, "fineweb")),
        _s2_row_from_s1(_s1_row(ids[2], far, "fineweb")),
    ]
    rows_b = [
        _s2_row_from_s1(_s1_row(ids[1], near, "fineweb-edu")),
        _s2_row_from_s1(_s1_row(ids[3], BASE_TEXT + "completely different tail " * 5, "fineweb-edu")),
        _s2_row_from_s1(_s1_row(ids[4], far2, "fineweb-edu")),
        _s2_row_from_s1(_s1_row(ids[5], base + " extra trailing sentence for variety", "fineweb-edu")),
    ]
    _write_s2_shard(out_root, "fineweb", "s0/part-0", rows_a)
    _write_s2_shard(out_root, "fineweb-edu", "s0/part-1", rows_b)
    return out_root, {"ids": ids, "rows_a": rows_a, "rows_b": rows_b}


def test_s3_minhash_detects_near_dup_and_rejects_far(tmp_path):
    out_root, ctx = _setup_s3_tree(tmp_path)
    ids = ctx["ids"]
    cfg = _cfg()
    run_s3(out_root, _profiles(), cfg, workers=2)

    rows = _read_s3(out_root, "fineweb", "s0/part-0") + _read_s3(
        out_root, "fineweb-edu", "s0/part-1"
    )
    by_id = {r["doc_id"]: r for r in rows}
    # ids[0] (base) 与 ids[1] (near, J≈0.85) 必须判重；survivor = fineweb-edu (优先级高)
    assert by_id[ids[1]]["s3_decision"] == "keep"
    assert by_id[ids[0]]["s3_decision"] == "drop_near_dup"
    # ids[2]/ids[4] 是 far 改写 (J≤0.55) → 不判重
    assert by_id[ids[2]]["s3_decision"] == "keep"
    assert by_id[ids[4]]["s3_decision"] == "keep"
    dec = pq.read_table(s3_decisions_path(out_root, "web")).to_pylist()
    dec_map = {d["doc_id"]: d["dup_of"] for d in dec}
    assert dec_map[ids[0]] == ids[1]


def test_s3_sig_deterministic_byte_level():
    base, near, _, _ = _variants()
    h = shingle_hashes(base, 5)
    s1 = minhash_signature(h, 128)
    s2 = minhash_signature(shingle_hashes(base, 5), 128)
    s3 = minhash_signature(shingle_hashes(near, 5), 128)
    assert s1.tobytes() == s2.tobytes()
    assert s1.tobytes() != s3.tobytes()
    assert s1.dtype == np.uint32 and s1.shape == (128,)


def test_s3_union_find_chain_clusters(tmp_path):
    out_root = tmp_path / "out"
    ids = _doc_ids("e", 3)
    words = BASE_TEXT.split()
    n = len(words)

    def spaced(tokens: str, positions: list[int]) -> str:
        w = list(words)
        for i in positions:
            w[i] = f"{tokens}{i}"
        return " ".join(w)

    step = max(1, n // 3)
    b_text = spaced("swapb", [step, 2 * step])
    c_text = spaced("swapc", [step, 2 * step])
    a_text = BASE_TEXT
    j_ab = dc.jaccard_sets(np.unique(shingle_hashes(a_text, 5)), np.unique(shingle_hashes(b_text, 5)))
    j_bc = dc.jaccard_sets(np.unique(shingle_hashes(b_text, 5)), np.unique(shingle_hashes(c_text, 5)))
    j_ac = dc.jaccard_sets(np.unique(shingle_hashes(a_text, 5)), np.unique(shingle_hashes(c_text, 5)))
    assert j_ab >= 0.8 and j_bc >= 0.8, (j_ab, j_bc)
    rows = [
        _s2_row_from_s1(_s1_row(ids[0], a_text, "fineweb")),
        _s2_row_from_s1(_s1_row(ids[1], b_text, "fineweb-edu")),
        _s2_row_from_s1(_s1_row(ids[2], c_text, "dclm-baseline")),
    ]
    _write_s2_shard(out_root, "fineweb", "s0/part-0", rows)
    cfg = _cfg()
    run_s3(out_root, _profiles(), cfg, workers=2)
    out = _read_s3(out_root, "fineweb", "s0/part-0")
    by_id = {r["doc_id"]: r for r in out}
    drops = [i for i in ids if by_id[i]["s3_decision"] == "drop_near_dup"]
    keeps = [i for i in ids if by_id[i]["s3_decision"] == "keep"]
    # A~B~C 链式聚成一簇，无论 A~C 是否直接相似
    assert len(keeps) == 1 and len(drops) == 2, ([by_id[i]["s3_decision"] for i in ids], (j_ab, j_bc, j_ac))
    dec = pq.read_table(s3_decisions_path(out_root, "web")).to_pylist()
    assert len(dec) == 2
    assert {d["dup_of"] for d in dec} == {keeps[0]}


def test_s3_idempotent(tmp_path):
    out_root, ctx = _setup_s3_tree(tmp_path)
    cfg = _cfg()
    run_s3(out_root, _profiles(), cfg, workers=2)
    first = {p: p.read_bytes() for p in sorted((out_root / "s3").rglob("*.parquet"))}
    first_dec = s3_decisions_path(out_root, "web").read_bytes()
    summaries = run_s3(out_root, _profiles(), cfg, workers=2)
    assert all(s.get("skipped") for s in summaries if s.get("stage") in ("s3_sig", "s3_collect", "s3_mat"))
    assert {p: p.read_bytes() for p in sorted((out_root / "s3").rglob("*.parquet"))} == first
    assert s3_decisions_path(out_root, "web").read_bytes() == first_dec


# ---------------------------------------------------------------- 端到端（真实 s1 分片）

def _real_shards() -> list[tuple[str, Path]]:
    cands = [
        ("cci3-hq", S1_ROOT / "cci3-hq" / "data" / "part_000310.jsonl.parquet"),
        ("dclm-baseline", S1_ROOT / "dclm-baseline" / "global-shard_09_of_10" / "local-shard_0_of_10" / "shard_00000074_processed.jsonl.zst.parquet"),
        ("fineweb", S1_ROOT / "fineweb" / "data" / "CC-MAIN-2019-22" / "004_00045.parquet.parquet"),
    ]
    return [(s, p) for s, p in cands if p.is_file()]


@pytest.mark.skipif(
    os.environ.get("YUFORMER_RUN_REAL_DATA_TESTS") != "1",
    reason="multi-GB real-data integration test is opt-in",
)
def test_end_to_end_real_s1_shards(tmp_path):
    shards = _real_shards()
    if len(shards) < 2:
        pytest.skip(f"real s1 shards not available: {shards}")
    out_root = tmp_path / "out"
    for source, path in shards[:5]:
        rel = path.name[: -len(".parquet")]
        dst = out_root / "s1" / source / (rel + ".parquet")
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.symlink_to(path)

    cfg = _cfg()
    profiles = _profiles()
    s2_summaries = run_s2(out_root, profiles, cfg, workers=4)
    s3_summaries = run_s3(out_root, profiles, cfg, workers=4)

    merge_in = sum(int(s.get("in_rows") or 0) for s in s2_summaries if s.get("stage") == "s2_merge")
    merge_out = sum(int(s.get("out_rows") or 0) for s in s2_summaries if s.get("stage") == "s2_merge")
    assert merge_in == merge_out
    s2_drops = sum(int(s.get("dropped") or 0) for s in s2_summaries if s.get("stage") == "s2_reduce")
    s2_drop_cols = sum(
        v for s in s2_summaries if s.get("stage") == "s2_merge" for k, v in (s.get("decisions") or {}).items() if k == "drop_exact_dup"
    )
    assert s2_drops == s2_drop_cols

    mat_in = sum(int(s.get("in_rows") or 0) for s in s3_summaries if s.get("stage") == "s3_mat")
    mat_out = sum(int(s.get("out_rows") or 0) for s in s3_summaries if s.get("stage") == "s3_mat")
    assert mat_in == mat_out
    s3_drops = sum(int(s.get("dropped") or 0) for s in s3_summaries if s.get("stage") == "s3_decide")
    s3_drop_cols = sum(
        v for s in s3_summaries if s.get("stage") == "s3_mat" for k, v in (s.get("decisions") or {}).items() if k == "drop_near_dup"
    )
    assert s3_drops == s3_drop_cols

    out_files = sorted((out_root / "s3").rglob("*.parquet"))
    assert out_files
    table = pq.read_table(out_files[0], schema=S3_SCHEMA)
    assert table.num_rows >= 0
    assert (out_root / "stage_s2_global_summary.json").is_file()
    assert (out_root / "stage_s3_global_summary.json").is_file()
    summary3 = json.loads((out_root / "stage_s3_global_summary.json").read_text("utf-8"))
    assert summary3["per_domain"]
