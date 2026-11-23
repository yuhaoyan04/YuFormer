from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from yuformer_data.scale import detectors
from yuformer_data.scale.detectors import (
    DECISION_DROP,
    DECISION_KEEP,
    DECISION_REVIEW,
    check_code_path,
    check_garbled,
    check_html,
    check_integrity,
    check_secrets,
    check_web_line_dup,
    evaluate,
    extract_path,
    normalize_text,
)
from yuformer_data.scale.filter import process_shard_s1, s1_output_paths
from yuformer_data.scale.ingest import (
    iter_raw_rows,
    map_row,
    process_shard_s0,
    shard_output_paths,
)
from yuformer_data.scale.profiles import load_profiles
from yuformer_data.scale.schema import CANONICAL_SCHEMA, make_doc_id

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = REPO_ROOT / "data" / "pretrain-raw"

GOOD_TEXT = (
    "The history of the steam engine stretches back into the first century AD; "
    "the aeolipile described by Hero of Alexandria is the earliest known example. "
    "In the following centuries the fundamentals were gradually developed."
)


def test_doc_id_deterministic():
    a = make_doc_id("fineweb", "data/CC-MAIN-2016-30/000_00027.parquet", 42)
    b = make_doc_id("fineweb", "data/CC-MAIN-2016-30/000_00027.parquet", 42)
    c = make_doc_id("fineweb", "data/CC-MAIN-2016-30/000_00027.parquet", 43)
    d = make_doc_id("dclm-baseline", "data/CC-MAIN-2016-30/000_00027.parquet", 42)
    assert a == b
    assert a != c
    assert a != d
    assert len(a) == 32


def test_normalize():
    t = "Café\u0301\u0007x\nline  with\t tabs   \n\n\n\nnext"
    out = normalize_text(t)
    assert "\u0007" not in out
    assert "Café" in out
    assert "\n\n\n" not in out
    assert "line  with" not in out


def test_integrity():
    assert check_integrity("")[0] == DECISION_DROP
    assert check_integrity("short")[0] == DECISION_DROP
    assert check_integrity(GOOD_TEXT) == (DECISION_KEEP, None)
    assert check_integrity("x" * 2_000_001)[0] == DECISION_REVIEW


def test_garbled_replacement_char():
    bad = ("ok words " * 200) + "\ufffd" * 50
    assert any(r == "garbled_replacement_char" for _, r in check_garbled(bad))
    assert check_garbled(GOOD_TEXT * 10) == []


def test_garbled_latin1_mojibake():
    bad = "This cafÃ© is naÃ¯ve " * 100
    assert any(r == "garbled_latin1_mojibake" for _, r in check_garbled(bad))
    assert check_garbled("Café naïve " * 100) == []


def test_garbled_control_chars():
    bad = ("fine text here " * 100) + "\x0b\x0c\x0b" * 30
    assert any(r == "garbled_control_chars" for _, r in check_garbled(bad))


def test_html_rules():
    tags = "<div><span>hello</span></div>" * 200
    assert any(r == "html_tag_ratio" for _, r in check_html(tags))
    script = GOOD_TEXT * 20 + "<script>var x=1;</script>"
    assert any(r == "html_script_style_block" for _, r in check_html(script))
    entities = "word " * 50 + "&lt;&gt;&amp;nbsp;" * 30
    assert any(r == "html_entity_flood" for _, r in check_html(entities))
    b64 = GOOD_TEXT * 20 + "QUJD" * 3000
    assert any(r == "base64_blob" for _, r in check_html(b64))
    assert check_html(GOOD_TEXT) == []


def test_secrets():
    hit = check_secrets('config = {"api_key": "ABCDEFGHIJKLMNOP1234"}')
    assert hit is not None and hit[0] == DECISION_REVIEW
    assert check_secrets("the password policy") is None


def test_web_line_dup():
    boiler = "\n".join(["unique header"] + ["same line"] * 40)
    assert check_web_line_dup(boiler) == (DECISION_DROP, "web_line_dup")
    varied = "\n".join(f"line number {i} content" for i in range(60))
    assert check_web_line_dup(varied) is None


def test_code_path():
    assert check_code_path("src/node_modules/foo/index.js") == (DECISION_DROP, "code_vendor_path")
    assert check_code_path("app/vendor/lib.py") == (DECISION_DROP, "code_vendor_path")
    assert check_code_path("dist/app.min.js") == (DECISION_DROP, "code_vendor_path")
    assert check_code_path("src/main.py") is None
    assert check_code_path(None) is None
    meta = json.dumps({"path": "x/node_modules/y.js", "text": "t"}, sort_keys=True)
    assert extract_path(meta) == "x/node_modules/y.js"


def test_evaluate_drop_beats_review():
    text = "x" * 2_000_001 + ' api_key: "ABCDEFGHIJKLMNOP"'
    verdict, lang = evaluate(text, "web", "{}", "")
    assert verdict.decision == DECISION_DROP
    assert "too_long" in verdict.reasons
    assert "secret_pattern" in verdict.reasons
    assert lang in ("unk", "en")


def test_evaluate_lang_skipped_without_model(monkeypatch):
    monkeypatch.delenv(detectors._LID_MODEL_ENV, raising=False)
    detectors.reset_lang_detector()
    verdict, lang = evaluate(GOOD_TEXT, "web", "{}", "")
    assert lang == "unk"
    assert verdict.metrics.get("lang_skipped") == 1.0


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    cols = {k: [r.get(k) for r in rows] for k in keys}
    pq.write_table(pa.table(cols), path)


def _synth_profiles() -> dict:
    return {
        "synp": {
            "domain": "web",
            "extensions": [".parquet"],
            "text_fields": ["text", "content"],
            "path_fields": ["url"],
            "score_fields": ["score"],
            "lang_fields": ["language"],
            "exclude_dirs": [],
            "mode": "text",
        },
        "syng": {
            "domain": "books",
            "extensions": [".json.gz"],
            "text_fields": ["text"],
            "path_fields": [],
            "score_fields": [],
            "lang_fields": [],
            "lang_default": "en",
            "exclude_dirs": [],
            "mode": "text",
        },
        "synt": {
            "domain": "books",
            "extensions": [".txt"],
            "text_fields": [],
            "path_fields": [],
            "score_fields": [],
            "lang_fields": [],
            "exclude_dirs": [],
            "mode": "text",
        },
    }


def _make_synth_tree(tmp_path: Path) -> Path:
    data_root = tmp_path / "raw"
    _write_parquet(
        data_root / "synp" / "part-000.parquet",
        [
            {"text": GOOD_TEXT, "url": "http://a", "score": 0.5, "language": "en", "junk": 1},
            {"text": "", "url": "http://b"},
            {"content": "content field text " * 10, "url": "http://c"},
        ],
    )
    gz_dir = data_root / "syng"
    gz_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(gz_dir / "books-0000.json.gz", "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"text": GOOD_TEXT, "id": "g1"}) + "\n")
        fh.write("{bad json\n")
        fh.write(json.dumps({"text": "another book " * 20, "id": "g2"}) + "\n")
    txt_dir = data_root / "synt"
    txt_dir.mkdir(parents=True, exist_ok=True)
    (txt_dir / "46812.txt").write_text("A whole plain text book.\n" + GOOD_TEXT, encoding="utf-8")
    return data_root


def test_ingest_mapping_parquet_gz_txt(tmp_path):
    data_root = _make_synth_tree(tmp_path)
    out_root = tmp_path / "out"
    prof = _synth_profiles()

    s = process_shard_s0(data_root / "synp" / "part-000.parquet", out_root, "synp", "part-000.parquet", prof["synp"])
    # H2 语义：空串/非字符串一律回退候选字段；text="" 且无 content 的行记 error（不再有静默空文本行）
    assert s["in_rows"] == 3 and s["out_rows"] == 2 and s["error_rows"] == 1
    out_path, _ = shard_output_paths(out_root, "s0", "synp", "part-000.parquet")
    rows = pq.read_table(out_path).to_pylist()
    assert rows[0]["text"] == GOOD_TEXT
    assert rows[0]["upstream_score"] == 0.5
    assert rows[0]["lang"] == "en"
    assert rows[0]["source"] == "synp" and rows[0]["domain"] == "web"
    assert rows[1]["text"].startswith("content field text")  # row1(text="")→error；row2 走 content 回退
    assert json.loads(rows[0]["meta"])["url"] == "http://a"
    assert json.loads(rows[0]["meta"])["junk"] == 1

    s = process_shard_s0(data_root / "syng" / "books-0000.json.gz", out_root, "syng", "books-0000.json.gz", prof["syng"])
    assert s["in_rows"] == 3 and s["out_rows"] == 2 and s["error_rows"] == 1
    out_path, _ = shard_output_paths(out_root, "s0", "syng", "books-0000.json.gz")
    rows = pq.read_table(out_path).to_pylist()
    assert rows[0]["lang"] == "en"
    assert rows[0]["doc_id"] == make_doc_id("syng", "books-0000.json.gz", 0)

    s = process_shard_s0(data_root / "synt" / "46812.txt", out_root, "synt", "46812.txt", prof["synt"])
    assert s["in_rows"] == 1 and s["out_rows"] == 1
    out_path, _ = shard_output_paths(out_root, "s0", "synt", "46812.txt")
    rows = pq.read_table(out_path).to_pylist()
    assert rows[0]["doc_id"] == make_doc_id("synt", "46812.txt", 0)
    assert "plain text book" in rows[0]["text"]


def test_ingest_idempotent(tmp_path):
    data_root = _make_synth_tree(tmp_path)
    out_root = tmp_path / "out"
    prof = _synth_profiles()["synp"]
    src = data_root / "synp" / "part-000.parquet"
    s1 = process_shard_s0(src, out_root, "synp", "part-000.parquet", prof)
    assert not s1.get("skipped")
    out_path, summary_path = shard_output_paths(out_root, "s0", "synp", "part-000.parquet")
    first_bytes = out_path.read_bytes()
    s2 = process_shard_s0(src, out_root, "synp", "part-000.parquet", prof)
    assert s2["skipped"] is True
    assert out_path.read_bytes() == first_bytes
    assert summary_path.is_file()


def test_s1_filter_routes_and_idempotent(tmp_path):
    data_root = _make_synth_tree(tmp_path)
    out_root = tmp_path / "out"
    process_shard_s0(data_root / "synp" / "part-000.parquet", out_root, "synp", "part-000.parquet", _synth_profiles()["synp"])
    s0_path = out_root / "s0" / "synp" / "part-000.parquet.parquet"

    s = process_shard_s1(s0_path, out_root, "synp", "part-000.parquet", "web")
    # H2 后：row1(text="") 在 S0 即 error，S1 只见 2 行——路由断言相应更新
    assert s["in_rows"] == 2 and s["out_rows"] == 2 and s["audit_rows"] == 0
    assert s["decisions"][DECISION_KEEP] == 2
    assert s["decisions"].get(DECISION_DROP, 0) == 0
    keep_path, audit_path, summary_path = s1_output_paths(out_root, "synp", "part-000.parquet")
    assert pq.read_table(keep_path).num_rows == 2
    assert pq.read_table(audit_path).num_rows == 0
    first = keep_path.read_bytes()
    s2 = process_shard_s1(s0_path, out_root, "synp", "part-000.parquet", "web")
    assert s2["skipped"] is True
    assert keep_path.read_bytes() == first


def test_s1_drop_goes_to_audit_only(tmp_path):
    data_root = tmp_path / "raw"
    bad = "<div>x</div>" * 300
    _write_parquet(data_root / "synp" / "part-000.parquet", [{"text": bad, "url": "http://x"}])
    out_root = tmp_path / "out"
    process_shard_s0(data_root / "synp" / "part-000.parquet", out_root, "synp", "part-000.parquet", _synth_profiles()["synp"])
    s0_path = out_root / "s0" / "synp" / "part-000.parquet.parquet"
    s = process_shard_s1(s0_path, out_root, "synp", "part-000.parquet", "web")
    assert s["decisions"][DECISION_DROP] == 1
    keep_path, audit_path, _ = s1_output_paths(out_root, "synp", "part-000.parquet")
    assert pq.read_table(keep_path).num_rows == 0
    audit = pq.read_table(audit_path).to_pylist()
    assert len(audit) == 1
    assert audit[0]["text_head"] == bad[:200]
    assert "html_tag_ratio" in audit[0]["reasons"]
    assert "text" not in pq.read_table(audit_path).schema.names


def _real_files() -> list[tuple[str, Path]]:
    return [
        ("wikitext", RAW_ROOT / "wikitext" / "wikitext-103-raw-v1" / "test-00000-of-00001.parquet"),
        ("dolma-books", RAW_ROOT / "dolma-books" / "books-0002.json.gz"),
        ("proof-pile-2", RAW_ROOT / "proof-pile-2" / "algebraic-stack" / "train" / "agda0000.jsonl.zst"),
    ]


@pytest.mark.skipif(
    os.environ.get("YUFORMER_RUN_REAL_DATA_TESTS") != "1",
    reason="multi-GB real-data integration test is opt-in",
)
def test_end_to_end_smoke_real(tmp_path):
    missing = [p for _, p in _real_files() if not p.is_file()]
    if missing:
        pytest.skip(f"real raw files not downloaded yet: {missing}")
    profiles = load_profiles()
    out_root = tmp_path / "out"
    for source, path in _real_files():
        rel = path.relative_to(RAW_ROOT / source).as_posix()
        s0 = process_shard_s0(path, out_root, source, rel, profiles[source])
        assert s0["out_rows"] > 0, f"{source}: s0 produced no rows"
        assert s0["in_rows"] >= s0["out_rows"]
        s0_path, _ = shard_output_paths(out_root, "s0", source, rel)
        table = pq.read_table(s0_path, schema=CANONICAL_SCHEMA)
        assert table.num_rows == s0["out_rows"]
        doc_ids = table.column("doc_id").to_pylist()
        assert len(doc_ids) == len(set(doc_ids))
        # H2 语义：error 行（如空文本字段）被省略，doc_id 是行号空间的不连续子序列——
        # 断言"唯一 + 全部属于行号宇宙 + 数量闭合"，不断言连续对齐
        universe = {make_doc_id(source, rel, i) for i in range(s0["in_rows"])}
        assert set(doc_ids) <= universe and len(doc_ids) == s0["out_rows"]

        s1 = process_shard_s1(s0_path, out_root, source, rel, profiles[source]["domain"])
        assert s1["in_rows"] == s0["out_rows"]
        assert s1["out_rows"] + s1["audit_rows"] == s1["in_rows"]
        assert s1["decisions"].get(DECISION_DROP, 0) == s1["audit_rows"]
        assert set(s1["decisions"]) <= {DECISION_KEEP, DECISION_REVIEW, DECISION_DROP}
        keep_path, audit_path, summary_path = s1_output_paths(out_root, source, rel)
        assert pq.read_table(keep_path).num_rows == s1["out_rows"]
        assert pq.read_table(audit_path).num_rows == s1["audit_rows"]
        assert summary_path.is_file()


# ---------------------------------------------------------------- qa 域 inline HTML 清洗（2026-09-26 stack-exchange 残留修复）

def test_clean_inline_html_strips_qa_tags():
    raw = 'Q: <p>Is it <b>realistic</b> to develop?</p> A: yes &amp; no &#40;parens&#41;'
    out = detectors.clean_inline_html(raw)
    assert "<p>" not in out and "<b>" not in out
    assert "&amp;" not in out and "&#40;" not in out
    assert "realistic" in out and "(parens)" in out
    cleaned_html_check = detectors.check_html(out, "qa")
    assert all(r != "html_tag_ratio" for _d, r in cleaned_html_check)


def test_clean_for_domain_qa_only():
    code_text = "x = a &amp;&amp; b  <stdio.h>"
    assert detectors.clean_for_domain(code_text, "code") == code_text
    assert detectors.clean_for_domain(code_text, "math") == code_text
    qa_out = detectors.clean_for_domain("<p>hello &amp; bye</p>", "qa")
    assert "<p>" not in qa_out and "&" in qa_out and "hello" in qa_out


def test_clean_inline_html_preserves_escaped_code_entities():
    raw = 'use <code>x &lt; 10</code> and <pre>y &gt; 5</pre>'
    out = detectors.clean_inline_html(raw)
    assert "x < 10" in out and "y > 5" in out
    assert "<code>" not in out
