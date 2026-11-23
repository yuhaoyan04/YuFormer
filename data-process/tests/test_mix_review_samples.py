from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_mix_review_samples.py"
SPEC = importlib.util.spec_from_file_location("build_mix_review_samples", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_quantile_rows_respect_contiguous_domains_and_nearest_rank():
    scores = np.array([0.4, 0.1, 0.3, 0.2, 9.0, 7.0, 8.0], dtype=np.float32)
    got = MOD.quantile_row_indices(
        scores,
        {"academic": 4, "web": 3},
        [0.25, 0.5, 0.75],
    )
    assert [(r["domain"], r["score"]) for r in got] == [
        ("academic", np.float32(0.2)),
        ("academic", np.float32(0.3)),
        ("academic", np.float32(0.3)),
        ("web", np.float32(8.0)),
        ("web", np.float32(8.0)),
        ("web", np.float32(9.0)),
    ]


def test_quantile_rows_tie_break_to_first_row():
    got = MOD.quantile_row_indices(
        np.array([0.1, 0.2, 0.2, 0.3], dtype=np.float32),
        {"web": 4},
        [0.5],
    )
    assert got[0]["global_row"] == 1
    assert got[0]["tie_count"] == 2


def test_redacted_excerpt_masks_pii_before_writing_review():
    text = "mail me at alice@example.com or 13812345678, id 11010519491231002X"
    got = MOD.redacted_excerpt(text, 400)
    assert "alice@example.com" not in got
    assert "13812345678" not in got
    assert "11010519491231002X" not in got
    assert "[EMAIL_REDACTED]" in got
    assert "[PHONE_REDACTED]" in got
    assert "[ID_REDACTED]" in got


def test_redacted_excerpt_samples_head_middle_and_tail():
    text = "H" * 100 + "M" * 100 + "T" * 100
    got = MOD.redacted_excerpt(text, 30)
    assert got.startswith("H" * 10)
    assert "M" * 10 in got
    assert got.endswith("T" * 10)
    assert got.count("source characters omitted") == 2


def test_markdown_labels_excerpt_as_truncated_and_full_text_as_training_input(tmp_path):
    report = {
        "mix_name": "mix_test",
        "generated_date": "20261019",
        "identity": {
            "manifest_sha256": "a",
            "members_sha256": "b",
            "tokenizer_sha256": "c",
        },
        "machine_verdict": "PASS",
        "n_samples": 1,
        "samples": [
            {
                "doc_id": "0123",
                "domain": "code",
                "quality_quantile": 0.5,
                "source": "source",
                "lang": "en",
                "n_tokens": 20,
                "retokenized_n_tokens": 20,
                "composite": 0.75,
                "automatic_flags": [],
                "excerpt": "first 400 chars",
                "excerpt_chars": 15,
                "excerpt_source_chars": 400,
                "text_chars": 1000,
                "excerpt_truncated": True,
            }
        ],
    }
    out = tmp_path / "review.md"
    MOD.write_markdown(out, report)
    text = out.read_text()
    assert "review-only excerpts" in text
    assert "review_source_chars=400/1000" in text
    assert "excerpt_truncated=true" in text
    assert "600 source characters are omitted" in text
    assert "Training uses the full text" in text


def test_markdown_renders_recorded_human_verdict(tmp_path):
    report = {
        "mix_name": "mix_test",
        "generated_date": "20261019",
        "identity": {
            "manifest_sha256": "a",
            "members_sha256": "b",
            "tokenizer_sha256": "c",
        },
        "machine_verdict": "PASS",
        "human_review_status": "PASS",
        "n_samples": 1,
        "samples": [
            {
                "doc_id": "0123",
                "domain": "books",
                "quality_quantile": 0.5,
                "source": "source",
                "lang": "en",
                "n_tokens": 20,
                "retokenized_n_tokens": 20,
                "composite": 0.75,
                "automatic_flags": ["char_run"],
                "excerpt": "text",
                "excerpt_chars": 4,
                "excerpt_source_chars": 4,
                "text_chars": 4,
                "excerpt_truncated": False,
                "human_verdict": "PASS",
                "human_reason": "Accepted | formatting metadata",
            }
        ],
    }
    out = tmp_path / "review.md"
    MOD.write_markdown(out, report)
    text = out.read_text()
    assert "Human gate: **PASS**" in text
    assert "| PASS | Accepted \\| formatting metadata |" in text
    assert "- Verdict: `[x] PASS  [ ] REJECT`" in text


def test_manifest_member_count_mismatch_fails():
    try:
        MOD.quantile_row_indices(np.array([0.1], dtype=np.float32), {"web": 2}, [0.5])
    except ValueError as exc:
        assert "manifest/member row mismatch" in str(exc)
    else:
        raise AssertionError("expected mismatch to fail")
