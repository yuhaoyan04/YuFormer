from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from yuformer_data.models import DataCategory, Decision, Sample, Stage
from yuformer_data.core import Pipeline, PipelineContext, Step
from yuformer_data.dedup import NearDeduper, hamming_distance, simhash64
from yuformer_data.training import (
    allocate_mix_quotas,
    assign_context_bucket,
    build_loss_mask,
    deterministic_shuffle_key,
    pack_tokenized_samples,
    prefix_context_mix,
    select_mixed_sample_ids,
    TokenizedSample,
)
from yuformer_data.pipelines import build_pipeline
from yuformer_data.steps.common import _normalize_text_value


def test_decision_monotone():
    s = Sample("ds", "1", Stage.PRETRAIN, DataCategory.WEB, {"text": "hello"})
    s.mark(Decision.REVIEW, "r1")
    assert s.decision == Decision.REVIEW
    s.mark(Decision.REVIEW, "r2")
    assert s.decision == Decision.REVIEW
    s.mark(Decision.DROP, "r3")
    assert s.decision == Decision.DROP
    s.mark(Decision.REVIEW, "r4")
    assert s.decision == Decision.DROP
    assert "r1" in s.reasons and "r3" in s.reasons
    print("test_decision_monotone: ok")


def test_normalize_text():
    assert _normalize_text_value("hello\r\nworld") == "hello\nworld"
    assert _normalize_text_value("a   b") == "a b"
    assert _normalize_text_value("a\n\n\n\nb") == "a\n\nb"
    print("test_normalize_text: ok")


def test_simhash():
    h1 = simhash64("the quick brown fox jumps over the lazy dog and runs away")
    h2 = simhash64("the quick brown fox jumps over the lazy dog and runs fast")
    h3 = simhash64("completely different text about random topics in science")
    assert hamming_distance(h1, h2) < hamming_distance(h1, h3)
    print(f"test_simhash: ok (d(h1,h2)={hamming_distance(h1,h2)}, d(h1,h3)={hamming_distance(h1,h3)})")


def test_near_dedup():
    dd = NearDeduper(threshold=3, bands=4)
    assert not dd.check_and_add("hello world this is a test document for dedup")
    assert not dd.check_and_add("totally different content about science and math")
    assert dd.check_and_add("hello world this is a test document for dedup!")
    print("test_near_dedup: ok")


def test_context_bucket():
    assert assign_context_bucket(100, [4096, 16384, 65536]) == "B4K"
    assert assign_context_bucket(5000, [4096, 16384, 65536]) == "B16K"
    assert assign_context_bucket(20000, [4096, 16384, 65536]) == "B64K"
    assert assign_context_bucket(70000, [4096, 16384, 65536]) == "gt_64K"
    print("test_context_bucket: ok")


def test_allocate_quotas():
    q = allocate_mix_quotas({"code": 0.5, "web": 0.3, "math": 0.2}, 11)
    assert sum(q.values()) == 11
    assert q == {"code": 6, "web": 3, "math": 2}
    print("test_allocate_quotas: ok")


def test_deterministic():
    k1 = deterministic_shuffle_key("id1", seed="test")
    k2 = deterministic_shuffle_key("id1", seed="test")
    assert k1 == k2
    print("test_deterministic: ok")


def test_packing():
    samples = [
        TokenizedSample(tokens=tuple(range(10)), source_id="a"),
        TokenizedSample(tokens=tuple(range(10)), source_id="b"),
        TokenizedSample(tokens=tuple(range(5)), source_id="c"),
    ]
    bins, over = pack_tokenized_samples(samples, sequence_length=15)
    assert len(over) == 0
    assert len(bins) >= 1
    assert all(b.cu_seqlens[0] == 0 for b in bins)
    print(f"test_packing: ok ({len(bins)} bins)")


def test_loss_mask():
    mask = build_loss_mask([1, 2, -100, 4, -100])
    assert mask == (1, 1, 0, 1, 0)
    print("test_loss_mask: ok")


def test_prefix_mix():
    base = {
        "B4K": ["a", "b", "c", "d", "e"],
        "B16K": ["f", "g", "h"],
        "B64K": ["i", "j"],
    }
    targets = {"B4K": 3, "B16K": 2, "B64K": 1}
    result = prefix_context_mix(base, targets, seed="test")
    all_ids = result["B4K"] + result["B16K"] + result["B64K"]
    assert len(set(all_ids)) == len(all_ids), "IDs should not overlap"
    assert len(result["B4K"]) == 3
    assert len(result["B16K"]) == 2
    assert len(result["B64K"]) == 1
    print(f"test_prefix_mix: ok (disjoint={len(set(all_ids)) == len(all_ids)})")


def test_select_mixed():
    sources = {
        "web": ["w1", "w2", "w3", "w4", "w5"],
        "code": ["c1", "c2", "c3"],
    }
    quotas = {"web": 3, "code": 2}
    ids = select_mixed_sample_ids(sources, quotas, seed="test")
    assert len(ids) == 5
    assert len(set(ids)) == 5
    print(f"test_select_mixed: ok ({len(ids)} ids, all unique)")


def test_pipeline_web():
    config = {
        "min_chars": 16,
        "context_lengths": [4096, 16384, 65536],
        "quality_policies": {
            "web": {"score_field": "quality_mean", "keep_min": 4.0, "review_min": 3.0},
        },
    }
    pipe = build_pipeline(DataCategory.WEB, config=config)
    sample = Sample(
        "test", "doc1", Stage.PRETRAIN, DataCategory.WEB,
        {"text": "This is a valid web document about science and technology. " * 5,
         "source": "web", "quality_mean": 4.5},
    )
    result = pipe.run(sample)
    assert result.decision == Decision.KEEP
    assert "context" in result.buckets
    assert "length" in result.buckets
    print(f"test_pipeline_web: ok (decision={result.decision.value}, context={result.buckets.get('context')})")


def test_pipeline_code():
    config = {"min_chars": 16, "context_lengths": [4096]}
    pipe = build_pipeline(DataCategory.CODE, config=config)
    sample = Sample(
        "test", "f1", Stage.PRETRAIN, DataCategory.CODE,
        {"text": "def add(a, b):\n    return a + b\n", "language": "python", "source": "github"},
    )
    result = pipe.run(sample)
    assert result.decision == Decision.KEEP
    assert result.buckets.get("language") == "python"
    print(f"test_pipeline_code: ok (decision={result.decision.value}, lang={result.buckets.get('language')})")


def test_pipeline_math():
    config = {"min_chars": 16, "context_lengths": [4096]}
    pipe = build_pipeline(DataCategory.MATH, config=config)
    sample = Sample(
        "test", "m1", Stage.PRETRAIN, DataCategory.MATH,
        {"question": "What is 2+2?", "solution": "We compute 2+2=4. Therefore the final answer is 4.",
         "source": "mathbook"},
    )
    result = pipe.run(sample)
    assert result.decision == Decision.KEEP
    print(f"test_pipeline_math: ok (decision={result.decision.value})")


if __name__ == "__main__":
    test_decision_monotone()
    test_normalize_text()
    test_simhash()
    test_near_dedup()
    test_context_bucket()
    test_allocate_quotas()
    test_deterministic()
    test_packing()
    test_loss_mask()
    test_prefix_mix()
    test_select_mixed()
    test_pipeline_web()
    test_pipeline_code()
    test_pipeline_math()
    print("\nAll tests passed!")
