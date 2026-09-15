from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from yuformer_data.cli import run_jsonl
from yuformer_data.models import DataCategory, Stage

config = {
    "min_chars": 16,
    "max_chars": 2_000_000,
    "context_lengths": [4096, 16384, 65536],
    "near_dedup": True,
    "near_dedup_threshold": 3,
    "quality_policies": {
        "web": {"score_field": "quality_mean", "keep_min": 4.0, "review_min": 3.0},
        "math": {"score_field": "quality_score", "keep_min": 0.8, "review_min": 0.5,
                 "hard_error_field": "hard_error_score", "hard_error_max": 3},
    },
}

test_data = [
    {"id": "doc1", "text": "This is a valid document about machine learning. " * 5, "source": "web", "quality_mean": 4.5},
    {"id": "doc2", "text": "short", "source": "web"},
    {"id": "doc3", "text": "This is a valid document about machine learning. " * 5, "source": "web", "quality_mean": 4.5},
    {"id": "doc4", "text": "This is another unique document about mathematics and calculus. " * 5, "source": "web", "quality_mean": 2.0},
]

import tempfile, json, os
tmp = Path(tempfile.mkdtemp())
input_path = tmp / "input.jsonl"
output_path = tmp / "output.jsonl"
manifest_path = tmp / "output.manifest.json"
summary_path = tmp / "output.summary.json"

with input_path.open("w") as f:
    for row in test_data:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

count = run_jsonl(
    str(input_path),
    str(output_path),
    stage=Stage.PRETRAIN,
    category=DataCategory.WEB,
    dataset="test-web",
    config=config,
    manifest_path=str(manifest_path),
    summary_path=str(summary_path),
)

print(f"Input: {len(test_data)} rows")
print(f"Output: {count} records")

with output_path.open() as f:
    for line in f:
        rec = json.loads(line)
        print(f"  {rec['source_id']:8s} decision={rec['decision']:6s} "
              f"context={rec['buckets'].get('context','?')} "
              f"length={rec['buckets'].get('length','?')} "
              f"reasons={rec['reasons']}")

with manifest_path.open() as f:
    manifest = json.load(f)
    print(f"Manifest: {manifest['records']} records, decisions={manifest['decisions']}")

with summary_path.open() as f:
    summary = json.load(f)
    print(f"Summary counters: {dict(summary['counters'])}")

if count == 1:
    print("\nstatus: ok")
else:
    print(f"\nstatus: EXPECTED 1 output record (doc2 too_short, doc3 exact_dup, doc4 quality<3.0), got {count}")
    sys.exit(1)
