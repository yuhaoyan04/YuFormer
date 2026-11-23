import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from yuformer_eval.audit import audit_result_manifest


def _manifest(samples_sha256, contamination_sha256):
    return {
        "schema_version": 1,
        "backend": {"name": "lm-evaluation-harness", "commit": "a" * 40},
        "model": {
            "checkpoint_sha256": "b" * 64,
            "tokenizer_sha256": "c" * 64,
            "chat_template_sha256": "d" * 64,
        },
        "run": {"seed": 1234, "limit": None, "fewshot_as_multiturn": False},
        "tasks": [
            {
                "name": "fixture",
                "dataset_revision": "e" * 40,
                "task_config_sha256": "f" * 64,
                "prompt_template_sha256": "1" * 64,
                "n_samples": 2,
                "value": 0.5,
                "higher_is_better": True,
                "contamination": {
                    "checked": True,
                    "report_path": "contamination.json",
                    "report_sha256": contamination_sha256,
                },
            }
        ],
        "artifacts": {"samples_path": "samples.jsonl", "samples_sha256": samples_sha256},
    }


def test_complete_manifest_and_raw_samples_pass(tmp_path):
    samples = tmp_path / "samples.jsonl"
    samples.write_text('{"prediction":"x"}\n')
    contamination = tmp_path / "contamination.json"
    contamination.write_text('{"matches":0}\n')
    manifest = _manifest(
        hashlib.sha256(samples.read_bytes()).hexdigest(),
        hashlib.sha256(contamination.read_bytes()).hexdigest(),
    )
    assert audit_result_manifest(manifest, root=tmp_path)["status"] == "PASS"


def test_partial_unpinned_or_tampered_results_fail(tmp_path):
    samples = tmp_path / "samples.jsonl"
    samples.write_text("first")
    contamination = tmp_path / "contamination.json"
    contamination.write_text('{"matches":0}\n')
    manifest = _manifest(
        hashlib.sha256(samples.read_bytes()).hexdigest(),
        hashlib.sha256(contamination.read_bytes()).hexdigest(),
    )
    manifest["run"]["limit"] = 10
    manifest["tasks"][0]["dataset_revision"] = "main"
    samples.write_text("tampered")
    result = audit_result_manifest(manifest, root=tmp_path)
    assert result["status"] == "FAIL"
    assert any("limit" in error for error in result["errors"])
    assert any("dataset_revision" in error for error in result["errors"])
    assert any("mismatch" in error for error in result["errors"])
