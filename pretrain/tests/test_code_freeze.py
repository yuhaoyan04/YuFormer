import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "pretrain/scripts/freeze_code_candidate.py"
SPEC = importlib.util.spec_from_file_location("freeze_code_candidate", SCRIPT)
FREEZE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(FREEZE)


def test_code_freeze_covers_formal_pretrain_source_and_operations_docs():
    files = set(FREEZE.collect_files(ROOT))
    assert {
        "pretrain/src/yuformer_pretrain/trainer.py",
        "pretrain/backends/torchtitan/prepare_env.sh",
        "pretrain/backends/torchtitan/yuformer_ext/model.py",
        "pretrain/docs/HOST_DRIVER_UPGRADE.md",
        "pretrain/docs/SOURCE_PROVENANCE.md",
        "pretrain/README.md",
        "pretrain/pyproject.toml",
        "model/src/yuformer_model/model.py",
        "model/src/yuformer_model/attention.py",
    } <= files


def test_code_freeze_excludes_runtime_reports_and_caches():
    files = set(FREEZE.collect_files(ROOT))
    assert "pretrain/tests/verify/verify_report.json" not in files
    assert not any("__pycache__" in path or ".pytest_cache" in path for path in files)


def test_code_freeze_identity_is_deterministic():
    first = FREEZE.build_manifest(ROOT)
    second = FREEZE.build_manifest(ROOT)
    assert first["files"] == second["files"]
    assert first["candidate_sha256"] == second["candidate_sha256"]
