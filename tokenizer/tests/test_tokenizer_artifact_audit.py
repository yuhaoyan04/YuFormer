import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_tokenizer_artifact.py"
SPEC = importlib.util.spec_from_file_location("tokenizer_artifact_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_missing_artifacts_fail(tmp_path):
    result = MODULE.audit(tmp_path)
    assert result["status"] == "FAIL"
    assert "tokenizer.json" in result["missing"]
