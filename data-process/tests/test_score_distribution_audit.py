import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


SCRIPT = Path(__file__).parents[1] / "scripts" / "audit_score_distributions.py"
SPEC = importlib.util.spec_from_file_location("audit_score_distributions", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


def _fixture(tmp_path: Path, *, chinese_rank_shift: float = 0.0, endpoint=False):
    root = tmp_path / "s4"
    for source, language in (("english-source", "en"), ("chinese-source", "zh-cn")):
        scores = [0.2 + (i % 600) / 1000 for i in range(2_000)]
        ranks = [0.2 + (i % 600) / 1000 for i in range(2_000)]
        if language.startswith("zh"):
            ranks = [max(0.0, value - chinese_rank_shift) for value in ranks]
        if endpoint and source == "english-source":
            scores = [0.0] * 2_000
        path = root / source / "part.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(
            pa.table({"lang": [language] * 2_000, "hybrid_v2": scores, "domain_quantile": ranks}),
            path,
        )
    scope = tmp_path / "scope.yaml"
    scope.write_text(
        yaml.safe_dump(
            {"audit_scope": {"included_sources": ["english-source", "chinese-source"]}}
        )
    )
    return root, scope


def test_distribution_audit_passes_healthy_groups(tmp_path):
    root, scope = _fixture(tmp_path)
    result = audit.audit_score_distributions(root, scope, bins=1_000, batch_size=333)
    assert result["status"] == "PASS"
    assert result["files_scanned"] == 2
    assert result["per_source"]["english-source"]["rows"] == 2_000
    assert result["per_source"]["chinese-source"]["rows"] == 2_000
    assert sum(row["rows"] for row in result["per_language"].values()) == 4_000
    assert result["language_bias"]["passed"] is True


def test_distribution_audit_rejects_endpoint_collapse_and_language_bias(tmp_path):
    root, scope = _fixture(tmp_path, chinese_rank_shift=0.2, endpoint=True)
    result = audit.audit_score_distributions(root, scope, bins=1_000)
    assert result["status"] == "FAIL"
    assert result["per_source"]["english-source"]["endpoint_ratio"] == 1.0
    assert result["language_bias"]["passed"] is False
