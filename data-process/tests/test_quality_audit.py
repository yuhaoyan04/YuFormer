import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_data_quality_20261012.py"
SPEC = importlib.util.spec_from_file_location("audit_data_quality", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class _Index:
    names = ["mmlu", "ceval"]

    def hit_codes_batch(self, shingles):
        mapping = {10: 0, 20: 1, 21: 1}
        return np.asarray([mapping.get(int(value), -1) for value in shingles], dtype=np.int16)


def test_decontam_audit_uses_distinct_per_benchmark_thresholds():
    shingles = np.asarray([10, 10, 20, 999], dtype="<u8")
    raw, qualified = MODULE.decontam_hit_summary(
        shingles, _Index(), {"mmlu": 1, "ceval": 2}
    )
    assert raw == {"mmlu": 1, "ceval": 1}
    assert qualified == ["mmlu"]

    raw, qualified = MODULE.decontam_hit_summary(
        np.asarray([20, 20, 21], dtype="<u8"), _Index(), {"ceval": 2}
    )
    assert raw == {"ceval": 2}
    assert qualified == ["ceval"]
