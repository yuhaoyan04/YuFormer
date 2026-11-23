import hashlib
from pathlib import Path
import shutil

import pytest

from yuformer_eval.source_binding import actual_training_source


def test_source_drift_is_rejected_before_loading_a_gpu_model(tmp_path):
    root = Path(__file__).resolve().parents[2]
    validator = tmp_path/'pretrain/backends/torchtitan/validate_contract.py'
    validator.parent.mkdir(parents=True)
    shutil.copyfile(root/'pretrain/backends/torchtitan/validate_contract.py', validator)
    model = tmp_path/'model/src/decoder.py'
    model.parent.mkdir(parents=True)
    model.write_text('gain = 1\n')
    digest = hashlib.sha256()
    for path in sorted((validator, model)):
        digest.update(str(path.relative_to(tmp_path)).encode())
        digest.update(b'\0')
        digest.update(path.read_bytes())
    expected = digest.hexdigest()
    pins = actual_training_source(tmp_path, expected)
    assert str(model) in pins
    model.write_text('gain = 2\n')
    with pytest.raises(ValueError, match='differs from the actual trained checkpoint'):
        actual_training_source(tmp_path, expected)
