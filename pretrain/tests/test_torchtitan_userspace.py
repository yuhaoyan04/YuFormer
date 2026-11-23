import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "backends" / "torchtitan" / "validate_userspace.py"
SPEC = importlib.util.spec_from_file_location("torchtitan_userspace", SCRIPT)
USERSPACE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(USERSPACE)


def test_version_tuple_accepts_local_wheel_suffix():
    assert USERSPACE.version_tuple("2.14.0+cu126") == (2, 14, 0)
    assert USERSPACE.version_tuple(None) == ()


def test_broken_shebangs_detects_relocated_virtualenv(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    valid = bin_dir / "valid"
    valid.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    broken = bin_dir / "broken"
    broken.write_text("#!/path/that/does/not/exist/python\n", encoding="utf-8")
    (bin_dir / "python").symlink_to("/bin/sh")
    assert USERSPACE.broken_shebangs(tmp_path) == ["broken"]


def test_userspace_report_publish_is_atomic(tmp_path):
    output = tmp_path / "health/userspace.json"
    USERSPACE.atomic_json(output, {"status": "ready"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "ready"}
    assert list(output.parent.glob(f".{output.name}.*")) == []
