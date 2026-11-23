import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "backends" / "torchtitan" / "preflight.py"
SPEC = importlib.util.spec_from_file_location("torchtitan_preflight", SCRIPT)
preflight = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(preflight)


def _ready_probe():
    return {
        "python": "3.12.8",
        "torch": "2.14.0+cu126",
        "torch_cuda": "12.6",
        "torchvision": "0.29.0",
        "torchao": "0.18.0",
        "torch_checkpointing": "0.1.0",
        "triton": "3.8.0",
        "nvidia_driver": "535.161.08",
        "gpu_capability": [9, 0],
        "cuda_available": True,
        "fsdp2": True,
        "dcp": True,
        "torchtitan": "0.3.0",
        "torchtitan_commit": preflight.LOCK["torchtitan"]["commit"],
        "torchtitan_source_dirty": False,
    }


def test_exact_locked_environment_is_ready():
    result = preflight.evaluate_probe(_ready_probe())
    assert result["status"] == "ready"
    assert result["failed_checks"] == []


def test_unqualified_version_or_dirty_source_is_blocked():
    for field, value, expected_check in (
        ("python", "3.13.0", "python_3_12"),
        ("torch", "2.15.0", "pytorch_2_14_0"),
        ("torch_cuda", "12.7", "cuda_build_matches_lock"),
        ("torchvision", "0.30.0", "torchvision_0_29_0"),
        ("torchao", "0.19.0", "torchao_0_18_0"),
        ("torch_checkpointing", "0.2.0", "torch_checkpointing_0_1_0"),
        ("triton", "3.9.0", "triton_3_8_0"),
        ("torchtitan_source_dirty", True, "clean_pinned_source"),
    ):
        probe = _ready_probe()
        probe[field] = value
        result = preflight.evaluate_probe(probe)
        assert result["status"] == "blocked"
        assert expected_check in result["failed_checks"]


def test_preflight_report_publish_is_atomic(tmp_path):
    target = tmp_path / "health/preflight.json"
    target.parent.mkdir()
    target.write_text('{"old": true}\n')
    preflight.atomic_write_text(target, '{"status": "ready"}\n')
    assert json.loads(target.read_text()) == {"status": "ready"}
    assert list(target.parent.glob(".preflight.json.tmp-*")) == []
