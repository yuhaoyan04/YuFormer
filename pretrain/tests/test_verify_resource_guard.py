import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


SOURCE = Path(__file__).resolve().parent / "verify/conftest.py"
SPEC = importlib.util.spec_from_file_location("verify_resource_guard", SOURCE)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


@pytest.mark.parametrize("enabled,visible", [(None, "0"), ("1", "")])
def test_verify_cannot_spawn_gpu_children_without_opt_in_and_visibility(monkeypatch, tmp_path, enabled, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    if enabled is None:
        monkeypatch.delenv("YUFORMER_RUN_GPU_VERIFY", raising=False)
    else:
        monkeypatch.setenv("YUFORMER_RUN_GPU_VERIFY", enabled)
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *a, **k: pytest.fail("GPU child was started"))
    with pytest.raises(pytest.skip.Exception):
        guard.run_trainer(tmp_path / "unused.yaml")


def test_verify_refuses_gpu_outside_parent_mask(monkeypatch, tmp_path):
    monkeypatch.setenv("YUFORMER_RUN_GPU_VERIFY", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2")
    monkeypatch.setattr(guard, "VGPU", "3")
    with pytest.raises(ValueError, match="outside CUDA_VISIBLE_DEVICES"):
        guard.run_trainer(tmp_path / "unused.yaml")


def test_cpu_collection_skips_only_gpu_integration_directory(monkeypatch):
    monkeypatch.delenv("YUFORMER_RUN_GPU_VERIFY", raising=False)
    markers = [[], []]
    items = [SimpleNamespace(fspath=path, add_marker=markers[index].append) for index, path
             in enumerate((SOURCE.parent / "test_v1_resume.py", SOURCE.parent.parent / "test_checkpoint.py"))]
    guard.pytest_collection_modifyitems(None, items)
    assert len(markers[0]) == 1
    assert markers[1] == []
