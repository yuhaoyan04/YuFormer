import importlib.util
import os
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/assert_run_lease.py"
SPEC = importlib.util.spec_from_file_location("assert_run_lease", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_assert_run_lease_accepts_owner_and_rejects_missing_token(monkeypatch, tmp_path):
    lease = MODULE.RunLease(tmp_path, "owner", heartbeat_seconds=3600).acquire()
    try:
        monkeypatch.setenv("YUFORMER_RUN_LEASE_TOKEN", lease.token)
        monkeypatch.setenv("YUFORMER_RUN_LEASE_GENERATION", str(lease.path))
        MODULE.assert_from_environment(tmp_path)
        monkeypatch.delenv("YUFORMER_RUN_LEASE_TOKEN")
        with pytest.raises(MODULE.ConfigError, match="absent"):
            MODULE.assert_from_environment(tmp_path)
    finally:
        lease.release()


def test_assert_run_lease_rejects_generation_outside_output(monkeypatch, tmp_path):
    outside = tmp_path / "outside/generation-00000000000000000001"
    outside.mkdir(parents=True)
    monkeypatch.setenv("YUFORMER_RUN_LEASE_TOKEN", "token")
    monkeypatch.setenv("YUFORMER_RUN_LEASE_GENERATION", str(outside))
    with pytest.raises(MODULE.ConfigError, match="outside"):
        MODULE.assert_from_environment(tmp_path / "run")
