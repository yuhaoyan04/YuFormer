import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_with_lease.py"
SPEC = importlib.util.spec_from_file_location("run_with_lease", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_run_with_lease_returns_child_status_and_releases(tmp_path):
    rc = MODULE.run(
        tmp_path, "first", [sys.executable, "-c", "raise SystemExit(7)"],
        poll_seconds=0.01,
    )
    assert rc == 7
    assert MODULE.run(
        tmp_path, "successor", [sys.executable, "-c", "pass"], poll_seconds=0.01
    ) == 0


def test_run_with_lease_rejects_concurrent_writer(tmp_path):
    first = MODULE.RunLease(tmp_path, "owner", heartbeat_seconds=3600).acquire()
    try:
        assert MODULE.run(
            tmp_path, "competitor", [sys.executable, "-c", "pass"], poll_seconds=0.01
        ) == 4
    finally:
        first.release()


def test_run_with_lease_terminates_child_when_generation_is_lost(monkeypatch, tmp_path):
    class LostLease:
        released = False
        token = "lost-token"
        path = tmp_path / ".run.lease/generation-00000000000000000001"

        def acquire(self):
            return self

        def renew(self):
            raise MODULE.ConfigError("generation fence")

        def release(self):
            self.released = True

    lease = LostLease()
    monkeypatch.setattr(MODULE, "RunLease", lambda *args, **kwargs: lease)
    rc = MODULE.run(
        tmp_path, "lost", [sys.executable, "-c", "import time; time.sleep(60)"],
        poll_seconds=0.01,
    )
    assert rc == 74
    assert lease.released is True
