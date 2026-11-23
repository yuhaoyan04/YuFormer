from __future__ import annotations

import json
import os
import time
from pathlib import Path

from yuformer_data.scale import driver


def test_stage_lease_rejects_a_second_writer(tmp_path: Path) -> None:
    first = driver._acquire_stage_lease(tmp_path, "s5")
    assert first is not None
    try:
        owner = json.loads((first.path / "owner.json").read_text())
        assert owner["stage"] == "s5"
        assert driver._acquire_stage_lease(tmp_path, "s5") is None
    finally:
        first.release()
    assert (first.path / f"released-{first.token}").is_file()
    second = driver._acquire_stage_lease(tmp_path, "s5")
    assert second is not None
    assert second.path != first.path
    second.release()


def test_stage_lease_recovers_an_expired_owner(
    tmp_path: Path, monkeypatch
) -> None:
    lock = tmp_path / ".driver_s5.lock"
    lock.mkdir()
    (lock / "owner.json").write_text(
        json.dumps({"stage": "s5", "token": "dead", "hostname": "old-pod", "pid": 7})
    )
    heartbeat = lock / "heartbeat-dead"
    heartbeat.touch()
    old = time.time() - 60
    os.utime(heartbeat, (old, old))
    monkeypatch.setattr(driver, "STAGE_LEASE_STALE_SEC", 1)

    lease = driver._acquire_stage_lease(tmp_path, "s5")
    assert lease is not None
    try:
        owner = json.loads((lease.path / "owner.json").read_text())
        assert owner["token"] == lease.token
        assert not list(tmp_path.glob(".driver_s5.lock.stale-*"))
        assert (lock / "owner.json").is_file(), "legacy generation remains immutable"
    finally:
        lease.release()

