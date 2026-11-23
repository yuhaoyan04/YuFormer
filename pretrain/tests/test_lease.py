import json
import os
import socket
import time

import pytest

from yuformer_pretrain.config import ConfigError
from yuformer_pretrain.lease import RunLease


def test_same_host_zombie_owner_does_not_block_recovery():
    child = os.fork()
    if child == 0:
        os._exit(0)
    try:
        deadline = time.monotonic() + 5
        from pathlib import Path
        while time.monotonic() < deadline:
            if Path(f"/proc/{child}/stat").read_text().split()[2] == "Z":
                break
            time.sleep(0.01)
        assert RunLease._same_host_owner_dead({"hostname": socket.gethostname(), "pid": child})
    finally:
        os.waitpid(child, 0)


def test_run_lease_rejects_concurrent_writer_and_releases(tmp_path):
    first = RunLease(tmp_path, "run-a").acquire()
    try:
        owner = json.loads((tmp_path / "run.lock").read_text())
        assert owner["run_name"] == "run-a"
        assert owner["pid"] > 0
        with pytest.raises(ConfigError, match="另一个训练任务占用"):
            RunLease(tmp_path, "run-b").acquire()
    finally:
        first.release()

    second = RunLease(tmp_path, "run-b").acquire()
    second.release()


def test_generation_fence_rejects_superseded_owner(tmp_path):
    first = RunLease(tmp_path, "run-a", heartbeat_seconds=3600).acquire()
    try:
        successor = tmp_path / ".run.lease" / "generation-00000000000000000002"
        successor.mkdir()
        token = "successor"
        (successor / "owner.json").write_text(json.dumps({"token": token, "started_unix": time.time()}))
        (successor / f"heartbeat-{token}").touch()
        with pytest.raises(ConfigError, match="generation fence"):
            first.assert_owner()
    finally:
        first.release()


def test_run_lease_immediately_recovers_dead_same_host_owner(tmp_path):
    generation = tmp_path / ".run.lease" / "generation-00000000000000000001"
    generation.mkdir(parents=True)
    token = "dead-owner"
    owner = {
        "token": token,
        "hostname": socket.gethostname(),
        "pid": 2**30,
        "started_unix": time.time(),
    }
    (generation / "owner.json").write_text(json.dumps(owner))
    (generation / f"heartbeat-{token}").touch()

    successor = RunLease(tmp_path, "recovered").acquire()
    try:
        assert successor.path is not None
        assert successor.path.name == "generation-00000000000000000002"
    finally:
        successor.release()
