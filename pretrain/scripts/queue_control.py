"""Live GPU allocations and completion-triggered watcher notifications."""
from __future__ import annotations

import json
import os
import signal
from pathlib import Path


def read_allocation(path: Path, group: str, allowed_gpus: set[int] | None = None) -> dict:
    value = json.loads(path.read_text())
    gpus = value["gpus"]
    shared = value.get("shared_gpus", [])
    if (value.get("group") != group or not isinstance(gpus, list)
            or any(type(gpu) is not int or not 0 <= gpu < 8 for gpu in gpus)
            or len(gpus) != len(set(gpus)) or not isinstance(shared, list)
            or not set(shared).issubset(gpus)
            or type(value.get("max_parallel")) is not int
            or not 0 <= value["max_parallel"] <= len(gpus)):
        raise ValueError("invalid live GPU allocation")
    if allowed_gpus is not None and not set(gpus).issubset(allowed_gpus):
        raise ValueError('GPU allocation exceeds the user resource policy')
    if group == "e1_performance" and shared:
        raise ValueError("performance measurements cannot receive shared GPU slots")
    return value


def notify_watcher(root: Path) -> bool:
    try:
        record = json.loads((root / "data/health/pretrain-watch-wakeup.json").read_text())
        pid = int(record["pid"])
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        expected = {os.fsencode(str(root / "pretrain/scripts/pretrain_readiness_watch.py")),
                    b"pretrain/scripts/pretrain_readiness_watch.py"}
        if record.get("signal") != "SIGUSR1" or not expected.intersection(argv):
            return False
        os.kill(pid, signal.SIGUSR1)
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False
