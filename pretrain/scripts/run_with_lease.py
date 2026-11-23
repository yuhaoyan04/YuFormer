#!/usr/bin/env python3
"""Run one process group under a cross-pod generation lease."""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pretrain/src"))

from yuformer_pretrain.config import ConfigError  # noqa: E402
from yuformer_pretrain.lease import RunLease  # noqa: E402


def terminate_group(process: subprocess.Popen[bytes], signum: int = signal.SIGTERM) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        return


def run(out_dir: Path, run_name: str, command: list[str], *, poll_seconds: float = 2.0) -> int:
    if not command:
        raise ValueError("a command is required")
    try:
        # The foreground loop owns renewal so loss of the generation fence can
        # synchronously terminate the child group before it publishes output.
        lease = RunLease(
            out_dir, run_name, heartbeat_seconds=3600, stale_seconds=10 * 60
        ).acquire()
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 4

    process: subprocess.Popen[bytes] | None = None
    previous_handlers: dict[int, object] = {}
    try:
        child_env = dict(os.environ)
        child_env.update({
            "YUFORMER_RUN_LEASE_HELD": "1",
            "YUFORMER_RUN_LEASE_TOKEN": lease.token,
            "YUFORMER_RUN_LEASE_GENERATION": str(lease.path),
        })
        process = subprocess.Popen(command, start_new_session=True, env=child_env)

        def forward(signum: int, _frame: object) -> None:
            assert process is not None
            terminate_group(process, signum)

        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous_handlers[signum] = signal.signal(signum, forward)
        while process.poll() is None:
            try:
                lease.renew()
            except (ConfigError, OSError) as exc:
                print(f"run lease lost; terminating child group: {exc}", file=sys.stderr)
                terminate_group(process)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    terminate_group(process, signal.SIGKILL)
                    process.wait(timeout=30)
                return 74
            try:
                return process.wait(timeout=poll_seconds)
            except subprocess.TimeoutExpired:
                pass
        return int(process.returncode or 0)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if process is not None and process.poll() is None:
            terminate_group(process, signal.SIGKILL)
            process.wait(timeout=30)
        lease.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    return run(args.out_dir, args.run_name, command)


if __name__ == "__main__":
    raise SystemExit(main())
