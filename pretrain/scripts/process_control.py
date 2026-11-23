"""Terminate an owned launcher tree across lease/torchrun session boundaries."""
import os
import signal
from pathlib import Path


def descendant_processes(root_pid: int) -> list[tuple[int, int]]:
    processes = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text()
            fields = raw[raw.rfind(")") + 2:].split()
            if fields[0] != "Z":
                processes[int(entry.name)] = (int(fields[1]), int(fields[19]))
        except (OSError, ValueError, IndexError):
            continue
    descendants, frontier = [], [root_pid]
    while frontier:
        parent = frontier.pop()
        for pid, (ppid, started) in processes.items():
            if pid == parent or ppid != parent:
                continue
            descendants.append((pid, started))
            frontier.append(pid)
    if root_pid in processes:
        descendants.insert(0, (root_pid, processes[root_pid][1]))
    return descendants


def terminate_process_tree(root_pid: int, signum: int = signal.SIGKILL) -> list[int]:
    targets = descendant_processes(root_pid)
    signalled = []
    # Children have separate process groups under both RunLease and torchrun.
    # Kill descendants before ancestors so none survive as untracked trainers.
    for pid, started in reversed(targets):
        if pid == os.getpid():
            raise ValueError("refusing to terminate the caller")
        try:
            raw = Path(f"/proc/{pid}/stat").read_text()
            fields = raw[raw.rfind(")") + 2:].split()
            if int(fields[19]) == started:
                os.kill(pid, signum)
                signalled.append(pid)
        except (OSError, ValueError, IndexError):
            continue
    return signalled
