"""Preserve invalidated evidence on NFS mounts that forbid directory rename."""
import errno
import os
from pathlib import Path


def move_artifact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(source, destination)
        return
    except OSError as exc:
        if exc.errno not in {errno.EPERM, errno.EACCES, errno.EXDEV} or not source.is_dir():
            raise
    destination.mkdir(exist_ok=True)
    for child in list(source.iterdir()):
        move_artifact(child, destination / child.name)
    try:
        source.rmdir()
    except OSError as exc:
        if exc.errno not in {errno.EPERM, errno.EACCES}:
            raise
