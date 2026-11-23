import errno
import os
from pathlib import Path
import stat
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import audit_1t_token_streams as auditor


def test_fsync_covers_file_bytes_metadata_and_publication_directories(tmp_path, monkeypatch):
    paths = [tmp_path / "train" / "tokens.bin", tmp_path / "val" / "meta.json"]
    for path in paths:
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"immutable publication")
    real_fsync, observed = os.fsync, []

    def record(descriptor):
        observed.append(stat.S_ISDIR(os.fstat(descriptor).st_mode))
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", record)
    result = auditor.checked_publication_fsync(tmp_path, paths)
    assert observed == [False, False, True, True]
    assert result["status"] == "PASS_CHECKED_FSYNC"
    assert len(result["files"]) == len(result["directories"]) == 2
    auditor.verify_publication_unchanged(tmp_path, result)


@pytest.mark.parametrize("directory", [False, True])
def test_writeback_error_in_file_or_directory_aborts_admission(tmp_path, monkeypatch, directory):
    path = tmp_path / "tokens.bin"
    path.write_bytes(b"tokens")
    real_fsync = os.fsync

    def fail(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode) == directory:
            raise OSError(errno.EIO, "injected fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError, match="injected fsync failure"):
        auditor.checked_publication_fsync(tmp_path, [path])


def test_close_error_is_not_silently_admitted(tmp_path, monkeypatch):
    path = tmp_path / "tokens.bin"
    path.write_bytes(b"tokens")
    real_close = os.close

    def fail(descriptor):
        real_close(descriptor)
        raise OSError(errno.EIO, "injected close failure")

    monkeypatch.setattr(os, "close", fail)
    with pytest.raises(OSError, match="injected close failure"):
        auditor.checked_publication_fsync(tmp_path, [path])


def test_atomic_replacement_during_fsync_is_rejected(tmp_path, monkeypatch):
    path, replacement = tmp_path / "tokens.bin", tmp_path / "replacement"
    path.write_bytes(b"same bytes")
    replacement.write_bytes(b"same bytes")
    real_fsync = os.fsync

    def replace(descriptor):
        real_fsync(descriptor)
        os.replace(replacement, path)

    monkeypatch.setattr(os, "fsync", replace)
    with pytest.raises(ValueError, match="changed during checked fsync"):
        auditor.checked_publication_fsync(tmp_path, [path])


def test_replacement_after_fsync_invalidates_full_audit(tmp_path):
    path, replacement = tmp_path / "tokens.bin", tmp_path / "replacement"
    path.write_bytes(b"same bytes")
    evidence = auditor.checked_publication_fsync(tmp_path, [path])
    replacement.write_bytes(b"same bytes")
    os.replace(replacement, path)
    with pytest.raises(ValueError, match="changed during audit"):
        auditor.verify_publication_unchanged(tmp_path, evidence)


def test_directory_replacement_is_rejected_even_when_file_inode_survives(tmp_path):
    directory = tmp_path / "publication"
    directory.mkdir()
    path = directory / "tokens.bin"
    path.write_bytes(b"same inode and bytes")
    evidence = auditor.checked_publication_fsync(tmp_path, [path])
    old = tmp_path / "old"
    directory.rename(old)
    directory.mkdir()
    os.link(old / "tokens.bin", path)
    with pytest.raises(ValueError, match="publication directory changed"):
        auditor.verify_publication_unchanged(tmp_path, evidence)


def test_symlink_cannot_substitute_for_a_published_file(tmp_path):
    target, link = tmp_path / "target", tmp_path / "tokens.bin"
    target.write_bytes(b"tokens")
    link.symlink_to(target)
    with pytest.raises(OSError):
        auditor.checked_publication_fsync(tmp_path, [link])


def test_fifo_is_rejected_without_waiting_for_a_writer(tmp_path):
    path = tmp_path / "tokens.bin"
    os.mkfifo(path)
    with pytest.raises(ValueError, match="requires a regular file"):
        auditor.checked_publication_fsync(tmp_path, [path])
