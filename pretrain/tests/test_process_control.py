import errno
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from artifact_archive import move_artifact
from process_control import descendant_processes, terminate_process_tree


def test_kill_reaches_child_in_another_session(tmp_path):
    receipt = tmp_path / "child.json"
    program = (
        "import json, subprocess, sys, time; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True); "
        f"open({str(receipt)!r},'w').write(json.dumps(child.pid)); time.sleep(60)"
    )
    parent = subprocess.Popen([sys.executable, "-c", program], start_new_session=True)
    try:
        deadline = time.monotonic() + 5
        while not receipt.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        child = json.loads(receipt.read_text())
        assert os.getpgid(parent.pid) != os.getpgid(child)
        assert {parent.pid, child} <= {pid for pid, _ in descendant_processes(parent.pid)}
        assert {parent.pid, child} <= set(terminate_process_tree(parent.pid, signal.SIGKILL))
        parent.wait(timeout=5)
        deadline = time.monotonic() + 5
        while Path(f"/proc/{child}/stat").exists() and time.monotonic() < deadline:
            raw = Path(f"/proc/{child}/stat").read_text()
            if raw[raw.rfind(")") + 2:].split()[0] == "Z":
                break
            time.sleep(0.02)
        assert not descendant_processes(child)
    finally:
        terminate_process_tree(parent.pid)
        parent.wait(timeout=5)


def test_archive_falls_back_when_nfs_forbids_directory_rename(monkeypatch, tmp_path):
    source, destination = tmp_path / "source", tmp_path / "archive"
    (source / "nested").mkdir(parents=True)
    (source / "nested/evidence.json").write_text('{"status":"FAIL"}')
    replace = os.replace

    def nfs_replace(left, right):
        if Path(left).is_dir():
            raise PermissionError(errno.EPERM, "NFS denies directory rename")
        return replace(left, right)

    monkeypatch.setattr(os, "replace", nfs_replace)
    move_artifact(source, destination)
    assert (destination / "nested/evidence.json").read_text() == '{"status":"FAIL"}'
    assert not list(source.rglob("*.json"))
