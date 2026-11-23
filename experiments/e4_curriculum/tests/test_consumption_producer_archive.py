import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from adopt_training_recipe import bind_consumption_producers


def test_new_final_audit_code_cannot_replace_the_actual_consumption_producer(tmp_path):
    sources = tmp_path/'experiments/e4_curriculum/scripts'
    sources.mkdir(parents=True)
    audit = sources/'audit_1t_overlap.py'
    controller = sources/'tokenize_1t_orders.py'
    audit.write_bytes(b'actual admitted-window extraction implementation')
    controller.write_bytes(b'actual twelve-stream production controller')
    audit_sha = hashlib.sha256(audit.read_bytes()).hexdigest()
    controller_sha = hashlib.sha256(controller.read_bytes()).hexdigest()
    report = {'stable':{'audit_tool_sha256':audit_sha, 'tokenization_tool_sha256':controller_sha}}
    assert len(bind_consumption_producers(tmp_path, report)) == 2
    archive = tmp_path/'report/source_snapshots'/f'audit_1t_overlap-{audit_sha}.py'
    archive.parent.mkdir(parents=True)
    original = audit.read_bytes()
    audit.write_bytes(b'new final full-overlap window interpretation')
    with pytest.raises(FileNotFoundError):
        bind_consumption_producers(tmp_path, report)
    archive.write_bytes(original)
    bound = bind_consumption_producers(tmp_path, report)
    assert str(archive.relative_to(tmp_path)) in bound
    assert str(audit.relative_to(tmp_path)) not in bound
    archive.write_bytes(b'corrupted historical implementation')
    with pytest.raises(ValueError, match='byte identity'):
        bind_consumption_producers(tmp_path, report)
