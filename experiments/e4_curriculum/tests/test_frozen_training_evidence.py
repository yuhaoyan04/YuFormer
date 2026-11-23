import hashlib
import json
import errno
import os
import stat
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from frozen_training_evidence import freeze_report_bindings, snapshot_files, production_output_dir, publish_checked
from training_data_origin import read_bound


def write(root, name, content):
    path = root/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reexporting_reports_cannot_change_frozen_proof_but_authorization_stays_live(tmp_path):
    report = 'data/health/torchtitan-qualification.json'
    auth = 'data/health/pretrain-user-full-pretrain-authorization.json'
    inputs = {report: write(tmp_path, report, '{"status":"PASS","time":1}'),
              auth: write(tmp_path, auth, '{"world_size":8}')}
    bound, archive = freeze_report_bindings(tmp_path, inputs)
    assert report not in bound and auth in bound
    write(tmp_path, report, '{"status":"PASS","time":2}')
    for name, digest in bound.items():
        read_bound(tmp_path/name, digest)
    saved = archive['files'][report]
    assert json.loads((tmp_path/saved['archive']).read_bytes())['time'] == 1
    write(tmp_path, auth, '{"world_size":4}')
    with pytest.raises(ValueError, match='byte identity'):
        read_bound(tmp_path/auth, bound[auth])


def test_historical_model_bytes_survive_production_model_replacement(tmp_path):
    name = 'model/configs/pretrain_final_model_configs.yaml'
    old_sha = write(tmp_path, name, 'parameters: 530107904\n')
    archive = snapshot_files(tmp_path, {str(tmp_path/name): old_sha})
    write(tmp_path, name, 'parameters: 1516600064\n')
    raw, digest = read_bound(tmp_path/archive['files'][name]['archive'], old_sha)
    assert digest == old_sha and b'530107904' in raw
    with pytest.raises(ValueError, match='byte identity'):
        snapshot_files(tmp_path, {name: old_sha})


def test_snapshots_are_idempotent_and_never_overwrite_corrupt_archived_objects(tmp_path):
    a, b = 'a.json', 'b.json'
    bindings = {a: write(tmp_path, a, 'same bytes'), b: write(tmp_path, b, 'same bytes')}
    first = snapshot_files(tmp_path, bindings)
    assert snapshot_files(tmp_path, dict(reversed(list(bindings.items())))) == first
    assert len(first['dependencies']) == 2
    (tmp_path/first['files'][a]['archive']).write_text('corrupt')
    with pytest.raises(ValueError, match='byte identity'):
        snapshot_files(tmp_path, bindings)


def test_existing_checkpoint_guard_uses_actual_declared_output_not_a_hardcoded_run(tmp_path):
    write(tmp_path, 'pretrain/configs/pretrain_base.env',
          'YUFORMER_RUNS_ROOT='+str(tmp_path/'custom-runs')+'\nYUFORMER_PRETRAIN_OUT=${YUFORMER_RUNS_ROOT}/chosen\n')
    output = production_output_dir(tmp_path)
    assert output == tmp_path/'custom-runs/chosen'
    checkpoint = output/'checkpoint/step-1/.metadata'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    assert list(production_output_dir(tmp_path).glob('checkpoint/step-*/.metadata')) == [checkpoint]


def test_unresolved_actual_production_output_fails_closed(tmp_path):
    write(tmp_path, 'pretrain/configs/pretrain_base.env', 'YUFORMER_PRETRAIN_OUT=${MISSING}/full\n')
    with pytest.raises(ValueError, match='unresolved'):
        production_output_dir(tmp_path)


@pytest.mark.parametrize('directory', [False, True])
def test_training_proof_writeback_failure_never_returns_a_successful_snapshot(tmp_path, monkeypatch, directory):
    name = 'proof.json'
    sha = write(tmp_path, name, '{"status":"PASS"}')
    original = os.fsync
    def fail(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode) == directory:
            raise OSError(errno.EIO, 'injected proof writeback failure')
        original(fd)
    monkeypatch.setattr(os, 'fsync', fail)
    with pytest.raises(OSError, match='proof writeback failure'):
        snapshot_files(tmp_path, {name:sha})


def test_immutable_proof_republication_checks_both_content_and_durability(tmp_path, monkeypatch):
    path = tmp_path/'proof.json'
    publish_checked(path, b'actual proof', immutable=True)
    with pytest.raises(ValueError, match='immutable'):
        publish_checked(path, b'different proof', immutable=True)
    def fail(fd):
        raise OSError(errno.EIO, 'injected existing proof writeback failure')
    monkeypatch.setattr(os, 'fsync', fail)
    with pytest.raises(OSError, match='existing proof'):
        publish_checked(path, b'actual proof', immutable=True)
