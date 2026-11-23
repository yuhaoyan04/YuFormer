import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from adoption_research_evidence import freeze_adoption_research_evidence


def test_final_source_runs_and_contracts_survive_live_report_changes(tmp_path):
    names = ('data/experiments/window_source_controls_20261114/old/result.json',
             'data/experiments/rope_source_controls_20261114/new/result.json',
             'experiments/e2_architecture/configs/rope_source_controls_20261114/new.yaml',
             'data/health/model-rope-source-controls.json',
             'experiments/e2_architecture/scripts/development_validation_review.py',
             'data/health/nearby-lr-controls.json',
             'data/health/lower-lr-control.json',
             'data/experiments/production_matched_controls_20261115/candidate/result.json',
             'experiments/e2_architecture/configs/production_matched_controls_20261115/candidate.yaml',
             'data/health/production-matched-controls.json',
             'data/health/production-matched-development-review.json',
             'data/experiments/production_shape_lr_controls_20261115_authorized/candidate/result.json',
             'experiments/e2_architecture/configs/production_shape_lr_controls_20261115_authorized/candidate.yaml',
             'data/health/production-shape-lr-controls.json',
             'data/health/production-shape-lr-contract-preview.json',
             'data/health/checkpoint-directory-durability-qualification.json')
    names += ('data/experiments/final_deterministic_production_controls_20261115/candidate/result.json',
              'experiments/e2_architecture/configs/final_deterministic_production_controls_20261115/candidate.yaml',
              'data/experiments/deterministic_flash_long_profiles_20261115/context16k/result.json',
              'report/patches/deterministic-flash-long-profiles/context16k.yaml',
              'data/health/final-deterministic-production-controls.json',
              'data/health/final-deterministic-production-review.json')
    inputs = {}
    for index, name in enumerate(names):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'actual':index}))
        inputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    bound, archive = freeze_adoption_research_evidence(tmp_path, inputs)
    assert set(archive['files']) == set(names)
    assert not set(names) & set(bound)
    for name in names:
        (tmp_path/name).write_text('later report export')
    for name, expected in bound.items():
        assert hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() == expected
    for name, expected in inputs.items():
        row = archive['files'][name]
        assert row['sha256'] == expected
        assert hashlib.sha256((tmp_path/row['archive']).read_bytes()).hexdigest() == expected


def test_changed_research_result_cannot_be_snapshotted_as_the_measured_result(tmp_path):
    name = 'data/experiments/rope_source_controls_20261114/new/result.json'
    path = tmp_path/name
    path.parent.mkdir(parents=True)
    path.write_text('changed')
    with pytest.raises(ValueError):
        freeze_adoption_research_evidence(tmp_path, {name:hashlib.sha256(b'measured').hexdigest()})


def test_actual_training_source_is_backed_up_without_weakening_live_guard(tmp_path):
    name = 'pretrain/backends/torchtitan/yuformer_ext/checkpoint.py'
    path = tmp_path/name
    path.parent.mkdir(parents=True)
    path.write_bytes(b'qualified source bytes')
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    bound, archive = freeze_adoption_research_evidence(tmp_path, {name:expected})
    assert bound[name] == expected
    row = archive['files'][name]
    path.write_bytes(b'later accidental source edit')
    assert hashlib.sha256(path.read_bytes()).hexdigest() != bound[name]
    assert (tmp_path/row['archive']).read_bytes() == b'qualified source bytes'


def test_source_backup_must_not_refresh_a_changed_admitted_source(tmp_path):
    name = 'model/src/yuformer_model/attention.py'
    path = tmp_path/name
    path.parent.mkdir(parents=True)
    path.write_bytes(b'changed source')
    with pytest.raises(ValueError,match='differs from its admitted binding'):
        freeze_adoption_research_evidence(tmp_path,{name:hashlib.sha256(b'admitted source').hexdigest()})
