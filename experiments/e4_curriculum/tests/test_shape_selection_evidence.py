import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import shape_selection_evidence as admission


def fixture(tmp_path, status='PASS_INDEPENDENT_PRODUCTION_SHAPE_GRID', run='run-a'):
    report = dict(status=status,training_source_sha256='current',preferred_case='candidate',
                  cases={'candidate':dict(run_dir=run,status='PASS_PRIMARY_AND_EXPANDED_ALL_NINE')},
                  source_inputs={'actual-producer.py':'bound'})
    raw = json.dumps(report).encode()
    (tmp_path/'review.json').write_bytes(raw)
    selected = dict(production_shape_validation_review=dict(case='candidate',protocol='protocol.json',report='review.json'),
                    sustained_measurement=dict(run_dir='run-a'),
                    evidence_sha256={'review.json':hashlib.sha256(raw).hexdigest(),'actual-producer.py':'bound'})
    return selected, report


def test_shape_run_cannot_qualify_by_quality_flag_alone(tmp_path):
    selection = dict(sustained_measurement=dict(
        run_dir='data/experiments/production_shape_lr_controls_20261115_authorized/candidate',
        quality_review_completed=True))
    with pytest.raises(ValueError,match='cannot omit'):
        admission.require_shape_quality_review(tmp_path,selection)


@pytest.mark.parametrize('change', ['failed_quality','another_checkpoint','missing_evidence','different_ranking'])
def test_bound_report_cannot_hide_failed_or_mismatched_shape_evidence(tmp_path,monkeypatch,change):
    selected, actual = fixture(tmp_path,
        status='FAIL_ALL_SHAPE_CANDIDATES' if change=='failed_quality' else 'PASS_INDEPENDENT_PRODUCTION_SHAPE_GRID',
        run='run-b' if change=='another_checkpoint' else 'run-a')
    if change=='missing_evidence':
        selected['evidence_sha256'].pop('actual-producer.py')
    if change=='different_ranking':
        selected['production_shape_validation_review']['case']='unselected'
    monkeypatch.setattr(admission,'review',lambda *args:actual)
    monkeypatch.setattr(admission,'source_sha256',lambda root:'current')
    with pytest.raises(ValueError):
        admission.require_shape_quality_review(tmp_path,selected)


def test_shape_quality_rejects_traversal_before_accessing_evidence(tmp_path):
    selected, _ = fixture(tmp_path)
    selected['production_shape_validation_review']['protocol']='../unrelated.json'
    with pytest.raises(ValueError,match='repository-relative'):
        admission.require_shape_quality_review(tmp_path,selected)
