import hashlib
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
import development_selection_evidence as admission


def fixture_selection(tmp_path, *, status='PASS_PRIMARY_AND_EXPANDED_ALL_NINE', actual_result='run-a/result.json'):
    report = {'status':status, 'source_inputs':{'producer.py':'bound-sha'}}
    raw = json.dumps(report).encode()
    (tmp_path/'review.json').write_bytes(raw)
    (tmp_path/'plan.json').write_text(json.dumps({'cases':{'candidate':{'primary_result':actual_result}}}))
    selection = dict(development_validation_review=dict(case='candidate', candidate_plan='plan.json',
        reference_plan='reference.json', report='review.json'), sustained_measurement=dict(run_dir='run-a'),
        evidence_sha256={'review.json':hashlib.sha256(raw).hexdigest(), 'producer.py':'bound-sha'})
    return selection, report


def test_failed_actual_quality_is_rejected_even_with_matching_report_hash(tmp_path, monkeypatch):
    selection, actual = fixture_selection(tmp_path, status='FAIL_PRIMARY_OR_EXPANDED_ALL_NINE')
    monkeypatch.setattr(admission, 'review', lambda *args:actual)
    with pytest.raises(ValueError, match='did not pass'):
        admission.require_development_review(tmp_path, selection)


def test_passing_quality_for_another_checkpoint_is_rejected(tmp_path, monkeypatch):
    selection, actual = fixture_selection(tmp_path, actual_result='another-run/result.json')
    monkeypatch.setattr(admission, 'review', lambda *args:actual)
    with pytest.raises(ValueError, match='another selected training checkpoint'):
        admission.require_development_review(tmp_path, selection)


def test_selection_must_bind_all_actual_quality_inputs(tmp_path, monkeypatch):
    selection, actual = fixture_selection(tmp_path)
    selection['evidence_sha256'].pop('producer.py')
    monkeypatch.setattr(admission, 'review', lambda *args:actual)
    with pytest.raises(ValueError, match='omits actual quality evidence'):
        admission.require_development_review(tmp_path, selection)
