import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import production_selection_evidence as admission


def fixture(tmp_path, status='PASS_PRODUCTION_PRIMARY_AND_EXPANDED_ALL_NINE', actual_result='selected/result.json'):
    actual = dict(status=status,training_source_sha256='current',source_inputs={'producer.py':'exact'})
    raw = json.dumps(actual).encode()
    (tmp_path/'review.json').write_bytes(raw)
    (tmp_path/'plan.json').write_text(json.dumps({'cases':{'larger_lr1e4_b8':{'primary_result':actual_result}}}))
    selected = dict(production_validation_review=dict(case='larger_lr1e4_b8',plan='plan.json',report='review.json'),
        sustained_measurement=dict(run_dir='selected'),evidence_sha256={'review.json':hashlib.sha256(raw).hexdigest(),'producer.py':'exact'})
    return selected,actual


def test_actual_production_measurement_cannot_omit_quality_proof(tmp_path):
    selected = dict(sustained_measurement=dict(run_dir='data/experiments/production_matched_controls_20261115/larger_lr1e4_b8'))
    with pytest.raises(ValueError,match='cannot omit'):
        admission.require_production_quality_review(tmp_path,selected)


def test_matching_hash_cannot_promote_actual_failed_production_quality(tmp_path,monkeypatch):
    selected,actual = fixture(tmp_path,status='FAIL_PRODUCTION_PRIMARY_OR_EXPANDED_ALL_NINE')
    monkeypatch.setattr(admission,'review',lambda path:actual)
    monkeypatch.setattr(admission,'source_sha256',lambda root:'current')
    with pytest.raises(ValueError,match='did not pass'):
        admission.require_production_quality_review(tmp_path,selected)


def test_production_quality_for_another_checkpoint_cannot_qualify_selected_model(tmp_path,monkeypatch):
    selected,actual = fixture(tmp_path,actual_result='another/result.json')
    monkeypatch.setattr(admission,'review',lambda path:actual)
    monkeypatch.setattr(admission,'source_sha256',lambda root:'current')
    with pytest.raises(ValueError,match='another selected checkpoint'):
        admission.require_production_quality_review(tmp_path,selected)
