"""Verify the exact actual-production pair used for the scientific selection."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'experiments/e2_architecture/scripts'))
from review_production_matched_controls import review
from freeze_1t_recipe import source_sha256


def require_production_quality_review(root, selection):
    if 'production_user_tradeoff_review' in selection:
        from user_tradeoff_selection_evidence import require_user_tradeoff_review
        return require_user_tradeoff_review(root, selection)
    marker = selection.get('production_validation_review')
    if marker is None:
        if selection.get('sustained_measurement',{}).get('run_dir','').startswith((
                'data/experiments/production_matched_controls_20261115/',
                'data/experiments/final_deterministic_production_controls_20261115/')):
            raise ValueError('the actual production selection cannot omit its paired quality proof')
        return {}
    if not isinstance(marker,dict) or set(marker) != {'case','plan','report'} or marker['case'] != 'larger_lr1e4_b8':
        raise ValueError('production quality selection requires its exact case/plan/report')
    root = Path(root).resolve()
    for key in ('plan','report'):
        path = Path(marker[key])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('production quality selection paths must be repository-relative')
    inputs = selection.get('evidence_sha256',{})
    path = root/marker['report']
    if marker['report'] not in inputs or hashlib.sha256(path.read_bytes()).hexdigest() != inputs[marker['report']]:
        raise ValueError('actual production quality report is missing from selected immutable evidence')
    actual = review(root/marker['plan'])
    stored = json.loads(path.read_bytes())
    if (stored != actual or actual['status'] != 'PASS_PRODUCTION_PRIMARY_AND_EXPANDED_ALL_NINE'
            or actual['training_source_sha256'] != source_sha256(root)):
        raise ValueError('actual current-source production primary and expanded quality did not pass')
    plan = json.loads((root/marker['plan']).read_bytes())
    if plan['cases'][marker['case']]['primary_result'] != selection['sustained_measurement']['run_dir']+'/result.json':
        raise ValueError('production quality proof belongs to another selected checkpoint')
    for name,expected in actual['source_inputs'].items():
        if inputs.get(name) != expected:
            raise ValueError('selected evidence omits actual production quality input: '+name)
    helper = Path(__file__)
    return {str(helper.relative_to(root)):hashlib.sha256(helper.read_bytes()).hexdigest()}
