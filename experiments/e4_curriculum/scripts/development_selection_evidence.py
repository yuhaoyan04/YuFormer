"""Require the exact development checkpoint review when admitting its selection."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'experiments/e2_architecture/scripts'))
from development_validation_review import review


def require_development_review(root, selection):
    row = selection.get('development_validation_review')
    if row is None:
        return {}
    if not isinstance(row, dict) or set(row) != {'case', 'candidate_plan', 'reference_plan', 'report'}:
        raise ValueError('development selection requires its exact case/plans/report')
    root = Path(root).resolve()
    for key in ('candidate_plan', 'reference_plan', 'report'):
        path = Path(row[key])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('development selection paths must be repository-relative')
    evidence = selection.get('evidence_sha256', {})
    expected = evidence.get(row['report'])
    path = root/row['report']
    if expected is None or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('development quality report is absent from exact selection evidence')
    stored = json.loads(path.read_bytes())
    actual = review(root, root/row['candidate_plan'], row['case'], root/row['reference_plan'])
    if stored != actual or actual['status'] != 'PASS_PRIMARY_AND_EXPANDED_ALL_NINE':
        raise ValueError('actual primary and expanded all-nine quality review did not pass')
    plan = json.loads((root/row['candidate_plan']).read_bytes())
    selected_result = selection['sustained_measurement']['run_dir']+'/result.json'
    if plan['cases'][row['case']]['primary_result'] != selected_result:
        raise ValueError('development quality review belongs to another selected training checkpoint')
    for name, expected in actual['source_inputs'].items():
        if evidence.get(name) != expected:
            raise ValueError('development selection omits actual quality evidence: '+name)
    helper = Path(__file__)
    return {str(helper.relative_to(root)):hashlib.sha256(helper.read_bytes()).hexdigest()}
