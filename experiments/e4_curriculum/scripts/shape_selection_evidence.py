"""Require independently reproduced grid quality before adopting a new shape."""
import hashlib
import json
from pathlib import Path

from production_shape_quality_review import review
from freeze_1t_recipe import source_sha256


def require_shape_quality_review(root, selection):
    marker = selection.get('production_shape_validation_review')
    run = selection.get('sustained_measurement',{}).get('run_dir','')
    if marker is None:
        if run.startswith('data/experiments/production_shape_lr_controls_20261115'):
            raise ValueError('shape-grid selection cannot omit its independent all-nine quality proof')
        return {}
    if not isinstance(marker,dict) or set(marker) != {'case','protocol','report'}:
        raise ValueError('shape selection needs its exact case/protocol/report')
    root = Path(root).resolve()
    for key in ('protocol','report'):
        path = Path(marker[key])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('shape quality paths must be repository-relative')
    inputs = selection.get('evidence_sha256',{})
    path = root/marker['report']
    if inputs.get(marker['report']) != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError('shape independent review is not bound by the selection')
    actual = review(root,marker['protocol'])
    if (json.loads(path.read_bytes()) != actual or actual['status'] != 'PASS_INDEPENDENT_PRODUCTION_SHAPE_GRID'
            or actual['training_source_sha256'] != source_sha256(root)):
        raise ValueError('actual current-source shape all-nine quality did not pass')
    case = marker['case']
    if (case != actual['preferred_case'] or actual['cases'][case]['run_dir'] != run
            or actual['cases'][case]['status'] != 'PASS_PRIMARY_AND_EXPANDED_ALL_NINE'):
        raise ValueError('shape quality proof belongs to another selected checkpoint or ranking')
    for name,expected in actual['source_inputs'].items():
        if inputs.get(name) != expected:
            raise ValueError('selection omits actual shape quality evidence: '+name)
    helper = Path(__file__)
    return {str(helper.relative_to(helper.parents[3])):hashlib.sha256(helper.read_bytes()).hexdigest()}
