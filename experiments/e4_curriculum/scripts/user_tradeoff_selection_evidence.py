"""Bind the accepted magnitude and actual selected checkpoint to adoption."""
import hashlib
import json
from pathlib import Path

from user_tradeoff_quality_review import review
from freeze_1t_recipe import source_sha256


def require_user_tradeoff_review(root,selection):
    marker=selection.get('production_user_tradeoff_review')
    if not isinstance(marker,dict) or set(marker)!={'case','plan','decision','report'}:
        raise ValueError('user tradeoff requires exact case/plan/decision/report evidence')
    if marker['case']!='larger_lr1e4_b8' or selection.get('parameters')!=1516600064:
        raise ValueError('user tradeoff cannot authorize another model')
    root=Path(root).resolve()
    for k in ('plan','decision','report'):
        p=Path(marker[k])
        if p.is_absolute() or '..' in p.parts:
            raise ValueError('user tradeoff paths must be repository-relative')
    bound=selection.get('evidence_sha256',{})
    path=root/marker['report']
    if bound.get(marker['report'])!=hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError('user tradeoff review is missing from bound selected evidence')
    actual=review(root,marker['plan'],marker['decision'])
    if (json.loads(path.read_bytes())!=actual or actual['status']!='PASS_EXPLICIT_USER_ACCEPTED_QUALITY_POLICY'
            or actual['training_source_sha256']!=source_sha256(root)):
        raise ValueError('actual selected-source quality did not meet the explicitly accepted bound')
    plan=json.loads((root/marker['plan']).read_bytes())
    if plan['cases'][marker['case']]['primary_result']!=selection['sustained_measurement']['run_dir']+'/result.json':
        raise ValueError('user tradeoff proof belongs to another selected checkpoint')
    for n,expected in actual['source_inputs'].items():
        if bound.get(n)!=expected:
            raise ValueError('selection omits actual user tradeoff input: '+n)
    p=Path(__file__)
    return {str(p.relative_to(p.parents[3])):hashlib.sha256(p.read_bytes()).hexdigest()}
