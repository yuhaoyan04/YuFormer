"""Fail closed at trained stage boundaries until bound capability audits pass."""
import hashlib
import json
import math
from pathlib import Path

from .window_release import bound_json

DOMAINS={'web','code','math','chinese','academic','books','encyclopedia','qa','multilingual'}
SUITES={'ruler','long_bpb','long_qa_code','short_core'}


def require_parent_evaluation(parent, *, next_seq, scope):
    if scope=='QUALIFICATION':
        return None
    run=Path(parent['run_identity']).parent
    identity=json.loads((run/'run_identity.json').read_bytes())
    complete=json.loads((run/'training_complete.json').read_bytes())
    result=json.loads((run/'result.json').read_bytes())
    accounting=json.loads((run/'training_accounting.json').read_bytes())
    if (complete.get('status')!='PASS' or result.get('status')!='PASS'
            or complete.get('identity_sha256')!=identity['identity_sha256']
            or result.get('identity_sha256')!=identity['identity_sha256']
            or complete.get('attempt_id')!=result.get('attempt_id')
            or accounting.get('attempt_id')!=complete.get('attempt_id')
            or accounting.get('step')!=parent['step'] or accounting.get('status')!='PASS'):
        raise ValueError('production stage parent must have a complete bound final training/validation result')
    for domain in DOMAINS:
        metric=result.get('metrics',{}).get(f'validation_metrics/domain/{domain}/loss',{})
        value=metric.get('last')
        if (type(value) not in (int,float) or not math.isfinite(value)
                or metric.get('last_step')!=parent['step']):
            raise ValueError('production stage parent lacks finite final validation in all nine domains')
    if next_seq==16384:
        return {'status':'PASS_COMPLETED_4K_BASELINE_VALIDATION'}
    binding=parent.get('evaluation_admission')
    if not isinstance(binding,dict):
        raise ValueError('64K production progression requires the actual 16K long/short capability admission')
    report=bound_json(binding['file'],binding['sha256'])
    if (report.get('status')!='PASS_STAGE_LONG_AND_SHORT_EVALUATION' or report.get('scope')!='PRODUCTION'
            or report.get('seq_len')!=16384 or report.get('step')!=parent['step']
            or report.get('checkpoint_handoff_sha256')!=parent['handoff_sha256']
            or report.get('run_identity_sha256')!=parent['run_identity_sha256']
            or set(report.get('suites',{}))!=SUITES
            or any(row.get('status')!='PASS' for row in report['suites'].values())
            or report.get('baseline_checkpoint_bound') is not True):
        raise ValueError('long/short capability admission does not qualify the actual completed 16K parent')
    inputs=report.get('source_inputs')
    if not isinstance(inputs,dict) or not inputs:
        raise ValueError('long/short capability report must bind its actual evaluator, cases and measurements')
    for name,sha in inputs.items():
        path=Path(name)
        if not path.is_absolute() or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
            raise ValueError('long/short capability evidence changed')
    return report
