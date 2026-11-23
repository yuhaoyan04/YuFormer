import hashlib
import json

import pytest

from yuformer_midtrain.evaluation_contract import require_parent_evaluation, DOMAINS, SUITES


def write(path,value):
    path.write_text(json.dumps(value,sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def parent(tmp_path):
    write(tmp_path/'run_identity.json',{'identity_sha256':'sealed'})
    write(tmp_path/'training_complete.json',{'status':'PASS','identity_sha256':'sealed','attempt_id':'attempt'})
    write(tmp_path/'result.json',{'status':'PASS','identity_sha256':'sealed','attempt_id':'attempt',
        'metrics':{f'validation_metrics/domain/{d}/loss':{'last':3.,'last_step':10} for d in DOMAINS}})
    write(tmp_path/'training_accounting.json',{'status':'PASS','attempt_id':'attempt','step':10})
    return {'run_identity':str(tmp_path/'run_identity.json'),'run_identity_sha256':'r'*64,
            'handoff_sha256':'h'*64,'step':10}


def test_completed_short_baseline_required_before_production_16k(parent):
    assert require_parent_evaluation(parent,next_seq=16384,scope='PRODUCTION')['status'].startswith('PASS')
    path=__import__('pathlib').Path(parent['run_identity']).parent/'result.json'
    result=json.loads(path.read_bytes());result['metrics'].pop('validation_metrics/domain/chinese/loss')
    write(path,result)
    with pytest.raises(ValueError,match='all nine'):
        require_parent_evaluation(parent,next_seq=16384,scope='PRODUCTION')


def test_64k_cannot_progress_on_boolean_evaluation_flags(parent):
    with pytest.raises(ValueError,match='actual 16K'):
        require_parent_evaluation(parent,next_seq=65536,scope='PRODUCTION')
    assert require_parent_evaluation(parent,next_seq=65536,scope='QUALIFICATION') is None


def test_capability_admission_binds_actual_parent_and_measurement_bytes(parent,tmp_path):
    evidence=tmp_path/'measurements.json';evidence.write_text('actual evaluation measurements')
    report={'status':'PASS_STAGE_LONG_AND_SHORT_EVALUATION','scope':'PRODUCTION','seq_len':16384,
        'step':10,'checkpoint_handoff_sha256':parent['handoff_sha256'],
        'run_identity_sha256':parent['run_identity_sha256'],'baseline_checkpoint_bound':True,
        'suites':{suite:{'status':'PASS'} for suite in SUITES},
        'source_inputs':{str(evidence):hashlib.sha256(evidence.read_bytes()).hexdigest()}}
    path=tmp_path/'admission.json';parent['evaluation_admission']={'file':str(path),'sha256':write(path,report)}
    assert require_parent_evaluation(parent,next_seq=65536,scope='PRODUCTION')==report
    evidence.write_text('changed measurements')
    with pytest.raises(ValueError,match='evidence changed'):
        require_parent_evaluation(parent,next_seq=65536,scope='PRODUCTION')
