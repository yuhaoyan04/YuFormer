import copy
import errno
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_1t_overlap as overlap
import training_source_windows as windows


def test_target_window_clips_real_document_intervals_and_cache_is_range_bound(tmp_path):
    order, ledger, output = [tmp_path/name for name in ('order.parquet','lengths.u32','consumed.records')]
    pq.write_table(pa.table({'doc_id':[f'{i:032x}' for i in range(3)],
                            'position':[0,1,2],'domain':['web','code','math']}),order)
    np.array([3,4,2],dtype='<u4').tofile(ledger)
    original = overlap.extract_consumed(order,ledger,output,identity='same-input',trained_tokens=4)
    shifted = overlap.extract_consumed(order,ledger,output,identity='same-input',trained_tokens=4,target_start=4)
    assert original['domain_tokens']=={'web':2,'code':2}
    assert shifted['schema_version']==3
    assert shifted['target_start']==4 and shifted['target_stop']==8
    assert shifted['domain_tokens']=={'code':3,'math':1}
    records=np.fromfile(output,dtype=overlap.CONSUMED_DTYPE)
    assert records[['doc_tokens','begin','end']].tolist()==[(4,1,4),(2,0,1)]


@pytest.mark.parametrize('start', [0,-1,True,1.5,10])
def test_invalid_or_out_of_capacity_target_start_is_rejected(start):
    with pytest.raises(ValueError):
        overlap.consumed_position_bounds(np.array([3,4,2],dtype='<u4'),4,target_start=start)


def test_target_window_on_document_boundaries_is_exact():
    bounds=overlap.consumed_position_bounds(np.array([3,4,2],dtype='<u4'),4,target_start=3)
    assert bounds=={'first_position':1,'last_position':1,'first_global_start':3,
                   'last_global_start':3,'target_start':3,'target_stop':7}


def test_consumed_record_writeback_failure_cannot_publish_a_success_cache(tmp_path,monkeypatch):
    order,ledger,output=[tmp_path/name for name in ('order.parquet','lengths.u32','consumed.records')]
    pq.write_table(pa.table({'doc_id':['0'*32],'position':[0],'domain':['web']}),order)
    np.array([5],dtype='<u4').tofile(ledger)
    def fail(_):
        raise OSError(errno.EIO,'injected consumed record writeback failure')
    monkeypatch.setattr(overlap.os,'fsync',fail)
    with pytest.raises(OSError,match='writeback failure'):
        overlap.extract_consumed(order,ledger,output,identity='test',trained_tokens=4)
    assert not output.exists()
    assert not output.with_suffix('.json').exists()


@pytest.fixture
def admitted(tmp_path,monkeypatch):
    helper=tmp_path/'experiments/e4_curriculum/scripts/training_source_windows.py'
    helper.parent.mkdir(parents=True)
    helper.write_bytes(Path(windows.__file__).read_bytes())
    monkeypatch.setattr(windows,'__file__',str(helper))
    order,ledger=tmp_path/'order.parquet',tmp_path/'lengths.u32'
    domains=['web']+list(windows.DOMAINS)
    pq.write_table(pa.table({'doc_id':[f'{i:032x}' for i in range(10)],
                            'position':list(range(10)),'domain':domains}),order)
    np.array([17]+[16]*9,dtype='<u4').tofile(ledger)
    output=tmp_path/'stable.consumed.records'
    prefix=overlap.extract_consumed(order,ledger,output,identity='test',trained_tokens=144)
    original_bytes=output.read_bytes()
    targets=dict.fromkeys(windows.DOMAINS,1/9)
    failed=overlap.audit_consumed_mixture(prefix['domain_tokens'],targets,
                                         absolute_tolerance=.002,relative_tolerance=.1)
    assert failed['status']=='FAIL'
    context={'total_source_tokens':161,'release_manifest_sha256':'r'*64,
             'stream_meta_sha256':'m'*64,'data_frozen_recipe_sha256':'f'*64}
    extracted,mixture,binding=windows.resolve_consumption_window(tmp_path,'stable',order,ledger,output,
        context=context,seq_len=1,stage_samples=144,targets=targets,prefix=prefix,prefix_mixture=failed,
        absolute_tolerance=.002,relative_tolerance=.1)
    report={'status':'PASS','stage':'stable','configured_target_tokens':144,
            'consumed':extracted,'mixture':mixture,'source_window':binding}
    return tmp_path,report,context,output,original_bytes


def validate(admitted,report=None):
    root,original,context,_,_=admitted
    return windows.require_window_admission(root,original if report is None else report,
        phase='stable',stage_samples=144,seq_len=1,release_sha256=context['release_manifest_sha256'],
        stream_meta_sha256=context['stream_meta_sha256'],
        data_frozen_recipe_sha256=context['data_frozen_recipe_sha256'])


def test_actual_selected_id_records_pass_without_overwriting_the_failed_prefix(admitted):
    root,report,_,prefix,before=admitted
    assert prefix.read_bytes()==before
    assert report['source_window']['source_sample_offset']==16
    assert report['source_window']['target_interval']==[17,161]
    assert report['consumed']['domain_tokens']==dict.fromkeys(windows.DOMAINS,16)
    result=validate(admitted)
    assert result['source_sample_offset']==16
    assert len(result['inputs'])==3


def test_window_binding_cannot_be_rebased_without_recounting(admitted):
    report=copy.deepcopy(admitted[1]);report['source_window']['source_sample_offset']=8
    with pytest.raises(ValueError,match='geometry/data/producer'):
        validate(admitted,report)


def test_nonpassing_mixture_cannot_reuse_a_valid_window_certificate(admitted):
    report=copy.deepcopy(admitted[1]);report['mixture']['status']='FAIL'
    with pytest.raises(ValueError):
        validate(admitted,report)


def test_window_certificate_is_hash_bound(admitted):
    root,report,*_=admitted
    path=root/report['source_window']['selection_report']['file']
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError):
        validate(admitted)


def test_replaced_id_record_file_invalidates_admission_even_with_identical_bytes(admitted):
    root,report,*_=admitted
    selection=json.loads((root/report['source_window']['selection_report']['file']).read_bytes())
    path=root/selection['consumption_records']['file']
    replacement=path.with_name('replacement');replacement.write_bytes(path.read_bytes());replacement.replace(path)
    with pytest.raises(ValueError,match='records changed'):
        validate(admitted)
