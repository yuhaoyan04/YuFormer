import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_disjoint_validation_review as review


def fixture(tmp_path, monkeypatch, chinese_delta=None, missing_batch=False):
    monkeypatch.setattr(review,'ROOT',tmp_path)
    protocol_path=tmp_path/'protocol.json'
    monkeypatch.setattr(review,'PROTOCOL',protocol_path)
    protocol={'domains':['academic','books','chinese','code','encyclopedia','math','multilingual','qa','web'],
              'training_source_sha256':'source', 'uncertainty':{'seed':20261114}}
    protocol_path.write_text(json.dumps(protocol))
    plan={'cases':{},'source_inputs':{}}
    for case,mean in [('matched_baseline_b8',8.),('larger_lr1e4_b8',7.9),('larger_lr2e4_b8',8.2)]:
        path=tmp_path/case/'scores.json'
        path.parent.mkdir()
        plan['cases'][case]={'report':str(path.relative_to(tmp_path)), 'checkpoint_metadata_sha256':case}
        domains={domain:{'batch_mean_losses':[mean]*44,'mean_loss':mean} for domain in protocol['domains']}
        if case=='larger_lr1e4_b8' and chinese_delta is not None:
            domains['chinese']={'batch_mean_losses':[8.+chinese_delta]*44,'mean_loss':8.+chinese_delta}
        if case=='larger_lr1e4_b8' and missing_batch:
            domains['chinese']['batch_mean_losses'].pop()
        path.write_text(json.dumps({'status':'PASS_READONLY_DISJOINT_VALIDATION',
            'protocol_sha256':review.bench.digest(protocol_path),'training_source_sha256':'source',
            'checkpoint_metadata_sha256':case,'source_inputs':{},'training_steps_executed':0,
            'checkpoint_modified':False,'domains':domains,'aggregate_mean_loss':mean}))
    return protocol,plan


def test_all_domains_must_improve_without_erasing_the_primary_failure(tmp_path,monkeypatch):
    protocol,plan=fixture(tmp_path,monkeypatch)
    result=review.review(protocol,plan)
    assert result['status']=='PASS_DISJOINT_ALL_NINE_MEAN_IMPROVEMENT'
    assert result['primary_review_preserved_as_failed'] is True
    assert all(row['paired_bootstrap_95_ci'][1]<0 for rows in result['comparisons'].values() for row in rows.values())


def test_even_a_small_chinese_regression_cannot_be_waived(tmp_path,monkeypatch):
    protocol,plan=fixture(tmp_path,monkeypatch,chinese_delta=.00001)
    result=review.review(protocol,plan)
    assert result['status']=='FAIL_DISJOINT_ALL_NINE_MEAN_IMPROVEMENT'
    assert result['comparisons']['matched_baseline_b8']['chinese']['mean_strictly_improves'] is False


def test_missing_paired_batches_do_not_create_a_quality_receipt(tmp_path,monkeypatch):
    protocol,plan=fixture(tmp_path,monkeypatch,missing_batch=True)
    with pytest.raises(ValueError,match='forty-four'):
        review.review(protocol,plan)
