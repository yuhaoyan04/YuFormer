import hashlib
import json
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import user_tradeoff_quality_review as assessment
import production_selection_evidence as admission


def fixture(tmp_path,monkeypatch,*,excess=False,other_bad=False,macro_bad=False,old_reply=False,other_model=False):
    domains=('web','code','math','chinese','academic','books','encyclopedia','qa','multilingual')
    ceiling=.010807349
    reports=[]
    for cohort in ('old','new'):
        folder=tmp_path/cohort
        (folder/'validation').mkdir(parents=True)
        run=folder/'run';run.mkdir()
        model=folder/'model.yaml'
        shape={'num_layers':18,'hidden_size':2560 if not (cohort=='new' and other_model) else 2048}
        model.write_text(yaml.safe_dump({'presets':{'candidate':shape}}))
        (run/'resolved_config.yaml').write_text(yaml.safe_dump({'model':{'preset':'candidate'},'optim':{'lr':.0001}}))
        (run/'run_identity.json').write_text(json.dumps({'model_configs':str(model)}))
        plan=folder/'validation/plan.json'
        plan.write_text(json.dumps({'cases':{'larger_lr1e4_b8':{'primary_result':f'{cohort}/run/result.json'}}}))
        relative=ceiling if cohort=='old' else ceiling+(.002 if excess else -.001)
        comparisons={d:dict(reference_mean=7.,candidate_mean=6.9,delta=-.1,mean_strictly_improves=True) for d in domains}
        comparisons['chinese']=dict(reference_mean=7.,candidate_mean=7.*(1+relative),delta=7.*relative,mean_strictly_improves=False)
        if cohort=='new' and other_bad:
            comparisons['code'].update(candidate_mean=7.1,delta=.1,mean_strictly_improves=False)
        primary={d:dict(baseline=v['reference_mean'],candidate=v['candidate_mean'],delta=v['delta'],mean_strictly_improves=v['mean_strictly_improves']) for d,v in comparisons.items()}
        report=dict(status='FAIL_PRODUCTION_PRIMARY_OR_EXPANDED_ALL_NINE',training_source_sha256='current',
                    primary_all_nine_passed=False,expanded_all_nine_passed=False,
                    aggregate_passed=not (cohort=='new' and macro_bad),
                    baseline_aggregate_mean_loss=7.,candidate_aggregate_mean_loss=6.9,
                    primary_comparison=primary,expanded_comparison=comparisons,
                    source_inputs={str(p.relative_to(tmp_path)):hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (plan,model,run/'resolved_config.yaml',run/'run_identity.json')})
        path=folder/'validation/review.json';path.write_text(json.dumps(report));reports.append(report)
    original=tmp_path/'old/validation/review.json'
    decision=dict(status='ACTIVE_USER_ACCEPTED_CURRENT_ACTUAL_PRODUCTION_QUALITY_TRADEOFF',
                  latest_user_answer='接受当前约 1.08% 的回退，优先保留 1.52B' if not old_reply else '接受已量化的小幅单域取舍，优先更大模型',
                  clarification_pending=False,formal_data_numeric_and_recovery_gates_unchanged=True,
                  accepted_candidate=dict(parameters=1516600064,peak_learning_rate=.0001,
                    quality_report='old/validation/review.json',quality_report_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                    chinese_relative_loss_delta=ceiling))
    (tmp_path/'decision.json').write_text(json.dumps(decision))
    helper=tmp_path/'quality-helper.py';helper.write_text('qualified helper bytes')
    monkeypatch.setattr(assessment,'__file__',str(helper))
    monkeypatch.setattr(assessment,'source_sha256',lambda root:'current')
    monkeypatch.setattr(assessment,'strict_review',lambda plan:reports[1])
    return original.read_bytes()


def test_accepted_bound_does_not_relabel_original_strict_failure(tmp_path,monkeypatch):
    old=fixture(tmp_path,monkeypatch)
    result=assessment.review(tmp_path,'new/validation/plan.json','decision.json')
    assert result['status']=='PASS_EXPLICIT_USER_ACCEPTED_QUALITY_POLICY'
    assert result['strict_original_endpoint_status']=='FAIL_PRODUCTION_PRIMARY_OR_EXPANDED_ALL_NINE'
    assert result['strict_all_nine_passed'] is False
    assert (tmp_path/'old/validation/review.json').read_bytes()==old


@pytest.mark.parametrize('case',['excess','other_bad','macro_bad'])
def test_quality_outside_explicitly_accepted_conditions_remains_failed(tmp_path,monkeypatch,case):
    fixture(tmp_path,monkeypatch,**{case:True})
    assert assessment.review(tmp_path,'new/validation/plan.json','decision.json')['status']=='FAIL_USER_ACCEPTED_QUALITY_BOUNDS'


@pytest.mark.parametrize('case',['old_reply','other_model'])
def test_old_magnitude_or_different_architecture_cannot_inherit_acceptance(tmp_path,monkeypatch,case):
    fixture(tmp_path,monkeypatch,**{case:True})
    with pytest.raises(ValueError):
        assessment.review(tmp_path,'new/validation/plan.json','decision.json')


def test_new_production_cohort_cannot_omit_its_paired_policy_evidence(tmp_path):
    selection={'sustained_measurement':{'run_dir':'data/experiments/final_deterministic_production_controls_20261115/candidate'}}
    with pytest.raises(ValueError,match='cannot omit'):
        admission.require_production_quality_review(tmp_path,selection)


def test_present_but_null_tradeoff_marker_cannot_bypass_real_adopter_hook(tmp_path):
    with pytest.raises(ValueError,match='exact case'):
        admission.require_production_quality_review(tmp_path,{'production_user_tradeoff_review':None})
