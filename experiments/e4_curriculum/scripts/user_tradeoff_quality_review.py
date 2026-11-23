"""Keep strict failures visible while checking the exact user-accepted tradeoff."""
import hashlib
import json
import math
from pathlib import Path
import sys

import yaml

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'experiments/e2_architecture/scripts'),str(ROOT/'pretrain/scripts')]
from review_production_matched_controls import review as strict_review
from freeze_1t_recipe import source_sha256


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def review(root,plan_path,decision_path):
    root=Path(root).resolve()
    def relative(value):
        p=Path(value)
        if p.is_absolute() or '..' in p.parts:
            raise ValueError('quality policy paths must be repository-relative')
        return p
    plan_path=root/relative(plan_path)
    decision_path=root/relative(decision_path)
    decision=json.loads(decision_path.read_bytes())
    if (decision.get('status')!='ACTIVE_USER_ACCEPTED_CURRENT_ACTUAL_PRODUCTION_QUALITY_TRADEOFF'
            or decision.get('latest_user_answer')!='接受当前约 1.08% 的回退，优先保留 1.52B'
            or decision.get('clarification_pending') is not False
            or decision.get('formal_data_numeric_and_recovery_gates_unchanged') is not True):
        raise ValueError('explicit actual-production magnitude acceptance is required; old0.15% reply is insufficient')
    approved=decision['accepted_candidate']
    if approved.get('parameters')!=1516600064 or approved.get('peak_learning_rate')!=.0001:
        raise ValueError('this user acceptance is limited to the actual1.52B LR1e-4 candidate')
    old_path=root/relative(approved['quality_report'])
    if digest(old_path)!=approved['quality_report_sha256']:
        raise ValueError('the original quantified user-accepted report changed')
    old=json.loads(old_path.read_bytes())
    accepted=old['expanded_comparison']['chinese']
    ceiling=accepted['delta']/accepted['reference_mean']
    if (old.get('status')!='FAIL_PRODUCTION_PRIMARY_OR_EXPANDED_ALL_NINE'
            or old.get('aggregate_passed') is not True
            or any(not v['mean_strictly_improves'] for d,v in old['expanded_comparison'].items() if d!='chinese')
            or not math.isclose(ceiling,approved['chinese_relative_loss_delta'],rel_tol=0,abs_tol=1e-15)
            or ceiling<=0 or ceiling>.011):
        raise ValueError('user decision does not bind the actual originally quantified eight-domain improvement')
    strict=strict_review(plan_path)
    stored_path=plan_path.parent/'review.json'
    if json.loads(stored_path.read_bytes())!=strict or strict['training_source_sha256']!=source_sha256(root):
        raise ValueError('new-source strict quality report must independently reproduce')
    model_inputs={}
    measured_models=[]
    for actual_plan,prior_report in ((old_path.parent/'plan.json',old),(plan_path,strict)):
        plan=json.loads(actual_plan.read_bytes())
        run=(root/plan['cases']['larger_lr1e4_b8']['primary_result']).parent
        config_path,identity_path=run/'resolved_config.yaml',run/'run_identity.json'
        config=yaml.safe_load(config_path.read_bytes())
        identity=json.loads(identity_path.read_bytes())
        model_path=Path(identity['model_configs'])
        if not model_path.is_absolute():model_path=root/model_path
        for p in (actual_plan,config_path,identity_path,model_path):
            name=str(p.relative_to(root))
            if prior_report['source_inputs'].get(name)!=digest(p):
                raise ValueError('actual user-accepted or new measured model provenance changed')
            model_inputs[name]=digest(p)
        if config['optim']['lr']!=.0001:
            raise ValueError('user tradeoff does not authorize a different candidate learning rate')
        measured_models.append(yaml.safe_load(model_path.read_bytes())['presets'][config['model']['preset']])
    if measured_models[0]!=measured_models[1]:
        raise ValueError('user tradeoff does not authorize another same-parameter-count architecture')
    domains=set(strict['expanded_comparison'])
    if domains!={'web','code','math','chinese','academic','books','encyclopedia','qa','multilingual'}:
        raise ValueError('all nine actual domains are required')
    expanded=strict['expanded_comparison']['chinese']
    primary=strict['primary_comparison']['chinese']
    expanded_relative=expanded['delta']/expanded['reference_mean']
    primary_relative=primary['delta']/primary['baseline']
    eight_primary=all(v['mean_strictly_improves'] for d,v in strict['primary_comparison'].items() if d!='chinese')
    eight_expanded=all(v['mean_strictly_improves'] for d,v in strict['expanded_comparison'].items() if d!='chinese')
    bounded=(expanded_relative<=ceiling+1e-12 and primary_relative<=ceiling+1e-12)
    passed=eight_primary and eight_expanded and strict['aggregate_passed'] and bounded
    inputs=dict(strict['source_inputs'])
    inputs.update(model_inputs)
    for p in (plan_path,stored_path,decision_path,old_path,Path(__file__)):
        inputs[str(p.relative_to(root))]=digest(p)
    return dict(status='PASS_EXPLICIT_USER_ACCEPTED_QUALITY_POLICY' if passed else 'FAIL_USER_ACCEPTED_QUALITY_BOUNDS',
                candidate='larger_lr1e4_b8',reference='matched_baseline_b8',
                training_source_sha256=strict['training_source_sha256'],source_inputs=inputs,
                strict_original_endpoint_status=strict['status'],strict_all_nine_passed=bool(
                    strict['primary_all_nine_passed'] and strict['expanded_all_nine_passed']),
                other_eight_primary_passed=eight_primary,other_eight_expanded_passed=eight_expanded,
                macro_passed=strict['aggregate_passed'],chinese_within_approved_bound=bounded,
                approved_maximum_relative_chinese_loss_delta=ceiling,
                actual_expanded_relative_chinese_loss_delta=expanded_relative,
                actual_primary_relative_chinese_loss_delta=primary_relative,
                expanded_comparison=strict['expanded_comparison'],primary_comparison=strict['primary_comparison'],
                candidate_aggregate_mean_loss=strict['candidate_aggregate_mean_loss'],
                baseline_aggregate_mean_loss=strict['baseline_aggregate_mean_loss'],
                interpretation='Explicitly accepted measured single-domain tradeoff; strict failed endpoints remain failed. Data, numeric, recovery, source, actual throughput and canary gates are unchanged.',
                production_qualified=False,formal_pretrain_started=False)
