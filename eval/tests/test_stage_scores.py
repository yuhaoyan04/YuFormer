import copy
import json
from pathlib import Path

import pytest

from yuformer_eval.stage_scores import decide, intrinsic_scores, require_policy, ruler_scores

ROOT = Path(__file__).resolve().parents[2]


def policy():
    return json.loads((ROOT/'eval/configs/context_stage_gate_v1.json').read_bytes())


def metrics():
    return {'ruler': {'macro': .2, 'non_needle_macro': .1},
        'long_bpb': {'macro': 2., 'domains': {d: {'value': 2.} for d in ('books', 'code', 'math')}},
        'short_core': {'domains': {d: {'value': 3.} for d in policy()['short_core']['domains']}},
        'long_qa_code': {'context_macro': .2, 'no_context_macro': .05, 'context_minus_control': .15}}


def passing_pair():
    before = metrics()
    after = copy.deepcopy(before)
    after['ruler']['macro'] = .3
    after['long_bpb']['macro'] = 1.9
    after['long_bpb']['domains']['books']['value'] = 1.7
    return before, after


def test_a_single_short_domain_regression_cannot_be_hidden_by_other_improvements():
    before, after = passing_pair()
    assert all(row['status'] == 'PASS' for row in decide(policy(), before, after).values())
    after['short_core']['domains']['chinese']['value'] += .00001
    after['short_core']['domains']['code']['value'] -= 1.
    assert decide(policy(), before, after)['short_core']['status'] == 'FAIL'


def test_needle_only_improvement_and_zero_code_scores_do_not_qualify_a_stage():
    before, after = passing_pair()
    after['ruler']['non_needle_macro'] = .09
    assert decide(policy(), before, after)['ruler']['status'] == 'FAIL'
    for metric in (before, after):
        metric['long_qa_code'].update(context_macro=0., no_context_macro=0., context_minus_control=0.)
    assert decide(policy(), before, after)['long_qa_code']['status'] == 'FAIL'


def test_bpb_uses_original_bytes_and_equal_domain_macro_not_token_count():
    rows = [({'suite': 'long_bpb'}, {'domain': domain, 'uncompressed_utf8_bytes': byte_count},
             {'nll_sum': 4., 'target_tokens': 2}) for domain, byte_count in [('books', 2), ('code', 4)]]
    value = intrinsic_scores(rows, domains=['books', 'code'], samples_per_domain=1, byte_normalized=True)
    assert value['domains']['books']['value'] == pytest.approx(2*value['domains']['code']['value'])
    assert value['macro'] == pytest.approx(sum(row['value'] for row in value['domains'].values())/2)


def test_removing_a_required_domain_or_enabling_regression_is_rejected():
    changed = policy()
    changed['short_core']['domains'].remove('chinese')
    with pytest.raises(ValueError):
        require_policy(changed)
    changed = policy()
    changed['short_core']['relative_regression_margin'] = .01
    with pytest.raises(ValueError):
        require_policy(changed)


def test_ruler_refuses_partial_needle_only_rosters():
    with pytest.raises(ValueError, match='all thirteen tasks'):
        ruler_scores([({'task': 'niah_single_1'}, {'outputs': ['cat']}, {'prediction': 'cat'})], samples_per_task=1)
