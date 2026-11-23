import copy

import pytest

from yuformer_eval.completed_run import DOMAINS, validate_completion


def final_fixture():
    identity = {'identity_sha256': 'id'}
    config = {'data': {'seq_len': 16384}, 'train': {'expected_iters': 7152, 'budget_tokens': 29997662208,
        'micro_batch_size': 2, 'grad_accum': 16}, 'parallel': {'world_size': 8}, 'midtrain': {'scope': 'PRODUCTION'}}
    complete = {'status': 'PASS', 'identity_sha256': 'id', 'attempt_id': 'actual'}
    result = {**complete, 'metrics': {'validation_metrics/domain/'+d+'/loss': {'last': 3., 'last_step': 7152}
                                    for d in DOMAINS}}
    accounting = {'status': 'PASS', 'attempt_id': 'actual', 'step': 7152}
    return identity, config, complete, result, accounting, copy.deepcopy(complete)


@pytest.mark.parametrize('fault', [None, 'partial', 'qualification', 'stale_domain', 'nonfinite', 'different_attempt'])
def test_only_actual_complete_production_with_final_finite_nine_domain_validation_is_accepted(fault):
    identity, config, complete, result, accounting, attempt = final_fixture()
    if fault == 'partial':
        accounting['step'] -= 1
    elif fault == 'qualification':
        config['midtrain']['scope'] = 'QUALIFICATION'
    elif fault == 'stale_domain':
        result['metrics']['validation_metrics/domain/chinese/loss']['last_step'] -= 128
    elif fault == 'nonfinite':
        result['metrics']['validation_metrics/domain/chinese/loss']['last'] = float('nan')
    elif fault == 'different_attempt':
        attempt['attempt_id'] = 'old'
    kwargs = {'seq_len': 16384, 'stage': {'updates': 7152, 'tokens': 29997662208}}
    if fault:
        with pytest.raises(ValueError):
            validate_completion(identity, config, complete, result, accounting, attempt, **kwargs)
    else:
        assert validate_completion(identity, config, complete, result, accounting, attempt, **kwargs)['step'] == 7152
