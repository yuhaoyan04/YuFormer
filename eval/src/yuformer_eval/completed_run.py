"""Reject incomplete runs before accepting stage capability measurements."""
import math

DOMAINS = {'academic', 'books', 'chinese', 'code', 'encyclopedia', 'math', 'multilingual', 'qa', 'web'}


def validate_completion(identity, config, completion, result, accounting, attempt, *, seq_len, stage):
    step = stage['updates']
    samples = step*config['parallel']['world_size']*config['train']['micro_batch_size']*config['train']['grad_accum']
    if (config['data']['seq_len'] != seq_len or config['train']['expected_iters'] != step
            or config['train']['budget_tokens'] != stage['tokens'] or samples*seq_len != stage['tokens']
            or config['parallel']['world_size'] != 8
            or any(row.get('status') != 'PASS' for row in (completion, result, accounting, attempt))
            or any(row.get('identity_sha256') != identity['identity_sha256'] for row in (completion, result, attempt))
            or not completion.get('attempt_id')
            or any(row.get('attempt_id') != completion['attempt_id'] for row in (result, accounting, attempt))
            or accounting.get('step') != step):
        raise ValueError('stage capability requires the actual complete production budget, run and final attempt')
    if seq_len != 4096 and config.get('midtrain', {}).get('scope') != 'PRODUCTION':
        raise ValueError('qualification context runs cannot supply production capability admission')
    for domain in DOMAINS:
        metric = result.get('metrics', {}).get('validation_metrics/domain/'+domain+'/loss', {})
        value = metric.get('last')
        if (metric.get('last_step') != step or type(value) not in (int, float)
                or not math.isfinite(value) or value < 0):
            raise ValueError('completed production checkpoint lacks finite final validation in every domain')
    return {'step': step, 'samples': samples, 'tokens': stage['tokens'], 'seq_len': seq_len}
