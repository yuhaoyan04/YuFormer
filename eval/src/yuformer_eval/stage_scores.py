"""Quantitative paired stage decisions from independently reconstructed cases."""
from collections import defaultdict
import math

RULER_TASKS = {'niah_single_1', 'niah_single_2', 'niah_single_3', 'niah_multikey_1',
    'niah_multikey_2', 'niah_multikey_3', 'niah_multivalue', 'niah_multiquery',
    'vt', 'cwe', 'fwe', 'qa_1', 'qa_2'}


def require_policy(policy):
    if (policy.get('policy_id') != 'yuformer-native-context-stage-gate-v1-20261115'
            or policy.get('workers') != 8 or policy.get('contexts') != [16384, 65536]
            or policy['short_core']['relative_regression_margin'] != 0.0
            or policy['ruler']['task_count'] != 13 or policy['ruler']['samples_per_task'] != 32
            or policy['long_bpb']['samples_per_domain'] != 8
            or policy['short_core']['samples_per_domain'] != 2816
            or policy['short_core']['target_tokens_per_case'] != 4096
            or policy['long_qa_code']['samples_per_language'] != 8
            or policy['long_qa_code']['similarity_threshold'] != .8
            or set(policy['short_core']['domains']) != {'academic', 'books', 'chinese', 'code',
                'encyclopedia', 'math', 'multilingual', 'qa', 'web'}
            or set(policy['long_bpb']['domains']) != {'books', 'code', 'math'}
            or set(policy['long_qa_code']['languages']) != {'python', 'cpp', 'java', 'typescript', 'rust', 'go'}):
        raise ValueError('stage scoring policy differs from the fixed production diagnostic protocol')
    requirements = {'ruler': ['require_strict_macro_improvement', 'require_non_needle_macro_nonregression'],
        'long_bpb': ['require_strict_equal_domain_macro_improvement', 'require_each_domain_nonregression'],
        'long_qa_code': ['require_context_macro_nonregression', 'require_context_minus_control_nonregression',
                         'require_at_least_one_correct_context_case'],
        'short_core': ['require_every_domain_nonregression']}
    if any(policy[suite][name] is not True for suite, names in requirements.items() for name in names):
        raise ValueError('stage capability requirements cannot be disabled')


def ruler_scores(rows, *, samples_per_task):
    scores = defaultdict(list)
    for entry, case, record in rows:
        task = entry['task']
        matches = [float(reference.lower() in record['prediction'].lower()) for reference in case['outputs']]
        score = max(matches) if task in {'qa_1', 'qa_2'} else sum(matches)/len(matches)
        scores[task].append(score)
    if set(scores) != RULER_TASKS or any(len(v) != samples_per_task for v in scores.values()):
        raise ValueError('RULER requires all thirteen tasks with the exact fixed case count')
    by_task = {task: sum(values)/len(values) for task, values in sorted(scores.items())}
    non_needle = [value for task, value in by_task.items() if not task.startswith('niah_')]
    return {'task_scores': by_task, 'macro': sum(by_task.values())/len(by_task),
            'non_needle_macro': sum(non_needle)/len(non_needle)}


def intrinsic_scores(rows, *, domains, samples_per_domain, byte_normalized, target_tokens=None):
    grouped = defaultdict(list)
    for entry, case, record in rows:
        if entry['suite'] != ('long_bpb' if byte_normalized else 'short_core'):
            raise ValueError('intrinsic measurement belongs to another suite')
        if target_tokens is not None and record['target_tokens'] != target_tokens:
            raise ValueError('short-core case does not score exactly the original 4096 targets')
        grouped[case['domain']].append((case, record))
    if set(grouped) != set(domains) or any(len(v) != samples_per_domain for v in grouped.values()):
        raise ValueError('intrinsic suite domain roster or original case count changed')
    result = {}
    for domain, values in sorted(grouped.items()):
        total = math.fsum(record['nll_sum'] for _, record in values)
        denominator = sum(case['uncompressed_utf8_bytes'] if byte_normalized else record['target_tokens']
                          for case, record in values)
        metric = total/denominator/(math.log(2) if byte_normalized else 1)
        if not math.isfinite(metric) or metric < 0:
            raise ValueError('nonfinite intrinsic metric')
        result[domain] = {'value': metric, 'nll_sum': total, 'denominator': denominator,
                          'cases': len(values), 'denominator_unit': 'UTF8_BYTES' if byte_normalized else 'TARGET_TOKENS'}
    return {'domains': result, 'macro': sum(row['value'] for row in result.values())/len(result)}


def code_scores(rows, *, languages, samples_per_language, dataset, score_function):
    grouped = defaultdict(list)
    scores = []
    seen = set()
    for _, case, record in rows:
        language, condition = case['language'], case['condition']
        key = (case['paired_function_index'], condition)
        if key in seen:
            raise ValueError('RepoQA function/control pair is duplicated')
        seen.add(key)
        repos = [repo for repo in dataset[language] if repo['repo'] == case['repo']]
        if len(repos) != 1:
            raise ValueError('RepoQA reference repository is ambiguous')
        score = score_function(record['prediction'], repos[0], language=language, name=case['name'])
        grouped[(language, condition)].append(float(score['pass_at_1_threshold_08']))
        scores.append({'index': case['index'], 'language': language, 'condition': condition, **score})
    expected = {(language, condition) for language in languages for condition in ('context', 'no_context')}
    if set(grouped) != expected or any(len(v) != samples_per_language for v in grouped.values()):
        raise ValueError('RepoQA requires every language and paired context/control case')
    functions = {index for index, _ in seen}
    if any((index, condition) not in seen for index in functions for condition in ('context', 'no_context')):
        raise ValueError('RepoQA context and control do not cover the same functions')
    by_language = {language: {condition: sum(grouped[(language, condition)])/samples_per_language
        for condition in ('context', 'no_context')} for language in languages}
    context = sum(row['context'] for row in by_language.values())/len(languages)
    control = sum(row['no_context'] for row in by_language.values())/len(languages)
    return {'languages': by_language, 'context_macro': context, 'no_context_macro': control,
            'context_minus_control': context-control, 'case_scores': scores}


def decide(policy, baseline, candidate):
    require_policy(policy)
    decisions = {}
    for suite in ('ruler', 'long_bpb', 'long_qa_code', 'short_core'):
        before, after = baseline[suite], candidate[suite]
        if suite == 'ruler':
            checks = {'strict_macro_improvement': after['macro'] > before['macro'],
                'non_needle_nonregression': after['non_needle_macro'] >= before['non_needle_macro']}
        elif suite == 'long_bpb':
            checks = {'strict_macro_improvement': after['macro'] < before['macro'],
                **{'domain_nonregression/'+d: after['domains'][d]['value'] <= before['domains'][d]['value']
                   for d in policy[suite]['domains']}}
        elif suite == 'short_core':
            checks = {'domain_nonregression/'+d: after['domains'][d]['value'] <= before['domains'][d]['value']
                      for d in policy[suite]['domains']}
        else:
            checks = {'context_nonregression': after['context_macro'] >= before['context_macro'],
                'context_benefit_nonregression': after['context_minus_control'] >= before['context_minus_control'],
                'nonzero_context_capability': after['context_macro'] > 0}
        decisions[suite] = {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
                            'baseline': before, 'candidate': after}
    return decisions
