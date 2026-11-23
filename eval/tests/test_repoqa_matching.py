import os
from pathlib import Path

import pytest


def test_original_repoqa_parser_and_matching_reject_wrong_function():
    source = os.environ.get('YUFORMER_REPOQA_SOURCE')
    if source is None:
        pytest.skip('requires the explicitly declared locked upstream and isolated Tree-sitter scorer environment')
    from yuformer_eval.repoqa_scoring import scorer
    score, pins = scorer(Path(source))
    text = 'def wanted(x):\n    return x + 1\n\ndef other(x):\n    return x * 7\n'
    repo = {'content': {'a.py': text}, 'needles': [
        {'name': 'wanted', 'path': 'a.py', 'start_line': 0, 'end_line': 2},
        {'name': 'other', 'path': 'a.py', 'start_line': 3, 'end_line': 5}]}
    correct = score('```python\ndef wanted(x):\n    return x + 1\n```', repo,
                    language='python', name='wanted')
    wrong = score('```python\ndef other(x):\n    return x * 7\n```', repo,
                  language='python', name='wanted')
    assert correct['pass_at_1_threshold_08'] is True and correct['similarity'] == 1
    assert wrong['pass_at_1_threshold_08'] is False and wrong['best_target'] == 'other'
    assert any('languages.so' in name for name in pins)
