from enum import Enum

import pytest

from yuformer_eval.repoqa_adapter import extract_definitions, reference_function, render_prompt


def test_prompt_follows_original_template_without_replacing_words_inside_values():
    prompt = render_prompt('instruction\ncode_context\ndescription\ninstruction',
                           'Read the description.', 'Look for instruction.', 'description = 1\n')
    assert prompt == ('Read the description.description = 1\n'
                      '\nFunction Description:Look for instruction.\nRead the description.')


def test_reference_preserves_exact_crlf_bytes():
    text = 'def f():\r\n    return 3\r\n'
    needle = {'path': 'a.py', 'start_byte': 0, 'end_byte': len(text)-1,
              'start_line': 0, 'end_line': 2}
    assert reference_function({'content': {'a.py': text}}, needle) == text[:-1]


def test_nonascii_byte_and_character_offset_disagreement_is_rejected():
    text = '# 中文\ndef f():\n    return 3\n'
    start = text.encode().index(b'def')
    needle = {'path': 'a.py', 'start_byte': start, 'end_byte': len(text.encode())-1,
              'start_line': 1, 'end_line': 3}
    with pytest.raises(ValueError, match='references disagree'):
        reference_function({'content': {'a.py': text}}, needle)


def test_first_line_indentation_is_the_only_permitted_line_range_difference():
    text = '    def f():\n        return 3\n'
    needle = {'path': 'a.py', 'start_byte': 4, 'end_byte': len(text)-1,
              'start_line': 0, 'end_line': 2}
    assert reference_function({'content': {'a.py': text}}, needle) == text[4:-1]
    needle['end_byte'] -= 1
    with pytest.raises(ValueError, match='references disagree'):
        reference_function({'content': {'a.py': text}}, needle)


def test_selected_ast_definitions_do_not_execute_upstream_module_side_effects(tmp_path):
    path = tmp_path/'source.py'
    path.write_text('raise RuntimeError("must not execute")\nclass Mode(Enum):\n X=1\ndef fn(x):\n return x+1\n')
    namespace = extract_definitions(path, {'Mode', 'fn'}, {'Enum': Enum})
    assert namespace['fn'](5) == 6
    assert namespace['Mode'].X.value == 1
    with pytest.raises(ValueError, match='roster changed'):
        extract_definitions(path, {'missing'}, {})
