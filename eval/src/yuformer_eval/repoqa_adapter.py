"""Run selected, locked upstream RepoQA definitions with a native tokenizer."""
import ast
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import List, Tuple

REVISION = 'ae876deb1365dbf5a15b0533723c8ed123eee586'


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def locked_upstream(source):
    source = Path(source).resolve()
    lock_path = source/'upstream.lock.json'
    lock = json.loads(lock_path.read_bytes())
    if lock.get('revision') != REVISION or not lock.get('files'):
        raise ValueError('RepoQA upstream revision or source roster mismatch')
    pins = {str(lock_path): digest(lock_path)}
    for name, binding in lock['files'].items():
        path = (source/name).resolve()
        if not path.is_relative_to(source):
            raise ValueError('RepoQA source path escapes the locked tree')
        data = path.read_bytes()
        blob = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if digest(path) != binding['sha256'] or blob != binding['git_blob_sha1']:
            raise ValueError('RepoQA original source byte binding failed')
        pins[str(path)] = binding['sha256']
    return pins


def extract_definitions(path, names, namespace):
    tree = ast.parse(Path(path).read_bytes(), filename=str(path))
    selected = []
    for node in tree.body:
        name = node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in names:
            selected.append(node)
    found = {node.name if hasattr(node, 'name') else node.targets[0].id for node in selected}
    if found != set(names) or len(selected) != len(names):
        raise ValueError('RepoQA selected original definition roster changed')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def context_builder(source, tokenizer):
    source = Path(source).resolve()
    pins = locked_upstream(source)

    class NativeTokenizer:
        def tokenize(self, text):
            return tokenizer.encode(text, add_special_tokens=False).tokens

    class NativeFactory:
        @staticmethod
        def from_pretrained(name):
            if name != 'codellama/CodeLlama-7b-Instruct-hf':
                raise ValueError('unexpected upstream tokenizer request')
            return NativeTokenizer()

    env = {'Enum': Enum, 'List': List, 'Tuple': Tuple, 'AutoTokenizer': NativeFactory}
    extract_definitions(source/'repoqa/utility.py', {'topological_sort'}, env)
    extract_definitions(source/'repoqa/search_needle_function.py',
        {'COMMENT_PREFIX', 'TEMPLATE', 'INSTRUCTION', 'CleanComment',
         '_backward_tokenizable_lines', '_forward_tokenizable_lines', 'make_code_context'}, env)
    return env, pins


def reference_function(repo, needle):
    text = repo['content'][needle['path']]
    start, end = needle['start_byte'], needle['end_byte']
    if type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise ValueError('invalid original RepoQA reference range')
    try:
        byte_slice = text.encode('utf8')[start:end].decode('utf8')
    except UnicodeDecodeError as error:
        raise ValueError('RepoQA byte boundary splits a UTF-8 codepoint') from error
    line_slice = '\n'.join(text.split('\n')[needle['start_line']:needle['end_line']])
    char_slice = text[start:end]
    if not byte_slice or byte_slice != char_slice or byte_slice != line_slice.lstrip(' \t'):
        raise ValueError('RepoQA byte, character and line references disagree')
    return byte_slice


def render_prompt(template, instruction, description, context):
    fields = {'instruction': instruction, 'code_context': context,
              'description': '\nFunction Description:'+description+'\n'}
    return ''.join(fields[key] for key in template.split('\n'))


def paired_contexts(env, tokenizer, repo, needle, *, language, contexts, generation_tokens):
    reference = reference_function(repo, needle)
    if not reference or len(tokenizer.encode(reference, add_special_tokens=False).ids) > 512:
        raise ValueError('reference exceeds the predeclared 512 native tokens')
    paths = env['topological_sort'](repo['dependency'])
    files = [(path, repo['content'][path]) for path in paths]
    position = (repo['needles'].index(needle)+0.5)/len(repo['needles'])
    result = []
    for seq in contexts:
        empty = render_prompt(env['TEMPLATE'], env['INSTRUCTION'], needle['description'], '')
        overhead = len(tokenizer.encode(empty, add_special_tokens=False).ids)
        budget = seq-generation_tokens-overhead-64
        if budget <= 0:
            raise ValueError('context reservation is infeasible')
        admitted = None
        for numerator in (100, 103, 106, 109, 112):
            requested_budget = budget*numerator//100
            built = env['make_code_context'](needle, files, position, requested_budget, language)
            context = built['code_context']
            prompt = render_prompt(env['TEMPLATE'], env['INSTRUCTION'], needle['description'], context)
            count = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
            if reference in context and .75*seq <= count <= seq-generation_tokens:
                admitted = (built, prompt, count, requested_budget)
                break
        if admitted is None:
            raise ValueError('reference presence, 75pct length floor or whole-prompt budget failed')
        built, prompt, count, requested_budget = admitted
        result.append({'seq_len': seq, 'input': prompt, 'input_tokens': count,
            'no_context_input': empty, 'position_ratio': position,
            'requested_upstream_budget': requested_budget,
            'needle_token_start': built['needle_token_start'], 'reference': reference})
    return result
