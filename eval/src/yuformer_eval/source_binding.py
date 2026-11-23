"""Bind inference to the actual training model and backend source closure."""
import ast
import hashlib
from pathlib import Path


def actual_training_source(root, expected_sha256):
    root = Path(root).resolve()
    validator = root/'pretrain/backends/torchtitan/validate_contract.py'
    tree = ast.parse(validator.read_bytes(), filename=str(validator))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_source_sha256']
    if len(selected) != 1:
        raise ValueError('actual training source digest definition is absent or duplicated')
    namespace = {'Path': Path, 'hashlib': hashlib}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(validator), 'exec'), namespace)
    if namespace['_source_sha256'](root) != expected_sha256:
        raise ValueError('evaluation model/backend source differs from the actual trained checkpoint')
    files = {p.resolve() for base in (root/'model/src', root/'pretrain/backends/torchtitan')
             for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    files.update(root/name for name in ('pretrain/scripts/run_with_lease.py',
        'pretrain/scripts/assert_run_lease.py', 'pretrain/src/yuformer_pretrain/lease.py'))
    files.add(Path(__file__).resolve())
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files) if p.is_file()}
