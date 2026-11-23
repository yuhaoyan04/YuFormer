#!/usr/bin/env python3
"""Execute an unmodified pinned RULER task with the explicit local tokenizer."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import runpy
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ruler-root',type=Path,required=True)
    parser.add_argument('--task-script',required=True)
    args,remaining=parser.parse_known_args()
    root=args.ruler_root.resolve()
    lock=json.loads((root/'upstream.lock.json').read_bytes())
    if lock['revision']!='db43f92ff634bbce1c33d1fb15858aec4bba509a':
        raise ValueError('RULER task generator revision differs from the frozen evaluation protocol')
    for name,row in lock['files'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('pinned RULER source changed: '+name)
    allowed={'niah','variable_tracking','common_words_extraction','freq_words_extraction','qa'}
    if args.task_script not in allowed:raise ValueError('unrecognized RULER task generator')
    sys.path[:0]=[str(root/'scripts/data/synthetic'),str(root/'scripts/data'),
        str(Path(__file__).resolve().parents[1]/'src')]
    from yuformer_eval.ruler_tokenizer import RulerYuFormerTokenizer
    import tokenizer as ruler_tokenizer
    # No upstream source edit, remote model code or alternate vocabulary: this
    # replaces only the HF tokenizer wrapper with its exact native decoder API.
    ruler_tokenizer.HFTokenizer=RulerYuFormerTokenizer
    script=root/'scripts/data/synthetic'/f'{args.task_script}.py'
    sys.argv=[str(script),*remaining]
    runpy.run_path(str(script),run_name='__main__')


if __name__=='__main__':main()
