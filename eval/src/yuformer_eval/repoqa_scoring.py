"""Use original locked RepoQA matching functions without model-client imports."""
from enum import Enum
import importlib.metadata
import math
from pathlib import Path
import re
from typing import Dict, Tuple

from .repoqa_adapter import digest, extract_definitions, locked_upstream


def scorer(source):
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    from tree_sitter_languages import get_language, get_parser

    source = Path(source).resolve()
    pins = locked_upstream(source)
    namespace = {'Enum': Enum, 'Dict': Dict, 'Tuple': Tuple, 're': re,
        'SmoothingFunction': SmoothingFunction, 'sentence_bleu': sentence_bleu,
        'get_language': get_language, 'get_parser': get_parser}
    extract_definitions(source/'repoqa/utility.py', {'FUNCTION_QUERY', 'COMMENT_QUERY'}, namespace)
    extract_definitions(source/'repoqa/metric.py', {'compute_function_similarity'}, namespace)
    extract_definitions(source/'repoqa/compute_score.py',
                        {'Result', 'remove_comments', 'sanitize_output', 'needle_evaluator'}, namespace)
    for name in ('nltk', 'tree_sitter', 'tree_sitter_languages', 'regex'):
        dist = importlib.metadata.distribution(name)
        for item in dist.files or []:
            if Path(str(item)).suffix in ('.py', '.so') or item.name in ('METADATA', 'RECORD'):
                path = Path(dist.locate_file(item)).resolve()
                if path.is_file():
                    pins[str(path)] = digest(path)
    for path in (Path(__file__).resolve(), Path(__file__).with_name('repoqa_adapter.py')):
        pins[str(path)] = digest(path)

    def score(prediction, repo, *, language, name):
        if not isinstance(prediction, str) or name not in {n['name'] for n in repo['needles']}:
            raise ValueError('actual RepoQA prediction/reference identity is missing')
        verdict, target, similarity = namespace['needle_evaluator'](prediction, name, repo, language, False)
        similarity = float(similarity)
        if not math.isfinite(similarity) or not 0 <= similarity <= 1:
            raise ValueError('upstream RepoQA similarity is nonfinite or outside [0,1]')
        best_match = verdict == namespace['Result'].BEST_MATCH
        return {'best_match': best_match, 'best_target': target, 'similarity': similarity,
                'pass_at_1_threshold_08': best_match and similarity >= .8}

    return score, pins
