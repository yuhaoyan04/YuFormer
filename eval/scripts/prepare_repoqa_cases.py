#!/usr/bin/env python3
"""Freeze native-tokenizer RepoQA diagnostic cases without running a model."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'eval/src'))

from tokenizers import Tokenizer
from yuformer_eval.repoqa_adapter import REVISION, context_builder, digest, paired_contexts

LANGUAGES = ('python', 'cpp', 'java', 'typescript', 'rust', 'go')
CONTEXTS = (4096, 16384, 65536)


def seal(path, value):
    with Path(path).open('xb') as handle:
        handle.write((json.dumps(value, sort_keys=True, indent=2)+'\n').encode())
        handle.flush()
        os.fsync(handle.fileno())


def prepare(root, source, output, *, seed, samples):
    root, source, output = [Path(p).resolve() for p in (root, source, output)]
    if output.exists() or type(samples) is not int or samples != 8 or not seed:
        raise ValueError('use a new output, an explicit seed and the fixed eight functions per language')
    gate_file = root/'data/TOKENIZER_READY.json'
    gate = json.loads(gate_file.read_bytes())
    tokenizer_path = Path(gate['tokenizer_path'])
    if not tokenizer_path.is_absolute():
        tokenizer_path = root/tokenizer_path
    if gate.get('status') != 'PASS' or digest(tokenizer_path) != gate['tokenizer_sha256']:
        raise ValueError('actual training tokenizer has not been admitted')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    env, pins = context_builder(source, tokenizer)
    asset_lock_path = source/'dataset.lock.json'
    asset = json.loads(asset_lock_path.read_bytes())
    dataset_path = source/('repoqa-'+asset['version']+'.json')
    gzip_path = dataset_path.with_suffix('.json.gz')
    if (asset.get('status') != 'PASS_OFFICIAL_VERSIONED_REPOQA_ASSET_DOWNLOADED'
            or asset.get('source_revision') != REVISION or asset['version'] != '2024-06-23'
            or digest(dataset_path) != asset['json_sha256'] or digest(gzip_path) != asset['gzip_sha256']
            or digest(source/'upstream.lock.json') != asset['upstream_source_lock_sha256']):
        raise ValueError('original versioned RepoQA dataset binding failed')
    dataset = json.loads(dataset_path.read_bytes())
    if set(dataset) != set(LANGUAGES):
        raise ValueError('all six actual release languages are required')
    for path in (asset_lock_path, dataset_path, gzip_path, gate_file, tokenizer_path,
                 Path(__file__).resolve(), ROOT/'eval/src/yuformer_eval/repoqa_adapter.py',
                 ROOT/'eval/docs/REPOQA_NATIVE_DIAGNOSTIC.md'):
        pins[str(path)] = digest(path)
    versions = {}
    for name in ('tokenizers',):
        dist = importlib.metadata.distribution(name)
        versions[name] = dist.version
        for item in dist.files or []:
            if Path(str(item)).suffix in ('.py', '.so') or item.name in ('METADATA', 'RECORD'):
                path = Path(dist.locate_file(item)).resolve()
                if path.is_file():
                    pins[str(path)] = digest(path)

    def guard():
        if any(digest(path) != sha for path, sha in pins.items()):
            raise ValueError('RepoQA case preparation source or tokenizer changed')

    guard()
    output.mkdir(parents=True)
    selections, rejected, all_cases = [], [], {seq: [] for seq in CONTEXTS}
    for language_index, language in enumerate(LANGUAGES):
        candidates = []
        identities = set()
        for repo in dataset[language]:
            for needle in repo['needles']:
                identity = (language, repo['repo'], needle['path'], needle['name'])
                if identity in identities:
                    raise ValueError('original RepoQA candidate identity is ambiguous')
                identities.add(identity)
                rank = hashlib.blake2b(json.dumps([seed, *identity], separators=(',', ':')).encode(),
                                       digest_size=32).hexdigest()
                candidates.append((rank, identity, repo, needle))
        admitted = 0
        for rank, identity, repo, needle in sorted(candidates, key=lambda item: (item[0], item[1])):
            try:
                pairs = paired_contexts(env, tokenizer, repo, needle, language=language,
                                         contexts=CONTEXTS, generation_tokens=768)
            except ValueError as error:
                rejected.append({'identity': identity, 'rank': rank, 'reason': str(error)})
                continue
            case_index = language_index*samples+admitted
            selections.append({'case_index': case_index, 'language': language, 'repo': repo['repo'],
                'commit_sha': repo['commit_sha'], 'path': needle['path'], 'name': needle['name'],
                'rank': rank, 'start_byte': needle['start_byte'], 'end_byte': needle['end_byte']})
            for pair in pairs:
                for condition in ('context', 'no_context'):
                    prompt = pair['input'] if condition == 'context' else pair['no_context_input']
                    all_cases[pair['seq_len']].append({'index': case_index*2+int(condition == 'no_context'),
                        'paired_function_index': case_index, 'input': prompt, 'outputs': [pair['reference']],
                        'input_tokens': len(tokenizer.encode(prompt, add_special_tokens=False).ids),
                        'language': language, 'repo': repo['repo'], 'name': needle['name'],
                        'path': needle['path'], 'condition': condition, 'position_ratio': pair['position_ratio'],
                        'requested_upstream_budget': pair['requested_upstream_budget']})
            admitted += 1
            if admitted == samples:
                break
        if admitted != samples:
            seal(output/'rejected-preparation.json', {'status': 'FAIL_INSUFFICIENT_FIXED_PROTOCOL_CASES',
                'language': language, 'admitted': admitted, 'rejected': rejected})
            raise ValueError('fixed native RepoQA protocol cannot cover '+language)
        print(json.dumps({'language': language, 'admitted_functions': admitted,
                          'rejected_so_far': len(rejected)}), flush=True)
    manifest = {'schema_version': 1, 'status': 'PASS_IMMUTABLE_PAIRED_REPOQA_CASE_MANIFEST',
        'scope': 'NATIVE_TOKENIZER_DIAGNOSTIC_SUBSET_WITH_NO_CONTEXT_CONTROLS_NOT_FULL_REPOQA_SCORE',
        'tokenizer_sha256': gate['tokenizer_sha256'], 'source_revision': REVISION,
        'dataset_version': asset['version'], 'seed': seed, 'samples_per_language': samples,
        'languages': LANGUAGES, 'contexts': CONTEXTS, 'generation_tokens': 768,
        'reference_max_tokens': 512, 'minimum_context_fraction': .75,
        'fixed_context_budget_multipliers': [1., 1.03, 1.06, 1.09, 1.12],
        'score_protocol': {'best_annotated_function_match': True, 'similarity_threshold': .8,
            'ignore_comments': False, 'similarity': 'UPSTREAM_WHITESPACE_BLEU_NLTK_SMOOTHING_METHOD4'},
        'contamination_scope': 'PUBLIC_REPOSITORIES_NOT_CERTIFIED_EXCLUDED_FROM_PRETRAINING',
        'selected_functions': selections, 'rejected_candidates': rejected,
        'source_inputs': pins, 'runtime': {'python': sys.version, 'packages': versions}, 'files': []}
    for seq in CONTEXTS:
        path = output/f'repoqa-{seq}.jsonl'
        with path.open('x') as handle:
            for row in all_cases[seq]:
                handle.write(json.dumps(row, sort_keys=True, separators=(',', ':'))+'\n')
            handle.flush()
            os.fsync(handle.fileno())
        manifest['files'].append({'file': str(path), 'sha256': digest(path),
            'suite': 'long_qa_code', 'dataset': 'repoqa', 'seq_len': seq,
            'samples': len(all_cases[seq]), 'generation_tokens': 768})
    guard()
    seal(output/'manifest.json', manifest)
    fd = os.open(output, os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'source', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--seed', required=True)
    parser.add_argument('--samples-per-language', type=int, required=True)
    args = parser.parse_args()
    result = prepare(args.root, args.source, args.output, seed=args.seed, samples=args.samples_per_language)
    print(json.dumps({'status': result['status'], 'files': len(result['files']),
                      'selected_functions': len(result['selected_functions'])}, sort_keys=True))
