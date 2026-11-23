#!/usr/bin/env python3
"""Evaluate an immutable case shard against an actual native checkpoint.

Workers produce measurements only. A separate paired admission must compare
the trained candidate with its actual baseline before stage progression.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'eval/src'), str(ROOT/'midtrain/src')]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case_roster(manifest_file, *, seq_len, worker_index, workers):
    if type(workers) is not int or workers <= 0 or not 0 <= worker_index < workers:
        raise ValueError('case worker index must belong to the explicit worker roster')
    manifest_file = Path(manifest_file).resolve()
    manifest = json.loads(manifest_file.read_bytes())
    statuses = {'PASS_IMMUTABLE_PAIRED_RULER_CASE_MANIFEST': 'ruler',
                'PASS_IMMUTABLE_PAIRED_REPOQA_CASE_MANIFEST': 'repoqa',
                'PASS_IMMUTABLE_PAIRED_NLL_CASE_MANIFEST': 'nll'}
    kind = statuses.get(manifest.get('status'))
    if kind is None:
        raise ValueError('evaluation requires an independently frozen paired case manifest')
    pins = {str(manifest_file): digest(manifest_file)}
    for name, sha in manifest.get('source_inputs', {}).items():
        path = Path(name)
        if not path.is_absolute() or digest(path) != sha:
            raise ValueError('case preparation dependency changed')
        pins[str(path)] = sha
    if not manifest.get('source_inputs'):
        raise ValueError('case manifest lacks actual tokenizer/source preparation bindings')
    rows, seen, total = [], set(), 0
    files = [row for row in manifest.get('files', []) if row.get('seq_len') == seq_len]
    if not files:
        raise ValueError('case manifest does not contain the requested context')
    for entry in files:
        path = Path(entry['file'])
        if not path.is_absolute() or digest(path) != entry['sha256']:
            raise ValueError('actual evaluation cases changed')
        pins[str(path)] = entry['sha256']
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if not records or len(records) != entry['samples']:
            raise ValueError('case file has another sample roster')
        for row in records:
            identity = (entry.get('task', entry.get('suite')), row.get('index'))
            if identity[0] is None or identity[1] is None or identity in seen:
                raise ValueError('evaluation case identity is absent or duplicated')
            seen.add(identity)
            if kind in {'ruler', 'repoqa'}:
                if (not isinstance(row.get('outputs'), list) or not row['outputs']
                        or any(not isinstance(x, str) or not x for x in row['outputs'])
                        or not isinstance(row.get('input'), str) or not row['input']):
                    raise ValueError('generation case lacks actual prompt/references')
                if type(entry.get('generation_tokens')) is not int or entry['generation_tokens'] <= 0:
                    raise ValueError('generation output token reservation must be explicit')
                if kind == 'repoqa' and (
                        entry.get('suite') != 'long_qa_code' or entry.get('dataset') != 'repoqa'
                        or row.get('language') not in {'python', 'cpp', 'java', 'typescript', 'rust', 'go'}
                        or row.get('condition') not in {'context', 'no_context'}
                        or type(row.get('paired_function_index')) is not int
                        or not all(isinstance(row.get(k), str) and row[k] for k in ('repo', 'name', 'path'))):
                    raise ValueError('RepoQA function/control identity is invalid')
            else:
                tokens = row.get('token_ids')
                if (entry.get('suite') not in {'long_bpb', 'long_qa_code', 'short_core'}
                        or not isinstance(tokens, list) or not 2 <= len(tokens) <= seq_len+1
                        or any(type(t) is not int or t < 0 for t in tokens)):
                    raise ValueError('NLL case token/role/context geometry is invalid')
                if entry['suite'] == 'long_bpb' and (
                        type(row.get('uncompressed_utf8_bytes')) is not int
                        or row['uncompressed_utf8_bytes'] <= 0):
                    raise ValueError('BPB requires a positive bound original UTF-8 byte count')
            if total % workers == worker_index:
                rows.append((entry, row))
            total += 1
    if not rows:
        raise ValueError('this case shard is empty; reduce the explicit worker count')
    return manifest, kind, rows, total, pins


def evaluate(args):
    started = time.monotonic()
    import torch
    from tokenizers import Tokenizer
    from yuformer_eval.native_inference import NativeCheckpointEvaluator
    from yuformer_eval.source_binding import actual_training_source
    if torch.cuda.device_count() != 1:
        raise ValueError('each independent evaluation worker must see exactly one explicitly assigned GPU')
    uuid = getattr(torch.cuda.get_device_properties(0), 'uuid', None)
    if uuid is None:
        raise ValueError('evaluation worker must record its actual CUDA device UUID')
    gpu_uuid = str(uuid)
    manifest, kind, roster, total, pins = case_roster(args.cases, seq_len=args.seq_len,
        worker_index=args.worker_index, workers=args.workers)
    output = args.output.resolve()
    if output.exists():
        raise ValueError('preserve previous measurements; select a new evaluation output directory')
    tokenizer_path = args.tokenizer.resolve()
    if digest(tokenizer_path) != manifest['tokenizer_sha256']:
        raise ValueError('evaluation tokenizer differs from the frozen cases')
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    for path in [Path(__file__), ROOT/'eval/src/yuformer_eval/native_inference.py', tokenizer_path]:
        pins[str(path.resolve())] = digest(path)
    handoff = json.loads(args.handoff.read_bytes())
    identity_file = Path(handoff['checkpoint']).parent.parent/'run_identity.json'
    identity = json.loads(identity_file.read_bytes())
    pins.update(actual_training_source(ROOT, identity['source_sha256']))
    evaluator = NativeCheckpointEvaluator(handoff_file=args.handoff, model_file=args.model,
        preset=args.preset, seq_len=args.seq_len, device=torch.device('cuda:0'))
    if evaluator.identity['tokenizer_sha256'] != manifest['tokenizer_sha256']:
        raise ValueError('actual checkpoint tokenizer differs from the evaluation cases')
    pins.update(evaluator.inputs)

    def guard():
        if any(digest(name) != sha for name, sha in pins.items()):
            raise ValueError('evaluation input/source changed during measurement')

    guard()
    output.mkdir(parents=True)
    predictions = output/'measurements.jsonl'
    with predictions.open('x') as handle:
        for entry, row in roster:
            record = {'index': row['index'], 'task': entry.get('task'), 'suite': entry.get('suite'),
                      'case_file_sha256': entry['sha256'], 'seq_len': args.seq_len}
            if kind in {'ruler', 'repoqa'}:
                prompt = tokenizer.encode(row['input']+row.get('answer_prefix', ''), add_special_tokens=False).ids
                generated = evaluator.generate(prompt, entry['generation_tokens'])
                record.update(prediction=tokenizer.decode(generated, skip_special_tokens=True),
                              outputs=row['outputs'], generated_token_ids=generated)
                if kind == 'repoqa':
                    record.update({key: row[key] for key in ('language', 'repo', 'name', 'path',
                                                            'condition', 'paired_function_index')})
            else:
                record.update(evaluator.nll(row['token_ids'], target_mask=row.get('target_mask')))
                for key in ('domain', 'doc_id', 'uncompressed_utf8_bytes'):
                    if key in row:
                        record[key] = row[key]
            handle.write(json.dumps(record, sort_keys=True)+'\n')
            handle.flush()
        os.fsync(handle.fileno())
    guard()
    receipt = {'status': 'PASS_NATIVE_CHECKPOINT_MEASUREMENTS',
        'scope': 'MEASUREMENTS_ONLY_NOT_STAGE_CAPABILITY_ADMISSION', 'kind': kind,
        'case_protocol_scope': manifest.get('scope'), 'seq_len': args.seq_len,
        'worker_index': args.worker_index, 'workers': args.workers,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'gpu_uuid': gpu_uuid,
        'measured_cases': len(roster), 'total_case_roster': total,
        'checkpoint_handoff_sha256': digest(args.handoff),
        'checkpoint_handoff_file': str(args.handoff.resolve()),
        'case_manifest_sha256': digest(args.cases),
        'case_manifest_file': str(args.cases.resolve()),
        'run_identity_sha256': json.loads(args.handoff.read_bytes())['run_identity_sha256'],
        'wallclock_seconds_including_load': time.monotonic()-started,
        'decoding': 'DETERMINISTIC_GREEDY_ACTUAL_EOS_NO_SAMPLING',
        'inference_attention': 'flash_flex_swa',
        'measurements': str(predictions), 'measurements_sha256': digest(predictions),
        'source_inputs': pins}
    with (output/'receipt.json').open('x') as handle:
        handle.write(json.dumps(receipt, sort_keys=True, indent=2)+'\n')
        handle.flush()
        os.fsync(handle.fileno())
    fd = os.open(output, os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cases', 'handoff', 'model', 'tokenizer', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--preset', required=True)
    parser.add_argument('--seq-len', type=int, choices=[4096, 16384, 65536], required=True)
    parser.add_argument('--worker-index', type=int, required=True)
    parser.add_argument('--workers', type=int, required=True)
    result = evaluate(parser.parse_args())
    print(json.dumps({key: result[key] for key in ('status', 'scope', 'measured_cases')}, sort_keys=True))
