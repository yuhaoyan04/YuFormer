#!/usr/bin/env python3
"""Bind actual heldout raw UTF-8 prefixes to audited validation token bytes."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'eval/src'))
from yuformer_eval.heldout_prefix import bounded_utf8_prefix, verify_encoded_document


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def stat_identity(stat):
    return dict(bytes=stat.st_size, device=stat.st_dev, inode=stat.st_ino, mtime_ns=stat.st_mtime_ns)


def seal(path, raw):
    with path.open('xb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def prepare(args):
    root, output = args.root.resolve(), args.output.resolve()
    if (output.exists() or args.samples_per_domain <= 0 or not args.contexts
            or len(set(args.contexts)) != len(args.contexts)
            or any(s not in (16384, 65536) for s in args.contexts)
            or not args.domains or len(set(args.domains)) != len(args.domains)):
        raise ValueError('new output, positive samples and explicit unique domains/long contexts are required')
    roster, audit, gate, release = [json.loads(p.read_bytes()) for p in
        (args.source_roster, args.stream_audit, args.tokenizer_gate, args.data_release)]
    if (roster['status'] != 'PASS_EXISTING_SOURCE_ROSTER_MATCHES_VALIDATION_REENCODE'
            or audit.get('status') != 'PASS' or gate.get('status') != 'PASS' or release.get('status') != 'PASS'
            or release['dataset_id'] != gate['dataset_id'] or release['snapshot_id'] != gate['snapshot_id']):
        raise ValueError('raw roster/physical stream/tokenizer/release admission is incomplete')
    tokenizer_file = Path(gate['tokenizer_path'])
    if not tokenizer_file.is_absolute():
        tokenizer_file = root/tokenizer_file
    if digest(tokenizer_file) != gate['tokenizer_sha256']:
        raise ValueError('actual native tokenizer changed')
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    if tokenizer.get_vocab_size() != gate['vocab_size']:
        raise ValueError('native vocabulary differs from its actual tokenizer gate')
    paths = [Path(__file__).resolve(), root/'eval/src/yuformer_eval/heldout_prefix.py',
        args.source_roster.resolve(), args.stream_audit.resolve(), args.tokenizer_gate.resolve(),
        args.data_release.resolve(), tokenizer_file]
    pins = {str(p): digest(p) for p in paths}
    roster_rows = [{'path': Path(row['file']).relative_to(roster['s3_root']).as_posix(), **row['stat']}
                   for row in roster['files']]
    roster_sha = hashlib.sha256(('s3-doc-index-v2\0'+json.dumps(
        roster_rows, sort_keys=True, separators=(',', ':'))).encode()).hexdigest()
    if roster_sha != roster['fingerprint']:
        raise ValueError('recorded raw source roster does not reproduce its actual admitted fingerprint')
    for name, sha in roster['source_inputs'].items():
        if digest(name) != sha:
            raise ValueError('source roster producer/validation binding changed')
        pins[name] = sha
    dist = importlib.metadata.distribution('tokenizers')
    for item in dist.files or []:
        path = Path(dist.locate_file(item)).resolve()
        if path.is_file() and (path.suffix in ('.py', '.so') or path.name in ('METADATA', 'RECORD')):
            pins[str(path)] = digest(path)
    source_files = roster['files']
    streams = {row['name']: row for row in audit['streams']}
    output.mkdir(parents=True)
    (output/'raw_sources').mkdir()
    (output/'prefixes').mkdir()
    manifest = {'schema_version': 1, 'status': 'PREPARING_IMMUTABLE_PAIRED_NLL_CASE_MANIFEST',
        'scope': 'ACTUAL_HELDOUT_RAW_UTF8_LONG_BPB_PAIRED_PREFIXES', 'seed': args.seed,
        'tokenizer_sha256': gate['tokenizer_sha256'], 'dataset_id': gate['dataset_id'],
        'domains': args.domains, 'contexts': args.contexts, 'samples_per_domain': args.samples_per_domain,
        'source_inputs': pins, 'source_documents': [], 'files': [],
        'metric_definition': 'Body-token conditional NLL after an EOD context token, divided by ln(2) and the exact original UTF-8 prefix byte count. No artificial trailing EOD target.',
        'selection_policy': 'fixed blake2b-ranked unique heldout documents long enough for the largest declared context; no model metric selection'}
    manifest['document_ledger_origin'] = 'independent full reencoding whose complete token bytes, order, tokenizer, document count and total labels match the unchanged original validation stream'
    seen = set()
    for domain_index, domain in enumerate(args.domains):
        row = streams['val_'+domain]
        stream, meta_file, order_file = [Path(row['artifacts'][k]) for k in ('stream', 'meta', 'order')]
        meta = json.loads(meta_file.read_bytes())
        expected_stat = {key: row['actual_'+key] for key in ('bytes', 'device', 'inode', 'mtime_ns')}
        reencode_meta_file = args.validation_reencode_dir/(meta['name']+'.meta.json')
        reencode = json.loads(reencode_meta_file.read_bytes())
        ledger = reencode['doc_lengths']
        ledger_file = args.validation_reencode_dir/ledger['file']
        if (row['status'] != 'PASS' or row['errors'] or not all(row['checks'].values())
                or stat_identity(stream.stat()) != expected_stat or digest(stream) != row['actual_sha256']
                or meta['sha256'] != row['actual_sha256'] or meta['tokenizer_sha256'] != gate['tokenizer_sha256']
                or meta['data_snapshot_id'] != gate['snapshot_id'] or meta['dtype'] not in ('<u2', '<u4')
                or reencode['sha256'] != meta['sha256'] or reencode['order_sha256'] != meta['order_sha256']
                or reencode['n_docs'] != meta['n_docs'] or reencode['total_tokens'] != meta['total_tokens']
                or reencode['tokenizer_sha256'] != gate['tokenizer_sha256']
                or ledger['dtype'] != '<u4' or ledger.get('includes_appended_eod') is not True
                or reencode['source_document_binding']['roster_fingerprint'] != roster['fingerprint']
                or digest(order_file) != meta['order_sha256'] or digest(ledger_file) != ledger['sha256']):
            raise ValueError('actual full raw/encoded heldout binding differs: '+domain)
        lengths = np.fromfile(ledger_file, dtype='<u4')
        if (len(lengths) != meta['n_docs'] or int(lengths.sum(dtype=np.uint64)) != meta['total_tokens']
                or meta['total_tokens']*np.dtype(meta['dtype']).itemsize != expected_stat['bytes']):
            raise ValueError('actual heldout document ledger geometry differs')
        original_ids = pq.read_table(order_file, columns=['doc_id'])['doc_id'].to_pylist()[:len(lengths)]
        if len(original_ids) != len(lengths) or len(set(original_ids)) != len(original_ids):
            raise ValueError('heldout source IDs are absent or duplicated')
        candidates = [i for i, length in enumerate(lengths) if int(length) >= max(args.contexts)+1 and original_ids[i] not in seen]
        candidates.sort(key=lambda i: hashlib.blake2b((args.seed+'\0'+domain+'\0'+original_ids[i]).encode(), digest_size=16).digest())
        if len(candidates) < args.samples_per_domain:
            raise ValueError('actual heldout long-document supply is insufficient: '+domain)
        selected = candidates[:args.samples_per_domain]
        # The existing tiny preparation cache is a lookup hint, never an
        # authority: every requested ID and the complete reencoded document are
        # checked against the actual original validation stream below.
        caches = []
        for path in (args.validation_reencode_dir/'.prep_cache').glob('*.npz'):
            with np.load(path, allow_pickle=False) as data:
                if len(data['spos']) == len(lengths):
                    caches.append((path, {k: np.array(data[k]) for k in ('sfi', 'srg', 'srow', 'spos')}))
        if len(caches) != 1:
            raise ValueError('existing validation lookup cache is absent or ambiguous')
        cache_file, cache = caches[0]
        if any(len(a) != len(lengths) for a in cache.values()) or not np.array_equal(np.sort(cache['spos']), np.arange(len(lengths))):
            raise ValueError('validation lookup positions do not cover the actual source roster')
        for path in (stream, meta_file, order_file, ledger_file, reencode_meta_file, cache_file):
            pins[str(path.resolve())] = digest(path)
        offsets = np.concatenate([np.array([0], dtype=np.uint64), np.cumsum(lengths, dtype=np.uint64)])
        arrays = np.memmap(stream, mode='r', dtype=meta['dtype'])
        cases = {seq: [] for seq in args.contexts}
        for local, document in enumerate(selected):
            position = int(np.flatnonzero(cache['spos'] == document)[0])
            file_index, group, source_row = [int(cache[k][position]) for k in ('sfi', 'srg', 'srow')]
            source = source_files[file_index]
            source_path = Path(source['file'])
            if stat_identity(source_path.stat()) != source['stat']:
                raise ValueError('raw source handle changed since admitted roster')
            with source_path.open('rb') as handle:
                if stat_identity(os.fstat(handle.fileno())) != source['stat']:
                    raise ValueError('opened raw source differs from admitted roster')
                parquet = pq.ParquetFile(handle)
                begin = sum(parquet.metadata.row_group(i).num_rows for i in range(group))
                group_rows = parquet.read_row_group(group, columns=['doc_id', 'text'], use_threads=False)
                local_row = source_row-begin
                if not 0 <= local_row < group_rows.num_rows or group_rows['doc_id'][local_row].as_py() != original_ids[document]:
                    raise ValueError('actual raw source ID/row differs from frozen heldout identity')
                text = group_rows['text'][local_row].as_py()
                if stat_identity(os.fstat(handle.fileno())) != source['stat']:
                    raise ValueError('raw source changed during document fetch')
            encoded = tokenizer.encode(text, add_special_tokens=False).ids
            eod = meta['document_boundary']['token_id']
            verify_encoded_document(encoded, arrays[int(offsets[document]):int(offsets[document+1])], eod_token_id=eod)
            doc_id = original_ids[document]
            seen.add(doc_id)
            raw_file = output/'raw_sources'/(domain+'-'+doc_id+'.txt')
            seal(raw_file, text.encode('utf-8'))
            pins[str(raw_file)] = digest(raw_file)
            manifest['source_documents'].append({'doc_id': doc_id, 'domain': domain, 'validation_document_index': document,
                'encoded_tokens_with_eod': int(lengths[document]), 'raw_file': str(raw_file), 'raw_sha256': pins[str(raw_file)],
                'source_file': str(source_path), 'source_stat': source['stat'], 'source_row': source_row, 'source_row_group': group})
            for seq in args.contexts:
                prefix = bounded_utf8_prefix(text, tokenizer, target_tokens=seq, eod_token_id=eod)
                prefix_file = output/'prefixes'/(str(seq)+'-'+doc_id+'.txt')
                seal(prefix_file, prefix['text'].encode('utf-8'))
                pins[str(prefix_file)] = digest(prefix_file)
                cases[seq].append({'index': domain_index*args.samples_per_domain+local, 'domain': domain, 'doc_id': doc_id,
                    'token_ids': prefix['token_ids'], 'uncompressed_utf8_bytes': prefix['uncompressed_utf8_bytes'],
                    'target_tokens': prefix['target_tokens'], 'raw_prefix_sha256': pins[str(prefix_file)]})
        for seq, records in cases.items():
            path = output/(str(seq)+'-'+domain+'.jsonl')
            seal(path, ''.join(json.dumps(row, sort_keys=True, separators=(',', ':'))+'\n' for row in records).encode())
            manifest['files'].append({'file': str(path), 'sha256': digest(path), 'suite': 'long_bpb', 'domain': domain,
                'seq_len': seq, 'samples': len(records)})
        print('prepared_actual_raw_utf8_domain', domain, len(selected), flush=True)
    if any(digest(path) != sha for path, sha in pins.items()):
        raise ValueError('heldout BPB source/producer changed during preparation')
    manifest['status'] = 'PASS_IMMUTABLE_PAIRED_NLL_CASE_MANIFEST'
    seal(output/'manifest.json', (json.dumps(manifest, sort_keys=True, indent=2)+'\n').encode())
    descriptor = os.open(output, os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'source-roster', 'stream-audit', 'tokenizer-gate', 'data-release', 'validation-reencode-dir', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--samples-per-domain', type=int, required=True)
    parser.add_argument('--domains', nargs='+', choices=['books', 'code', 'math'], required=True)
    parser.add_argument('--contexts', nargs='+', type=int, choices=[16384, 65536], required=True)
    parser.add_argument('--seed', required=True)
    result = prepare(parser.parse_args())
    print(json.dumps({'status': result['status'], 'scope': result['scope'], 'files': len(result['files']),
                      'source_documents': len(result['source_documents'])}, sort_keys=True))
