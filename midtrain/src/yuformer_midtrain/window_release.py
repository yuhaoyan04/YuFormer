"""Map a fully admitted long-window release without rereading terabytes at startup."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .window_records import WindowRecords

CHECKS = {'source_binding', 'exact_window_bytes', 'document_bounds', 'heldout_disjointness',
          'replay_quota', 'domain_mixture', 'target_overlap', 'publication_durability'}


def dataset_id(release):
    components = {'index': release['record_index']['sha256'],
                  'byte_audit': release['source_byte_audit']['sha256'],
                  'admission': release['admission']['sha256'], 'seq_len': release['seq_len'],
                  'tokenizer_sha256': release['tokenizer_sha256']}
    raw = json.dumps(components, sort_keys=True, separators=(',', ':')).encode()
    return 'yuformer-dataset-'+hashlib.sha256(raw).hexdigest()[:20]


def stat_identity(value):
    return {'bytes': value.st_size, 'device': value.st_dev,
            'inode': value.st_ino, 'mtime_ns': value.st_mtime_ns}


def bound_path(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('window release assets must use explicit root-relative paths')
    return Path(root)/name


def bound_json(path, expected):
    if (not isinstance(expected, str) or len(expected) != 64
            or any(c not in '0123456789abcdef' for c in expected)):
        raise ValueError('window release dependency SHA-256 is required')
    path = Path(path)
    with path.open('rb') as handle:
        before = stat_identity(os.fstat(handle.fileno()))
        raw = handle.read()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('window release dependency SHA-256 mismatch')
        if stat_identity(os.fstat(handle.fileno())) != before or stat_identity(path.stat()) != before:
            raise ValueError('window release dependency changed while reading')
    return json.loads(raw)


def edge_sha256(handle):
    size = os.fstat(handle.fileno()).st_size
    handle.seek(0)
    digest = hashlib.sha256(str(size).encode('ascii'))
    digest.update(handle.read(1 << 20))
    if size > 1 << 20:
        handle.seek(max(0, size-(1 << 20)))
        digest.update(handle.read(1 << 20))
    return digest.hexdigest()


def load_window_release(path, *, expected_sha256, source_root, seq_len, tokenizer_sha256):
    if not Path(source_root).is_absolute() or not Path(path).is_absolute():
        raise ValueError('long-window release and external source root must be absolute declared paths')
    release = bound_json(path, expected_sha256)
    if (release.get('schema_version') != 1 or release.get('status') != 'PASS_LONG_CONTEXT_WINDOW_RELEASE'
            or release.get('layout') != 'source-index-s-plus-one-v1' or release.get('seq_len') != seq_len
            or release.get('tokenizer_sha256') != tokenizer_sha256 or release.get('short_replay_fraction') != .25):
        raise ValueError('long-window release is unqualified or has another length/tokenizer/layout')
    index = release.get('record_index', {})
    samples = release.get('samples')
    if (type(samples) is not int or samples <= 0 or samples % 8
            or release.get('target_labels') != samples*seq_len
            or index.get('dtype') != '<u8' or index.get('columns') != ['source_id', 'token_offset', 'short_replay']
            or index.get('bytes') != samples*3*8):
        raise ValueError('window release record geometry does not close exact target labels')
    if release.get('dataset_id') != dataset_id(release):
        raise ValueError('long-window dataset identity does not match its immutable components')
    binding = release.get('admission', {})
    admission = bound_json(bound_path(source_root, binding.get('file')), binding.get('sha256'))
    checks = admission.get('checks', {})
    if (admission.get('status') != 'PASS_LONG_WINDOW_DATA_ADMISSION' or set(checks) != CHECKS
            or any(value != 'PASS' for value in checks.values())
            or admission.get('record_index_sha256') != index.get('sha256')
            or admission.get('target_labels') != release['target_labels']
            or admission.get('seq_len') != seq_len):
        raise ValueError('long-window source/mixture/overlap/heldout/durability admission is incomplete')
    audit_binding = release.get('source_byte_audit', {})
    audit = bound_json(bound_path(source_root, audit_binding.get('file')), audit_binding.get('sha256'))
    if audit.get('status') != 'PASS_LONG_SOURCE_BYTE_BINDING':
        raise ValueError('long-window source bytes need a bound complete audit')
    sources = release.get('sources', [])
    if not isinstance(sources, list) or not sources or admission.get('sources') != sources:
        raise ValueError('window source roster differs from the admitted bytes')
    arrays = []
    for row in sources:
        artifact = bound_path(source_root, row.get('file'))
        if (row.get('role') not in ('long_document_source', 'packed_short_replay')
                or row.get('dtype') not in ('<u2', '<u4')
                or not isinstance(row.get('sha256'), str) or len(row['sha256']) != 64
                or any(c not in '0123456789abcdef' for c in row['sha256'])
                or row not in audit.get('artifacts', [])):
            raise ValueError('window source has no exact audited role/dtype/byte identity')
        with artifact.open('rb') as handle:
            before = stat_identity(os.fstat(handle.fileno()))
            if (before != row.get('stat') or type(row.get('tokens')) is not int
                    or row['tokens']*np.dtype(row['dtype']).itemsize != before['bytes']
                    or edge_sha256(handle) != row.get('edge_sha256')):
                raise ValueError('audited long-window source changed before mapping')
            array = np.memmap(handle, dtype=row['dtype'], mode='r')
            if stat_identity(os.fstat(handle.fileno())) != before or stat_identity(artifact.stat()) != before:
                raise ValueError('long-window source changed while mapping')
            arrays.append(array)
    artifact = bound_path(source_root, index.get('file'))
    with artifact.open('rb') as handle:
        before = stat_identity(os.fstat(handle.fileno()))
        if (before['bytes'] != index['bytes']
                or hashlib.file_digest(handle, 'sha256').hexdigest() != index.get('sha256')):
            raise ValueError('long-window index size/SHA-256 mismatch')
        records = np.memmap(handle, dtype='<u8', mode='r', shape=(samples, 3))
        if stat_identity(os.fstat(handle.fileno())) != before or stat_identity(artifact.stat()) != before:
            raise ValueError('long-window index changed while mapping')
    data = WindowRecords(arrays, records, seq_len=seq_len, identity=expected_sha256)
    for source_id, source in enumerate(sources):
        expected_kind = int(source['role'] == 'packed_short_replay')
        if np.any(data.index[data.index[:, 0] == source_id, 2] != expected_kind):
            raise ValueError('long/short window kinds disagree with the source roles')
    return data
