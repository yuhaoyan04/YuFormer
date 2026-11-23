import hashlib
import json

import numpy as np
import pytest

from yuformer_midtrain.window_release import load_window_release, stat_identity, edge_sha256, CHECKS, dataset_id


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def release(tmp_path):
    sources = []
    for i, role in enumerate(('long_document_source', 'packed_short_replay')):
        path = tmp_path/f'source{i}.bin'
        np.arange(i*1000, i*1000+256, dtype='<u2').tofile(path)
        with path.open('rb') as handle:
            edge = edge_sha256(handle)
        sources.append({'file': path.name, 'dtype': '<u2', 'tokens': 256, 'role': role,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'edge_sha256': edge,
            'stat': stat_identity(path.stat())})
    rows = np.array([(int(i in (1, 6)), i*12, int(i in (1, 6))) for i in range(8)], dtype='<u8')
    path = tmp_path/'index.bin'
    rows.tofile(path)
    index = {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
             'bytes': path.stat().st_size, 'dtype': '<u8',
             'columns': ['source_id', 'token_offset', 'short_replay']}
    admission = {'status': 'PASS_LONG_WINDOW_DATA_ADMISSION', 'checks': dict.fromkeys(CHECKS, 'PASS'),
                 'record_index_sha256': index['sha256'], 'target_labels': 64, 'seq_len': 8, 'sources': sources}
    admission_sha = write(tmp_path/'admission.json', admission)
    audit_sha = write(tmp_path/'byte-audit.json', {'status': 'PASS_LONG_SOURCE_BYTE_BINDING', 'artifacts': sources})
    document = {'schema_version': 1, 'status': 'PASS_LONG_CONTEXT_WINDOW_RELEASE',
        'layout': 'source-index-s-plus-one-v1', 'seq_len': 8, 'tokenizer_sha256': 'a'*64,
        'short_replay_fraction': .25, 'samples': 8, 'target_labels': 64, 'record_index': index,
        'admission': {'file': 'admission.json', 'sha256': admission_sha},
        'source_byte_audit': {'file': 'byte-audit.json', 'sha256': audit_sha}, 'sources': sources}
    path = tmp_path/'release.json'
    document['dataset_id'] = dataset_id(document)
    digest = write(path, document)
    return path, digest, document, admission


def load(path, digest):
    return load_window_release(path, expected_sha256=digest, source_root=path.parent,
                               seq_len=8, tokenizer_sha256='a'*64)


def test_exact_admitted_window_records_map_without_flat_stream_drift(release):
    path, digest, _, _ = release
    records = load(path, digest)
    for i in range(8):
        x, y = records.sample(i)
        start = i*12 + (1000 if i in (1, 6) else 0)
        assert x.tolist() == list(range(start, start+8))
        assert y.tolist() == list(range(start+1, start+9))


@pytest.mark.parametrize('fault', ['source_bytes', 'index_bytes', 'missing_overlap_gate',
                                  'wrong_tokenizer', 'path_escape', 'incomplete_release'])
def test_changed_or_partially_admitted_release_is_rejected(release, fault):
    path, digest, document, admission = release
    if fault == 'source_bytes':
        (path.parent/'source0.bin').write_bytes(b'\0'*512)
    elif fault == 'index_bytes':
        with (path.parent/'index.bin').open('r+b') as handle:
            handle.write(b'\xff'*8)
    elif fault == 'missing_overlap_gate':
        admission['checks'].pop('target_overlap')
        document['admission']['sha256'] = write(path.parent/'admission.json', admission)
        digest = write(path, document)
    elif fault == 'wrong_tokenizer':
        document['tokenizer_sha256'] = 'b'*64
        digest = write(path, document)
    elif fault == 'path_escape':
        document['record_index']['file'] = '../index.bin'
        digest = write(path, document)
    else:
        document['status'] = 'PREPARED_NOT_ADMITTED'
        digest = write(path, document)
    with pytest.raises(ValueError):
        load(path, digest)
