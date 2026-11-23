import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('short_core_preparation', ROOT/'eval/scripts/prepare_short_core_cases.py')
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


def fixture(tmp_path, count=8193):
    tokenizer = tmp_path/'tokenizer.json'
    tokenizer.write_text('{}\n')
    gate = tmp_path/'gate.json'
    gate.write_text(json.dumps({'status': 'PASS', 'dataset_id': 'dataset', 'snapshot_id': 'snapshot',
        'tokenizer_path': str(tokenizer), 'tokenizer_sha256': PREPARE.digest(tokenizer), 'vocab_size': 65536}))
    release = tmp_path/'release.json'
    release.write_text(json.dumps({'status': 'PASS', 'dataset_id': 'dataset', 'snapshot_id': 'snapshot'}))
    rows = []
    for domain in PREPARE.DOMAINS:
        stream, meta, order = [tmp_path/(domain+suffix) for suffix in ('.u2', '.meta.json', '.order')]
        np.arange(count, dtype='<u2').tofile(stream)
        order.write_text('frozen heldout document roster\n')
        metadata = {'sha256': PREPARE.digest(stream), 'tokenizer_sha256': PREPARE.digest(tokenizer),
            'order_sha256': PREPARE.digest(order), 'total_tokens': count, 'dtype': '<u2'}
        meta.write_text(json.dumps(metadata))
        rows.append({'name': 'val_'+domain, 'status': 'PASS', 'checks': {'full_sha256_exact': True},
            'errors': [], 'artifacts': dict(stream=str(stream), meta=str(meta), order=str(order)),
            'actual_sha256': metadata['sha256'], 'actual_total_tokens_with_eod': count,
            **{'actual_'+k: v for k, v in PREPARE.identity(stream.stat()).items()}})
    audit = tmp_path/'audit.json'
    audit.write_text(json.dumps({'status': 'PASS', 'streams': rows}))
    return dict(root=tmp_path, audit_file=audit, gate_file=gate, release_file=release,
                output=tmp_path/'cases', samples=2, contexts=[4096, 16384, 65536])


def test_short_targets_are_identical_at_every_context_and_have_global_case_ids(tmp_path):
    args = fixture(tmp_path)
    manifest = PREPARE.prepare(**args)
    assert manifest['status'] == 'PASS_IMMUTABLE_PAIRED_NLL_CASE_MANIFEST'
    assert len(manifest['files']) == 27
    indices = []
    for domain in PREPARE.DOMAINS:
        entries = [r for r in manifest['files'] if r['domain'] == domain]
        assert len({r['sha256'] for r in entries}) == 1
        rows = [json.loads(line) for line in Path(entries[0]['file']).read_text().splitlines()]
        assert rows[0]['token_ids'] == list(range(4097))
        assert rows[1]['token_ids'] == list(range(4096, 8193))
        indices.extend(r['index'] for r in rows)
    assert sorted(indices) == list(range(18))


def test_s_plus_one_capacity_is_required_before_publication(tmp_path):
    args = fixture(tmp_path, count=8192)
    with pytest.raises(ValueError, match='heldout binding/capacity'):
        PREPARE.prepare(**args)
    assert not args['output'].exists()


def test_changed_heldout_bytes_cannot_become_a_new_case_release(tmp_path):
    args = fixture(tmp_path)
    stream = tmp_path/'books.u2'
    with stream.open('r+b') as handle:
        handle.seek(20)
        handle.write(b'\xff\xff')
    with pytest.raises(ValueError, match='heldout binding/capacity'):
        PREPARE.prepare(**args)
    assert not args['output'].exists()
