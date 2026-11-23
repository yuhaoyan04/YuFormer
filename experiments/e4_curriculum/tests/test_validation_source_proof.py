import hashlib
import json
from pathlib import Path
import struct
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'experiments/e4_curriculum/scripts'), str(ROOT/'pretrain/scripts')]
from frozen_training_evidence import snapshot_files
from validation_source_proof import require_validation_source_proof, DOMAINS


def write(root, name, value):
    path = root/name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else (json.dumps(value, sort_keys=True)+'\n').encode())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def proof(root, fault=None):
    helper = 'experiments/e4_curriculum/scripts/validation_source_proof.py'
    write(root, helper, (ROOT/helper).read_bytes())
    producer = 'pretrain/scripts/materialize_tokens.py'
    producer_sha = write(root, producer, b'source-verified-encoder')
    fingerprint = 'f'*64
    bindings = {producer: producer_sha}
    rows, results = [], {}
    for domain in sorted(DOMAINS):
        old_name = f'data/token_streams/m3_20261020_yf64k/val_{domain}.meta.json'
        new_name = f'data/processed/audits/validation_reencode_20261114/streams/val_{domain}.meta.json'
        payload = struct.pack('<HH', 42, 0)
        token_sha = hashlib.sha256(payload).hexdigest()
        old = dict(bytes=4, n_docs=1, dtype='<u2', total_tokens=2, sha256=token_sha,
                   order_sha256='a'*64, tokenizer_sha256='b'*64,
                   tokenizer_gate_sha256='c'*64, data_snapshot_id='snapshot')
        bindings[old_name] = write(root, old_name, old)
        ledger_name = str(Path(new_name).parent/f'val_{domain}.doc_lengths.u32')
        bindings[ledger_name] = write(root, ledger_name, struct.pack('<I', 2))
        new = dict(old, materializer_tool_sha256=producer_sha,
            source_document_binding=dict(schema_version=4, expected_doc_ids_verified=True,
                order_bytes_verified=True, source_handles_verified=True, tokenizer_bytes_verified=True,
                roster_fingerprint=fingerprint),
            doc_lengths=dict(file=Path(ledger_name).name, sha256=bindings[ledger_name], rows=1, sum_tokens=2))
        if domain == 'web' and fault == 'wrong_source_ids':
            new['source_document_binding']['expected_doc_ids_verified'] = False
        if domain == 'web' and fault == 'wrong_roster':
            new['source_document_binding']['roster_fingerprint'] = 'e'*64
        bindings[new_name] = write(root, new_name, new)
        results[domain] = dict(status='PASS_EXACT_BYTE_EQUALITY', tokens=2, documents=1,
            old_meta_sha256=bindings[old_name], new_meta=new_name, new_meta_sha256=bindings[new_name],
            original_and_reencoded_sha256=token_sha, new_source_document_ids_verified=True)
        rows.append(dict(name='val_'+domain, status='PASS', actual_sha256=token_sha))
    if fault == 'missing_domain':
        del results['web']
    if fault == 'missing_ledger':
        del bindings['data/processed/audits/validation_reencode_20261114/streams/val_web.doc_lengths.u32']
    name = 'data/health/validation-full-reencode-audit.json'
    bindings[name] = write(root, name, dict(status='PASS_ALL_NINE_FULL_VALIDATION_STREAMS',
        results=results, production_assets_modified=False))
    archive = snapshot_files(root, bindings)
    write(root, 'data/health/validation-source-proof-archive.json',
        dict(status='PASS_IMMUTABLE_VALIDATION_SOURCE_PROOF_ARCHIVE',
             manifest=archive['manifest'], manifest_sha256=archive['manifest_sha256']))
    if fault == 'wrong_admitted_bytes':
        rows[0]['actual_sha256'] = 'd'*64
    if fault == 'old_meta_drift':
        write(root, 'data/token_streams/m3_20261020_yf64k/val_web.meta.json', {'different': True})
    if fault == 'different_decoder':
        write(root, producer, b'changed decoder')
    if fault == 'corrupt_archive':
        write(root, archive['files'][name]['archive'], b'corrupt')
    return {'status': 'PASS', 'streams': rows}, fingerprint, archive


def test_all_nine_source_proofs_are_bound_to_immutable_evidence_and_actual_admission(tmp_path):
    audit, fingerprint, archive = proof(tmp_path)
    inputs, info = require_validation_source_proof(tmp_path, audit, fingerprint)
    assert info['domains'] == sorted(DOMAINS)
    assert info['sha256'] == archive['manifest_sha256']
    assert archive['manifest'] in inputs
    assert 'data/health/validation-full-reencode-audit.json' not in inputs
    for name, expected in inputs.items():
        assert hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() == expected
    write(tmp_path, 'data/health/validation-full-reencode-audit.json', {'status': 'reexported'})
    assert require_validation_source_proof(tmp_path, audit, fingerprint) == (inputs, info)


@pytest.mark.parametrize('fault', ['missing_domain', 'wrong_source_ids', 'wrong_roster',
    'missing_ledger', 'wrong_admitted_bytes', 'old_meta_drift', 'different_decoder', 'corrupt_archive'])
def test_forged_or_drifted_validation_source_proof_cannot_enter_training_freeze(tmp_path, fault):
    audit, fingerprint, _ = proof(tmp_path, fault)
    with pytest.raises(ValueError):
        require_validation_source_proof(tmp_path, audit, fingerprint)
