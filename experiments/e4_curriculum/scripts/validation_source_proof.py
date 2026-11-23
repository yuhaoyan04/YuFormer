"""Bind independent validation re-encoding to actual admitted validation bytes."""
import hashlib
import json
from pathlib import Path

from training_data_origin import read_bound
from source_document_binding import verified_source_document_binding

DOMAINS = {'web', 'code', 'math', 'chinese', 'academic', 'books', 'encyclopedia', 'qa', 'multilingual'}


def checked_path(root, name):
    if not isinstance(name, str) or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('invalid archived validation proof path')
    return root/name


def require_validation_source_proof(root, audit, roster_fingerprint):
    root = Path(root).resolve()
    if audit.get('status') != 'PASS':
        raise ValueError('validation source proof requires admitted physical stream bytes')
    index = json.loads((root/'data/health/validation-source-proof-archive.json').read_bytes())
    if index.get('status') != 'PASS_IMMUTABLE_VALIDATION_SOURCE_PROOF_ARCHIVE':
        raise ValueError('independent full validation source proof has not been archived')
    manifest_path = checked_path(root, index.get('manifest'))
    manifest_raw, manifest_sha = read_bound(manifest_path, index.get('manifest_sha256'))
    manifest = json.loads(manifest_raw)
    if manifest.get('purpose') != 'IMMUTABLE_TRAINING_QUALIFICATION_EVIDENCE':
        raise ValueError('validation source proof uses an unidentified archive')
    records, dependencies = {}, {str(manifest_path.relative_to(root)): manifest_sha}
    for name, record in manifest['files'].items():
        path = checked_path(root, record.get('archive'))
        raw, digest = read_bound(path, record.get('sha256'))
        if len(raw) != record.get('bytes'):
            raise ValueError('validation source proof object length differs')
        records[name] = (raw, digest)
        dependencies[str(path.relative_to(root))] = digest
    report_name = 'data/health/validation-full-reencode-audit.json'
    if report_name not in records:
        raise ValueError('archive lacks the completed independent validation report')
    report = json.loads(records[report_name][0])
    if (report.get('status') != 'PASS_ALL_NINE_FULL_VALIDATION_STREAMS'
            or set(report.get('results', {})) != DOMAINS
            or report.get('production_assets_modified') is not False):
        raise ValueError('validation source proof must cover all nine unmodified production streams')
    rows = {row['name']: row for row in audit.get('streams', [])}
    materializer = 'pretrain/scripts/materialize_tokens.py'
    producer_sha = hashlib.sha256((root/materializer).read_bytes()).hexdigest()
    if materializer not in records or records[materializer][1] != producer_sha:
        raise ValueError('validation source proof belongs to another source-verified decoder')
    for domain in sorted(DOMAINS):
        result = report['results'][domain]
        old_name = f'data/token_streams/m3_20261020_yf64k/val_{domain}.meta.json'
        if old_name not in records or records[old_name][1] != result.get('old_meta_sha256'):
            raise ValueError('validation source proof lacks its original metadata: '+domain)
        old_raw, _ = read_bound(root/old_name, result['old_meta_sha256'])
        old = json.loads(old_raw)
        new_name = result.get('new_meta')
        checked_path(root, new_name)
        if new_name not in records or records[new_name][1] != result.get('new_meta_sha256'):
            raise ValueError('validation source proof lacks its independently generated metadata: '+domain)
        new = json.loads(records[new_name][0])
        ledger = new.get('doc_lengths') or {}
        ledger_name = str((Path(new_name).parent/str(ledger.get('file', ''))))
        if (ledger_name not in records or records[ledger_name][1] != ledger.get('sha256')
                or ledger.get('rows') != new.get('n_docs') or ledger.get('sum_tokens') != new.get('total_tokens')):
            raise ValueError('validation source proof lacks its exact document-length ledger: '+domain)
        row = rows.get('val_'+domain) or {}
        expected = result.get('original_and_reencoded_sha256')
        if (result.get('status') != 'PASS_EXACT_BYTE_EQUALITY'
                or result.get('new_source_document_ids_verified') is not True
                or not verified_source_document_binding(new)
                or new['source_document_binding']['roster_fingerprint'] != roster_fingerprint
                or new.get('materializer_tool_sha256') != producer_sha
                or row.get('status') != 'PASS' or row.get('actual_sha256') != expected
                or old.get('sha256') != expected or new.get('sha256') != expected
                or result.get('tokens') != old.get('total_tokens')
                or result.get('documents') != old.get('n_docs')
                or any(old.get(key) != new.get(key) for key in
                    ('bytes', 'n_docs', 'dtype', 'total_tokens', 'order_sha256',
                     'tokenizer_sha256', 'tokenizer_gate_sha256', 'data_snapshot_id'))):
            raise ValueError('validation source proof differs from actual admitted data: '+domain)
    tool = 'experiments/e4_curriculum/scripts/validation_source_proof.py'
    tool_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    read_bound(root/tool, tool_sha)
    dependencies[tool] = tool_sha
    return dependencies, {'manifest': str(manifest_path.relative_to(root)), 'sha256': manifest_sha,
                          'domains': sorted(DOMAINS), 'production_assets_modified': False}
