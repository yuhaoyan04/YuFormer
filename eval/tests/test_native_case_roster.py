import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('native_checkpoint_measurement_worker',
    ROOT/'eval/scripts/evaluate_native_checkpoint.py')
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(tmp_path, rows, **entry):
    source = tmp_path/'source.txt'
    source.write_text('frozen original source\n')
    cases = tmp_path/'cases.jsonl'
    cases.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'status': 'PASS_IMMUTABLE_PAIRED_NLL_CASE_MANIFEST',
        'source_inputs': {str(source): digest(source)},
        'files': [{'file': str(cases), 'sha256': digest(cases), 'samples': len(rows),
                   'seq_len': 16384, 'suite': 'long_bpb', **entry}]}))
    return manifest


def test_every_case_is_assigned_exactly_once_across_eight_workers(tmp_path):
    rows = [{'index': index, 'token_ids': [7, 8, 9], 'uncompressed_utf8_bytes': 5}
            for index in range(29)]
    manifest = freeze(tmp_path, rows)
    assigned = []
    for rank in range(8):
        _, _, roster, total, _ = WORKER.case_roster(manifest, seq_len=16384, worker_index=rank, workers=8)
        assert total == 29
        assigned.extend(row['index'] for _, row in roster)
    assert sorted(assigned) == list(range(29))


def test_bpb_cannot_be_reported_without_original_byte_denominator(tmp_path):
    manifest = freeze(tmp_path, [{'index': 0, 'token_ids': [7, 8]}])
    with pytest.raises(ValueError, match='UTF-8 byte count'):
        WORKER.case_roster(manifest, seq_len=16384, worker_index=0, workers=1)


def test_duplicate_case_identity_cannot_inflate_an_evaluation(tmp_path):
    row = {'index': 0, 'token_ids': [7, 8], 'uncompressed_utf8_bytes': 2}
    manifest = freeze(tmp_path, [row, row])
    with pytest.raises(ValueError, match='duplicated'):
        WORKER.case_roster(manifest, seq_len=16384, worker_index=0, workers=1)


@pytest.mark.parametrize('fault', ['cases', 'source', 'context'])
def test_case_or_source_drift_and_wrong_context_are_rejected(tmp_path, fault):
    manifest = freeze(tmp_path, [{'index': 0, 'token_ids': [7, 8], 'uncompressed_utf8_bytes': 2}])
    seq = 16384
    if fault == 'context':
        seq = 65536
    else:
        target = tmp_path/('cases.jsonl' if fault == 'cases' else 'source.txt')
        target.write_text('changed\n')
    with pytest.raises(ValueError):
        WORKER.case_roster(manifest, seq_len=seq, worker_index=0, workers=1)


def test_repoqa_context_and_control_are_distinct_and_keep_original_function_identity(tmp_path):
    rows = [{'index': i, 'input': 'Retrieve the described function.', 'outputs': ['def f(): return 1'],
             'language': 'python', 'repo': 'repo', 'name': 'f', 'path': 'a.py',
             'condition': condition, 'paired_function_index': 0}
            for i, condition in enumerate(('context', 'no_context'))]
    manifest = freeze(tmp_path, rows, suite='long_qa_code', dataset='repoqa', generation_tokens=768)
    data = json.loads(manifest.read_bytes())
    data['status'] = 'PASS_IMMUTABLE_PAIRED_REPOQA_CASE_MANIFEST'
    manifest.write_text(json.dumps(data))
    _, kind, roster, total, _ = WORKER.case_roster(manifest, seq_len=16384, worker_index=0, workers=1)
    assert kind == 'repoqa' and total == 2
    assert [row['condition'] for _, row in roster] == ['context', 'no_context']
    rows[1]['condition'] = 'unbound alternative'
    cases = tmp_path/'cases.jsonl'
    cases.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    data['files'][0]['sha256'] = digest(cases)
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='control identity'):
        WORKER.case_roster(manifest, seq_len=16384, worker_index=0, workers=1)
