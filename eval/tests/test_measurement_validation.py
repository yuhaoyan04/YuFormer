import pytest
import json
from pathlib import Path

from yuformer_eval.measurements import digest, load_measurements, merge_pins, validate_record


def test_target_mask_and_true_bpb_byte_count_are_bound_independently():
    entry = {'suite': 'long_bpb', 'sha256': 'a'*64}
    case = {'index': 0, 'token_ids': [1, 2, 3, 4], 'target_mask': [True, False, True],
            'domain': 'books', 'uncompressed_utf8_bytes': 6}
    record = {'index': 0, 'task': None, 'suite': 'long_bpb', 'case_file_sha256': 'a'*64,
              'seq_len': 16384, 'nll_sum': 4., 'target_tokens': 2, 'mean_nll': 2.,
              'domain': 'books', 'uncompressed_utf8_bytes': 6}
    validate_record(entry, case, record, kind='nll', seq_len=16384)
    for key, value in [('target_tokens', 3), ('uncompressed_utf8_bytes', 5),
                       ('nll_sum', float('nan')), ('mean_nll', 4.), ('domain', 'code')]:
        with pytest.raises(ValueError):
            validate_record(entry, case, {**record, key: value}, kind='nll', seq_len=16384)


def test_generation_cannot_use_another_reference_or_exceed_reserved_tokens():
    entry = {'task': 'qa_1', 'sha256': 'a'*64, 'generation_tokens': 2}
    case = {'index': 8, 'outputs': ['answer']}
    record = {'index': 8, 'task': 'qa_1', 'suite': None, 'case_file_sha256': 'a'*64,
              'seq_len': 16384, 'prediction': 'answer', 'outputs': ['answer'], 'generated_token_ids': [5]}
    validate_record(entry, case, record, kind='ruler', seq_len=16384)
    for fault in ({'outputs': ['other']}, {'generated_token_ids': [1, 2, 3]}, {'index': 7}):
        with pytest.raises(ValueError):
            validate_record(entry, case, {**record, **fault}, kind='ruler', seq_len=16384)


def test_workers_cannot_silently_merge_different_source_revisions(tmp_path):
    path = str(tmp_path/'model.py')
    pins = {path: 'a'*64}
    with pytest.raises(ValueError, match='disagree'):
        merge_pins(pins, {path: 'b'*64})


def measurement_fixture(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source = tmp_path/'source.txt'
    source.write_text('original source\n')
    cases = tmp_path/'cases.jsonl'
    rows = [{'index': i, 'token_ids': [1, 2], 'domain': 'books', 'uncompressed_utf8_bytes': 2}
            for i in range(17)]
    cases.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'status': 'PASS_IMMUTABLE_PAIRED_NLL_CASE_MANIFEST',
        'source_inputs': {str(source): digest(source)}, 'files': [{'seq_len': 16384,
            'suite': 'long_bpb', 'file': str(cases), 'sha256': digest(cases), 'samples': 17}]}))
    handoff = tmp_path/'handoff.json'
    handoff.write_text(json.dumps({'status': 'PASS_COMPLETE_CHECKPOINT_BYTE_SNAPSHOT',
                                  'run_identity_sha256': 'e'*64}))
    receipts = []
    for worker in range(8):
        output = tmp_path/f'measurements-{worker}.jsonl'
        measured = [{'index': row['index'], 'task': None, 'suite': 'long_bpb',
            'case_file_sha256': digest(cases), 'seq_len': 16384, 'nll_sum': 2.,
            'target_tokens': 1, 'mean_nll': 2., 'domain': 'books', 'uncompressed_utf8_bytes': 2}
            for row in rows if row['index'] % 8 == worker]
        output.write_text(''.join(json.dumps(row)+'\n' for row in measured))
        receipt = tmp_path/f'receipt-{worker}.json'
        receipt.write_text(json.dumps({'status': 'PASS_NATIVE_CHECKPOINT_MEASUREMENTS',
            'kind': 'nll', 'seq_len': 16384, 'workers': 8, 'worker_index': worker,
            'case_manifest_file': str(manifest), 'case_manifest_sha256': digest(manifest),
            'checkpoint_handoff_file': str(handoff), 'checkpoint_handoff_sha256': digest(handoff),
            'run_identity_sha256': 'e'*64, 'total_case_roster': 17,
            'gpu_uuid': f'UNIT_TEST_GPU_{worker}', 'cuda_visible_devices': str(worker),
            'wallclock_seconds_including_load': 1., 'measured_cases': len(measured),
            'measurements': str(output), 'measurements_sha256': digest(output),
            'source_inputs': {str(p): digest(p) for p in (
                root/'eval/scripts/evaluate_native_checkpoint.py',
                root/'eval/src/yuformer_eval/native_inference.py')}}))
        receipts.append(receipt)
    return manifest, handoff, receipts


@pytest.mark.parametrize('fault', [None, 'missing_worker', 'duplicate_gpu', 'wrong_checkpoint', 'wrong_shard'])
def test_complete_eight_worker_reconstruction_rejects_roster_checkpoint_and_device_faults(tmp_path, fault):
    manifest, handoff, receipts = measurement_fixture(tmp_path)
    if fault == 'missing_worker':
        receipts.pop()
    elif fault:
        path = receipts[-1]
        receipt = json.loads(path.read_bytes())
        if fault == 'duplicate_gpu':
            receipt['gpu_uuid'] = 'UNIT_TEST_GPU_0'
        elif fault == 'wrong_checkpoint':
            receipt['checkpoint_handoff_sha256'] = 'f'*64
        else:
            output = Path(receipt['measurements'])
            rows = [json.loads(line) for line in output.read_bytes().splitlines()]
            rows[0]['index'] = 0
            output.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            receipt['measurements_sha256'] = digest(output)
        path.write_text(json.dumps(receipt))
    if fault:
        with pytest.raises(ValueError):
            load_measurements(manifest, receipts, seq_len=16384, handoff_file=handoff)
    else:
        _, kind, actual, _ = load_measurements(manifest, receipts, seq_len=16384, handoff_file=handoff)
        assert kind == 'nll' and [row['index'] for _, _, row in actual] == list(range(17))
