"""Reconstruct complete, checkpoint-bound measurement rosters independently."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def merge_pins(destination, incoming):
    for name, sha in incoming.items():
        if not Path(name).is_absolute() or not isinstance(sha, str) or len(sha) != 64:
            raise ValueError('measurement provenance must use absolute files with SHA-256 digests')
        if name in destination and destination[name] != sha:
            raise ValueError('evaluation workers disagree about a shared dependency')
        destination[name] = sha


def validate_record(entry, case, record, *, kind, seq_len):
    if (record.get('seq_len') != seq_len or record.get('index') != case['index']
            or record.get('task') != entry.get('task') or record.get('suite') != entry.get('suite')
            or record.get('case_file_sha256') != entry['sha256']):
        raise ValueError('actual measurement does not match its assigned immutable case')
    if kind == 'nll':
        tokens = case['token_ids']
        mask = case.get('target_mask', [True]*(len(tokens)-1))
        if (len(mask) != len(tokens)-1 or any(type(x) is not bool for x in mask)
                or sum(mask) <= 0 or type(record.get('target_tokens')) is not int
                or record.get('target_tokens') != sum(mask)):
            raise ValueError('actual NLL denominator differs from the original target mask')
        total, mean = record.get('nll_sum'), record.get('mean_nll')
        if (type(total) not in (int, float) or type(mean) not in (int, float)
                or not math.isfinite(total) or not math.isfinite(mean) or min(total, mean) < 0
                or not math.isclose(mean, total/sum(mask), rel_tol=1e-10, abs_tol=1e-12)):
            raise ValueError('NLL sum/mean is nonfinite or inconsistent with its target denominator')
        for key in ('domain', 'doc_id', 'uncompressed_utf8_bytes'):
            if key in case and record.get(key) != case[key]:
                raise ValueError('NLL source/domain/UTF-8 denominator differs from its frozen case')
    else:
        generated = record.get('generated_token_ids')
        if (not isinstance(record.get('prediction'), str) or record.get('outputs') != case['outputs']
                or not isinstance(generated, list) or not 1 <= len(generated) <= entry['generation_tokens']
                or any(type(t) is not int or t < 0 for t in generated)):
            raise ValueError('generation output/reference/token reservation is invalid')
        if kind == 'repoqa' and any(record.get(k) != case[k] for k in
                ('language', 'repo', 'name', 'path', 'condition', 'paired_function_index')):
            raise ValueError('actual RepoQA function/control identity changed')


def load_measurements(case_manifest_file, receipt_files, *, seq_len, handoff_file, expected_workers=8):
    case_manifest_file, handoff_file = [Path(p).resolve() for p in (case_manifest_file, handoff_file)]
    worker_file = Path(__file__).resolve().parents[2]/'scripts/evaluate_native_checkpoint.py'
    spec = importlib.util.spec_from_file_location('_bound_native_measurement_roster', worker_file)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    manifest, kind, roster, total, pins = worker.case_roster(case_manifest_file,
        seq_len=seq_len, worker_index=0, workers=1)
    handoff = json.loads(handoff_file.read_bytes())
    handoff_sha = digest(handoff_file)
    if handoff.get('status') != 'PASS_COMPLETE_CHECKPOINT_BYTE_SNAPSHOT':
        raise ValueError('measurements require an actual native checkpoint byte handoff')
    merge_pins(pins, {str(p): digest(p) for p in (worker_file, Path(__file__).resolve(), handoff_file)})
    receipts = [Path(p).resolve() for p in receipt_files]
    if len(receipts) != expected_workers or len(set(receipts)) != len(receipts):
        raise ValueError('evaluation requires one distinct receipt for every declared worker')
    by_index, gpu_uuids = {}, set()
    for receipt_path in receipts:
        receipt = json.loads(receipt_path.read_bytes())
        index = receipt.get('worker_index')
        if (receipt.get('status') != 'PASS_NATIVE_CHECKPOINT_MEASUREMENTS'
                or receipt.get('kind') != kind or receipt.get('seq_len') != seq_len
                or receipt.get('workers') != expected_workers or type(index) is not int
                or not 0 <= index < expected_workers or index in by_index
                or receipt.get('case_manifest_file') != str(case_manifest_file)
                or receipt.get('case_manifest_sha256') != digest(case_manifest_file)
                or receipt.get('checkpoint_handoff_file') != str(handoff_file)
                or receipt.get('checkpoint_handoff_sha256') != handoff_sha
                or receipt.get('run_identity_sha256') != handoff['run_identity_sha256']
                or receipt.get('total_case_roster') != total):
            raise ValueError('worker/checkpoint/case roster binding failed')
        uuid = receipt.get('gpu_uuid')
        if (not isinstance(uuid, str) or not uuid or uuid in gpu_uuids
                or receipt.get('cuda_visible_devices') != str(index)):
            raise ValueError('evaluation did not use the declared distinct eight-GPU assignment')
        seconds = receipt.get('wallclock_seconds_including_load')
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('evaluation must retain its actual positive finite measurement time')
        gpu_uuids.add(uuid)
        actual = Path(receipt['measurements'])
        if not actual.is_absolute() or digest(actual) != receipt['measurements_sha256']:
            raise ValueError('actual measurement byte file changed')
        records = [json.loads(line) for line in actual.read_bytes().splitlines()]
        expected = [(entry, case) for offset, (entry, case) in enumerate(roster)
                    if offset % expected_workers == index]
        if len(records) != len(expected) or receipt.get('measured_cases') != len(expected):
            raise ValueError('worker omitted, duplicated or added actual cases')
        for (entry, case), record in zip(expected, records, strict=True):
            validate_record(entry, case, record, kind=kind, seq_len=seq_len)
        source_inputs = receipt.get('source_inputs')
        if not isinstance(source_inputs, dict) or not source_inputs:
            raise ValueError('worker did not bind its actual measurement source')
        if any(source_inputs.get(str(path)) != digest(path) for path in
               (worker_file, worker_file.parent.parent/'src/yuformer_eval/native_inference.py')):
            raise ValueError('worker lacks the actual native model measurement implementation')
        merge_pins(pins, source_inputs)
        merge_pins(pins, {str(receipt_path): digest(receipt_path), str(actual): digest(actual)})
        by_index[index] = records
    if set(by_index) != set(range(expected_workers)):
        raise ValueError('actual evaluation worker roster is incomplete')
    combined = []
    cursors = [0]*expected_workers
    for offset, (entry, case) in enumerate(roster):
        index = offset % expected_workers
        record = by_index[index][cursors[index]]
        cursors[index] += 1
        combined.append((entry, case, record))
    if any(digest(name) != sha for name, sha in pins.items()):
        raise ValueError('case, measurement or worker source changed during independent reconstruction')
    return manifest, kind, combined, pins
