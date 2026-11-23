"""Admit only the actual current-backend DCP persistence qualification."""
import hashlib
import json
from pathlib import Path

from freeze_1t_recipe import source_sha256

REPORT = 'data/health/checkpoint-directory-durability-qualification.json'
SOURCE_PATHS = {'pretrain/backends/torchtitan/yuformer_ext/checkpoint.py',
                'pretrain/scripts/qualify_checkpoint_durability.py'}
EXPECTED_CASES = {'actual_dcp_roundtrip_and_publication_order', 'pre_metadata_directory_fsync_error',
                  'post_metadata_directory_fsync_error', 'file_fsync_cannot_be_disabled',
                  'two_rank:disabled', 'two_rank:async', 'two_rank:async_with_pinned_mem'}


def require_checkpoint_directory_durability(root):
    root = Path(root).resolve()
    path = root/REPORT
    if not path.is_file():
        raise ValueError('actual current-backend checkpoint directory persistence qualification is missing')
    raw = path.read_bytes()
    report = json.loads(raw)
    cases = report.get('tests', [])
    inputs = report.get('source_inputs', {})
    if (report.get('status') != 'PASS_CHECKPOINT_DIRECTORY_DURABILITY' or len(cases) != 7
            or {row.get('name') for row in cases} != EXPECTED_CASES
            or any(row.get('status') != 'PASS' for row in cases)
            or report.get('world_size') != 2 or report.get('device') != 'cpu'
            or set(inputs) != SOURCE_PATHS):
        raise ValueError('checkpoint directory persistence needs all seven actual current-backend cases')
    for name, expected in inputs.items():
        source = root/name
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError('checkpoint persistence qualification producer/backend changed: '+name)
    if report.get('training_source_sha256') != source_sha256(root):
        raise ValueError('checkpoint persistence proof belongs to another training source')
    helper = Path(__file__)
    return {**inputs, REPORT:hashlib.sha256(raw).hexdigest(),
            str(helper.relative_to(root)):hashlib.sha256(helper.read_bytes()).hexdigest()}
