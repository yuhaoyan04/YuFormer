import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from checkpoint_durability_evidence import require_checkpoint_directory_durability, EXPECTED_CASES, SOURCE_PATHS, REPORT


def test_missing_actual_current_backend_proof_blocks_adoption(tmp_path):
    with pytest.raises(ValueError, match='qualification is missing'):
        require_checkpoint_directory_durability(tmp_path)


def test_seven_pass_labels_without_exact_producer_code_are_rejected(tmp_path):
    report = dict(status='PASS_CHECKPOINT_DIRECTORY_DURABILITY', world_size=2, device='cpu',
        tests=[dict(name=name,status='PASS') for name in EXPECTED_CASES],
        source_inputs={name:'wrong-sha' for name in SOURCE_PATHS})
    for name in SOURCE_PATHS:
        path = tmp_path/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('different backend or producer')
    path = tmp_path/REPORT
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='producer/backend changed'):
        require_checkpoint_directory_durability(tmp_path)
