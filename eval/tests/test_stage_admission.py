import hashlib
import json

import pytest

from yuformer_eval.stage_admission import complete_checkpoint, create_admission, verify_admission


def test_case_asset_admission_cannot_substitute_for_a_completed_native_checkpoint(tmp_path):
    path = tmp_path/'case-audit.json'
    path.write_text(json.dumps({'status': 'PASS_IMMUTABLE_PAIRED_NLL_CASE_MANIFEST'}))
    binding = {'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match='actual complete production checkpoint'):
        complete_checkpoint(binding, {}, seq_len=16384,
                            budget={'stages': [{'seq_len': 16384, 'updates': 7152, 'tokens': 29997662208}]})


def test_partial_suites_are_rejected_before_any_model_capability_is_scored(tmp_path):
    path = tmp_path/'spec.json'
    path.write_text(json.dumps({'seq_len': 16384, 'baseline': {'receipts': {'ruler': []}},
                                'candidate': {'receipts': {'ruler': []}}}))
    with pytest.raises(ValueError, match='all four suites'):
        create_admission(path)


def test_fail_or_handwritten_pass_cannot_bypass_recomputation(tmp_path):
    path = tmp_path/'report.json'
    path.write_text(json.dumps({'status': 'FAIL_STAGE_LONG_AND_SHORT_EVALUATION'}))
    with pytest.raises(ValueError, match='has not passed'):
        verify_admission(path)
    path.write_text(json.dumps({'status': 'PASS_STAGE_LONG_AND_SHORT_EVALUATION',
                                'spec_file': str(tmp_path/'absent-actual-evidence.json')}))
    with pytest.raises(FileNotFoundError):
        verify_admission(path)
