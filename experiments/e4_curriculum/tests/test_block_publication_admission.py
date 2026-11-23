import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from publish_training_block_manifest import require_admission, STAGES, calibration


def valid_inputs():
    domains = calibration.DOMAINS
    audit = {'status': 'PASS', 'roster_exact': True, 'train_doc_lengths_audited': True,
             'train_source_rows_verified': True, 'streams': [{'name': name, 'status': 'PASS'}
                 for name in list(STAGES)+['val_'+domain for domain in domains]]}
    targets = {phase: {domain: 1/9 for domain in domains} for phase in STAGES}
    mixtures = {phase: {'status': 'PASS', 'stage': phase, 'configured_target_tokens': 90,
        'mixture': {'status': 'PASS', 'consumed_tokens': 90, 'absolute_tolerance': .002,
            'relative_tolerance': .1, 'domains': {domain: {'pass': True, 'tokens': 10,
                'target_weight': 1/9} for domain in domains}}} for phase in STAGES}
    return audit, mixtures, targets


def check(audit, mixtures, targets):
    require_admission(audit, mixtures, [9, 9, 9], 10, targets)


def test_complete_original_consumption_gates_are_required():
    check(*valid_inputs())


@pytest.mark.parametrize('flag', ['roster_exact', 'train_doc_lengths_audited', 'train_source_rows_verified'])
def test_any_missing_full_data_admission_blocks_publication(flag):
    audit, mixtures, targets = valid_inputs()
    audit[flag] = False
    with pytest.raises(ValueError, match='full-byte/source/ledger'):
        check(audit, mixtures, targets)


def test_changed_consumption_is_recomputed_even_if_every_pass_boolean_is_true():
    audit, mixtures, targets = valid_inputs()
    rows = mixtures['stable']['mixture']['domains']
    rows['web']['tokens'] += 1; rows['code']['tokens'] -= 1
    with pytest.raises(ValueError, match='original frozen domain gate'):
        check(audit, mixtures, targets)


def test_larger_tolerance_cannot_authorize_the_mapping():
    audit, mixtures, targets = valid_inputs()
    mixtures['warmup']['mixture']['absolute_tolerance'] = .02
    with pytest.raises(ValueError, match='exact configured'):
        check(audit, mixtures, targets)


def test_wrong_domain_target_cannot_replace_the_original_frozen_recipe():
    audit, mixtures, targets = valid_inputs()
    mixtures['anneal']['mixture']['domains']['web']['target_weight'] = .2
    with pytest.raises(ValueError, match='original frozen domain gate'):
        check(audit, mixtures, targets)


def test_missing_validation_stream_blocks_the_manifest_even_with_pass_rollup():
    audit, mixtures, targets = valid_inputs()
    audit['streams'].pop()
    with pytest.raises(ValueError, match='roster is not exact'):
        check(audit, mixtures, targets)
