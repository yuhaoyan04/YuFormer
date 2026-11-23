import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_budget_lr_controls import require_matched_contract


def config():
    return {'run_name': 'control', 'model': {'configs': '/same.model', 'preset': 'candidate'},
            'optim': {'lr': .0002, 'warmup_steps': 10}, 'seed': 19,
            'data': {'stages': ['stream'], 'stage_sample_limits': [1024]},
            'train': {'micro_batch_size': 8, 'grad_accum': 16}}


def test_peak_lr_and_isolated_copy_names_are_the_only_allowed_differences():
    control = config()
    candidate = copy.deepcopy(control)
    candidate['optim']['lr'] = .0003
    candidate['run_name'] = 'independent_control'
    candidate['model']['configs'] = '/identical-bytes-isolated-copy.model'
    require_matched_contract(candidate, control)


@pytest.mark.parametrize('group,key,value', [('optim', 'warmup_steps', 20),
    ('data', 'stages', ['different']), ('data', 'stage_sample_limits', [2048]),
    ('train', 'micro_batch_size', 4), ('train', 'grad_accum', 32), ('model', 'preset', 'other')])
def test_qualifying_difference_cannot_hide_as_lr_control(group, key, value):
    control = config()
    candidate = copy.deepcopy(control)
    candidate[group][key] = value
    with pytest.raises(ValueError, match='other than peak learning rate'):
        require_matched_contract(candidate, control)
