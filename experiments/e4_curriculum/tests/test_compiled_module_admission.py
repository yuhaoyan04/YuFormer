import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from adopt_training_recipe import require_compiled_module_cases


def row():
    return {'status':'PASS', 'cases':[{'layer':layer,'batch_size':batch,'seq_len':length,'status':'PASS'}
            for layer,batch,length in ((0,2,129),(5,2,129),(0,8,4096),(5,8,4096))]}


def test_actual_four_shape_roster_is_admissible():
    require_compiled_module_cases(row())


@pytest.mark.parametrize('change', ['old_three_shapes','duplicate_shape','failed_shape'])
def test_missing_global_4k_or_failed_numerics_cannot_be_admitted(change):
    value=copy.deepcopy(row())
    if change=='old_three_shapes':value['cases'].pop()
    elif change=='duplicate_shape':value['cases'][-1]=copy.deepcopy(value['cases'][0])
    else:value['cases'][-1]['status']='FAIL'
    with pytest.raises(ValueError, match='global 4K B8'):
        require_compiled_module_cases(value)
