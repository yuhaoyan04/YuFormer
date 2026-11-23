import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml

PROTOTYPE = Path(__file__).resolve().parents[1]/'prototypes/pretrain_scripts'
HELPER_SPEC = importlib.util.spec_from_file_location('_candidate_production_dependencies',
    PROTOTYPE/'production_dependency_identity.py')
DEPENDENCIES = importlib.util.module_from_spec(HELPER_SPEC)
HELPER_SPEC.loader.exec_module(DEPENDENCIES)
production_asset_paths = DEPENDENCIES.production_asset_paths

SPEC = importlib.util.spec_from_file_location('_candidate_qualification_identity', PROTOTYPE/'qualification_identity.py')
QUALIFICATION = importlib.util.module_from_spec(SPEC)
_previous = sys.modules.get('production_dependency_identity')
sys.modules['production_dependency_identity'] = DEPENDENCIES
try:
    SPEC.loader.exec_module(QUALIFICATION)
finally:
    if _previous is None:
        sys.modules.pop('production_dependency_identity', None)
    else:
        sys.modules['production_dependency_identity'] = _previous


def build_assets(tmp_path, block=False):
    folder = tmp_path/'pretrain/configs'
    folder.mkdir(parents=True)
    model = tmp_path/'production.model.yaml'
    model.write_text('num_layers: 16\n')
    config = {'model': {'configs': '${YUFORMER_MODEL_CONFIGS}'}, 'data': {}}
    if block:
        config['data'] = {'sampling_policy': 'block_permutation',
                          'block_permutation_manifest': '${YUFORMER_BLOCK_ORDER}'}
    (folder/'pretrain_base.yaml').write_text(yaml.safe_dump(config))
    (folder/'pretrain_base.env').write_text(f'YUFORMER_MODEL_CONFIGS={model}\nYUFORMER_BLOCK_ORDER={tmp_path}/order.json\n')
    (tmp_path/'order.json').write_text(json.dumps({'stages': [{'permutation': {'file': 'blocks.u32'}}]}))
    (tmp_path/'blocks.u32').write_bytes(bytes(16))
    return model


def test_actual_model_change_invalidates_qualification_even_if_env_path_is_unchanged(tmp_path):
    model = build_assets(tmp_path)
    before = QUALIFICATION.qualification_identity(tmp_path)
    model.write_text('num_layers: 18\n')
    after = QUALIFICATION.qualification_identity(tmp_path)
    assert before['sha256'] != after['sha256']
    assert any(row['path'] == 'production.model.yaml' for row in after['files'])


def test_changed_block_artifact_invalidates_qualification(tmp_path):
    build_assets(tmp_path, block=True)
    before = QUALIFICATION.qualification_identity(tmp_path)
    (tmp_path/'blocks.u32').write_bytes(bytes([1])*16)
    assert before['sha256'] != QUALIFICATION.qualification_identity(tmp_path)['sha256']


def test_unresolved_production_model_path_is_rejected(tmp_path):
    build_assets(tmp_path)
    (tmp_path/'pretrain/configs/pretrain_base.env').write_text('OTHER=value\n')
    with pytest.raises(ValueError, match='unresolved production dependency'):
        production_asset_paths(tmp_path)
