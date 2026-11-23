import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

source = Path(__file__).resolve().parents[1]/'scripts/research_control_identity.py'
spec = importlib.util.spec_from_file_location('research_control_identity_test', source)
identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)


def test_actual_training_dependencies_are_bound_while_unrelated_production_publishes(tmp_path, monkeypatch):
    root = tmp_path
    configs = root/'experiments/e2_architecture/configs/test'
    configs.mkdir(parents=True)
    corpus = root/'data/research'
    corpus.mkdir(parents=True)
    production = root/'data/token_streams/pretrain-1t/anneal.meta.json'
    production.parent.mkdir(parents=True)
    production.write_text('pending')
    core = root/'model.py'
    core.write_text('original core')
    for name in ('model.yaml','release.json','audit.json','tokenizer.json','train.meta.json'):
        (corpus/name).write_text('{}')
    (corpus/'gate.json').write_text(json.dumps({'tokenizer_path':str(corpus/'tokenizer.json')}))
    domains = ('web','code','math','chinese','academic','books','encyclopedia','qa','multilingual')
    for domain in domains:
        (corpus/f'val_{domain}.meta.json').write_text('{}')
    data = {key:str(corpus/name) for key,name in (
        ('release_manifest','release.json'),('token_stream_audit','audit.json'),('tokenizer_gate','gate.json'))}
    data.update(meta=[str(corpus/'train.meta.json')], val_dir=str(corpus))
    config = configs/'case.yaml'
    config.write_text(yaml.safe_dump({'model':{'configs':str(corpus/'model.yaml')},'data':data}))
    (configs/'case.env').write_text('YUFORMER_CONFIG='+str(config)+'\n')

    def qualification(_):
        return {'files':[{'path':str(p.relative_to(root)), 'bytes':p.stat().st_size,
                          'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in (core,production)]}

    monkeypatch.setattr(identity, 'qualification_identity', qualification)
    monkeypatch.setattr(identity, 'production_asset_paths', lambda _: {production})
    fingerprint = lambda: identity.research_control_identity(root, configs)['sha256']
    initial = fingerprint()
    production.write_text('actually published production bytes')
    assert fingerprint() == initial
    for dependency in (core, corpus/'train.meta.json',corpus/'val_chinese.meta.json',corpus/'tokenizer.json',config):
        before = dependency.read_bytes()
        dependency.write_bytes(before+b'\n')
        assert fingerprint() != initial
        dependency.write_bytes(before)
    (corpus/'val_qa.meta.json').unlink()
    with pytest.raises(ValueError, match='dependency is missing'):
        fingerprint()
