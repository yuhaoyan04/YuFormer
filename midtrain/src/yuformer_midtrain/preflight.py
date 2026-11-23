"""Seal long-stage assets and code before any distributed CUDA allocation."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import yaml

from .registry import validate_midtrain_contract
from .window_release import bound_json, stat_identity, edge_sha256


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seal_bytes(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError('run directory is bound to another stage/config/source: '+str(path))
        return
    with path.open('xb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def preflight(config_path, *, publish=False):
    root = Path(os.environ['YUFORMER_REPO_ROOT']).resolve()
    backend = Path(os.environ['YUFORMER_SHARED_BACKEND_ROOT']).resolve()
    if not backend.is_absolute() or not (backend/'yuformer_ext/config_registry.py').is_file():
        raise ValueError('declare the actual shared TorchTitan backend directory')
    shared = import_path('_midtrain_shared_contract', backend/'validate_contract.py')
    raw = os.path.expandvars(Path(config_path).read_text())
    if '${' in raw:
        raise ValueError('unresolved long-stage environment variable')
    config = yaml.load(raw, Loader=shared.ContractLoader)
    projected = {k: v for k, v in config.items() if k != 'midtrain'}
    shared._validate_config_fields(projected)
    if (config['backend'] != 'torchtitan-v0.3.0' or not Path(config['out_dir']).is_absolute()
            or config.get('experiment_status') in {'superseded', 'invalidated', 'probe_only'}):
        raise ValueError('long-stage backend/output/status contract is invalid')
    geometry, gate, records = validate_midtrain_contract(config)
    stage = config['midtrain']
    # The active allocation is independently reproduced from the nominal user
    # decision and measured runtime proposal, rather than trusting a budget label.
    contract_root = Path(os.environ['YUFORMER_BUDGET_CONTRACT_ROOT']).resolve()
    import sys
    sys.path.insert(0, str(contract_root))
    from total_budget_contract import load_total_budget_contract
    plan = yaml.safe_load((root/'experiments/e4_curriculum/configs/e4_1t_wsd_context.yaml').read_bytes())
    bundle = load_total_budget_contract(root, plan)
    active = root/plan['global_training_budget']['effective_file']
    if Path(stage['approved_budget_file']).resolve() != active or digest(active) != stage['approved_budget_sha256']:
        raise ValueError('long stage does not bind the actual effective whole-chain allocation')
    release = bound_json(stage['window_release'], stage['window_release_sha256'])
    sources = {str((root/name).resolve()): sha for name, sha in bundle['inputs'].items()}
    source_files = [Path(config_path), Path(config['model']['configs']), Path(config['data']['tokenizer_gate']),
                    Path(stage['window_release']), Path(stage['initial_checkpoint']['handoff_file']),
                    root/'experiments/e4_curriculum/configs/e4_1t_wsd_context.yaml']
    for folder in (root/'midtrain/src', root/'midtrain/scripts'):
        source_files += [p for p in folder.rglob('*') if p.suffix in {'.py', '.sh'} and '__pycache__' not in p.parts]
    sources.update({str(p.resolve()): digest(p) for p in source_files})
    validation = []
    audit_path = Path(config['data']['token_stream_audit'])
    audit = json.loads(audit_path.read_bytes())
    if audit.get('status') != 'PASS' or audit.get('roster_exact') is not True:
        raise ValueError('long-stage validation requires the complete physical byte audit')
    by_name = {row['name']: row for row in audit['streams']}
    val_dir = Path(config['data']['val_dir'])
    domains = {'web','code','math','chinese','academic','books','encyclopedia','qa','multilingual'}
    if {p.name for p in val_dir.glob('val_*.meta.json')} != {'val_'+d+'.meta.json' for d in domains}:
        raise ValueError('long-stage validation must retain all nine heldout domains')
    for domain in sorted(domains):
        meta_path = val_dir/f'val_{domain}.meta.json'
        binary = val_dir/f'val_{domain}.tokens.bin'
        meta = json.loads(meta_path.read_bytes())
        row = by_name['val_'+domain]
        expected = {k: row['actual_'+k] for k in ('bytes','device','inode','mtime_ns')}
        with binary.open('rb') as handle:
            if (stat_identity(os.fstat(handle.fileno())) != expected
                    or edge_sha256(handle) != meta['edge_sha256']
                    or meta['tokenizer_sha256'] != gate['tokenizer_sha256']
                    or config['data']['val_samples_per_domain'] > (meta['total_tokens']-1)//geometry['seq_len']):
                raise ValueError('heldout validation byte/tokenizer/capacity binding failed: '+domain)
        sources[str(meta_path)] = digest(meta_path)
        validation.append({'binary':str(binary),'meta':str(meta_path), 'meta_sha256':digest(meta_path),
                           'edge_sha256':meta['edge_sha256']})
    tokenizer = Path(gate['tokenizer_path'])
    if not tokenizer.is_absolute():
        tokenizer = root/tokenizer
    if digest(tokenizer) != gate['tokenizer_sha256']:
        raise ValueError('long-stage tokenizer bytes changed')
    sources[str(tokenizer)] = gate['tokenizer_sha256']
    sources[str(audit_path)] = digest(audit_path)
    identity = {'schema_version':1, 'runtime_state_schema_version':2,
        'run_name':config['run_name'], 'backend':config['backend'],
        'resolved_config_sha256':hashlib.sha256(raw.encode()).hexdigest(),
        'source_sha256':shared._source_sha256(root), 'stage_source_inputs':sources,
        'model_configs':str(Path(config['model']['configs']).resolve()),
        'model_configs_sha256':digest(config['model']['configs']),
        'tokenizer_gate':str(Path(config['data']['tokenizer_gate']).resolve()),
        'tokenizer_gate_sha256':digest(config['data']['tokenizer_gate']),
        'tokenizer_id':gate['tokenizer_id'], 'tokenizer_sha256':gate['tokenizer_sha256'],
        'tokenizer_path':str(tokenizer), 'tokenizer_artifact_sha256':gate['tokenizer_sha256'],
        'release_manifest':stage['window_release'], 'release_manifest_sha256':stage['window_release_sha256'],
        'dataset_id':release['dataset_id'], 'snapshot_id':release['dataset_id'],
        'token_stream_audit':str(audit_path), 'token_stream_audit_sha256':digest(audit_path),
        'streams':release['sources'], 'validation_streams':validation,
        'training_sample_order':{'algorithm':'independent-source-index-s-plus-one-v1',
            'scope':geometry['scope'], 'samples':geometry['samples'],
            'record_index_sha256':release['record_index']['sha256']}}
    identity['identity_sha256'] = hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if publish:
        out = Path(config['out_dir'])
        seal_bytes(out/'resolved_config.yaml',raw.encode())
        seal_bytes(out/'run_identity.json',(json.dumps(identity,sort_keys=True,indent=2)+'\n').encode())
    return {'status':'PASS_MIDTRAIN_PREFLIGHT','out_dir':config['out_dir'],'identity':identity}
