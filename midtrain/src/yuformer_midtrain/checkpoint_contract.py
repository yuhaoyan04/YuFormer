"""Bind stage initialization to the actual checkpoint, model and prior context."""
import hashlib
import io
import json
import pickle
from pathlib import Path

import yaml

from .window_release import bound_json, stat_identity


def checkpoint_scalars(path, metadata_sha256):
    import torch
    from torch.distributed.checkpoint.metadata import BytesStorageMetadata
    path = Path(path)
    raw = (path/'.metadata').read_bytes()
    if hashlib.sha256(raw).hexdigest() != metadata_sha256:
        raise ValueError('parent checkpoint metadata SHA-256 mismatch')
    metadata = pickle.loads(raw)
    names = {'train_state.step', 'train_state.global_tokens_seen', 'train_state.samples_consumed',
             'dataloader.consumed_samples', 'dataloader.seq_len', 'lr_scheduler.last_epoch'}
    if not names <= metadata.state_dict_metadata.keys():
        raise ValueError('stage parent must contain a complete training checkpoint')
    storage = {index.fqn: info for index, info in metadata.storage_data.items() if index.fqn in names}
    result = {}
    for name in names:
        if not isinstance(metadata.state_dict_metadata[name], BytesStorageMetadata):
            raise ValueError('parent checkpoint scalar has an unsupported storage type')
        info = storage[name]
        relative = Path(info.relative_path)
        if (relative.is_absolute() or '..' in relative.parts or info.offset < 0
                or not 0 < info.length <= 2*1024*1024 or getattr(info, 'transform_descriptors', None)):
            raise ValueError('parent checkpoint scalar shard path/bounds/transform is invalid')
        shard = path/relative
        with shard.open('rb') as handle:
            before = stat_identity(shard.stat())
            handle.seek(info.offset)
            payload = handle.read(info.length)
            if len(payload) != info.length or stat_identity(shard.stat()) != before:
                raise ValueError('parent checkpoint scalar shard changed or is truncated')
        result[name] = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    return result


def validate_parent(parent, *, budget, next_seq, scope, model, tokenizer_sha256):
    checkpoint = Path(parent['path']).resolve()
    identity_path = Path(parent['run_identity']).resolve()
    identity = bound_json(identity_path, parent['run_identity_sha256'])
    unsigned = {k: v for k, v in identity.items() if k != 'identity_sha256'}
    if hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(',', ':')).encode()).hexdigest() != identity.get('identity_sha256'):
        raise ValueError('stage parent run identity seal is invalid')
    raw = identity_path.with_name('resolved_config.yaml').read_bytes()
    if hashlib.sha256(raw).hexdigest() != identity.get('resolved_config_sha256'):
        raise ValueError('stage parent resolved configuration changed')
    config = yaml.safe_load(raw)
    previous_seq = 4096 if next_seq == 16384 else 16384
    if config['data']['seq_len'] != previous_seq or identity.get('tokenizer_sha256') != tokenizer_sha256:
        raise ValueError('stage parent has another context or tokenizer')
    model_raw = Path(identity['model_configs']).read_bytes()
    if hashlib.sha256(model_raw).hexdigest() != identity.get('model_configs_sha256'):
        raise ValueError('stage parent model configuration changed')
    previous_model = yaml.safe_load(model_raw)['presets'][config['model']['preset']]
    architectural = lambda x: {k: v for k, v in x.items() if k not in
        {'max_seq_len', 'rope_base', 'rope_theta', 'selective_recompute'}}
    if architectural(previous_model) != architectural(model):
        raise ValueError('weight-only stage transition cannot change the learned architecture')
    step = parent.get('step')
    if (type(step) is not int or step <= 0
            or checkpoint != identity_path.parent/'checkpoint'/f'step-{step}'):
        raise ValueError('stage parent step/path does not belong to its actual run identity')
    if scope == 'PRODUCTION':
        previous_budget = next(s for s in budget['stages'] if s['seq_len'] == previous_seq)
        if step != previous_budget['updates'] or step != config['train']['expected_iters']:
            raise ValueError('production context extension must start from the completed previous stage')
    values = checkpoint_scalars(checkpoint, parent['metadata_sha256'])
    samples = step*config['parallel']['world_size']*config['train']['micro_batch_size']*config['train']['grad_accum']
    if (values['train_state.step'] != step or values['lr_scheduler.last_epoch'] != step
            or values['dataloader.seq_len'] != previous_seq
            or values['train_state.samples_consumed'] != samples
            or values['dataloader.consumed_samples'] != samples
            or values['train_state.global_tokens_seen'] != samples*previous_seq):
        raise ValueError('stage parent actual checkpoint step/cursor/scheduler does not match its declaration')
    return {'step': step, 'seq_len': previous_seq, 'samples': samples, 'path': str(checkpoint)}


def verify_parent_handoff(parent):
    handoff = bound_json(parent['handoff_file'], parent['handoff_sha256'])
    checkpoint = Path(parent['path']).resolve()
    producer = Path(__file__).resolve().parents[2]/'scripts/create_checkpoint_handoff.py'
    if (handoff.get('status') != 'PASS_COMPLETE_CHECKPOINT_BYTE_SNAPSHOT'
            or handoff.get('checkpoint') != str(checkpoint)
            or handoff.get('metadata_sha256') != parent['metadata_sha256']
            or handoff.get('run_identity_sha256') != parent['run_identity_sha256']
            or handoff.get('producer_sha256') != hashlib.sha256(producer.read_bytes()).hexdigest()):
        raise ValueError('stage parent handoff does not bind its actual checkpoint and producer')
    rows = handoff.get('artifacts', [])
    files = {p.name for p in checkpoint.iterdir() if p.is_file()}
    if not rows or {row.get('file') for row in rows} != files or len(rows) != len(files):
        raise ValueError('stage parent handoff file roster changed')
    for row in rows:
        name = row.get('file')
        if (not isinstance(name, str) or Path(name).name != name
                or stat_identity((checkpoint/name).stat()) != row.get('stat')
                or not isinstance(row.get('sha256'), str) or len(row['sha256']) != 64):
            raise ValueError('stage parent shard changed since its complete byte snapshot')
    for name, expected in handoff.get('source_inputs', {}).items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError('stage parent handoff verifier/producer source changed')
    return handoff
