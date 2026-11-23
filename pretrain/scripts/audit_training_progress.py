#!/usr/bin/env python3
"""Read-only checks of actual logged learning rates, losses and checkpoint cursors."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle

import yaml

from collect_torchtitan_result import lineage_series


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def lr_factor(epoch, *, total, warmup, decay, minimum):
    if not 0 <= epoch <= total or not 0 <= warmup <= total or not 0 < decay <= total-warmup:
        raise ValueError('invalid observed scheduler epoch/horizon')
    stable_end = total+1-decay
    if epoch < warmup:
        return (epoch+1)/warmup
    if epoch < stable_end:
        return 1.
    progress = (epoch+1-stable_end)/decay
    return minimum+(1-minimum)*.5*(1+math.cos(math.pi*progress))


def expected_lr(config, epoch):
    total, optim = config['train']['expected_iters'], config['optim']
    warmup = optim['warmup_steps']
    if optim['lr_schedule'] == 'wsd':
        decay = optim['decay_steps']
    elif optim['lr_schedule'] == 'cosine':
        decay = total-warmup
    else:
        raise ValueError('unsupported monitored learning-rate schedule')
    return optim['lr']*lr_factor(epoch, total=total, warmup=warmup, decay=decay,
                                 minimum=optim.get('min_lr_ratio', 0.))


def verify_scalar_series(config, series):
    lr_tags = sorted(tag for tag in series if tag.startswith('lr/'))
    losses = series.get('loss_metrics/global_avg_loss', [])
    if not losses:
        return {'status': 'WAITING_FIRST_TRAINING_METRICS'}
    if not lr_tags:
        raise ValueError('actual optimizer learning-rate metrics are missing')
    for tag, points in series.items():
        if (tag.startswith(('lr/', 'loss_metrics/', 'validation_metrics/domain/'))
                or tag in ('validation_metrics/loss', 'grad_norm')):
            if any(not math.isfinite(point['value']) for point in points):
                raise ValueError('nonfinite observed training metric: '+tag)
    observed = []
    checked_steps = set()
    for tag in lr_tags:
        for point in series[tag]:
            step, actual = int(point['step']), point['value']
            if step < 1 or step > config['train']['expected_iters']:
                raise ValueError('observed LR step exceeds the configured budget')
            expected = expected_lr(config, step-1)
            if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-12):
                raise ValueError(f'actual optimizer LR differs from the qualified curve: {tag} step={step} actual={actual} expected={expected}')
            observed.append({'tag': tag, 'step': step, 'actual': actual, 'expected': expected})
            checked_steps.add(step)
    if not checked_steps:
        raise ValueError('actual optimizer LR series is empty')
    latest = max(point['step'] for point in losses)
    last_lr = max(checked_steps)
    return {'status': 'PASS_OBSERVED_PREFIX' if last_lr >= latest else 'WAITING_EVENT_FLUSH',
        'latest_logged_step': latest, 'latest_lr_step': last_lr, 'lr_groups': len(lr_tags),
        'checked_lr_points': len(observed), 'last_learning_rates': [v for v in observed if v['step'] == last_lr],
        'checkpoint_expected_lr_uses_epoch': 'saved_step',
        'logged_lr_expected_epoch': 'optimizer_update_step_minus_one'}


def checkpoint_scalars(path):
    import torch
    from torch.distributed.checkpoint.metadata import BytesStorageMetadata
    path = Path(path)
    metadata_path = path/'.metadata'
    with metadata_path.open('rb') as handle:
        before = os.fstat(handle.fileno())
        raw = handle.read()
        metadata = pickle.loads(raw)
        stat = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
        if stat(os.fstat(handle.fileno())) != stat(before) or stat(metadata_path.stat()) != stat(before):
            raise ValueError('checkpoint metadata changed while reading')
    names = {'train_state.step', 'train_state.global_tokens_seen', 'train_state.samples_consumed',
             'dataloader.consumed_samples', 'dataloader.seq_len', 'lr_scheduler.last_epoch',
             'dataloader.sampling_policy', 'dataloader.block_permutation_manifest_sha256'}
    names.update(k for k in metadata.state_dict_metadata if k.startswith('optimizer.param_groups.') and k.endswith('.lr'))
    values = {}
    storage = {index.fqn: info for index, info in metadata.storage_data.items()
               if index.fqn in names}
    for name in names & metadata.state_dict_metadata.keys():
        if not isinstance(metadata.state_dict_metadata[name], BytesStorageMetadata):
            raise ValueError('checkpoint scalar unexpectedly uses tensor storage: '+name)
        info = storage[name]
        relative = Path(info.relative_path)
        if relative.is_absolute() or '..' in relative.parts or getattr(info, 'transform_descriptors', None):
            raise ValueError('unsupported checkpoint scalar storage path/transform')
        if not 0 < info.length <= 2*1024*1024 or info.offset < 0:
            raise ValueError('checkpoint scalar storage bounds are invalid')
        shard = path/relative
        with shard.open('rb') as handle:
            before = stat(os.fstat(handle.fileno()))
            handle.seek(info.offset)
            payload = handle.read(info.length)
            if len(payload) != info.length or stat(os.fstat(handle.fileno())) != before or stat(shard.stat()) != before:
                raise ValueError('checkpoint scalar shard changed/truncated while reading')
        values[name] = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    return values, sha(raw)


def verify_checkpoint(config, values, order_identity):
    step = values['train_state.step']
    if type(step) is not int or values['lr_scheduler.last_epoch'] != step:
        raise ValueError('checkpoint scheduler epoch differs from trainer step')
    samples_per_update = config['train']['micro_batch_size']*config['train']['grad_accum']*config['parallel']['world_size']
    expected_samples = step*samples_per_update
    if (values['train_state.samples_consumed'] != expected_samples
            or values['dataloader.consumed_samples'] != expected_samples
            or values['train_state.global_tokens_seen'] != expected_samples*config['data']['seq_len']
            or values['dataloader.seq_len'] != config['data']['seq_len']):
        raise ValueError('checkpoint sample/token cursor disagrees with optimizer steps')
    if config['data'].get('sampling_policy', 'sequential') == 'block_permutation':
        if (values.get('dataloader.sampling_policy') != 'block_permutation'
                or values.get('dataloader.block_permutation_manifest_sha256') != order_identity['manifest_sha256']):
            raise ValueError('checkpoint restored another training sample order')
    actual = [v for key, v in values.items() if key.startswith('optimizer.param_groups.') and key.endswith('.lr')]
    expected = expected_lr(config, step)
    if not actual or any(not isinstance(v, (float, int)) or not math.isfinite(v) or
                         not math.isclose(v, expected, rel_tol=1e-10, abs_tol=1e-14) for v in actual):
        raise ValueError('checkpoint optimizer LR differs from its restored scheduler epoch')
    ends, total = [], 0
    for n in config['data'].get('stage_sample_limits', []):
        total += n
        ends.append(total)
    import bisect
    stage = bisect.bisect_right(ends, expected_samples) if ends else None
    if stage == len(ends):
        stage = None
    return {'status': 'PASS', 'step': step, 'samples_consumed': expected_samples,
        'global_tokens_seen': expected_samples*config['data']['seq_len'],
        'next_logical_stage_index': stage, 'optimizer_groups_checked': len(actual),
        'saved_optimizer_lr': actual[0], 'expected_saved_optimizer_lr': expected}


def verify_reference_schedule(config):
    import torch
    from torchtitan.components.optimizer import LRSchedulersContainer, OptimizersContainer, default_adamw
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer_config = default_adamw(lr=1.)
    optimizer_config.implementation = 'for-loop'
    optimizers = OptimizersContainer(optimizer_config, model_parts=[model])
    optim, total = config['optim'], config['train']['expected_iters']
    scheduler = LRSchedulersContainer.Config(warmup_steps=optim['warmup_steps'], total_steps=total,
        decay_ratio=optim['decay_steps']/total if optim['lr_schedule'] == 'wsd' else None,
        decay_type='cosine', min_lr_factor=optim.get('min_lr_ratio', 0.))
    reference = next(iter(scheduler.build(optimizers=optimizers, training_steps=total))).lr_lambdas[0]
    maximum = max(abs(expected_lr(config, epoch)/optim['lr']-reference(epoch)) for epoch in range(total+1))
    if maximum > 1e-12:
        raise ValueError('independent monitoring curve differs from the pinned actual TorchTitan scheduler')
    import inspect
    path = Path(inspect.getfile(LRSchedulersContainer))
    return {'status': 'PASS', 'scheduler_epochs_checked': total+1,
        'maximum_multiplier_error': maximum, 'scheduler_source_sha256': sha(path.read_bytes())}


def audit(run, *, check_checkpoint=False):
    run = Path(run).resolve()
    config_raw = (run/'resolved_config.yaml').read_bytes()
    config = yaml.safe_load(config_raw)
    identity = json.loads((run/'run_identity.json').read_bytes())
    if sha(config_raw) != identity['resolved_config_sha256']:
        raise ValueError('resolved training config changed since launch')
    bound = dict(identity)
    recorded = bound.pop('identity_sha256')
    if sha(json.dumps(bound, sort_keys=True, separators=(',', ':')).encode()) != recorded:
        raise ValueError('run identity manifest was modified')
    attempt = json.loads((run/'current_attempt.json').read_bytes())
    if attempt['identity_sha256'] != recorded:
        raise ValueError('active attempt belongs to another run identity')
    files, series = lineage_series(run, attempt)
    result = verify_scalar_series(config, series)
    result.update(run_dir=str(run), attempt_id=attempt['attempt_id'], identity_sha256=recorded,
        config_sha256=sha(config_raw), event_files=len(files), observer_tool_sha256=sha(Path(__file__).read_bytes()))
    if check_checkpoint:
        paths = [p for p in (run/'checkpoint').glob('step-*') if p.name.removeprefix('step-').isdigit() and (p/'.metadata').is_file()]
        if paths:
            latest = max(paths, key=lambda p: int(p.name.removeprefix('step-')))
            values, digest = checkpoint_scalars(latest)
            result['checkpoint'] = dict(verify_checkpoint(config, values, identity.get('training_sample_order')),
                path=str(latest), metadata_sha256=digest)
        else:
            result['checkpoint'] = {'status': 'WAITING_FIRST_COMPLETE_CHECKPOINT'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--check-checkpoint', action='store_true')
    parser.add_argument('--check-reference-scheduler', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        result = audit(args.run, check_checkpoint=args.check_checkpoint)
        if args.check_reference_scheduler:
            result['reference_scheduler'] = verify_reference_schedule(
                yaml.safe_load((args.run/'resolved_config.yaml').read_bytes()))
    except Exception as exc:
        result = {'status': 'FAIL', 'run_dir': str(args.run), 'error': repr(exc)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name+f'.tmp-{os.getpid()}')
        temporary.write_text(json.dumps(result, sort_keys=True, indent=2)+'\n')
        os.replace(temporary, args.output)
    print(json.dumps(result, sort_keys=True))
    return 1 if result['status'] == 'FAIL' else 0


if __name__ == '__main__':
    raise SystemExit(main())
