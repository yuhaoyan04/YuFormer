"""Read checkpoint weights and evaluate a predeclared disjoint held-out window."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'pretrain/backends/torchtitan'), str(ROOT/'model/src'),
               str(ROOT/'pretrain/scripts'), str(ROOT/'experiments/e4_curriculum/scripts')]
from frozen_training_evidence import publish_checked
from freeze_1t_recipe import source_sha256
from qualification_identity import qualification_identity


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint_identity(folder):
    return {p.name: dict(bytes=p.stat().st_size, inode=p.stat().st_ino,
                         device=p.stat().st_dev, mtime_ns=p.stat().st_mtime_ns)
            for p in sorted(folder.iterdir()) if p.is_file()}


def file_identity(path):
    value = path.stat()
    return dict(bytes=value.st_size, device=value.st_dev, inode=value.st_ino,
                mtime_ns=value.st_mtime_ns)


def evaluate_case(case, plan):
    from torchtitan.tools.logging import init_logger
    from torchtitan.observability import structured_logger as sl
    from yuformer_ext.config_registry import yuformer_from_yaml

    init_logger()
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    protocol = json.loads((ROOT/plan['protocol']).read_bytes())
    if world != 8 or case not in protocol['cases']:
        raise ValueError('held-out review requires the exact declared eight-rank case roster')
    for name, expected in plan['source_inputs'].items():
        if digest(ROOT/name) != expected:
            raise ValueError('held-out review input changed: '+name)
    if any(file_identity(ROOT/row['file']) != row['stat'] for row in plan['validation_streams'].values()):
        raise ValueError('fully audited validation stream changed before evaluation')
    if source_sha256(ROOT) != protocol['training_source_sha256']:
        raise ValueError('held-out review training code differs from the actual checkpoints')
    record = plan['cases'][case]
    checkpoint = ROOT/record['checkpoint']
    before = checkpoint_identity(checkpoint)
    if (before != record['checkpoint_stat']
            or digest(checkpoint/'.metadata') != record['checkpoint_metadata_sha256']):
        raise ValueError('actual step-200 checkpoint changed before evaluation')
    identity = qualification_identity(ROOT)
    config_path = ROOT/record['config']
    os.environ['YUFORMER_CONFIG'] = str(config_path)
    config = yuformer_from_yaml()
    config.checkpoint.initial_load_path = str(checkpoint)
    config.checkpoint.initial_load_model_only = True
    config.checkpoint.folder = 'unused-readonly-evaluation-checkpoint'
    config.checkpoint.async_mode = 'disabled'
    sl.init_structured_logger(source='disjoint_heldout_review',
                              output_dir=config.dump_folder, enable=False)
    trainer = None
    try:
        trainer = config.build()
        if not trainer.checkpointer.load(step=-1):
            raise ValueError('actual checkpoint model weights were not loaded')
        if trainer.step != 0 or trainer.dataloader.state_dict()['consumed_samples'] != 0:
            raise ValueError('read-only review must not restore or advance training state')
        for model in trainer.model_parts:
            model.eval()
        primary = json.loads((ROOT/record['primary_result']).read_bytes())
        results = {}
        batch = protocol['local_batch_size']
        seq = protocol['seq_len']
        val = ROOT/'data/token_streams/m3_20261020_yf64k'
        started = time.monotonic()

        @torch.no_grad()
        def batch_loss(tokens, first):
            start = first+rank*batch
            windows = np.stack([np.asarray(tokens[s*seq:s*seq+seq+1], dtype=np.int64)
                                for s in range(start, start+batch)])
            x = torch.from_numpy(windows[:, :-1].copy()).to(trainer.device)
            y = torch.from_numpy(windows[:, 1:].copy()).to(trainer.device)
            inputs, labels, extra = trainer.post_dataloading_process({'input':x}, y)
            with trainer.train_context():
                prediction = trainer.model_parts[0](inputs, **extra)
                loss, _ = trainer.loss_fn(prediction, labels)
            total = loss.detach().to(torch.float64)
            dist.all_reduce(total)
            if not bool(torch.isfinite(total)):
                raise ValueError('nonfinite held-out loss')
            return float(total.item()/(world*batch*seq))

        for domain in protocol['domains']:
            tokens = np.memmap(val/f'val_{domain}.tokens.bin', mode='r', dtype='<u2')
            stop = protocol['source_sample_offset']+protocol['samples_per_domain']
            if stop*seq+1 > len(tokens):
                raise ValueError('held-out review would wrap or repeat validation samples')
            original = batch_loss(tokens, 0)
            expected = primary['metrics'][f'validation_metrics/domain/{domain}/loss']['last']
            if abs(original-expected) > 1e-5:
                raise ValueError(f'original 64-sample metric did not reproduce: {domain}: {original} vs {expected}')
            values = [batch_loss(tokens, protocol['source_sample_offset']+i*world*batch)
                      for i in range(protocol['global_batches_per_domain'])]
            results[domain] = dict(primary_reproduced_loss=original,
                                   original_primary_loss=expected,
                                   batch_mean_losses=values, mean_loss=float(np.mean(values)),
                                   samples=protocol['samples_per_domain'],
                                   target_tokens=protocol['samples_per_domain']*seq)
            if rank == 0:
                print(json.dumps(dict(case=case, domain=domain, **results[domain])), flush=True)
            del tokens
        if (checkpoint_identity(checkpoint) != before
                or digest(checkpoint/'.metadata') != record['checkpoint_metadata_sha256']
                or qualification_identity(ROOT) != identity
                or any(file_identity(ROOT/row['file']) != row['stat'] for row in plan['validation_streams'].values())
                or any(digest(ROOT/name) != expected for name,expected in plan['source_inputs'].items())):
            raise ValueError('held-out review source, checkpoint, or data identity changed')
        if trainer.step != 0 or any(p.grad is not None for model in trainer.model_parts for p in model.parameters()):
            raise ValueError('held-out review unexpectedly advanced or differentiated the model')
        if rank == 0:
            report = dict(status='PASS_READONLY_DISJOINT_VALIDATION', case=case,
                          plan_sha256=plan['_validated_plan_sha256'],
                          protocol_sha256=digest(ROOT/plan['protocol']),
                          training_source_sha256=protocol['training_source_sha256'],
                          checkpoint_metadata_sha256=record['checkpoint_metadata_sha256'],
                          source_qualification_identity=identity, source_inputs=plan['source_inputs'],
                          domains=results, aggregate_mean_loss=float(np.mean([v['mean_loss'] for v in results.values()])),
                          training_steps_executed=0, checkpoint_modified=False,
                          elapsed_seconds=time.monotonic()-started, production_qualified=False)
            publish_checked(ROOT/record['report'], (json.dumps(report, sort_keys=True, indent=2)+'\n').encode())
    finally:
        if trainer is not None:
            trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    if digest(args.plan) != args.plan_sha256:
        raise ValueError('actual evaluation plan changed before launch')
    plan = json.loads(args.plan.read_bytes())
    plan['_validated_plan_sha256'] = args.plan_sha256
    evaluate_case(args.case, plan)
