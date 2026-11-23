#!/usr/bin/env python3
"""CPU audit of actual final production checkpoint counters and validation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'eval/src'), str(ROOT/'midtrain/src')]

import yaml
from yuformer_eval.completed_run import validate_completion
from yuformer_eval.measurements import digest
from yuformer_eval.source_binding import actual_training_source


def audit(handoff_file, budget_file, *, budget_sha256, seq_len, output):
    handoff_file, budget_file, output = [Path(p).resolve() for p in (handoff_file, budget_file, output)]
    if output.exists() or digest(budget_file) != budget_sha256:
        raise ValueError('use a new audit output and the actual frozen total production budget')
    budget = json.loads(budget_file.read_bytes())
    stages = [s for s in budget['stages'] if s['seq_len'] == seq_len]
    if len(stages) != 1:
        raise ValueError('explicit production stage is absent or duplicated')
    handoff = json.loads(handoff_file.read_bytes())
    checkpoint = Path(handoff['checkpoint'])
    run = checkpoint.parent.parent
    paths = {name: run/(name+'.json') for name in
             ('run_identity', 'training_complete', 'result', 'training_accounting', 'current_attempt')}
    if not all(p.is_file() for p in paths.values()):
        raise ValueError('production run has not published all complete final artifacts')
    objects = {key: json.loads(path.read_bytes()) for key, path in paths.items()}
    identity = objects['run_identity']
    unsigned = {k: v for k, v in identity.items() if k != 'identity_sha256'}
    if (hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            != identity['identity_sha256'] or digest(paths['run_identity']) != handoff['run_identity_sha256']):
        raise ValueError('actual final checkpoint identity seal changed')
    config_path = run/'resolved_config.yaml'
    if digest(config_path) != identity['resolved_config_sha256']:
        raise ValueError('actual final checkpoint configuration changed')
    config = yaml.safe_load(config_path.read_bytes())
    completed = validate_completion(identity, config, objects['training_complete'], objects['result'],
        objects['training_accounting'], objects['current_attempt'], seq_len=seq_len, stage=stages[0])
    if checkpoint.name != 'step-'+str(completed['step']):
        raise ValueError('handoff does not identify the actual completed final update')
    pins = actual_training_source(ROOT, identity['source_sha256'])
    from yuformer_midtrain.checkpoint_contract import checkpoint_scalars, verify_parent_handoff
    verify_parent_handoff({'path': str(checkpoint), 'metadata_sha256': handoff['metadata_sha256'],
        'run_identity_sha256': handoff['run_identity_sha256'], 'handoff_file': str(handoff_file),
        'handoff_sha256': digest(handoff_file)})
    scalars = checkpoint_scalars(checkpoint, handoff['metadata_sha256'])
    expected = {'train_state.step': completed['step'], 'lr_scheduler.last_epoch': completed['step'],
        'train_state.global_tokens_seen': completed['tokens'], 'train_state.samples_consumed': completed['samples'],
        'dataloader.consumed_samples': completed['samples'], 'dataloader.seq_len': seq_len}
    if scalars != expected:
        raise ValueError('actual final native checkpoint counter/scheduler/cursor mismatch')
    for path in [*paths.values(), config_path, handoff_file, budget_file, Path(__file__).resolve(),
                 ROOT/'eval/src/yuformer_eval/completed_run.py', ROOT/'eval/src/yuformer_eval/measurements.py',
                 ROOT/'midtrain/src/yuformer_midtrain/checkpoint_contract.py',
                 ROOT/'midtrain/src/yuformer_midtrain/window_release.py', checkpoint/'.metadata']:
        pins[str(path)] = digest(path)
    if any(digest(path) != sha for path, sha in pins.items()):
        raise ValueError('final production checkpoint audit source changed')
    report = {'status': 'PASS_ACTUAL_COMPLETE_PRODUCTION_CHECKPOINT_FINAL_VALIDATION_AND_NATIVE_COUNTERS',
        'scope': 'FINAL_COMPLETENESS_IDENTITY_NINE_DOMAIN_VALIDATION_AND_NATIVE_SCALARS_NOT_CAPABILITY',
        **completed, 'checkpoint': str(checkpoint), 'checkpoint_handoff_file': str(handoff_file),
        'checkpoint_handoff_sha256': digest(handoff_file), 'run_identity_sha256': handoff['run_identity_sha256'],
        'internal_run_identity_sha256': identity['identity_sha256'], 'tokenizer_sha256': identity['tokenizer_sha256'],
        'native_scalars': scalars, 'checkpoint_artifacts': handoff['artifacts'],
        'source_inputs': pins, 'producer': str(Path(__file__).resolve()), 'producer_sha256': digest(__file__)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        handle.write((json.dumps(report, sort_keys=True, indent=2)+'\n').encode())
        handle.flush()
        os.fsync(handle.fileno())
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('handoff', 'budget', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--budget-sha256', required=True)
    parser.add_argument('--seq-len', type=int, choices=[4096, 16384, 65536], required=True)
    args = parser.parse_args()
    report = audit(args.handoff, args.budget, budget_sha256=args.budget_sha256,
                   seq_len=args.seq_len, output=args.output)
    print(json.dumps({key: report[key] for key in ('status', 'scope', 'step', 'seq_len')}))
