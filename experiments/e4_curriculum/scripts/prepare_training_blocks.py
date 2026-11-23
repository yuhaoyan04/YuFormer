#!/usr/bin/env python3
"""Prepare independent stage mappings while token encoding continues unchanged."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import yaml

import calibrate_training_tokens as calibration
import training_block_permutation as blocks

ROOT = Path(__file__).resolve().parents[3]


def prepare(destination, *, seed, block_samples):
    base_path = ROOT/'pretrain/configs/pretrain_base.yaml'
    release_path = ROOT/'data/processed/releases/pretrain-1t-frozen.release.json'
    base = yaml.safe_load(base_path.read_bytes())
    release = json.loads(release_path.read_bytes())
    if release.get('status') != 'PASS' or set(release.get('stages', {})) != {'warmup', 'stable', 'anneal'}:
        raise ValueError('block preparation requires the complete selected-order release')
    limits = base['data'].get('stage_sample_limits')
    if not isinstance(limits, list) or len(limits) != 3 or not isinstance(seed, str) or not seed:
        raise ValueError('block preparation needs explicit stage budgets and independent seed')
    geometry = {'seq_len': base['data']['seq_len'], 'stage_sample_limits': limits,
                'block_samples': block_samples, 'seed': seed, 'algorithm': blocks.ALGORITHM,
                'release_manifest_sha256': calibration.digest(release_path),
                'dataset_id': release['dataset_id'], 'snapshot_id': release['snapshot_id'],
                'data_frozen_recipe_sha256': release['frozen_recipe_sha256'],
                'generator_tool_sha256': calibration.digest(blocks.__file__),
                'preparation_tool_sha256': calibration.digest(__file__)}
    destination.mkdir(parents=True, exist_ok=True)
    with calibration.RunLease(destination, 'independent-causal-training-index-preparation'):
        plan_path = destination/'mapping_plan.json'
        if plan_path.exists():
            old = json.loads(plan_path.read_bytes())
            if old.get('geometry') != geometry:
                raise ValueError('existing training mapping belongs to a different data/seed/code contract')
            for row in old['stages']:
                if calibration.digest(destination/row['permutation']['file']) != row['permutation']['sha256']:
                    raise ValueError('prepared training mapping bytes changed')
            return old
        rows = []
        for index, (phase, samples) in enumerate(zip(('warmup', 'stable', 'anneal'), limits)):
            print(f'preparing {phase}: {samples} causal samples, blocks={samples//block_samples}', flush=True)
            permutation = blocks.build_permutation(samples, block_samples, seed=seed, phase=phase)
            path = destination/f'{phase}.blocks.u32'
            raw = permutation.tobytes()
            expected = hashlib.sha256(raw).hexdigest()
            if path.exists():
                if calibration.digest(path) != expected:
                    raise ValueError('prepared stage mapping differs from independent seed: '+phase)
            else:
                temporary = path.with_name(path.name+f'.tmp-{os.getpid()}')
                temporary.write_bytes(raw)
                os.replace(temporary, path)
            rows.append({'stage_index': index, 'phase': phase, 'stage_sample_limit': samples,
                'permutation_population_samples': samples, 'source_order': release['stages'][phase]['order'],
                'permutation': {'file': path.name, 'sha256': expected, 'bytes': len(raw), 'dtype': '<u4'}})
        plan = {'schema_version': 1, 'status': 'PREPARED_INDEX_ARRAYS_AWAITING_FULL_STREAM_GATES',
                'purpose': 'INDEPENDENT_CAUSAL_BLOCK_MAPPING_PREPARATION', 'production_qualified': False,
                'geometry': geometry, 'stages': rows,
                'next_gate': 'bind complete source-verified stream metadata and PASS full-byte/consumed-mixture audits before publishing a schema-v2 production manifest'}
        calibration.atomic_json(plan_path, plan)
        calibration.atomic_json(ROOT/'data/health/training-block-preparation.json',
            dict(plan, directory=str(destination), plan_sha256=calibration.digest(plan_path)))
        return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--seed', required=True)
    parser.add_argument('--block-samples', type=int, required=True)
    args = parser.parse_args()
    result = prepare(args.out_dir.resolve(), seed=args.seed, block_samples=args.block_samples)
    print(json.dumps({'status': result['status'], 'stages': len(result['stages']),
                      'array_bytes': sum(row['permutation']['bytes'] for row in result['stages'])}), flush=True)
