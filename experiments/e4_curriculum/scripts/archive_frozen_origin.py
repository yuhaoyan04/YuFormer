#!/usr/bin/env python3
"""Preserve the complete byte-bound origin before any training-only adoption."""
import hashlib
import json
import os
from pathlib import Path

import calibrate_training_tokens as calibration

ROOT = Path(__file__).resolve().parents[3]


def archive():
    frozen = ROOT / 'experiments/e4_curriculum/results/frozen_1t_recipe.json'
    frozen_sha = calibration.digest(frozen)
    recipe = json.loads(frozen.read_bytes())
    if recipe.get('status') != 'PASS':
        raise ValueError('only an existing PASS frozen origin can be archived')
    directory = ROOT / f'data/processed/recipe_origins/{frozen_sha}'
    objects = directory / 'objects'
    objects.mkdir(parents=True, exist_ok=True)
    bindings = dict(recipe['inputs'])
    for name, expected in recipe['outputs'].items():
        if name in bindings and bindings[name] != expected:
            raise ValueError('conflicting original input/output identity')
        bindings[name] = expected
    bindings[str(frozen.relative_to(ROOT))] = frozen_sha
    for name in ('data/processed/releases/pretrain-1t-frozen.release.json',
                 'data/health/torchtitan-qualification.json',
                 'data/health/production-recovery-qualification.json',
                 'data/health/production-throughput-canary.json'):
        bindings[name] = calibration.digest(ROOT / name)
    records = {}
    for name, expected in sorted(bindings.items()):
        source = ROOT / name
        destination = objects / expected
        with source.open('rb') as handle:
            before = calibration.materializer._stat_identity(os.fstat(handle.fileno()))
            temporary = destination.with_name(destination.name+f'.tmp-{os.getpid()}')
            digest = hashlib.sha256()
            with temporary.open('wb') as output:
                while value := handle.read(8 << 20):
                    output.write(value)
                    digest.update(value)
            if (digest.hexdigest() != expected or
                    calibration.materializer._stat_identity(os.fstat(handle.fileno())) != before or
                    calibration.materializer.source_file_identity(source) != before):
                temporary.unlink()
                raise ValueError('frozen origin source does not match its byte identity: ' + name)
        if destination.exists():
            if calibration.digest(destination) != expected:
                raise ValueError('archived content address is corrupt')
            temporary.unlink()
        else:
            os.replace(temporary, destination)
        if calibration.digest(destination) != expected:
            raise ValueError('archived frozen origin copy SHA mismatch')
        records[name] = {'sha256': expected, 'bytes': before['bytes'],
                         'archive': str(destination.relative_to(ROOT)), 'original_stat': before}
    report = {'schema_version': 1, 'status': 'PASS_IMMUTABLE_ORIGIN_ARCHIVE',
              'purpose': 'PRESERVE_ORIGINAL_RECIPE_BYTES_NOT_PRODUCTION_ADOPTION',
              'frozen_recipe_sha256': frozen_sha, 'files': records,
              'archive_tool_sha256': calibration.digest(__file__),
              'large_token_streams_and_orders': 'preserved in place; not copied or rebound'}
    manifest = directory / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_bytes()) != report:
        raise ValueError('immutable origin archive manifest differs')
    calibration.atomic_json(manifest, report)
    print(json.dumps({'status': report['status'], 'files': len(records),
                      'manifest': str(manifest.relative_to(ROOT)),
                      'sha256': calibration.digest(manifest)}), flush=True)


if __name__ == '__main__':
    archive()
