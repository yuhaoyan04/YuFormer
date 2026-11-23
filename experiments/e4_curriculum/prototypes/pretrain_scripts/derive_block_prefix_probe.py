"""Derive bounded qualification budgets from the exact production block order."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def derive(parent_path, destination, *, stage_limits, parent_stage_indices, publish=True):
    parent_path, destination = Path(parent_path), Path(destination)
    if destination.parent.resolve() != parent_path.parent.resolve() or destination == parent_path:
        raise ValueError('prefix probe must be a distinct sibling of the immutable parent manifest')
    with parent_path.open('rb') as handle:
        before = os.fstat(handle.fileno())
        raw = handle.read()
        parent = json.loads(raw)
        after = os.fstat(handle.fileno())
        identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if identity(before) != identity(after) or identity(parent_path.stat()) != identity(before):
            raise ValueError('production permutation manifest changed during probe derivation')
    if (parent.get('schema_version') != 2 or parent.get('scope') != 'PRODUCTION'
            or parent.get('status') != 'PASS' or parent.get('parent_manifest')
            or parent.get('purpose') != 'TRAINING_CAUSAL_BLOCK_PERMUTATION'
            or parent.get('algorithm') != 'blake2b-128-independent-causal-block-sort-v1'
            or not isinstance(parent.get('stages'), list) or not parent['stages']
            or type(parent.get('block_samples')) is not int or parent['block_samples'] <= 0):
        raise ValueError('prefix probe needs a complete production block-order manifest')
    if not stage_limits or len(stage_limits) != len(parent_stage_indices):
        raise ValueError('prefix probe stage roster is incomplete')
    block_samples = parent['block_samples']
    result = {k: v for k, v in parent.items() if k not in ('stages', 'scope')}
    result.update(scope='PREFIX_PROBE', parent_manifest={'file': parent_path.name,
        'sha256': hashlib.sha256(raw).hexdigest()}, stages=[])
    previous = -1
    for child_index, (source_index, samples) in enumerate(zip(parent_stage_indices, stage_limits)):
        if (type(source_index) is not int or source_index <= previous or source_index >= len(parent['stages'])
                or type(samples) is not int or samples <= 0 or samples % block_samples):
            raise ValueError('invalid prefix-probe stage index/consumption budget')
        source = parent['stages'][source_index]
        if (source.get('stage_index') != source_index or samples > source['stage_sample_limit']
                or source.get('permutation_population_samples') != source['stage_sample_limit']):
            raise ValueError('prefix probe exceeds its production parent population')
        result['stages'].append(dict(source, stage_index=child_index, parent_stage_index=source_index,
                                     stage_sample_limit=samples))
        previous = source_index
    content = (json.dumps(result, sort_keys=True, indent=2)+'\n').encode()
    if destination.exists():
        if destination.read_bytes() != content:
            raise ValueError('prefix probe already exists with a different immutable contract')
    elif publish:
        temporary = destination.with_name(destination.name+f'.tmp-{os.getpid()}')
        temporary.write_bytes(content)
        os.replace(temporary, destination)
    return result
