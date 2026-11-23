"""Build explicit short qualification budgets over the actual production order."""
import copy
import hashlib
import json
from pathlib import Path

from derive_block_prefix_probe import derive


def resolve(value, environment, root):
    for key, replacement in environment.items():
        value = value.replace('${'+key+'}', replacement)
    if '${' in value:
        raise ValueError('unresolved production probe input: '+value)
    path = Path(value)
    return (path if path.is_absolute() else Path(root)/path).resolve()


def build_probe(root, base, environment, *, role, publish=False):
    if (base['data'].get('sampling_policy') != 'block_permutation'
            or len(base['data'].get('stages', [])) != 3 or len(base['data'].get('meta', [])) != 3
            or base['optim']['lr_schedule'] != 'wsd'):
        raise ValueError('production probes require the explicit three-stage WSD block-order contract')
    parent_path = resolve(base['data']['block_permutation_manifest'], environment, root)
    parent_raw = parent_path.read_bytes()
    parent_sha = hashlib.sha256(parent_raw).hexdigest()
    if role == 'recovery':
        steps, stage_steps, parent_indices = 24, [8, 8, 8], [0, 1, 2]
    elif role == 'canary':
        steps, stage_steps, parent_indices = 200, [200], [0]
    else:
        raise ValueError('unknown production probe role')
    samples_per_update = (int(base['train']['micro_batch_size'])*int(base['train']['grad_accum'])
                          *int(base['parallel']['world_size']))
    limits = [n*samples_per_update for n in stage_steps]
    generator_sha = hashlib.sha256(Path(__file__).with_name('derive_block_prefix_probe.py').read_bytes()).hexdigest()
    child_path = parent_path.with_name(f'probe-{role}-{steps}-{parent_sha[:16]}-{generator_sha[:12]}.json')
    child = derive(parent_path, child_path, stage_limits=limits, parent_stage_indices=parent_indices,
                   publish=publish)
    if child['parent_manifest']['sha256'] != parent_sha:
        raise ValueError('production order changed while deriving its qualification probe')
    config = copy.deepcopy(base)
    config['experiment_status'] = 'qualification'
    config['data'].update(stages=[base['data']['stages'][i] for i in parent_indices],
        meta=[base['data']['meta'][i] for i in parent_indices], stage_sample_limits=limits,
        block_permutation_manifest=str(child_path))
    config['train'].update(expected_iters=steps,
        budget_tokens=steps*samples_per_update*int(base['data']['seq_len']))
    if role == 'recovery':
        config['optim'].update(warmup_steps=8, decay_steps=8)
        config['train'].update(save_every=4, eval_every=8, log_every=1, keep_last_n=3)
    else:
        config['optim'].update(lr_schedule='cosine', warmup_steps=10, min_lr_ratio=.1)
        config['optim'].pop('decay_steps', None)
        config['train'].update(save_every=100, eval_every=100, log_every=10, keep_last_n=2)
    child_sha = hashlib.sha256((json.dumps(child, sort_keys=True, indent=2)+'\n').encode()).hexdigest()
    identity = {'role': role, 'steps': steps, 'parent_order_manifest': str(parent_path),
        'parent_order_manifest_sha256': parent_sha, 'probe_order_manifest': str(child_path),
        'probe_order_manifest_sha256': child_sha, 'probe_stage_samples': limits,
        'parent_stage_indices': parent_indices, 'tokens_per_update': samples_per_update*int(base['data']['seq_len']),
        'production_optimizer': copy.deepcopy(base['optim']), 'probe_optimizer': copy.deepcopy(config['optim']),
        'production_seed': base['seed'], 'probe_seed': config['seed'],
        'model_config_file': str(resolve(base['model']['configs'], environment, root))}
    return config, identity
