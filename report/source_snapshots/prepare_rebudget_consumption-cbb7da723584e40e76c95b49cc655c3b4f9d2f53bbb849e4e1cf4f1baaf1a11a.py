"""Prepare exact consumption proofs for an approved budget without publishing it."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import yaml

from audit_1t_overlap import extract_consumed, audit_consumed_mixture, require_published_stat
from frozen_training_evidence import publish_checked
from global_training_budget import validate_global_budget
from training_data_origin import read_bound
from training_source_windows import resolve_consumption_window, require_window_admission, stat_identity, POLICY

ROOT = Path(__file__).resolve().parents[3]
PHASES = ('warmup', 'stable', 'anneal')


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def prepare(root=ROOT):
    root = Path(root).resolve()
    budget_path = root/'data/health/user-total-pretrain-and-length-budget.json'
    raw, budget_sha = read_bound(budget_path)
    budget = json.loads(raw)
    stage = validate_global_budget(budget)[0]
    release_path = root/'data/processed/releases/pretrain-1t-frozen.release.json'
    release_raw, release_sha = read_bound(release_path)
    release = json.loads(release_raw)
    frozen_path = root/'data/processed/recipe_origins'/release['frozen_recipe_sha256']/'objects'/release['frozen_recipe_sha256']
    frozen_raw, _ = read_bound(frozen_path, release['frozen_recipe_sha256'])
    frozen = json.loads(frozen_raw)
    if release.get('status') != 'PASS' or frozen.get('status') != 'PASS':
        raise ValueError('rebudget consumption needs the original PASS physical data recipe')
    folder = root/'data/processed/budget_consumption'/('total1t-including-long-'+budget_sha[:16])
    folder.mkdir(parents=True, exist_ok=True)
    cache = folder/'exact_records'
    cache.mkdir(exist_ok=True)
    base = yaml.safe_load((root/'pretrain/configs/pretrain_base.yaml').read_bytes())
    base['data']['stage_sample_limits'] = [p['samples'] for p in stage['phases']]
    base['train'].update(budget_tokens=stage['tokens'], expected_iters=stage['updates'])
    base['optim'].update(warmup_steps=stage['phases'][0]['updates'], decay_steps=stage['phases'][2]['updates'])
    preview = folder/'budget-geometry-preview.yaml'
    publish_checked(preview, yaml.safe_dump(base, sort_keys=False).encode(), immutable=True)
    pins = {str(p.relative_to(root)): digest(p) for p in
        (budget_path, release_path, frozen_path, preview, Path(__file__),
         root/'experiments/e4_curriculum/scripts/training_source_windows.py',
         root/'experiments/e4_curriculum/scripts/audit_1t_overlap.py')}
    state = {'pid': os.getpid(), 'status': 'PREPARING_EXACT_REBUDGET_CONSUMPTION',
             'production_qualified': False, 'production_configuration_modified': False,
             'budget_sha256': budget_sha, 'source_inputs': pins, 'phases': {}}
    health = root/'data/health/approved-budget-consumption-preparation.json'

    def save():
        publish_checked(health, (json.dumps(state, sort_keys=True, indent=2)+'\n').encode())

    def guard():
        for name, sha in pins.items():
            read_bound(root/name, sha)

    save()
    try:
        for i, phase in enumerate(PHASES):
            guard()
            samples, seq = stage['phases'][i]['samples'], stage['seq_len']
            meta_path = root/f'data/token_streams/pretrain-1t/{phase}.meta.json'
            meta_raw, meta_sha = read_bound(meta_path)
            meta = json.loads(meta_raw)
            order = root/release['stages'][phase]['order']
            row = next(v for v in release['artifacts'] if v['path'] == str(order.relative_to(root)))
            order_stat = require_published_stat(order, row)
            ledger = meta_path.parent/meta['doc_lengths']['file']
            ledger_stat = stat_identity(ledger.stat())
            old_path = root/f'data/health/pretrain-1t-{phase}-mixture-preflight.json'
            old_raw, old_sha = read_bound(old_path)
            old = json.loads(old_raw)
            state.update(active_phase=phase, status='PREPARING_EXACT_REBUDGET_CONSUMPTION')
            save()
            if old.get('configured_target_tokens') == samples*seq:
                # Preserve the actual earlier proof as an immutable object;
                # canonical health files may be republished after final adoption.
                snapshot = folder/(phase+'-prior-admission-'+old_sha+'.json')
                publish_checked(snapshot, old_raw, immutable=True)
                report = copy.deepcopy(old)
                report['unchanged_geometry_parent'] = {'file': str(snapshot.relative_to(root)), 'sha256': old_sha}
            else:
                if digest(ledger) != meta['doc_lengths']['sha256']:
                    raise ValueError('rebudget consumption ledger SHA mismatch')
                identity = f"{row['sha256']}:{meta['doc_lengths']['sha256']}:{samples}:{seq}"
                prefix = cache/(phase+'.consumed.records')
                extracted = extract_consumed(order, ledger, prefix, identity=identity, trained_tokens=samples*seq)
                targets = frozen['stages'][phase]['target_final_domain_weights']
                mixture = audit_consumed_mixture(extracted['domain_tokens'], targets,
                    absolute_tolerance=.002, relative_tolerance=.1)
                context = {'release_manifest_sha256': release_sha,
                    'data_frozen_recipe_sha256': release['frozen_recipe_sha256'],
                    'stream_meta_sha256': meta_sha, 'order_sha256': row['sha256'],
                    'doc_lengths_sha256': meta['doc_lengths']['sha256'],
                    'total_source_tokens': meta['total_tokens'],
                    'base_config_sha256': digest(preview), 'budget_geometry_file': str(preview.relative_to(root)),
                    'approved_budget_sha256': budget_sha}
                extracted, mixture, window = resolve_consumption_window(root, phase, order, ledger, prefix,
                    context=context, seq_len=seq, stage_samples=samples, targets=targets,
                    prefix=extracted, prefix_mixture=mixture, absolute_tolerance=.002, relative_tolerance=.1)
                report = {'status': mixture['status'], 'stage': phase, 'mixture': mixture,
                    'configured_target_tokens': samples*seq, 'consumed': extracted, 'source_window': window,
                    'evidence_sha256': {**pins, str(meta_path.relative_to(root)): meta_sha},
                    'order_stat': order_stat, 'ledger_stat': ledger_stat,
                    'audit_tool_sha256': digest(root/'experiments/e4_curriculum/scripts/audit_1t_overlap.py'),
                    'tokenization_tool_sha256': digest(root/'experiments/e4_curriculum/scripts/tokenize_1t_orders.py')}
            report['consumption_preparation_tool'] = {'file': str(Path(__file__).relative_to(root)),
                'sha256': digest(__file__)}
            report['approved_budget_sha256'] = budget_sha
            require_window_admission(root, report, phase=phase, stage_samples=samples, seq_len=seq,
                release_sha256=release_sha, stream_meta_sha256=meta_sha,
                data_frozen_recipe_sha256=release['frozen_recipe_sha256'])
            if require_published_stat(order, row) != order_stat or stat_identity(ledger.stat()) != ledger_stat:
                raise ValueError('physical sources changed during rebudget consumption proof')
            output = folder/(phase+'-mixture-admission.json')
            publish_checked(output, (json.dumps(report, sort_keys=True, indent=2)+'\n').encode(), immutable=True)
            state['phases'][phase] = {'report': str(output.relative_to(root)), 'sha256': digest(output),
                'tokens': samples*seq, 'source_sample_offset': report['source_window']['source_sample_offset']}
            save()
        guard()
        state.update(status='PREPARED_EXACT_CONSUMPTION_ADMISSIONS_NOT_PRODUCTION_PUBLISHED',
            remaining_gates=['new exact block permutation', 'temporal admission', 'exact consumed overlap',
                             'runtime and model qualification', 'training recipe adoption'])
        save()
        return state
    except Exception as exc:
        state.update(status='FAIL_CLOSED', error=repr(exc))
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    prepare(args.root)
