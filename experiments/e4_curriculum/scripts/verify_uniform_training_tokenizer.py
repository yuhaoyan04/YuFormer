#!/usr/bin/env python3
"""E133 independent calibration diagnostic; preserve all production identities."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import yaml

import calibrate_training_tokens as calibration
import uniform_selected_sample as uniform

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / 'data/processed/tokenizer_calibration/post-selection-11326994-uniform-v2'
HEALTH = ROOT / 'data/health/uniform-training-tokenizer-pilot.json'
SEED = 'yuformer-e133-independent-calibration-20261114-v1'


def run():
    release_path = ROOT / 'data/processed/releases/pretrain-1t-frozen.release.json'
    frozen_path = ROOT / 'experiments/e4_curriculum/results/frozen_1t_recipe.json'
    canonical_report = ROOT / calibration.REPORT
    paths = [release_path, frozen_path, canonical_report, Path(calibration.__file__),
             ROOT / 'pretrain/scripts/materialize_tokens.py']
    protected = {str(p.relative_to(ROOT)): calibration.digest(p) for p in paths}
    state = {'status': 'RUNNING', 'pid': os.getpid(), 'work_dir': str(WORK),
             'purpose': 'INDEPENDENT_SELECTED_POPULATION_DIAGNOSTIC_ONLY',
             'sampling': uniform.SAMPLING, 'seed': SEED,
             'protected_inputs': protected, 'workers': 16,
             'minimum_reference_tokens_per_domain': 100_000_000,
             'does_not_replace_formal_full_stream_gates': True,
             'wrapper_tool_sha256': calibration.digest(__file__),
             'sampler_tool_sha256': calibration.digest(uniform.__file__)}
    original_atomic = calibration.atomic_json

    def publish():
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        original_atomic(HEALTH, state)

    publish()
    try:
        release = json.loads(release_path.read_text())
        if release.get('status') != 'PASS':
            raise ValueError('parent release must pass')
        artifacts = {row['path']: row for row in release['artifacts']}
        populations, expected_sha = {}, {}
        with calibration.RunLease(WORK, 'uniform-selected-training-tokenizer-diagnostic'):
            for phase, row in release['stages'].items():
                manifest = ROOT / row['manifest']
                with manifest.open('rb') as handle:
                    before = calibration.materializer._stat_identity(os.fstat(handle.fileno()))
                    digest = calibration.materializer._hash_open_file(handle)
                    if digest != artifacts[row['manifest']]['sha256']:
                        raise ValueError('manifest differs from selected release: ' + phase)
                    populations[phase] = json.load(handle)['selected']
                    if (calibration.materializer._stat_identity(os.fstat(handle.fileno())) != before or
                            calibration.materializer.source_file_identity(manifest) != before):
                        raise ValueError('selected manifest changed during read')
                expected_sha[phase] = artifacts[row['order']]['sha256']
                directory = WORK / phase
                directory.mkdir(parents=True, exist_ok=True)
                parent = directory / 'parent-order.parquet'
                # Use the immutable selected inode; sampler hashes and reads the same FD.
                if not parent.exists():
                    parent.symlink_to(ROOT / row['order'])
                if parent.resolve() != (ROOT / row['order']).resolve():
                    raise ValueError('diagnostic parent link differs from selected order')

            def sample(parent, target, destination):
                phase = parent.parent.name
                state.update(active_phase=phase, active_operation='full-population-independent-rank-scan')
                publish()
                result = uniform.sample_order(parent, target, destination,
                    population=populations[phase], expected_parent_sha256=expected_sha[phase],
                    seed=SEED, phase=phase)
                state.update(active_operation='source-bound-tokenization')
                publish()
                return result

            def publish_report(path, value):
                if value.get('purpose') == 'TRAINING_TOKENIZER_CALIBRATION':
                    value.update(purpose=state['purpose'], sampling=uniform.SAMPLING, seed=SEED,
                                 wrapper_tool_sha256=state['wrapper_tool_sha256'],
                                 sampler_tool_sha256=state['sampler_tool_sha256'])
                original_atomic(path, value)

            # The legacy materializer/measurement path is reused only inside this process.
            calibration.sample_order = sample
            calibration.atomic_json = publish_report
            calibration.REPORT = str((WORK / 'measurement.json').relative_to(ROOT))
            report = calibration.build(ROOT, WORK,
                ROOT / 'tokenizer/tokenizers/m3_20261020/yf-64k-bpe/tokenizer.json',
                16, 100_000_000, release_path)
        for relative, expected in protected.items():
            if calibration.digest(ROOT / relative) != expected:
                raise ValueError('protected production input changed: ' + relative)
        if (calibration.digest(__file__) != state['wrapper_tool_sha256'] or
                calibration.digest(uniform.__file__) != state['sampler_tool_sha256']):
            raise ValueError('diagnostic source changed while sampling')
        frozen = json.loads(frozen_path.read_text())
        base = yaml.safe_load((ROOT / 'pretrain/configs/pretrain_base.yaml').read_text())
        forecasts = {}
        for index, phase in enumerate(('warmup', 'stable', 'anneal')):
            domain_rows = report['stages'][phase]['domains']
            predicted = {d: populations[phase][d]['selected_tokens'] * domain_rows[d]['ratio']
                         for d in calibration.DOMAINS}
            total = sum(predicted.values())
            target = frozen['stages'][phase]['target_final_domain_weights']
            errors = {d: abs(predicted[d]/total-target[d]) for d in target}
            required = int(base['data']['stage_sample_limits'][index]) * int(base['data']['seq_len']) + 1
            forecasts[phase] = {'predicted_actual_tokens': total,
                'required_causal_stream_tokens': required, 'predicted_capacity_sufficient': total >= required,
                'predicted_weights': {d: v/total for d, v in predicted.items()},
                'target_weights': target, 'absolute_errors': errors,
                'max_absolute_error': max(errors.values()),
                'prediction_within_formal_absolute_tolerance': max(errors.values()) <= .002,
                'formal_full_stream_status': 'SEPARATE_GATE_REQUIRED'}
        state.update(status='PASS_DIAGNOSTIC' if all(v['predicted_capacity_sufficient'] and
            v['prediction_within_formal_absolute_tolerance'] for v in forecasts.values())
            else 'PREDICTION_REVIEW_REQUIRED', forecasts=forecasts,
            measurement_report=str(WORK / 'measurement.json'),
            measurement_sha256=calibration.digest(WORK / 'measurement.json'),
            protected_inputs_unchanged=True,
            note='Uniform finite samples have uncertainty; complete actual consumed-stream gates are authoritative.')
        publish()
    except BaseException as exc:
        state.update(status='BLOCKED_DIAGNOSTIC', error=f'{type(exc).__name__}: {exc}')
        publish()
        raise


if __name__ == '__main__':
    run()
