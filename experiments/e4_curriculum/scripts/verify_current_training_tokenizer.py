#!/usr/bin/env python3
"""Diagnose tokenizer ratios on new selected orders without changing frozen inputs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import sys

import yaml
import calibrate_training_tokens as calibration

ROOT = Path(__file__).resolve().parents[3]
HEALTH = ROOT / 'data/health/post-selection-tokenizer-pilot.json'
WORK = ROOT / 'data/processed/tokenizer_calibration/post-selection-11326994-verification'


def sha(path):
    return calibration.digest(path)


def publish(value):
    value['updated_at'] = datetime.now(timezone.utc).isoformat()
    temporary = HEALTH.with_name(HEALTH.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, HEALTH)


def run():
    canonical = ROOT / calibration.REPORT
    frozen_path = ROOT / 'experiments/e4_curriculum/results/frozen_1t_recipe.json'
    release_path = ROOT / 'data/processed/releases/pretrain-1t-frozen.release.json'
    protected = {str(p.relative_to(ROOT)): sha(p) for p in (canonical, frozen_path, release_path)}
    state = {'status': 'RUNNING', 'pid': os.getpid(), 'work_dir': str(WORK),
             'purpose': 'POST_SELECTION_DIAGNOSTIC_ONLY', 'protected_inputs': protected,
             'wrapper_tool_sha256': sha(__file__), 'workers': 16,
             'minimum_reference_tokens_per_domain': 100_000_000,
             'does_not_replace_formal_full_stream_gates': True}
    publish(state)
    calibration.REPORT = str((WORK / 'measurement.json').relative_to(ROOT))
    assert ROOT / calibration.REPORT != canonical
    try:
        with calibration.RunLease(WORK, 'post-selection-tokenizer-verification'):
            report = calibration.build(ROOT, WORK,
                ROOT / 'tokenizer/tokenizers/m3_20261020/yf-64k-bpe/tokenizer.json',
                16, 100_000_000, release_path)
        for relative, expected in protected.items():
            if sha(ROOT / relative) != expected:
                raise ValueError('protected production input changed: ' + relative)
        frozen = json.loads(frozen_path.read_text())
        base = yaml.safe_load((ROOT / 'pretrain/configs/pretrain_base.yaml').read_text())
        forecasts = {}
        for index, stage in enumerate(('warmup', 'stable', 'anneal')):
            config = yaml.safe_load((ROOT / frozen['stages'][stage]['config']).read_text())
            manifest = json.loads((ROOT / 'data/processed/mix_order' / (config['mix_name'] + '.manifest.json')).read_text())
            ratios = report['stages'][stage]['domains']
            predicted = {domain: sum(sources.values()) * ratios[domain]['ratio']
                         for domain, sources in manifest['selected_tokens_by_source'].items()}
            total = sum(predicted.values())
            target = frozen['stages'][stage]['target_final_domain_weights']
            errors = {domain: abs(predicted[domain] / total - target[domain]) for domain in target}
            required = int(base['data']['stage_sample_limits'][index]) * int(base['data']['seq_len']) + 1
            forecasts[stage] = {'predicted_actual_tokens': total, 'required_causal_stream_tokens': required,
                               'predicted_capacity_sufficient': total >= required,
                               'predicted_weights': {d: v / total for d, v in predicted.items()},
                               'target_weights': target, 'absolute_errors': errors,
                               'max_absolute_error': max(errors.values()),
                               'prediction_within_formal_absolute_tolerance': max(errors.values()) <= .002,
                               'formal_full_stream_status': 'SEPARATE_GATE_REQUIRED'}
        state.update(status='PASS_DIAGNOSTIC' if all(v['predicted_capacity_sufficient']
            and v['prediction_within_formal_absolute_tolerance'] for v in forecasts.values())
            else 'PREDICTION_REVIEW_REQUIRED', forecasts=forecasts,
            measurement_report=str(WORK / 'measurement.json'),
            measurement_sha256=sha(WORK / 'measurement.json'), protected_inputs_unchanged=True,
            note='Pilot predictions have sampling uncertainty; complete actual consumed-stream gates remain authoritative.')
        publish(state)
    except BaseException as exc:
        state.update(status='BLOCKED_DIAGNOSTIC', error=f'{type(exc).__name__}: {exc}')
        publish(state)
        raise


if __name__ == '__main__':
    run()
