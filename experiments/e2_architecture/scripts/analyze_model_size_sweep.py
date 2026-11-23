#!/usr/bin/env python3
"""Build a screening comparison and reject allocator-thrashing candidates."""
from pathlib import Path
import json
import math
import re
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
ANSI = re.compile(r'\x1b\[[0-9;]*m')
STEP = re.compile(r'\bstep:\s*(\d+)\s+loss:')


def memory_pressure(log, *, skip_steps=8, final_step=24):
    current_step = 0
    all_warnings = 0
    steady_warnings = 0
    with Path(log).open(errors='replace') as handle:
        for line in handle:
            match = STEP.search(ANSI.sub('', line))
            if match:
                current_step = max(current_step, int(match.group(1)))
            if 'memory allocation failed with OOM' in line:
                all_warnings += 1
                if skip_steps <= current_step < final_step:
                    steady_warnings += 1
    return {'allocation_retry_warnings_total': all_warnings,
            'allocation_retry_warnings_steady': steady_warnings,
            'requires_smaller_batch_comparison': steady_warnings > 0}


def analyze(root=ROOT):
    rows = []
    groups = {}
    for group, filename in [('size', 'model-size-sweep.json'),
                            ('width_and_batch', 'model-width-followup.json'),
                            ('width_frontier', 'model-width-frontier.json'),
                            ('attention_layout', 'model-attention-layouts.json')]:
        path = root / 'data/health' / filename
        if not path.is_file():
            continue
        state = json.loads(path.read_text())
        groups[group] = {'status': state['status'], 'pid': state.get('pid'),
                         'active_variant': state.get('active_variant'), 'error': state.get('error')}
        cases = list(state.get('cases', {}).values())
        baseline = next((v for v in cases if v['candidate'] == 'baseline_053b'
                         and v['evaluation']['status'] == 'PASS_SCREENING'), None)
        baseline_tps = (baseline['evaluation']['p10_tps_per_device'] if baseline else
                        (state.get('baseline_reference') or {}).get('p10_tps_per_device'))
        for case in cases:
            evaluation = case['evaluation']
            row = {'group': group, 'candidate': case['candidate'], 'variant': case['variant']['name'],
                   'parameters': case['parameters'], 'microbatch': case['variant']['microbatch'],
                   'grad_accum': case['variant']['grad_accum'], 'status': evaluation['status'],
                   'p10_tps_per_device': evaluation.get('p10_tps_per_device'),
                   'projected_1t_days_duty_09': evaluation.get('projected_1t_days_duty_09'),
                   'median_mfu_pct': evaluation.get('median_mfu_pct'),
                   'production_qualified': False, 'log': case['log']}
            pressure = memory_pressure(case['log'])
            row.update(pressure)
            ratio = row['p10_tps_per_device'] / baseline_tps if row['p10_tps_per_device'] and baseline_tps else None
            row['throughput_ratio_vs_group_baseline'] = ratio
            tps, days = row['p10_tps_per_device'], row['projected_1t_days_duty_09']
            measured = (row['status'] == 'PASS_SCREENING' and tps is not None and days is not None
                        and math.isfinite(tps) and tps > 0 and math.isfinite(days) and days > 0)
            row['eligible_throughput_screening'] = measured and not pressure['requires_smaller_batch_comparison']
            row['within_30_days_without_memory_pressure'] = bool(row['eligible_throughput_screening']
                and row['projected_1t_days_duty_09'] <= 30)
            row['within_10pct_slowdown_without_memory_pressure'] = bool(row['within_30_days_without_memory_pressure']
                and ratio is not None and ratio >= .9)
            rows.append(row)
    choose = lambda key: max((v for v in rows if v[key]), key=lambda v: v['parameters'], default=None)
    return {'status': 'SCREENING_COMPLETE' if groups and all(v['status'] == 'COMPLETE_SCREENING'
                for v in groups.values()) else 'SCREENING_IN_PROGRESS',
            'observed_at': datetime.now(timezone.utc).isoformat(), 'groups': groups, 'measurements': rows,
            'largest_under_10pct_slowdown': choose('within_10pct_slowdown_without_memory_pressure'),
            'largest_within_30_days': choose('within_30_days_without_memory_pressure'),
            'production_qualified': False,
            'limitations': ['Short runs measure throughput, not model quality.',
                'Any new production architecture requires a sustained canary, LR/numerical/recovery qualification and identity freeze.',
                'Allocator warnings after warmup require a smaller-batch comparison and cannot authorize production.']}


def main():
    report = analyze()
    path = ROOT / 'data/health/model-scale-sweet-point.json'
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    temp.replace(path)
    print(json.dumps({'status': report['status'], 'completed_measurements': len(report['measurements']),
                      'largest_under_10pct_slowdown': (report['largest_under_10pct_slowdown'] or {}).get('variant'),
                      'largest_within_30_days': (report['largest_within_30_days'] or {}).get('variant')}))


if __name__ == '__main__':
    main()
