#!/usr/bin/env python3
"""Refresh an admitted half-day report; optionally commit and verify publication.

No training launch, dependency changes, cleanup or force pushes. Authentication
uses the repository-local Git helper and an explicitly provided GitHub CLI.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

PASS = 'PASS_HALF_DAY_OPERATIONAL_REVIEW'
ATTENTION = 'HALF_DAY_REVIEW_DOMAIN_REGRESSION_REQUIRES_ATTENTION'


def validate_observation(review, note):
    if review['status'] == PASS and not review['regressed_domains']:
        return
    if review['status'] != ATTENTION or not note or review['reviewed_at'] not in note:
        raise ValueError('non-PASS review requires a current, independently written observation note')


def validate_paths(paths, extra, root, protected):
    for path in paths:
        p = Path(path)
        if p.is_absolute() or '..' in p.parts:
            raise ValueError('repository path escapes root')
        if not (path == 'README.md' or path.startswith('report/') or path in extra):
            raise ValueError('unexpected change requires explicit inclusion: '+path)
        resolved = (root/p).resolve()
        if (root/p).is_symlink() or not resolved.is_relative_to(root) or str(resolved) in protected:
            raise ValueError('change intersects frozen dependency or escapes root: '+path)


def main(args):
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'GIT_TERMINAL_PROMPT': '0',
           'MPLCONFIGDIR': os.environ.get('MPLCONFIGDIR', '/tmp/yuformer-report-matplotlib')}

    def run(command):
        result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=180)
        if result.returncode:
            raise RuntimeError(f'{command[0]} exit{result.returncode}: {result.stderr[-2000:]} {result.stdout[-1200:]}')
        return result.stdout.strip(), result.stderr.strip()

    def changed_paths():
        tracked = run(['git', 'diff', 'HEAD', '--name-only', '-z'])[0]
        untracked = run(['git', 'ls-files', '--others', '--exclude-standard', '-z'])[0]
        return {p for value in (tracked, untracked) for p in value.split('\0') if p}

    cadence = json.loads((root/'data/health/assistant-monitoring-cadence.json').read_text())
    source = Path(cadence['last_review_evidence'])
    latest = sorted((root/'data/health').glob('assistant-half-day-pretrain-review-*.json'))[-1]
    if source.resolve() != latest.resolve(): raise ValueError('cadence and latest review disagree')
    review = json.loads(source.read_text())
    if review['status'] != cadence['last_review_status']: raise ValueError('review status mismatch')
    note = (root/args.observation_note).read_text() if args.observation_note else None
    validate_observation(review, note)
    stamp = source.stem.removeprefix('assistant-half-day-pretrain-review-')
    report = f'report/PRETRAIN_HALF_DAY_REVIEW_{stamp}.md'
    if not (root/report).is_file(): raise ValueError('original human review missing')
    state_path = root/'data/health/repository-publication-20261123.json'
    state = json.loads(state_path.read_text())
    if args.publish and (run(['git', 'symbolic-ref', '--short', 'HEAD'])[0] != 'main'
            or run(['git', 'rev-parse', 'HEAD'])[0] != state['remote_verified_commit']):
        raise ValueError('publish requires the previously verified main HEAD; inspect local changes')
    if any(row['review'] == report for row in state.get('published_reviews', [])):
        if changed_paths(): raise ValueError('already published review has new changes; inspect manually')
        print(json.dumps({'status': 'ALREADY_PUBLISHED', 'commit': state['remote_verified_commit']}))
        return
    protected = json.loads((root/'data/repository_maintenance/20261123-protected-before-cleanup.json').read_text())
    extra = set(args.include_path)
    validate_paths(changed_paths(), extra, root, protected)
    run([sys.executable, 'tools/export_pretrain_report.py'])
    run([args.plot_python, 'tools/plot_pretrain_metrics.py'])
    m = review['training_metrics']
    native = len(review['checkpoint_receipts_rechecked'])
    lr = sum(row['observations_checked'] for row in review['lr_contracts'])
    ops = root/'report/ACTIVE_PRETRAIN_OPERATIONS.md'
    text = ops.read_text()
    first, last = text.index('Latest complete review: '), text.index('Actual run:')
    summary = (f"Latest complete review: {review['reviewed_at']}, status{review['status']}, "
        f"step{m['step']}, {review['labels_at_last_logged_step']:,}labels, phase{review['phase']}. "
        f"Regressed domains={review['regressed_domains']}. Sameattempt/source/cursor/LR checks passed;"
        f"{native}native control receipts and{lr}actual LR scalars rechecked. Forecast total"
        f"{review['projected_total_days']:.3f}days, remaining{review['whole_chain_remaining_days']:.3f}days. "
        f"See {report} and report/metrics/latest_pretrain_status.json for losses and reference steps. "
        "P016/P018 histories and unknown causes stay preserved; later lower points are not causal repairs. "
        "Prior domain census covers72000..80000 only. Fixed64samples/domain are point estimates; "
        "final capability remains unmeasured.\n\n")
    text = text[:first]+summary+text[last:]
    text = re.sub(r'Next review \S+\. CPU-only source', f"Next review {review['next_review_at']}. CPU-only source", text, count=1)
    text = re.sub(r'Updated .*?\. This file', f'Updated {datetime.now(timezone.utc).isoformat()}. This file', text, count=1)
    ops.write_text(text)
    with (root/'report/metrics/pretrain_validation.csv').open() as handle:
        count = sum(1 for _ in csv.DictReader(handle))
    checks_path = root/'report/metrics/repository_release_checks.json'
    checks = json.loads(checks_path.read_text())
    checks['latest_half_day_followup'] = {'review': report, 'review_status': review['status'],
        'step': m['step'], 'validation_rows': count,
        'maintenance_tests_reused_from': checks['maintenance_tests']['last_run_at'],
        'scope': 'Refresh report/metrics; existing test evidence covers its recorded source version. No repeated GPU qualification or raw-token/weight full scan. Explicitly included new code needs separate validation.'}
    checks_path.write_text(json.dumps(checks, ensure_ascii=False, sort_keys=True, indent=2)+'\n')
    changed = changed_paths()
    validate_paths(changed, extra, root, protected)
    for name in changed:
        if name.endswith('.json'): json.loads((root/name).read_text())
    run(['git', 'add', '--', *sorted(changed)])
    audit = json.loads(run([sys.executable, 'tools/audit_repository.py'])[0])
    if audit['failures']: raise ValueError('staged audit failed')
    print(json.dumps({'audit': audit['status'], 'validation_rows': count, 'changed_files': len(changed)}), flush=True)
    if not args.publish:
        print(json.dumps({'status': 'LOCAL_REPORTS_STAGED_NOT_PUBLISHED'}))
        return
    if not args.gh: raise ValueError('--gh or YUFORMER_GH_CLI is required to verify authenticated login')
    if run([args.gh, 'api', 'user', '--jq', '.login'])[0] != 'yuhaoyan04':
        raise ValueError('unexpected authenticated GitHub account')
    expected = state['remote_verified_commit']
    if run(['git', 'rev-parse', 'HEAD'])[0] != expected: raise ValueError('local HEAD changed; inspect manually')
    url = run(['git', 'remote', 'get-url', '--push', 'origin'])[0]
    if url != 'https://github.com/yuhaoyan04/YuFormer.git': raise ValueError('unexpected push destination')
    if run(['git', 'ls-remote', url, 'refs/heads/main'])[0].split()[0] != expected:
        raise ValueError('remote advanced; inspect and integrate before publication')
    print(run(['git', '-c', 'user.name=yuhaoyan04', '-c', 'user.email=yuhaoyan04@users.noreply.github.com',
        'commit', '-m', f"Publish step-{m['step']} half-day training review"])[0], flush=True)
    head = run(['git', 'rev-parse', 'HEAD'])[0]
    if run(['git', 'status', '--porcelain'])[0]: raise ValueError('worktree changed during commit')
    out, err = run(['git', '-c', 'pack.threads=2', 'push', 'origin', 'main'])
    if run(['git', 'ls-remote', url, 'refs/heads/main'])[0].split()[0] != head:
        raise ValueError('remote SHA verification failed')
    state['published_commits'].append(head)
    state.update(status='REMOTE_MAIN_VERIFIED_WITH_HALF_DAY_UPDATE', latest_half_day_commit=head,
        remote_verified_commit=head, updated_at=datetime.now(timezone.utc).isoformat(),
        push_returncode=0, push_stdout=out, push_stderr=err)
    state.setdefault('published_reviews', []).append({'review': report, 'commit': head, 'verified_at': state['updated_at']})
    state_path.write_text(json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2)+'\n')
    print(json.dumps({'status': state['status'], 'remote_verified_commit': head,
        'next_review_at': review['next_review_at'], 'validation_rows': count, 'worktree': 'CLEAN'}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--gh', default=os.environ.get('YUFORMER_GH_CLI'))
    parser.add_argument('--plot-python', default=os.environ.get('YUFORMER_PLOT_PYTHON', 'python'))
    parser.add_argument('--observation-note', help='Current independent analysis required for a non-PASS review')
    parser.add_argument('--include-path', action='append', default=[], help='Explicitly reviewed additional repository path')
    main(parser.parse_args())
