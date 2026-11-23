#!/usr/bin/env python3
"""Inspect staged Git blobs and maintained document links before publication."""
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = [rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    rb'gh[pousr]_[A-Za-z0-9]{36,}', rb'github_pat_[A-Za-z0-9_]{70,}',
    rb'AKIA[A-Z0-9]{16}', rb'hf_[A-Za-z0-9]{30,}', rb'sk-proj-[A-Za-z0-9_-]{40,}']


def main():
    raw = subprocess.check_output(['git', 'ls-files', '-s', '-z'], cwd=ROOT)
    entries = []
    for row in raw.split(b'\0'):
        if row:
            metadata, path = row.split(b'\t', 1)
            mode, oid, stage = metadata.split()
            if stage != b'0': raise ValueError('unmerged index')
            entries.append((path.decode(), oid.decode(), mode.decode()))
    objects = subprocess.run(['git', 'cat-file', '--batch'], cwd=ROOT,
        input=''.join(oid+'\n' for _, oid, _ in entries).encode(),
        stdout=subprocess.PIPE, check=True).stdout
    cursor = 0
    failures = []
    total = 0
    largest = []
    for path, oid, mode in entries:
        end = objects.index(b'\n', cursor)
        actual_oid, kind, size_text = objects[cursor:end].split()
        size = int(size_text)
        body = objects[end+1:end+1+size]
        cursor = end+size+2
        assert actual_oid.decode() == oid and kind == b'blob'
        total += size
        largest.append((size, path))
        if size > 20*1024**2: failures.append((path, 'blob exceeds20MiB'))
        parts = Path(path).parts
        if (path.startswith(('data/', 'report/history/', '.venv')) or path.endswith('.distcp')
                or '.ipynb_checkpoints' in parts or '__pycache__' in parts):
            failures.append((path, 'runtime or cache artifact'))
        if mode == '120000': failures.append((path, 'symlink needs explicit review'))
        for pattern in SECRET_PATTERNS:
            if re.search(pattern, body): failures.append((path, 'credential-shaped content; inspect locally'))
    maintained = ['README.md', 'report/TECHNICAL_REPORT.md', 'report/metrics/README.md',
        'docs/engineering/PLAYBOOK.md', 'docs/engineering/INCIDENTS.md',
        'docs/engineering/REPOSITORY_MAINTENANCE.md']
    for name in maintained:
        path = ROOT/name
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if target.startswith(('https:', 'http:', '#', 'mailto:')): continue
            value = target.split('#', 1)[0]
            if not (path.parent/value).exists(): failures.append((name, 'broken link: '+target))
    whitespace = subprocess.run(['git', 'diff', '--cached', '--check'], cwd=ROOT,
                                capture_output=True, text=True)
    if whitespace.returncode:
        # Never print full added lines, which could contain sensitive content.
        notices = [line for line in whitespace.stdout.splitlines()
                   if re.match(r'^.+:[0-9]+: .+\.$', line)]
        failures.extend((notice, 'whitespace check') for notice in notices[:20])
        if not notices: failures.append(('git', 'diff --check failed'))
    result = {'status': 'FAIL' if failures else 'PASS_STAGED_SIZE_SECRET_BOUNDARY_AND_CURRENT_LINK_CHECKS',
        'files': len(entries), 'total_bytes': total,
        'largest': sorted(largest, reverse=True)[:5], 'failures': failures,
        'scope': 'Pattern-based secret check is not a complete security audit; core training code qualifications are bound separately.'}
    print(json.dumps(result, ensure_ascii=False))
    if failures: raise SystemExit(1)


if __name__ == '__main__': main()
