#!/usr/bin/env python3
"""Plan or remove a narrow set of obsolete artifacts; preserve active bindings.

This intentionally has no age-based, data-root-wide or checkpoint-wide deletion.
The only heavy target is the explicitly retired pre-calibration canary archive.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def protected_bindings(root: Path) -> set[Path]:
    paths = set()
    sources = [root/'experiments/e4_curriculum/results/frozen_1t_recipe.json',
               root/'eval/configs/authorized_context_chain_20261115_v1.json',
               root/'data/health/live-pretrain-supervisor-recovery-20261118.json']
    reviews = sorted((root/'data/health').glob('assistant-half-day-pretrain-review-*.json'))
    if reviews:
        sources.append(reviews[-1])
    for source in sources:
        paths.add(source.resolve())
        data = json.loads(source.read_bytes())
        for key in ('inputs', 'outputs', 'source_inputs'):
            for value in data.get(key, {}):
                paths.add((root/value).resolve())
    # These roots are never deletion candidates, including unlisted native files.
    paths.add((root/'data').resolve())
    paths.add((root/'.git').resolve())
    return paths


def assert_safe(candidate: Path, root: Path, protected: set[Path]) -> None:
    root, candidate = root.resolve(), candidate.absolute()
    if candidate.is_symlink() or not candidate.is_relative_to(root) or candidate == root:
        raise ValueError('unsafe cleanup boundary: '+str(candidate))
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('cleanup escapes workspace')
    for path in protected:
        if resolved == path or resolved in path.parents or path in resolved.parents:
            raise ValueError('cleanup intersects protected dependency: '+str(path))
    if candidate.is_dir() and any(p.is_symlink() for p in candidate.rglob('*')):
        raise ValueError('cleanup tree contains symlinks')


def candidates(root: Path) -> list[Path]:
    found = []
    skip = {'data', '.git', '.venv-data', '.venv', 'venv', 'history', 'tokenizers', 'baselines'}
    junk = {'.ipynb_checkpoints', '__pycache__', '.pytest_cache'}
    for base, dirs, _ in os.walk(root):
        for name in list(dirs):
            if name in junk:
                found.append(Path(base)/name)
                dirs.remove(name)
            elif name in skip:
                dirs.remove(name)
    stale = root/'report/history/before-training-calibration-20261113T215043Z/stale-production-canary-run'
    for name in ('checkpoint', 'structured_logs'):
        path = stale/name
        if path.exists():
            found.append(path)
    return sorted(found)


def opened_files() -> set[tuple[int, int]]:
    result = set()
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            for fd in (process/'fd').iterdir():
                try:
                    stat = fd.stat()
                    result.add((stat.st_dev, stat.st_ino))
                except (FileNotFoundError, PermissionError):
                    pass
        except (FileNotFoundError, PermissionError):
            pass
    return result


def run(root: Path, apply: bool) -> dict:
    root = root.resolve()
    protected = protected_bindings(root)
    opened = opened_files()
    selected = candidates(root)
    listing = []
    for path in selected:
        assert_safe(path, root, protected)
        files = []
        for member in sorted(path.rglob('*')):
            if not member.is_file():
                continue
            stat = member.stat()
            if (stat.st_dev, stat.st_ino) in opened:
                raise ValueError('candidate is open in a process: '+str(member))
            row = {'path': str(member.relative_to(root)), 'bytes': stat.st_size,
                   'mtime_ns': stat.st_mtime_ns, 'inode': stat.st_ino, 'device': stat.st_dev}
            if member.name == '.metadata':
                row['sha256'] = digest(member)
            files.append(row)
        listing.append({'path': str(path.relative_to(root)), 'files': files,
                        'bytes': sum(row['bytes'] for row in files)})
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = root/'report/metrics'/f'cleanup-{timestamp}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {'status': 'PLANNED_NOT_DELETED', 'created_at': datetime.now(timezone.utc).isoformat(),
              'scope': 'OBSOLETE_CACHES_AND_RETIRED_PRECALIBRATION_CANARY_ONLY',
              'targets': listing, 'total_bytes': sum(item['bytes'] for item in listing),
              'protected_dependency_count': len(protected),
              'protected_paths': sorted(str(p) for p in protected),
              'authorization': 'User explicitly requested obsolete local data/code/docs cleanup on2026-11-23.',
              'large_shard_hashes': 'Not rescanned; stat identities retained. Native metadata hashes retained. These are obsolete artifacts, not current qualification or formal checkpoints.'}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)+'\n')
    if apply:
        for item, path in zip(listing, selected):
            assert_safe(path, root, protected_bindings(root))
            for row in item['files']:
                stat = (root/row['path']).stat()
                if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != (row['device'], row['inode'], row['bytes'], row['mtime_ns']):
                    raise ValueError('cleanup candidate changed after plan')
            shutil.rmtree(path)
        result['status'] = 'REMOVED_OBSOLETE_ARTIFACTS_PROTECTED_DEPENDENCIES_RETAINED'
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)+'\n')
    return {'status': result['status'], 'targets': len(listing),
            'bytes': result['total_bytes'], 'manifest': str(output)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.apply)))
