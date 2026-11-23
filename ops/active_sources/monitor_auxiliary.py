"""Read-only host and data-index observations; never modifies pipeline state."""
import json
import os
from pathlib import Path


def process_observation(pid):
    process = Path('/proc') / str(int(pid))
    try:
        fields = (process / 'stat').read_text().rsplit(') ', 1)[1].split()
        status = dict(line.split(':', 1) for line in (process / 'status').read_text().splitlines() if ':' in line)
        io = dict(line.split(':', 1) for line in (process / 'io').read_text().splitlines())
        return {'alive': fields[0] != 'Z',
                'cpu_seconds': round((int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK'), 2),
                'rss_gib': round(int(status.get('VmRSS', '0 kB').split()[0]) / 1024 ** 2, 3),
                'read_chars': int(io['rchar']), 'read_bytes': int(io['read_bytes'])}
    except (OSError, ValueError, IndexError, KeyError):
        return {'alive': False}


def auxiliary_observations(root, tail_lines):
    result = {}
    try:
        report = json.loads((root / 'data/health/source-document-index-build.json').read_text())
        index = {key: report.get(key) for key in ('status', 'pid', 'started_at', 'rows', 'files', 'elapsed_seconds', 'error', 'legacy_entire_source_mapping_equals_rebuilt_mapping') if key in report}
        if report.get('pid'):
            index.update(process_observation(report['pid']))
        lines = tail_lines(root / 'report/logs/source-document-index-build.log')
        index['latest_progress'] = lines[-1] if lines else None
        index['unexpected_exit'] = report.get('status') == 'RUNNING' and index.get('alive') is False
        result['source_document_index'] = index
    except (OSError, ValueError, TypeError):
        pass
    try:
        base = Path('/sys/fs/cgroup/memory')
        usage, limit, failures = [int((base / filename).read_text()) for filename in ('memory.usage_in_bytes', 'memory.limit_in_bytes', 'memory.failcnt')]
        result['container_memory'] = {'used_gib': round(usage / 1024 ** 3, 2), 'limit_gib': round(limit / 1024 ** 3, 2), 'headroom_gib': round((limit - usage) / 1024 ** 3, 2), 'memory_limit_hits': failures}
    except (OSError, ValueError):
        pass
    try:
        boundary = json.loads((root / 'data/health/s6-fd-fix-boundary.json').read_text())
        observed = {key: boundary.get(key) for key in
                    ('status', 'stage', 'pid', 'controller_pid', 'child_pid', 'updated_at', 'error') if key in boundary}
        observed.update(process_observation(boundary['pid']))
        observed['unexpected_exit'] = boundary.get('status') == 'RUNNING' and not observed.get('alive')
        result['s6_fd_source_fix'] = observed
    except (OSError, ValueError, KeyError, TypeError):
        pass
    result['monitor_schema_version'] = 2
    return result
