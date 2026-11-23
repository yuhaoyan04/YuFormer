import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
from monitor_metrics import latest_training_metrics
from monitor_auxiliary import auxiliary_observations
import time

root = Path('/opt/huawei/explorer-env/dataset/yyh_arcagi/YUFormer/YuFormer')
lock = (root / 'data/health/authorized-pretrain-monitor.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
(root / 'data/health/authorized-pretrain-monitor.pid').write_text(str(os.getpid()) + '\n')

def tail_lines(path):
    path = Path(path)
    if not path.is_file():
        return []
    with path.open('rb') as handle:
        handle.seek(max(0, path.stat().st_size - 65536))
        return handle.read().decode(errors='replace').splitlines()

last_train_metrics = None

while True:
    state = json.loads((root / 'data/health/authorized-pretrain-pipeline.json').read_text())
    pid = state.get('pid')
    try:
        os.kill(pid, 0)
        stat = Path(f'/proc/{pid}/stat').read_text()
        alive = stat[stat.rfind(')') + 2:].split()[0] != 'Z'
    except (OSError, TypeError):
        alive = False
    phase = state.get('stage')
    log = state.get('stages', {}).get(phase, {}).get('log')
    lines = tail_lines(log) if log else []
    row = {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
           'pid': pid, 'status': state.get('status'), 'stage': phase, 'alive': alive,
           'error': state.get('error'),
           'last_log': re.sub(r'\x1b\[[0-9;]*m', '', lines[-1]) if lines else None}
    if phase and 'qualification' in phase:
        p = root / 'data/health/torchtitan-qualification.json'
        if p.exists():
            q = json.loads(p.read_text())
            row['qualification'] = {k: v.get('status') for k, v in q.get('stages', {}).items()}
    if phase == 'production_recovery':
        p = root / 'data/health/production-recovery-qualification.json'
        if p.exists():
            q = json.loads(p.read_text())
            row['recovery'] = {k: v.get('status') for k, v in q.get('stages', {}).items()}
    if phase in ('orders', 'tokens'):
        filename = ('pretrain-1t-order-materialization.json' if phase == 'orders'
                    else 'pretrain-1t-tokenization.json')
        try:
            data_state = json.loads((root / 'data/health' / filename).read_text())
            row['data_jobs'] = {k: {j: v.get(j) for j in
                                      ('status', 'attempts', 'total_attempts', 'returncode')}
                                for k, v in data_state.get('jobs', {}).items()}
            active = next((v for v in data_state.get('jobs', {}).values()
                           if v.get('status') == 'RUNNING'), {})
            inner = Path(active.get('log', '/nonexistent'))
            row['data_active_started_at'] = active.get('started_at')
            if inner.is_file():
                with inner.open('rb') as handle:
                    handle.seek(max(0, inner.stat().st_size - 8192))
                    tail = handle.read().decode(errors='replace').splitlines()
                row['data_last_log'] = tail[-1] if tail else None
                row['data_log_age_seconds'] = round(time.time() - inner.stat().st_mtime)
                started = datetime.datetime.fromisoformat(active['started_at']).timestamp()
                row['data_log_from_current_attempt'] = inner.stat().st_mtime >= started
        except (OSError, ValueError, TypeError):
            pass
    if phase == 'orders':
        try:
            generation = max((root / 'data/processed/.driver_s6.lock').glob('generation-*'))
            heartbeat = next(generation.glob('heartbeat-*'))
            row['s6_lease_generation'] = generation.name
            row['s6_heartbeat_age_seconds'] = round(time.time() - heartbeat.stat().st_mtime)
        except (OSError, ValueError, StopIteration):
            pass
    production = state.get('stages', {}).get('full_pretrain', {})
    row['full_pretrain_launched'] = bool(production.get('pid'))
    if phase == 'full_pretrain':
        observed = latest_training_metrics(lines)
        if observed is not None:
            last_train_metrics = observed
        row['training_metrics'] = last_train_metrics
        try:
            gpu = subprocess.run(['nvidia-smi',
                '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True)
            labels = ('index', 'utilization_pct', 'memory_used_mib', 'memory_total_mib', 'temperature_c', 'power_w')
            row['gpu_inventory'] = [dict(zip(labels, [float(v.strip()) if v.strip() != 'N/A' else None
                                                     for v in line.split(',')]))
                                    for line in gpu.stdout.splitlines() if line.strip()]
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            row['gpu_query_error'] = type(exc).__name__
    checkpoints = list((root / 'data/runs/pretrain-final-1t-8gpu/checkpoint').glob('step-*/.metadata'))
    row['latest_complete_production_checkpoint'] = max(
        (int(p.parent.name.removeprefix('step-')) for p in checkpoints), default=None)
    if phase == 'orders' and alive:
        try:
            generation = max((root / 'data/processed/.driver_s6.lock').glob('generation-*'))
            owner = json.loads((generation / 'owner.json').read_text())
            proc = Path(f"/proc/{owner['pid']}")
            io = dict(line.split(': ', 1) for line in (proc / 'io').read_text().splitlines())
            row['s6_process_io'] = {key: int(io[key]) for key in ('rchar', 'read_bytes', 'syscr')}
            status = dict(line.split(':', 1) for line in (proc / 'status').read_text().splitlines() if ':' in line)
            row['s6_process_rss_gib'] = round(int(status.get('VmRSS', '0 kB').split()[0]) / 1024 ** 2, 3)
            stat = (proc / 'stat').read_text()
            fields = stat[stat.rfind(')') + 2:].split()
            row['s6_process_cpu_seconds'] = round((int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK'), 2)
        except (OSError, ValueError, KeyError):
            pass
    try:
        launch = json.loads((root / 'data/health/production-recovery-full-accumulation-launch.json').read_text())
        auxiliary_pid = int(launch['pid'])
        auxiliary_proc = Path(f'/proc/{auxiliary_pid}/stat')
        auxiliary_alive = (auxiliary_proc.is_file()
                           and auxiliary_proc.read_text().split(') ', 1)[1].split()[0] != 'Z')
        recovery_path = root / 'data/health/production-recovery-qualification.json'
        recovery = json.loads(recovery_path.read_text()) if recovery_path.is_file() else {}
        if recovery.get('production_contract', {}).get('gradient_accumulation') != launch['gradient_accumulation']:
            recovery = {}
        auxiliary = {'pid': auxiliary_pid, 'alive': auxiliary_alive,
                     'gradient_accumulation': launch['gradient_accumulation'],
                     'formal_pretrain': False, 'report_status': recovery.get('status'),
                     'stages': {name: value.get('status') for name, value in recovery.get('stages', {}).items()}}
        logs = list((root / 'data/experiments/production_recovery').glob('q7_*/qualification.log'))
        if logs:
            latest = max(logs, key=lambda path: path.stat().st_mtime)
            auxiliary['latest_case'] = latest.parent.name
            auxiliary['latest_training_metrics'] = latest_training_metrics(tail_lines(latest))
        row['parallel_production_recovery'] = auxiliary
    except (OSError, ValueError, KeyError):
        pass
    row.update(auxiliary_observations(root, tail_lines))
    row['observational_only'] = True
    output = root / 'data/health/authorized-pretrain-live-monitor.json'
    temporary = output.with_name(output.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, output)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    time.sleep(30)
