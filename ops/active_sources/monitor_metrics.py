import math
import re

ANSI = re.compile(r'\x1b\[[0-9;]*m')


def latest_training_metrics(lines):
    for raw in reversed(lines):
        line = ANSI.sub('', raw)
        if re.search(r'\bvalidate\s+step:', line):
            continue
        step = re.search(r'\bstep:\s*(\d+)', line)
        loss = re.search(r'\bloss:\s*(\S+)', line)
        tps = re.search(r'\btps:\s*([\d,]+)', line)
        if not (step and loss and tps):
            continue
        try:
            value = float(loss.group(1))
            row = {'step': int(step.group(1)), 'loss': value if math.isfinite(value) else None,
                   'loss_is_finite': math.isfinite(value),
                   'tps_per_device': int(tps.group(1).replace(',', ''))}
            for label, pattern in (
                ('grad_norm', r'\bgrad_norm:\s*([\d.eE+\-]+)'),
                ('mfu_pct', r'\bmfu:\s*([\d.]+)'),
                ('memory_gib', r'\bmemory:\s*([\d.]+)GiB')):
                match = re.search(pattern, line)
                if match:
                    row[label] = float(match.group(1))
            timestamp = re.search(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', line)
            if timestamp:
                row['log_timestamp_utc'] = timestamp.group(0)
            return row
        except ValueError:
            continue
    return None
