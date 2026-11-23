"""Review all 200 actual sustained updates rather than a short-probe default."""
import re

ANSI = re.compile(r'\x1b\[[0-9;]*m')
STEP = re.compile(r'\bstep:\s*(\d+)\s+loss:')


def review_memory_log(raw, *, steps=200, skip_steps=50):
    current, all_warnings, steady_warnings, observed = 0, 0, 0, set()
    for line in raw.decode(errors='replace').splitlines():
        match = STEP.search(ANSI.sub('', line))
        if match:
            step = int(match.group(1))
            current = max(current, step)
            observed.add(step)
        if 'memory allocation failed with OOM' in line:
            all_warnings += 1
            if skip_steps <= current <= steps:
                steady_warnings += 1
    if observed != set(range(1, steps+1)):
        raise ValueError('sustained memory review requires the actual complete update log')
    if steady_warnings:
        raise ValueError('late allocator retries require a measured smaller-batch comparison')
    return {'status':'PASS_SUSTAINED_MEMORY_REVIEW', 'steps':steps, 'skip_steps':skip_steps,
            'allocation_retry_warnings_total':all_warnings,
            'allocation_retry_warnings_steady':steady_warnings,
            'requires_smaller_batch_comparison':False}
