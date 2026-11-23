"""Check both CUDA allocator diagnostics and TorchTitan's counter summaries."""
import re

RETRY_COUNT = re.compile(r'\b([0-9]+) CUDA memory allocation retries\b')


def require_no_allocator_retries(raw):
    cpp_failures, summary_warnings, maximum_reported = 0, 0, 0
    for line in raw.decode(errors='replace').splitlines():
        if 'memory allocation failed with OOM' in line:
            cpp_failures += 1
        match = RETRY_COUNT.search(line)
        if match:
            count = int(match.group(1))
            maximum_reported = max(maximum_reported, count)
            if count > 0:
                summary_warnings += 1
    if cpp_failures or summary_warnings:
        raise ValueError('actual allocator retry detected in CUDA diagnostics or TorchTitan counter summaries')
    return {'status':'PASS_NO_ALLOCATOR_RETRIES', 'cuda_failure_messages':cpp_failures,
            'positive_counter_messages':summary_warnings, 'maximum_reported_retries':maximum_reported,
            'scope':'complete actual log including initialization and checkpoint saves'}
