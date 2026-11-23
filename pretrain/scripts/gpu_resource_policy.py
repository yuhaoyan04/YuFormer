"""Fail-closed GPU ownership limits for the experiment scheduler and queues."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def visible_gpu_indices() -> set[int] | None:
    """Return the physical indices exposed by nvidia-smi, or None if unknown."""
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if probe.returncode:
        return None
    try:
        return {int(line.strip()) for line in probe.stdout.splitlines() if line.strip()}
    except ValueError:
        return None


def resource_policy(health: Path) -> dict:
    path = health / 'gpu-resource-policy.json'
    if not path.exists():
        raise ValueError('GPU resource policy is missing; refusing experiment admission')
    value = json.loads(path.read_text())
    gpus = value.get('allowed_gpu_indices')
    width = value.get('max_concurrent_jobs')
    if (value.get('status') != 'ACTIVE' or not isinstance(gpus, list) or not gpus
            or any(type(gpu) is not int or not 0 <= gpu < 8 for gpu in gpus)
            or len(gpus) != len(set(gpus)) or type(width) is not int or width != len(gpus)
            or type(value.get('allow_busy_gpu_sharing')) is not bool):
        raise ValueError('Invalid GPU resource policy; refusing experiment admission')
    available = visible_gpu_indices()
    if available is None:
        raise ValueError('Cannot verify GPU resource policy against live hardware; refusing experiment admission')
    unavailable = sorted(set(gpus) - available)
    if unavailable:
        raise ValueError(
            f'GPU resource policy references unavailable device indices {unavailable}; '
            f'live inventory is {sorted(available)}'
        )
    return value


def allowed_gpu_indices(health: Path) -> set[int]:
    return set(resource_policy(health)['allowed_gpu_indices'])
