"""Read training budgets independently from immutable data capacity contracts."""
from pathlib import Path

import yaml


def screen_budget(root: Path, stream_capacity_budget: int) -> int:
    path = root / 'pretrain/configs/research_budget.yaml'
    value = int(yaml.safe_load(path.read_text())['screen_budget_tokens']) if path.is_file() else stream_capacity_budget
    if not 0 < value <= stream_capacity_budget:
        raise ValueError('training budget must be positive and within materialized stream capacity')
    return value
