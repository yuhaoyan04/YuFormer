"""Fail-closed diagnostics for grouped RLVR rollouts."""
from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Iterable


def _quantile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: list[float]) -> dict[str, float | None]:
    return {
        "min": min(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "p50": _quantile(values, 0.5),
        "p95": _quantile(values, 0.95),
        "max": max(values) if values else None,
    }


def audit_rollouts(
    records: Iterable[dict[str, Any]],
    *,
    expected_group_size: int,
    max_response_tokens: int,
    variance_epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Validate one rollout batch and identify groups eligible for GRPO updates.

    Every record requires ``prompt_id``, globally unique ``rollout_id``, finite
    ``reward``, and ``response_tokens``. Optional trainer metrics are aggregated
    when present: ``kl``, ``importance_ratio``, ``clip_low``, ``clip_high``, and
    ``truncated``.
    """
    if expected_group_size < 2:
        raise ValueError("expected_group_size must be at least 2")
    if max_response_tokens <= 0:
        raise ValueError("max_response_tokens must be positive")
    if variance_epsilon < 0:
        raise ValueError("variance_epsilon must be non-negative")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rollout_ids: set[str] = set()
    errors: list[dict[str, Any]] = []
    optional: dict[str, list[float]] = defaultdict(list)
    lengths: list[float] = []
    rewards: list[float] = []
    truncated_count = 0
    rows = 0

    for row_number, record in enumerate(records, start=1):
        rows += 1
        row_errors: list[str] = []
        if not isinstance(record, dict):
            errors.append({"row": row_number, "errors": ["record must be an object"]})
            continue
        prompt_id = record.get("prompt_id")
        rollout_id = record.get("rollout_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            row_errors.append("prompt_id must be a non-empty string")
        if not isinstance(rollout_id, str) or not rollout_id:
            row_errors.append("rollout_id must be a non-empty string")
        elif rollout_id in rollout_ids:
            row_errors.append("rollout_id must be globally unique")
        else:
            rollout_ids.add(rollout_id)

        reward = record.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            row_errors.append("reward must be numeric")
        elif not math.isfinite(float(reward)):
            row_errors.append("reward must be finite")
        length = record.get("response_tokens")
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            row_errors.append("response_tokens must be a non-negative integer")
        elif length > max_response_tokens:
            row_errors.append("response_tokens exceeds max_response_tokens")
        truncated = record.get("truncated", False)
        if not isinstance(truncated, bool):
            row_errors.append("truncated must be boolean")

        for name in ("kl", "importance_ratio"):
            value = record.get(name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                row_errors.append(f"{name} must be finite numeric when present")
        for name in ("clip_low", "clip_high"):
            value = record.get(name)
            if value is not None and not isinstance(value, bool):
                row_errors.append(f"{name} must be boolean when present")

        if row_errors:
            errors.append({"row": row_number, "rollout_id": rollout_id, "errors": row_errors})
            continue
        groups[prompt_id].append(record)
        reward_value = float(reward)
        rewards.append(reward_value)
        lengths.append(float(length))
        truncated_count += int(truncated)
        for name in ("kl", "importance_ratio"):
            if record.get(name) is not None:
                optional[name].append(float(record[name]))
        for name in ("clip_low", "clip_high"):
            if record.get(name) is not None:
                optional[name].append(float(record[name]))

    group_diagnostics: list[dict[str, Any]] = []
    eligible_prompt_ids: list[str] = []
    zero_variance = 0
    all_zero = 0
    all_one = 0
    for prompt_id in sorted(groups):
        group = groups[prompt_id]
        if len(group) != expected_group_size:
            errors.append(
                {
                    "prompt_id": prompt_id,
                    "errors": [
                        f"group has {len(group)} rollouts; expected {expected_group_size}"
                    ],
                }
            )
        group_rewards = [float(record["reward"]) for record in group]
        variance = statistics.pvariance(group_rewards)
        is_zero_variance = variance <= variance_epsilon
        zero_variance += int(is_zero_variance)
        all_zero += int(all(value == 0.0 for value in group_rewards))
        all_one += int(all(value == 1.0 for value in group_rewards))
        if not is_zero_variance and len(group) == expected_group_size:
            eligible_prompt_ids.append(prompt_id)
        group_diagnostics.append(
            {
                "prompt_id": prompt_id,
                "rollouts": len(group),
                "reward_mean": statistics.fmean(group_rewards),
                "reward_variance": variance,
                "zero_variance": is_zero_variance,
            }
        )

    if rows == 0:
        errors.append({"row": None, "errors": ["rollout batch is empty"]})
    if groups and not eligible_prompt_ids:
        errors.append({"row": None, "errors": ["batch has no non-zero-variance complete group"]})

    valid_rows = sum(len(group) for group in groups.values())
    group_count = len(groups)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "rows": rows,
        "valid_rows": valid_rows,
        "groups": group_count,
        "expected_group_size": expected_group_size,
        "eligible_groups": len(eligible_prompt_ids),
        "eligible_prompt_ids": eligible_prompt_ids,
        "zero_variance_groups": zero_variance,
        "zero_variance_fraction": zero_variance / group_count if group_count else None,
        "all_zero_groups": all_zero,
        "all_one_groups": all_one,
        "reward": _summary(rewards),
        "response_tokens": _summary(lengths),
        "truncate_fraction": truncated_count / valid_rows if valid_rows else None,
        "group_diagnostics": group_diagnostics,
        "errors": errors,
    }
    for name in ("kl", "importance_ratio"):
        result[name] = _summary(optional[name])
        result[f"{name}_coverage"] = len(optional[name]) / valid_rows if valid_rows else None
    for name in ("clip_low", "clip_high"):
        result[f"{name}_fraction"] = (
            statistics.fmean(optional[name]) if optional[name] else None
        )
        result[f"{name}_coverage"] = len(optional[name]) / valid_rows if valid_rows else None
    return result
