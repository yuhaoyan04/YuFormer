#!/usr/bin/env python3
"""Collect a completed TorchTitan run into a fail-closed result contract."""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


REQUIRED_TAG_SUFFIXES = (
    "loss_metrics/global_avg_loss",
    "validation_metrics/loss",
    "throughput(tps)",
    "mfu(%)",
)
EXPECTED_VALIDATION_DOMAINS = {
    "academic", "books", "chinese", "code", "encyclopedia",
    "math", "multilingual", "qa", "web",
}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def scalar_series(event_files: list[Path]) -> dict[str, list[dict[str, float | int]]]:
    merged: dict[str, dict[int, tuple[float, float]]] = {}
    for event_file in event_files:
        accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                old = merged.setdefault(tag, {}).get(event.step)
                if old is None or event.wall_time >= old[0]:
                    merged[tag][event.step] = (event.wall_time, event.value)
    return {
        tag: [{"step": step, "value": value, "wall_time": wall_time}
              for step, (wall_time, value) in sorted(by_step.items())]
        for tag, by_step in sorted(merged.items())
    }


def resolve_tag(series: dict[str, list[dict[str, Any]]], suffix: str) -> str | None:
    # TorchTitan emits both training ``throughput(tps)`` and
    # ``validation_metrics/throughput(tps)``.  The canonical exact tag must
    # win before trying namespaced suffixes; otherwise a valid run is rejected
    # as ambiguous merely because validation throughput is also present.
    if suffix in series:
        return suffix
    matches = [tag for tag in series if tag.endswith("/" + suffix)]
    return matches[0] if len(matches) == 1 else None


def active_event_files(run_dir: Path) -> list[Path]:
    attempts = run_dir / "attempts"
    return sorted(
        path for path in run_dir.glob("**/events.out.tfevents.*")
        if attempts not in path.parents
    )


def attempt_event_lineage(run_dir: Path, attempt: dict[str, Any]):
    """Select ancestor events only through checkpoints inherited by this attempt."""
    groups = [(active_event_files(run_dir), None)]
    current, seen, upper_bound = attempt, set(), None
    while current.get("parent_attempt_id"):
        current_id = current["attempt_id"]
        if current_id in seen:
            raise ValueError("cycle in attempt event lineage")
        seen.add(current_id)
        resume_step = int(current.get("resumed_from_step", 0))
        upper_bound = resume_step if upper_bound is None else min(upper_bound, resume_step)
        archive = run_dir / "attempts" / f"superseded-before-{current_id}" / "events"
        groups.append((sorted(archive.glob("**/events.out.tfevents.*")), upper_bound))
        parent_path = run_dir / "attempts" / f"{current['parent_attempt_id']}.json"
        if not parent_path.is_file():
            break
        current = json.loads(parent_path.read_text(encoding="utf-8"))
        if current.get("identity_sha256") != attempt.get("identity_sha256"):
            raise ValueError("attempt event lineage crosses run identities")
    return list(reversed(groups))


def lineage_series(run_dir: Path, attempt: dict[str, Any]):
    merged: dict[str, dict[int, dict]] = {}
    files = []
    for paths, upper_bound in attempt_event_lineage(run_dir, attempt):
        files.extend(paths)
        for tag, points in scalar_series(paths).items():
            for point in points:
                if upper_bound is None or int(point["step"]) <= upper_bound:
                    merged.setdefault(tag, {})[int(point["step"])] = point
    return files, {tag: [points[step] for step in sorted(points)]
                   for tag, points in merged.items()}


def summarize(run_dir: Path) -> dict[str, Any]:
    identity_path = run_dir / "run_identity.json"
    config_path = run_dir / "resolved_config.yaml"
    completion_path = run_dir / "training_complete.json"
    attempt_path = run_dir / "current_attempt.json"
    if (not identity_path.is_file() or not config_path.is_file()
            or not completion_path.is_file() or not attempt_path.is_file()):
        raise FileNotFoundError(
            "run identity, resolved config, current attempt, and training completion are required"
        )
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    context_path = run_dir / "execution_context.json"
    execution_context = (
        json.loads(context_path.read_text(encoding="utf-8"))
        if context_path.is_file() else {"schema_version": 1, "loss_only": False,
                                        "performance_metrics_excluded": []}
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected_steps = int(config["train"]["expected_iters"])
    tokens_per_step = (int(config["data"]["seq_len"])
                       * int(config["train"]["micro_batch_size"])
                       * int(config["train"]["grad_accum"])
                       * int(config["parallel"]["world_size"]))
    trained_tokens = expected_steps * tokens_per_step
    accounting_path = run_dir / "training_accounting.json"
    accounting = (json.loads(accounting_path.read_text(encoding="utf-8"))
                  if accounting_path.is_file() else {})
    event_files, series = lineage_series(run_dir, attempt)
    resolved = {suffix: resolve_tag(series, suffix) for suffix in REQUIRED_TAG_SUFFIXES}
    issues: list[str] = []
    if int(identity.get("runtime_state_schema_version", 1)) >= 2:
        if (accounting.get("status") != "PASS"
                or accounting.get("attempt_id") != attempt.get("attempt_id")
                or accounting.get("step") != expected_steps):
            issues.append("runtime_accounting_missing_or_unbound")
        if not accounting.get("topology_reshard", False):
            if accounting.get("global_tokens_seen") != trained_tokens:
                issues.append("runtime_token_count_mismatch")
            if accounting.get("consumed_samples") != trained_tokens // int(config["data"]["seq_len"]):
                issues.append("runtime_sample_cursor_mismatch")
    if completion.get("status") != "PASS" or int(completion.get("returncode", -1)) != 0:
        issues.append("training_process_not_successful")
    if completion.get("identity_sha256") != identity.get("identity_sha256"):
        issues.append("completion_identity_mismatch")
    if (not attempt.get("attempt_id")
            or completion.get("attempt_id") != attempt.get("attempt_id")
            or attempt.get("identity_sha256") != identity.get("identity_sha256")):
        issues.append("completion_attempt_mismatch")
    if (context_path.is_file()
            and execution_context.get("attempt_id") != attempt.get("attempt_id")):
        issues.append("execution_context_attempt_mismatch")
    if not event_files:
        issues.append("tensorboard_event_files_missing")
    for suffix in REQUIRED_TAG_SUFFIXES:
        tag = resolved[suffix]
        if tag is None:
            issues.append(f"metric_missing_or_ambiguous:{suffix}")
        elif any(not math.isfinite(float(point["value"])) for point in series[tag]):
            issues.append(f"non_finite_metric:{suffix}")
    train_tag = resolved["loss_metrics/global_avg_loss"]
    final_step = max((int(point["step"]) for point in series.get(train_tag or "", [])), default=-1)
    log_frequency = int(config["train"].get("log_every", 1))
    minimum_final_log = expected_steps - (expected_steps % log_frequency)
    if final_step < minimum_final_log:
        issues.append(f"incomplete_metrics:{final_step}<{minimum_final_log}")

    metrics: dict[str, Any] = {}
    selected_series: dict[str, list[dict[str, float | int]]] = {}
    for suffix, tag in resolved.items():
        points = series.get(tag or "", [])
        if points:
            values = [float(point["value"]) for point in points]
            metrics[suffix] = {
                "tag": tag, "count": len(points), "first_step": int(points[0]["step"]),
                "last_step": int(points[-1]["step"]), "first": values[0], "last": values[-1],
                "min": min(values), "max": max(values),
            }
            selected_series[suffix] = [
                {"step": int(point["step"]), "value": float(point["value"])}
                for point in points
            ]
    for tag, points in series.items():
        marker = "validation_metrics/domain/"
        if marker not in tag or not tag.endswith("/loss") or not points:
            continue
        canonical = tag[tag.index(marker):]
        values = [float(point["value"]) for point in points]
        if any(not math.isfinite(value) for value in values):
            issues.append(f"non_finite_metric:{canonical}")
        metrics[canonical] = {
            "tag": tag, "count": len(points), "first_step": int(points[0]["step"]),
            "last_step": int(points[-1]["step"]), "first": values[0], "last": values[-1],
            "min": min(values), "max": max(values),
        }
        selected_series[canonical] = [
            {"step": int(point["step"]), "value": float(point["value"])}
            for point in points
        ]
    domain_metrics = [key for key in metrics if key.startswith("validation_metrics/domain/")]
    observed_domains = {
        key.removeprefix("validation_metrics/domain/").removesuffix("/loss")
        for key in domain_metrics
    }
    if observed_domains != EXPECTED_VALIDATION_DOMAINS:
        issues.append(
            "validation_domain_roster:"
            + ",".join(sorted(observed_domains))
        )
    aggregate_validation = metrics.get("validation_metrics/loss") or {}
    if int(aggregate_validation.get("last_step", -1)) != expected_steps:
        issues.append(
            f"final_validation_missing:{aggregate_validation.get('last_step')}!={expected_steps}"
        )
    nonfinal_domains = sorted(
        key for key in domain_metrics
        if int((metrics.get(key) or {}).get("last_step", -1)) != expected_steps
    )
    if nonfinal_domains:
        issues.append("final_domain_validation_missing:" + ",".join(nonfinal_domains))
    return {
        "schema_version": 1, "status": "PASS" if not issues else "FAIL",
        "observed_at": datetime.now(timezone.utc).isoformat(), "run_dir": str(run_dir),
        "run_name": identity.get("run_name"), "identity_sha256": identity.get("identity_sha256"),
        "attempt_id": attempt.get("attempt_id"),
        "execution_context": execution_context,
        "runtime_accounting": accounting,
        "expected_steps": expected_steps, "final_train_step": final_step,
        "trained_steps": expected_steps, "trained_tokens": trained_tokens,
        "configured_budget_shortfall_tokens": (
            int(config["train"]["budget_tokens"]) - trained_tokens
        ),
        "hyperparameters": {
            "world_size": int(config["parallel"]["world_size"]),
            "model_preset": config["model"]["preset"],
            "learning_rate": float(config["optim"]["lr"]),
            "budget_tokens": int(config["train"]["budget_tokens"]),
            "tokens_per_step": tokens_per_step,
            "trained_steps": expected_steps,
            "trained_tokens": trained_tokens,
        },
        "event_files": [str(path) for path in event_files], "issues": issues,
        "metrics": metrics, "series": selected_series,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    result = summarize(run_dir)
    output = args.output or (run_dir / "result.json")
    atomic_json(output, result)
    print(json.dumps({"status": result["status"], "run_name": result["run_name"],
                      "issues": result["issues"]}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
