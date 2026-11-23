from __future__ import annotations

import argparse
import atexit
import json
import logging
import multiprocessing as mp
import os
import socket
import sys
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from . import decontam as dct
from . import detectors
from .dedup_common import load_dedup_config
from .dedup_exact import run_s2
from .dedup_minhash import run_s3
from .dedup_v2 import run_ndv2, run_ndv2_refresh
from .filter import process_shard_s1
from .ingest import process_shard_s0
from .length_bucket import run_s5
from . import materialize as mat
from .profiles import DEFAULT_PROFILES_PATH, load_profiles
from .score import (
    DEFAULT_SCORER_CONFIG,
    build_compact_quantile_tables,
    build_quantile_tables,
    load_quantile_tables,
    load_scorer_config,
    load_quality_model,
    merge_s4_global_summary,
    process_shard_quantile,
    process_shard_score,
    resolve_model_path,
)
from .score_bert import bert_tasks, resolve_bert_model_dir, run_stage_bert

log = logging.getLogger("yuformer.scale")

PROGRESS_EVERY = 50
# B1 修复：worker 被 SIGKILL/OOM 杀死时 mp.Pool.imap_unordered 会永久阻塞且无任何报错。
# 停滞守卫：超过该时长无分片完成则硬退出（os._exit，不依赖可能已损坏的 Pool），
# 断点幂等保证重启无损——把"无声僵死"变成"干净退出可重启"。
STALL_TIMEOUT_SEC = 14400  # 2026-09-22: 1800→4h——s3_verify 单桶（web 域 ~470 万篇候选）实测 >30min，旧值会 livelock（每 30min 杀一次、在途桶重做）
STAGE_LEASE_HEARTBEAT_SEC = 30
STAGE_LEASE_STALE_SEC = 24 * 60 * 60

_WORKER: dict[str, Any] = {}


class _StageLease:
    """Cross-pod single-writer lease backed by an atomic NFS mkdir.

    A plain pidfile only proves liveness inside the current PID namespace.  All
    heavy stages write shared files, so two pods must contend on an operation
    that the shared filesystem serializes.  NFS directory creation provides
    that primitive; the heartbeat permits recovery after a dead pod without
    treating a multi-day healthy job as stale.
    """

    def __init__(self, path: Path, token: str, owner_pid: int) -> None:
        self.path = path
        self.token = token
        self.owner_pid = owner_pid
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def _heartbeat(self) -> None:
        heartbeat = self.path / f"heartbeat-{self.token}"
        failures = 0
        while not self.stop.wait(STAGE_LEASE_HEARTBEAT_SEC):
            try:
                owner = json.loads((self.path / "owner.json").read_text("utf-8"))
                if owner.get("token") != self.token:
                    log.error("stage lease ownership lost: %s", self.path)
                    os._exit(74)
                heartbeat.touch(exist_ok=True)
                failures = 0
            except FileNotFoundError:
                log.error("stage lease disappeared: %s", self.path)
                os._exit(74)
            except (OSError, ValueError):
                failures += 1
                log.exception("stage lease heartbeat failed: %s", self.path)
                if failures >= 3:
                    os._exit(74)

    def release(self) -> None:
        if os.getpid() != self.owner_pid:
            return
        self.stop.set()
        # Do not delete/reuse the same NFS directory immediately.  A client may
        # retain cached owner metadata and accidentally delete a successor's
        # lease.  Token-scoped release markers are harmless if a late atexit
        # callback writes into a successor directory: the successor ignores a
        # marker whose token is not its own.
        try:
            (self.path / f"released-{self.token}").touch(exist_ok=True)
        except OSError:
            pass


def _read_stage_lease(path: Path) -> tuple[dict[str, Any], float]:
    try:
        owner = json.loads((path / "owner.json").read_text("utf-8"))
    except (OSError, ValueError):
        owner = {}
    try:
        token = str(owner.get("token") or "unknown")
        heartbeat = path / f"heartbeat-{token}"
        age = time.time() - (heartbeat if heartbeat.exists() else path).stat().st_mtime
    except OSError:
        age = 0.0
    return owner, max(0.0, age)


def _acquire_stage_lease(out_root: Path, stage: str) -> _StageLease | None:
    """Acquire an append-only shared lease, or return ``None`` if active.

    Some NFS deployments allow creating children in a root-squashed directory
    but reject renaming that directory from its parent.  Therefore the stable
    lock root is never renamed or deleted.  Successive owners contend on the
    same deterministic generation directory; only one atomic mkdir can win.
    Legacy v1 owner files in the root are treated as generation zero.
    """
    lock_root = out_root / f".driver_{stage}.lock"
    token = uuid.uuid4().hex
    owner = {
        "schema_version": 1,
        "stage": stage,
        "token": token,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "started_unix": time.time(),
    }
    try:
        lock_root.mkdir(exist_ok=True)
    except OSError:
        log.exception("stage %s: failed to create shared lease root %s", stage, lock_root)
        return None

    for _attempt in range(8):
        generations: list[tuple[int, Path]] = []
        try:
            for path in lock_root.glob("generation-*"):
                if not path.is_dir():
                    continue
                try:
                    generations.append((int(path.name.removeprefix("generation-")), path))
                except ValueError:
                    continue
        except OSError:
            log.exception("stage %s: failed to inspect shared lease %s", stage, lock_root)
            return None

        if generations:
            current_number, current_path = max(generations)
        elif (lock_root / "owner.json").is_file():
            current_number, current_path = 0, lock_root
        else:
            current_number, current_path = 0, None

        if current_path is not None:
            current, age = _read_stage_lease(current_path)
            current_token = str(current.get("token") or "unknown")
            released = (current_path / f"released-{current_token}").is_file()
            if not released and age <= STAGE_LEASE_STALE_SEC:
                log.warning(
                    "stage %s: shared lease held by host=%s pid=%s age=%.0fs",
                    stage,
                    current.get("hostname", "unknown"),
                    current.get("pid", "unknown"),
                    age,
                )
                return None

        generation_path = lock_root / f"generation-{current_number + 1:020d}"
        try:
            generation_path.mkdir()
        except FileExistsError:
            continue
        except OSError:
            log.exception("stage %s: failed to acquire shared lease %s", stage, generation_path)
            return None
        try:
            (generation_path / "owner.json").write_text(
                json.dumps(owner, ensure_ascii=False, sort_keys=True), encoding="utf-8"
            )
            (generation_path / f"heartbeat-{token}").touch()
        except BaseException:
            # Leave an incomplete, fresh generation as a fail-closed lease.  A
            # later attempt may advance after its stale timeout.
            raise
        lease = _StageLease(generation_path, token, os.getpid())
        atexit.register(lease.release)
        return lease
    return None


def _install_stall_guard(stage: str) -> Any:
    """返回 (beat, stop)。stage 结束必须调 stop()——否则旧守卫线程会继续计时而误杀后续 stage。"""
    state = {"last": time.monotonic()}
    stop = threading.Event()

    def beat() -> None:
        state["last"] = time.monotonic()

    def watch() -> None:
        while not stop.wait(60):
            stall = time.monotonic() - state["last"]
            if stall > STALL_TIMEOUT_SEC:
                log.error(
                    "progress[%s]: stall guard tripped — no shard completed in %ds "
                    "(likely worker OOM-kill / pool hang). Hard exit for clean restart (resume-safe).",
                    stage, int(stall),
                )
                os._exit(3)

    threading.Thread(target=watch, daemon=True).start()
    return beat, stop


def _init_worker(profiles_path: str, data_root: str, out_root: str) -> None:
    _WORKER["profiles"] = load_profiles(profiles_path)
    _WORKER["data_root"] = Path(data_root)
    _WORKER["out_root"] = Path(out_root)


def _init_worker_s4(scorer_cfg_path: str, out_root: str) -> None:
    cfg = load_scorer_config(scorer_cfg_path)
    _WORKER["scorer_cfg"] = cfg
    _WORKER["out_root"] = Path(out_root)
    _WORKER["scorer_model"] = load_quality_model(cfg)


def _run_shard(task: tuple[str, str, str]) -> dict[str, Any]:
    stage, source, relpath = task
    profiles = _WORKER["profiles"]
    data_root = _WORKER["data_root"]
    out_root = _WORKER["out_root"]
    profile = profiles[source]
    try:
        if stage == "s0":
            return process_shard_s0(data_root / source / relpath, out_root, source, relpath, profile)
        s0_path = out_root / "s0" / source / (relpath + ".parquet")
        return process_shard_s1(s0_path, out_root, source, relpath, profile["domain"])
    except Exception as exc:
        return {
            "stage": stage,
            "source": source,
            "relpath": relpath,
            "shard_fatal": f"{type(exc).__name__}: {exc}",
        }


def _run_shard_s4(task: tuple[str, str, str]) -> dict[str, Any]:
    stage, source, relpath = task
    out_root = _WORKER["out_root"]
    cfg = _WORKER["scorer_cfg"]
    try:
        if stage == "s4_score":
            s1_path = out_root / "s1" / source / (relpath + ".parquet")
            return process_shard_score(
                s1_path, out_root, source, relpath, cfg, model=_WORKER["scorer_model"]
            )
        return process_shard_quantile(
            out_root, source, relpath, cfg, tables=_WORKER.get("quantile_tables")
        )
    except Exception as exc:
        return {
            "stage": stage,
            "source": source,
            "relpath": relpath,
            "shard_fatal": f"{type(exc).__name__}: {exc}",
        }


def _shard_tasks_s0(profiles: dict[str, dict[str, Any]], data_root: Path) -> list[tuple[str, str, str]]:
    tasks = []
    for source, profile in profiles.items():
        src_dir = data_root / source
        if not src_dir.is_dir():
            if profile.get("optional"):
                log.warning("source dir missing (optional, skipped): %s", source)
                continue
            raise FileNotFoundError(f"source dir missing: {src_dir}")
        exclude = set(profile.get("exclude_dirs", []))
        exts = tuple(profile["extensions"])
        for path in sorted(src_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(src_dir).as_posix()
            if rel.split("/", 1)[0] in exclude or "/.cache/" in f"/{rel}":
                continue
            if rel.endswith(exts):
                tasks.append(("s0", source, rel))
    return sorted(tasks, key=lambda t: (t[1], t[2]))


def _shard_tasks_s1(profiles: dict[str, dict[str, Any]], out_root: Path) -> list[tuple[str, str, str]]:
    tasks = []
    s0_root = out_root / "s0"
    for source in profiles:
        src_dir = s0_root / source
        if not src_dir.is_dir():
            continue
        for path in sorted(src_dir.rglob("*.parquet")):
            rel = path.relative_to(src_dir).as_posix()
            tasks.append(("s1", source, rel[: -len(".parquet")]))
    return sorted(tasks, key=lambda t: (t[1], t[2]))


def _imap_tolerant(
    pool: mp.pool.Pool, tasks: list[tuple[str, str, str]], runner: Any = _run_shard
) -> Iterator[dict[str, Any]]:
    it = pool.imap_unordered(runner, tasks, chunksize=1)
    idx = 0
    while True:
        try:
            yield next(it)
        except StopIteration:
            return
        except Exception as exc:
            idx += 1
            log.error("pool-level shard failure (%d): %s", idx, exc)
            yield {"shard_fatal": f"pool:{type(exc).__name__}: {exc}"}


def run_stage(
    stage: str,
    tasks: list[tuple[str, str, str]],
    profiles_path: str,
    data_root: Path,
    out_root: Path,
    workers: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    started = time.monotonic()
    done = 0
    rows = 0
    decisions: Counter[str] = Counter()
    beat, guard_stop = _install_stall_guard(stage)
    try:
        with mp.Pool(processes=workers, initializer=_init_worker, initargs=(profiles_path, str(data_root), str(out_root))) as pool:
            for summary in _imap_tolerant(pool, tasks):
                summaries.append(summary)
                done += 1
                beat()
                rows += int(summary.get("in_rows") or 0)
                dec = summary.get("decisions") or {}
                for k, v in dec.items():
                    decisions[k] += v
                if done % PROGRESS_EVERY == 0 or done == len(tasks):
                    elapsed = time.monotonic() - started
                    rate = done / max(elapsed, 1e-9)
                    log.info(
                        "progress[%s]: %d/%d shards, %.2f shards/s, %.0f docs/s, decisions so far: %s",
                        stage, done, len(tasks), rate, rows / max(elapsed, 1e-9), dict(decisions),
                    )
    finally:
        guard_stop.set()
    return summaries


def _shard_tasks_s4(out_root: Path, stage: str) -> list[tuple[str, str, str]]:
    tasks = []
    root = out_root / ("s1" if stage == "s4_score" else "s4_scored")
    for src_dir in sorted(root.iterdir()) if root.is_dir() else []:
        if not src_dir.is_dir():
            continue
        for path in sorted(src_dir.rglob("*.parquet")):
            rel = path.relative_to(src_dir).as_posix()
            tasks.append((stage, src_dir.name, rel[: -len(".parquet")]))
    return sorted(tasks, key=lambda t: (t[1], t[2]))


def run_stage_s4(
    stage: str,
    tasks: list[tuple[str, str, str]],
    scorer_cfg_path: str,
    out_root: Path,
    workers: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    started = time.monotonic()
    done = 0
    rows = 0
    beat, guard_stop = _install_stall_guard(stage)
    if stage == "s4_quantile":
        # E-32/E-33：分位表 web 域 22GB，per-worker 实载/mmap 全量表在 160 并发下
        # 都会 OOM 或随机缺页拖尾；父进程构建 compact 表（rank 等距采样 2^25 点/域，
        # 物理 ~150MB）后 fork，全 worker COW 共享。worker 侧 tables=None 时仍会
        # 自行 load 全量表（直接 API 调用/测试的兜底路径，勿用于大规模并发）。
        _WORKER["quantile_tables"] = build_compact_quantile_tables(load_quantile_tables(out_root))
        log.info(
            "stage %s: compact quantile tables ready (%s)",
            stage, {k: len(v) for k, v in _WORKER["quantile_tables"].items()},
        )
    try:
        with mp.Pool(processes=workers, initializer=_init_worker_s4, initargs=(scorer_cfg_path, str(out_root))) as pool:
            for summary in _imap_tolerant(pool, tasks, runner=_run_shard_s4):
                summaries.append(summary)
                done += 1
                beat()
                rows += int(summary.get("in_rows") or 0)
                if done % PROGRESS_EVERY == 0 or done == len(tasks):
                    elapsed = time.monotonic() - started
                    rate = done / max(elapsed, 1e-9)
                    log.info(
                        "progress[%s]: %d/%d shards, %.2f shards/s, %.0f docs/s",
                        stage, done, len(tasks), rate, rows / max(elapsed, 1e-9),
                    )
    finally:
        guard_stop.set()
    return summaries


def write_global_summary(out_root: Path, stage: str, summaries: list[dict[str, Any]]) -> Path:
    per_source: dict[str, Counter[str]] = {}
    per_domain: dict[str, Counter[str]] = {}
    reason_totals: Counter[str] = Counter()
    decision_totals: Counter[str] = Counter()
    profiles = _WORKER.get("profiles") or {}
    total = Counter()
    for s in summaries:
        src = s.get("source", "?")
        acc = per_source.setdefault(src, Counter())
        for key in ("in_rows", "out_rows", "audit_rows", "error_rows", "input_bytes", "output_bytes"):
            acc[key] += int(s.get(key) or 0)
        acc["elapsed_sec"] += float(s.get("elapsed_sec") or 0.0)
        decision_totals.update(s.get("decisions") or {})
        reason_totals.update(s.get("reasons") or {})
        for key, v in (s.get("decisions") or {}).items():
            total[f"decisions.{key}"] += v
        dom = "?"
        if src in profiles:
            dom = profiles[src]["domain"]
        dacc = per_domain.setdefault(dom, Counter())
        for key in ("in_rows", "out_rows", "audit_rows"):
            dacc[key] += int(s.get(key) or 0)
    payload = {
        "stage": stage,
        "shards": len(summaries),
        "shard_errors": sum(1 for s in summaries if s.get("errors") or s.get("shard_fatal")),
        "totals": dict(total),
        "decisions": dict(decision_totals),
        "reasons": dict(reason_totals),
        "per_source": {k: dict(v) for k, v in sorted(per_source.items())},
        "per_domain": {k: dict(v) for k, v in sorted(per_domain.items())},
    }
    path = out_root / f"stage_{stage}_global_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8")
    return path


def fail_fast_checks(data_root: Path, out_root: Path, profiles_path: str) -> None:
    import pyarrow  # noqa: F401
    import zstandard  # noqa: F401

    if not data_root.is_dir():
        raise SystemExit(f"data-root not found: {data_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    profiles = load_profiles(profiles_path)
    missing = [
        name for name, p in profiles.items()
        if not (data_root / name).is_dir() and not p.get("optional")
    ]
    if missing:
        raise SystemExit(f"source dirs missing on disk: {missing}")
    lid = detectors.get_lang_detector()
    if lid is None:
        log.warning(
            "fasttext lid model not available (env %s); lang detection skipped, lang='unk'",
            detectors._LID_MODEL_ENV,
        )


def fail_fast_checks_s4(out_root: Path, scorer_cfg_path: str, need_tables: bool) -> dict[str, Any]:
    import fasttext  # noqa: F401

    if not (out_root / "s1").is_dir():
        raise SystemExit(f"s4 requires s1 outputs: {out_root / 's1'} missing (run --stage s1 first)")
    cfg = load_scorer_config(scorer_cfg_path)
    resolve_model_path(cfg)
    if need_tables and not (out_root / "s4_quantiles" / "tables" / "_build.summary.json").is_file():
        raise SystemExit(
            "s4 quantile pass requires pass1 aggregate tables "
            f"({out_root / 's4_quantiles' / 'tables' / '_build.summary.json'}); "
            "run --stage s4 (full) or score all shards first, then build tables"
        )
    return cfg


def run_s4(args: argparse.Namespace, out_root: Path) -> None:
    scorer_cfg_path = str(args.scorer)
    cfg = fail_fast_checks_s4(out_root, scorer_cfg_path, need_tables=False)

    tasks = _shard_tasks_s4(out_root, "s4_score")
    if args.sources:
        tasks = [t for t in tasks if t[1] in set(args.sources)]
    if args.limit:
        tasks = tasks[: args.limit]
    log.info("stage s4 score pass: %d shards", len(tasks))
    score_summaries = (
        run_stage_s4("s4_score", tasks, scorer_cfg_path, out_root, args.workers) if tasks else []
    )

    build_summary: dict[str, Any] | None = None
    quantile_summaries: list[dict[str, Any]] = []
    score_errors = sum(1 for s in score_summaries if s.get("errors") or s.get("shard_fatal"))
    if score_errors:
        # E-27 竞态修复：打分有 error 分片时禁止建分位表——否则用不完整 pairs 建出
        # 坏表且被 _build.summary.json 幂等 skip 永久生效（重启补齐打分也不会刷表）。
        log.warning(
            "stage s4: %d score shard errors——跳过分位表构建（补齐打分后重跑 --stage s4 刷新）",
            score_errors,
        )
    elif args.stage == "s4":
        build_summary = build_quantile_tables(out_root, cfg=cfg)
        log.info(
            "stage s4 quantile tables built: %d rows across %d domains",
            build_summary.get("total_rows", 0),
            len(build_summary.get("domains", {})),
        )
        rule_summary = build_quantile_tables(
            out_root, score_field="rule_composite", cfg=cfg
        )
        log.info(
            "stage s4 rule quantile tables built: %d rows across %d domains",
            rule_summary.get("total_rows", 0),
            len(rule_summary.get("domains", {})),
        )
        hybrid_summary = build_quantile_tables(
            out_root, score_field="hybrid_composite", cfg=cfg
        )
        log.info(
            "stage s4 hybrid quantile tables built: %d rows across %d domains",
            hybrid_summary.get("total_rows", 0),
            len(hybrid_summary.get("domains", {})),
        )
        hybrid2_summary = build_quantile_tables(
            out_root, score_field="hybrid_v2", cfg=cfg
        )
        log.info(
            "stage s4 hybrid_v2 (production) quantile tables built: %d rows across %d domains",
            hybrid2_summary.get("total_rows", 0),
            len(hybrid2_summary.get("domains", {})),
        )
        qtasks = _shard_tasks_s4(out_root, "s4_quantile")
        if args.sources:
            qtasks = [t for t in qtasks if t[1] in set(args.sources)]
        if args.limit:
            qtasks = qtasks[: args.limit]
        log.info("stage s4 quantile pass: %d shards", len(qtasks))
        quantile_summaries = (
            run_stage_s4("s4_quantile", qtasks, scorer_cfg_path, out_root, args.workers)
            if qtasks
            else []
        )

    path = merge_s4_global_summary(out_root, score_summaries, quantile_summaries, build_summary, cfg)
    log.info("stage s4 global summary: %s", path)
    errors = sum(1 for s in score_summaries + quantile_summaries if s.get("errors") or s.get("shard_fatal"))
    if errors:
        log.warning("stage s4 finished with %d shard errors (will retry on next run)", errors)


def run_s4_build(args: argparse.Namespace, out_root: Path) -> None:
    """Rebuild one global score distribution without rerunning inference."""
    if not (out_root / "s4_quantiles" / "pairs").is_dir():
        raise SystemExit("s4_build requires completed s4_quantiles/pairs artifacts")
    cfg = load_scorer_config(str(args.scorer))
    summary = build_quantile_tables(
        out_root,
        force=bool(args.force),
        score_field=str(args.score_field),
        cfg=cfg,
    )
    log.info(
        "stage s4_build: field=%s rows=%d domains=%d",
        args.score_field,
        int(summary.get("total_rows") or 0),
        len(summary.get("domains") or {}),
    )


def run_s4_bert(args: argparse.Namespace, out_root: Path) -> None:
    """GPU worker stage：s3 keep 池 → s4_bert bert_score。"""
    import torch  # noqa: F401

    scorer_cfg_path = str(args.scorer)
    cfg = load_scorer_config(scorer_cfg_path)
    resolve_bert_model_dir(cfg)
    tasks = bert_tasks(out_root)
    if args.sources:
        tasks = [t for t in tasks if t[1] in set(args.sources)]
    if args.limit:
        tasks = tasks[: args.limit]
    log.info("stage s4_bert: %d shards on gpus %s", len(tasks), args.gpus)
    summaries = (
        run_stage_bert(tasks, scorer_cfg_path, out_root, args.gpus,
                       _install_stall_guard("s4_bert"), log=log.info)
        if tasks else []
    )
    errors = sum(1 for s in summaries if s.get("errors") or s.get("shard_fatal"))
    total_scored = sum(int(s.get("scored_rows") or 0) for s in summaries)
    total_elapsed = sum(float(s.get("elapsed_sec") or 0.0) for s in summaries)
    payload = {
        "stage": "s4_bert",
        "shards": len(summaries),
        "shard_errors": errors,
        "scored_rows": total_scored,
        "agg_rows_per_sec": round(
            total_scored / max(total_elapsed / max(len(args.gpus), 1), 1e-9), 1
        ),
        "gpus": list(args.gpus),
        "per_source": {},
    }
    per_source: Counter[str] = Counter()
    for s in summaries:
        per_source[s.get("source", "?")] += int(s.get("scored_rows") or 0)
    payload["per_source"] = dict(sorted(per_source.items()))
    path = out_root / "stage_s4_bert_global_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8")
    log.info("stage s4_bert global summary: %s", path)
    if errors:
        log.warning("stage s4_bert finished with %d shard errors (will retry on next run)", errors)


def run_decontam(args: argparse.Namespace, out_root: Path) -> None:
    """S7 评测去污染：基准 8-gram 索引（幂等构建）→ 分片并行扫描产决策表。"""
    cfg = dct.load_decontam_config(args.decontam_config)
    cfg_path = str(args.decontam_config or dct.DEFAULT_DECONTAM_CONFIG_PATH)
    index_dir = Path(cfg["index_dir"])
    idx = dct.build_benchmark_index(cfg, index_dir)
    if idx.get("errors"):
        raise SystemExit(
            "decontam benchmark index is incomplete; refusing to scan with missing benchmark paths: "
            f"{idx.get('errors')}"
        )
    log.info(
        "decontam index ready: %d unique %d-gram shingles across %d benchmarks (%s, skipped=%s)",
        int(idx.get("n_unique_shingles") or 0), int(cfg["ngram_n"]),
        len(cfg["benchmarks"]), index_dir, idx.get("skipped", False),
    )
    try:
        tasks = dct.shard_tasks(out_root, cfg, args.sources, args.limit)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    log.info("stage decontam: %d shards (input=%s/)", len(tasks), cfg["input_stage"])
    summaries = (
        dct.run_decontam_scan(tasks, cfg_path, index_dir, args.workers,
                              _install_stall_guard("decontam"))
        if tasks else []
    )
    published, coverage = dct.collect_published_summaries(out_root, cfg)
    path = dct.write_global_summary(out_root, published, coverage=coverage, cfg=cfg)
    log.info("stage decontam global summary: %s", path)
    errors = sum(1 for s in summaries if s.get("errors") or s.get("shard_fatal"))
    if errors:
        log.warning(
            "stage decontam finished with %d shard errors — 未成功分片下次自动重试（E-16："
            "验收时首先检查 stage_decontam_global_summary.json 的 shard_errors 是否为 0）",
            errors,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="yuformer_data.scale.driver")
    parser.add_argument("--stage", choices=["s0", "s1", "s2", "s3", "s4", "s4_build", "s4_bert", "s5", "s6", "s6_val", "score", "dedup", "decontam", "ndv2", "ndv2_refresh", "all"], required=True)
    parser.add_argument("--workers", type=int, default=160)
    parser.add_argument("--data-root", default="../data/pretrain-raw")
    parser.add_argument("--out-root", default="../data/processed")
    parser.add_argument("--profiles", default=str(DEFAULT_PROFILES_PATH))
    parser.add_argument("--scorer", default=str(DEFAULT_SCORER_CONFIG))
    parser.add_argument("--dedup-config", default=None)
    parser.add_argument("--decontam-config", default=None, help="decontam 阶段配置（默认 configs/decontam.yaml）")
    parser.add_argument("--mix-config", default=None, help="s6/s6_val: mix 配比配置（如 configs/mix_m2.yaml）")
    parser.add_argument("--tokenizer", default=None, help="s5: tokenizer 目录或 tokenizer.json（默认 YUFORMER_TOKENIZER 或仓库 yf-64k-bpe）")
    parser.add_argument("--allow-e17-web", action="store_true", help="s5: web 域接受 E-17 占位（fast track，FineWeb 立场）")
    parser.add_argument("--allow-e17-code", action="store_true", help="s5: code 域接受 E-17 占位（E-31：code verify 实测 4-9 天，v1.0 用 shingle 预物化重攻）")
    parser.add_argument("--gpus", nargs="*", default=None, help="s4_bert: CUDA 设备号列表（默认全部可见 GPU，一进程一卡）")
    parser.add_argument("--sources", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0, help="max shards per stage (smoke)")
    parser.add_argument(
        "--score-field",
        choices=["composite", "rule_composite", "hybrid_composite", "hybrid_v2"],
        default="hybrid_v2",
        help="s4_build: rebuild only this score distribution",
    )
    parser.add_argument("--force", action="store_true", help="s4_build: ignore a matching build contract")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)

    # Shared operational hold used while repairing an upstream generation.
    # A deliberately launched owner may bypass it, while unattended legacy
    # orchestrators fail non-zero and cannot race into S6 with a stale mix name.
    if (
        args.stage in ("s6", "s6_val")
        and (out_root.parent / "PAUSE_S6").is_file()
        and os.environ.get("YUFORMER_ALLOW_PAUSED_S6") != "1"
    ):
        log.warning("stage %s paused by %s", args.stage, out_root.parent / "PAUSE_S6")
        return 75

    # E-41/E-99：重 stage 必须是跨 Pod 单写者。旧 pidfile 只能在当前 PID
    # namespace 验证存活，实测会放行另一 Pod 的 S5，导致 parquet/summary
    # 被不同 worker 交叉覆盖。共享 NFS 上的 mkdir 是原子竞争点，心跳用于容器
    # 死亡后的最终恢复；抢不到时返回非零，防止编排器误报成功。
    if args.stage in ("s4", "s4_build", "s5", "s6", "s6_val", "decontam", "ndv2"):
        out_root.mkdir(parents=True, exist_ok=True)
        lease = _acquire_stage_lease(out_root, args.stage)
        if lease is None:
            return 75
        # 保留 pidfile 作为运维可观测信号，不再用它做跨 Pod 互斥。
        (out_root / f".driver_{args.stage}.pid").write_text(str(os.getpid()))

    if args.stage == "s4_bert":
        if args.gpus is None:
            import torch

            args.gpus = [str(i) for i in range(torch.cuda.device_count())]
        if not args.gpus:
            raise SystemExit("s4_bert requires at least one GPU (none visible)")
        run_s4_bert(args, out_root)
        return 0

    if args.stage in ("s4", "score"):
        run_s4(args, out_root)
        return 0

    if args.stage == "s4_build":
        run_s4_build(args, out_root)
        return 0

    if args.stage == "decontam":
        run_decontam(args, out_root)
        return 0

    if args.stage in ("s5", "s6", "s6_val"):
        profiles = load_profiles(args.profiles)
        if args.sources:
            unknown = set(args.sources) - set(profiles)
            if unknown:
                raise SystemExit(f"unknown sources: {sorted(unknown)}")
        if args.stage == "s5":
            run_s5(out_root, profiles, args.workers, tokenizer=args.tokenizer,
                   sources=args.sources, limit=args.limit,
                   guard=_install_stall_guard("s5"),
                   allow_e17_web=bool(getattr(args, "allow_e17_web", False)),
                   allow_e17_code=bool(getattr(args, "allow_e17_code", False)))
            return 0
        if args.sources:
            profiles = {k: v for k, v in profiles.items() if k in set(args.sources)}
        if not args.mix_config:
            raise SystemExit(f"{args.stage} 需要 --mix-config <yaml>（如 configs/mix_m2.yaml）")
        if args.stage == "s6_val":
            mat.run_s6_val(out_root, profiles, args.mix_config, args.workers,
                           _install_stall_guard, sources=args.sources)
            return 0
        mat.run_s6(out_root, profiles, args.mix_config, args.workers,
                   _install_stall_guard, sources=args.sources)
        return 0

    profiles = load_profiles(args.profiles)
    if args.sources:
        unknown = set(args.sources) - set(profiles)
        if unknown:
            raise SystemExit(f"unknown sources: {sorted(unknown)}")
        profiles = {k: v for k, v in profiles.items() if k in set(args.sources)}

    _WORKER["profiles"] = profiles

    if args.stage in ("s2", "s3", "dedup"):
        if args.stage in ("s2", "dedup") and not (out_root / "s1").is_dir():
            raise SystemExit(f"s2 依赖 s1 产物：{out_root / 's1'} 不存在，请先运行 --stage s1")
        if args.stage == "s3" and not (out_root / "s2").is_dir():
            raise SystemExit(f"s3 依赖 s2 产物：{out_root / 's2'} 不存在，请先运行 --stage s2（或 --stage dedup）")
        cfg = load_dedup_config(args.dedup_config)
        if args.stage in ("s2", "dedup"):
            run_s2(out_root, profiles, cfg, args.workers, sources=args.sources, limit=args.limit)
            log.info("stage s2 global summary: %s", out_root / "stage_s2_global_summary.json")
        if args.stage in ("s3", "dedup"):
            run_s3(out_root, profiles, cfg, args.workers, sources=args.sources, limit=args.limit)
            log.info("stage s3 global summary: %s", out_root / "stage_s3_global_summary.json")
        return 0

    if args.stage in ("ndv2", "ndv2_refresh"):
        # ndv2：web/code 候选签名验证 + 全文精度门 + 外排直接边决策。
        # 依赖 s3_pairs_bydoc / s3_store / s3_csig 以及 candidate flat ctext 索引。
        if not (out_root / "s3").is_dir():
            raise SystemExit(f"ndv2 依赖 s3 产物：{out_root / 's3'} 不存在，请先运行 --stage s3")
        cfg = load_dedup_config(args.dedup_config)
        if args.stage == "ndv2":
            run_ndv2(out_root, profiles, cfg, args.workers, sources=args.sources)
            log.info("stage ndv2 global summary: %s", out_root / "stage_ndv2_global_summary.json")
        else:
            run_ndv2_refresh(out_root, profiles, cfg, args.workers, sources=args.sources, limit=args.limit)
            log.info("stage ndv2_refresh global summary: %s", out_root / "stage_ndv2_refresh_global_summary.json")
        return 0

    fail_fast_checks(data_root, out_root, args.profiles)

    stages = ["s0", "s1"] if args.stage == "all" else [args.stage]
    for stage in stages:
        tasks = _shard_tasks_s0(profiles, data_root) if stage == "s0" else _shard_tasks_s1(profiles, out_root)
        tasks = sorted(tasks, key=lambda t: (t[1], t[2]))
        if args.limit:
            tasks = tasks[: args.limit]
        log.info("stage %s: %d shards", stage, len(tasks))
        if not tasks:
            continue
        summaries = run_stage(stage, tasks, args.profiles, data_root, out_root, args.workers)
        path = write_global_summary(out_root, stage, summaries)
        log.info("stage %s global summary: %s", stage, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
