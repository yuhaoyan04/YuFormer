"""Checkpoint（宪法 6：marker 协议 + 原子写）。

目录布局（ckpt_root = <out_dir>/checkpoints）：
  ckpt-00001234/
    model.pt      FSDP full state_dict（rank0 收集，cpu offload）
    optim.pt      优化器 full state_dict（rank0）
    cursor.json   DataCursor + iter + config_sha256
    rng-rank-*.pt 每个 data-parallel rank 各自的 python/numpy/torch/cuda RNG states
    rng.pt        rank0 RNG（兼容旧 checkpoint/离线工具）
    complete.json 完整写入哨兵
  latest.json     {"iter": int, "config_sha256": str, "run_name": str}（原子写最后落）

RESUME_ITER 环境变量（对齐 ZGCM launch_train.sh）：设置时必须与 latest.json 的
iter 相等，否则拒绝启动（防跑错 checkpoint）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    StateDictType,
)

MARKER_NAME = "latest.json"


def ckpt_dir(root: Path, it: int) -> Path:
    return root / f"ckpt-{it:08d}"


def _named_optim_state_dict(model: FSDP, opt: torch.optim.Optimizer) -> dict:
    """no_shard 路径：优化器 state 按参数 FQN 键控（orig params 全量有效，无需 FSDP 转换）。

    直接序列化 opt.state（param 对象键控）+ param_groups 超参，绕开 torch 2.5
    Optimizer.state_dict() 的 int 索引打包（多优化器场景下映射不可恢复）。
    """
    fqn_by_id = {id(p): n for n, p in model.named_parameters()}
    state: dict[str, dict] = {}
    for p, s in opt.state.items():
        fqn = fqn_by_id.get(id(p))
        if fqn is None:
            raise KeyError(f"optimizer state 参数不在 model.named_parameters() 中: {id(p)}")
        state[fqn] = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in s.items()}
    groups = [
        {k: v for k, v in g.items() if k != "params"}
        for g in opt.param_groups
    ]
    params = [[fqn_by_id[id(p)] for p in g["params"]] for g in opt.param_groups]
    return {"state": state, "param_groups": groups, "params": params}


def _load_named_optim_state_dict(model: FSDP, opt: torch.optim.Optimizer, osd: dict) -> None:
    p_by_fqn = dict(model.named_parameters())
    opt.state.clear()
    for fqn, s in osd["state"].items():
        p = p_by_fqn[fqn]
        opt.state[p] = {
            k: (v.to(p.device) if torch.is_tensor(v) else v) for k, v in s.items()
        }
    for g, g_saved, fqns in zip(opt.param_groups, osd["param_groups"], osd["params"]):
        g["params"] = [p_by_fqn[f] for f in fqns]
        for k, v in g_saved.items():
            g[k] = v


def save_checkpoint(
    ckpt_root: Path,
    it: int,
    model: FSDP,
    optimizers: list[torch.optim.Optimizer],
    cursor_state: dict,
    rng_state: dict,
    config_sha256: str,
    run_name: str,
    rank: int,
    sharding: str = "no_shard",
) -> Path:
    ckpt_root.mkdir(parents=True, exist_ok=True)
    distributed = torch.distributed.is_initialized()
    if not distributed and rank != 0:
        raise ValueError("未初始化 distributed 时 save_checkpoint 的 rank 必须为 0")
    published: list[int | None] = [None]
    if rank == 0:
        marker_path = ckpt_root / MARKER_NAME
        if marker_path.is_file():
            published[0] = int(json.loads(marker_path.read_text("utf-8"))["iter"])
    if distributed:
        torch.distributed.broadcast_object_list(published, src=0)
    if published[0] is not None and published[0] >= it:
        raise RuntimeError(
            f"拒绝非单调 checkpoint 发布: latest iter={published[0]}, 待保存 iter={it}；"
            "可能有并发任务或 --no-resume 误写同一 out_dir"
        )
    # 所有 rank 共用同一临时目录，各自写唯一命名的 RNG 文件。rank0 独占目录生命周期。
    tmp = ckpt_root / f".tmp-ckpt-{it:08d}"
    if sharding == "no_shard":
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model_sd = model.state_dict()
        optim_sds = [_named_optim_state_dict(model, o) for o in optimizers]
    else:
        if len(optimizers) != 1:
            raise ValueError("full_shard 仅支持单优化器（FSDP.optim_state_dict 不支持多优化器）")
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            model_sd = model.state_dict()
            optim_sds = [FSDP.optim_state_dict(model, optimizers[0])]
    if rank == 0:
        import shutil

        # 清理同一 iteration 由当前或升级前（带 pid 后缀）协议留下的未发布临时目录。
        for residue in ckpt_root.glob(f".tmp-ckpt-{it:08d}*"):
            if residue.is_dir():
                shutil.rmtree(residue)
        tmp.mkdir(parents=True)
    if distributed:
        torch.distributed.barrier()

    torch.save(rng_state, tmp / f"rng-rank-{rank:05d}.pt")
    if rank == 0:
        torch.save(model_sd, tmp / "model.pt")
        torch.save(optim_sds, tmp / "optim.pt")
        (tmp / "cursor.json").write_text(
            json.dumps(
                {
                    "iter": it,
                    "cursor": cursor_state,
                    "config_sha256": config_sha256,
                    "run_name": run_name,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        # 保留旧文件名，让尚未升级的单 rank 离线读取工具仍可工作。
        torch.save(rng_state, tmp / "rng.pt")
    marker = {
        "iter": it,
        "config_sha256": config_sha256,
        "run_name": run_name,
        "dir": f"ckpt-{it:08d}",
    }
    tmp_marker = ckpt_root / f".tmp-marker-{os.getpid()}-{it}"
    if rank == 0:
        tmp_marker.write_text(json.dumps(marker, sort_keys=True), encoding="utf-8")
    if distributed:
        torch.distributed.barrier()
    if rank == 0:
        # 2026-10-16 E-25：SFS 禁止目录 rename（os.replace(dir) 抛 EPERM，8 连崩根因）——
        # 改为"建最终目录 + 逐文件 rename（文件 rename SFS 支持）+ complete.json 哨兵"。
        # 崩溃在中间步 → 无 marker/无 complete.json，resume 不认可，语义同原子写。
        final = ckpt_dir(ckpt_root, it)
        final.mkdir(parents=True, exist_ok=True)
        for f in sorted(tmp.iterdir()):
            os.replace(f, final / f.name)
        (final / "complete.json").write_text(
            json.dumps({"iter": it}, sort_keys=True), encoding="utf-8")
        tmp.rmdir()
        os.replace(tmp_marker, ckpt_root / MARKER_NAME)
    if distributed:
        torch.distributed.barrier()
    return ckpt_dir(ckpt_root, it)


def load_marker(ckpt_root: Path) -> dict[str, Any]:
    marker_path = ckpt_root / MARKER_NAME
    if not marker_path.is_file():
        raise FileNotFoundError(f"checkpoint marker 缺失: {marker_path}（无可续训断点）")
    return json.loads(marker_path.read_text("utf-8"))


def prune_checkpoints(ckpt_root: Path, keep_last_n: int, keep_every: int = 0) -> list[Path]:
    """删除已完成的旧 checkpoint，始终保留 latest、最近 N 个和周期里程碑。

    ``keep_last_n <= 0`` 表示关闭清理。没有 ``complete.json`` 的目录不在这里处理，
    避免把故障取证现场误判成可回收 checkpoint。
    """
    if keep_last_n <= 0:
        return []
    if keep_every < 0:
        raise ValueError("keep_every 必须 >= 0")
    complete: list[tuple[int, Path]] = []
    for path in ckpt_root.glob("ckpt-[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]"):
        if path.is_symlink() or not (path / "complete.json").is_file():
            continue
        try:
            it = int(path.name.removeprefix("ckpt-"))
        except ValueError:
            continue
        complete.append((it, path))
    complete.sort()
    protected = {it for it, _ in complete[-keep_last_n:]}
    marker_path = ckpt_root / MARKER_NAME
    if marker_path.is_file():
        protected.add(int(load_marker(ckpt_root)["iter"]))
    if keep_every > 0:
        protected.update(it for it, _ in complete if it % keep_every == 0)

    import shutil

    removed: list[Path] = []
    for it, path in complete:
        if it not in protected:
            shutil.rmtree(path)
            removed.append(path)
    return removed


def resolve_resume_iter(ckpt_root: Path, resume_iter: int | None = None) -> int:
    marker = load_marker(ckpt_root)
    env_iter = os.environ.get("RESUME_ITER")
    it = int(marker["iter"])
    if env_iter is not None and int(env_iter) != it:
        import sys

        print(f"RESUME_ITER={env_iter} != checkpoint iter={it}（marker 协议拒绝：跑错 checkpoint？）",
              file=sys.stderr)
        raise SystemExit(4)
    if resume_iter is not None and resume_iter != it:
        raise SystemExit(f"resume={resume_iter} != checkpoint iter={it}")
    return it


def load_checkpoint(
    ckpt_root: Path,
    it: int,
    model: FSDP,
    optimizers: list[torch.optim.Optimizer],
    rank: int,
    sharding: str = "no_shard",
) -> tuple[dict, dict]:
    """返回 (cursor_state, rng_state)。model/optim 原地加载。

    full state_dict 由各 rank 自行从磁盘读入后在 FSDP FULL_STATE_DICT 上下文加载
    （加载侧不做 rank0_only，避免广播语义差异；文件读取由 page cache 摊薄）。
    no_shard 优化器 state 为 FQN 键控字典，直接映射回参数对象。
    """
    d = ckpt_dir(ckpt_root, it)
    if not d.is_dir():
        raise FileNotFoundError(f"checkpoint 缺失: {d}")
    rank_rng = d / f"rng-rank-{rank:05d}.pt"
    if rank_rng.is_file():
        rng_path = rank_rng
    elif list(d.glob("rng-rank-*.pt")):
        raise FileNotFoundError(
            f"checkpoint 含逐 rank RNG，但缺少 rank={rank} 的状态: {rank_rng}；"
            "请使用与保存时相同的 world size"
        )
    else:
        # 兼容升级前只保存 rank0 rng.pt 的 checkpoint。
        rng_path = d / "rng.pt"
    cursor = json.loads((d / "cursor.json").read_text("utf-8"))
    model_sd = torch.load(d / "model.pt", map_location="cpu", weights_only=True)
    full_osds = torch.load(d / "optim.pt", map_location="cpu", weights_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
        model.load_state_dict(model_sd)
    if sharding == "no_shard":
        for o, osd in zip(optimizers, full_osds):
            _load_named_optim_state_dict(model, o, osd)
    else:
        if len(optimizers) != 1:
            raise ValueError("full_shard 仅支持单优化器")
        sharded = FSDP.optim_state_dict_to_load(model, optimizers[0], full_osds[0])
        optimizers[0].load_state_dict(sharded)
    rng_state = torch.load(rng_path, map_location="cpu", weights_only=False)
    return cursor, rng_state


def get_rng_state() -> dict:
    import numpy as np
    import random

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    import numpy as np
    import random

    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
