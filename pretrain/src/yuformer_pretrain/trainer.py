"""pretrain 训练循环（宪法 1/2/4/6 的运行闭包核心）。

并行：单机 FSDP（全分片），BF16 autocast 混合精度（fp32 master），
activation checkpointing 可开关；优化器 AdamW / Muon（2D 路由，见 optim.py）。
数据：S6 物化的 token 流 + 全局样本游标（见 data.py），checkpoint 存
model+optimizers+cursor+rng，marker 协议防跑错（见 checkpoint.py）。
数据并行切分：每步全局 S = mbs×accum×world 样本，rank r 消费其中
第 [r×(S/world), (r+1)×(S/world)) 段（S/world = mbs×accum），由全局
游标 + rank 偏移推出，断点续训后各 rank 数据切分不变。
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from . import checkpoint as ckpt
from .config import ConfigError, PretrainConfig, load_config
from .data import (
    DataCursor,
    PrefetchLoader,
    TokenStream,
    concat_streams,
    distributed_sample_index,
    load_data_release,
    open_val_streams,
)
from .lease import RunLease
from .optim import build_optimizers, lr_at, set_lrs, step as optim_step, zero_grad
from .run_identity import build_run_identity, ensure_run_identity, write_attempt_manifest
from .tracking import RunTracker, canonical_sha256

log = logging.getLogger("yuformer.pretrain")

H800_BF16_PEAK = 989e12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YuFormer pretrain")
    p.add_argument("--config", required=True)
    p.add_argument("--allow-iter-override", action="store_true")
    p.add_argument("--max-iters", type=int, default=0, help="smoke 用：截断迭代数（0=按预算）")
    p.add_argument("--no-resume", action="store_true", help="忽略已有 checkpoint 从头训")
    return p.parse_args()


def init_distributed() -> tuple[int, int, int]:
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29517"
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, world, local_rank


class _ActivationCheckpointWrapper(torch.nn.Module):
    def __init__(self, block: torch.nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        from torch.utils.checkpoint import checkpoint

        return checkpoint(self.block, *args, use_reentrant=False)


def build_fsdp_model(cfg: PretrainConfig, device: torch.device) -> FSDP:
    import sys

    from yuformer_model import ModelConfig, create_model
    from yuformer_model.transformer import TransformerBlock

    model_src = Path(cfg.model_configs).resolve().parent.parent / "src"
    if str(model_src) not in sys.path:
        sys.path.insert(0, str(model_src))
    model_cfg = ModelConfig.from_preset(cfg.model_preset, config_path=cfg.model_configs)
    if cfg.train.selective_recompute != "none" and cfg.train.activation_checkpointing:
        raise ConfigError("selective_recompute 与 activation_checkpointing（全层）互斥，二选一")
    model_cfg.selective_recompute = cfg.train.selective_recompute
    model_cfg.bf16_hygiene = cfg.train.bf16_hygiene
    model_cfg.attn_impl = cfg.train.attn_impl
    torch.manual_seed(cfg.seed)
    model = create_model(model_cfg)
    if cfg.train.activation_checkpointing:
        # 必须先于 torch.compile 包装：compile 返回 OptimizedModule 后对其改
        # .layers 不触及 _orig_mod，包装会静默失效（E-51 实测踩坑）。
        model.layers = torch.nn.ModuleList([_ActivationCheckpointWrapper(b) for b in model.layers])
    if cfg.train.compile:
        # E-51：在 FSDP 包装前编译底层 module（no_shard 下单卡语义不变；
        # dynamic=False 固定形状，避免多 shape 重编译）。
        model = torch.compile(model, mode=cfg.train.compile_mode, dynamic=False)
    auto_wrap = partial(transformer_auto_wrap_policy, transformer_layer_cls={TransformerBlock})
    mp = MixedPrecision(
        param_dtype=torch.float32,
        reduce_dtype=torch.bfloat16 if cfg.train.bf16 else torch.float32,
        buffer_dtype=torch.float32,
    )
    strategy = {
        "no_shard": ShardingStrategy.NO_SHARD,
        "full_shard": ShardingStrategy.FULL_SHARD,
    }[cfg.parallel.sharding]
    model = model.to(device)
    return FSDP(
        model, device_id=device, auto_wrap_policy=auto_wrap, mixed_precision=mp,
        sharding_strategy=strategy, use_orig_params=True,
    )


def batch_to_device(tokens: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.from_numpy(tokens).to(device)
    return t[:, :-1].contiguous(), t[:, 1:].contiguous()


def forward_loss(model: FSDP, x: torch.Tensor, y: torch.Tensor, bf16: bool,
                 ce_chunk: int = 0) -> torch.Tensor:
    # E-48：ce_chunk>0 走分块融合 CE（不物化全量 fp32 logits）；0 保留旧对拍路径。
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
        if ce_chunk > 0:
            return model(x, labels=y, ce_chunk=ce_chunk)
        logits = model(x)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))


@torch.no_grad()
def eval_val_loss(model: FSDP, streams: list[TokenStream], samples_per_domain: int,
                  mbs: int, bf16: bool, device: torch.device, max_tokens: int,
                  ce_chunk: int = 0) -> dict[str, float]:
    model.eval()
    out = {}
    for s in streams:
        n = min(samples_per_domain, s.samples, max(1, max_tokens // s.seq_len))
        # E-77 根因修复：评估批必须整除 mbs（丢弃余数）——残批（如 50=6×8+2 的 2）
        # 触发 torch.compile 按新形状重编译（dynamo cache_size_limit=8 打满后每轮 val
        # 重编译一次，inductor 工件每次永久驻留 ~13GB、步速随之阶梯劣化直至 OOM）。
        # 整除后评估形状与训练完全一致，命中同一编译图，零重编译。
        n -= n % mbs
        if n == 0:
            continue
        losses: list[torch.Tensor] = []
        for i in range(0, n, mbs):
            idxs = list(range(i, min(i + mbs, n)))
            arr = np.stack([s.sample(j) for j in idxs])
            x, y = batch_to_device(arr, device)
            losses.append(forward_loss(model, x, y, bf16, ce_chunk) * len(idxs))
        out[s.bin_path.name.replace(".tokens.bin", "")] = torch.stack(losses).sum().item() / n
    model.train()
    return out


def _gpu_neighbor_snapshot() -> dict:
    """E-50：MFU 门禁测量协议——每 log 点记录 GPU util/显存基线，与吞吐并排入
    metrics.jsonl，事后可拟合争用系数（用户裁定 2026-10-11：按峰值邻居负载承诺）。"""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()
        utils, mems = [], []
        for line in out:
            u, m = line.split(", ")
            utils.append(int(u)); mems.append(int(m))
        return {"gpu_util_all": utils, "gpu_mem_mb_all": mems}
    except Exception:
        return {}


def run(cfg: PretrainConfig, max_iters: int, no_resume: bool, rank: int, world: int) -> None:
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    out_dir = Path(cfg.out_dir)
    ckpt_root = out_dir / "checkpoints"
    cfg_sha = canonical_sha256(cfg.raw)

    dataset_release = load_data_release(
        cfg.data.release_manifest,
        required=cfg.data.require_release,
    )
    expected_snapshot_id = dataset_release["snapshot_id"] if dataset_release else None
    streams = [
        TokenStream.open(
            b,
            m,
            cfg.data.seq_len,
            require_eod=cfg.data.require_eod,
            expected_snapshot_id=expected_snapshot_id,
        )
        for b, m in zip(cfg.data.stages, cfg.data.meta)
    ]
    tokenizer_shas = {s.meta.get("tokenizer_sha256") for s in streams}
    if None in tokenizer_shas or "" in tokenizer_shas or len(tokenizer_shas) != 1:
        raise ConfigError(
            "所有训练 token stream 必须声明且共享同一 tokenizer_sha256，得到 "
            f"{sorted(str(x) for x in tokenizer_shas)}"
        )
    train_tokenizer_sha = str(next(iter(tokenizer_shas)))
    repo_root = Path(__file__).resolve().parents[3]
    identity = build_run_identity(
        run_name=cfg.run_name,
        config=cfg.raw,
        model_config_path=cfg.model_configs,
        tokenizer_sha256=train_tokenizer_sha,
        data=[
            {
                "path": stream.bin_path.as_posix(),
                "sha256": stream.meta.get("sha256"),
                "edge_sha256": stream.meta.get("edge_sha256"),
                "total_tokens": stream.meta.get("total_tokens"),
                "source": stream.meta.get("source"),
            }
            for stream in streams
        ],
        dataset_release=dataset_release,
        source_paths=[repo_root / "model" / "src", repo_root / "pretrain" / "src"],
        repo_root=repo_root,
        backend="pytorch-fsdp1-compat",
    )
    if rank == 0:
        ensure_run_identity(out_dir, identity)
        write_attempt_manifest(out_dir, identity, world_size=world)
    if world > 1:
        torch.distributed.barrier()
    total, _ = concat_streams(streams)
    cursor = DataCursor(total, cfg.data.epochs)
    iters = cfg.derived_iters
    if max_iters:
        iters = min(iters, max_iters)
    spr = cfg.train.micro_batch_size * cfg.train.grad_accum  # samples per rank per step
    cursor.require(iters * spr * world)

    tracker = RunTracker(out_dir / "results", cfg.run_name, cfg.raw, cfg.seed) if rank == 0 else None

    model = build_fsdp_model(cfg, device)
    n_params = model.module.num_parameters()
    n_nonemb = model.module.num_parameters(non_embedding=True)
    optimizers = build_optimizers(model, cfg.optim)

    start_it = 0
    if no_resume and (ckpt_root / ckpt.MARKER_NAME).is_file():
        raise ConfigError(
            f"--no-resume 不能写入已有 checkpoint 的 out_dir: {out_dir}；请使用新的 out_dir"
        )
    if not no_resume and (ckpt_root / ckpt.MARKER_NAME).is_file():
        start_it = ckpt.resolve_resume_iter(ckpt_root)
        if start_it >= iters:
            raise SystemExit(f"checkpoint iter {start_it} >= 目标 iters {iters}")
        cursor_state, rng_state = ckpt.load_checkpoint(ckpt_root, start_it, model, optimizers, rank,
                                                       cfg.parallel.sharding)
        if cursor_state.get("config_sha256") != cfg_sha:
            raise SystemExit("checkpoint config_sha256 与当前配置不符（防跑错：改配置请换 run_name）")
        cursor.load_state_dict(cursor_state["cursor"])
        ckpt.set_rng_state(rng_state)
        cursor.require((iters - start_it) * spr * world)
        if rank == 0 and tracker:
            archive = tracker.rewind_to_step(start_it, reason="resume_from_checkpoint")
            if archive is not None:
                log.info("archived post-checkpoint metrics suffix: %s", archive)
        log.info("rank %d: resume from iter %d (consumed=%d)", rank, start_it, cursor.consumed)
    elif cfg.init_weights_from:
        # 2026-10-15 V6/WSD 退火段语义：仅载权重，优化器/游标/rng 全新（ZGCM --pretrained-checkpoint）。
        src = Path(cfg.init_weights_from)
        if not (src / "model.pt").is_file():
            raise ConfigError(f"init_weights_from 无 model.pt: {src}")
        if src.resolve() == Path(cfg.out_dir).resolve():
            raise ConfigError("init_weights_from 不能与 out_dir 相同（防自覆盖）")
        if rank == 0 or world == 1:
            sd = torch.load(src / "model.pt", map_location="cpu", weights_only=True)
            tgt = model.module if hasattr(model, "module") else model
            missing, unexpected = tgt.load_state_dict(sd, strict=True)
            log.info("init_weights_from %s loaded (missing=%d unexpected=%d)，优化器/游标全新",
                     src, len(missing), len(unexpected))
        if world > 1:
            torch.distributed.barrier()

    mc = model.module.config
    flops_per_token = 6 * n_nonemb + 12 * mc.num_layers * cfg.data.seq_len * mc.hidden_size

    val_streams: list[TokenStream] = []
    val_skipped: list[str] = []
    if cfg.data.val_dir:
        val_streams, val_skipped = open_val_streams(
            cfg.data.val_dir,
            cfg.data.seq_len,
            mc.vocab_size,
            tokenizer_sha256=train_tokenizer_sha,
            require_eod=cfg.data.require_eod,
            expected_snapshot_id=expected_snapshot_id,
        )
        for name in val_skipped:
            log.warning(
                "跳过 val 流 %s：vocab_size 或 tokenizer_sha256 与当前训练不兼容",
                name,
            )

    if rank == 0 and tracker:
        tracker.write_manifest(
            params=n_params,
            params_non_embedding=n_nonemb,
            iters=iters,
            tokens_per_step=cfg.tokens_per_step,
            budget_tokens=cfg.train.budget_tokens,
            model_preset=cfg.model_preset,
            tokenizer_sha256=train_tokenizer_sha,
            data_edge_sha256=[s.meta.get("edge_sha256") for s in streams],
            data_sha256=[s.meta.get("sha256") for s in streams],
            data_source=[s.meta.get("source") for s in streams],
            data_stage_samples=[s.samples for s in streams],
            data_stage_switch_at_samples=[int(x) for x in np.cumsum([s.samples for s in streams])[:-1]],
            dataset_release=dataset_release,
            world_size=world,
            run_id=identity["run_id"],
            backend=identity["backend"],
            val_skipped=val_skipped,
        )

    if cursor.consumed % (spr * world) != 0:
        raise ConfigError(
            f"checkpoint consumed_samples={cursor.consumed} 未对齐全局 step 大小 {spr * world}"
        )
    resume_consumed = cursor.consumed
    local_cursor = DataCursor(total, cfg.data.epochs)
    local_cursor.capacity = 1 << 62  # 本地视图不做容量检查（全局 cursor 已查）
    sample_index_fn = partial(
        distributed_sample_index,
        resume_consumed=resume_consumed,
        samples_per_rank=spr,
        rank=rank,
        world_size=world,
    )

    step_times: list[float] = []
    t_start = time.monotonic()
    acc_loss = float("nan")
    model.train()
    with PrefetchLoader(
        streams,
        local_cursor,
        num_workers=2,
        sample_index_fn=sample_index_fn,
    ) as source:
        for it in range(start_it, iters):
            lr = lr_at(it, iters, cfg.optim)
            set_lrs(optimizers, lr)
            zero_grad(optimizers)
            t0 = time.monotonic()
            acc_loss = 0.0
            for _ in range(cfg.train.grad_accum):
                arrs = [source.next_sample() for _ in range(cfg.train.micro_batch_size)]
                x, y = batch_to_device(np.stack(arrs), device)
                loss = forward_loss(model, x, y, cfg.train.bf16, cfg.train.ce_chunk) / cfg.train.grad_accum
                loss.backward()
                acc_loss += loss.item()
            model.clip_grad_norm_(cfg.optim.clip_grad)
            optim_step(optimizers)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.monotonic() - t0
            step_times.append(dt)
            cursor.advance(spr * world)

            if rank == 0 and tracker and (it % cfg.train.log_every == 0 or it == iters - 1):
                tracker.log_metrics(step=it + 1, loss=acc_loss, lr=lr,
                                    consumed_samples=cursor.consumed, step_time=round(dt, 4),
                                    **_gpu_neighbor_snapshot())
                log.info("iter %d/%d loss=%.4f lr=%.2e t=%.3fs", it + 1, iters, acc_loss, lr, dt)

            if cfg.train.eval_every > 0 and (it + 1) % cfg.train.eval_every == 0 and val_streams:
                vals = eval_val_loss(model, val_streams, cfg.data.val_samples_per_domain,
                                     cfg.train.micro_batch_size, cfg.train.bf16, device,
                                     cfg.data.val_max_tokens_per_domain)
                if rank == 0 and tracker:
                    tracker.log_metrics(step=it + 1, **{f"val_loss_{k}": v for k, v in vals.items()})
                    log.info("iter %d val: %s", it + 1, {k: round(v, 4) for k, v in vals.items()})
                # E-77：val/save 的临时大块分配（评估批 + FSDP full-state-dict 收集）在
                # expandable_segments 池里部分驻留（实测每 500 步交点 +13~28GB、四轮 OOM
                # 128k 臂、步速拖成阶梯）。池状态决定单次 empty_cache 能否整体归还（128k 有效、
                # 32k/64k 无效——同代码同 iter 验证），故三重释放覆盖：周期 200 iters + val 后 + save 后。
                # 只归还空闲段，数值与断点语义零影响。
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if (cfg.train.save_every > 0 and (it + 1) % cfg.train.save_every == 0) or it + 1 == iters:
                ckpt.save_checkpoint(ckpt_root, it + 1, model, optimizers, cursor.state_dict(),
                                     ckpt.get_rng_state(), cfg_sha, cfg.run_name, rank,
                                     cfg.parallel.sharding)
                if rank == 0:
                    log.info("saved checkpoint at iter %d", it + 1)
                    removed = ckpt.prune_checkpoints(
                        ckpt_root,
                        keep_last_n=cfg.train.keep_last_n,
                        keep_every=cfg.train.keep_every,
                    )
                    if removed:
                        log.info("checkpoint retention removed: %s", [p.name for p in removed])
                if world > 1:
                    # rank0 清理旧目录期间其余 rank 不得提前进入下一步 collective。
                    torch.distributed.barrier()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if torch.cuda.is_available() and (it + 1) % 200 == 0:
                torch.cuda.empty_cache()

    if rank == 0 and tracker:
        wall = time.monotonic() - t_start
        tail = step_times[max(10, len(step_times) // 10):] or step_times
        mean_t = sum(tail) / len(tail)
        achieved = flops_per_token * cfg.tokens_per_step / mean_t
        tracker.log_metrics(step=iters, final_loss=acc_loss, wallclock_total=round(wall, 1),
                            mean_step_time=round(mean_t, 4), achieved_tflops=round(achieved / 1e12, 2),
                            mfu=round(achieved / H800_BF16_PEAK, 4))
        tracker.write_manifest(
            params=n_params,
            params_non_embedding=n_nonemb,
            iters=iters,
            tokens_per_step=cfg.tokens_per_step,
            budget_tokens=cfg.train.budget_tokens,
            model_preset=cfg.model_preset,
            tokenizer_sha256=train_tokenizer_sha,
            data_edge_sha256=[s.meta.get("edge_sha256") for s in streams],
            data_sha256=[s.meta.get("sha256") for s in streams],
            data_source=[s.meta.get("source") for s in streams],
            data_stage_samples=[s.samples for s in streams],
            data_stage_switch_at_samples=[int(x) for x in np.cumsum([s.samples for s in streams])[:-1]],
            dataset_release=dataset_release,
            world_size=world,
            run_id=identity["run_id"],
            backend=identity["backend"],
            val_skipped=val_skipped,
            mfu=achieved / H800_BF16_PEAK,
            mean_step_time=mean_t,
            achieved_tflops=achieved / 1e12,
            wallclock_total=wall,
        )
        # E-62：调度器臂完成判据 <out>/results/final.json——此前无人写它，臂会无限重跑
        import json as _json
        _fp = out_dir / "results" / "final.json"
        _fp.parent.mkdir(parents=True, exist_ok=True)
        _fp.write_text(_json.dumps({
            "status": "done", "run_name": cfg.run_name, "iters": iters,
            "final_loss": acc_loss, "mfu": achieved / H800_BF16_PEAK,
            "mean_step_time": mean_t, "wallclock_total": wall,
        }, indent=2, sort_keys=True))
        log.info("wrote %s", _fp)
        if torch.cuda.is_available():
            log.info("VRAM_REPORT peak_allocated_mb=%d peak_reserved_mb=%d",
                     int(torch.cuda.max_memory_allocated() / 2**20),
                     int(torch.cuda.max_memory_reserved() / 2**20))
        log.info("done: %d iters, mean step %.3fs, %.1f TFLOPS, MFU=%.1f%%",
                 iters, mean_t, achieved / 1e12, 100 * achieved / H800_BF16_PEAK)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    cfg = load_config(args.config, allow_iter_override=args.allow_iter_override)
    if cfg.backend != "pytorch-fsdp1-compat":
        raise ConfigError(
            f"config backend={cfg.backend!r} cannot run in the compatibility trainer; "
            "run the pinned TorchTitan backend after preflight and adapter qualification"
        )
    rank_hint = int(os.environ.get("RANK", "0"))
    lease = RunLease(cfg.out_dir, cfg.run_name) if rank_hint == 0 else None
    try:
        if lease is not None:
            lease.acquire()
        rank, world, _ = init_distributed()
        if world != cfg.parallel.world_size:
            raise SystemExit(f"world_size {world} != 配置 {cfg.parallel.world_size}")
        run(cfg, args.max_iters, args.no_resume, rank, world)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if lease is not None:
            lease.release()


if __name__ == "__main__":
    main()
