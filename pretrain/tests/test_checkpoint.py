"""checkpoint.py 单测（CPU，world_size=1 路径）：save→load 往返、RNG 恢复确定性、
marker 协议（RESUME_ITER 不符 exit 4）、tmp 残留的原子生效语义。

torch 2.2 的 FSDP 无法在纯 CPU 上运行（_lazy_init 的 _init_streams 依赖 CUDA
Stream），而 no_shard 单卡下 FSDP.state_dict_type(FULL_STATE_DICT) 对
state_dict/load_state_dict 是恒等包装，故用 no-op context manager shim 替换
checkpoint 模块内的 FSDP 符号；FSDP 真正的分片/gather 语义由
scripts/smoke_train.py 在 GPU 上端到端覆盖。
"""
import contextlib
import json
import os
import random

import numpy as np
import pytest
import torch

from yuformer_pretrain import checkpoint as ckpt
from yuformer_pretrain.config import OptimConfig
from yuformer_pretrain.data import DataCursor
from yuformer_pretrain.muon import Muon
from yuformer_pretrain.optim import build_optimizers


class _PlainFSDPShim:
    """no_shard 单卡下 state_dict_type 是恒等语义，用 no-op 顶替 torch FSDP。"""

    @staticmethod
    @contextlib.contextmanager
    def state_dict_type(module, *args, **kwargs):
        yield


@pytest.fixture
def plain_fsdp(monkeypatch):
    monkeypatch.setattr(ckpt, "FSDP", _PlainFSDPShim)


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 16),
    )


class _TinyMixedModel(torch.nn.Module):
    """muon 路由代表：embedding/标量走 AdamW，2D 矩阵走 Muon。"""

    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(32, 8)
        self.norm = torch.nn.LayerNorm(8)
        self.proj = torch.nn.Linear(8, 16)

    def forward(self, idx):
        return self.proj(self.norm(self.embed_tokens(idx)))


def _optim_cfg(**kw):
    base = dict(optimizer="adamw", lr=1e-3, weight_decay=0.0, warmup_steps=0,
                lr_schedule="constant", min_lr_ratio=0.1, adam_betas=(0.9, 0.95),
                muon_momentum=0.95, muon_nesterov=True, muon_ns_steps=5, clip_grad=1.0)
    base.update(kw)
    return OptimConfig(**base)


def _train_steps(model, opts, batches):
    for x in batches:
        loss = torch.nn.functional.mse_loss(model(x), x)
        for o in opts:
            o.zero_grad()
        loss.backward()
        for o in opts:
            o.step()
    return model


def _snapshot_params(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def _snapshot_opt_state(model, opt):
    fqn_by_id = {id(p): n for n, p in model.named_parameters()}
    return {
        fqn_by_id[id(p)]: {
            k: (v.detach().clone() if torch.is_tensor(v) else v)
            for k, v in st.items()
        }
        for p, st in opt.state.items()
    }


def _assert_opt_state_equal(model, opt, snap):
    fqn_by_id = {id(p): n for n, p in model.named_parameters()}
    assert len(opt.state) == len(snap)
    for p, st in opt.state.items():
        for k, v in st.items():
            if torch.is_tensor(v):
                assert torch.equal(v, snap[fqn_by_id[id(p)]][k])
            else:
                assert v == snap[fqn_by_id[id(p)]][k]


def _assert_rng_equal(a, b):
    assert set(a) == set(b)
    assert a["python"] == b["python"]
    assert a["numpy"][0] == b["numpy"][0]
    assert np.array_equal(a["numpy"][1], b["numpy"][1])
    assert a["numpy"][2] == b["numpy"][2]
    assert torch.equal(a["torch"], b["torch"])
    if "cuda" in a:  # 有 GPU 时 rng.pt 含 cuda 四件套；加载侧 map_location=cpu
        for x, y in zip(a["cuda"], b["cuda"]):
            assert torch.equal(x.cpu(), y.cpu())


def _write_marker(root, it, sha="sha-A", run_name="run"):
    root.mkdir(parents=True, exist_ok=True)
    (root / ckpt.MARKER_NAME).write_text(
        json.dumps({"iter": it, "config_sha256": sha, "run_name": run_name,
                    "dir": f"ckpt-{it:08d}"}, sort_keys=True),
        encoding="utf-8",
    )


def _distributed_save_worker(rank, root, rendezvous):
    torch.distributed.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2
    )
    try:
        model = _tiny_model(seed=9)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        torch.manual_seed(100 + rank)
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        ckpt.save_checkpoint(
            root, 3, model, [opt], {"consumed_samples": 48},
            rng_state, "sha-A", "run", rank=rank,
        )
    finally:
        torch.distributed.destroy_process_group()


def test_ckpt_dir_and_marker_layout(tmp_path, plain_fsdp):
    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    root = tmp_path / "checkpoints"
    saved = ckpt.save_checkpoint(root, 7, model, [opt], {"consumed_samples": 56},
                                 ckpt.get_rng_state(), "sha-A", "run", rank=0)
    assert saved == root / "ckpt-00000007"
    assert {p.name for p in saved.iterdir()} == {
        "model.pt", "optim.pt", "cursor.json", "rng.pt", "rng-rank-00000.pt", "complete.json"
    }
    marker = ckpt.load_marker(root)
    assert marker == {"iter": 7, "config_sha256": "sha-A", "run_name": "run",
                      "dir": "ckpt-00000007"}
    # 生效后不留 tmp 痕迹（原子 os.replace）
    assert not list(root.glob(".tmp-*"))


def test_distributed_save_persists_each_rank_rng(tmp_path, plain_fsdp):
    ctx = torch.multiprocessing.get_context("fork")
    root = tmp_path / "checkpoints"
    rendezvous = tmp_path / "gloo-rendezvous"
    procs = [ctx.Process(target=_distributed_save_worker, args=(rank, root, rendezvous))
             for rank in range(2)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=20)
        assert proc.exitcode == 0

    saved = ckpt.ckpt_dir(root, 3)
    assert (saved / "rng-rank-00000.pt").is_file()
    assert (saved / "rng-rank-00001.pt").is_file()
    rng0 = torch.load(saved / "rng-rank-00000.pt", weights_only=False)
    rng1 = torch.load(saved / "rng-rank-00001.pt", weights_only=False)
    assert not torch.equal(rng0["torch"], rng1["torch"])
    assert (saved / "complete.json").is_file()
    assert ckpt.load_marker(root)["iter"] == 3


def test_rank_rng_requires_matching_topology_and_loads_legacy(tmp_path, plain_fsdp):
    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    root = tmp_path / "checkpoints"
    saved_rng = ckpt.get_rng_state()
    ckpt.save_checkpoint(root, 1, model, [opt], {"consumed_samples": 8},
                         saved_rng, "sha-A", "run", rank=0)

    with pytest.raises(FileNotFoundError, match="缺少 rank=1"):
        ckpt.load_checkpoint(root, 1, model, [opt], rank=1)

    # 升级前的 checkpoint 只有 rng.pt，仍可恢复。
    (root / "ckpt-00000001" / "rng-rank-00000.pt").unlink()
    _, loaded = ckpt.load_checkpoint(root, 1, model, [opt], rank=0)
    _assert_rng_equal(loaded, saved_rng)


def test_checkpoint_retention_keeps_recent_latest_and_milestones(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    for it in (10, 20, 30, 40):
        d = ckpt.ckpt_dir(root, it)
        d.mkdir()
        (d / "complete.json").write_text(json.dumps({"iter": it}))
    incomplete = ckpt.ckpt_dir(root, 5)
    incomplete.mkdir()
    (incomplete / "model.pt").write_bytes(b"forensics")
    _write_marker(root, 40)

    removed = ckpt.prune_checkpoints(root, keep_last_n=2, keep_every=20)
    assert [p.name for p in removed] == ["ckpt-00000010"]
    assert {p.name for p in root.glob("ckpt-*")} == {
        "ckpt-00000005", "ckpt-00000020", "ckpt-00000030", "ckpt-00000040"
    }
    assert ckpt.prune_checkpoints(root, keep_last_n=0, keep_every=1) == []


def test_checkpoint_publish_iteration_must_be_monotonic(tmp_path, plain_fsdp):
    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    root = tmp_path / "checkpoints"
    ckpt.save_checkpoint(root, 5, model, [opt], {"consumed_samples": 40},
                         ckpt.get_rng_state(), "sha-A", "run", rank=0)
    with pytest.raises(RuntimeError, match="非单调 checkpoint"):
        ckpt.save_checkpoint(root, 5, model, [opt], {"consumed_samples": 40},
                             ckpt.get_rng_state(), "sha-A", "run", rank=0)
    with pytest.raises(RuntimeError, match="latest iter=5, 待保存 iter=4"):
        ckpt.save_checkpoint(root, 4, model, [opt], {"consumed_samples": 32},
                             ckpt.get_rng_state(), "sha-A", "run", rank=0)


def test_save_load_roundtrip_restores_everything(tmp_path, plain_fsdp):
    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
    g = torch.Generator().manual_seed(1)
    batches = [torch.randn(8, 16, generator=g) for _ in range(3)]
    _train_steps(model, [opt], batches)
    model.eval()
    x = torch.randn(4, 16, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        y_before = model(x)
    params_before = _snapshot_params(model)
    opt_before = _snapshot_opt_state(model, opt)

    root = tmp_path / "checkpoints"
    rng_saved = ckpt.get_rng_state()
    ckpt.save_checkpoint(root, 7, model, [opt], {"consumed_samples": 123},
                         rng_saved, "sha-A", "run", rank=0)

    # 破坏现场：参数漂移 + optimizer state 清零
    perturb = torch.Generator().manual_seed(99)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=perturb))
    for st in opt.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                v.zero_()

    cursor, rng_loaded = ckpt.load_checkpoint(root, 7, model, [opt], rank=0)
    # 迭代状态 / consumed_samples 游标
    assert cursor["iter"] == 7
    assert cursor["config_sha256"] == "sha-A"
    assert cursor["run_name"] == "run"
    assert cursor["cursor"] == {"consumed_samples": 123}
    dc = DataCursor(total_samples=10_000, epochs=1)
    dc.load_state_dict(cursor["cursor"])
    assert dc.consumed == 123
    # model 参数逐位恢复
    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), params_before[n])
    # optimizer state（step/exp_avg/exp_avg_sq）逐位恢复
    _assert_opt_state_equal(model, opt, opt_before)
    # rng 四件套完整往返（cuda 键视机器而定）
    _assert_rng_equal(rng_loaded, rng_saved)
    # 前向输出一致
    ckpt.set_rng_state(rng_loaded)
    with torch.no_grad():
        y_after = model(x)
    assert torch.equal(y_before, y_after)


def test_save_load_roundtrip_muon_route_multi_optimizer(tmp_path, plain_fsdp,
                                                        monkeypatch):
    # fused AdamW 仅 CUDA 可用：CPU 路径强制 fused=False（同 smoke 之外的单测约定）
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = _TinyMixedModel()
    opts = build_optimizers(model, _optim_cfg(optimizer="muon"))
    assert [type(o) for o in opts] == [torch.optim.AdamW, Muon]
    g = torch.Generator().manual_seed(3)
    idx = [torch.randint(0, 32, (4, 8), generator=g) for _ in range(2)]
    for x in idx:
        loss = model(x).pow(2).mean()
        for o in opts:
            o.zero_grad()
        loss.backward()
        for o in opts:
            o.step()
    snaps = [_snapshot_opt_state(model, o) for o in opts]
    params_before = _snapshot_params(model)

    root = tmp_path / "checkpoints"
    ckpt.save_checkpoint(root, 2, model, opts, {"consumed_samples": 8},
                         ckpt.get_rng_state(), "sha-A", "run", rank=0)
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(0.5)
    for o in opts:
        for st in o.state.values():
            for v in st.values():
                if torch.is_tensor(v):
                    v.zero_()

    cursor, _ = ckpt.load_checkpoint(root, 2, model, opts, rank=0)
    assert cursor["iter"] == 2
    for n, p in model.named_parameters():
        assert torch.equal(p.detach(), params_before[n])
    for o, snap in zip(opts, snaps):
        _assert_opt_state_equal(model, o, snap)


def test_resume_training_step_matches_uninterrupted(tmp_path, plain_fsdp):
    """断点续训一致性（smoke_train 断言 2 的单元版）：A 训 3 步存档再训 2 步；
    B 同构造训 3 步后从档恢复再训同样 2 步——loss 与参数逐位一致。"""
    batches = [torch.randn(8, 16, generator=torch.Generator().manual_seed(11))
               for _ in range(5)]

    def build():
        m = _tiny_model(seed=3)
        o = torch.optim.AdamW(m.parameters(), lr=5e-3, betas=(0.9, 0.95))
        return m, o

    def step(m, o, x):
        loss = torch.nn.functional.mse_loss(m(x), x)
        o.zero_grad()
        loss.backward()
        o.step()
        return loss.item()

    m_a, o_a = build()
    m_b, o_b = build()
    for i in range(3):
        assert step(m_a, o_a, batches[i]) == step(m_b, o_b, batches[i])
    ckpt.save_checkpoint(tmp_path / "ck", 3, m_a, [o_a], {"consumed_samples": 24},
                         ckpt.get_rng_state(), "sha-A", "run", rank=0)
    losses_a = [step(m_a, o_a, batches[i]) for i in range(3, 5)]

    with torch.no_grad():  # B 模拟漂移后从档恢复
        for p in m_b.parameters():
            p.mul_(0.5)
    cursor, rng = ckpt.load_checkpoint(tmp_path / "ck", 3, m_b, [o_b], rank=0)
    assert cursor["cursor"]["consumed_samples"] == 24
    ckpt.set_rng_state(rng)
    losses_b = [step(m_b, o_b, batches[i]) for i in range(3, 5)]
    assert losses_b == losses_a
    for (_, pa), (_, pb) in zip(m_a.named_parameters(), m_b.named_parameters()):
        assert torch.equal(pa, pb)


def _draw_triplet():
    return (
        random.random(),
        np.random.rand(4).tolist(),
        torch.rand(4).tolist(),
    )


def test_rng_state_restores_draws_bitwise(monkeypatch):
    # 强制无 GPU 路径：cuda RNG 由 smoke_train.py 在 GPU 上覆盖
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    _draw_triplet()  # 推进状态，避开初始种子的平凡点
    state = ckpt.get_rng_state()
    assert set(state) == {"python", "numpy", "torch"}
    expected = [_draw_triplet() for _ in range(3)]
    ckpt.set_rng_state(state)
    for want in expected:
        assert _draw_triplet() == want  # 同操作重放，浮点逐位一致


def test_set_rng_state_empty_is_noop(monkeypatch):
    # 无 GPU 依赖：只用 CPU 三件套
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    _draw_triplet()  # 推进流
    state_after = ckpt.get_rng_state()
    ckpt.set_rng_state({})  # 空状态应为 no-op：不改变任何流
    draw_direct = _draw_triplet()
    ckpt.set_rng_state(state_after)  # 回到同一状态重放
    assert _draw_triplet() == draw_direct


def test_rng_roundtrip_through_disk(tmp_path):
    state = ckpt.get_rng_state()
    torch.save(state, tmp_path / "rng.pt")
    loaded = torch.load(tmp_path / "rng.pt", map_location="cpu", weights_only=False)
    _assert_rng_equal(loaded, state)
    # 打乱现场后从磁盘状态恢复，随机序列与原始状态重放逐位一致
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    ckpt.set_rng_state(state)
    want = _draw_triplet()
    ckpt.set_rng_state(loaded)
    assert _draw_triplet() == want


def test_resolve_resume_iter_marker_match(tmp_path, monkeypatch):
    _write_marker(tmp_path, 5)
    monkeypatch.delenv("RESUME_ITER", raising=False)
    assert ckpt.resolve_resume_iter(tmp_path) == 5
    monkeypatch.setenv("RESUME_ITER", "5")
    assert ckpt.resolve_resume_iter(tmp_path) == 5
    assert ckpt.resolve_resume_iter(tmp_path, resume_iter=5) == 5


def test_resolve_resume_iter_env_mismatch_exit4(tmp_path, monkeypatch, capsys):
    _write_marker(tmp_path, 5)
    monkeypatch.setenv("RESUME_ITER", "4")
    with pytest.raises(SystemExit) as e:
        ckpt.resolve_resume_iter(tmp_path)
    assert e.value.code == 4
    assert "marker 协议拒绝" in capsys.readouterr().err


def test_resolve_resume_iter_arg_mismatch_exit(tmp_path, monkeypatch):
    _write_marker(tmp_path, 5)
    monkeypatch.delenv("RESUME_ITER", raising=False)
    with pytest.raises(SystemExit):
        ckpt.resolve_resume_iter(tmp_path, resume_iter=4)


def test_load_marker_missing_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match="marker"):
        ckpt.load_marker(tmp_path)


def test_config_sha256_guard_contract(tmp_path, plain_fsdp):
    """宪法 6 防跑错：trainer.run 用 `cursor["config_sha256"] != cfg_sha` 拒绝启动，
    守卫本身在 trainer.py（e2e 覆盖）；本测试锁定 checkpoint 层契约——
    load_checkpoint 返回的 cursor 必须携带 config_sha256 供守卫比对。"""
    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    root = tmp_path / "checkpoints"
    ckpt.save_checkpoint(root, 3, model, [opt], {"consumed_samples": 12},
                         ckpt.get_rng_state(), "sha-A", "run", rank=0)
    cursor, _ = ckpt.load_checkpoint(root, 3, model, [opt], rank=0)
    assert cursor["config_sha256"] == "sha-A"
    # 换配置 → sha 不等 → trainer 守卫拒绝（此处验证判据可判定）
    assert cursor["config_sha256"] != "sha-B"


def test_interrupted_save_never_publishes_partial(tmp_path, plain_fsdp):
    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    root = tmp_path / "checkpoints"
    ckpt.save_checkpoint(root, 5, model, [opt], {"consumed_samples": 40},
                         ckpt.get_rng_state(), "sha-A", "run", rank=0)
    # 模拟 it=10 保存中断：tmp 目录残留、latest.json 未更新
    residue = root / f".tmp-ckpt-{10:08d}-{os.getpid()}"
    residue.mkdir(parents=True)
    (residue / "model.pt").write_bytes(b"partial-bytes")
    marker = ckpt.load_marker(root)
    assert marker["iter"] == 5
    assert marker["dir"] == "ckpt-00000005"
    with pytest.raises(FileNotFoundError, match="checkpoint 缺失"):
        ckpt.load_checkpoint(root, 10, model, [opt], rank=0)
    # 重存 it=10：清掉同名 tmp 残留后原子生效，marker 前进到 10
    ckpt.save_checkpoint(root, 10, model, [opt], {"consumed_samples": 80},
                         ckpt.get_rng_state(), "sha-A", "run", rank=0)
    assert ckpt.load_marker(root)["iter"] == 10
    cursor, _ = ckpt.load_checkpoint(root, 10, model, [opt], rank=0)
    assert cursor["cursor"] == {"consumed_samples": 80}
    assert not residue.exists()
    assert not list(root.glob(".tmp-*"))


def test_named_optim_state_dict_rejects_unknown_param():
    model = _tiny_model()
    foreign = torch.nn.Parameter(torch.randn(4, 4))
    opt = torch.optim.AdamW([foreign], lr=1e-3)
    (foreign**2).sum().backward()
    opt.step()
    with pytest.raises(KeyError, match="optimizer state"):
        ckpt._named_optim_state_dict(model, opt)
