"""Muon 单步/短程 smoke：随机模型上 loss 下降、无 NaN、步进确定。"""
import torch

from yuformer_pretrain.muon import Muon, zeropower_via_newtonschulz5


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
    )


def _fixed_batch(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(64, 32, generator=g)


def test_newton_schulz_orthogonalizes():
    torch.manual_seed(0)
    G = torch.randn(48, 16)
    X = zeropower_via_newtonschulz5(G)
    # NS5 把奇异值压到 ~1 附近（已知区间约 [0.7, 1.15]）
    sv = torch.linalg.svdvals(X)
    assert 0.6 < sv.min().item() and sv.max().item() < 1.3


def test_muon_step_loss_decreases_no_nan():
    model = _tiny_model()
    opt = Muon(model.parameters(), lr=0.02, momentum=0.95, weight_decay=0.0)
    x = _fixed_batch()
    losses = []
    for _ in range(20):
        loss = torch.nn.functional.mse_loss(model(x), x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0]


def test_muon_deterministic():
    losses_runs = []
    for run in range(2):
        model = _tiny_model(seed=7)
        opt = Muon(model.parameters(), lr=0.02)
        x = _fixed_batch()
        losses = []
        for _ in range(5):
            loss = torch.nn.functional.mse_loss(model(x), x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        losses_runs.append(losses)
    assert losses_runs[0] == losses_runs[1]
