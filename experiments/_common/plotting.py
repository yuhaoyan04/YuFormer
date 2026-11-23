"""E1 scaling law 拟合与图表（experiments/_common 共享基建，2026-10-08 实现）。

契约
====
- 拟合 L(N,D) = E + A/N^α + B/D^β：对 (α,β) 粗+细两阶段网格搜索，
  每个候选 (α,β,E) 下对 (A,B) 做**线性域**最小二乘（模型对 A/B 线性，
  正规方程 2×2 闭式解，非负约束取五种边界候选的 RMSE 最小者）。
  注：任务描述的 log(L-E) 联合回归是加法模型的近似写法，线性域解更精确且
  同样廉价，故采用线性域（E 网格搜索因此也不受 L-E>0 约束）。
- 置信带：parametric bootstrap（固定 seed=20261008，默认 200 次），
  y* = ŷ + RMSE·z, z~N(0,1)，每次重采样全网格重拟合，取 95% 分位带。
- 数据契约（与 tracker.py / trainer.py 对齐）：
  每臂目录 <arms_dir>/<arm>/ 下 metrics.jsonl + manifest.json。
  最终 loss 取值顺序：eval_val_loss → val_loss_* 均值（trainer 实际写的键）
  → final_loss → loss（均取 step 最大的记录）；
  N = manifest["params"]；D = manifest["config"]["train"]["budget_tokens"]。
- 产出：<out>/figures/E1-3.png、E1-4.png、fit_result.json（sort_keys，tmp+os.replace）。
- 依赖：numpy + 标准库（matplotlib 惰性导入；import 失败时 CLI 跳过画图并报 SKIP）。

用法：
  python3 plotting.py --arms-dir <dir> --out <dir> [--c-budget <FLOPs> ...]
  python3 plotting.py --selftest
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

SEED = 20261008
N_BOOT = 200
# 文献对照值：Chinchilla (Hoffmann 2022) 参数拟合 α/β；
# Kaplan 2020 单变量指数 α_N/α_D（计划文档 "0.76/0.09" 系 0.076/0.095 笔误）。
LIT_KAPLAN_2020 = {"alpha": 0.076, "beta": 0.095}
LIT_CHINCHILLA_2022 = {"alpha": 0.34, "beta": 0.28}

# 拟合网格默认值（规格：α,β ∈ [0.05, 1.2]；E ∈ [0, 0.5] 粗网格）
GRID_MIN, GRID_MAX = 0.05, 1.2
COARSE_STEP, FINE_STEP = 0.05, 0.01
E_RANGE = (0.0, 0.5)
E_COARSE_STEP, E_FINE_STEP = 0.02, 0.004

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class FitResult:
    """L(N,D) = E + A/N^α + B/D^β 拟合结果。"""

    E: float
    A: float
    alpha: float
    B: float
    beta: float
    rmse: float
    n_points: int
    residuals: np.ndarray = field(repr=False, default=None)  # type: ignore[assignment]

    def predict(self, N: Any, D: Any) -> np.ndarray:
        """按拟合参数预测 loss（支持标量/数组）。"""
        N = np.asarray(N, dtype=np.float64)
        D = np.asarray(D, dtype=np.float64)
        return self.E + self.A * N ** (-self.alpha) + self.B * D ** (-self.beta)

    def to_dict(self) -> dict[str, Any]:
        return {
            "E": round(self.E, 6),
            "A": round(self.A, 6),
            "alpha": round(self.alpha, 6),
            "B": round(self.B, 6),
            "beta": round(self.beta, 6),
            "rmse": round(self.rmse, 6),
            "n_points": self.n_points,
        }


# ---------------------------------------------------------------------------
# 网格拟合核心
# ---------------------------------------------------------------------------


def _solve_cell_scalar(
    N: np.ndarray, D: np.ndarray, L: np.ndarray, alpha: float, beta: float, E: float
) -> tuple[float, float, float]:
    """单个 (α,β,E) 下对 (A,B) 做非负线性最小二乘，返回 (A, B, rmse)。

    候选集合：{A,B 自由（若解非负）, B=0, A=0, A=B=0}，取 RMSE 最小者。
    """
    xa = N ** (-alpha)
    xb = D ** (-beta)
    y = L - E
    s_aa = float(xa @ xa)
    s_bb = float(xb @ xb)
    s_ab = float(xa @ xb)
    s_ay = float(xa @ y)
    s_by = float(xb @ y)
    sse0 = float(y @ y)
    n = len(N)

    best = (0.0, 0.0, sse0)  # (A, B, SSE)
    det = s_aa * s_bb - s_ab * s_ab
    if det > 1e-30:
        a = (s_bb * s_ay - s_ab * s_by) / det
        b = (s_aa * s_by - s_ab * s_ay) / det
        if a >= 0.0 and b >= 0.0:
            sse = float(np.sum ( (y - a * xa - b * xb) ** 2))
            best = (a, b, sse)
    if s_ay > 0.0 and s_aa > 0.0:  # B=0 边界
        a = s_ay / s_aa
        sse = sse0 - s_ay * a
        if sse < best[2]:
            best = (a, 0.0, sse)
    if s_by > 0.0 and s_bb > 0.0:  # A=0 边界
        b = s_by / s_bb
        sse = sse0 - s_by * b
        if sse < best[2]:
            best = (0.0, b, sse)
    rmse = math.sqrt(max(best[2], 0.0) / n)
    return best[0], best[1], rmse


def _solve_grid(
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    alphas: np.ndarray,
    betas: np.ndarray,
    es: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    """在 (α,β,E) 网格上全向量化求解，返回 (E, A, alpha, B, beta, rmse) 最优解。"""
    xa = N[None, :] ** (-alphas[:, None])  # (m, n)
    xb = D[None, :] ** (-betas[:, None])  # (k, n)
    s_aa = (xa * xa).sum(axis=1)  # (m,)
    s_bb = (xb * xb).sum(axis=1)  # (k,)
    s_ab = xa @ xb.T  # (m, k)
    b_aa = s_aa[:, None] * np.ones_like(s_ab)  # 广播视图
    b_bb = s_bb[None, :] * np.ones_like(s_ab)

    best: tuple[float, float, float, float, float, float] | None = None
    for e in es:
        y = L - e
        s_ay = xa @ y  # (m,)
        s_by = xb @ y  # (k,)
        sse0 = float(y @ y)

        # SSE(a,b) = Σy² - 2a·S_ay - 2b·S_by + 2ab·S_ab + a²·S_aa + b²·S_bb（展开式，免残差矩阵）
        det = s_aa[:, None] * s_bb[None, :] - s_ab * s_ab
        with np.errstate(divide="ignore", invalid="ignore"):
            a_sol = (s_bb[None, :] * s_ay[:, None] - s_ab * s_by[None, :]) / det
            b_sol = (s_aa[:, None] * s_by[None, :] - s_ab * s_ay[:, None]) / det
        sse_full = np.full_like(s_ab, np.inf)
        mask = (det > 1e-30) & (a_sol >= 0.0) & (b_sol >= 0.0)
        if mask.any():
            sse_full[mask] = (
                sse0
                - 2.0 * a_sol[mask] * np.broadcast_to(s_ay[:, None], s_ab.shape)[mask]
                - 2.0 * b_sol[mask] * np.broadcast_to(s_by[None, :], s_ab.shape)[mask]
                + 2.0 * a_sol[mask] * b_sol[mask] * s_ab[mask]
                + a_sol[mask] ** 2 * b_aa[mask]
                + b_sol[mask] ** 2 * b_bb[mask]
            )

        # 边界候选：B=0（A=S_ay/S_aa，要求 S_ay>0）、A=0（B=S_by/S_bb，要求 S_by>0）、A=B=0
        drop_a = np.where(s_ay > 0, s_ay**2 / np.maximum(s_aa, 1e-300), 0.0)
        drop_b = np.where(s_by > 0, s_by**2 / np.maximum(s_bb, 1e-300), 0.0)
        sse_b0 = sse0 - drop_a[:, None] * np.ones_like(s_ab)
        sse_a0 = sse0 - drop_b[None, :] * np.ones_like(s_ab)
        sse_cell = np.minimum(np.minimum(sse_full, sse_b0), np.minimum(sse_a0, sse0))

        idx = np.unravel_index(int(np.argmin(sse_cell)), sse_cell.shape)
        rmse = math.sqrt(max(float(sse_cell[idx]), 0.0) / len(N))
        if best is None or rmse < best[5]:
            a_f, b_f, _ = _solve_cell_scalar(N, D, L, float(alphas[idx[0]]), float(betas[idx[1]]), float(e))
            best = (float(e), a_f, float(alphas[idx[0]]), b_f, float(betas[idx[1]]), rmse)
    assert best is not None
    return best


def _refine_grid(
    best_a: float, best_b: float, best_e: float, *, e_half: float = 0.06
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在最优点附近生成细网格（步长 FINE_STEP / E_FINE_STEP，夹到全局范围）。

    E 细网格半宽取 0.06（≈3 个粗步长）：粗 (α,β) 分辨率不足时 E 会偏移 1-2 个
    粗步长，半宽 0.02 会把真值 E 排除在细网格外（自测合成数据即此情形）。
    """
    alphas = np.round(
        np.clip(np.arange(best_a - 0.05, best_a + 0.05 + 1e-9, FINE_STEP), GRID_MIN, GRID_MAX), 4
    )
    betas = np.round(
        np.clip(np.arange(best_b - 0.05, best_b + 0.05 + 1e-9, FINE_STEP), GRID_MIN, GRID_MAX), 4
    )
    es = np.round(np.clip(np.arange(best_e - e_half, best_e + e_half + 1e-9, E_FINE_STEP), *E_RANGE), 4)
    return np.unique(alphas), np.unique(betas), np.unique(es)


def fit_lnbd(
    points: Iterable[tuple[float, float, float]],
    *,
    e_range: tuple[float, float] = E_RANGE,
) -> FitResult:
    """拟合 L(N,D) = E + A/N^α + B/D^β。

    points: [(N, D, loss), ...]，至少 5 个点（5 参数）。
    (α,β) 粗网格 [0.05,1.2]×0.05 → 细网格 ±0.05×0.01（最多 3 轮细化，RMSE 不再
    相对下降 1e-10 即停）；E 粗 0.02 → 细 0.004（半宽 0.06）。
    返回 FitResult（含残差，供 bootstrap 使用）。
    """
    pts = [(float(n), float(d), float(l)) for n, d, l in points]
    if len(pts) < 5:
        raise ValueError(f"fit_lnbd 至少需要 5 个 (N,D,loss) 点，得到 {len(pts)}")
    N = np.array([p[0] for p in pts], dtype=np.float64)
    D = np.array([p[1] for p in pts], dtype=np.float64)
    L = np.array([p[2] for p in pts], dtype=np.float64)
    if (L <= 0).any():
        raise ValueError("loss 必须为正（拟合与双对数绘图前提）")

    alphas = np.round(np.arange(GRID_MIN, GRID_MAX + 1e-9, COARSE_STEP), 4)
    betas = np.round(np.arange(GRID_MIN, GRID_MAX + 1e-9, COARSE_STEP), 4)
    es = np.round(np.arange(e_range[0], e_range[1] + 1e-9, E_COARSE_STEP), 4)
    E_, A_, al_, B_, be_, rmse_ = _solve_grid(N, D, L, alphas, betas, es)
    for _ in range(3):
        fa, fb, fe = _refine_grid(al_, be_, E_)
        e_n, a_n, al_n, b_n, be_n, rm_n = _solve_grid(N, D, L, fa, fb, fe)
        if rmse_ - rm_n <= 1e-10 * max(rmse_, 1.0):
            break
        E_, A_, al_, B_, be_, rmse_ = e_n, a_n, al_n, b_n, be_n, rm_n

    fit = FitResult(
        E=E_, A=A_, alpha=al_, B=B_, beta=be_, rmse=rmse_, n_points=len(pts),
        residuals=L - (E_ + A_ * N ** (-al_) + B_ * D ** (-be_)),
    )
    return fit


def bootstrap_fits(
    fit: FitResult,
    points: Sequence[tuple[float, float, float]],
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> list[FitResult]:
    """parametric bootstrap：y* = ŷ + RMSE·z（z~N(0,1)），每次全网格重拟合。"""
    rng = np.random.default_rng(seed)
    pts = [(float(n), float(d), float(l)) for n, d, l in points]
    N = np.array([p[0] for p in pts])
    D = np.array([p[1] for p in pts])
    yhat = fit.predict(N, D)
    out: list[FitResult] = []
    for _ in range(n_boot):
        y_star = yhat + fit.rmse * rng.standard_normal(len(pts))
        out.append(fit_lnbd(list(zip(N.tolist(), D.tolist(), y_star.tolist()))))
    return out


def curve_band(
    fits: Sequence[FitResult], C: float, n_curve: int = 64
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """给定 bootstrap 拟合集，返回 C=6ND 约束下曲线带的网格与 95% 分位。

    返回 (N_grid, lo, hi, median)，loss = L(N, C/6N)。
    """
    lo_n, hi_n = _band_n_range(fits, C)
    n_grid = np.exp(np.linspace(math.log(lo_n), math.log(hi_n), n_curve))
    preds = np.stack([f.predict(n_grid, C / (6.0 * n_grid)) for f in fits])
    return (
        n_grid,
        np.percentile(preds, 2.5, axis=0),
        np.percentile(preds, 97.5, axis=0),
        np.percentile(preds, 50.0, axis=0),
    )


def _band_n_range(fits: Sequence[FitResult], C: float) -> tuple[float, float]:
    lo = min(f.A for f in fits if f.A > 0) if any(f.A > 0 for f in fits) else 1e6
    hi = max(f.A for f in fits)
    # 覆盖带在数据规模附近即可：用参数最优点的 ±10 倍范围
    centers = [optimal_n(f, C) for f in fits]
    c = float(np.median(centers))
    return c * 0.3, c * 3.0


def optimal_n(fit: FitResult, C: float) -> float:
    """参数闭式最优：dL(N, C/6N)/dN = 0 → N* = [(Aα/Bβ)·(C/6)^β]^{1/(α+β)}。"""
    if fit.A <= 0 or fit.B <= 0:
        raise ValueError("A/B 非正，无内部最优解（退化拟合）")
    return ((fit.A * fit.alpha) / (fit.B * fit.beta) * (C / 6.0) ** fit.beta) ** (
        1.0 / (fit.alpha + fit.beta)
    )


# ---------------------------------------------------------------------------
# isoflop 前沿与外推
# ---------------------------------------------------------------------------


def isoflop_frontier(
    points: Sequence[tuple[float, float, float]],
    C_values: Sequence[float],
    *,
    rel_tol: float = 0.03,
) -> list[dict[str, Any]]:
    """每条等算力线 C=6ND 上拟合 log(loss) vs log(N) 抛物线，取 argmin 得 N*(C)。

    至少需要 3 个点才能二次拟合；点不足的线条跳过（返回条目前带 "skipped": true）。
    返回 [{C, n_points, N_star, D_star, ratio_D_over_N, min_loss, points:[...]}]。
    """
    pts = [(float(n), float(d), float(l)) for n, d, l in points]
    out: list[dict[str, Any]] = []
    for C in C_values:
        group = [
            (n, d, l)
            for (n, d, l) in pts
            if abs(6.0 * n * d - C) <= rel_tol * C
        ]
        entry: dict[str, Any] = {"C": float(C), "n_points": len(group)}
        if len(group) < 3:
            entry["skipped"] = True
            out.append(entry)
            continue
        ns = np.array([g[0] for g in group])
        ls = np.array([g[2] for g in group])
        lx = np.log(ns)
        ly = np.log(ls)
        a2, a1, a0 = np.polyfit(lx, ly, 2)
        grid = np.exp(np.linspace(lx.min() - 0.15 * (lx.max() - lx.min() + 1e-9),
                                  lx.max() + 0.15 * (lx.max() - lx.min() + 1e-9), 512))
        vals = a2 * np.log(grid) ** 2 + a1 * np.log(grid) + a0
        i_star = int(np.argmin(vals))
        n_star = float(grid[i_star])
        d_star = float(C / (6.0 * n_star))
        entry.update(
            N_star=n_star,
            D_star=d_star,
            ratio_D_over_N=d_star / n_star,
            min_loss=float(math.exp(vals[i_star])),
            points=[[round(float(n), 1), round(float(d), 1), round(float(l), 6)] for n, d, l in group],
        )
        out.append(entry)
    return out


def extrapolate(
    fit: FitResult, C_budget: float, N_grid: Sequence[float]
) -> tuple[float, float, float]:
    """在 C=6ND 约束下 minimize L(N, C/6N)：网格 argmin + 邻域黄金分割细化。

    返回 (N*, D*, predicted_loss)。
    """
    grid = np.asarray([float(n) for n in N_grid], dtype=np.float64)
    if (grid <= 0).any():
        raise ValueError("N_grid 必须为正")
    losses = fit.predict(grid, C_budget / (6.0 * grid))
    i = int(np.argmin(losses))
    # 局部细化区间：最优网格点的相邻两格（边界处向外扩一格）
    if i == 0:
        lo, hi = grid[0] * 0.5, grid[min(1, len(grid) - 1)]
    elif i == len(grid) - 1:
        lo, hi = grid[max(len(grid) - 2, 0)], grid[-1] * 2.0
    else:
        lo, hi = grid[i - 1], grid[i + 1]

    def f(log_n: float) -> float:
        n = math.exp(log_n)
        return float(fit.predict(n, C_budget / (6.0 * n)))

    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = math.log(lo), math.log(hi)
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(80):
        if abs(b - a) < 1e-12:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    n_star = math.exp((a + b) / 2.0)
    d_star = C_budget / (6.0 * n_star)
    return n_star, d_star, float(fit.predict(n_star, d_star))


def lit_comparison_table(fit: FitResult) -> dict[str, Any]:
    """表 E1-2：我们的 α/β vs Kaplan 2020 vs Chinchilla 2022。"""
    return {
        "ours": {
            "alpha": round(fit.alpha, 4),
            "beta": round(fit.beta, 4),
            "E": round(fit.E, 4),
            "A": round(fit.A, 4),
            "B": round(fit.B, 4),
            "source": "fit_lnbd（本实验拟合）",
        },
        "kaplan2020": {**LIT_KAPLAN_2020, "source": "Kaplan et al. 2020（L(N)/L(D) 单变量指数）"},
        "chinchilla2022": {**LIT_CHINCHILLA_2022, "source": "Hoffmann et al. 2022（L(N,D) 参数拟合）"},
    }


# ---------------------------------------------------------------------------
# 数据收集（tracker.py / trainer.py 契约）
# ---------------------------------------------------------------------------


def final_loss(metrics: Sequence[dict[str, Any]]) -> tuple[float, str] | None:
    """从 metrics.jsonl 记录中取最终 loss，返回 (loss, 取值键名)。

    优先级：eval_val_loss → val_loss_* 均值（trainer 实际写入的键）→
    final_loss → loss；均取 step 最大的记录。
    """
    if not metrics:
        return None
    last = max(metrics, key=lambda r: int(r.get("step", -1)))

    def _mean_key(rec: dict[str, Any], prefix: str) -> float | None:
        vals = [v for k, v in rec.items() if k.startswith(prefix) and isinstance(v, (int, float))]
        return float(sum(vals) / len(vals)) if vals else None

    for key in ("eval_val_loss",):
        if isinstance(last.get(key), (int, float)):
            return float(last[key]), key
    mean_val = _mean_key(last, "val_loss_")
    if mean_val is not None:
        return mean_val, "val_loss_*"
    for key in ("final_loss", "loss"):
        if isinstance(last.get(key), (int, float)):
            return float(last[key]), key
    return None


def collect_arms(arms_dir: str | Path) -> list[dict[str, Any]]:
    """扫描 <arms_dir>/<arm>/，返回 [{arm, N, D, loss, loss_key}]（N=params，D=budget_tokens）。"""
    root = Path(arms_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"arms 目录不存在: {root}")
    out: list[dict[str, Any]] = []
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        m_metrics, m_manifest = arm_dir / "metrics.jsonl", arm_dir / "manifest.json"
        if not (m_metrics.is_file() and m_manifest.is_file()):
            continue
        manifest = json.loads(m_manifest.read_text(encoding="utf-8"))
        metrics = [
            json.loads(line)
            for line in m_metrics.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        got = final_loss(metrics)
        if got is None:
            print(f"warn: {arm_dir.name} metrics 中无可用 loss，跳过", file=sys.stderr)
            continue
        loss, loss_key = got
        N = manifest.get("params")
        D = manifest.get("config", {}).get("train", {}).get("budget_tokens")
        if N is None or D is None:
            print(f"warn: {arm_dir.name} manifest 缺 params/config.train.budget_tokens，跳过", file=sys.stderr)
            continue
        out.append(
            {
                "arm": arm_dir.name,
                "N": float(N),
                "D": float(D),
                "loss": float(loss),
                "loss_key": loss_key,
            }
        )
    if not out:
        raise SystemExit(f"{root} 下没有可用臂（需要 <arm>/metrics.jsonl + manifest.json）")
    return out


# ---------------------------------------------------------------------------
# 画图（matplotlib 惰性导入；风格对齐 experiments/e0_tokenizer/scripts/plot_e0.py）
# ---------------------------------------------------------------------------

PALETTE = ["#219ebc", "#023047", "#8ecae6", "#ffb703", "#fb8500", "#6a994e", "#bc4749"]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def fig_e13(
    arms: Sequence[dict[str, Any]],
    fit: FitResult,
    frontier: Sequence[dict[str, Any]],
    boot: Sequence[FitResult],
    out_path: str | Path,
) -> None:
    """图 E1-3：isoflop 曲线族（loss-N 散点 + 抛物线拟合 + L(N,D) 拟合曲线 + 95% 带）。"""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 5.8))
    lines = [f for f in frontier if not f.get("skipped")]
    on_frontier = {(p[0], p[1]) for f in lines for p in f["points"]}
    # 非等算力线臂（如 L(N) 固定 D=10B）以灰点标出（仍参与 L(N,D) 拟合）
    for a in arms:
        if (round(a["N"], 1), round(a["D"], 1)) not in on_frontier:
            ax.plot(a["N"], a["loss"], ".", color="#999999", ms=4)
    if any((round(a["N"], 1), round(a["D"], 1)) not in on_frontier for a in arms):
        ax.plot([], [], ".", color="#999999", label="L(N) 臂（D=10B，非等算力线）")
    for i, entry in enumerate(lines):
        color = PALETTE[i % len(PALETTE)]
        for (n, d, l) in entry["points"]:
            ax.plot(n, l, "o", color=color, ms=6)
        ns = np.exp(
            np.linspace(
                math.log(min(p[0] for p in entry["points"])) * 0.85,
                math.log(max(p[0] for p in entry["points"])) * 1.18,
                128,
            )
        )
        c = entry["C"]
        ds = c / (6.0 * ns)
        ax.plot(ns, fit.predict(ns, ds), "-", color=color, lw=1.8,
                label=f"C={c:.1e}：L(N,D) 拟合")
        if boot:
            n_band, lo, hi, med = curve_band(boot, c)
            ax.fill_between(n_band, lo, hi, color=color, alpha=0.15, lw=0)
            ax.plot(n_band, med, "--", color=color, lw=0.9)
        ax.plot(entry["N_star"], entry["min_loss"], "*", color=color, ms=14,
                label=f"C={c:.1e}：N*={entry['N_star'] / 1e6:.0f}M")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("模型参数 N")
    ax.set_ylabel("held-out loss")
    ax.set_title(
        "图 E1-3 isoflop 曲线族：等算力 C=6ND 下 loss vs N"
        f"（E={fit.E:.3f}, α={fit.alpha:.3f}, β={fit.beta:.3f}）"
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_e14(
    frontier: Sequence[dict[str, Any]],
    fit: FitResult,
    out_path: str | Path,
    c_extra: Sequence[float] = (),
) -> None:
    """图 E1-4：最优前沿 D*/N* vs 算力（双对数）+ Chinchilla ≈20 对照线。"""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    lines = [f for f in frontier if not f.get("skipped")]
    cs_obs = [f["C"] for f in lines]
    cs = sorted(set(cs_obs) | set(float(c) for c in c_extra))
    cs_curve = np.exp(np.linspace(math.log(min(cs) * 0.5), math.log(max(cs) * 5.0), 256))
    ratios = []
    for c in cs_curve:
        n_star = optimal_n(fit, float(c))
        ratios.append(c / (6.0 * n_star * n_star))
    ax.plot(cs_curve, ratios, "-", color="#219ebc", lw=1.8,
            label=f"参数拟合前沿（α={fit.alpha:.3f}, β={fit.beta:.3f}）")
    for entry in lines:
        ax.plot(entry["C"], entry["ratio_D_over_N"], "o", color="#023047", ms=7)
        ax.annotate(
            f"N*={entry['N_star'] / 1e6:.0f}M\nD/N={entry['ratio_D_over_N']:.1f}",
            (entry["C"], entry["ratio_D_over_N"]),
            textcoords="offset points", xytext=(6, -18), fontsize=7.5,
        )
    ax.axhline(20.0, ls="--", color="#fb8500", lw=1.2, label="Chinchilla ≈20 tokens/param")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("算力预算 C（FLOPs，C=6ND）")
    ax.set_ylabel("最优数据/参数比 D*/N*（tokens/param）")
    ax.set_title("图 E1-4 最优前沿：D*/N* vs 算力预算（点=抛物线拟合 argmin，线=参数拟合闭式解）")
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def atomic_write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name + ".tmp-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, p)


def cluster_c_values(
    points: Sequence[tuple[float, float, float]], *, rel_tol: float = 0.02
) -> tuple[list[float], int]:
    """把观测点按 C=6ND 相对容差贪心聚类成等算力线，返回 (线中心列表, 被跳过线条数)。

    L(N) 臂（D 恒定、每臂一条线）点不足 3 个无法二次拟合，由 frontier 标记 skipped，
    这里只负责聚类。
    """
    cs = sorted(6.0 * float(n) * float(d) for n, d, _ in points)
    clusters: list[list[float]] = []
    for c in cs:
        if clusters and abs(c - clusters[-1][-1]) <= rel_tol * clusters[-1][-1]:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    return [float(np.median(cl)) for cl in clusters], 0


def run_pipeline(
    arms: list[dict[str, Any]],
    out_dir: str | Path,
    c_budgets: Sequence[float] = (),
    *,
    make_figures: bool = True,
) -> dict[str, Any]:
    """拟合 + 前沿 + 外推 + 图表 + fit_result.json；返回结果字典。"""
    points = [(a["N"], a["D"], a["loss"]) for a in arms]
    fit = fit_lnbd(points)
    cs, _ = cluster_c_values(points)
    frontier_all = isoflop_frontier(points, cs)
    frontier = [f for f in frontier_all if not f.get("skipped")]
    n_lines_skipped = sum(1 for f in frontier_all if f.get("skipped"))
    boot = bootstrap_fits(fit, points)

    alpha_samples = sorted(f.alpha for f in boot)
    beta_samples = sorted(f.beta for f in boot)
    e_samples = sorted(f.E for f in boot)

    max_c = max(6.0 * a["N"] * a["D"] for a in arms)
    extrapolations: dict[str, Any] = {}
    # 外推网格覆盖观测 N 的 1/3 ~ 10 倍（10× 算力外推的最优点通常仍在界内）
    n_grid = np.exp(
        np.linspace(
            math.log(min(a["N"] for a in arms) / 3.0),
            math.log(max(a["N"] for a in arms) * 10.0),
            256,
        )
    )
    for label, c in [("auto_10x_max_c", max_c * 10.0)] + [
        (f"c_budget_{c:.2e}", float(c)) for c in c_budgets
    ]:
        n_star, d_star, pl = extrapolate(fit, c, n_grid)
        extrapolations[label] = {
            "C": c,
            "N_star": round(n_star, 1),
            "D_star": round(d_star, 1),
            "predicted_loss": round(pl, 6),
        }

    result: dict[str, Any] = {
        "arms": [
            {"arm": a["arm"], "N": a["N"], "D": a["D"], "loss": a["loss"], "loss_key": a["loss_key"]}
            for a in arms
        ],
        "fit": fit.to_dict(),
        "bootstrap": {
            "seed": SEED,
            "n_boot": len(boot),
            "alpha_p05": round(alpha_samples[len(alpha_samples) // 20], 4),
            "alpha_p50": round(alpha_samples[len(alpha_samples) // 2], 4),
            "alpha_p95": round(alpha_samples[-1 - len(alpha_samples) // 20], 4),
            "beta_p05": round(beta_samples[len(beta_samples) // 20], 4),
            "beta_p50": round(beta_samples[len(beta_samples) // 2], 4),
            "beta_p95": round(beta_samples[-1 - len(beta_samples) // 20], 4),
            "E_p05": round(e_samples[len(e_samples) // 20], 4),
            "E_p50": round(e_samples[len(e_samples) // 2], 4),
            "E_p95": round(e_samples[-1 - len(e_samples) // 20], 4),
        },
        "isoflop_frontier": frontier,
        "n_isoflop_lines_skipped": n_lines_skipped,
        "literature_comparison": lit_comparison_table(fit),
        "extrapolations": extrapolations,
    }

    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    if make_figures:
        fig_e13(arms, fit, frontier, boot, out / "figures" / "E1-3.png")
        fig_e14(frontier, fit, out / "figures" / "E1-4.png", [e["C"] for e in extrapolations.values()])
    atomic_write_json(out / "fit_result.json", result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="E1 scaling law 拟合与图表（experiments/_common）")
    p.add_argument("--arms-dir", help="臂结果根目录（<arm>/metrics.jsonl + manifest.json）")
    p.add_argument("--out", help="产出目录（figures/E1-3.png、E1-4.png、fit_result.json）")
    p.add_argument("--c-budget", type=float, action="append", default=[],
                   help="外推算力预算 FLOPs（可多次指定；另自动输出 10× 最大臂算力点）")
    p.add_argument("--selftest", action="store_true", help="合成数据自测（无需 GPU/网络/真实数据）")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.arms_dir or not args.out:
        raise SystemExit("--arms-dir 与 --out 必填（或 --selftest）")
    arms = collect_arms(args.arms_dir)
    result = run_pipeline(arms, args.out, args.c_budget)
    print(f"拟合完成：E={result['fit']['E']} α={result['fit']['alpha']} β={result['fit']['beta']} "
          f"RMSE={result['fit']['rmse']}（{len(arms)} 臂）")
    print(f"产出：{Path(args.out) / 'fit_result.json'}，figures/E1-3.png，figures/E1-4.png")
    return 0


# ---------------------------------------------------------------------------
# 自测：合成数据复现 Chinchilla 参数（19 臂 E1 网格），端到端 CLI
# ---------------------------------------------------------------------------


def _synthetic_arms() -> list[dict[str, Any]]:
    """按 E1 真实网格（5 L(N) + 14 isoflop）用已知参数生成无噪声数据。"""
    E_true, A_true, al_true, B_true, be_true = 0.4, 350.0, 0.34, 380.0, 0.28

    def L(n: float, d: float) -> float:
        return E_true + A_true / n ** al_true + B_true / d ** be_true

    n_by_scale = {"50m": 53e6, "100m": 105e6, "200m": 203e6, "400m": 411e6, "800m": 815e6}
    arms: list[dict[str, Any]] = []
    for scale, n in n_by_scale.items():
        arms.append({"arm": f"e1_ln_{scale}", "N": n, "D": 10e9, "loss": L(n, 10e9)})
    for ci, c in enumerate((4.8e18, 1.4e19, 4.8e19)):
        for scale, n in n_by_scale.items():
            d = int(c / (6 * n))
            if 0.5e9 <= d <= 100e9:
                arms.append({"arm": f"e1_iso_c{ci}_{scale}", "N": n, "D": float(d), "loss": L(n, d)})
    return arms


def selftest() -> int:
    """合成数据自测：参数复现、前沿、外推、CLI 端到端；matplotlib 缺失时 SKIP 图表。"""
    checks: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append(name)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            raise SystemExit(f"selftest 失败: {name} {detail}")

    arms = _synthetic_arms()
    check("synthetic_arms", len(arms) == 19, f"{len(arms)} 臂")

    points = [(a["N"], a["D"], a["loss"]) for a in arms]
    fit = fit_lnbd(points)
    tol = dict(E=0.02, A=30.0, alpha=0.02, B=30.0, beta=0.02)
    ok = (
        abs(fit.E - 0.4) <= tol["E"]
        and abs(fit.alpha - 0.34) <= tol["alpha"]
        and abs(fit.beta - 0.28) <= tol["beta"]
        and abs(fit.A - 350.0) <= tol["A"]
        and abs(fit.B - 380.0) <= tol["B"]
    )
    check(
        "fit_lnbd 参数复现",
        ok,
        f"E={fit.E:.4f}(真0.4) α={fit.alpha:.4f}(真0.34) β={fit.beta:.4f}(真0.28) RMSE={fit.rmse:.2e}",
    )

    cs, _ = cluster_c_values(points)
    # 合成网格聚类出 7 条线：3 条 isoflop + ln_800m 并入 4.8e19 线（差 1.9%）
    # + 4 条 L(N) 单点线（点不足 3 个，跳过）
    check("C 聚类", len(cs) == 7, f"{[f'{c:.2e}' for c in cs]}")
    frontier_all = isoflop_frontier(points, cs)
    good = [f for f in frontier_all if not f.get("skipped")]
    check("isoflop_frontier 可拟合 3 条 / 跳过 4 条",
          len(good) == 3 and sum(1 for f in frontier_all if f.get("skipped")) == 4)
    # 无噪声合成数据下，抛物线 argmin 应与参数闭式最优一致（一个数量级内）
    for entry in good:
        n_closed = optimal_n(fit, entry["C"])
        ratio = entry["N_star"] / n_closed
        check(
            f"N* 一致 C={entry['C']:.1e}",
            0.5 <= ratio <= 2.0,
            f"抛物线 N*={entry['N_star'] / 1e6:.0f}M vs 闭式 {n_closed / 1e6:.0f}M",
        )

    n_star, d_star, pl = extrapolate(fit, 4.8e20, [1e8, 2e8, 4e8, 8e8, 1.6e9])
    check(
        "extrapolate 45 天量级预算",
        n_star > 0 and abs(6 * n_star * d_star - 4.8e20) / 4.8e20 < 1e-6 and pl > fit.E,
        f"N*={n_star / 1e6:.0f}M D*={d_star / 1e9:.1f}B loss={pl:.3f}",
    )

    lit = lit_comparison_table(fit)
    check(
        "lit_comparison_table",
        set(lit) == {"ours", "kaplan2020", "chinchilla2022"}
        and lit["chinchilla2022"]["alpha"] == 0.34
        and lit["chinchilla2022"]["beta"] == 0.28
        and lit["kaplan2020"]["alpha"] == 0.076,
    )

    boot = bootstrap_fits(fit, points, n_boot=20, seed=SEED)
    check("bootstrap 确定性", len(boot) == 20 and boot[0].alpha == bootstrap_fits(fit, points, n_boot=1, seed=SEED)[0].alpha)

    # CLI 端到端：写假臂目录 → run_pipeline → 校验产出
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        arms_root = Path(td) / "arms"
        for a in arms:
            d = arms_root / a["arm"]
            d.mkdir(parents=True)
            rec = {"step": 100, "wallclock": 1.0, "loss": 9.9, "val_loss_synth": a["loss"]}
            (d / "metrics.jsonl").write_text(json.dumps(rec, sort_keys=True) + "\n", encoding="utf-8")
            manifest = {
                "arm_name": a["arm"],
                "seed": 20261001,
                "params": a["N"],
                "config": {"train": {"budget_tokens": int(a["D"])}},
            }
            atomic_write_json(d / "manifest.json", manifest)
        out_dir = Path(td) / "out"
        result = run_pipeline(collect_arms(arms_root), out_dir, [4.8e20], make_figures=False)
        fit_json = json.loads((out_dir / "fit_result.json").read_text(encoding="utf-8"))
        check(
            "CLI fit_result.json",
            fit_json == json.loads(json.dumps(result, sort_keys=True))
            and "c_budget_4.80e+20" in fit_json["extrapolations"],
        )
        check(
            "final_loss 取值链（val_loss_*）",
            all(a["loss_key"] == "val_loss_*" for a in collect_arms(arms_root)),
        )
        # matplotlib 可能因环境问题不可用（如与 numpy 二进制不兼容）：SKIP 图表断言
        try:
            import matplotlib  # noqa: F401

            pts = [(a["N"], a["D"], a["loss"]) for a in arms]
            fit_st = fit_lnbd(pts)
            fig_e13(arms, fit_st, result["isoflop_frontier"], [], out_dir / "figures" / "E1-3.png")
            fig_e14(result["isoflop_frontier"], fit_st, out_dir / "figures" / "E1-4.png")
            check(
                "图表产出 E1-3/E1-4",
                (out_dir / "figures" / "E1-3.png").is_file()
                and (out_dir / "figures" / "E1-4.png").is_file(),
            )
            mpl_ok = True
        except ImportError as exc:
            mpl_ok = False
            print(f"[SKIP] 图表产出 E1-3/E1-4 — matplotlib 不可用：{exc}")

    print(f"selftest 完成：{len(checks)} 项 PASS"
          + ("（图表部分 SKIP：matplotlib 不可用）" if not mpl_ok else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
