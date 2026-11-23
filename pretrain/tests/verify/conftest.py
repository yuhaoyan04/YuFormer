"""原子化验证套件公共夹具（V1-V5）。

资源契约：GPU 验证须显式启用，并使用配置选定的空闲 GPU。
普通 pytest/CPU 检查不启动 GPU 子任务；每个训练子任务峰值显存受预算约束。

报告：每个测试经 `vreport` fixture 落 JSON 片段到 VERIFY_OUT/report.d/，
并合并进 tests/verify/verify_report.json（含 PASS/FAIL 状态）。
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

log = logging.getLogger("yuformer.verify")

REPO_ROOT = Path(__file__).resolve().parents[3]
PRETRAIN_SRC = REPO_ROOT / "pretrain" / "src"
MODEL_SRC = REPO_ROOT / "model" / "src"
MODEL_CONFIGS = REPO_ROOT / "model" / "configs" / "model_configs.yaml"
TRACKER_ROOT = REPO_ROOT / "experiments" / "_common"

VERIFY_OUT = Path(os.environ.get("YUFORMER_VERIFY_OUT", "/tmp/yuformer_verify"))
REPORT_DIR = VERIFY_OUT / "report.d"
REPORT_PATH = Path(__file__).resolve().parent / "verify_report.json"

VGPU = os.environ.get("YUFORMER_VERIFY_GPU", "0")
VRAM_BUDGET_MB = 3200


def gpu_verification_enabled() -> bool:
    return (os.environ.get("YUFORMER_RUN_GPU_VERIFY") == "1"
            and os.environ.get("CUDA_VISIBLE_DEVICES") != "")


def pytest_collection_modifyitems(config, items):
    if gpu_verification_enabled():
        return
    directory = Path(__file__).resolve().parent
    for item in items:
        if Path(str(item.fspath)).resolve().is_relative_to(directory):
            item.add_marker(pytest.mark.skip(
                reason="GPU integration suite requires YUFORMER_RUN_GPU_VERIFY=1 and an idle GPU"))


def pytest_sessionstart(session):
    """A formal suite run starts from an empty evidence set.

    Individual test invocations keep their diagnostic fragments.  ``run_all``
    opts into reset so stale V6 or earlier-run fragments cannot contaminate the
    V1-V5 acceptance report.
    """
    if os.environ.get("YUFORMER_VERIFY_RESET") != "1":
        return
    shutil.rmtree(REPORT_DIR, ignore_errors=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(
            {
                "suite": "yuformer-pretrain-verify",
                "scope": os.environ.get("YUFORMER_VERIFY_SCOPE", "unspecified"),
                "status": "RUNNING",
                "items": {},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def pytest_sessionfinish(session, exitstatus):
    if os.environ.get("YUFORMER_VERIFY_RESET") != "1":
        return
    _merge_report()
    report = json.loads(REPORT_PATH.read_text())
    failed = sorted(
        name for name, item in report["items"].items() if item.get("status") != "PASS"
    )
    report.update(
        {
            "status": "PASS" if exitstatus == 0 and report["items"] and not failed else "FAIL",
            "pytest_exitstatus": int(exitstatus),
            "passed": len(report["items"]) - len(failed),
            "failed": failed,
        }
    )
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


def wait_gpu_free(min_mb: int, timeout_s: int = 1800) -> int:
    """Wait for an exclusively idle GPU with enough memory."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        used = subprocess.run(
            ["nvidia-smi", "-i", VGPU, "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip()
        used_mb, total_mb = (int(value.strip()) for value in used.split(","))
        processes = subprocess.run(
            ["nvidia-smi", "-i", VGPU, "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout.strip()
        free = total_mb - used_mb
        if used_mb <= 1024 and not processes and free >= min_mb:
            return free
        time.sleep(20)
    raise TimeoutError(f"GPU {VGPU} did not become exclusively idle within {timeout_s}s")


def _proc_gpu_mem_mb(pid: int) -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15,
    ).stdout
    for line in out.splitlines():
        p, m = line.split(", ")
        if int(p) == pid:
            return int(m)
    return 0


class GpuPeakMonitor:
    """后台轮询【本进程】GPU 显存占用（nvidia-smi 按进程归因），取峰值。

    按 GPU 总量差值会被同卡邻居（E0 臂等其它租户）的分配抖动污染
    （V4 首跑实测邻居噪声 >400MB）；按 PID 归因才符合 <3GB 资源契约。"""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.peak = 0
        self._stop = threading.Event()
        self._th: threading.Thread | None = None

    def __enter__(self) -> "GpuPeakMonitor":
        def _poll() -> None:
            while not self._stop.is_set():
                try:
                    self.peak = max(self.peak, _proc_gpu_mem_mb(self.pid))
                except Exception:
                    pass
                self._stop.wait(0.4)

        self._th = threading.Thread(target=_poll, daemon=True)
        self._th.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._th:
            self._th.join(timeout=5)


def make_token_stream(
    directory: Path,
    name: str,
    n_tokens: int,
    seed: int = 0,
    pattern: str | int = "random",
    vocab: int = 32768,
) -> tuple[Path, Path]:
    """合成 token 流（契约同 scripts/materialize_tokens.py 产物）。

    pattern: "random" 均匀随机 | "chain" 可学习确定性链（a'=(7a+13)%512 掺 10% 噪声）
             | int 常量 token（V3 课程序检测用）。
    """
    from yuformer_pretrain.data import _edge_sha256

    rng = np.random.default_rng(seed)
    if pattern == "chain":
        arr = np.empty(n_tokens, dtype=np.uint16)
        arr[0] = rng.integers(0, 512)
        noise = rng.random(n_tokens) < 0.1
        for i in range(1, n_tokens):
            a = int(arr[i - 1])
            arr[i] = rng.integers(0, vocab) if noise[i] else (7 * (a % 512) + 13) % 512
    elif isinstance(pattern, int):
        arr = np.full(n_tokens, pattern, dtype=np.uint16)
    else:
        arr = rng.integers(0, vocab, size=n_tokens, dtype=np.uint16)
    bin_path = directory / f"{name}.tokens.bin"
    arr.tofile(bin_path)
    import hashlib as hl

    h = hl.sha256()
    with open(bin_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    meta = {
        "name": name,
        "total_tokens": int(n_tokens),
        "bytes": int(bin_path.stat().st_size),
        "sha256": h.hexdigest(),
        "edge_sha256": _edge_sha256(bin_path),
        "tokenizer_sha256": "synthetic-verify",
        "tokenizer": "synthetic",
        "source": f"synthetic:{name}",
        "vocab_size": vocab,
        "n_docs": 0,
    }
    meta_path = directory / f"{name}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return bin_path, meta_path


def write_config(
    path: Path,
    *,
    run_name: str,
    out_dir: Path,
    stages: list[Path],
    metas: list[Path],
    seq_len: int,
    budget_tokens: int,
    micro_batch_size: int,
    grad_accum: int,
    optimizer: str = "adamw",
    lr: float = 2e-4,
    warmup_steps: int = 10,
    lr_schedule: str = "constant",
    min_lr_ratio: float = 0.1,
    epochs: int = 10,
    seed: int = 1234,
    model_preset: str = "tiny",
    model_configs: Path = MODEL_CONFIGS,
    activation_checkpointing: bool = False,
    bf16: bool = True,
    eval_every: int = 0,
    log_every: int = 1,
    save_every: int = 0,
    world_size: int = 1,
    val_dir: Path | None = None,
    init_weights_from: str | None = None,
) -> Path:
    tokens_per_step = seq_len * micro_batch_size * grad_accum * world_size
    assert budget_tokens % tokens_per_step == 0, "budget 必须整除 tokens_per_step"
    cfg = {
        "run_name": run_name,
        "seed": seed,
        "model": {"preset": model_preset, "configs": str(model_configs)},
        "out_dir": str(out_dir),
        **({"init_weights_from": init_weights_from} if init_weights_from else {}),
        "data": {
            "stages": [str(s) for s in stages],
            "meta": [str(m) for m in metas],
            "val_dir": str(val_dir) if val_dir else "",
            "seq_len": seq_len,
            "epochs": epochs,
        },
        "optim": {
            "optimizer": optimizer,
            "lr": lr,
            "weight_decay": 0.1,
            "warmup_steps": warmup_steps,
            "lr_schedule": lr_schedule,
            "min_lr_ratio": min_lr_ratio,
            "adam_betas": [0.9, 0.95],
            "muon_momentum": 0.95,
            "muon_nesterov": True,
            "muon_ns_steps": 5,
            "clip_grad": 1.0,
        },
        "train": {
            "budget_tokens": budget_tokens,
            "micro_batch_size": micro_batch_size,
            "grad_accum": grad_accum,
            "expected_iters": budget_tokens // tokens_per_step,
            "activation_checkpointing": activation_checkpointing,
            "bf16": bf16,
            "eval_every": eval_every,
            "log_every": log_every,
            "save_every": save_every,
        },
        "parallel": {"world_size": world_size, "sharding": "no_shard"},
    }
    import yaml

    p = path / f"{run_name}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return p


def run_trainer(
    cfg_path: Path,
    extra: list[str] | None = None,
    env_extra: dict | None = None,
    timeout: int = 2400,
    attempts: int = 2,
) -> tuple[subprocess.CompletedProcess, int, Path]:
    """子进程跑 trainer，返回 (proc, 峰值显存 MB, log_path)。

    显存预算硬断言在此。共享 vGPU 上瞬时显存空窗（邻居/兄弟任务峰值、死进程泄漏
    不可回收——E-23 同族）会随机以 OutOfMemoryError / CUBLAS_STATUS_ALLOC_FAILED /
    unspecified launch failure 砸在训练进程上：对这三类错误自动重试（默认 2 次尝试），
    其余失败（如 marker 负向用例）不重试、原样返回。
    """
    if not gpu_verification_enabled():
        pytest.skip("GPU integration is disabled; refusing to override the parent GPU mask")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and VGPU not in [value.strip() for value in visible.split(",")]:
        raise ValueError("verification GPU is outside CUDA_VISIBLE_DEVICES")
    if env_extra and "CUDA_VISIBLE_DEVICES" in env_extra and env_extra["CUDA_VISIBLE_DEVICES"] != VGPU:
        raise ValueError("verification subprocess cannot override the selected GPU")
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": VGPU,
        "PYTHONPATH": f"{PRETRAIN_SRC}:{MODEL_SRC}:{env.get('PYTHONPATH', '')}",
        "YUFORMER_TRACKER_ROOT": str(TRACKER_ROOT),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # trainer 默认 MASTER_PORT=29517 为硬编码（launch.sh 契约显式传）——验证套件
        # 侧按 PID 取端口，避免与同机其它训练臂（E0 等）撞车（实测撞过：errno 98）
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(29517 + (os.getpid() % 500)),
        "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0",
    })
    if env_extra:
        env.update(env_extra)
    cmd = [sys.executable, "-m", "yuformer_pretrain.trainer", "--config", str(cfg_path)]
    if extra:
        cmd += extra
    # 日志落文件而非 PIPE：PrefetchLoader fork worker 会继承 PIPE 写端，trainer 退出后
    # communicate() 永远等不到 EOF（V1 实测挂死 20 分钟）；文件重定向无此问题。
    log_path = cfg_path.with_suffix(".log")
    retryable = ("OutOfMemoryError", "CUBLAS_STATUS_ALLOC_FAILED", "unspecified launch failure")
    proc: subprocess.CompletedProcess | None = None
    peak = 0
    for attempt in range(1, attempts + 1):
        wait_gpu_free(VRAM_BUDGET_MB // 2, timeout_s=1800)  # 点火前置：至少一半预算的空窗
        with open(log_path, "w", encoding="utf-8") as log_f:
            popen = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
            with GpuPeakMonitor(popen.pid) as mon:
                popen.wait(timeout=timeout)
        out = log_path.read_text(errors="replace")
        # 显存峰值口径：优先子进程自报（torch.cuda.max_memory_allocated——vGPU 环境下
        # nvidia-smi 按 PID 归因对新进程失效、按 GPU 总量差值被邻居噪声污染，两者均实测不可信；
        # 自报缺省时退回按 PID 轮询峰值）
        peak = mon.peak
        for line in out.splitlines():
            if "VRAM_REPORT" in line:
                for tok in line.split():
                    if tok.startswith("peak_allocated_mb="):
                        peak = max(peak, int(tok.split("=", 1)[1]))
        proc = subprocess.CompletedProcess(cmd, popen.returncode, out[-6000:], out[-6000:])
        if proc.returncode == 0 or attempt == attempts or not any(s in out for s in retryable):
            break
        log.warning("run_trainer: %s 第 %d 次尝试被显存空窗击毁，重试…", cfg_path.name, attempt)
    if peak > VRAM_BUDGET_MB and proc.returncode == 0:
        raise AssertionError(f"本进程峰值显存 {peak}MB 超预算 {VRAM_BUDGET_MB}MB（训练虽成功但违反资源契约）")
    assert proc is not None
    return proc, peak, log_path


def read_metrics(out_dir: Path, run_name: str) -> dict[int, dict]:
    """metrics 落盘路径 = <out_dir>/results/<run_name>/metrics.jsonl
    （checkpoint 在 <out_dir>/checkpoints/，与 run_name 平级——trainer 布局）。"""
    metrics = out_dir / "results" / run_name / "metrics.jsonl"
    out: dict[int, dict] = {}
    if metrics.is_file():
        for line in metrics.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                # 同 step 多记录（iter 末 loss 记录 + final 汇总记录）合并，避免互相覆盖
                out[int(r["step"])] = {**out.get(int(r["step"]), {}), **r}
    return out


def read_manifest(out_dir: Path, run_name: str) -> dict:
    return json.loads((out_dir / "results" / run_name / "manifest.json").read_text())


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    setattr(item, f"rep_{rep.when}", rep)


@pytest.fixture
def vreport(request):
    """每项验证的 JSON 报告片段，测试结束（含失败）合并入 verify_report.json。"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    data: dict = {"name": request.node.name, "status": "PASS", "checks": {}}
    yield data
    if request.node.rep_call.failed if hasattr(request.node, "rep_call") else False:
        data["status"] = "FAIL"
        data["error"] = str(request.node.rep_call.longrepr)[:2000]
    frag = REPORT_DIR / f"{request.node.name}.json"
    frag.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, default=str))
    _merge_report()


def _merge_report() -> None:
    report: dict = {"suite": "yuformer-pretrain-verify", "items": {}}
    if REPORT_PATH.is_file():
        report = json.loads(REPORT_PATH.read_text())
    for frag in sorted(REPORT_DIR.glob("*.json")):
        d = json.loads(frag.read_text())
        report["items"][d["name"]] = {k: v for k, v in d.items() if k != "name"}
    temporary = REPORT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, REPORT_PATH)
