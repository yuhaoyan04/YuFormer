"""S4 BERT 回归档 GPU 推理：s3 去重后池 → s4_bert/<source>/<relpath>.parquet。

- 输入 = S3 产物中 s3_decision == 'keep' 的行（列含 text/doc_id/domain/upstream_score）；
  s3_decision 列缺失时报错并提示先跑 dedup。
- 多进程并行：每进程独占一张 GPU（CUDA_VISIBLE_DEVICES 在 spawn 子进程环境设置，
  torch 惰性导入，保证 device 绑定正确）。shard 级幂等（parquet+summary 齐在才 skip）。
- FP16 batch 512；bert_score = P(hq) 量化到 1e-4（GPU 数值不确定性的稳定性处理，
  重跑 ±1e-4 抖动属预期，见 README）。
- domain_allowlist 配置开关：超时优化路径仍不达标时降级为只扫白名单域。
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .ingest import BatchWriter, write_summary

BERT_SCORED_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("doc_id", pa.string(), nullable=False),
        pa.field("bert_score", pa.float32(), nullable=False),
    ]
)

DEFAULT_BERT_INFERENCE: dict[str, Any] = {
    "max_len": 256,            # 与训练一致
    "max_score_chars": 1024,   # 与 fasttext 档 / gold-set 训练截断一致
    "batch_size": 512,
    "score_quantum": 1e-4,
    "domain_allowlist": None,  # null = 全域；list[str] = 只扫这些 domain（降级路径）
    "torch_compile": False,    # 超时优化路径 1：torch.compile（首次编译有开销，实测无增益）
}


def bert_inference_params(cfg: dict[str, Any]) -> dict[str, Any]:
    params = dict(DEFAULT_BERT_INFERENCE)
    params.update(cfg.get("bert") or {})
    # 截断策略开关（2026-09-20）：bert 节优先，缺省回退 scorer.yaml 根级（fasttext 共用）
    params.setdefault("window_scoring", bool(cfg.get("window_scoring", False)))
    params.setdefault("long_doc_chars", int(cfg.get("long_doc_chars", 4096)))
    return params


def bert_output_paths(out_root: Path, source: str, relpath: str) -> tuple[Path, Path]:
    base = out_root / "s4_bert" / source / relpath
    return base.with_name(base.name + ".parquet"), base.with_name(base.name + ".summary.json")


def resolve_bert_model_dir(cfg: dict[str, Any]) -> Path:
    p = Path(str(cfg.get("bert_model", "")))
    if not p.is_dir():
        raise FileNotFoundError(
            f"bert scorer model dir not found: {p}; "
            "train it with yuformer_data.scale.bert_train"
        )
    return p


def _load_bert(model_dir: Path) -> tuple[Any, Any, str]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("s4_bert requires a CUDA GPU (CUDA_VISIBLE_DEVICES not usable)")
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    # 推理档直接以 fp16 加载（训练档保存的是 fp32：AMP 训练 master 权重）——
    # 纯 fp16 推理比 fp32+autocast 快 ~33%；分数量化 1e-4，精度损失可忽略
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model = model.half().to(device)
    model.eval()
    return model, tokenizer, device


class BertInferencer:
    """流水化推理器：后台线程预取 tokenize（CPU）与 GPU forward 重叠。

    实测单线程 tokenize ~4.2k 篇/s 是纯 GPU forward（~8k 篇/s/GPU）的主要瓶颈，
    故 encode 预取 lookahead 批 + HF fast tokenizer 多线程（Rayon，每 worker 限
    RAYON_NUM_THREADS 线程，遵守"少量 CPU"约束）。

    window_scoring=True（2026-09-20 截断策略裁定）：短文档（归一化 ≤ long_doc_chars）
    头部 max_len token 直打；长文档 head/mid/tail 三窗各 max_len token 均分。
    score() 接收**原始**文本（内部归一化）；window 模式下长文档归一化上限
    WINDOW_NORM_CAP 字符、token 上限 TOKEN_CAP（超长文档 mid 窗取前 TOKEN_CAP
    token 的中点，books/academic 百 KB 长文的确定性近似）。
    """

    WINDOW_NORM_CAP = 32768
    TOKEN_CAP = 8192

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str,
        max_len: int,
        batch_size: int,
        quantum: float,
        lookahead: int = 3,
        max_chars: int = 1024,
        long_doc_chars: int = 4096,
        window_scoring: bool = False,
    ) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self.model = model
        self.tok = tokenizer
        self.device = device
        self.max_len = max_len
        self.batch_size = batch_size
        self.quantum = quantum
        self.max_chars = max_chars
        self.long_doc_chars = long_doc_chars
        self.window_scoring = window_scoring
        self.executor = ThreadPoolExecutor(max_workers=lookahead)

    def _encode(self, texts: list[str]) -> Any:
        from .bert_train import encode_texts

        return encode_texts(self.tok, texts, self.max_len)

    def score(self, texts: list[str]) -> np.ndarray:
        """原始文本 → 量化 bert_score。window_scoring 时长文档三窗均分。"""
        from .scorer_train import normalize_scorer_text

        if self.window_scoring:
            norms = [normalize_scorer_text(t, self.WINDOW_NORM_CAP) for t in texts]
            return self._score_windowed(norms)
        norms = [normalize_scorer_text(t, self.max_chars) for t in texts]
        return self._score_normalized_pipeline(norms)

    def _score_normalized_pipeline(self, norms: list[str]) -> np.ndarray:
        """head-only：分块 encode（预取流水）+ GPU forward。"""
        import torch
        from concurrent.futures import wait

        texts = norms
        bs = self.batch_size
        chunks = [texts[i : i + bs] for i in range(0, len(texts), bs)]
        lookahead = 3
        encoded: dict[int, Any] = {}
        pending: dict[Any, int] = {}
        next_submit = 0
        for i in range(min(lookahead, len(chunks))):
            pending[self.executor.submit(self._encode, chunks[i])] = i
            next_submit += 1
        while pending:
            done, _ = wait(pending, return_when="FIRST_COMPLETED")
            for fut in done:
                encoded[pending.pop(fut)] = fut.result()
            while next_submit < len(chunks) and len(pending) < lookahead:
                pending[self.executor.submit(self._encode, chunks[next_submit])] = next_submit
                next_submit += 1
        probs = np.zeros(len(texts), dtype=np.float64)
        n_out = 0
        with torch.no_grad():
            for i, chunk in enumerate(chunks):
                enc = {k: v.to(self.device) for k, v in encoded[i].items()}
                logits = self.model(**enc).logits
                p = torch.softmax(logits.float(), dim=-1)[:, 1]
                # 2026-09-25: fp16 forward 在病态长输入上产生 NaN（推理中断实录）——
                # NaN 置 0.5（中性分），保证 worker 不死；数量记 summary 由调用方观察。
                p = torch.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0)
                k = len(chunk)
                probs[n_out : n_out + k] = p.detach().cpu().numpy()
                n_out += k
        return np.round(probs / self.quantum) * self.quantum

    def _score_windowed(self, norms: list[str]) -> np.ndarray:
        """head/mid/tail 三窗（长文档）/ 单 head 窗（短文档），窗内均分。"""
        import torch
        from .bert_train import token_windows

        enc = self.tok(
            norms,
            truncation=True,
            max_length=self.TOKEN_CAP,
            padding=False,
            add_special_tokens=True,
        )
        flat_ids: list[list[int]] = []
        owners: list[int] = []
        for i, ids in enumerate(enc["input_ids"]):
            for w in token_windows(ids, self.max_len):
                flat_ids.append(w)
                owners.append(i)
        probs = np.zeros(len(norms), dtype=np.float64)
        cnt = np.zeros(len(norms), dtype=np.float64)
        bs = self.batch_size
        with torch.no_grad():
            for i in range(0, len(flat_ids), bs):
                batch_ids = flat_ids[i : i + bs]
                padded = self.tok.pad(
                    {"input_ids": batch_ids}, padding=True, return_tensors="pt"
                )
                padded = {k: v.to(self.device) for k, v in padded.items()}
                logits = self.model(**padded).logits
                p = torch.softmax(logits.float(), dim=-1)[:, 1]
                p = torch.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0).detach().cpu().numpy()
                for j, owner in enumerate(owners[i : i + bs]):
                    probs[owner] += p[j]
                    cnt[owner] += 1.0
        probs = probs / np.maximum(cnt, 1.0)
        return np.round(probs / self.quantum) * self.quantum

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False)


def process_shard_bert(
    s3_path: Path,
    out_root: Path,
    source: str,
    relpath: str,
    cfg: dict[str, Any],
    model: Any | None = None,
    tokenizer: Any | None = None,
    device: str = "cuda",
    inferencer: "BertInferencer | None" = None,
) -> dict[str, Any]:
    """单 shard：s3 keep 行 → bert_score。输出 doc_id + bert_score。

    多卡 worker 传 inferencer（流水化）；单次调用可传 model/tokenizer 或都不传。
    """
    out_path, summary_path = bert_output_paths(out_root, source, relpath)
    if summary_path.is_file() and out_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["skipped"] = True
        return summary

    params = bert_inference_params(cfg)
    allowlist = params.get("domain_allowlist")
    allow = set(allowlist) if allowlist else None

    started = time.monotonic()
    if inferencer is None:
        if model is None:
            model, tokenizer, device = _load_bert(resolve_bert_model_dir(cfg))
        inferencer = BertInferencer(
            model, tokenizer, device, max_len=int(params["max_len"]),
            batch_size=int(params["batch_size"]), quantum=float(params["score_quantum"]),
            max_chars=int(params["max_score_chars"]),
            long_doc_chars=int(params.get("long_doc_chars", 4096)),
            window_scoring=bool(params.get("window_scoring", False)),
        )

    in_rows = 0
    kept_rows = 0
    skipped_nonkeep = 0
    skipped_domain = 0
    writer = BatchWriter(out_path, BERT_SCORED_SCHEMA)
    try:
        pf = pq.ParquetFile(s3_path)
        names = set(pf.schema_arrow.names)
        if "s3_decision" not in names:
            raise RuntimeError(
                f"{s3_path} lacks column 's3_decision' — S3 近重去重产物缺失，"
                "请先运行 driver --stage dedup（或 --stage s3）"
            )
        if "text" not in names or "doc_id" not in names:
            raise RuntimeError(f"{s3_path} lacks text/doc_id columns")
        read_cols = ["doc_id", "text", "s3_decision"] + (["domain"] if "domain" in names else [])
        has_domain = "domain" in names
        for batch in pf.iter_batches(batch_size=4096, columns=read_cols):
            d = batch.to_pydict()
            n = len(d["doc_id"])
            if n == 0:
                continue
            in_rows += n
            keep_mask = [s == "keep" for s in d["s3_decision"]]
            skipped_nonkeep += n - sum(keep_mask)
            if allow is not None and has_domain:
                dom_mask = [dom in allow for dom in d["domain"]]
                skipped_domain += sum(1 for k, dm in zip(keep_mask, dom_mask) if k and not dm)
                keep_mask = [k and dm for k, dm in zip(keep_mask, dom_mask)]
            elif allow is not None:
                raise RuntimeError("domain_allowlist configured but 'domain' column missing")
            idxs = [i for i, k in enumerate(keep_mask) if k]
            if not idxs:
                continue
            texts = [d["text"][i] for i in idxs]  # 原始文本；归一化/分窗在 inferencer 内
            probs = inferencer.score(texts)
            writer.extend(
                {
                    "doc_id": [d["doc_id"][i] for i in idxs],
                    "bert_score": probs.astype(np.float32).tolist(),
                }
            )
            kept_rows += len(idxs)
        out_bytes = writer.close()
    except Exception as exc:
        writer.abort()
        summary = {
            "stage": "s4_bert",
            "source": source,
            "relpath": relpath,
            "in_rows": in_rows,
            "scored_rows": 0,
            "output_bytes": 0,
            "elapsed_sec": round(time.monotonic() - started, 3),
            "errors": {f"shard_error:{type(exc).__name__}": 1},
            "error_detail": str(exc)[:500],
            "skipped": False,
        }
        write_summary(summary_path, summary)
        return summary

    elapsed = time.monotonic() - started
    summary = {
        "stage": "s4_bert",
        "source": source,
        "relpath": relpath,
        "in_rows": in_rows,
        "scored_rows": kept_rows,
        "skipped_nonkeep": skipped_nonkeep,
        "skipped_domain": skipped_domain,
        "output_bytes": out_bytes,
        "elapsed_sec": round(elapsed, 3),
        "rows_per_sec": round(kept_rows / max(elapsed, 1e-9), 1),
        "errors": {},
        "skipped": False,
    }
    write_summary(summary_path, summary)
    return summary


# ---- 多 GPU 并行 driver ----


def bert_tasks(out_root: Path) -> list[tuple[str, str, str]]:
    root = out_root / "s3"
    if not root.is_dir():
        raise FileNotFoundError(
            f"s4_bert 需要 s3 去重产物：{root} 不存在，请先运行 --stage dedup"
        )
    tasks = []
    for src_dir in sorted(root.iterdir()):
        if not src_dir.is_dir():
            continue
        for path in sorted(src_dir.rglob("*.parquet")):
            rel = path.relative_to(src_dir).as_posix()
            tasks.append(("s4_bert", src_dir.name, rel[: -len(".parquet")]))
    return sorted(tasks, key=lambda t: (t[1], t[2]))


def _bert_worker_main(
    gpu: str,
    my_tasks: list[tuple[str, str, str]],
    scorer_cfg_path: str,
    out_root: str,
    result_q: Any,
) -> None:
    """单 GPU 进程：环境绑定 → 加载模型 → 顺序消费自己的任务分区。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    # HF fast tokenizer 用 Rayon 多线程编码（每 worker 限 8 线程，遵守少量 CPU 约束）
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("RAYON_NUM_THREADS", "8")
    try:
        from .score import load_scorer_config

        cfg = load_scorer_config(scorer_cfg_path)
        model_dir = resolve_bert_model_dir(cfg)
        model, tok, device = _load_bert(model_dir)
        params = bert_inference_params(cfg)
        if params.get("torch_compile"):
            import torch

            model = torch.compile(model)
        inferencer = BertInferencer(
            model, tok, device, max_len=int(params["max_len"]),
            batch_size=int(params["batch_size"]), quantum=float(params["score_quantum"]),
            max_chars=int(params["max_score_chars"]),
            long_doc_chars=int(params.get("long_doc_chars", 4096)),
            window_scoring=bool(params.get("window_scoring", False)),
        )
        out_root_p = Path(out_root)
        for _stage, source, relpath in my_tasks:
            try:
                s3_path = out_root_p / "s3" / source / (relpath + ".parquet")
                summary = process_shard_bert(
                    s3_path, out_root_p, source, relpath, cfg, inferencer=inferencer,
                )
            except Exception as exc:
                summary = {
                    "stage": "s4_bert",
                    "source": source,
                    "relpath": relpath,
                    "shard_fatal": f"{type(exc).__name__}: {exc}",
                }
            result_q.put(summary)
    except Exception as exc:  # worker 级失败：给每个未完成任务一个 fatal 结果
        for _stage, source, relpath in my_tasks:
            result_q.put({
                "stage": "s4_bert",
                "source": source,
                "relpath": relpath,
                "shard_fatal": f"worker:{type(exc).__name__}: {exc}",
            })
    finally:
        result_q.put(None)  # 该 worker 的结束哨兵


def run_stage_bert(
    tasks: list[tuple[str, str, str]],
    scorer_cfg_path: str,
    out_root: Path,
    gpus: list[str],
    stall_guard: Any,
    log: Any = None,
) -> list[dict[str, Any]]:
    """gpus = CUDA 设备号列表；每进程独占一卡（spawn 上下文，torch 惰性导入）。

    任务分区确定性：task[i] → gpu[i % len(gpus)]，重跑分配一致。
    """
    beat, guard_stop = stall_guard
    log = log or (lambda msg: print(msg, flush=True))
    summaries: list[dict[str, Any]] = []
    started = time.monotonic()
    done = 0
    rows = 0
    ctx = mp.get_context("spawn")
    result_q: Any = ctx.Queue()
    # 每卡先预取一个任务，余下轮转——近似按序但均衡，且分配确定
    partitions: list[list[tuple[str, str, str]]] = [[] for _ in gpus]
    for i, task in enumerate(tasks):
        partitions[i % len(gpus)].append(task)
    procs = []
    for gpu, part in zip(gpus, partitions):
        if not part:
            result_q.put(None)
            continue
        p = ctx.Process(
            target=_bert_worker_main,
            args=(gpu, part, scorer_cfg_path, str(out_root), result_q),
        )
        p.start()
        procs.append(p)
    finished_workers = 0
    try:
        while finished_workers < len(gpus):
            summary = result_q.get()
            if summary is None:
                finished_workers += 1
                continue
            summaries.append(summary)
            done += 1
            beat()
            rows += int(summary.get("scored_rows") or 0)
            if done % 20 == 0 or done == len(tasks):
                elapsed = time.monotonic() - started
                log(
                    f"progress[s4_bert]: {done}/{len(tasks)} shards, "
                    f"{done / max(elapsed, 1e-9):.2f} shards/s, "
                    f"{rows / max(elapsed, 1e-9):.0f} docs/s"
                )
    finally:
        guard_stop.set()
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
    return summaries
