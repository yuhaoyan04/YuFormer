"""数据加载（宪法 4：保序消费 + 样本级断点续训）。

契约（S6 → pretrain，经 scripts/materialize_tokens.py 物化）：
  <name>.tokens.bin  uint16/uint32 LE 平铺 token 流（顺序 = mix_order 顺序表 doc 顺序，
                     每篇正文末尾有 EOD，拼接成流后由本模块切定长样本）
  <name>.meta.json   {"total_tokens", "bytes", "sha256", "edge_sha256",
                      "tokenizer_sha256", "source"}

样本 = 顺序读 seq_len+1 个 token → input=t[:-1], target=t[1:]（seq_len 个监督 token）。
游标 = 全局样本序号（跨 stage 拼接、epoch 内按 stage 顺序，跨 epoch 取模回绕），
checkpoint 存 consumed_samples，任意断点重启消费顺序与 worker 调度无关。
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .config import ConfigError


def load_data_release(path: str | Path | None, *, required: bool) -> dict | None:
    """Load a PASS data release and return the immutable release identity."""
    if not path:
        if required:
            raise ConfigError("正式训练要求 data.release_manifest")
        return None
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ConfigError(f"data release manifest 缺失: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ConfigError(f"data release manifest 无法读取: {manifest_path}: {exc}") from exc
    if manifest.get("schema_version") != 1:
        raise ConfigError(f"data release schema_version 不支持: {manifest.get('schema_version')}")
    if manifest.get("status") != "PASS" or not manifest.get("dataset_id"):
        raise ConfigError(
            f"data release 未通过门禁: status={manifest.get('status')} "
            f"failed_gates={manifest.get('failed_gates')}"
        )
    if not manifest.get("snapshot_id"):
        raise ConfigError("data release 缺少 content-addressed snapshot_id")
    return {
        "path": manifest_path.as_posix(),
        "dataset_id": str(manifest["dataset_id"]),
        "snapshot_id": str(manifest["snapshot_id"]),
        "spec_sha256": manifest.get("spec_sha256"),
        "release_name": manifest.get("release_name"),
    }


def load_data_snapshot(path: str | Path) -> dict:
    """Load a hashed candidate snapshot; gate status may still be BLOCKED."""
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise ConfigError(f"data snapshot manifest 缺失: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ConfigError(f"data snapshot manifest 无法读取: {manifest_path}: {exc}") from exc
    if manifest.get("schema_version") != 1 or not manifest.get("snapshot_id"):
        raise ConfigError("data snapshot manifest 必须包含 schema_version=1 与 snapshot_id")
    return {
        "path": manifest_path.as_posix(),
        "snapshot_id": str(manifest["snapshot_id"]),
        "release_name": manifest.get("release_name"),
        "status": manifest.get("status"),
        "artifacts": manifest.get("artifacts", []),
    }


def _edge_sha256(path: Path, edge_bytes: int = 1 << 20) -> str:
    size = path.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode("ascii"))
    with open(path, "rb") as f:
        h.update(f.read(edge_bytes))
        if size > edge_bytes:
            f.seek(max(0, size - edge_bytes))
            h.update(f.read(edge_bytes))
    return h.hexdigest()


@dataclass
class TokenStream:
    bin_path: Path
    meta_path: Path
    seq_len: int
    total_tokens: int
    samples: int
    meta: dict
    dtype: np.dtype = np.dtype("<u2")  # E-42：词表 >65536 的流用 "<u4"（meta.dtype 声明）

    @classmethod
    def open(
        cls,
        bin_path: str | Path,
        meta_path: str | Path,
        seq_len: int,
        *,
        require_eod: bool = False,
        expected_snapshot_id: str | None = None,
    ) -> "TokenStream":
        bp, mp_ = Path(bin_path), Path(meta_path)
        if not bp.is_file():
            raise ConfigError(f"token stream 缺失: {bp}")
        if not mp_.is_file():
            raise ConfigError(f"token stream meta 缺失: {mp_}")
        meta = json.loads(mp_.read_text("utf-8"))
        if bp.stat().st_size != int(meta["bytes"]):
            raise ConfigError(f"{bp} size {bp.stat().st_size} != meta.bytes {meta['bytes']}")
        if _edge_sha256(bp) != meta["edge_sha256"]:
            raise ConfigError(f"{bp} edge_sha256 与 meta 不符（数据快照被改动？）")
        total = int(meta["total_tokens"])
        dt = np.dtype(meta.get("dtype", "<u2"))  # E-42：缺省 u2 兼容既有流
        if dt.itemsize not in (2, 4):
            raise ConfigError(f"{mp_} 声明的 dtype {dt} 不受支持（仅 <u2/<u4）")
        if require_eod:
            boundary = meta.get("document_boundary")
            if not isinstance(boundary, dict) or boundary.get("policy") != "append_eod":
                raise ConfigError(f"{mp_} 未声明 append_eod 文档边界")
            if not isinstance(boundary.get("token_id"), int):
                raise ConfigError(f"{mp_} 的 EOD token_id 无效")
            if int(boundary.get("count") or -1) != int(meta.get("n_docs") or -2):
                raise ConfigError(f"{mp_} 的 EOD count 与 n_docs 不一致")
        if expected_snapshot_id is not None and meta.get("data_snapshot_id") != expected_snapshot_id:
            raise ConfigError(
                f"{mp_} 的 data_snapshot_id={meta.get('data_snapshot_id')} "
                f"与 release snapshot={expected_snapshot_id} 不一致"
            )
        return cls(bp, mp_, seq_len, total, total // (seq_len + 1), meta, dt)

    def sample(self, sample_index: int) -> np.ndarray:
        """第 sample_index 个样本（stage 内序号，调用方负责取模/边界）。"""
        span = self.seq_len + 1
        off = sample_index * span
        raw = np.memmap(self.bin_path, dtype=self.dtype, mode="r",
                        offset=off * self.dtype.itemsize, shape=(span,))
        return np.array(raw, dtype=np.int64)


def concat_streams(streams: list[TokenStream]) -> tuple[int, np.ndarray]:
    """全局样本序号 → (stage, stage 内序号) 映射表（stage 顺序拼接）。"""
    sizes = np.array([s.samples for s in streams], dtype=np.int64)
    total = int(sizes.sum())
    lut = np.zeros((total, 2), dtype=np.int64)
    start = 0
    for i, n in enumerate(sizes):
        lut[start:start + int(n), 0] = i
        lut[start:start + int(n), 1] = np.arange(int(n))
        start += int(n)
    return total, lut


def open_val_streams(
    val_dir: str | Path,
    seq_len: int,
    model_vocab: int,
    tokenizer_sha256: str | None = None,
    require_eod: bool = False,
    expected_snapshot_id: str | None = None,
) -> tuple[list[TokenStream], list[str]]:
    """打开同 tokenizer 的 ``val_*.tokens.bin``，其余流跳过并记名。

    vocab_size 只能防止 token id 越界，不能证明 token id 的语义一致。验证流必须由与
    训练流完全相同的 tokenizer.json 物化；以 SHA-256 作身份门禁。只匹配 ``val_*``
    也避免训练流与验证流共目录时把训练数据误当 held-out 数据。
    """
    streams: list[TokenStream] = []
    skipped: list[str] = []
    for f in sorted(Path(val_dir).glob("val_*.tokens.bin")):
        s = TokenStream.open(
            f,
            f.with_name(f.name.replace(".tokens.bin", ".meta.json")),
            seq_len,
            require_eod=require_eod,
            expected_snapshot_id=expected_snapshot_id,
        )
        sv = int(s.meta.get("vocab_size") or 0)
        stream_tok = s.meta.get("tokenizer_sha256")
        if sv > model_vocab or (tokenizer_sha256 is not None and stream_tok != tokenizer_sha256):
            skipped.append(f.name)
            continue
        streams.append(s)
    return streams, skipped


def distributed_sample_index(
    local_index: int,
    *,
    resume_consumed: int,
    samples_per_rank: int,
    rank: int,
    world_size: int,
) -> int:
    """把 rank 本地连续序号映射到无重叠的全局样本序号。

    每个优化步的全局样本块大小为 ``samples_per_rank * world_size``；rank ``r``
    取得该块中第 ``r * samples_per_rank`` 开始的连续子块。``resume_consumed`` 是
    checkpoint 中已提交的全局样本数，因此恢复前后的映射逐样本一致。
    """
    if local_index < 0 or resume_consumed < 0:
        raise ConfigError("样本序号与 resume_consumed 必须非负")
    if samples_per_rank < 1 or world_size < 1:
        raise ConfigError("samples_per_rank 与 world_size 必须 >= 1")
    if rank < 0 or rank >= world_size:
        raise ConfigError(f"rank={rank} 超出 [0, {world_size})")
    step, within_rank = divmod(local_index, samples_per_rank)
    return (
        resume_consumed
        + step * samples_per_rank * world_size
        + rank * samples_per_rank
        + within_rank
    )


class DataCursor:
    """游标 = consumed_samples（单调全局样本序号）；容量 = 总样本 × epochs。"""

    def __init__(self, total_samples: int, epochs: int) -> None:
        if total_samples < 1:
            raise ConfigError("token stream 样本数为 0（数据量不足一个 seq_len+1？）")
        self.total_samples = total_samples
        self.capacity = total_samples * epochs
        self.consumed = 0

    def state_dict(self) -> dict:
        return {"consumed_samples": self.consumed}

    def load_state_dict(self, state: dict) -> None:
        self.consumed = int(state["consumed_samples"])

    def require(self, n: int) -> None:
        if self.consumed + n > self.capacity:
            raise ConfigError(
                f"数据预算不足：还需 {n} 样本，剩余 {self.capacity - self.consumed} "
                f"（capacity={self.capacity}，consumed={self.consumed}）"
            )

    def advance(self, n: int) -> None:
        self.require(n)
        self.consumed += n


def _worker(in_q: mp.Queue, out_q: mp.Queue) -> None:
    cache: dict[str, np.memmap] = {}
    while True:
        item = in_q.get()
        if item is None:
            return
        bin_path, dtype_str, seq_len, local_idx, gidx = item
        mm = cache.get(bin_path)
        if mm is None:
            mm = np.memmap(bin_path, dtype=np.dtype(dtype_str), mode="r")  # E-42
            cache[bin_path] = mm
        span = seq_len + 1
        off = local_idx * span
        out_q.put((gidx, np.array(mm[off:off + span], dtype=np.int64)))


class PrefetchLoader:
    """双进程预取：worker w 服务 gidx % num_workers == w，主进程按序重组。

    next_sample() 每返回一个样本游标 +1（精确记账，中途放弃安全）；
    样本→worker 分配是序号的静态函数 → 与调度时序无关，断点续训逐位一致。
    """

    def __init__(self, streams: list[TokenStream], cursor: DataCursor, num_workers: int = 2,
                 in_flight: int = 8,
                 sample_index_fn: Callable[[int], int] | None = None) -> None:
        self.streams = streams
        self.cursor = cursor
        self.total, self.lut = concat_streams(streams)
        self.num_workers = max(1, min(num_workers, 8))
        self.window = in_flight * self.num_workers
        self._ctx = mp.get_context("fork")
        self._procs: list[mp.Process] = []
        self._in: list[mp.Queue] = []
        self._out: mp.Queue = self._ctx.Queue(maxsize=self.window)
        self._issued = 0
        self._received = 0
        self._pending: dict[int, np.ndarray] = {}
        self._base: int = 0
        self._sample_index_fn = sample_index_fn or (lambda i: i)

    def _dispatch(self, gidx: int) -> None:
        data_idx = self._sample_index_fn(gidx)
        stage, local = self.lut[data_idx % self.total]
        s = self.streams[int(stage)]
        self._in[gidx % self.num_workers].put(
            (str(s.bin_path), s.dtype.str, s.seq_len, int(local), gidx))  # E-42

    def __enter__(self) -> "PrefetchLoader":
        self._base = self.cursor.consumed
        for _ in range(self.num_workers):
            in_q = self._ctx.Queue(maxsize=self.window)
            p = self._ctx.Process(target=_worker, daemon=True, args=(in_q, self._out))
            self._in.append(in_q)
            self._procs.append(p)
            p.start()
        return self

    def __exit__(self, *exc: object) -> None:
        for q in self._in:
            try:
                q.put_nowait(None)
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    def next_sample(self) -> np.ndarray:
        start = self.cursor.consumed
        while start not in self._pending:
            while self._issued - self._received < self.window:
                self._dispatch(self._base + self._issued)
                self._issued += 1
            gidx, arr = self._out.get()
            self._received += 1
            if gidx >= start:  # 丢弃前序迭代残留（数据相同，防重复记账）
                self._pending[gidx] = arr
        arr = self._pending.pop(start)
        self.cursor.advance(1)
        return arr
