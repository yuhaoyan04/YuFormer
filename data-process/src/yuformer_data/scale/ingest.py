from __future__ import annotations

import gzip
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from .schema import CANONICAL_SCHEMA, META_MAX_CHARS, make_doc_id

WRITE_BATCH_ROWS = 2048
# 单行 MB 级的大行分片（books/code）只按行数 flush 会攒出数 GB 缓冲——
# 叠加 160 worker 有 OOM 风险；按缓冲字符数设硬顶（H1 修复）
FLUSH_MAX_CHARS = 256_000_000


def shard_output_paths(out_root: Path, stage: str, source: str, relpath: str) -> tuple[Path, Path]:
    base = out_root / f"{stage}" / source / relpath
    return base.with_name(base.name + ".parquet"), base.with_name(base.name + ".summary.json")


def is_done(summary_path: Path, output_path: Path) -> bool:
    return summary_path.is_file() and output_path.is_file()


def _open_text(path: Path) -> io.TextIOBase:
    name = path.name.lower()
    if name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    if name.endswith(".zst"):
        import zstandard

        raw = path.open("rb")
        reader = zstandard.ZstdDecompressor().stream_reader(raw)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def iter_raw_rows(path: Path, profile: dict[str, Any]) -> Iterator[tuple[int, dict[str, Any] | None]]:
    if path.suffix.lower() == ".parquet":
        pf = pq.ParquetFile(path)
        idx = 0
        for batch in pf.iter_batches(batch_size=WRITE_BATCH_ROWS):
            for row in batch.to_pylist():
                yield idx, row
                idx += 1
        return
    if path.suffix.lower() == ".txt":
        with path.open("rt", encoding="utf-8", errors="replace") as fh:
            yield 0, {"text": fh.read()}
        return
    idx = 0
    with _open_text(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                yield idx, None
                idx += 1
                continue
            if isinstance(obj, dict):
                yield idx, obj
                idx += 1
            else:
                yield idx, None
                idx += 1


def _first_str(row: dict[str, Any], keys: list[str]) -> str | None:
    # 只接受非空字符串；空串/None/其他类型一律回退到下一个候选字段（H2 修复）
    for key in keys:
        val = row.get(key)
        if isinstance(val, str) and val:
            return val
    return None


def _first_float(row: dict[str, Any], keys: list[str]) -> float | None:
    for key in keys:
        val = row.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
        if isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                continue
    return None


def _lang_from_row(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        val = row.get(key)
        if isinstance(val, str) and val:
            return val
        if isinstance(val, dict) and val:
            return max(val, key=lambda k: val[k])
    return ""


def _qa_text(row: dict[str, Any], profile: dict[str, Any]) -> str | None:
    import ast

    q = row.get(profile.get("q_field", "question"))
    if not isinstance(q, str) or not q:
        return None
    raw_answers = row.get(profile.get("a_field", "answers"))
    answers: list[dict[str, Any]] = []
    if isinstance(raw_answers, list):
        answers = [a for a in raw_answers if isinstance(a, dict)]
    elif isinstance(raw_answers, str) and raw_answers:
        try:
            parsed = ast.literal_eval(raw_answers)
            if isinstance(parsed, list):
                answers = [a for a in parsed if isinstance(a, dict)]
        except (ValueError, SyntaxError):
            pass
    best = ""
    if answers:
        selected = [a for a in answers if a.get("selected") is True]
        pool = selected or answers
        best = str(max(pool, key=lambda a: float(a.get("pm_score") or 0.0)).get("text") or "")
    parts = [f"Q: {q.strip()}"]
    if best:
        parts.append(f"A: {best.strip()}")
    return "\n".join(parts)


def map_row(
    row: dict[str, Any],
    profile: dict[str, Any],
    source: str,
    relpath: str,
    row_idx: int,
) -> dict[str, Any] | None:
    if profile.get("mode") == "qa":
        text = _qa_text(row, profile)
        used_fields = {profile.get("q_field", "question"), profile.get("a_field", "answers")}
    else:
        text_keys = list(profile["text_fields"]) or ["text"]
        text = _first_str(row, text_keys)
        used_fields = set(text_keys)
    if text is None:
        return None
    score = _first_float(row, list(profile["score_fields"]))
    lang = _lang_from_row(row, list(profile["lang_fields"])) or profile.get("lang_default", "")
    meta_obj = {k: v for k, v in row.items() if k not in used_fields}
    meta = json.dumps(meta_obj, ensure_ascii=False, sort_keys=True, default=str)
    meta = meta[:META_MAX_CHARS]
    return {
        "doc_id": make_doc_id(source, relpath, row_idx),
        "text": text,
        "domain": profile["domain"],
        "source": source,
        "lang": lang,
        "upstream_score": score,
        "meta": meta,
    }


class BatchWriter:
    def add(self, row: dict[str, Any]) -> None:
        for name, buf in zip(self._names, self._bufs):
            buf.append(row.get(name))
        self.rows += 1
        self._buf_chars += sum(len(v) for v in row.values() if isinstance(v, str))
        if self.rows % WRITE_BATCH_ROWS == 0 or self._buf_chars >= FLUSH_MAX_CHARS:
            self.flush()

    def extend(self, cols: dict[str, list[Any]]) -> None:
        n = len(cols[self._names[0]])
        for name, buf in zip(self._names, self._bufs):
            buf.extend(cols[name])
            if n and isinstance(cols[name][0], str):
                self._buf_chars += sum(len(s) for s in cols[name])
        self.rows += n
        if self.rows % WRITE_BATCH_ROWS < n or self._buf_chars >= FLUSH_MAX_CHARS:
            self.flush()

    def flush(self) -> None:
        if not self._bufs[0]:
            return
        batch = pa.record_batch([pa.array(buf) for buf in self._bufs], schema=self.schema)
        self._writer.write_batch(batch)
        self._bufs = [[] for _ in self._names]
        self._buf_chars = 0

    def __init__(self, path: Path, schema: pa.Schema):
        self.path = path
        self.schema = schema
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
        self._writer = pq.ParquetWriter(self._tmp, schema, compression="zstd")
        self._names = schema.names
        self._bufs: list[list[Any]] = [[] for _ in self._names]
        self._buf_chars = 0
        self.rows = 0

    def close(self) -> int:
        self.flush()
        self._writer.close()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self._tmp, self.path)
        return self.path.stat().st_size

    def abort(self) -> None:
        try:
            self._writer.close()
        except Exception:
            pass
        self._tmp.unlink(missing_ok=True)


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def safe_iter_batches(pf: pq.ParquetFile, **kwargs: Any) -> Iterator[pa.RecordBatch]:
    """0-row-group parquet（上游空分片的合法产物）在 pyarrow≥17 下 iter_batches 抛
    OSError "file only has 0 row groups"（E-37：2026-10-08 pyarrow 12→18 升级暴露，
    旧版静默空迭代）。统一按"空文件=零批次"处理，下游各 stage 免重复判空。"""
    if pf.metadata.num_rows == 0:
        return
    yield from pf.iter_batches(**kwargs)


def process_shard_s0(src_path: Path, out_root: Path, source: str, relpath: str, profile: dict[str, Any]) -> dict[str, Any]:
    out_path, summary_path = shard_output_paths(out_root, "s0", source, relpath)
    if is_done(summary_path, out_path):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["skipped"] = True
        return summary

    started = time.monotonic()
    in_rows = out_rows = error_rows = 0
    writer = BatchWriter(out_path, CANONICAL_SCHEMA)
    try:
        for row_idx, raw in iter_raw_rows(src_path, profile):
            in_rows += 1
            if raw is None:
                error_rows += 1
                continue
            try:
                rec = map_row(raw, profile, source, relpath, row_idx)
            except Exception:
                error_rows += 1
                continue
            if rec is None:
                error_rows += 1
                continue
            writer.add(rec)
            out_rows += 1
        out_bytes = writer.close()
    except Exception as exc:
        writer.abort()
        out_bytes = 0
        error_rows += 1
        elapsed = time.monotonic() - started
        summary = {
            "stage": "s0",
            "source": source,
            "relpath": relpath,
            "input_bytes": src_path.stat().st_size,
            "output_bytes": out_bytes,
            "in_rows": in_rows,
            "out_rows": out_rows,
            "error_rows": error_rows,
            "errors": {f"shard_error:{type(exc).__name__}": 1},
            "elapsed_sec": round(elapsed, 3),
            "skipped": False,
        }
        write_summary(summary_path, summary)
        return summary

    elapsed = time.monotonic() - started
    input_bytes = src_path.stat().st_size
    summary = {
        "stage": "s0",
        "source": source,
        "relpath": relpath,
        "input_bytes": input_bytes,
        "output_bytes": out_bytes,
        "in_rows": in_rows,
        "out_rows": out_rows,
        "error_rows": error_rows,
        "errors": {},
        "elapsed_sec": round(elapsed, 3),
        "mb_per_sec": round(input_bytes / max(elapsed, 1e-9) / 1e6, 2),
        "skipped": False,
    }
    write_summary(summary_path, summary)
    return summary
