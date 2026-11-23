from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .ingest import BatchWriter, safe_iter_batches, write_summary
from .schema import S1_SCHEMA
from .scorer_train import normalize_scorer_text

DEFAULT_SCORER_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "scorer.yaml"

SCORED_SCHEMA: pa.Schema = (
    S1_SCHEMA.append(pa.field("self_score", pa.float32(), nullable=False))
    .append(pa.field("composite", pa.float32(), nullable=False))
    .append(pa.field("upstream_norm", pa.float32(), nullable=True))
    .append(pa.field("fasttext_score", pa.float32(), nullable=False))
    .append(pa.field("bert_score", pa.float32(), nullable=True))
    .append(pa.field("composite_v2", pa.float32(), nullable=False))
    .append(pa.field("rule_score", pa.float32(), nullable=False))
    .append(pa.field("rule_composite", pa.float32(), nullable=False))
    .append(pa.field("hybrid_composite", pa.float32(), nullable=False))
    .append(pa.field("llm_distill", pa.float32(), nullable=True))
    .append(pa.field("hybrid_v2", pa.float32(), nullable=False))
)
S4_SCHEMA: pa.Schema = SCORED_SCHEMA.append(
    pa.field("domain_quantile", pa.float32(), nullable=False)
)
PAIRS_SCHEMA: pa.Schema = pa.schema(
    [
            pa.field("domain", pa.string(), nullable=False),
        pa.field("composite", pa.float32(), nullable=False),
        pa.field("rule_composite", pa.float32(), nullable=False),
        pa.field("hybrid_composite", pa.float32(), nullable=False),
        pa.field("hybrid_v2", pa.float32(), nullable=False),
    ]
)

HIST_BIN_COUNT = 20
CORR_SAMPLE_MAX = 2000
PREDICT_BATCH = 1000
log = logging.getLogger("yuformer.scale.score")


def load_scorer_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_SCORER_CONFIG
    if not p.is_file():
        raise FileNotFoundError(f"scorer config not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict) or not cfg.get("composite"):
        raise ValueError(f"scorer config invalid: {p}")
    # Asset paths belong to the data-process module layout, never to the
    # caller's cwd. This keeps ``python -m ...`` identical from repo root and
    # from data-process/ and prevents silent goldrule/fasttext fallback.
    for key in ("model", "model_fallback_bin", "bert_model", "goldrule_model"):
        raw = cfg.get(key)
        if raw:
            asset = Path(str(raw))
            if not asset.is_absolute():
                asset = (p.parent.parent / asset).resolve()
            cfg[key] = str(asset)
    cfg["_config_path"] = str(p)
    return cfg


def resolve_model_path(cfg: dict[str, Any]) -> Path:
    primary = Path(str(cfg.get("model", "")))
    if primary.is_file():
        return primary
    fallback = Path(str(cfg.get("model_fallback_bin", "")))
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        f"scorer model not found: {primary} (nor fallback {fallback}); "
        "train it with yuformer_data.scale.scorer_train"
    )


def normalize_upstream(source: str, score: float | None, cfg: dict[str, Any]) -> float | None:
    if score is None:
        return None
    table = cfg.get("upstream_normalize") or {}
    lo, hi = table.get(source, [0.0, 1.0])
    if hi <= lo:
        return None
    return float(min(1.0, max(0.0, (float(score) - lo) / (hi - lo))))


def composite_vector(
    sources: list[str],
    upstream_scores: list[float | None],
    self_scores: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray:
    w_u = float(cfg["composite"]["upstream_weight"])
    w_s = float(cfg["composite"]["self_weight"])
    n = len(sources)
    table = cfg.get("upstream_normalize") or {}
    lo = np.zeros(n, dtype=np.float64)
    hi = np.ones(n, dtype=np.float64)
    for src in set(sources):
        bounds = table.get(src)
        if bounds:
            mask = np.array([s == src for s in sources])
            lo[mask] = float(bounds[0])
            hi[mask] = float(bounds[1])
    up = np.array([np.nan if s is None else float(s) for s in upstream_scores], dtype=np.float64)
    valid = ~np.isnan(up) & (hi > lo)
    nu = np.clip((up - lo) / np.maximum(hi - lo, 1e-12), 0.0, 1.0)
    comp = np.where(valid, w_u * nu + w_s * self_scores.astype(np.float64), self_scores.astype(np.float64))
    return np.clip(comp, 0.0, 1.0)


def _upstream_norm_array(
    sources: list[str], upstream_scores: list[float | None], cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized upstream scores and validity; invalid rows stay NaN."""
    n = len(sources)
    table = cfg.get("upstream_normalize") or {}
    lo = np.zeros(n, dtype=np.float64)
    hi = np.ones(n, dtype=np.float64)
    for src in set(sources):
        bounds = table.get(src)
        if bounds:
            mask = np.array([s == src for s in sources])
            lo[mask] = float(bounds[0])
            hi[mask] = float(bounds[1])
    up = np.array([np.nan if s is None else float(s) for s in upstream_scores], dtype=np.float64)
    valid = ~np.isnan(up) & (hi > lo)
    nu = np.where(
        valid,
        np.clip((up - lo) / np.maximum(hi - lo, 1e-12), 0.0, 1.0),
        np.nan,
    )
    return nu, valid


def composite_v2_weights(cfg: dict[str, Any]) -> tuple[float, float, float]:
    """三轨权重 (w_u, w_f, w_b)；配置缺 composite_v2 时回退 legacy 双轨权重（w_b=0）。"""
    c2 = cfg.get("composite_v2") or {}
    if c2:
        return (
            float(c2.get("upstream_weight", 0.25)),
            float(c2.get("fasttext_weight", 0.35)),
            float(c2.get("bert_weight", 0.40)),
        )
    leg = cfg.get("composite") or {}
    return float(leg.get("upstream_weight", 0.5)), float(leg.get("self_weight", 0.5)), 0.0


def load_goldrule_model(cfg: dict[str, Any]) -> Any | None:
    """goldrule_v2（TF-IDF+LR 蒸馏模型，2026-09-28）。缺失返回 None（轨自动降级）。"""
    import joblib

    path = Path(str(cfg.get("goldrule_model", "")))
    if not path.is_file():
        return None
    runtime_path = path.with_name(path.stem + ".runtime.json")
    if not runtime_path.is_file():
        raise RuntimeError(f"goldrule runtime contract missing: {runtime_path}")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    import sklearn

    actual_versions = {
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
    }
    mismatches = {
        package: {"required": required, "actual": actual_versions.get(package)}
        for package, required in (runtime.get("required_packages") or {}).items()
        if actual_versions.get(package) != required
    }
    if mismatches:
        raise RuntimeError(
            "goldrule runtime contract mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    # E-34：工件在 numpy 2.x 环境 pickle（引用 numpy._core.*），numpy 1.x 环境
    # 直接 unpickle 报 ModuleNotFoundError: numpy._core（2026-10-08 S4 重跑时
    # 11 个 bert 重算分片因此 shard_fatal）。兼容别名 numpy._core→numpy.core
    # （只读加载，TF-IDF/LR 不触 _core 新增 API；已实测加载+predict 正常）。
    if not np.lib.NumpyVersion(np.__version__).major >= 2:
        # Import sklearn's compiled extensions before installing the pickle-only
        # alias.  Otherwise extensions built against NumPy's public C API can
        # resolve the synthetic ``numpy._core`` module and fail import_array.
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
        from sklearn.linear_model import LogisticRegression  # noqa: F401
        import numpy.core.multiarray  # noqa: F401

        sys.modules.setdefault("numpy._core", np.core)
        sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
    return joblib.load(path)


def goldrule_scores_batch(bundle: Any, texts: list[str], max_chars: int = 512) -> np.ndarray:
    """goldrule_v2 批量打分。trunc 512 = 生产吞吐权衡点（385 docs/s/核，
    全池 ~21h @128w）；3000 字符版 60 docs/s 不可用（见 finalize 脚本头注）。"""
    truncated = [(t or "")[:max_chars] for t in texts]
    X = bundle["vectorizer"].transform(truncated)
    return bundle["classifier"].predict_proba(X)[:, 1].astype(np.float32)


def hybrid_v2_vector(
    llm_distill: np.ndarray,
    upstream_norm: np.ndarray,
    rule: np.ndarray,
    cfg: dict[str, Any],
) -> np.ndarray:
    """生产选取轨（2026-09-28）：0.5·llm_distill + 0.3·upstream + 0.2·rule，
    缺失轨（llm_distill 整轨缺失 / upstream 行缺失）按权重归一。三路独立信号集成：
    learned 内容级质量（修复文体偏置）+ 上游分（编辑信号）+ 规则结构分。"""
    hcfg = cfg.get("hybrid_v2") or {}
    w_llm = float(hcfg.get("llm_distill_weight", 0.5))
    w_up = float(hcfg.get("upstream_weight", 0.3))
    w_rule = float(hcfg.get("rule_weight", 0.2))
    up = upstream_norm.astype(np.float64)
    if llm_distill is None:
        w_llm = 0.0
        llm = np.zeros(len(up))
    else:
        llm = llm_distill.astype(np.float64)
    up_ok = ~np.isnan(up)
    w_sum = w_llm + w_rule + w_up
    total = (
        w_llm * llm
        + w_rule * rule.astype(np.float64)
        + np.where(up_ok, w_up, 0.0) * np.nan_to_num(up)
    )
    denom = np.where(up_ok, w_sum, w_sum - w_up)
    return np.clip(total / np.maximum(denom, 1e-9), 0.0, 1.0)


def rule_composite_vector(upstream_norm: np.ndarray, rule: np.ndarray) -> np.ndarray:
    """规则兜底复合分（2026-09-27 用户裁定）：0.6·upstream_norm + 0.4·rule_score，
    按行归一（upstream 缺失行退化纯 rule_score）。免训练、全确定性——learned 轨
    CV 不过门时的生产选取路径；goldrule 研究子集的对照基线。"""
    up = np.where(np.isnan(upstream_norm.astype(np.float64)), np.nan, upstream_norm.astype(np.float64))
    w_up = np.where(np.isnan(up), 0.0, 0.6)
    w_rule = 1.0 - w_up
    total = np.where(np.isnan(up), rule.astype(np.float64), w_up * np.nan_to_num(up) + w_rule * rule.astype(np.float64))
    return np.clip(total, 0.0, 1.0)


def hybrid_composite_vector(
    langs: list[str], rule_comp: np.ndarray, fasttext: np.ndarray
) -> np.ndarray:
    """语言感知混合轨（2026-09-27 样本抽检实证）：fasttext 判杀英文语义垃圾
    （SEO/spinner，rule 对其失效——形式特征全健康）但对中文 96% 近零（域内排序
    退化）；rule 中文健康。互补设计：zh 行 = rule_composite 原样；非 zh 行 =
    0.6·rule_composite + 0.4·fasttext（垃圾文 fasttext→0 拉低总分，好文双高）。"""
    zh_mask = np.fromiter((str(l).startswith("zh") for l in langs), dtype=bool, count=len(langs))
    en_side = 0.6 * rule_comp + 0.4 * np.nan_to_num(fasttext.astype(np.float64))
    return np.where(zh_mask, rule_comp, en_side)


def composite_v2_vector(
    sources: list[str],
    upstream_scores: list[float | None],
    fasttext_scores: np.ndarray,
    bert_scores: np.ndarray | None,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """三轨融合：composite_v2 = Σ w_i·track_i / Σ w_i(该行可用轨)。

    上游分缺失 → 该行 w_u 置 0（权重在 fasttext/bert 间按原比例重归一）；
    bert 分缺失（None 或 NaN）→ 该行 w_b 置 0；全 bert 缺失时整体退化为双轨
    composite（与 legacy composite 数值一致）。
    返回 (composite_v2, upstream_norm, bert_used_mask)。
    """
    w_u, w_f, w_b = composite_v2_weights(cfg)
    n = len(sources)
    nu, up_valid = _upstream_norm_array(sources, upstream_scores, cfg)
    ft = fasttext_scores.astype(np.float64)
    if bert_scores is None:
        bert = np.zeros(n, dtype=np.float64)
        bert_valid = np.zeros(n, dtype=bool)
    else:
        bert = np.asarray(bert_scores, dtype=np.float64)
        bert_valid = ~np.isnan(bert)
        bert = np.where(bert_valid, bert, 0.0)  # 0*NaN=NaN 防护
    w_u_row = np.where(up_valid, w_u, 0.0)
    w_b_row = np.where(bert_valid, w_b, 0.0)
    w_f_row = np.full(n, w_f, dtype=np.float64)
    wsum = w_u_row + w_f_row + w_b_row
    comp = (
        w_u_row * np.nan_to_num(nu)
        + w_f_row * ft
        + w_b_row * bert
    ) / np.maximum(wsum, 1e-12)
    return np.clip(comp, 0.0, 1.0), nu, bert_valid


def load_bert_score_map(out_root: Path, source: str, relpath: str) -> dict[str, float] | None:
    """读 s4_bert/<source>/<relpath>.parquet 的 (doc_id → bert_score)；无产物返回 None。"""
    from .score_bert import bert_output_paths

    out_path, _summary = bert_output_paths(out_root, source, relpath)
    if not out_path.is_file():
        return None
    pf = pq.ParquetFile(out_path)
    ids = []
    scores = []
    for batch in safe_iter_batches(pf, batch_size=8192, columns=["doc_id", "bert_score"]):
        d = batch.to_pydict()
        ids.extend(d["doc_id"])
        scores.extend(d["bert_score"])
    return dict(zip(ids, scores))


def load_quality_model(cfg: dict[str, Any]) -> Any:
    import fasttext

    return fasttext.load_model(str(resolve_model_path(cfg)))


def predict_self_scores(
    model: Any,
    texts: list[str],
    max_chars: int,
    long_doc_chars: int = 4096,
    window_scoring: bool = False,
) -> np.ndarray:
    """fasttext 质量分。window_scoring=True 时长文档（>long_doc_chars）走
    head/mid/tail 三窗各 max_chars 字符、均分（与 BERT 轨同策略，2026-09-20）。"""
    from .scorer_train import predict_hq_proba, scorer_windows

    if not window_scoring:
        norm = [normalize_scorer_text(t, max_chars) for t in texts]
        return predict_hq_proba(model, norm, batch_size=PREDICT_BATCH).astype(np.float32)
    flat: list[str] = []
    owners: list[int] = []
    for i, t in enumerate(texts):
        for w in scorer_windows(t, max_chars, long_doc_chars):
            flat.append(w)
            owners.append(i)
    p = predict_hq_proba(model, flat, batch_size=PREDICT_BATCH).astype(np.float64)
    out = np.zeros(len(texts), dtype=np.float64)
    cnt = np.zeros(len(texts), dtype=np.float64)
    np.add.at(out, owners, p)
    np.add.at(cnt, owners, 1.0)
    return (out / np.maximum(cnt, 1.0)).astype(np.float32)


def scored_output_paths(out_root: Path, source: str, relpath: str) -> tuple[Path, Path, Path]:
    base = out_root / "s4_scored" / source / relpath
    pairs_base = out_root / "s4_quantiles" / "pairs" / source / relpath
    return (
        base.with_name(base.name + ".parquet"),
        pairs_base.with_name(pairs_base.name + ".parquet"),
        base.with_name(base.name + ".summary.json"),
    )


def s4_output_paths(out_root: Path, source: str, relpath: str) -> tuple[Path, Path]:
    base = out_root / "s4" / source / relpath
    return base.with_name(base.name + ".parquet"), base.with_name(base.name + ".summary.json")


def _hist_counts(values: np.ndarray, bins: int = HIST_BIN_COUNT) -> list[int]:
    if len(values) == 0:
        return [0] * bins
    idx = np.clip((values.astype(np.float64) * bins).astype(int), 0, bins - 1)
    return np.bincount(idx, minlength=bins).tolist()


def _corr_sample_pairs(
    upstream: list[float | None], self_scores: list[float], max_keep: int = CORR_SAMPLE_MAX
) -> list[list[float]]:
    pairs = [(u, s) for u, s in zip(upstream, self_scores) if u is not None]
    if len(pairs) > max_keep:
        stride = len(pairs) / max_keep
        pairs = [pairs[int(i * stride)] for i in range(max_keep)]
    return [[float(u), float(s)] for u, s in pairs]


def _model_fingerprint(cfg: dict[str, Any]) -> str | None:
    """scorer 模型组合指纹（fasttext md5 + goldrule joblib md5）。
    E-24 家族修复：旧产物 skip 必须校验模型指纹，否则任一模型重训后静默沿用旧分数。"""
    try:
        parts = [hashlib.md5(resolve_model_path(cfg).read_bytes()).hexdigest()]
    except FileNotFoundError:
        parts = ["none"]
    gr_path = Path(str(cfg.get("goldrule_model", "")))
    parts.append(hashlib.md5(gr_path.read_bytes()).hexdigest() if gr_path.is_file() else "none")
    return "|".join(parts)


def _file_identity(path: Path) -> list[int]:
    if not path.is_file():
        return [-1, -1]
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def _sha256_optional(path: Path) -> str:
    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection_semantic_sha256(cfg: dict[str, Any], score_field: str) -> str:
    """Identity only the inputs that can change the production selection field."""
    if score_field != "hybrid_v2":
        raise ValueError(f"selection contract unsupported for {score_field}")
    goldrule = Path(str(cfg.get("goldrule_model") or ""))
    cache_key = json.dumps(
        {
            "score_field": score_field,
            "hybrid_v2": cfg.get("hybrid_v2") or {},
            "upstream_normalize": cfg.get("upstream_normalize") or {},
            "goldrule_path": str(goldrule),
            "goldrule_identity": _file_identity(goldrule),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    cache = cfg.setdefault("_selection_semantic_cache", {})
    if cache.get("key") == cache_key:
        return str(cache["sha256"])
    payload = {
        "score_field": score_field,
        "formula": "hybrid_v2_v2_missing_upstream_renormalized",
        "hybrid_v2": cfg.get("hybrid_v2") or {},
        "upstream_normalize": cfg.get("upstream_normalize") or {},
        "rule_score": {"implementation": "rule_scores_batch_v1", "max_chars": 2048},
        "goldrule_sha256": _sha256_optional(goldrule),
        "goldrule_max_chars": 512,
    }
    result = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    cache.update({"key": cache_key, "sha256": result})
    return result


def selection_contract_base(
    s1_path: Path,
    scored_path: Path,
    pairs_path: Path,
    cfg: dict[str, Any],
    score_field: str = "hybrid_v2",
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "score_field": score_field,
        "s1_fingerprint": _file_identity(s1_path),
        "scored_fingerprint": _file_identity(scored_path),
        "pairs_fingerprint": _file_identity(pairs_path),
        "semantic_sha256": selection_semantic_sha256(cfg, score_field),
    }


def effective_hybrid_v2(
    stored: np.ndarray,
    upstream_scores: list[float | None] | np.ndarray,
    cfg: dict[str, Any],
    representation: str,
) -> np.ndarray:
    """Return current hybrid_v2 semantics from a declared stored representation.

    Production shards written before the v2 missing-upstream fix used zero as a
    valid upstream value.  For those rows the old score is the unnormalized
    numerator divided by the full weight sum, so it can be repaired exactly
    without rerunning the goldrule model.
    """
    values = np.asarray(stored, dtype=np.float64)
    if representation == "current_v2":
        return values.astype(np.float32)
    if representation != "legacy_zero_upstream_v1":
        raise ValueError(f"unsupported hybrid_v2 representation: {representation}")
    hcfg = cfg.get("hybrid_v2") or {}
    full = sum(
        float(hcfg.get(key, default))
        for key, default in (
            ("llm_distill_weight", 0.5),
            ("upstream_weight", 0.3),
            ("rule_weight", 0.2),
        )
    )
    retained = full - float(hcfg.get("upstream_weight", 0.3))
    if full <= 0.0 or retained <= 0.0:
        raise ValueError("hybrid_v2 weights cannot repair missing upstream scores")
    # float64 conversion maps Arrow/Python None to NaN in native NumPy code;
    # avoid a Python predicate over billions of production rows.
    upstream = np.asarray(upstream_scores, dtype=np.float64)
    missing = np.isnan(upstream)
    repaired = values.copy()
    repaired[missing] *= full / retained
    return np.clip(repaired, 0.0, 1.0).astype(np.float32)


def validate_selection_contract(
    s1_path: Path,
    scored_path: Path,
    pairs_path: Path,
    summary: dict[str, Any],
    cfg: dict[str, Any],
    audit_sha256: str,
    score_field: str = "hybrid_v2",
) -> str | None:
    contract = (summary.get("selection_contracts") or {}).get(score_field) or {}
    expected = selection_contract_base(s1_path, scored_path, pairs_path, cfg, score_field)
    if any(contract.get(k) != v for k, v in expected.items()):
        return "selection artifact identity or semantic config is stale"
    provenance = contract.get("provenance")
    if provenance == "computed_current":
        if contract.get("representation") != "current_v2":
            return "computed selection representation is stale"
        if summary.get("processing_contract") != _score_processing_contract(
            s1_path,
            scored_path.parents[2],
            str(summary.get("source") or ""),
            str(summary.get("relpath") or ""),
            cfg,
            _model_fingerprint(cfg),
        ):
            return "computed selection parent processing contract is stale"
    elif provenance == "legacy_exact_audit":
        if contract.get("representation") != "legacy_zero_upstream_v1":
            return "legacy selection representation is stale"
        if not audit_sha256 or contract.get("audit_sha256") != audit_sha256:
            return "legacy selection audit is missing or stale"
    else:
        return "selection provenance is missing or unsupported"
    return None


def certify_legacy_selection_shard(
    s1_path: Path,
    out_root: Path,
    source: str,
    relpath: str,
    cfg: dict[str, Any],
    audit_sha256: str,
    score_field: str = "hybrid_v2",
) -> dict[str, Any]:
    """Structurally bind an old score shard after dataset-level exact recomputation."""
    scored_path, pairs_path, summary_path = scored_output_paths(
        out_root, source, relpath
    )
    if not all(p.is_file() for p in (s1_path, scored_path, pairs_path, summary_path)):
        raise FileNotFoundError(f"S4 selection shard incomplete: {source}/{relpath}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("errors") or summary.get("shard_fatal"):
        raise ValueError(f"S4 score summary contains errors: {source}/{relpath}")
    if not audit_sha256:
        raise ValueError("legacy S4 certification requires exact audit SHA")
    if validate_selection_contract(
        s1_path,
        scored_path,
        pairs_path,
        summary,
        cfg,
        audit_sha256,
        score_field,
    ) is None:
        return summary
    rows = int(pq.ParquetFile(s1_path).metadata.num_rows)
    scored_pf = pq.ParquetFile(scored_path)
    pairs_pf = pq.ParquetFile(pairs_path)
    if int(scored_pf.metadata.num_rows) != rows or int(pairs_pf.metadata.num_rows) != rows:
        raise ValueError(f"S4/S1 row closure failed: {source}/{relpath}")
    if int(summary.get("in_rows") or 0) != rows:
        raise ValueError(f"S4 summary row closure failed: {source}/{relpath}")
    if score_field not in scored_pf.schema_arrow.names or score_field not in pairs_pf.schema_arrow.names:
        raise ValueError(f"{score_field} missing from S4 artifacts: {source}/{relpath}")
    llm_idx = scored_pf.schema_arrow.get_field_index("llm_distill")
    if llm_idx < 0:
        raise ValueError(f"llm_distill missing from legacy S4 artifact: {source}/{relpath}")
    llm_nulls = 0
    for row_group in range(scored_pf.metadata.num_row_groups):
        stats = scored_pf.metadata.row_group(row_group).column(llm_idx).statistics
        if stats is None or stats.null_count is None:
            raise ValueError(
                f"llm_distill null statistics unavailable: {source}/{relpath}"
            )
        llm_nulls += int(stats.null_count)
    if llm_nulls:
        raise ValueError(
            f"legacy transform requires llm_distill on every row; found {llm_nulls}: "
            f"{source}/{relpath}"
        )
    if s1_path.stat().st_mtime_ns > min(
        scored_path.stat().st_mtime_ns, pairs_path.stat().st_mtime_ns
    ):
        raise ValueError(f"S1 is newer than legacy S4 artifacts: {source}/{relpath}")
    contract = selection_contract_base(
        s1_path, scored_path, pairs_path, cfg, score_field
    )
    contract.update(
        {
            "provenance": "legacy_exact_audit",
            "representation": "legacy_zero_upstream_v1",
            "audit_sha256": audit_sha256,
        }
    )
    summary.setdefault("selection_contracts", {})[score_field] = contract
    summary["output_bytes"] = scored_path.stat().st_size
    summary["pairs_bytes"] = pairs_path.stat().st_size
    write_summary(summary_path, summary)
    return summary


def _config_fingerprint(cfg: dict[str, Any]) -> str:
    """Hash every semantic scorer option while ignoring loader-only metadata."""
    payload = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _score_processing_contract(
    s1_path: Path,
    out_root: Path,
    source: str,
    relpath: str,
    cfg: dict[str, Any],
    model_fp: str | None,
) -> dict[str, Any]:
    bert_path = out_root / "s4_bert" / source / (relpath + ".parquet")
    return {
        "schema_version": 2,
        "s1_fingerprint": _file_identity(s1_path),
        "bert_fingerprint": _file_identity(bert_path),
        "scorer_config_sha256": _config_fingerprint(cfg),
        "model_fingerprint": model_fp,
    }


def process_shard_score(
    s1_path: Path,
    out_root: Path,
    source: str,
    relpath: str,
    cfg: dict[str, Any],
    model: Any | None = None,
) -> dict[str, Any]:
    out_path, pairs_path, summary_path = scored_output_paths(out_root, source, relpath)
    model_fp = _model_fingerprint(cfg)
    processing_contract = _score_processing_contract(
        s1_path, out_root, source, relpath, cfg, model_fp
    )
    if summary_path.is_file() and out_path.is_file() and pairs_path.is_file():
        summary = json.loads(summary_path.read_text("utf-8"))
        if summary.get("processing_contract") == processing_contract and not summary.get("errors"):
            summary["skipped"] = True
            return summary

    started = time.monotonic()
    in_rows = 0
    if model is None:
        model = load_quality_model(cfg)
    goldrule_bundle = load_goldrule_model(cfg)
    # fasttext 轨开关（2026-09-28）：来源代理标签已判死刑（E-25），生产路径 =
    # hybrid_v2（llm_distill+upstream+rule）。关闭 fasttext 省三窗推理（吞吐
    # 112→~200 rows/s/worker）；列保留 schema（填 0）供历史对照。
    # E-36 收官：真值解析——yaml 1.1 会把 off 解析为布尔 False，字符串比较永不命中
    # （该轨因此白跑了 10 天）；现统一按"显式 false/off/no/0 即关"判定。
    ft_track = cfg.get("fasttext_track", True)
    skip_fasttext = ft_track is False or (
        isinstance(ft_track, str) and ft_track.strip().lower() in ("off", "false", "no", "0")
    )
    max_chars = int(cfg.get("inference", {}).get("max_score_chars", 20000))

    writer = BatchWriter(out_path, SCORED_SCHEMA)
    pairs_writer = BatchWriter(pairs_path, PAIRS_SCHEMA)
    per_domain_hist: dict[str, dict[str, list[int]]] = {}
    corr_pairs: list[list[float]] = []
    s1_names = S1_SCHEMA.names
    bert_map = load_bert_score_map(out_root, source, relpath)
    bert_joined = bert_map is not None
    try:
        pf = pq.ParquetFile(s1_path)
        for batch in safe_iter_batches(pf, batch_size=4096):
            d = batch.to_pydict()
            n = len(d["text"])
            if n == 0:
                continue
            texts = d["text"]
            inf_cfg = cfg.get("inference") or {}
            window_scoring = bool(cfg.get("window_scoring", inf_cfg.get("window_scoring", False)))
            long_doc_chars = int(cfg.get("long_doc_chars", 4096))
            if skip_fasttext:
                self_scores = np.zeros(n, dtype=np.float64)
            else:
                self_scores = predict_self_scores(
                    model, texts, max_chars, long_doc_chars, window_scoring
                )
            sources = d["source"]
            upstream = d["upstream_score"]
            composite = composite_vector(sources, upstream, self_scores, cfg)
            comp_f32 = composite.astype(np.float32)
            if bert_map is not None:
                bert = np.array(
                    [bert_map.get(doc_id, np.nan) for doc_id in d["doc_id"]], dtype=np.float64
                )
            else:
                bert = None
            comp_v2, up_norm, bert_used = composite_v2_vector(sources, upstream, self_scores, bert, cfg)
            comp_v2_f32 = comp_v2.astype(np.float32)
            from .rule_score import rule_scores_batch

            # rule 截 2048（生产权衡：300KB 长文全量正则 146/s → ~400/s；排序
            # spearman(2048,8192)=0.67 在 hybrid_v2 的 0.2 权重下噪声 <7% 可容忍）
            rule = rule_scores_batch(texts, min(max_chars, 2048))
            rule_comp = rule_composite_vector(up_norm, rule)
            rec_cols: dict[str, Any] = {name: d[name] for name in s1_names}
            rec_cols["self_score"] = self_scores.tolist()
            rec_cols["composite"] = comp_f32.tolist()
            rec_cols["upstream_norm"] = [
                None if np.isnan(v) else float(np.float32(v)) for v in up_norm
            ]
            rec_cols["fasttext_score"] = self_scores.tolist()
            if bert is None:
                rec_cols["bert_score"] = [None] * n
            else:
                rec_cols["bert_score"] = [
                    float(np.float32(b)) if u else None for b, u in zip(bert, bert_used)
                ]
            rec_cols["composite_v2"] = comp_v2_f32.tolist()
            rec_cols["rule_score"] = rule.tolist()
            rec_cols["rule_composite"] = rule_comp.astype(np.float32).tolist()
            hybrid = hybrid_composite_vector(d["lang"], rule_comp, self_scores)
            rec_cols["hybrid_composite"] = hybrid.astype(np.float32).tolist()
            if goldrule_bundle is not None:
                llm_distill = goldrule_scores_batch(goldrule_bundle, texts)
                rec_cols["llm_distill"] = llm_distill.tolist()
            else:
                llm_distill = None
                rec_cols["llm_distill"] = [None] * n
            hybrid2 = hybrid_v2_vector(llm_distill, up_norm, rule, cfg)
            rec_cols["hybrid_v2"] = hybrid2.astype(np.float32).tolist()
            writer.extend(rec_cols)
            # 分位数排序键 = composite_v2（bert 缺失时数值上等于 legacy composite）；
            # rule_composite 并行落 pairs——S6 score_field 可切换（2026-09-27 用户裁定兜底轨）
            pairs_writer.extend(
                {
                    "domain": d["domain"],
                    "composite": comp_v2_f32.tolist(),
                    "rule_composite": rule_comp.astype(np.float32).tolist(),
                    "hybrid_composite": hybrid.astype(np.float32).tolist(),
                    "hybrid_v2": hybrid2.astype(np.float32).tolist(),
                }
            )
            self64 = self_scores.astype(np.float64)
            comp64 = comp_f32.astype(np.float64)
            for dom in set(d["domain"]):
                mask = np.array([x == dom for x in d["domain"]])
                h = per_domain_hist.setdefault(
                    dom, {"self_score": [0] * HIST_BIN_COUNT, "composite": [0] * HIST_BIN_COUNT}
                )
                for idx in np.minimum((self64[mask] * HIST_BIN_COUNT).astype(int), HIST_BIN_COUNT - 1):
                    h["self_score"][int(idx)] += 1
                for idx in np.minimum((comp64[mask] * HIST_BIN_COUNT).astype(int), HIST_BIN_COUNT - 1):
                    h["composite"][int(idx)] += 1
            corr_pairs.extend(_corr_sample_pairs(upstream, self_scores.tolist()))
            in_rows += n
        out_bytes = writer.close()
        pairs_bytes = pairs_writer.close()
    except Exception as exc:
        writer.abort()
        pairs_writer.abort()
        summary = {
            "stage": "s4_score",
            "source": source,
            "relpath": relpath,
            "in_rows": in_rows,
            "output_bytes": 0,
            "pairs_bytes": 0,
            "hist": {},
            "corr_pairs": [],
            "bert_joined": bert_joined,
            "processing_contract": processing_contract,
            "elapsed_sec": round(time.monotonic() - started, 3),
            "errors": {f"shard_error:{type(exc).__name__}": 1},
            "skipped": False,
        }
        write_summary(summary_path, summary)
        return summary

    elapsed = time.monotonic() - started
    corr_pairs = corr_pairs[:CORR_SAMPLE_MAX]
    selection_contract = selection_contract_base(
        s1_path, out_path, pairs_path, cfg, "hybrid_v2"
    )
    selection_contract["provenance"] = "computed_current"
    selection_contract["representation"] = "current_v2"
    summary = {
        "stage": "s4_score",
        "source": source,
        "relpath": relpath,
        "in_rows": in_rows,
        "output_bytes": out_bytes,
        "pairs_bytes": pairs_bytes,
        "hist": {k: per_domain_hist[k] for k in sorted(per_domain_hist)},
        "corr_pairs": corr_pairs,
        "bert_joined": bert_joined,
        "scorer_md5": model_fp,
        "processing_contract": processing_contract,
        "selection_contracts": {"hybrid_v2": selection_contract},
        "elapsed_sec": round(elapsed, 3),
        "rows_per_sec": round(in_rows / max(elapsed, 1e-9), 1),
        "errors": {},
        "skipped": False,
    }
    write_summary(summary_path, summary)
    return summary


def quantile_tables_dir(out_root: Path) -> Path:
    return out_root / "s4_quantiles" / "tables"


def build_quantile_tables(
    out_root: Path,
    force: bool = False,
    score_field: str = "composite",
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """score_field ∈ {composite, rule_composite}——分位表按选取键分文件物化
    （<domain>.npy / <domain>.rule.npy），_build.summary.json 同步后缀防混用。"""
    if score_field not in ("composite", "rule_composite", "hybrid_composite", "hybrid_v2"):
        raise ValueError(f"unsupported score_field: {score_field}")
    suffix = {
        "composite": "", "rule_composite": ".rule", "hybrid_composite": ".hybrid",
        "hybrid_v2": ".hybrid2",
    }[score_field]
    tables_dir = quantile_tables_dir(out_root)
    pairs_root = out_root / "s4_quantiles" / "pairs"
    if not pairs_root.is_dir():
        raise FileNotFoundError(f"quantile pass1 pairs missing: {pairs_root}")
    shards = sorted(pairs_root.rglob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"no pass1 pair shards under: {pairs_root}")
    manifest = hashlib.sha256()
    selection_manifest = hashlib.sha256()
    selection_by_rel: dict[str, dict[str, Any]] = {}
    s1_by_rel: dict[str, Path] = {}
    if score_field == "hybrid_v2":
        cfg = cfg or load_scorer_config()
        audit_path = out_root / "audits" / "s4_hybrid_v2_legacy_audit.json"
        audit_sha256 = _sha256_optional(audit_path)
        if audit_sha256 == "missing":
            audit_sha256 = ""
    total_bytes = 0
    for shard_index, shard in enumerate(shards, 1):
        stat = shard.stat()
        total_bytes += stat.st_size
        manifest.update(
            f"{shard.relative_to(pairs_root).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode(
                "utf-8"
            )
        )
        if score_field == "hybrid_v2":
            relpath = shard.relative_to(pairs_root)
            score_summary = (out_root / "s4_scored" / relpath).with_suffix(
                ".summary.json"
            )
            if not score_summary.is_file():
                raise FileNotFoundError(
                    f"S4 selection summary missing for quantile input: {score_summary}"
                )
            summary_payload = json.loads(score_summary.read_text(encoding="utf-8"))
            selection = (summary_payload.get("selection_contracts", {})).get(
                "hybrid_v2"
            )
            if not selection:
                raise ValueError(
                    f"hybrid_v2 selection contract missing: {score_summary}"
                )
            source = str(summary_payload.get("source") or "")
            original_relpath = str(summary_payload.get("relpath") or "")
            scored_path, expected_pairs, _ = scored_output_paths(
                out_root, source, original_relpath
            )
            if expected_pairs != shard:
                raise ValueError(f"S4 pair path/summary identity mismatch: {relpath}")
            s1_candidates = [
                out_root / "s1" / source / (original_relpath + ".parquet"),
                out_root / "s1" / source / original_relpath,
            ]
            expected_s1 = selection.get("s1_fingerprint")
            s1_path = next(
                (p for p in s1_candidates if _file_identity(p) == expected_s1),
                s1_candidates[0],
            )
            reason = validate_selection_contract(
                s1_path,
                scored_path,
                shard,
                summary_payload,
                cfg,
                audit_sha256,
            )
            if reason:
                raise ValueError(f"invalid hybrid_v2 selection contract: {reason}: {relpath}")
            selection_by_rel[relpath.as_posix()] = selection
            s1_by_rel[relpath.as_posix()] = s1_path
            selection_manifest.update(relpath.as_posix().encode("utf-8"))
            selection_manifest.update(b"\0")
            selection_manifest.update(
                json.dumps(
                    selection, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            )
            selection_manifest.update(b"\n")
        if shard_index % 500 == 0 or shard_index == len(shards):
            log.info(
                "s4 quantile contract scan: %d/%d shards field=%s",
                shard_index,
                len(shards),
                score_field,
            )
    input_contract = {
        "schema_version": 2,
        "pair_shards": len(shards),
        "pair_bytes": total_bytes,
        "pair_manifest_sha256": manifest.hexdigest(),
        "score_field": score_field,
    }
    if score_field == "hybrid_v2":
        input_contract["selection_contract_manifest_sha256"] = (
            selection_manifest.hexdigest()
        )
    build_id = hashlib.sha256(
        json.dumps(
            input_contract, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()[:16]
    done_path = tables_dir / f"_build{suffix}.summary.json"
    if not force and done_path.is_file():
        summary = json.loads(done_path.read_text("utf-8"))
        if summary.get("input_contract") == input_contract:
            summary["skipped"] = True
            return summary

    started = time.monotonic()

    per_domain: dict[str, list[np.ndarray]] = {}

    def load_value_shard(path: Path) -> tuple[list[tuple[str, np.ndarray]], int]:
        """Read and normalize one pair shard without mutating shared state."""
        try:
            pf = pq.ParquetFile(path)
        except Exception as exc:
            raise RuntimeError(f"cannot open S4 quantile pair shard: {path}") from exc
        # Production pair shards are normally assigned to one coarse domain.
        # Prove that from Parquet statistics once, then avoid decoding and
        # object-sorting the same domain string for billions of rows.  Fall
        # back to row-level grouping whenever statistics are absent or mixed.
        domain_idx = pf.schema_arrow.get_field_index("domain")
        constant_domain: str | None = None
        domain_stats_complete = domain_idx >= 0 and pf.metadata.num_row_groups > 0
        if domain_stats_complete:
            for row_group in range(pf.metadata.num_row_groups):
                stats = pf.metadata.row_group(row_group).column(domain_idx).statistics
                if (
                    stats is None
                    or stats.null_count is None
                    or int(stats.null_count) != 0
                    or stats.min is None
                    or stats.max is None
                    or stats.min != stats.max
                ):
                    domain_stats_complete = False
                    break
                value = (
                    stats.min.decode("utf-8")
                    if isinstance(stats.min, bytes)
                    else str(stats.min)
                )
                if constant_domain is None:
                    constant_domain = value
                elif constant_domain != value:
                    domain_stats_complete = False
                    break
        if not domain_stats_complete:
            constant_domain = None
        rel = path.relative_to(pairs_root)
        selection = selection_by_rel.get(rel.as_posix())
        representation = str((selection or {}).get("representation") or "current_v2")
        upstream: np.ndarray | None = None
        upstream_all_missing = False
        if score_field == "hybrid_v2" and representation == "legacy_zero_upstream_v1":
            s1_pf = pq.ParquetFile(s1_by_rel[rel.as_posix()])
            upstream_idx = s1_pf.schema_arrow.get_field_index("upstream_score")
            null_count = 0
            stats_complete = upstream_idx >= 0
            if stats_complete:
                for row_group in range(s1_pf.metadata.num_row_groups):
                    stats = s1_pf.metadata.row_group(row_group).column(upstream_idx).statistics
                    if stats is None or stats.null_count is None:
                        stats_complete = False
                        break
                    null_count += int(stats.null_count)
            rows = int(s1_pf.metadata.num_rows)
            if rows != int(pf.metadata.num_rows):
                raise ValueError(f"S1/pairs row closure failed: {rel}")
            if stats_complete and null_count == rows:
                upstream_all_missing = True
            elif not stats_complete or null_count:
                upstream = s1_pf.read(
                    columns=["upstream_score"], use_threads=False
                ).column("upstream_score").combine_chunks().to_numpy(
                    zero_copy_only=False
                )
        pair_columns = (
            [score_field]
            if constant_domain is not None
            else ["domain", score_field]
        )
        table = pf.read(columns=pair_columns, use_threads=False)
        composites = (
            table.column(score_field)
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            .astype(np.float32)
        )
        if score_field == "hybrid_v2" and representation == "legacy_zero_upstream_v1":
            if upstream_all_missing:
                hcfg = (cfg or {}).get("hybrid_v2") or {}
                full_weight = sum(
                    float(hcfg.get(key, default))
                    for key, default in (
                        ("llm_distill_weight", 0.5),
                        ("upstream_weight", 0.3),
                        ("rule_weight", 0.2),
                    )
                )
                retained_weight = full_weight - float(hcfg.get("upstream_weight", 0.3))
                composites = np.clip(
                    composites * (full_weight / retained_weight), 0.0, 1.0
                ).astype(np.float32)
            elif upstream is not None:
                composites = effective_hybrid_v2(
                    composites, upstream, cfg or {}, representation
                )
        n = len(composites)
        if constant_domain is not None:
            return [(constant_domain, composites)], n
        domains = table.column("domain").to_pylist()
        if not domains:
            return [], 0
        order = np.argsort(np.array(domains, dtype=object), kind="stable")
        sorted_domains = [domains[i] for i in order]
        sorted_vals = composites[order]
        groups: list[tuple[str, np.ndarray]] = []
        start = 0
        while start < n:
            end = start
            dom = sorted_domains[start]
            while end < n and sorted_domains[end] == dom:
                end += 1
            groups.append((dom, sorted_vals[start:end]))
            start = end
        return groups, n

    # Multiple bounded readers hide cold network-filesystem latency without
    # allowing completed shard arrays to accumulate outside ``per_domain``.
    # NumPy/Arrow release the GIL here; Arrow's own per-file pool is disabled
    # above to avoid workers*CPU nested thread oversubscription.
    reader_workers = max(1, int(os.environ.get("YUFORMER_S4_BUILD_WORKERS", "8")))
    max_pending = reader_workers * 2
    total = 0
    completed = 0
    shard_iter = iter(shards)
    with ThreadPoolExecutor(max_workers=reader_workers) as executor:
        pending: dict[Future[tuple[list[tuple[str, np.ndarray]], int]], Path] = {}
        for _ in range(min(max_pending, len(shards))):
            path = next(shard_iter)
            pending[executor.submit(load_value_shard, path)] = path
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                path = pending.pop(future)
                try:
                    groups, rows = future.result()
                except Exception as exc:
                    raise RuntimeError(f"S4 quantile value scan failed: {path}") from exc
                for domain, values in groups:
                    per_domain.setdefault(domain, []).append(values)
                total += rows
                completed += 1
                if completed % 250 == 0 or completed == len(shards):
                    log.info(
                        "s4 quantile value scan: %d/%d shards rows=%d field=%s workers=%d",
                        completed,
                        len(shards),
                        total,
                        score_field,
                        reader_workers,
                    )
                try:
                    next_path = next(shard_iter)
                except StopIteration:
                    continue
                pending[executor.submit(load_value_shard, next_path)] = next_path

    tables_dir.mkdir(parents=True, exist_ok=True)
    table_meta: dict[str, dict[str, Any]] = {}
    for dom in sorted(per_domain):
        arr = np.concatenate(per_domain[dom]).astype(np.float32)
        arr.sort()
        npy_path = tables_dir / f"{_domain_key(dom)}{suffix}.{build_id}.npy"
        tmp_npy = npy_path.with_name(npy_path.name + f".tmp-{os.getpid()}")
        with tmp_npy.open("wb") as handle:
            np.save(handle, arr)
        os.replace(tmp_npy, npy_path)
        table_meta[dom] = {
            "n": int(len(arr)),
            "min": float(arr[0]) if len(arr) else None,
            "max": float(arr[-1]) if len(arr) else None,
            "npy": npy_path.name,
            "md5": hashlib.md5(arr.tobytes()).hexdigest(),
        }
    summary = {
        "stage": "s4_quantile_build",
        "score_field": score_field,
        "pair_shards": len(shards),
        "total_rows": total,
        "domains": table_meta,
        "input_contract": input_contract,
        "build_id": build_id,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "skipped": False,
    }
    tmp = done_path.with_name(done_path.name + ".tmp")
    tmp.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8")
    tmp.replace(done_path)
    return summary


def _domain_key(domain: str) -> str:
    return hashlib.blake2b(domain.encode("utf-8"), digest_size=8).hexdigest()


def load_quantile_tables(out_root: Path) -> dict[str, np.ndarray]:
    tables_dir = quantile_tables_dir(out_root)
    done_path = tables_dir / "_build.summary.json"
    if not done_path.is_file():
        raise FileNotFoundError(
            f"quantile tables not built: {done_path} (run the s4 driver stage / quantile build first)"
        )
    meta = json.loads(done_path.read_text(encoding="utf-8"))["domains"]
    tables: dict[str, np.ndarray] = {}
    for dom, info in meta.items():
        # mmap 而非载入内存：分位表全集 ~25GB（web 域 22GB），quantile pass 每 worker
        # 都调用本函数——实载入会在 160 worker 下复制 160 份（~4TB）直接 OOM 挂死
        # （E-32，2026-10-08）；mmap 后物理页由 page cache 共享。
        # 注意：clean MAP_PRIVATE 页仍计入各进程 RSS（实测单 worker 触表后 RSS 16GB+），
        # 且随机二分缺页拖尾严重——quantile pass 应再经 build_compact_quantile_tables
        # 压缩后父进程预载共享（E-33），不要直接拿本函数的返回跑 160 并发。
        tables[dom] = np.load(tables_dir / info["npy"], mmap_mode="r")
    return tables


COMPACT_TABLE_POINTS = 2 ** 25  # 每域 rank 等距采样点数（E-33，q 误差上界 1/(m-1)≈3e-8）


def build_compact_quantile_tables(
    tables: dict[str, np.ndarray], max_points: int = COMPACT_TABLE_POINTS
) -> dict[str, np.ndarray]:
    """全量分位表的 rank 等距紧凑采样（E-33）。

    quantile pass 每 worker 对每行做 searchsorted：全量表 web 域 22GB，
    160 worker 下 RSS 口径内存 ~3.5TB 必然 OOM（E-32），mmap 也救不了
    RSS 计帐与随机缺页拖尾。compact 表按 rank 等距取 min(m, n) 点，
    仍是 sorted 数组 → quantile_of 原样可用；误差上界 1/(m-1)≈3e-8，
    远小于 precision=1e-6 的舍入步长（只会令 ~6% 行的舍入结果差一个
    1e-6 步，等价于 top-k 截断线附近数百行样本的成员变化，可忽略）。
    父进程构建一次，fork 后全 worker COW 共享（物理 ~150MB）。
    """
    compact: dict[str, np.ndarray] = {}
    for dom, arr in tables.items():
        n = len(arr)
        if n <= max_points:
            compact[dom] = np.array(arr)  # 小域精确拷贝驻留（arr 为 memmap 需 copy）
            continue
        idx = np.linspace(0, n - 1, max_points).round().astype(np.int64)
        compact[dom] = np.ascontiguousarray(arr[idx])
    return compact


def quantile_of(sorted_arr: np.ndarray, values: np.ndarray, precision: float) -> np.ndarray:
    n = len(sorted_arr)
    if n == 0:
        return np.zeros(len(values), dtype=np.float64)
    if n == 1:
        return (values >= sorted_arr[0]).astype(np.float64)
    left = np.searchsorted(sorted_arr, values, side="left")
    lo = sorted_arr[np.clip(left - 1, 0, n - 1)]
    hi = sorted_arr[np.clip(left, 0, n - 1)]
    span = np.maximum(hi - lo, 1e-12)
    frac = np.clip((values - lo) / span, 0.0, 1.0)
    rank = np.where(
        left == 0, 0.0, np.where(left >= n, float(n - 1), left - 1 + frac)
    )
    q = np.clip(rank / (n - 1), 0.0, 1.0)
    return np.round(q / precision) * precision


def process_shard_quantile(
    out_root: Path,
    source: str,
    relpath: str,
    cfg: dict[str, Any],
    tables: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    out_path, summary_path = s4_output_paths(out_root, source, relpath)
    scored_path, _, _ = scored_output_paths(out_root, source, relpath)
    build_path = quantile_tables_dir(out_root) / "_build.summary.json"
    build_sha256 = (
        hashlib.sha256(build_path.read_bytes()).hexdigest() if build_path.is_file() else "missing"
    )
    processing_contract = {
        "schema_version": 2,
        "scored_fingerprint": _file_identity(scored_path),
        "quantile_build_sha256": build_sha256,
        "quantile_score_column": "composite_v2",
        "precision": float(cfg.get("quantile", {}).get("precision", 1e-6)),
    }
    if summary_path.is_file() and out_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("processing_contract") == processing_contract and not summary.get("errors"):
            summary["skipped"] = True
            return summary
    if not scored_path.is_file():
        raise FileNotFoundError(f"s4_scored shard missing: {scored_path}")

    started = time.monotonic()
    precision = float(cfg.get("quantile", {}).get("precision", 1e-6))
    if tables is None:
        tables = load_quantile_tables(out_root)

    in_rows = 0
    writer = BatchWriter(out_path, S4_SCHEMA)
    try:
        pf = pq.ParquetFile(scored_path)
        for batch in safe_iter_batches(pf, batch_size=4096):
            rows = batch.to_pylist()
            if not rows:
                continue
            domains = [r["domain"] for r in rows]
            # 排序键 = composite_v2（三轨融合分；bert 缺失时 = legacy composite）
            composite = np.array([r["composite_v2"] for r in rows], dtype=np.float32)
            q = np.zeros(len(rows), dtype=np.float64)
            dom_arr = np.array(domains, dtype=object)
            order = np.argsort(dom_arr, kind="stable")
            sorted_doms = [domains[i] for i in order]
            sorted_comp = composite[order]
            start = 0
            n = len(sorted_doms)
            while start < n:
                end = start
                dom = sorted_doms[start]
                while end < n and sorted_doms[end] == dom:
                    end += 1
                table = tables.get(dom)
                if table is None or len(table) == 0:
                    q[order[start:end]] = 0.0
                else:
                    q[order[start:end]] = quantile_of(table, sorted_comp[start:end], precision)
                start = end
            for i, row in enumerate(rows):
                rec = {name: row.get(name) for name in SCORED_SCHEMA.names}
                rec["domain_quantile"] = float(np.float32(q[i]))
                writer.add(rec)
            in_rows += len(rows)
        out_bytes = writer.close()
    except Exception as exc:
        writer.abort()
        summary = {
            "stage": "s4_quantile",
            "source": source,
            "relpath": relpath,
            "in_rows": in_rows,
            "output_bytes": 0,
            "elapsed_sec": round(time.monotonic() - started, 3),
            "errors": {f"shard_error:{type(exc).__name__}": 1},
            "processing_contract": processing_contract,
            "skipped": False,
        }
        write_summary(summary_path, summary)
        return summary

    elapsed = time.monotonic() - started
    summary = {
        "stage": "s4_quantile",
        "source": source,
        "relpath": relpath,
        "in_rows": in_rows,
        "output_bytes": out_bytes,
        "elapsed_sec": round(elapsed, 3),
        "rows_per_sec": round(in_rows / max(elapsed, 1e-9), 1),
        "errors": {},
        "processing_contract": processing_contract,
        "skipped": False,
    }
    write_summary(summary_path, summary)
    return summary


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    rx = _rankdata(x)
    ry = _rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def merge_s4_global_summary(
    out_root: Path,
    score_summaries: list[dict[str, Any]],
    quantile_summaries: list[dict[str, Any]],
    build_summary: dict[str, Any] | None,
    cfg: dict[str, Any],
) -> Path:
    per_source: dict[str, Counter[str]] = {}
    per_domain_hist: dict[str, dict[str, list[int]]] = {}
    corr_by_source: dict[str, list[list[float]]] = {}
    for s in score_summaries:
        src = s.get("source", "?")
        acc = per_source.setdefault(src, Counter())
        acc["in_rows"] += int(s.get("in_rows") or 0)
        acc["elapsed_sec"] += float(s.get("elapsed_sec") or 0.0)
        for dom, h in (s.get("hist") or {}).items():
            dh = per_domain_hist.setdefault(dom, {"self_score": [0] * HIST_BIN_COUNT, "composite": [0] * HIST_BIN_COUNT})
            for key in ("self_score", "composite"):
                for i, v in enumerate(h.get(key, [])):
                    dh[key][i] += int(v)
        corr_by_source.setdefault(src, []).extend(s.get("corr_pairs") or [])

    total_rows = sum(int(s.get("in_rows") or 0) for s in score_summaries)
    total_elapsed = sum(float(s.get("elapsed_sec") or 0.0) for s in score_summaries)
    quantile_rows = sum(int(s.get("in_rows") or 0) for s in quantile_summaries)

    spearman_by_source = {}
    for src, pairs in sorted(corr_by_source.items()):
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        spearman_by_source[src] = spearman(xs, ys)

    try:
        model_path = resolve_model_path(cfg)
        model_md5 = hashlib.md5(model_path.read_bytes()).hexdigest()
    except FileNotFoundError:
        model_path = Path(str(cfg.get("model", "?")))
        model_md5 = None

    payload = {
        "stage": "s4",
        "score_shards": len(score_summaries),
        "quantile_shards": len(quantile_summaries),
        "shard_errors": sum(1 for s in score_summaries + quantile_summaries if s.get("errors") or s.get("shard_fatal")),
        "total_rows": total_rows,
        "quantile_rows": quantile_rows,
        "wall_rows_per_sec": round(total_rows / max(total_elapsed, 1e-9), 1),
        "model": str(model_path),
        "model_md5": model_md5,
        "composite": {k: cfg["composite"][k] for k in sorted(cfg["composite"])},
        "composite_v2": {k: cfg["composite_v2"][k] for k in sorted(cfg["composite_v2"])}
        if cfg.get("composite_v2")
        else None,
        "upstream_normalize": {k: cfg.get("upstream_normalize", {}).get(k) for k in sorted(cfg.get("upstream_normalize", {}))},
        "quantile_build": build_summary,
        "per_source": {k: dict(v) for k, v in sorted(per_source.items())},
        "hist_per_domain": {k: per_domain_hist[k] for k in sorted(per_domain_hist)},
        "spearman_upstream_vs_self_score": spearman_by_source,
    }
    path = out_root / "stage_s4_global_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8")
    return path
