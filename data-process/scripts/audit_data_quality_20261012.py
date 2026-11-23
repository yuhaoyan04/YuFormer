#!/usr/bin/env python3
"""Published pretrain mix quality-gate audit.

seed=20261012 确定性抽样 9 域 × 300（web 500，共 3200 篇），逐篇机器检查：
语言分布 / 文本卫生 / 退化文本 / 短文档 / PII / 去污染回验 / hybrid_v2 分位健全性。

只读源数据，产出写 data/audit_data_quality_20261012/（report.json + samples.tsv）。
"""
from __future__ import annotations

import hashlib
import heapq
import json
import os
import re
import sys
import time
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/opt/huawei/explorer-env/dataset/yyh_arcagi/YUFormer/YuFormer")
DATA = ROOT / "data"
MIX = DATA / "processed" / "mix_members" / "mix_m3_e1_100b.parquet"
S3 = DATA / "processed" / "s3"
S4 = DATA / "processed" / "s4_scored"
QUANT = DATA / "processed" / "s4_quantiles" / "tables"
DECONTAM = DATA / "decontam_index"
LID = DATA / "models" / "lid.176.ftz"
OUT = DATA / "audit_data_quality_20261019"
SEED = "20261019"

sys.path.insert(0, str(ROOT / "data-process" / "src"))
from yuformer_data.scale.decontam import (  # noqa: E402
    BenchmarkIndex,
    doc_shingle_hashes,
    effective_min_distinct_hits,
    load_decontam_config,
)

SAMPLE_N = {"web": 800}  # 9 域 × 300 + web 追加 500 = 3200（契约口径）
DEFAULT_N = 300


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def decontam_hit_summary(
    shingles: np.ndarray,
    index: BenchmarkIndex,
    thresholds: dict[str, int],
) -> tuple[dict[str, int], list[str]]:
    """Return distinct hit counts and benchmarks meeting the production policy."""
    codes = index.hit_codes_batch(shingles)
    raw: dict[str, int] = {}
    qualified: list[str] = []
    for code in np.unique(codes[codes >= 0]):
        name = index.names[int(code)]
        count = int(len(np.unique(shingles[codes == code])))
        raw[name] = count
        if count >= int(thresholds.get(name, 1)):
            qualified.append(name)
    return raw, sorted(qualified)

# 域 → 期望语言（top-1 占比阈值）
LANG_EXPECT = {
    "web": ("en", 0.70),
    "chinese": ("zh", 0.70),
    "academic": ("en", 0.50),
    "math": ("en", 0.50),
    "qa": ("en", 0.50),
    "books": ("en", 0.50),
    "encyclopedia": (None, 0.0),   # wikipedia-en/zh + wikitext 混合，只看分布
    "multilingual": (None, 0.0),   # fineweb-2 多语，无单一期望
    "code": (None, 0.0),           # LID 对代码无意义
}

TAG_RE = re.compile(r"<[^<>\n]{1,200}>")
SCRIPT_STYLE_RE = re.compile(r"(?is)<\s*(script|style)[\s>]")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
IDCARD_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
URL_RE = re.compile(r"https?://", re.I)
CHAR_RUN_RE = re.compile(r"(.)\1{49,}", re.S)
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def u_hash(doc_id: str) -> float:
    h = hashlib.blake2b(doc_id.encode(), digest_size=8, person=SEED.encode()[:8]).digest()
    return int.from_bytes(h, "little") / 2**64


def stratified_sample() -> tuple[dict[str, list[int]], dict]:
    """三遍流式读 mix parquet（2GB 内存安全）：域计数+token 总量 → top-K 行号 → 取出字段。"""
    pf = pq.ParquetFile(MIX)
    n_rg = pf.metadata.num_row_groups
    t0 = time.monotonic()
    counts: Counter = Counter()
    total_tokens = 0
    rg_bounds = [0]
    for rg in range(n_rg):
        tbl = pf.read_row_group(rg, columns=["domain", "n_tokens"])
        counts.update(tbl.column("domain").to_pylist())
        total_tokens += int(np.sum(tbl.column("n_tokens").to_numpy()))
        rg_bounds.append(rg_bounds[-1] + tbl.num_rows)
    print(f"  pass1 counts {dict(counts)} tokens={total_tokens/1e9:.2f}B {time.monotonic()-t0:.0f}s", flush=True)

    targets = {d: SAMPLE_N.get(d, DEFAULT_N) for d in counts}
    heaps: dict[str, list] = {d: [] for d in counts}  # max-heap of (u, row)
    for rg in range(n_rg):
        col = pf.read_row_group(rg, columns=["doc_id", "domain"])
        ids = col.column("doc_id").to_pylist()
        doms = col.column("domain").to_pylist()
        base = rg_bounds[rg]
        for i, (did, d) in enumerate(zip(ids, doms)):
            h = heaps[d]
            u = u_hash(did)
            if len(h) < targets[d]:
                heapq.heappush(h, (-u, base + i))
            elif u < -h[0][0]:
                heapq.heapreplace(h, (-u, base + i))
    sel_rows: dict[str, list[int]] = {
        d: sorted(r for _, r in h) for d, h in heaps.items() if h
    }
    print(f"  pass2 selected {sum(len(v) for v in sel_rows.values())} rows", flush=True)

    wanted = {r for rows in sel_rows.values() for r in rows}
    picked: dict[int, dict] = {}
    for rg in range(n_rg):
        lo, hi = rg_bounds[rg], rg_bounds[rg + 1]
        hit = [r for r in wanted if lo <= r < hi]
        if not hit:
            continue
        tbl = pf.read_row_group(rg, columns=["doc_id", "domain", "composite", "n_tokens"])
        # Production members use row groups up to 50M rows.  Converting every
        # value to Python objects just to keep a few sampled offsets inflated
        # this audit to tens of GiB RSS.  Take the selected Arrow rows first.
        local_rows = np.asarray([row - lo for row in hit], dtype=np.int64)
        tbl = tbl.take(pa.array(local_rows, type=pa.int64()))
        ids = tbl.column("doc_id").to_pylist()
        comps = tbl.column("composite").to_pylist()
        ntoks = tbl.column("n_tokens").to_pylist()
        doms = tbl.column("domain").to_pylist()
        for i, r in enumerate(hit):
            picked[r] = {
                "doc_id": ids[i], "domain": doms[i],
                "composite": float(comps[i]), "n_tokens": int(ntoks[i]),
            }
    meta = {
        "mix_rows": pf.metadata.num_rows,
        "mix_tokens": total_tokens,
        "domain_counts": dict(counts),
        "sample_plan": targets,
    }
    return picked, meta


def load_reusable_sample_roster(
    audit_dir: Path, manifest_path: Path, mix_name: str
) -> tuple[dict[int, dict], dict, dict]:
    """Reuse only the deterministic roster after proving its dataset identity."""
    report_path = audit_dir / "report.json"
    tsv_path = audit_dir / "samples.tsv"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    identity = report.get("identity") or {}
    checks = {
        "mix_name": identity.get("mix_name") == mix_name,
        "seed": str(report.get("seed")) == SEED,
        "members_sha256": identity.get("members_sha256") == sha256_file(MIX),
        "manifest_sha256": identity.get("manifest_sha256") == sha256_file(manifest_path),
    }
    if not all(checks.values()):
        raise SystemExit(f"reusable sample roster identity mismatch: {checks}")
    picked: dict[int, dict] = {}
    with tsv_path.open(encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle, delimiter="\t")):
            picked[index] = {
                "doc_id": row["doc_id"],
                "domain": row["domain"],
                "composite": 0.0,
                "n_tokens": int(row["n_tokens"]),
            }
    counts = Counter(row["domain"] for row in picked.values())
    expected = {domain: SAMPLE_N.get(domain, DEFAULT_N) for domain in counts}
    if len(picked) != 3200 or len({row["doc_id"] for row in picked.values()}) != len(picked):
        raise SystemExit("reusable sample roster must contain 3,200 unique doc_ids")
    if any(counts[domain] != target for domain, target in expected.items()):
        raise SystemExit(f"reusable sample roster domain counts mismatch: {dict(counts)}")
    mix_meta = dict(report["mix"])
    mix_meta.pop("file", None)
    provenance = {
        "mode": "identity_verified_reuse",
        "source_identity_checks": checks,
    }
    print(f"  reused identity-bound sample roster: {len(picked)} rows", flush=True)
    return picked, mix_meta, provenance


def check_index_cache() -> tuple[list[Path], bool]:
    files = sorted(S3.rglob("*.parquet"))
    fp = hashlib.sha256(
        json.dumps([(str(f.relative_to(S3)), f.stat().st_size) for f in files],
                   separators=(",", ":")).encode()
    ).hexdigest()[:16]
    tag = (S3 / ".doc_index_cache" / "fingerprint.txt")
    cached = tag.is_file() and tag.read_text().strip() == fp
    print(f"  s3 files={len(files)} fp={fp} cache_ok={cached}", flush=True)
    return files, cached


def fetch_by_rows(files: list[Path], files_s4: list[Path] | None,
                  locs: dict[int, tuple[int, int]]) -> dict[int, dict]:
    """locs: sample_idx → (fidx, row)。按文件批量取 s3(doc_id/text/domain/source)；
    s4 镜像行组布局与 s3 不同（4096 vs 2048），需按各自行组边界独立映射。"""
    out: dict[int, dict] = {}
    by_file: dict[int, list[int]] = defaultdict(list)
    for si, (fi, _) in locs.items():
        by_file[fi].append(si)
    for fi, sis in sorted(by_file.items()):
        pf = pq.ParquetFile(files[fi])
        md = pf.metadata
        bounds = np.concatenate([[0], np.cumsum([md.row_group(i).num_rows for i in range(md.num_row_groups)])])
        rows = np.array([locs[si][1] for si in sis])
        rg_of = np.searchsorted(bounds, rows, side="right") - 1
        s4_path = files_s4[fi] if files_s4 is not None else None
        pf4 = pq.ParquetFile(s4_path) if s4_path and s4_path.is_file() else None
        bounds4 = None
        if pf4 is not None:
            md4 = pf4.metadata
            if md4.num_rows == md.num_rows:
                bounds4 = np.concatenate([[0], np.cumsum([md4.row_group(i).num_rows for i in range(md4.num_row_groups)])])
            else:
                pf4 = None  # 行数不一致：不做镜像假设
        for rg in np.unique(rg_of):
            sel = [si for si, rgz in zip(sis, rg_of) if rgz == rg]
            lo = int(bounds[rg])
            tbl = pf.read_row_group(int(rg), columns=["doc_id", "text", "domain", "source"])
            ids = tbl.column("doc_id").to_pylist()
            texts = tbl.column("text").to_pylist()
            doms = tbl.column("domain").to_pylist()
            srcs = tbl.column("source").to_pylist()
            # s4：把全局行号映射到 s4 行组（按行号排序批量读）
            h2: dict[int, float] = {}
            if pf4 is not None:
                order = sorted(sel, key=lambda s: locs[s][1])
                for s4rg in np.unique(np.searchsorted(bounds4, [locs[s][1] for s in order], side="right") - 1):
                    grp = [s for s in order
                           if np.searchsorted(bounds4, locs[s][1], side="right") - 1 == s4rg]
                    lo4 = int(bounds4[s4rg])
                    t4 = pf4.read_row_group(int(s4rg), columns=["doc_id", "hybrid_v2"])
                    ids4 = t4.column("doc_id").to_pylist()
                    h4 = t4.column("hybrid_v2").to_pylist()
                    for s in grp:
                        i4 = locs[s][1] - lo4
                        if 0 <= i4 < len(ids4):
                            h2[s] = (ids4[i4], float(h4[i4]))
            for si in sel:
                i = locs[si][1] - lo
                rec = {
                    "s3_doc_id": ids[i], "text": texts[i] or "",
                    "s3_domain": doms[i], "source": srcs[i],
                    "hybrid_v2": None, "s4_aligned": False,
                }
                got = h2.get(si)
                if got and got[0] == ids[i]:
                    rec["hybrid_v2"] = got[1]
                    rec["s4_aligned"] = True
                out[si] = rec
    return out


def text_checks(text: str) -> dict:
    n = max(len(text), 1)
    fffd = text.count("\ufffd")
    ctrl = len(CTRL_RE.findall(text))
    tags = TAG_RE.findall(text)
    tag_chars = sum(len(t) for t in tags)
    lines = [l for l in text.splitlines() if l.strip()]
    uniq_lines = len(set(lines))
    line_dup = (1 - uniq_lines / len(lines)) if len(lines) >= 4 else 0.0
    words = text[:20000].split()
    ngram_dup = 0.0
    n_sh = 0
    if len(words) >= 108:
        sh = set()
        dup = 0
        for j in range(len(words) - 7):
            g = " ".join(words[j:j + 8])
            if g in sh:
                dup += 1
            else:
                sh.add(g)
        n_sh = len(words) - 7
        ngram_dup = dup / n_sh
    return {
        "fffd_ratio": fffd / n,
        "ctrl_ratio": ctrl / n,
        "html_tag_ratio": tag_chars / n,
        "html_script_style": bool(SCRIPT_STYLE_RE.search(text)),
        "url_density": len(URL_RE.findall(text)) / (n / 1000),
        "char_repeat": bool(CHAR_RUN_RE.search(text)),
        "line_dup_ratio": line_dup,
        "ngram_dup_ratio": ngram_dup,
        "n_8grams": n_sh,
    }


def main(argv: list[str] | None = None) -> None:
    global MIX, S3, S4, QUANT, DECONTAM, LID, OUT, SEED
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed-root", type=Path, default=DATA / "processed")
    ap.add_argument("--mix", default="mix_m3_e1_100b")
    ap.add_argument("--decontam-index", type=Path, default=DATA / "decontam_index")
    ap.add_argument(
        "--decontam-config",
        type=Path,
        default=ROOT / "data-process" / "configs" / "decontam.yaml",
    )
    ap.add_argument("--lid-model", type=Path, default=DATA / "models" / "lid.176.ftz")
    ap.add_argument("--out-dir", type=Path, default=DATA / "audit_data_quality_20261019")
    ap.add_argument("--seed", default="20261019")
    ap.add_argument(
        "--reuse-samples-from",
        type=Path,
        default=None,
        help="reuse report.json/samples.tsv roster only when mix, manifest, seed and hashes match",
    )
    ap.add_argument(
        "--preload-id-index",
        action="store_true",
        help="warm the 97GB sorted ID index with large sequential reads before search",
    )
    args = ap.parse_args(argv)
    processed = args.processed_root.resolve()
    MIX = processed / "mix_members" / f"{args.mix}.parquet"
    S3 = processed / "s3"
    S4 = processed / "s4_scored"
    QUANT = processed / "s4_quantiles" / "tables"
    DECONTAM = args.decontam_index.resolve()
    decontam_cfg = load_decontam_config(args.decontam_config)
    LID = args.lid_model.resolve()
    OUT = args.out_dir.resolve()
    SEED = str(args.seed)
    manifest_path = processed / "mix_order" / f"{args.mix}.manifest.json"
    for path in (MIX, S3, S4, QUANT, DECONTAM, LID, manifest_path):
        if not path.exists():
            raise SystemExit(f"missing required input: {path}")
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    print("== pass1-3: stratified sample from mix ==", flush=True)
    if args.reuse_samples_from is not None:
        picked, mix_meta, sampling_provenance = load_reusable_sample_roster(
            args.reuse_samples_from.resolve(), manifest_path, args.mix
        )
    else:
        picked, mix_meta = stratified_sample()
        sampling_provenance = {"mode": "fresh_deterministic_scan"}
    all_rows = sorted(picked)
    row2si = {r: i for i, r in enumerate(all_rows)}

    print("== index cache check ==", flush=True)
    files, cache_ok = check_index_cache()
    if not cache_ok:
        raise SystemExit("doc_index_cache fingerprint mismatch — 需重建索引（本审计不自动重建）")
    files_s4 = [Path(str(f).replace("/s3/", "/s4_scored/", 1)) for f in files]

    print("== memmap index + searchsorted ==", flush=True)
    ids = np.load(S3 / ".doc_index_cache" / "ids.npy", mmap_mode="r")
    fidx = np.load(S3 / ".doc_index_cache" / "fidx.npy", mmap_mode="r")
    ridx = np.load(S3 / ".doc_index_cache" / "ridx.npy", mmap_mode="r")
    if args.preload_id_index:
        print(f"  sequential preload {ids.nbytes / 2**30:.1f} GiB ID index", flush=True)
        # Warm the shared page cache with large sequential reads.  np.array(ids,
        # copy=True) used page-fault-sized mmap reads and retained a second
        # 96.6-GiB anonymous copy, which was slower on NFS and doubled RSS.
        ids_path = S3 / ".doc_index_cache" / "ids.npy"
        with ids_path.open("rb", buffering=0) as handle:
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(
                    handle.fileno(), 0, 0, getattr(os, "POSIX_FADV_SEQUENTIAL", 2)
                )
            while handle.read(64 << 20):
                pass
        ids_search = ids
    else:
        ids_search = ids
    q = np.frombuffer(b"".join(bytes.fromhex(picked[r]["doc_id"]) for r in all_rows), dtype="V16")
    order = np.argsort(q, kind="stable")  # 排序后二分探测单调 → 网络存储顺序读
    qs = q[order]
    pos_s = np.searchsorted(ids_search, qs)
    pos_s = np.clip(pos_s, 0, len(ids_search) - 1)
    hit_s = np.asarray(ids_search[pos_s]) == qs
    pos = np.empty(len(q), dtype=np.int64)
    pos[order] = pos_s
    found = np.empty(len(q), dtype=bool)
    found[order] = hit_s
    locs: dict[int, tuple[int, int]] = {}
    missing_text: list[str] = []
    for k, r in enumerate(all_rows):
        if found[k]:
            locs[row2si[r]] = (int(fidx[pos[k]]), int(ridx[pos[k]]))
        else:
            missing_text.append(picked[r]["doc_id"])
    del ids_search, ids
    print(f"  index hit {len(locs)}/{len(all_rows)} missing={len(missing_text)}", flush=True)

    print("== fetch text (s3) + hybrid_v2 (s4 mirror) ==", flush=True)
    fetched = fetch_by_rows(files, files_s4, locs)
    n_s4_aligned = sum(1 for v in fetched.values() if v["s4_aligned"])
    print(f"  fetched text={len(fetched)} s4_aligned={n_s4_aligned}", flush=True)

    print("== load lid + decontam index + quantile tables ==", flush=True)
    import fasttext
    lid = fasttext.load_model(str(LID))
    bidx = BenchmarkIndex(DECONTAM)
    qmeta = json.loads((QUANT / "_build.hybrid2.summary.json").read_text())
    qtab = {d: np.load(QUANT / m["npy"], mmap_mode="r") for d, m in qmeta["domains"].items()}

    print("== per-doc checks ==", flush=True)
    samples: list[dict] = []
    for si, r in enumerate(all_rows):
        base = dict(picked[r])
        f = fetched.get(si)
        rec: dict = {**base, "text_found": f is not None}
        if f is None:
            rec.update({"lang": None, "hybrid_v2": None, "checks": {}, "decontam_hits": [],
                        "pct_hybrid_v2": None, "s4_aligned": False})
            samples.append(rec)
            continue
        text = f["text"]
        rec.update({"source": f["source"], "s3_domain": f["s3_domain"], "hybrid_v2": f["hybrid_v2"],
                    "s4_aligned": f["s4_aligned"], "text": text})
        # 语言
        head = text[:4000].replace("\n", " ").strip()
        if head:
            lab, prob = lid.predict(head, k=1)
            lang = lab[0].replace("__label__", "")
            if lang in ("zh-CN", "zh-TW"):
                lang = "zh"
            rec["lang"] = lang
            rec["lang_prob"] = float(prob[0])
        else:
            rec["lang"] = None
            rec["lang_prob"] = 0.0
        # 卫生/退化
        rec["checks"] = text_checks(text)
        # PII
        rec["pii"] = {
            "email": len(EMAIL_RE.findall(text)),
            "phone": len(PHONE_RE.findall(text)),
            "idcard": len(IDCARD_RE.findall(text)),
        }
        # 去污染：与生产同为全文扫描，并独立复算逐 benchmark distinct-hit 阈值。
        sh, _ = doc_shingle_hashes(text, 8, cap=None)
        thresholds = effective_min_distinct_hits(decontam_cfg, f["source"])
        raw_hits, qualified_hits = decontam_hit_summary(sh, bidx, thresholds)
        rec["decontam_raw_distinct_hits"] = raw_hits
        rec["decontam_thresholds"] = thresholds
        rec["decontam_hits"] = qualified_hits
        rec["n_shingles"] = int(len(sh))
        if (si + 1) % 400 == 0:
            print(f"  checked {si+1}/{len(all_rows)} {time.monotonic()-t0:.0f}s", flush=True)
        samples.append(rec)

    # 分位：按域批量（排序后二分探测单调 → 网络存储顺序读）
    print("== percentiles (batched per domain) ==", flush=True)
    for dom, recs in defaultdict(list, {d: [r for r in samples if r["domain"] == d] for d in
                                        set(r["domain"] for r in samples)}).items():
        vals = sorted((r["hybrid_v2"], i) for i, r in enumerate(recs) if r.get("hybrid_v2") is not None)
        if not vals:
            continue
        arr = qtab[dom]
        v = np.array([x[0] for x in vals])
        p = np.searchsorted(arr, v, side="right") / len(arr)
        for (val, i), pct in zip(vals, p):
            recs[i]["pct_hybrid_v2"] = float(pct)
        print(f"  {dom}: {len(vals)} scored", flush=True)

    # ---- 聚合 ----
    print("== aggregate ==", flush=True)
    report: dict = {
        "audit": "published_mix_data_quality",
        "seed": SEED,
        "identity": {
            "mix_name": args.mix,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "members_sha256": sha256_file(MIX),
            "decontam_global_sha256": sha256_file(processed / "stage_decontam_global_summary.json"),
        },
        "mix": {"file": str(MIX.relative_to(ROOT)), **mix_meta},
        "sampling": sampling_provenance,
        "index": {
            "cache_ok": cache_ok,
            "s3_files": len(files),
            "lookup_mode": "sequential_preload" if args.preload_id_index else "mmap_binary_search",
        },
        "decontam_index": {
            "benchmarks": bidx.names,
            "n_unique_shingles": int(bidx.n),
            "policy_config": args.decontam_config.resolve().as_posix(),
            "policy_config_sha256": sha256_file(args.decontam_config.resolve()),
        },
        "per_domain": {},
    }
    manual: dict[str, list] = defaultdict(list)
    dom_samples: dict[str, list[dict]] = defaultdict(list)
    for rec in samples:
        dom_samples[rec["domain"]].append(rec)

    for dom, recs in sorted(dom_samples.items()):
        n = len(recs)
        got = [r for r in recs if r.get("text_found")]
        langs = Counter(r["lang"] for r in got if r.get("lang"))
        exp_lang, exp_thr = LANG_EXPECT.get(dom, (None, 0.0))
        exp_share = (langs.get(exp_lang, 0) / len(got)) if (exp_lang and got) else None

        def med(key):
            vals = sorted(r["checks"][key] for r in got if key in r.get("checks", {}))
            return vals[len(vals) // 2] if vals else None

        flags = {
            "fffd": sum(1 for r in got if r["checks"]["fffd_ratio"] > 1e-4),
            "ctrl": sum(1 for r in got if r["checks"]["ctrl_ratio"] > 1e-3),
            "html": sum(1 for r in got if r["checks"]["html_tag_ratio"] > 0.05 or r["checks"]["html_script_style"]),
            "url_dense": sum(1 for r in got if r["checks"]["url_density"] > 0.5),
            "char_repeat": sum(1 for r in got if r["checks"]["char_repeat"]),
            "line_dup": sum(1 for r in got if r["checks"]["line_dup_ratio"] > 0.55),
            "ngram_dup": sum(1 for r in got if r["checks"]["ngram_dup_ratio"] > 0.5 and r["checks"]["n_8grams"] >= 100),
        }
        pii = {
            "email": sum(1 for r in got if r["pii"]["email"] > 0),
            "phone": sum(1 for r in got if r["pii"]["phone"] > 0),
            "idcard": sum(1 for r in got if r["pii"]["idcard"] > 0),
        }
        for r in got:
            if r["pii"]["email"] or r["pii"]["phone"] or r["pii"]["idcard"]:
                manual["pii"].append(r["doc_id"])
            if r["decontam_hits"]:
                manual["decontam"].append({"doc_id": r["doc_id"], "hits": r["decontam_hits"]})
            elif r.get("decontam_raw_distinct_hits"):
                manual["decontam_below_threshold"].append({
                    "doc_id": r["doc_id"],
                    "source": r["source"],
                    "distinct_hits": r["decontam_raw_distinct_hits"],
                    "thresholds": {
                        name: r["decontam_thresholds"][name]
                        for name in r["decontam_raw_distinct_hits"]
                    },
                })
            if exp_lang and r.get("lang") and r["lang"] != exp_lang and r.get("lang_prob", 0) > 0.5:
                manual["lang_mismatch"].append({"doc_id": r["doc_id"], "lang": r["lang"], "domain": dom})
            if r["checks"]["char_repeat"] or (r["checks"]["ngram_dup_ratio"] > 0.5 and r["checks"]["n_8grams"] >= 100):
                manual["degenerate"].append(r["doc_id"])

        pcts = sorted(r["pct_hybrid_v2"] for r in got if r.get("pct_hybrid_v2") is not None)
        h2s = sorted(r["hybrid_v2"] for r in got if r.get("hybrid_v2") is not None)
        score_block = {
            "n_score_missing": n - len(h2s),
            "hybrid_v2_median": h2s[len(h2s) // 2] if h2s else None,
            "pct_median": pcts[len(pcts) // 2] if pcts else None,
            "pct_p10": pcts[len(pcts) // 10] if pcts else None,
            "pct_min": pcts[0] if pcts else None,
        }
        short_n = sum(1 for r in recs if r["n_tokens"] < 64)
        dom_rep = {
            "n_sampled": n,
            "n_text_found": len(got),
            "n_s4_aligned": sum(1 for r in got if r.get("s4_aligned")),
            "lang_top5": dict(langs.most_common(5)),
            "expected_lang": exp_lang,
            "expected_lang_share": exp_share,
            "hygiene_flags": flags,
            "hygiene_median": {
                "fffd_ratio": med("fffd_ratio"), "ctrl_ratio": med("ctrl_ratio"),
                "html_tag_ratio": med("html_tag_ratio"), "url_density": med("url_density"),
            },
            "degenerate_flags": {
                "char_repeat": flags["char_repeat"],
                "line_dup": flags["line_dup"],
                "ngram_dup": flags["ngram_dup"],
            },
            "ngram_dup_median": med("ngram_dup_ratio"),
            "line_dup_median": med("line_dup_ratio"),
            "short_docs_lt64": short_n,
            "short_docs_share": short_n / n,
            "pii_docs": pii,
            "decontam_docs_with_hits": sum(1 for r in got if r["decontam_hits"]),
            "decontam_total_hits": sum(len(r["decontam_hits"]) for r in got),
            "decontam_docs_with_raw_hits": sum(
                1 for r in got if r.get("decontam_raw_distinct_hits")
            ),
            "score": score_block,
        }
        report["per_domain"][dom] = dom_rep

    report["manual_review"] = {
        "pii_doc_ids": sorted(set(manual["pii"])),
        "decontam": manual["decontam"],
        "decontam_below_threshold": manual["decontam_below_threshold"],
        "lang_mismatch": manual["lang_mismatch"],
        "degenerate_doc_ids": sorted(set(manual["degenerate"])),
    }
    report["missing_text_doc_ids"] = missing_text
    report["elapsed_sec"] = round(time.monotonic() - t0, 1)

    machine_checks = {
        "all_sample_text_resolved": not missing_text
        and all(v["n_text_found"] == v["n_sampled"] for v in report["per_domain"].values()),
        "all_sample_scores_aligned": all(
            v["n_s4_aligned"] == v["n_sampled"]
            and v["score"]["n_score_missing"] == 0
            for v in report["per_domain"].values()
        ),
        "no_benchmark_overlap": all(
            v["decontam_docs_with_hits"] == 0 for v in report["per_domain"].values()
        ),
        "text_encoding_hygiene": all(
            v["hygiene_flags"]["fffd"] == 0
            and v["hygiene_flags"]["ctrl"] == 0
            and v["hygiene_flags"]["html"] == 0
            for v in report["per_domain"].values()
        ),
        "expected_language_share": all(
            v["expected_lang"] is None
            or float(v["expected_lang_share"] or 0.0)
            >= LANG_EXPECT[d][1]
            for d, v in report["per_domain"].items()
        ),
    }
    report["machine_checks"] = machine_checks
    report["status"] = "PASS" if all(machine_checks.values()) else "FAIL"

    # samples.tsv
    samples_path = OUT / "samples.tsv"
    samples_tmp = OUT / f".samples.tsv.tmp-{os.getpid()}"
    with samples_tmp.open("w", encoding="utf-8") as fh:
        fh.write("idx\tdoc_id\tdomain\tsource\tlang\tn_tokens\thybrid_v2\tpct_hybrid_v2\t"
                 "flag_fffd\tflag_ctrl\tflag_html\tflag_url\tflag_charrep\tflag_linedup\tflag_ngramdup\t"
                 "flag_short\tpii\tdecontam_hits\ttext_head300\n")
        for i, r in enumerate(samples):
            c = r.get("checks", {})
            pii = r.get("pii", {})
            pii_s = ";".join(k for k, v in pii.items() if v) or ""
            head = (r.get("text") or "")[:300].replace("\t", " ").replace("\r", " ").replace("\n", " ⏎ ").replace('"', "'")
            fh.write("\t".join(str(x) for x in [
                i, r["doc_id"], r["domain"], r.get("source", ""), r.get("lang", ""),
                r["n_tokens"],
                f"{r['hybrid_v2']:.6f}" if r.get("hybrid_v2") is not None else "missing",
                f"{r['pct_hybrid_v2']:.4f}" if r.get("pct_hybrid_v2") is not None else "",
                int(c.get("fffd_ratio", 0) > 1e-4), int(c.get("ctrl_ratio", 0) > 1e-3),
                int(c.get("html_tag_ratio", 0) > 0.05 or c.get("html_script_style", False)),
                int(c.get("url_density", 0) > 0.5), int(c.get("char_repeat", False)),
                int(c.get("line_dup_ratio", 0) > 0.55),
                int(c.get("ngram_dup_ratio", 0) > 0.5 and c.get("n_8grams", 0) >= 100),
                int(r["n_tokens"] < 64), pii_s, "|".join(r.get("decontam_hits", [])),
                head,
            ]) + "\n")
    os.replace(samples_tmp, samples_path)

    roster = {
        "schema_version": 1,
        "mix_name": args.mix,
        "seed": SEED,
        "members_sha256": report["identity"]["members_sha256"],
        "manifest_sha256": report["identity"]["manifest_sha256"],
        "sample_plan": mix_meta["sample_plan"],
        "samples": [
            {
                "doc_id": r["doc_id"],
                "domain": r["domain"],
                "n_tokens": int(r["n_tokens"]),
            }
            for r in samples
        ],
    }
    roster_path = OUT / "sample_roster.json"
    roster_tmp = OUT / f".sample_roster.json.tmp-{os.getpid()}"
    roster_tmp.write_text(
        json.dumps(roster, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(roster_tmp, roster_path)
    report["sampling"] = {
        **sampling_provenance,
        "roster": roster_path.resolve().as_posix(),
        "roster_sha256": sha256_file(roster_path),
        "samples_tsv": samples_path.resolve().as_posix(),
        "samples_tsv_sha256": sha256_file(samples_path),
        "n_samples": len(samples),
    }
    report_path = OUT / "report.json"
    report_tmp = OUT / f".report.json.tmp-{os.getpid()}"
    report_tmp.write_text(
        json.dumps(report, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(report_tmp, report_path)
    print(
        f"done status={report['status']} in {time.monotonic()-t0:.0f}s -> {OUT}",
        flush=True,
    )
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
