"""s3 keep 池分层抽样质检：乱码/boilerplate/领域纯度/长度分布/决策与分数对照。

确定性分层抽样（blake2b seed）：每源取 1 个代表分片，行内按 hash 前 n 抽样。
只读 (doc_id, text, lang, s2_decision, s3_decision) 列，IO 轻量。
输出：JSON 报告（sort_keys）+ 人审样本摘录（每域 3 篇头部 400 字符）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

MOJIBAKE_RE = re.compile(r"Ã[\x80-\xbf©®°±²³µ¶·¸¹º¼½¾¿]|â€|ï¼|Ã©")
BOILER_RE = re.compile(
    r"cookie|javascript|sign in|log ?in|subscribe|newsletter|all rights reserved|"
    r"privacy policy|terms of (service|use)|click here|related (articles|posts)|"
    r"share (this|on)|follow us|advertisement|©",
    re.I,
)
NAV_RE = re.compile(r"(home|menu|search|contact|about us)\s*(\||·|»|>)", re.I)
URL_TOKEN_RE = re.compile(r"https?://|www\.|\.(com|net|org|html|php|aspx)\b", re.I)
SUSPICIOUS_CJK_MIX = re.compile(r"[\u4e00-\u9fff]")


def sample_rows(path: Path, n: int, seed: str):
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    rg = pf.read_row_group(0, columns=["doc_id", "text", "lang", "s2_decision", "s3_decision"]).to_pylist()
    if len(rg) <= n:
        return rg, total
    picked = []
    for i, r in enumerate(rg):
        h = hashlib.blake2b(f"{seed}|{path.name}|{i}".encode(), digest_size=8).digest()
        if int.from_bytes(h, "big") % max(1, len(rg) // n) == 0:
            picked.append(r)
        if len(picked) >= n:
            break
    return picked, total


def text_metrics(text: str) -> dict:
    n = len(text)
    words = text.split()
    nw = len(words)
    upper_ratio = sum(1 for c in text if c.isupper()) / max(n, 1)
    digit_ratio = sum(1 for c in text if c.isdigit()) / max(n, 1)
    ws = sum(1 for c in text if c.isspace())
    space_ratio = ws / max(n, 1)
    uniq_words = len({w.lower() for w in words})
    uniq_ratio = uniq_words / max(nw, 1)
    moji = len(MOJIBAKE_RE.findall(text))
    boiler = len(BOILER_RE.findall(text))
    nav = len(NAV_RE.findall(text))
    url = len(URL_TOKEN_RE.findall(text))
    fffd = text.count("\ufffd")
    avg_word = sum(len(w) for w in words) / max(nw, 1)
    return {
        "chars": n,
        "words": nw,
        "avg_word_len": round(avg_word, 2),
        "upper_ratio": round(upper_ratio, 4),
        "digit_ratio": round(digit_ratio, 4),
        "space_ratio": round(space_ratio, 4),
        "word_uniq_ratio": round(uniq_ratio, 4),
        "mojibake_hits": moji,
        "boilerplate_hits": boiler,
        "nav_hits": nav,
        "url_hits": url,
        "replacement_char": fffd,
    }


def pick_shard(source_dir: Path, seed: str) -> Path | None:
    files = sorted(source_dir.rglob("*.parquet"))
    files = [f for f in files if not f.name.endswith(".tmp") and ".summary" not in f.name]
    if not files:
        return None
    h = int.from_bytes(hashlib.blake2b(f"{seed}|{source_dir.name}".encode(), digest_size=8).digest(), "big")
    return files[h % len(files)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed-root", default="../data/processed")
    ap.add_argument("--stage-dir", default="s3")
    ap.add_argument("--per-source", type=int, default=400)
    ap.add_argument("--seed", default="quality-spotcheck-v1")
    ap.add_argument("--out", default=None)
    ap.add_argument("--show-samples", type=int, default=0)
    args = ap.parse_args(argv)
    root = Path(args.processed_root).resolve()
    stage = root / args.stage_dir
    if not stage.is_dir():
        print(f"missing {stage}", file=sys.stderr)
        return 2

    report: dict = {"stage_dir": args.stage_dir, "per_source": args.per_source, "seed": args.seed, "sources": {}}
    samples_for_human: dict[str, list[dict]] = {}
    for src_dir in sorted(p for p in stage.iterdir() if p.is_dir()):
        shard = pick_shard(src_dir, args.seed)
        if shard is None:
            continue
        try:
            rows, total = sample_rows(shard, args.per_source, args.seed)
        except Exception as e:
            report["sources"][src_dir.name] = {"error": f"{type(e).__name__}: {e}", "shard": str(shard)}
            continue
        mets = [text_metrics(r["text"] or "") for r in rows]
        n = max(len(rows), 1)
        agg = {
            "shard": str(shard.relative_to(root)),
            "shard_rows": total,
            "sampled": len(rows),
            "avg_chars": sum(m["chars"] for m in mets) // n,
            "p50_chars": sorted(m["chars"] for m in mets)[len(mets) // 2],
            "max_chars": max(m["chars"] for m in mets),
            "min_chars": min(m["chars"] for m in mets),
            "avg_words": sum(m["words"] for m in mets) // n,
            "avg_word_uniq_ratio": round(sum(m["word_uniq_ratio"] for m in mets) / n, 4),
            "docs_with_mojibake": sum(1 for m in mets if m["mojibake_hits"] > 2 or m["replacement_char"] > 2),
            "docs_with_boilerplate": sum(1 for m in mets if m["boilerplate_hits"] >= 3),
            "docs_with_nav": sum(1 for m in mets if m["nav_hits"] >= 2),
            "docs_url_heavy": sum(1 for m in mets if m["url_hits"] >= 8),
            "docs_very_short": sum(1 for m in mets if m["chars"] < 200),
            "docs_very_long": sum(1 for m in mets if m["chars"] > 500_000),
            "s2_drop_in_sample": sum(1 for r in rows if r.get("s2_decision") == "drop_exact_dup"),
            "s3_drop_in_sample": sum(1 for r in rows if r.get("s3_decision") == "drop_near_dup"),
            "lang_top": {},
        }
        langs: dict[str, int] = {}
        for r in rows:
            langs[r.get("lang") or "unk"] = langs.get(r.get("lang") or "unk", 0) + 1
        agg["lang_top"] = dict(sorted(langs.items(), key=lambda kv: -kv[1])[:5])
        report["sources"][src_dir.name] = agg

        keep_rows = [r for r in rows if r.get("s2_decision") != "drop_exact_dup" and r.get("s3_decision") != "drop_near_dup"]
        by_len = sorted(keep_rows, key=lambda r: len(r["text"] or ""))
        for bucket_i, r in enumerate([by_len[len(by_len) // 4], by_len[len(by_len) // 2], by_len[3 * len(by_len) // 4]]):
            samples_for_human.setdefault(src_dir.name, []).append(
                {
                    "quantile": [0.25, 0.5, 0.75][bucket_i],
                    "doc_id": r["doc_id"],
                    "chars": len(r["text"] or ""),
                    "lang": r.get("lang"),
                    "head": (r["text"] or "")[:400].replace("\n", "\\n"),
                }
            )

    flagged = {}
    for src, agg in report["sources"].items():
        if "error" in agg:
            flagged[src] = ["read_error"]
            continue
        fl = []
        n = max(agg["sampled"], 1)
        if agg["docs_with_mojibake"] / n > 0.01:
            fl.append(f"mojibake>{agg['docs_with_mojibake']}/{n}")
        if agg["docs_with_boilerplate"] / n > 0.15:
            fl.append(f"boilerplate>{agg['docs_with_boilerplate']}/{n}")
        if agg["docs_very_short"] / n > 0.10:
            fl.append(f"very_short>{agg['docs_very_short']}/{n}")
        if agg["avg_word_uniq_ratio"] < 0.25:
            fl.append(f"low_uniq_ratio={agg['avg_word_uniq_ratio']}")
        if fl:
            flagged[src] = fl
    report["flagged"] = flagged

    print(json.dumps({k: v for k, v in report.items() if k != "sources"}, indent=2, sort_keys=True))
    print("\n=== per-source metrics ===")
    for src, agg in report["sources"].items():
        if "error" in agg:
            print(f"{src:22s} ERROR {agg['error']}")
            continue
        print(
            f"{src:22s} rows={agg['shard_rows']:>9,} chars(p50)={agg['p50_chars']:>7,} uniq={agg['avg_word_uniq_ratio']:.2f} "
            f"moji={agg['docs_with_mojibake']:>3d} boiler={agg['docs_with_boilerplate']:>3d} short={agg['docs_very_short']:>3d} "
            f"s2drop={agg['s2_drop_in_sample']:>3d} s3drop={agg['s3_drop_in_sample']:>3d} lang={list(agg['lang_top'].items())[:2]}"
        )
    if args.show_samples:
        print("\n=== human review samples ===")
        print(json.dumps(samples_for_human, ensure_ascii=False, indent=1, sort_keys=True))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        report["human_samples"] = samples_for_human
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True, ensure_ascii=False)
        print(f"\nreport written: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
