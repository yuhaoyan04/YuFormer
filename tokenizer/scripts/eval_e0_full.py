"""E0 全量指标采集：分域压缩率 / fertility / 编码吞吐，自训 3 档 vs 工业 baseline。

确定性抽样（blake2b）；每源限 --per-source 行、每行截断 --max-chars（长文截断对
bytes/token 影响可忽略，对吞吐口径统一）。输出 sort_keys JSON 供图表脚本消费。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer

SOURCE_DOMAIN = {
    "fineweb": "web", "fineweb-edu": "web", "dclm-baseline": "web", "fineweb-2": "web",
    "github-code": "code", "starcoderdata": "code",
    "proof-pile-2": "math", "automathtext": "math",
    "arxiv-latex-corpus": "academic", "pes2o": "academic",
    "dolma-books": "books",
    "wikipedia-en": "encyclopedia", "wikipedia-zh": "encyclopedia", "wikitext": "encyclopedia",
    "cci3-hq": "chinese", "skypile-150b": "chinese",
    "stack-exchange": "qa",
}
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def words(text: str) -> int:
    if _CJK.search(text):
        return sum(1 for c in text if _CJK.match(c)) + len([w for w in text.split() if not _CJK.search(w)])
    return len(text.split())


def sample_texts(path: Path, n: int, max_chars: int, seed: str) -> list[str]:
    """按行组顺序读取直到取满 n 条非空样本（确定性：同文件同参数同结果）。
    不做全文件扫描——tokenizer_corpus 源文件最大 950 万行，逐行扫描是 IO 浪费。"""
    _ = seed
    pf = pq.ParquetFile(path)
    texts: list[str] = []
    n_rg = pf.metadata.num_row_groups
    for rg in range(n_rg):
        tbl = pf.read_row_group(rg, columns=["text"])
        col = tbl.column(0).to_pylist()
        step = max(1, len(col) // max(n // 3, 1))
        for i, t in enumerate(col):
            if i % step != 0:
                continue
            s = (t or "")[:max_chars]
            if s.strip():
                texts.append(s)
                if len(texts) >= n:
                    return texts
        if rg >= 8 and len(texts) >= n // 2:
            break
    return texts


def eval_tokenizer(tok: Tokenizer, name: str, corpus_cache: dict[str, list[str]]):
    per_source_out = {}
    for src, texts in corpus_cache.items():
        if not texts:
            continue
        t0 = time.perf_counter()
        encs = tok.encode_batch(texts, add_special_tokens=False)
        dt = time.perf_counter() - t0
        n_tok = sum(len(e.ids) for e in encs)
        n_bytes = sum(len(t.encode("utf-8")) for t in texts)
        n_chars = sum(len(t) for t in texts)
        n_words = sum(words(t) for t in texts)
        per_source_out[src] = {
            "domain": SOURCE_DOMAIN.get(src, "?"),
            "samples": len(texts),
            "bytes": n_bytes,
            "chars": n_chars,
            "words": n_words,
            "tokens": n_tok,
            "bytes_per_token": round(n_bytes / n_tok, 4),
            "chars_per_token": round(n_chars / n_tok, 4),
            "tokens_per_word": round(n_tok / max(n_words, 1), 4),
            "throughput_mb_s": round(n_bytes / max(dt, 1e-9) / 1e6, 2),
        }
    domains: dict[str, dict] = {}
    for src, m in per_source_out.items():
        d = domains.setdefault(m["domain"], {"bytes": 0, "chars": 0, "words": 0, "tokens": 0, "samples": 0})
        for k in ("bytes", "chars", "words", "tokens", "samples"):
            d[k] += m[k]
    for dname, d in domains.items():
        d["bytes_per_token"] = round(d["bytes"] / d["tokens"], 4)
        d["chars_per_token"] = round(d["chars"] / d["tokens"], 4)
        d["tokens_per_word"] = round(d["tokens"] / max(d["words"], 1), 4)
    return {"tokenizer": name, "per_source": per_source_out, "per_domain": domains}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dir", default="../data/tokenizer_corpus")
    ap.add_argument("--ours", nargs="+", default=["yf-32k-bpe", "yf-64k-bpe", "yf-128k-bpe"])
    ap.add_argument("--ours-root", default="tokenizers")
    ap.add_argument("--baselines", nargs="*", default=["baselines/gpt2-50k.tokenizer.json", "baselines/qwen2.5-151k.tokenizer.json"])
    ap.add_argument("--per-source", type=int, default=20000)
    ap.add_argument("--max-chars", type=int, default=8192)
    ap.add_argument("--seed", default="e0-full-v1")
    ap.add_argument("--out", default="../experiments/e0_tokenizer/results/e0_full_metrics.json")
    args = ap.parse_args(argv)
    corpus_dir = Path(args.corpus_dir)
    if not corpus_dir.is_dir():
        print(f"corpus dir missing: {corpus_dir}", file=sys.stderr)
        return 2

    jobs = []
    for name in args.ours:
        p = Path(args.ours_root) / name / "tokenizer.json"
        jobs.append((name, p))
    for b in args.baselines:
        p = Path(b)
        jobs.append((p.stem.replace(".tokenizer", ""), p))
    corpus_cache: dict[str, list[str]] = {}
    for src_file in sorted(corpus_dir.glob("*.parquet")):
        src = src_file.stem
        corpus_cache[src] = sample_texts(src_file, args.per_source, args.max_chars, args.seed)
        print(f"[sampled] {src}: {len(corpus_cache[src])} texts", file=sys.stderr)
    results = []
    for name, path in jobs:
        if not path.is_file():
            print(f"skip {name}: {path} not found", file=sys.stderr)
            continue
        tok = Tokenizer.from_file(str(path))
        r = eval_tokenizer(tok, name, corpus_cache)
        vocab = tok.get_vocab_size()
        r["vocab_size"] = vocab
        tot_tok = sum(m["tokens"] for m in r["per_source"].values())
        tot_b = sum(m["bytes"] for m in r["per_source"].values())
        r["overall_bytes_per_token"] = round(tot_b / tot_tok, 4)
        results.append(r)
        print(f"[done] {name}: vocab={vocab:,} overall={r['overall_bytes_per_token']}", file=sys.stderr)

    out = {
        "seed": args.seed,
        "per_source": args.per_source,
        "max_chars": args.max_chars,
        "corpus_dir": str(corpus_dir),
        "tokenizers": results,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True, ensure_ascii=False)
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
