"""goldrule 标注集构造：从 s3 keep 池分层抽样（域 × rule 分位 × 长度），LLM 标注用。

分布对齐原则（FineWeb-Edu 关键设计，修复 E-25 文体偏置）：
  - 样本来自待打分的目标池本身（不是精英源）
  - 每域按 rule_score 五分位分层（全质量谱覆盖——含"普通但正常"的中段）
  - 每条截断 max_chars（LLM 上下文与成本控制；头中尾三窗保留长文信息）
确定性：blake2b(seed|domain|layer|rank) 全程可复现。
输出：parquet（doc_id/domain/lang/text_head/chars/rule_score/source）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DOMAIN_SOURCES = {
    "web": ["fineweb", "fineweb-edu", "dclm-baseline"],
    "code": ["github-code", "starcoderdata"],
    "math": ["proof-pile-2", "automathtext"],
    "academic": ["pes2o", "arxiv-latex-corpus"],
    "books": ["dolma-books"],
    "encyclopedia": ["wikipedia-en", "wikipedia-zh"],
    "chinese": ["cci3-hq", "skypile-150b"],
    "qa": ["stack-exchange"],
    "multilingual": ["fineweb-2"],
}
DOMAIN_QUOTA = {
    "web": 1000, "code": 450, "math": 400, "academic": 350, "books": 250,
    "encyclopedia": 400, "chinese": 700, "qa": 300, "multilingual": 150,
}
N_LAYERS = 5


def _h(domain: str, layer: int, rank: int, salt: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(f"{salt}|{domain}|{layer}|{rank}".encode(), digest_size=8).digest(), "big"
    )


def sample_domain(out_root: Path, domain: str, sources: list[str], quota: int, seed: str, max_chars: int):
    pool = []
    for src in sources:
        src_dir = out_root / "s3" / src
        if not src_dir.is_dir():
            continue
        files = sorted(src_dir.rglob("*.parquet"))
        for f in (files[:: max(1, len(files) // 4)][:4] if len(files) > 4 else files):
            try:
                pf = pq.ParquetFile(f)
            except Exception:
                continue
            for batch in pf.iter_batches(batch_size=1024, columns=["doc_id", "text", "lang", "s2_decision"]):
                for r in batch.to_pylist():
                    if r["s2_decision"] != "keep" or not r["text"] or len(r["text"]) < 200:
                        continue
                    pool.append((r["doc_id"], r["text"], r["lang"], src))
                if len(pool) >= quota * 6:
                    break
            if len(pool) >= quota * 6:
                break
        if len(pool) >= quota * 6:
            break
    return pool


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed-root", default="../data/processed")
    ap.add_argument("--out", default="../data/models/goldrule/annotate_set.parquet")
    ap.add_argument("--seed", default="goldrule-v1")
    ap.add_argument("--max-chars", type=int, default=3000)
    args = ap.parse_args(argv)
    out_root = Path(args.processed_root).resolve()

    from yuformer_data.scale.rule_score import rule_score

    rows = []
    for domain, sources in DOMAIN_SOURCES.items():
        quota = DOMAIN_QUOTA[domain]
        pool = sample_domain(out_root, domain, sources, quota, args.seed, args.max_chars)
        if not pool:
            print(f"[skip] {domain}: no pool", file=sys.stderr)
            continue
        scores = np.array([rule_score(t[:8192]) for _d, t, _l, _s in pool])
        qs = np.percentile(scores, np.linspace(0, 100, N_LAYERS + 1))
        per_layer = quota // N_LAYERS
        for layer in range(N_LAYERS):
            lo, hi = qs[layer], qs[layer + 1]
            cands = [
                i for i, s in enumerate(scores)
                if (lo <= s <= hi if layer == N_LAYERS - 1 else lo <= s < hi)
            ]
            picked = 0
            rank = 0
            while picked < per_layer and rank < len(cands) * 4:
                i = cands[_h(domain, layer, rank, args.seed) % len(cands)]
                d, t, lang, src = pool[i]
                if not any(x[0] == d for x in rows[-per_layer:]):
                    rows.append((d, domain, lang, t[: args.max_chars], len(t), float(scores[i]), src))
                    picked += 1
                rank += 1
        print(f"[ok] {domain}: pool={len(pool)} picked={sum(1 for r in rows if r[1] == domain)}", file=sys.stderr)

    table = pa.table(
        {
            "doc_id": [r[0] for r in rows],
            "domain": [r[1] for r in rows],
            "lang": [r[2] for r in rows],
            "text": [r[3] for r in rows],
            "full_chars": [r[4] for r in rows],
            "rule_score": [r[5] for r in rows],
            "source": [r[6] for r in rows],
        }
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out, compression="zstd")
    print(json.dumps({"rows": len(rows), "out": str(out)}, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
