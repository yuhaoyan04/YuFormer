"""S5 单分片白盒压测：真实分片上量测吞吐与 ru_maxrss（E-21 口径），不走 driver 的全源覆盖硬门。

用法：PYTHONPATH=src python scripts/stress_s5.py --source dclm-baseline --relpath <rel> [--runs 2]
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

from yuformer_data.scale import length_bucket as lb


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--relpath", required=True)
    ap.add_argument("--domain", default=None)
    ap.add_argument("--out-root", default="../data/processed")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--keep-output", action="store_true", help="保留 s5 产物（默认测完删除，避免污染正式物化）")
    args = ap.parse_args(argv)
    out_root = Path(args.out_root).resolve()
    if args.domain:
        domain = args.domain
    else:
        from yuformer_data.scale.profiles import load_profiles

        domain = load_profiles(None)[args.source]["domain"]
    ctx = lb.load_context_lengths()
    results = []
    for run in range(args.runs):
        if run > 0:
            p, s = lb.s5_output_paths(out_root, args.source, args.relpath)
            for f in (p, s):
                if f.is_file():
                    f.unlink()
        t0 = time.time()
        summary = lb.process_shard_s5(out_root, args.source, args.relpath, domain, ctx)
        wall = time.time() - t0
        r = {
            "source": args.source,
            "relpath": args.relpath,
            "domain": domain,
            "run": run,
            "in_rows": summary.get("in_rows"),
            "out_rows": summary.get("out_rows"),
            "total_tokens": summary.get("total_tokens"),
            "decisions": summary.get("decisions"),
            "buckets": summary.get("buckets"),
            "errors": summary.get("errors"),
            "error_message": summary.get("error_message"),
            "elapsed_reported": summary.get("elapsed_sec"),
            "wall_sec": round(wall, 1),
            "rows_per_s": round((summary.get("in_rows") or 0) / max(wall, 1e-9), 1),
            "tokens_per_s": round((summary.get("total_tokens") or 0) / max(wall, 1e-9), 1),
            "ru_maxrss_gb": round(rss_gb(), 3),
        }
        results.append(r)
        print(json.dumps(r, ensure_ascii=False, sort_keys=True))
    if not args.keep_output:
        p, s = lb.s5_output_paths(out_root, args.source, args.relpath)
        for f in (p, s):
            if f.is_file():
                f.unlink()
        print("output cleaned (dry-run mode)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
