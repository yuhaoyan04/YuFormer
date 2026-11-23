#!/bin/bash
# E0-Q2 128k 流重建驱动（E-74）：物化到临时名 → 全量验证 → cp 转正 → 生成正式 meta。
# 背景：2026-10-13 物化因 materialize_tokens.py pass2a 写死 <u2（E-42 只改声明 dtype 漏改落盘），
# 128k token id≥65536 被静默回绕，整流报废；且 NFS rename EPERM 留下半成品。已修复脚本（E-74）。
# 幂等：正式产物已存在则直接退出。日志：data/rebuild_128k_20261014.log。
set -uo pipefail
REPO_ROOT="${YUFORMER_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO_ROOT"
LOG=data/rebuild_128k_20261014.log
exec >>"$LOG" 2>&1

FINAL_BIN=data/token_streams/e0q2_yf-128k-bpe.tokens.bin
FINAL_META=data/token_streams/e0q2_yf-128k-bpe.meta.json
TMP_NAME=e0q2_yf-128k-bpe.rebuild-tmp
TMP_BIN=data/token_streams/${TMP_NAME}.tokens.bin
TMP_META=data/token_streams/${TMP_NAME}.meta.json

echo "==== rebuild e0q2_yf-128k-bpe start $(date '+%F %H:%M:%S') ===="
echo "[backup] 已删除的损坏 meta 原内容（e0q2_yf-128k-bpe.meta.json, 2026-10-13 04:38, bytes 与 total_tokens*4 差 2 字节、内容实为 u2 回绕数据）:"
cat <<'META'
{
  "bytes": 19226647246,
  "dtype": "<u4",
  "edge_sha256": "53b0c1931d04d6037b5643d7b65d72dbaf95a3645d800d054a7380181f574123",
  "n_docs": 9495268,
  "name": "e0q2_yf-128k-bpe",
  "s3_root": "data/processed/s3",
  "seq_note": "裸文本 token 拼接（add_special_tokens=False，与 S5 计长同口径）",
  "sha256": "2604998e0a95ac8940c446dc58e78af20c4b562d4e105098c876d0afa68f65d2",
  "source": "s6:data/processed/mix_order/mix_m3_e1_100b_fast_stage1.parquet",
  "tokenizer": "tokenizer/tokenizers/yf-128k-bpe/tokenizer.json",
  "tokenizer_sha256": "4716da80ebd2affb16f8a278352fe0ef0118c6d54cf57a34fbcfb57c1de870ac",
  "total_tokens": 4806661811,
  "vocab_size": 131072
}
META

if [ -f "$FINAL_META" ] && [ -f "$FINAL_BIN" ]; then
    echo "final 产物已存在——幂等退出"
    exit 0
fi

echo "---- materialize ($TMP_NAME) $(date '+%F %H:%M:%S')"
PYTHONPATH=pretrain/src HF_HUB_OFFLINE=1 HOME=/root PATH=/usr/bin:/bin \
    /usr/bin/python3 pretrain/scripts/materialize_tokens.py \
    --order data/processed/mix_order/mix_m3_e1_100b_fast_stage1.parquet \
    --s3-root data/processed/s3 \
    --tokenizer tokenizer/tokenizers/yf-128k-bpe/tokenizer.json \
    --out-dir data/token_streams \
    --name "$TMP_NAME" --max-tokens 10000000000
rc=$?
echo "materialize rc=$rc $(date '+%F %H:%M:%S')"
if [ $rc -ne 0 ]; then echo "REBUILD_FAIL materialize rc=$rc"; exit $rc; fi

echo "---- verify $(date '+%F %H:%M:%S')"
PYTHONPATH=pretrain/src HF_HUB_OFFLINE=1 HOME=/root PATH=/usr/bin:/bin \
    /usr/bin/python3 - <<'PYEOF'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "pretrain/src")
from yuformer_pretrain.data import _edge_sha256      # noqa: E402
from yuformer_pretrain.tracking import file_sha256   # noqa: E402

VOCAB = 131072
tmp_meta = Path("data/token_streams/e0q2_yf-128k-bpe.rebuild-tmp.meta.json")
tmp_bin = Path("data/token_streams/e0q2_yf-128k-bpe.rebuild-tmp.tokens.bin")
m = json.loads(tmp_meta.read_text())
size = tmp_bin.stat().st_size
errs = []

def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        errs.append(msg)

check(size % 4 == 0, f"bytes%4==0 (bytes={size})")
check(size == m["bytes"], f"meta.bytes==file_size ({m['bytes']} vs {size})")
check(size // 4 == m["total_tokens"], f"total_tokens==bytes//4 ({m['total_tokens']} vs {size//4})")
check(m["dtype"] == "<u4", f'meta.dtype ({"<u4"} vs {m["dtype"]})')
check(m["vocab_size"] == VOCAB, f'meta.vocab_size ({m["vocab_size"]})')
check(m["n_docs"] == 9495268, f'meta.n_docs ({m["n_docs"]})')

rng = np.random.default_rng(20261014)
max_seen = -1
def scan(off: int, nbytes: int, tag: str) -> int:
    global max_seen
    with open(tmp_bin, "rb") as f:
        f.seek(off)
        a = np.frombuffer(f.read(nbytes), dtype="<u4")
    n_bad = int((a >= VOCAB).sum())
    max_seen = max(max_seen, int(a.max()))
    print(f"  sample {tag}: n={a.size} max_id={int(a.max())} n>=vocab={n_bad}", flush=True)
    return n_bad

bad = scan(0, 16 << 20, "head")
bad += scan(max(0, size - (16 << 20)), 16 << 20, "tail")
for i in range(64):
    off = int(rng.integers(0, max(1, size - (4 << 20)))) & ~3
    bad += scan(off, 4 << 20, f"rand{i:02d}")
check(bad == 0, f"全采样 u4 读 id<{VOCAB} (bad={bad}, max_seen={max_seen})")

from tokenizers import Tokenizer
tok = Tokenizer.from_file("tokenizer/tokenizers/yf-128k-bpe/tokenizer.json")
with open(tmp_bin, "rb") as f:
    head = np.frombuffer(f.read(1 << 20), dtype="<u4")[:4096]
txt = tok.decode(head.tolist())
check(len(txt.strip()) > 0, f"头部 4096 token decode 非空 ({len(txt)} chars)")

sha = file_sha256(tmp_bin)
edge = _edge_sha256(tmp_bin)
check(sha == m["sha256"], "sha256 全量重算一致")
check(edge == m["edge_sha256"], "edge_sha256 一致")
tsha = hashlib.sha256(Path("tokenizer/tokenizers/yf-128k-bpe/tokenizer.json").read_bytes()).hexdigest()
check(tsha == m["tokenizer_sha256"], "tokenizer_sha256 一致")

if errs:
    print(f"VERIFY_FAIL ({len(errs)} 项)")
    sys.exit(1)
print(f"VERIFY_OK total_tokens={size//4} bytes={size} sha256={sha}", flush=True)
PYEOF
rc=$?
echo "verify rc=$rc $(date '+%F %H:%M:%S')"
if [ $rc -ne 0 ]; then echo "REBUILD_FAIL verify rc=$rc"; exit $rc; fi

echo "---- promote $(date '+%F %H:%M:%S')"
cp "$TMP_BIN" "$FINAL_BIN"
PYTHONPATH=pretrain/src HOME=/root PATH=/usr/bin:/bin /usr/bin/python3 - <<'PYEOF'
import json
from pathlib import Path
m = json.loads(Path("data/token_streams/e0q2_yf-128k-bpe.rebuild-tmp.meta.json").read_text())
m["name"] = "e0q2_yf-128k-bpe"
Path("data/token_streams/e0q2_yf-128k-bpe.meta.json").write_text(
    json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("final meta written: name=e0q2_yf-128k-bpe")
PYEOF
rm -f "$TMP_BIN" "$TMP_META"
echo "REBUILD_OK $(date '+%F %H:%M:%S')"
