#!/bin/bash
echo "REFUSED: E0-Q2 was invalidated; configs moved to pretrain/configs/_holding/invalidated_e0q2" >&2
exit 4
# E0-Q2 自动点火：DATA_READY + GPU 窗口 + unigram 就绪 → 四臂并行（每臂 1 卡）
# 幂等：已起的臂不重启；某 tokenizer 缺失自动降级为 3 臂。
# E-73：仓库根由脚本位置推导（YUFORMER_ROOT 可覆盖）；director python 已死 → python3。
REPO_ROOT="${YUFORMER_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO_ROOT"
LOG=data/e0q2_autolaunch.log
while true; do
  if [ ! -f data/DATA_READY.json ]; then sleep 300; continue; fi
  if [ ! -f data/health/gpu_window_ready.json ]; then sleep 300; continue; fi
  echo "[$(date '+%F %H:%M:%S')] DATA + GPU 就绪——准备 E0-Q2 数据（多 tokenizer 子集）" >> $LOG
  /usr/bin/python3 experiments/e0_tokenizer/scripts/prepare_e0q2_data.py >> $LOG 2>&1
  gpu=0
  for tok in yf-32k-bpe yf-64k-bpe yf-128k-bpe yf-64k-unigram; do
    [ -f data/token_streams/e0q2_${tok}.meta.json ] || continue
    if pgrep -f "e0q2_${tok}" > /dev/null; then continue; fi
    echo "[$(date '+%F %H:%M:%S')] launch arm ${tok} on GPU ${gpu}" >> $LOG
    OUT=data/experiments/e0q2/${tok}
    mkdir -p $OUT
    YUFORMER_PRETRAIN_OUT=$OUT YUFORMER_TOKEN_STREAMS=data/token_streams \
      bash pretrain/scripts/launch.sh pretrain/configs/e0q2/e0q2_${tok}.env >> $LOG 2>&1 &
    gpu=$((gpu+1))
    [ $gpu -ge 4 ] && break
  done
  echo "[$(date '+%F %H:%M:%S')] E0-Q2 arms launched (${gpu})——退出点火循环" >> $LOG
  break
done
