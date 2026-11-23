#!/bin/bash
echo "REFUSED: legacy scheduler depends on invalidated E0 streams and the compatibility trainer; use the future TorchTitan experiment launcher after D0/T0/I0" >&2
exit 4
# 实验调度器 v3（2026-10-14 修订，E-73）：v2 + ① 仓库根由脚本位置推导（YUFORMER_ROOT 可覆盖，
# 去硬编码绝对路径）；② launch_arm 接上 out_name（e0q2 输出目录剥 e0q2_ 前缀，与 env 内
# YUFORMER_PRETRAIN_OUT 一致——此前死参数导致 arm_done 找错 final.json、臂无限重跑）；
# ③ 启动前输出路径重复段防御（历史拼接 bug 曾把日志写进 e2/<arm>/e2/<arm>/）；
# ④ free_gpus 注释与代码对齐（限定 0-1）。其余契约不变（幂等跳过/单卡臂/GPU≥60GB）。
# ④b 2026-10-16：YUFORMER_SCHED_GROUPS 环境变量可限定入队组（如 "e0q2" 只跑 E0-Q2，
# 阶段门由接手执行顺序锁定——词表定案前不启动 E1/E2）；缺省 = 全队列。
# ④c 2026-10-16（点火实抓）：启动期显存分配滞后于 free_gpus 采样（首 iter 建模型/FSDP
# 期间 ~60-90s 卡上无显存），同一轮会把多张臂堆到同一张卡——本轮已认领 GPU 进 CLAIMED
# 互斥，跨轮则由 free_gpus 的 ≥60GB 内存判据自然拦截。
# 队列顺序：E0-Q2（4）→ E2（11）→ E1（19）——便宜的决策实验先出，E1 isoflop 殿后。
# GPU 分配（2026-10-14 修订）：0-6 跑实验舰队（PRETRAIN_EXPERIMENTS_PLAN 的 ~4 天
# 总账按 8 卡宽度估算）；GPU 7 预留给 MFU 冲刺/E1 门禁复测，不占队。
REPO_ROOT="${YUFORMER_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
LOG=data/experiments/scheduler.log
mkdir -p data/experiments

arm_done() {  # E-71b：$1=out_name（results/final.json 在 data/experiments/<out_name>/ 下）
  [ -f "data/experiments/$1/results/final.json" ]
}

arm_running() { pgrep -f "$1" > /dev/null; }

free_gpus() {  # 输出空闲 >=${YUFORMER_SCHED_FREE_MB:-60000}MB 的卡号（0-6；GPU 7 留给 MFU 冲刺，见文件头；
  # 阈值可经 YUFORMER_SCHED_FREE_MB 收紧——如 75000 实现"每卡一臂"独占，60000 允许两臂共存）
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' -v need="${YUFORMER_SCHED_FREE_MB:-60000}" '$1 < 7 && $3 - $2 >= need {print $1}'
}

launch_arm() {  # $1=group $2=arm $3=out_name $4=gpu
  local out=data/experiments/$3
  # E-73：路径防御——out 不得含 <group>/<arm>/<group>/ 重复段或空段（防嵌套目录回归）
  case "/$out/" in
    *"/$1/$2/$1/"*|*"//"*)
      echo "[$(date '+%F %H:%M:%S')] REFUSE $1/$2: 输出路径含重复段 ($out)" >> $LOG
      return 1 ;;
  esac
  mkdir -p "$out"
  echo "[$(date '+%F %H:%M:%S')] launch $1/$2 (out=$3) on GPU$4" >> $LOG
  CUDA_VISIBLE_DEVICES=$4 YUFORMER_PRETRAIN_OUT="$(pwd)/$out" \
    bash pretrain/scripts/launch.sh "pretrain/configs/$1/$2.env" >> "$out/launch.log" 2>&1 &
}

build_queue() {  # E0-Q2 → E2 → E1，arm = env 文件名去后缀；YUFORMER_SCHED_GROUPS 可限定
  QUEUE=""
  for g in ${YUFORMER_SCHED_GROUPS:-e0q2 e2 e1}; do
    for e in $(ls pretrain/configs/$g/*.env 2>/dev/null | sort); do
      case "$e" in *_lf.env) continue ;; esac  # E-71：排除 live-fire 克隆
      QUEUE="$QUEUE $g:$(basename "$e" .env)"
    done
  done
}

while true; do
  [ -f data/DATA_READY.json ] || { sleep 300; continue; }
  # 数据准备（E0-Q2 四 tokenizer 10B 流；E1/E2 复用 e0q2_yf-64k-bpe 或 stage1 全量流——E-47）
  if [ ! -f data/token_streams/e0q2_yf-64k-bpe.meta.json ] || [ ! -f data/token_streams/e0q2_yf-128k-bpe.meta.json ]; then
    python3 experiments/e0_tokenizer/scripts/prepare_e0q2_data.py >> $LOG 2>&1 || { sleep 600; continue; }
  fi

  build_queue
  PENDING=0
  CLAIMED=""   # ④c：本轮已认领 GPU（启动期显存未落盘，free_gpus 会重复选中同卡）
  for entry in $QUEUE; do
    group="${entry%%:*}"; arm="${entry#*:}"
    if [ "$group" = "e0q2" ]; then
      outname="e0q2/${arm#e0q2_}"
    else
      outname="$group/$arm"
    fi
    arm_done "$outname" && continue
    arm_running "$arm" && { PENDING=$((PENDING+1)); continue; }
    gpu=""
    for g in $(free_gpus); do
      case " $CLAIMED " in
        *" $g "*) continue ;;
      esac
      gpu=$g; break
    done
    [ -z "$gpu" ] && { PENDING=$((PENDING+1)); continue; }
    launch_arm "$group" "$arm" "$outname" "$gpu" && CLAIMED="$CLAIMED $gpu"
    PENDING=$((PENDING+1))
    sleep 30
  done

  if [ "$PENDING" -eq 0 ]; then
    echo "[$(date '+%F %H:%M:%S')] ALL EXPERIMENTS DONE" >> $LOG
    echo '{"ts": "'$(date -Iseconds)'", "status": "all_experiments_done"}' > data/EXPERIMENTS_DONE.json
    break
  fi
  sleep 600
done
