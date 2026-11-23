#!/usr/bin/env bash
# 内存清道夫：每 8 分钟 drop caches，防 cgroup OOM（page cache 计入容器限额）
while true; do
  sleep 300
  sync
  echo 3 > /proc/sys/vm/drop_caches 2>/dev/null && echo "[janitor] $(date '+%F %T') caches dropped" >> /opt/huawei/explorer-env/dataset/yyh_arcagi/YUFormer/YuFormer/data/janitor.log
done
