# MFU 短程性能探针（兼容后端，非正式训练配置）

用途：在单卡上对兼容后端做 200-step、157,286,400-token 的有界性能诊断。四份配置都显式标记 `experiment_status: probe_only` 与 `backend: pytorch-fsdp1-compat`；它们不能作为正式训练预算、收敛结论或 TorchTitan 性能结论。

## 用法（只在预留卡上跑，例：GPU 7）

```bash
cd pretrain && CUDA_VISIBLE_DEVICES=7 bash scripts/launch.sh \
  ../experiments/_common/mfu_gate/mfu_gate_ws1.env
```

- 配置本身固定为 200 steps（~50 min @ mbs12/accum16）；稳态分析直接读
  `<out>/results/metrics.jsonl`（每 log 点含 step_time + gpu_util_all/gpu_mem_mb_all 邻居基线）。
- 数据口径：v1 声明的 mix_m2（full S6）流物化前，用 mix_m3_e1_100b_fast_stage1（24.9B tokens，
  seq 4096 / vocab 65536 与 pretrain-0.5b 匹配）；MFU 与数据内容无关，只与 shape/dtype 路径有关。
- MFU = flops_per_token × tokens_per_step / mean_step_time / 989e12（H800 BF16 峰值，trainer 同口径）。

## 历史测量

| 日期 | 配置 | 结果 | 记录 |
|---|---|---|---|
| 2026-10-14 | v1 落地版单卡等价（adamw+mbs12+compile+sel-mlp） | 见 IGNITION §八 | EXPERIMENT_IGNITION_PLAN.md |

## 短窗探针（非门禁级，20 iters，决策支持用）

门禁复测若不达 18%，用 `mfu_gate_p1_probefull/p2_swaoff/p3_flash.{yaml,env}` 做
单变量 20-iter 归因（均基于 mfu_gate_ws1，一次只改一处；结果记录在 IGNITION §八）：
p1=preset 换 probe_full（全注意力无门控，复现 §7.1 口径）；p2=preset 换 e2-swaoff
（全注意力，§6.2 裁定路线）；p3=attn_impl 换 flash（E-52，SWA 层 window 内核）。
注意 flops_per_token：probe_full/e2-swaoff=3.8760G，pretrain-0.5b=4.0458G
（门控层参数差，analyze.py 需按 preset 传值）。
