# 实验点火规划（Phase B 启动前检查单 + MFU 实证，2026-10-10 夜）

> **SUPERSEDED / DO NOT EXECUTE（2026-10-18）**：本文保留为历史事故与性能证据，
> 其中 `DATA_READY`、E0 token streams、FSDP1/Muon/constant-LR 配置和旧 scheduler 均已失效。
> 当前点火条件是 D0/T0/I0；formal E1/E2 由 `gen_e1e2_arms.py` 生成并要求 TorchTitan、
> PASS data release、EOD 与随机顺序流。执行规范见 `report/PROJECT_AUDIT_20261018.md` 和
> `experiments/e1_scaling/README.md`。

> **预算口径更新（2026-10-20）**：本文下方的 1.5T / 45 天估算也是历史诊断，已被 1T
> final-tokenizer WSD 合同（50B warmup + 800B stable + 150B anneal）取代。当前运行规范见
> `report/PRETRAIN_STACK_AUDIT_20261020.md`、`pretrain/configs/pretrain_base.yaml` 和
> `experiments/e4_curriculum/configs/e4_1t_wsd_context.yaml`。

> **E-75（2026-10-21）**：正式 100B final-tokenizer stream 已取代本文历史 e0q2/stage1
> 容量路径；E1 增补固定约 300M 的五点 `L(D)` 截面，矩阵现为 24 臂。本文下方 E-74 的
> 19 臂容量表仅保留事故记录，不再是可执行合同。

> 前置：DATA_READY（S6 fast track 全链完成信号）。本文是点火前的最终核对与实证记录。
> 修订：E-47（点火准备批）、E-73（e0q2 容量核定）、E-74（E1 臂容量核定）——所有修复已落代码并通过测试/复演。

## 一、已就绪清单（点火前全部完成 ✅）

| 项 | 状态 | 证据 |
|---|---|---|
| 训练栈冒烟（torch 2.2.0a0 + FSDP no_shard + Muon + 断点续训） | ✅ | smoke_train：muon_loss↓、resume 逐位一致、marker 拒错 iter |
| 33 个臂 env 的 YUFORMER_PYTHON | ✅ 全修 | director env 已死 → /usr/bin/python3（E-47） |
| 大预算臂数据指向 | ✅ 7 臂改指 stage1 全量流 | c0_50m/c1_*/c2_*（预算 11B-76B > e0q2 流 10B 会 fail-fast） |
| e0q2 iters/budget 容量核定（E-73） | ✅ 四臂按流容量 min 修正 | 32k=38146 / 64k=38137 / 128k=18331 / unigram=37199；budget=iters×262144 与容量对平（身份守卫+容量 fail-fast 均保留） |
| E1 19 臂容量核定（E-74） | ✅ 8 超限臂修复 + 11 臂复核 | 5 ln 钳 38137/9,997,385,728；3 iso（c1_50m/c2_100m/c2_200m）改 stage1+stage2 双流；训练栈自身校验路径复演 19/19 PASS（见第八节） |
| e2-l48-h848 hidden 848→832 | ✅ | 13 头 × 64 head_dim（848/13 不整除） |
| torch 版本串解析 + SWA padding NaN | ✅ | attention.py 正则解析 + 全 False 行补对角（24/24 测试） |
| eval_every/save_every=0 除零 | ✅ 加守卫 | 探针配置暴露，臂配置=500 不受影响 |
| 调度器 v2 | ✅ 重写 | python3 + 队列从 configs 动态生成（33 臂自动入队）+ arm_done 修复 |
| tokenize 关卡 B3/B4 | ✅ | 行组局部索引（E-42）+ dtype 自适应 u2/u4 + loader 按 meta 解码 |
| quick_eval / plotting 基建 | ✅ | 12+13 项自测全过；benchmark 数据待下载（见二.4） |

## 二、MFU 实证（0.5B / 4K / 单卡 H800，2026-10-10 夜探针）

| 配置 | step 时间 | TFLOPS | MFU |
|---|---|---|---|
| SWA mask + Muon, mbs4 | 1.80s | 73.8 | **7.5%** |
| 全注意力(is_causal) + Muon, mbs4 | 1.24s | 102.4 | 10.3% |
| 全注意力 + AdamW, mbs4 | 1.02s | 124.2 | **12.6%** |
| mbs8/16+ | OOM | — | fp32 logits 每 mbs 1.07GB（chunked CE 未实现） |

**关键结论：当前栈单卡 MFU 现实值 10-13%，不是计划假设的 30-40%。** 分解：Muon 比 AdamW
贵 ~2.3 点；SWA mask 路径比 is_causal 贵 ~2.8 点；mbs 上不去的根因是 fp32 logits（CE 前
向 1×mbs GB + 反向 1×mbs GB）。注意 GPU 为共享卡（邻居恒定占 12.9GB，可用 ~66GB）。

**对计划的影响（按 26K tokens/s/卡 @7.5% 算）**：
- E1 实验总预算 ~100B+ tokens → 100B/26K = 44 卡·天 ≫ 原计划 ~4 卡·天。**E1 按当前 MFU 不可行**。
- 正式 pretrain 1.5T：4 卡 = 167 天；8 卡 = 83 天——都远超 45 天硬顶。

## 三、MFU 提升路线（Plan C 冲刺，2026-10-11 起 3-4 天，8 卡独占）

**已完成（首夜战果，E-48/E-49）**：
| 项 | 结果 |
|---|---|
| 分块融合 CE（E-48，默认开 ce_chunk=8192） | 验收全过（对拍<1e-4/smoke 回归过）；解除 logits 对 mbs 的锁死 |
| selective recompute（E-49，config `selective_recompute: mlp`） | 省 ~7GB 激活（vs full-ckpt 的 30% FLOPs 税） |
| expandable_segments（launch.sh 默认注入） | 回收 5.7GB 碎片，共享卡上 mbs8 从 OOM 变可跑 |
| **mbs8 组合（全注意力+sel-mlp+chunked CE）实测** | **MFU 11.5%**（臂默认配置 7.5% → +53%） |

**冲刺剩余项（用户裁定顺序）**：① SWA 感知 flash 路径（4K 下等价于全注意力 is_causal，
已实证 +37%；正式版考虑 pretrain 预设 attention_every=1，SWA 留给 midtrain 256K 时代）→
② bf16 residual/norm 卫生 → ③ torch.compile（最后；**数值口径改"同 build 内续训一致"**）。
增量验收线：≥10%（已过 11.5%）→ ≥14% → **≥18% 即停**（45 天硬顶只需 18%，不追求 35%）。

**⚠️ 测量污染声明**：GPU 为共享卡（邻居租户 8 进程 ×13GB、6 卡 100% util），MFU 门禁测量
须选低负载卡 + 记录基线 util；最终 pretrain 的可行性口径 = 本机有效吞吐（含共居），
机器现实如此，18% 门禁按有效值考核。

## 三.五、Plan C 排期与门禁（2026-10-10 用户裁定）

1. **首日 live-fire**：E2 单臂 2-3B tokens capped，真调度器/真数据/真 marker（3-4 卡时）。
   ⚠️ 前置：需 DATA_READY（e0q2 数据流依赖 mix_order）。live-fire 用独立 out 目录
   （data/experiments/livefire/），避免污染正式臂的 final.json 幂等标记。
2. **冲刺期**（1-4 天）：8 卡给冲刺独占；逐项验收（上表）。
3. **实验期**（4-5 天起）：全量 E0Q2+E2（~1.4 天 @18% 口径）→ E1 19 臂不缩
   （MFU 若落 13-18% 之间，带实测重算预算后再裁定：延期 / pretrain 缩 1.0-1.2T / 加卡）。
4. **E1 门禁**：冲刺验收达标才启动，不缩臂。
5. **pretrain 点火门**：≥18% 必须是 **30 分钟真实训练的持续实测**，非 microbenchmark。

## 四、点火 Runbook（DATA_READY 后）

1. `bash experiments/_common/experiment_scheduler.sh`（等 DATA_READY 门自动放行）
   - 内部先跑 prepare_e0q2_data.py（4 tokenizer × 10B 流物化，~2-4h CPU）
   - 队列：E0-Q2 4 臂（单卡并行×4）→ E2 10 臂（轮转）→ E1 19 臂（轮转）
   - 每臂 out：data/experiments/<group>/<arm>/（metrics.jsonl + manifest.json + results/final.json；e0q2 臂剥 e0q2_ 前缀 = env 的 YUFORMER_PRETRAIN_OUT，E-73 起 arm_done/launch_arm 统一按此判定）
2. 臂完成判定：results/final.json；调度器幂等，可安全重跑。
3. 全部完成后：plotting.py --arms-dir data/experiments --out experiments/e1_scaling/results
   → fit_result.json + E1-3/E1-4 图 + (N*,D*) 外推；quick_eval 对关键 ckpt 跑 GSM8K/MMLU 子集。
4. 决策会输入：E0-Q2 词表定案 / E2 架构定案 / E1 (N*,D*) + MFU 实绩修正的 45 天可行性。

## 四、点火 Runbook（DATA_READY 后自动/手动步骤）

1. `bash experiments/_common/experiment_scheduler.sh`（等 DATA_READY 门自动放行）
   - 内部先跑 prepare_e0q2_data.py（4 tokenizer × 10B 流物化，~2-4h CPU）
   - 队列：E0-Q2 4 臂（单卡并行×4）→ E2 10 臂（轮转）→ E1 19 臂（轮转）
   - 每臂 out：data/experiments/<group>/<arm>/（metrics.jsonl + manifest.json + results/final.json；e0q2 臂剥 e0q2_ 前缀 = env 的 YUFORMER_PRETRAIN_OUT，E-73 起 arm_done/launch_arm 统一按此判定）
2. 臂完成判定：results/final.json；调度器幂等，可安全重跑。
3. 全部完成后：plotting.py --arms-dir data/experiments --out experiments/e1_scaling/results
   → fit_result.json + E1-3/E1-4 图 + (N*,D*) 外推；quick_eval 对关键 ckpt 跑 GSM8K/MMLU 子集。
4. 决策会输入：E0-Q2 词表定案 / E2 架构定案 / E1 (N*,D*) + MFU 实绩修正的 45 天可行性。

## 五、待办（非阻塞）

- quick_eval benchmark 数据：HF 直连超时（curl 28），走 pip 内部镜像源或 hf-mirror 端点重试；
  落到 data/eval/{gsm8k,mmlu,humaneval}.jsonl（schema 见 _common/README.md）。
- 8 卡占用观察：邻居租户 12.9GB/卡正在向多卡扩散，调度器 free_gpus 按 ≥60GB 过滤已适配。

## 六、用户裁定补丁（2026-10-11 凌晨）

### 6.1 MFU 门禁测量协议（修订"选低负载卡"——E-50）
- 门禁 = **持续有效吞吐（tokens/day）实测**，测量窗口必须**覆盖邻居高负载时段**
  （≥24h 或含已知高峰）；低负载窗口测出的是"无争用上限"，不能做 45 天承诺。
- 每次 MFU 测量记录邻居 util 基线（已落实：trainer 每 log 点把全卡 util/显存写入
  metrics.jsonl 的 `gpu_util_all`/`gpu_mem_mb_all` 字段）——事后拟合争用系数，
  pretrain 排期按"峰值邻居负载下仍达标"承诺，不按均值。
- CPU 邻居（nzq 8×sokoban 各占 ~1 核）：176 核上无感，但 ndv2/S6 物化/dataloader
  叠加时留意 IO 互踩。

### 6.2 pretrain 全注意力 + SWA 留给 midtrain（附条件）
- ✅ E2 补 SWA on/off 对照臂：**e2_swaoff**（attention_every=1 全注意力，其余同
  pretrain-0.5b；端口 29631）——SWA 臂（l16/nogate 等）保留，"混合注意力"消融数据齐。
- ⚠️ 风险在案：pretrain 全注意力 → midtrain 引入 SWA = 架构切换（注意力熵分布突变），
  无现成先例（ZGCM 从头训 SWA）。**fallback**：pretrain+midtrain 都全注意力——
  0.5B@64K 全注意力注意力开销仅 ~9% FLOPs（256K 才是 ~36%），经济可接受；
  SWA 降级为纯研究模块。midtrain 最终选型 = E2 SWA 臂数据 + 一次廉价 midtrain-pilot。

### 6.3 full 版 S6 的 s5 清库规则（防静默沿用旧 join）
fast track 的 S6 用的是 E-17 豁免口径的 s5 产物。**ndv2_refresh 完成 → full 版 S6
点火前**：必须清 web/code/math 三域的 s5 产物并重跑 s5（math 域当时静默跳过、门③
管不到它）；不清则 full 版静默沿用旧 s5 数据。此规则已入 DECISION_RECORD_20261010
已知风险清单（R-2 家族），执行时按指纹清单核对。

### 6.4 flash 冲刺验收顺序（防数值回归误读为性能数据）
flash 路径定型后：**先复跑 smoke_train 断点一致性 → 再测 MFU**。顺序不可反。

## 七、MFU 冲刺战果（E-51/E-52，2026-10-13，共享卡实测）

**验收线 ≥18%：达标。** 30 分钟持续实测（非 30-iter 快照）MFU **21.9%**
（mbs6+compile，1500 iters 无中断，邻居共居 24-38GB 的有效口径）；
短窗探针峰值 **23.9%**（mbs12+compile，干净 13GB 邻居口径）。

### 7.1 演进表（0.5B/4K 全注意力探针 probe_full，单卡 H800，E-50 口径）

| 配置 | mbs | step(s) | TFLOPS | MFU | 判定 |
|---|---|---|---|---|---|
| eager 基线（复跑历史 11.5% 锚点） | 8 | 2.367 | 107 | 10.8% | 基线 |
| +torch.compile（default，FSDP 前编底层 module） | 8 | 1.297 | 196 | **19.8%** | +9.0 点，主杠杆 |
| 同上 | 12 | 1.614 | 236 | **23.9%** | 峰值（探针） |
| 同上 | 16 | — | — | OOM | >65GB 分配 |
| compile + bf16_hygiene | 12 | 1.826 | 209 | 21.1% | **拒绝**（变慢且数值有风险） |
| compile + selective=none | 8 | 1.518 | 167 | 16.9% | sel-mlp(+2.9 点) 保留 |
| compile + Muon（NS5） | 8 / 12 | 1.554 / 2.447 | 163 / 156 | 16.5% / 15.7% | **AdamW 胜**（NS5 在 0.5B 贵 3-8 点） |
| compile + max-autotune | 8 | — | — | crash | cudagraphs×FSDP 状态机冲突，弃 |
| compile + activation_checkpointing（全层） | 16 | 4.318 | 118 | 11.9% | **拒绝**（重算税 ~50%，且暴露包装顺序 bug，已修） |
| **compile  sustained 验证** | **6** | **0.880** | **217** | **21.9%** | **30min/1500iters，验收达标** |

### 7.2 冲刺项裁定
1. **SDPA 后端审计**：torch 2.2 下全注意力无 mask 路径已走 FLASH（pytorch_flash
   kernel）；SWA bool mask 自动落到 **mem_efficient**（cutlass sm80，非 math，不物化
   S²）——两条路都没有 math 灾难，无需重表达 mask。
2. **bf16 卫生**：实测负收益（-2.8 点），开关保留默认关，不做。
3. **torch.compile**：+9~13 点，numrics 低位变化（iter31 loss 差 1e-3，iter40 一致）。
   **resume 一致性**：40 截断续训 vs 全新跑，loss 差异 max 7.3e-3，落在两次全新跑
   run-to-run 差异（max 1.2e-2，flash bwd 原子加天然非逐位）之内——"同 build 内续训
   一致"成立；跨 build/跨开关的 checkpoint 不保证可载（state_dict 带 _orig_mod 前缀，
   fail-fast 优于静默错位）。
4. **flash-attn 包（E-52）**：环境预装 2.0.4（无 window/GQA）；PyPI 预编译轮与
   torch 2.2 ABI 不兼容（undefined symbol），**源码构建 2.6.3 成功**（nvcc 12.3/sm90，
   TORCH_CUDA_ARCH_LIST=9.0）。director env 已死后于 2026-10-14 在系统 python 重建：
   `pip3 install flash-attn==2.6.3 --no-build-isolation`（MAX_JOBS=48，~27 min）
   再次成功；同时给 attention 加了版本能力 guard（window_size 需 >=2.3.0，上游
   2.3.0 引入，考据自各 tag 源码），不满足在 ModelConfig init 期 fail-fast，
   2.0.4 被拦截不再到 forward 才 TypeError。新增 `attn_impl: sdpa|flash`（默认 sdpa）：flash 路径
   GQA 原生免 repeat_interleave、SWA 走 window_size 免 bool mask、免 S² 显存；
   单测 parity 通过（SWA/full 两形态 vs SDPA 参考）。**注意**：torch 2.2 dynamo 无法
   穿透 flash_attn 的 autograd.Function（需 `torch._dynamo.disable` 显式 graph break，
   已处理）；window_size=(-1,-1) 才是"无窗"（(0,0) 是只 attend 自己，已踩坑修正）。
   **训练中实测：未完成**——2026-10-13 夜共享卡 5/6/7 被邻居长任务占满（50-67GB），
   仅有的窗口全部让位给 sustained 验收与主阶梯；flash_mbs4 训练验证曾因邻居 surge
   OOM 中断。parity 单测 + SDPA 后端审计（SWA 现走 mem_efficient 非 math）支撑
   "无回归"结论，mbs 解放收益（probe 免 repeat_interleave ~8.6GB saved tensors，
   SWA 层 mem_efficient→flash window）留待点火前 20-iter 复测（成本 ~10 min）。
5. **Muon vs AdamW**：0.5B 档 AdamW 完胜（MFU 口径）；Muon 作为研究臂保留。

### 7.3 点火配置建议（pretrain_v1.yaml diff）
```yaml
train:
  compile: true          # E-51 主杠杆
  compile_mode: default  # max-autotune 与 FSDP 冲突；no-cudagraphs 变体未复测
  selective_recompute: mlp
  micro_batch_size: 12   # 单卡 ~62GB 峰值（含碎片）；4 卡 no_shard 每卡同构
  # attn_impl: flash     # E-52：待 7.4 阶梯实测确认后开启（SWA 层收益最大）
optim:
  optimizer: adamw
```
前置：4 卡 no_shard world_size=4 的 compile smoke（本次只有 world_size=1 实证）；
compile 首 iter 编译 ~2-4 min 一次性成本。

### 7.4 mbs × 峰值显存 × MFU 阶梯（probe_full，compile+sel-mlp+adamw）
| mbs | 峰值显存(整机,含13GB邻居) | MFU | 备注 |
|---|---|---|---|
| 4 | ~26GB | 14.7-21.4% | 区间=邻居负载差 |
| 6 | ~39GB | 21.9%（30min sustained） | 验收档 |
| 8 | ~64GB | 19.8% | — |
| 12 | ~62GB | 23.9% | 峰值 |
| 16 | >65GB OOM | — | flash 路径理论上可解（免 repeat_interleave ~8.6GB saved tensors），**训练实测未完成**（共享卡无窗口） |
| 24/32 | 仅 full-ckpt 下 ~41GB | ~12% | 重算税不可接受 |

（显存口径说明：expandable_segments 的 reserved 偏高，表值为整机峰值减 13GB 常量
邻居的估值；fit/OOM 事实为硬数据。）

## 八、E-74：E1 19 臂数据容量核定（2026-10-14）

**问题**。调度器 e1 组 19 臂中，8 臂的数据预算折算样本需求超过其 token 流容量，启动即被
守卫 fail-fast。守卫链（训练代码自身）：`config.validate()` 身份守卫
（`budget_tokens // tokens_per_step == expected_iters`，config.py:141）→
`TokenStream.open`（`samples = total_tokens // (seq_len+1)`，data.py:64）→
`DataCursor.require(derived_iters × mbs × accum × world ≤ Σsamples × epochs)`
（trainer.py:187）。注：E-73 任务清点口径为 9 臂；逐臂复算实际超限 **8 臂**
（5 ln + 3 iso），其余 11 臂余量均为正。

**裁决依据**：
- `docs/research/实验设计.md`（09-19 锁定）**E1-Q1**：L(N) 固定 **D=10B** × N∈{50M,…,800M}
  ——D 轴不可动；钳制偏差 ≤0.03% 视为不动。
- 同文档 **E1-Q3**：isoflop 三条等算力线 **C'∈{4.8e18, 1.4e19, 4.8e19}**——算力点
  C' 不可动，故 iso 臂 budget_tokens/expected_iters 一律不动。
- `experiments/PRETRAIN_EXPERIMENTS_PLAN.md`：E1/E2 共用 **100B mix** 物化集
  （= `mix_m3_e1_100b_fast_stage1` 24.86B + `mix_m3_e1_100b_fast_stage2` 全局打乱 74.58B）；
  E-47 先例（预算 >10B 的 iso 臂从 e0q2 流改指 stage1）；E-73 先例
  （e0q2_yf-64k-bpe 臂按流容量钳 38137 iters、budget=iters×262144）。
- 双流拼接是 data.py 现成机制（`concat_streams` 跨 stage LUT，全局样本序号连续，
  断点续训语义不变）。

**处置**：
1. **5 个 ln 臂**（e0q2_yf-64k-bpe 流，容量 2,440,805 样本）：需求 2,441,344
   （38146×64）超 539 样本 → 按 E-73 同型钳制 `expected_iters=38137`、
   `budget_tokens=9,997,385,728`（=38137×262144；D 10B→9.997B，−0.026%），
   钳后余量恰 37 样本。
2. **3 个 iso 臂**（c1_50m 44.0B / c2_100m 76.2B / c2_200m 39.4B）：需求
   10.75M / 18.60M / 9.62M 样本超 stage1 容量 6,067,853 → `data.stages/meta` 改
   stage1+stage2 双流（LUT 容量 24,271,634），budget/expected_iters 不动。设计意图吻合：
   E1 共用 100B 池；且单流无解——stage2 单流容量 18,203,781 样本 < c2_100m 需求
   18,601,152，双流是不改算力点的唯一解法。
3. **其余 11 臂**：复核余量均为正（最小 c2_800m：+44,389），不动作。

**逐臂核算表**（需求 = expected_iters × micro_batch × accum；容量 = Σ(total_tokens//4097) × epochs）：

| 臂 | 预算 tokens | iters | 需求样本 | 指向流（容量） | 余量 | 处置 |
|---|---:|---:|---:|---|---:|---|
| e1_iso_c0_50m | 15,094,339,622 | 57,580 | 3,685,120 | stage1（6,067,853） | +2,382,733 | 不动 |
| e1_iso_c0_100m | 7,619,047,619 | 29,064 | 1,860,096 | e0q2_yf-64k-bpe（2,440,805） | +580,709 | 不动 |
| e1_iso_c0_200m | 3,940,886,699 | 15,033 | 962,112 | e0q2_yf-64k-bpe | +1,478,693 | 不动 |
| e1_iso_c0_400m | 1,946,472,019 | 7,425 | 475,200 | e0q2_yf-64k-bpe | +1,965,605 | 不动 |
| e1_iso_c0_800m | 981,595,092 | 3,744 | 239,616 | e0q2_yf-64k-bpe | +2,201,189 | 不动 |
| e1_iso_c1_50m | 44,025,157,232 | 167,942 | 10,748,288 | stage1+stage2（24,271,634） | +13,523,346 | **E-74 双流** |
| e1_iso_c1_100m | 22,222,222,222 | 84,771 | 5,425,344 | stage1（6,067,853） | +642,509 | 不动 |
| e1_iso_c1_200m | 11,494,252,873 | 43,847 | 2,806,208 | stage1 | +3,261,645 | 不动 |
| e1_iso_c1_400m | 5,677,210,056 | 21,656 | 1,385,984 | e0q2_yf-64k-bpe | +1,054,821 | 不动 |
| e1_iso_c1_800m | 2,862,985,685 | 10,921 | 698,944 | e0q2_yf-64k-bpe | +1,741,861 | 不动 |
| e1_iso_c2_100m | 76,190,476,190 | 290,643 | 18,601,152 | stage1+stage2（24,271,634） | +5,670,482 | **E-74 双流** |
| e1_iso_c2_200m | 39,408,866,995 | 150,332 | 9,621,248 | stage1+stage2（24,271,634） | +14,650,386 | **E-74 双流** |
| e1_iso_c2_400m | 19,464,720,194 | 74,252 | 4,752,128 | stage1（6,067,853） | +1,315,725 | 不动 |
| e1_iso_c2_800m | 9,815,950,920 | 37,444 | 2,396,416 | e0q2_yf-64k-bpe | +44,389 | 不动 |
| e1_ln_50m | 9,997,385,728 | 38,137 | 2,440,768 | e0q2_yf-64k-bpe（2,440,805） | +37 | **E-74 钳制**（原 10B/38,146，超 539） |
| e1_ln_100m | 9,997,385,728 | 38,137 | 2,440,768 | e0q2_yf-64k-bpe | +37 | **E-74 钳制**（同上） |
| e1_ln_200m | 9,997,385,728 | 38,137 | 2,440,768 | e0q2_yf-64k-bpe | +37 | **E-74 钳制**（同上） |
| e1_ln_400m | 9,997,385,728 | 38,137 | 2,440,768 | e0q2_yf-64k-bpe | +37 | **E-74 钳制**（同上） |
| e1_ln_800m | 9,997,385,728 | 38,137 | 2,440,768 | e0q2_yf-64k-bpe | +37 | **E-74 钳制**（同上） |

**改动文件清单**（仅 `pretrain/configs/e1/` 与本节文档）：
- `e1_ln_{50m,100m,200m,400m,800m}.yaml`：`budget_tokens` 10,000,000,000→9,997,385,728、
  `expected_iters` 38146→38137，yaml 头注 E-74 钳制原因。
- `e1_iso_c1_50m.yaml`、`e1_iso_c2_100m.yaml`、`e1_iso_c2_200m.yaml`：`data.stages/meta`
  为 stage1+stage2 双流 + E-74 头注（两流 data 改动在本节开始前已存在于工作区，
  E-74 逐臂复算核定其符合裁决，补齐注释并纳入本节统一验证）；budget/expected_iters 不动。
- `e1_iso_c1_50m.env`、`e1_iso_c2_100m.env`、`e1_iso_c2_200m.env`：`YUFORMER_ARM_DATA`
  行尾注释 E-47→E-74（该变量已被 yaml 内联流名取代，仅留档）。
- 未触碰：`pretrain/src/`、e0q2/e2 全部配置、`data/experiments/` 运行产物。

**复演验证**：用训练代码自身校验路径逐臂复演（source env → `load_config()`/`validate()`
→ `TokenStream.open` → `concat_streams` → `DataCursor.require`），**19/19 PASS**；
不启动训练、不占 GPU。复演脚本 `/tmp/e1_guard_audit.py`（临时文件不入库，可重跑）。
