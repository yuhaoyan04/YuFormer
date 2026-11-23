# Pretrain 前置实验执行计划（2026-10-21）

正式 pretrain 配方由三组证据依次确定：E1 拟合规模与数据预算，E2 选择架构形状，E3 选择
数据配比。可执行合同以 `experiments/_common/gen_e1e2_arms.py`、生成后的 YAML/env、
`pretrain/scripts/validate_formal_contracts.py` 和 readiness watcher 为准。

## 正式矩阵

| 阶段 | 规模 | token budget | 主要产出 |
|---|---:|---:|---|
| LR pilot | 15 臂（3 个规模 × 5 个对数间隔 LR） | 500M/臂 | bracketed 稳定 LR、真实吞吐与显存边界 |
| Batch/LR pilot | 9 臂（1M/2M/4M tokens/update × 3 LR） | 1.996B/臂（共同 step-aligned） | 联合选择 global batch 与 batch-scaled LR |
| E1 scaling | 24 臂 | 1B–98.46B/臂 | `L(N,D)`、bootstrap 置信区间、45 天可行前沿 |
| E2 架构消融 | 10 点 × 3 seeds = 30 臂 | 10B/臂 | scaling 归一后的架构排名与跨 seed 方差 |
| E3 数据配比 | 8 臂 | 约 10B/臂 | 分域响应与最终 mixture |

E3 配置在八条 token stream 完成审计后原子生成，因此 formal roster 在此之前是 78，生成后是
86。身份审计必须 exact-roster PASS，额外、缺失或陈旧结果都不能进入分析。

## E1：Scaling law

- `L(N)`：五个规模，固定 10B 数据。
- `L(D)`：固定约 300M 参数，数据量为 1B、3B、10B、30B、98.46B。
- IsoFLOP：三条算力线，共 14 个 `(N,D)` 点。
- 拟合 `L(N,D)=E+A/N^α+B/D^β`，报告残差、R²、归一化 RMSE、leave-one-out
  诊断和 bootstrap 置信区间。
- 45 天可行性使用吞吐 bootstrap 的保守 95% 上界，不能使用单次峰值吞吐。

## E2：Iso-param 架构消融

10 个唯一架构点为 `l16`、`l32`、`l48`、`ffn2x`、`ffn4x`、`nogate`、`noqkn`、
`fullrope`、`swaoff`、`untied`。每点使用 seeds 20261001、20261002、20261003。

所有臂共享数据顺序、token budget、AdamW/cosine 定义、评测集和 TorchTitan commit。分析器
使用 E1 scaling fit 将每臂 loss 折算到统一 non-embedding 参数量，再聚合均值和标准差；三组
seed 做配对差值 95% CI，在与经验最优无法区分的集合中选择估算 FLOPs 最低者。禁止将不同
参数量或 FLOPs 的 raw loss 直接解释为架构收益。

## E3：数据配比

八臂共享架构、token budget、验证集和顺序生成规则，只改变 mixture。训练数据使用 E3 专属
order，验证数据使用固定 base release，避免验证集随实验臂变化。选择报告必须同时给出聚合
loss、分域 loss、吞吐和关键下游指标。单 seed screening 不宣称统计显著性；在经验最优
aggregate loss 的 0.002 实用等价带内优先选择最接近 M2 baseline 的配比，避免追逐微小噪声。

## 自动执行 DAG

```text
data/tokenizer gates
  -> TorchTitan numerical + recovery + reshard qualification
  -> 15-arm LR bracket
  -> 9-arm batch/LR joint pilot
  -> E1 scaling + fit
  -> E2 30 arms + normalized analysis
  -> E3 8 arms + mixture analysis
  -> freeze recipe
  -> sustained 4-GPU frozen-model throughput canary
  -> materialize 50B warmup / 800B stable / 150B annealing streams
  -> exact-roster, full-hash, overlap and production-contract audits
```

GPU 队列需要四张空闲卡；CPU 数据阶段可与不依赖它的分析并行。readiness watcher 对可恢复失败
使用有界重试和依赖代际触发，外部驱动或 GPU 占用不消耗重试次数。人工深度检查按任务尺度
限频：小时级约 15 分钟，天级每 2–3 小时；后台仅做低成本存活与失败信号检测。

## 点火门

- 数据 release、tokenizer、EOD、随机顺序、容量和 overlap 审计全部 PASS。
- TorchTitan 环境 preflight 与数值、断点续训、reshard 资格测试全部 PASS。
- 训练配方只引用冻结的 tokenizer、order release、代码和配置哈希。
- 50B/800B/150B 三阶段严格覆盖 1T tokens，WSD LR 边界和 context scaling 由真实调度器合同验证。
- batch pilot 选出的 tokens/update 与 batch-scaled LR 必须写入正式配方，并重新对齐三段边界。
- 正式吞吐承诺来自冻结模型的 200-step、4×H800 持续 canary；以稳态 p10 吞吐推算 1T 工期，
  45 天门禁未通过时不能发布 production contract。
