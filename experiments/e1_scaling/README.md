# E1 Scaling Law

## 2026-11-07 配对 seed 复验更新

在每点 1 seed 的 E1 screen 上，按冻结的 24 个设计点逐点补跑 seed
20261002/20261003；seed 20261001 的 screen 结果作为第三个配对观测。因此最终分析
要求 24 点 × 3 seed、72 个结果身份全部通过，并审计配置预算上限、实际训练 token
与 optimizer-step 取整短差。seed-aware 拟合用每点三 seed held-out loss 的算术均值；
身份审计公共入口、复验准备器与分析器都要求配置名精确匹配标准 24 点 roster；
参数区间在每次 bootstrap 中对所有设计点共用一组 seed 索引，以保留配对协方差。
固定 bootstrap RNG seed 为 20261107，并写入拟合 JSON。
同步 seed-index bootstrap 是在复验队列启动后根据配对设计作出的透明方法修订，而非原始预注册；
contracts manifest 留存旧/新方法及输入 SHA-256，最终结果应按该修订报告。
拟合报告还按三条 isoFLOP 家族的中位实际 `6ND` 计算解析的 compute-optimal
`(N*,D*)` 与 `D*/N*`，用同一组 seed bootstrap 样本给出区间；曲线导出生成单独的
compute-optimal frontier PNG/SVG。该结果是拟合形式下的解析最小值，不是直接训练观测。
闭式为 `N*=[(Aα)/(Bβ)·(C/6)^β]^(1/(α+β))`、`D*=C/(6N*)`。
只有完整 24 点拟合和身份审计均 PASS 时才绘制为合格前沿；否则图中保留 pending 状态。
manifest 同时记录前沿整体状态及 c0/c1/c2 有效家族，不能用部分前沿冒充完整结果。
只有拟合质量、预算、seed roster 和身份门禁全部通过，才允许把结果用于 1T-token
生产候选决策；E1 复验及 scaling 门禁完成前不得启动全量 1T 预训练。当前运行记录和
收尾命令见 `report/PRETRAIN_1B_EXECUTION_20261022.md`。

2026-11-02 当前范围：仅 E1 scaling 与必要 LR/batch/独立性能证据；E2/E3 及其
复验延期。LR 21 臂、batch 15 臂与独立性能 5 臂已完成，E1 24 臂自动流水线运行。
batch 固定为已声明的 1M-token 参考基线，E1 screen 实际为 262,144 tokens/update；
2M/4M 搜索边界限制保留，不宣称全局最优 batch。

研究目标是在固定 tokenizer、PASS 数据 release、模型族和训练后端下拟合 compute-optimal
`L(N,D)`，而不是用单点 loss 给 0.5B/1T 候选背书。数据 release 与 tokenizer gate 已
**PASS**：E0 在九域 held-out 语料选择 64K BBPE，实验训练/validation streams 已通过审计。
2026-10-22 修复 EOD/PAD embedding 语义后，当前训练代码已通过 12 项 GPU 资格，
包含真实 800M/64 次梯度累积、SIGKILL 恢复与跨拓扑 reshard；LR/batch pilot 已完成。

实验分三阶段：

1. 对 50M/200M/800M 三个锚点做 bracketed peak-LR pilot；每个锚点扫描中心学习率的
   0.25×/0.5×/1×/2×/4×，固定 warmup 比例和 cosine 定义，以 held-out token cross-entropy
   与稳定性共同
   选取按参数规模缩放的学习率规则；
2. 在 200M 锚点完成 1M/2M/4M tokens/update 与匹配预算的 LR 扫描；当前协议固定
   具有双侧 LR 证据的 1M 参考基线，并迁移学习率，不作全局 batch 最优化结论；
3. 在约 50M/100M/200M/400M/800M 五档执行三条 isoFLOP 线，并加入固定 1B-token 的
   `L(N)` 截面。E1 的 24 个唯一设计点各跑 1 seed，并用受控 `L(N)`/`L(D)` 截面、跨点
   bootstrap、残差与 leave-one-out 门禁约束拟合不确定性。跨点 bootstrap 不代表重复
   seed 方差；E2/E3 比较延期，不报告其选择结论。失败 run 保留并分类。

第一阶段的机器合同已落在 `pretrain/configs/e1_pilot/`：50M、200M、800M 三个锚点各扫描
中心学习率的 0.25×/0.5×/1×/2×/4×，每臂固定 500M tokens，共 15 臂。
200M 的原最优点在上界，已通过 `pretrain/configs/lr_pilot_search.json` 声明额外
1.8e-3/2.4e-3/4.8e-3 三臂，配置位于 `pretrain/configs/e1_pilot_extension/`。
补测由 `prepare_lr_extensions.py` 从基线复制，保持 seed、数据、batch、预算、调度和评测相同。
独立队列复用空闲 GPU，不中断正在进行的 800M 实验。选型按实际 LR 排序，必须等全部
已声明的 LR 臂完成且最优点两侧都有有效对照；补测身份同时纳入审计。
`night_research_policy.json` 声明有预算上限的自动 LR/batch 扩展。200M 的独立
`lr_200m_batch_anchor.json` 通过损失与身份门禁后即可启动 batch，无需等待 800M；
完整 LR 选型仍是 E1/E2 的前置条件。随后
`pretrain/configs/batch_pilot/` 的 9 臂各训练 1,996,488,704 tokens；该预算可被最大 4M
tokens/update 整除，保证三个 batch 候选实际消费完全一致。batch 边界补测也保留该预算、
seed、数据与 batch，只改变 peak LR。已开跑配置禁止重写。batch pilot 在单卡上以梯度累积模拟
global batch，用于隔离优化批量效应；多卡通信、显存和实际吞吐由 TorchTitan qualification 与
正式 pilot 的运行指标另行验收。全部匹配预算证据、选中基线的双侧 LR、身份与稳定性
门禁通过后生成 E1 的 LR；未选中 2M/4M 的边界限制继续记录。短实验的 tokens/update
上限为 262,144，按平方根规则将 LR
换算到该 batch，且每臂至少 128 次更新；生产 batch 的原始选择单独保留供四卡 canary 验证。

所有正式臂由 `experiments/_common/gen_e1e2_arms.py` 生成，当前共 24 个候选配置：5 个固定
1B 的 `L(N)` 点、5 个固定约 300M 的 `L(D)` 点，以及 14 个 isoFLOP 点。`L(D)` 数据量为
0.1B/0.3B/1B/3B/9.846B。isoFLOP 三档为 4.8e17/1.4e18/4.8e18，以真实模型形状计算的
统一 FLOPs/token（含词表投影和 attention）配平，误差小于一次更新的 FLOPs。算法 FLOPs
不计 activation recomputation；实际性能独立测量。预算唯一来源为
`pretrain/configs/research_budget.yaml`。它们显式要求
TorchTitan、EOD 与 PASS data release，使用 AdamW、约 1%（5–2000 step）warmup 和 cosine；
当前 peak LR 已通过 `lr_resolution.json` 解析生成。24 个活跃 E1 环境文件及已完成的
21 个 LR、15 个 batch 环境文件已经绑定
`mix_m3_e1_100b_stage2`、64K 最终 token stream、PASS data release 和
`data/TOKENIZER_READY.json`。正式隔离环境锁定 Python 3.12 / PyTorch 2.14 cu126 /
TorchTitan v0.3.0，已在本节点通过资格；后续代码变更仍须重验资格。

结果同时记录 train/held-out token cross-entropy（固定 tokenizer，因此可跨臂直接比较）、真实消费
tokens、non-embedding/total 参数、实际 FLOPs、
wall time、MFU 与故障。拟合报告包含 bootstrap CI、残差图、留一规模敏感性和预测区间；
四卡工期估计对单卡吞吐先乘 0.70 的保守分布式效率，不假设线性扩展；正式点火前仍需以选定
生产架构的持续四卡 canary 替换该临时系数。
所有臂都在最终 optimizer step 强制执行 aggregate 与九域 held-out 验证；结果收集器拒绝
验证步早于最终训练步或缺失任一域的运行，防止短程臂因验证间隔而获得陈旧 loss。
下游评测仅作能力验证，不参与训练 loss 拟合。

## 自动结果导出

readiness 在任务完成唤醒或定期审计后，调用 `pretrain/scripts/render_e1_scaling.py`，
自动更新 `experiments/e1_scaling/results/curves/`：

分析环境依赖声明于 `pretrain/pyproject.toml` 的 `analysis` extra（SciPy、Matplotlib）；
当前控制器 Python 已具备该依赖，绘图不导入训练环境、不占 GPU。

- `scaling_curves.png` / `.svg`：固定数据的 L(N)、固定模型的 L(D)、三条 isoFLOP
  家族以及联合拟合残差。N 轴统一使用 non-embedding 参数，D 使用实际训练 tokens。
- `endpoints.csv`：完成点、实测 held-out loss、实际 N/D 及 run identity。
- `manifest.json`：输入与文件哈希、图表状态、排除点、已审计点数和研究结论范围。

未完成或身份未通过的实验不进入点表；保留有效但非单调的观测，不为得到平滑曲线
删除数据。只有 24 点齐备且拟合质量、完整身份审计均 PASS，图表才标记
`ANALYSIS_AND_IDENTITY_PASS`；此前为 `EXPLORATORY_PROGRESS`。失败拟合不会推进
配方冻结。三 seed 复验期间，最终 gate 的权威拟合文件为
`experiments/e1_scaling/results/seed_replication_fit.json`，screen-only 的
`scaling_fit_screening_20261107.json` 不能替代它。图片是结果视图，正式门禁仍以 seed-aware
fit 和 72-run 身份审计为准。
相同输入不重复绘图；文件损坏会在下一次调用时重建。
