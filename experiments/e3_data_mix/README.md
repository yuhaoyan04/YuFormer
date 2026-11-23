# E3 数据配比

**研究问题**：领域响应曲线 / 质量数量 trade-off / 打分器贡献
**设计权威**：[`../../docs/research/实验设计.md E3 节`](../../docs/research/实验设计.md)（问题/假设/矩阵/图表编号以此为准）

## 结构

- `configs/` — 一臂一配置（自包含：模型/数据/优化器/预算/种子）
- `scripts/` — launcher（配置→CLI）+ 收集/拟合/画图
- `results/` — 指标 JSON / 拟合参数 / 图表 PNG（大数据走外部存储）

## 已冻结的主矩阵

`configs/e3_matrix.yaml` 固定八臂、每臂约 10B reference tokens 的物化容量：M2 基线、code/math/chinese/
academic/encyclopedia 五个 2× 响应臂、M1 原生比例和 M3 质量前缀。单域 2× 臂只从 web
扣除相同权重，其余域保持不变，因此响应可解释。

所有臂只从已经通过 release gate 的 `mix_m3_e1_100b` members 派生，并复用同一份九域
held-out validation。`deterministic_broad` 在已审计域池内做稳定哈希抽样；
`quality_prefix` 按 composite 降序取完整文档前缀。两种策略显式区分，避免把领域权重效应与
质量深度效应混在一个变量里。`scripts/audit_matrix.py` 强制检查八臂 roster、权重归一、
10B token 配额、源域容量、release/tokenizer snapshot 一致性；当前静态门禁为 PASS。

训练量独立读取 `pretrain/configs/research_budget.yaml`：当前初筛每臂 1B 最终 tokenizer
tokens，按完整 optimizer updates 向下对齐。保持既有物化矩阵、release 和 token binaries
的身份不变；训练消费前缀的长度绑定到各臂配置。入选配比与 M2 基线再各做三个 seed、每臂
3B 的独立复验；基线入选时去重。aggregate loss 及九域非劣性复验 PASS 才能冻结配方。

运行结果必须包含九域独立 validation loss。缺少任一域时，结果收集器会返回 FAIL，不能进入
配比响应拟合或 50B/800B/150B 最终配方冻结。

## 自动执行链

`scripts/materialize_orders.py` 从审计通过的 100B members/order 派生八个确定性顺序表，
以完整文档为边界且不超过每域 token 配额。`deterministic_broad` 复用源数据的内容哈希顺序，
让各臂共享随机数；`quality_prefix` 按 `(composite desc, doc_id asc)` 选择后恢复内容哈希训练顺序。
产物 manifest 记录输入与八个输出的全量 SHA-256，并可直接作为派生 data release。

`scripts/tokenize_orders.py` 逐臂调用正式 tokenizer 物化器，保留 part cache 并进行臂级有限重试；
`scripts/audit_token_streams.py` 独立重读八条训练流和九条共享验证流的全部字节，只有 17/17 精确
roster 才会 PASS。随后 `scripts/generate_configs.py` 根据 E2 胜出架构、LR 和八条流的共同容量
生成等算力配置，避免把 S5 选择 token 数误当成 BBPE 实际容量。`scripts/analyze_results.py`
要求八臂全部具备九域 loss，输出响应斜率、质量/数量差值和最终配比候选。选择阈值独立记录在
`configs/e3_analysis_policy.yaml`，不会改变已经物化的数据矩阵身份；候选相对 M2 任一 held-out
领域 loss 的回归不得超过 2%，随后才在 0.002 聚合 loss 实用等价带内选择最接近 M2 的配方。

以上步骤由 `pretrain/scripts/pretrain_readiness_watch.py --auto-bootstrap` 持久编排；顺序表、
分词、全量审计、配置生成和训练队列都有状态文件、进程检测与最多三次退避重试。
