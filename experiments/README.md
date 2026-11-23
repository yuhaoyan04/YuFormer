# YuFormer Experiments

按研究问题组织的实验目录（每个实验 = 运行闭包：README + configs + scripts + results）。
实验的问题定义、假设、矩阵、图表产出以 [`../docs/research/实验设计.md`](../docs/research/实验设计.md) 为权威；
本目录负责"怎么跑、结果放哪"。

| 目录 | 研究问题 | 报告图表 |
|---|---|---|
| `e0_tokenizer/` | 为什么 BBPE / 词表大小缩放 / 训练成本 | 图 E0-1~E0-3 |
| `e1_scaling/` | L(N) / L(D) / isoflop / 下游 scaling | 图 E1-1~E1-6（核心章） |
| `e2_architecture/` | 深度宽度 / FFN 比例 / 注意力配比 / 组件消融 | 图 E2-1~E2-5 |
| `e3_data_mix/` | 领域响应曲线 / 质量数量 trade-off / 打分器贡献 | 图 E3-1~E3-3 |
| `e4_curriculum/` | 课程式 vs 打乱 / 长上下文渐进外推 | 图 E4-1~E4-3 |

## 约定

- **一臂一配置**：`configs/<arm_name>.yaml` 自包含（模型/数据/优化器/预算），launcher 只做配置→CLI 翻译。
- **`results/` 只存小产物**：指标 JSON、拟合参数、图表 PNG；训练数据/checkpoint 经环境变量指向外部存储，不入库。
- **每臂必有 manifest**：数据快照（mix config hash / tokenizer hash）、模型 config hash、随机种子——可回溯（工程宪法 #4/#5）。
- 共享基础设施（实验追踪/画图/快速评测）在 `_common/`，各实验只读引用。
