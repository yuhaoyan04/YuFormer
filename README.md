# YuFormer

从数据治理、自训 tokenizer、模型实现，到八卡预训练、长上下文和后训练的可审计 LLM 工程项目。

**2026-11-26：1.5166B 正式预训练正在八张 H800 上运行。** 最近半日复核为第 84760 步，已消费 355.509B training labels，处于 stable 阶段；全链路预计 31.69 天，剩余约 21.09 天。这是运行时预测，最终能力尚未完成。最新验证聚合 loss 2.273167，逐域回升观察为 `{"validation_metrics/domain/books/loss": 0.0595506593523476, "validation_metrics/domain/code/loss": 0.3622287703848759}`。

| 入口 | 内容 |
|---|---|
| [技术报告](report/TECHNICAL_REPORT.md) | 当前架构、数据、选型、预算、结果与限制 |
| [实验数据](report/metrics/README.md) | 验证曲线、规模实测、机器快照与来源哈希 |
| [工程经验手册](docs/engineering/PLAYBOOK.md) | 数据、内存、训练、恢复、评测与后训练排障 |
| [异常案例库](docs/engineering/INCIDENTS.md) | 症状、根因、修复、验证及可迁移规则 |
| [维护规范](docs/engineering/REPOSITORY_MAINTENANCE.md) | 发布、清理、证据保留与训练冻结边界 |

## 当前训练

- 模型：1,516,600,064 参数，18L/H2560/FFN6912，40 Q / 10 KV heads，混合滑窗与全局注意力，自训 65536 词表 BBPE。
- 后端：锁定 TorchTitan v0.3.0、FSDP2、AdamW，确定性 Flash 全局 attention 与编译 Flex 滑窗。每次更新 4,194,304 labels。
- 总预算：**966,254,395,392 labels（约 0.966T），包含 4K WSD、16K 和 64K**；启动工期上限 33 天。
- 4K：218453 updates，warmup/stable/decay 为 11921/170770/35762，peak LR 1e-4、末端比例 0.1。
- 16K/64K：7152/4768 updates，各自 warmup 与 cosine decay，25% 短样本 replay。阶段间只加载权重，同阶段恢复完整状态。
- 长阶段系统与数据资格通过；正式长阶段和长短能力验收等待真实父阶段完成。

预算真源：[effective_total_training_budget.json](experiments/e4_curriculum/results/effective_total_training_budget.json)。其中启动标记属于冻结前快照，不能随运行状态修改；当前状态见带时间的 [metrics](report/metrics/README.md)。

## 目录与完成边界

| 目录 | 作用与状态 |
|---|---|
| `data-process/`、`tokenizer/` | 正式三阶段数据和 64K tokenizer 已准备并验收 |
| `model/`、`pretrain/` | 生产 1.5166B 已过八卡资格，正式 4K 运行中 |
| `midtrain/`、`eval/` | 长阶段数据和八卡系统/恢复已通过；最终能力待测 |
| `sft/` | 对话与 assistant-only packing 契约已实现，完整训练闭环待接入 |
| `rl/` | verl/SGLang、rollout 契约已实现，模型 adapter 和训练闭环待完成 |

E1 scaling NRMSE 3.663% 超过预注册 3% 门槛，保留 FAIL 和用户豁免。短时筛查、系统资格、训练期验证及最终能力分别报告。

## 环境与复现

生产依赖见 [backend.lock.json](pretrain/backends/torchtitan/backend.lock.json)，各目录 README 和 `docs/` 说明入口。RL 使用独立环境。源码、配置、测试、正式 tokenizer 和轻量指标进入 Git；大型语料、权重与运行归档通过外部资产和哈希管理。克隆源码不自动获得全部训练资产，也不构成启动重复正式训练的指令。

参考工程：[ZGCM-1](https://github.com/zgcagi/ZGCM-1)。历史材料的日期表示当时事实，当前状态以上述快照为准。
