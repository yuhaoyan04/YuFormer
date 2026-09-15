# YuFormer

从零复现 ZGCM-1 式大语言模型全链路：**数据收集 → 数据处理 → 数据配比 → Tokenizer → 模型架构 → Pretrain → SFT → RL**，并配套技术报告。

## 项目定位

- 参考对象：[ZGCM-1](https://github.com/zgcagi/ZGCM-1)（7.39B、4.19T tokens、数学推理 + Agentic Search）
- 目标：以可运行、可回溯、可写进简历的小规模全链路复现，验证对 LLM 训练工程体系的完整理解
- 调研基线：[docs/research/ZGCM-1-认知框架.md](docs/research/ZGCM-1-认知框架.md)

## 规划模块（对应 ZGCM-1 五大阶段目录）

| 目录 | 阶段 | 状态 |
|---|---|---|
| `data-process/` | 数据收集 / 清洗 / 去重 / 配比 | 待开发 |
| `tokenizer/` | 分词器训练与评估 | 待开发 |
| `pretrain/` | 预训练 | 待开发 |
| `midtrain/` | 长上下文中期训练 | 待开发 |
| `sft/` | 监督微调（通用 + Agentic） | 待开发 |
| `rl/` | GRPO 强化学习（可验证奖励） | 待开发 |
| `report/` | 技术报告 | 待开发 |

## 环境与版本管理

- 仓库：https://github.com/yuhaoyan04/YuFormer
- 每个阶段目录遵循"运行闭包"原则：自带 code + configs + docs + launcher

## Status

🚧 项目初始化阶段——调研已完成，等待开发计划确定。
