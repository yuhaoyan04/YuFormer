# YuFormer Data Process

AI-native 数据治理流水线——清洗、去重、过滤、打标、配比、长度分桶。

## 数据领域与数据集

YuFormer 按 **10 个领域** 组织预训练数据，每个领域选定 2-5 个最优开源数据集，全部经 HuggingFace 验证可下载。

| 领域 | 数据集 | HF 路径 | 体量 | 用途 |
|---|---|---|---|---|
| **通用网页** | FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | 1.3T tokens | 教育质量分数，课程式排序 |
| | FineWeb | `HuggingFaceFW/fineweb` | 15T tokens | 通用网页主干 |
| | DCLM-Baseline | `mlfoundations/dclm-baseline-1.0` | 4T tokens | OLMo-2 使用 |
| | C4 | `allenai/c4` | 200B tokens | 经典基线 |
| **代码** | The Stack v2 Dedup | `bigcode/the-stack-v2-dedup` | 30TB | 600+ 语言 GitHub 源码 |
| | StarCoderData | `bigcode/starcoderdata` | 783GB | StarCoder 训练集 |
| **数学** | Proof-Pile-2 | `EleutherAI/proof-pile-2` | 55B tokens | arXiv+OpenWebMath+AlgebraicStack |
| | OpenWebMath | `open-web-math/open-web-math` | 14.7B tokens | 数学网页 |
| | AutoMathText | `math-ai/AutoMathText` | 14B tokens | LLM 筛选数学文本 |
| | MathPile | `GAIR/MathPile` | 9.8B tokens | 多源数学 |
| **学术** | peS2o | `allenai/peS2o` | 40M 论文 | S2ORC 清洗版 |
| | arXiv LaTeX Corpus | `KiteFishAI/arxiv-tex-corpus-full` | 80GB | arXiv LaTeX 源码 |
| **文学** | PG19 | `deepmind/pg19` | 28k 本书 | 公有领域书籍，长文本 |
| **百科** | Wikipedia (EN) | `wikimedia/wikipedia` | 4B tokens | 事实知识 |
| | Wikipedia (ZH) | `wikimedia/wikipedia` | 1.7B tokens | 中文百科 |
| | WikiText-103 | `Salesforce/wikitext` | 103M tokens | 精选文章 |
| **问答** | Stack Exchange | `HuggingFaceH4/stack-exchange-preferences` | 10M Q&A | 专家问答 |
| **中文** | SkyPile-150B | `Skywork/SkyPile-150B` | 150B tokens | 中文网页语料 |
| | CCI3-Data | `BAAI/CCI3-Data` | 1TB | 智源中文互联网 |
| | CCI3-HQ | `BAAI/CCI3-HQ` | 518GB | CCI3 高质量子集 |
| **多语言** | FineWeb-2 | `HuggingFaceFW/fineweb-2` | 4.3TB | 1000+ 语言 |

下载方式见 [`scripts/download_pretrain_data.py`](scripts/download_pretrain_data.py)，支持按 `max_size_gb` 限制下载量。

## 流水线架构

```
JSONL 输入
  → DatasetAdapter 字段映射
  → 类别首步归一化
  → COMMON_PREFIX（NFKC 归一 / 完整性 / 来源 / 脱敏）
  → 类别专属步骤（code/web/agentic/instruction/math/reasoning）
  → COMMON_SUFFIX（质量策略 → 精确去重 → 近似去重 → 计长分桶 → 质量分桶）
  → 清洗后 JSONL + summary + manifest
```

### 三层核心抽象

1. **Sample + Decision**：全管线统一记录载体，三态决策 keep/review/drop（只降不升）
2. **Step + Pipeline**：流式执行引擎，DROP 即短路
3. **Adapter + Profile**：数据集专属规则以插件挂载，主干保持小而稳定

### Context Length 分桶（阶段性学习支持）

YuFormer 的创新点之一：**数据处理阶段就按可配置的上下文长度阶梯对样本分桶**，支撑 pretrain/midtrain 的渐进式长上下文训练。

在 `configs/cleaning_config.yaml` 中配置：
```yaml
context_lengths: [4096, 16384, 65536]  # 对应 pretrain stage 1/2/3
```

处理时每个样本被分配到 `B4K` / `B16K` / `B64K` / `gt_64K` 桶。配比阶段用**互斥前缀嵌套**（mix4K ⊂ mix16K ⊂ mix64K）构造各阶段的训练数据——短样本对更长阶段保持资格，但每条 ID 只进一个最终 mix。

这与 ZGCM-1 的 `mix16⊂mix64⊂mix256` 机制一致，但边界完全可配置。

## 使用

```bash
# 清洗单个数据集
PYTHONPATH=src python -m yuformer_data --input data.jsonl --output cleaned.jsonl \
  --stage pretrain --category code --dataset my-code-corpus

# 带质量策略与去重配置
PYTHONPATH=src python -m yuformer_data --input data.jsonl --output cleaned.jsonl \
  --stage pretrain --category web --dataset my-web-corpus \
  --config configs/quality_policies.json --near-dedup

# 冒烟测试
PYTHONPATH=src python scripts/smoke_test.py
```

## 输出格式

每条输出记录（JSONL，`sort_keys=True`）：
```json
{
  "buckets": {"length": "1_4096", "quality": "main", ...},
  "category": "code",
  "data": {...},
  "dataset": "my-corpus",
  "decision": "keep",
  "flags": [],
  "metrics": {"estimated_tokens": 1024.0},
  "reasons": [],
  "record_sha256": "a1b2...",
  "source_id": "doc-001",
  "stage": "pretrain"
}
```

另输出 `*.summary.json`（per-step 计数器）和 `*.manifest.json`（从输出重建的决策计数）。
