# E0 Tokenizer

**研究问题**：为什么 BBPE / 词表大小随模型规模缩放 / 训练成本与原理
**设计权威**：[`../../docs/research/实验设计.md E0 节`](../../docs/research/实验设计.md)（问题/假设/矩阵/图表编号以此为准）

> **2026-10-19 复审：以下 2026-09-26 结果只作为历史 pilot，不构成 T0。** 当时的 M2
> tokenizer corpus 含当前 release 已排除的 arXiv/SkyPile，未绑定 S6 train roster，也未证明
> 与 val 互斥；现存三档 tokenizer 还缺少当前 chat contract 的两个 tool-call special token。
> 正式 E0 将用 `m3_coverage_balanced_v1` 的同一批 train-only 文档重训 32K/64K/128K，重新做
> 独立 val 压缩率和固定原始 byte 预算的 BPB 对照后再冻结 tokenizer id。现有 64K 选择是待
> 验证假设，不能直接进入 scaling。

## 结构

- `configs/` — 一臂一配置（自包含：模型/数据/优化器/预算/种子）
- `scripts/` — launcher（配置→CLI）+ 收集/拟合/画图
- `results/` — 指标 JSON / 拟合参数 / 图表 PNG（大数据走外部存储）

## 历史臂清单（2026-09-26 pilot 指标，已失效）

| 臂 | 配置 | 状态 | 关键结果 |
|---|---|---|---|
| yf-32k-bpe | BBPE 32,768 | ✅ 训练+评估 | overall 3.59 B/tok；中文 3.91；code 2.86 |
| yf-64k-bpe | BBPE 65,536 | ✅ 训练+评估 | **overall 3.86 B/tok（Qwen2.5-151k 的 98.5%，词表仅 43%）**；已定为 pretrain-0.5b 预设 |
| yf-128k-bpe | BBPE 131,072 | ✅ 训练+评估 | overall 4.07 B/tok；64k→128k 边际 +5.4% 但 embedding 参数 ×2（tied 下占总参 14%→27%） |
| gpt2-50k | baseline | ✅ 评估 | overall 2.36 B/tok——中文崩到 1.41（无 CJK 子词），印证"词表必须匹配训练分布" |
| qwen2.5-151k | baseline | ✅ 评估 | overall 3.92 B/tok；code 3.35 优于 yf-64k（3.07）——大词表的 code 子词优势 |
| ~~unigram~~ | — | ❌ 2026-09-25 取消 | HF tokenizers 训练器 Rust 级崩溃 ×2（TryFromInt/Internal）——训练器成熟度本身成为 BBPE 胜出实证 |

### 数据与产物

- `results/e0_full_metrics.json`——分域（web/code/math/academic/books/encyclopedia/chinese/qa）压缩率 + fertility（tokens/word，CJK 逐字口径）+ 编码吞吐（单线程 2.4-2.9 MB/s，5 档同量级）
- `results/compression_summary.json`——200k 样本 × 25GB M2 加权语料的总表（vocab 利用率 0.95-0.99、unk=0、roundtrip=1.0）
- `results/figures/E0-1_domain_compression.png`——图 E0-1 分域压缩率分组柱状图（5 tokenizer × 8 域）
- `results/figures/E0-2_vocab_ablation.png`——图 E0-2 词表三联图（压缩率/embedding 占比/吞吐）
- 采集脚本：`tokenizer/scripts/eval_e0_full.py`（--per-source 20000，确定性行组抽样）；画图：`scripts/plot_e0.py`

### 待补（E0-Q3 训练成本账目）

- 三档训练的 wall time / 峰值内存 / 语料量账目**缺失**（历史训练未计时）——下次重训任一档时记录（建议随 E1 前的 tokenizer 冻结重训一并做）
- E0-Q2 代理训练（固定 10B tokens 小模型 loss 对比）依赖 pretrain 基础设施，排期至 E1 前 |
