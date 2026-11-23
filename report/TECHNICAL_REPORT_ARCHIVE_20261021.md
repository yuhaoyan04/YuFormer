# YuFormer 技术报告（草稿 v0.2）

> 日期：2026-10-21 ｜ 状态：**进行中**（正式数据门禁已通过；E3 tokenization 进行中；TorchTitan 运行时资格测试等待驱动升级）
> 写作原则：做了什么写什么。没有的结果不承诺，未完成的明确标注。所有数字可回溯到仓库内文档与产物（路径随文给出）。
> 素材冲突处以主代理裁定口径为准，并在文中以「⚠️ 存疑」标注。

---

## 1. 项目概述

**YuFormer 的目标**：站在 ZGCM-1 的工程范式上，**从零训练一个 0.5B 大语言模型**，并完整走通
数据 → scaling/架构/配比实验 → pretrain → SFT → RL → 评测 → 技术报告的全链路。
ZGCM-1 是一位先行者训练并开源的 7.39B 生产级模型；YuFormer 继承它的工程骨架
（运行闭包、自包含配置、确定主义、三态审计），但模型实现、tokenizer、训练栈全部自己写，
并在每个模块留下有报告价值的消融结论。

**规模**（2026-09-16 锁定，见 `docs/research/预训练规模核算.md`）：

| 维度 | 数值 |
|---|---|
| 数据池 | 19 源 ~15.3TB raw（15,269.61 GB 实测，download_report.json）/ ~5.1T tokens（清洗后 ~4.4T，见 §3） |
| 模型 | 0.5B（pretrain-0.5b 预设：32L/h1024，tied ~473M 参数） |
| pretrain 预算 | **1T final-tokenizer tokens**：50B warmup + 800B stable + 150B anneal |
| 上下文 | 4K base pretrain；另计 50B 的 4K→16K→64K context-extension 实验 |
| 下游 | midtrain / SFT / RL / eval，总周期 1–2 个月 |

**方法一句话**：数据决定上限——用工业级管线（严格清洗低丢弃、全量打分排序、top-k 确定性
选取）从 ~4.4T 清洗池中按阶段物化 1T final-tokenizer 训练流；配置不拍脑袋——scaling /
架构 / 配比三组实验（E1/E2/E3）跑完再冻结最终 pretrain 配方。

**当前进度**（截至 2026-10-21）：

- ✅ 数据与 tokenizer 门禁通过；最终采用 `yf-64k-bpe`，九域验证流和 100B 研究流审计通过。
- 🔄 E3 八臂顺序已物化，tokenization 逐臂进行；完成后自动进入流审计与配比实验配置生成。
- ✅ pretrain 代码冻结候选通过 198 项回归；正式合同固定为 1T WSD（50B/800B/150B）。
- ⏳ TorchTitan v0.3.0 数值、恢复与 reshard 资格测试等待主机驱动从 535.161.08 升级到
  CUDA 13 栈要求的 ≥580.65.06；E1/E2/E3 训练结果及正式 1T 流因此尚未完成。
- ⏳ base pretrain / context extension / SFT / RL / eval：**未开始**。

## 2. 工程范式与架构

### 2.1 从 ZGCM-1 继承什么

ZGCM-1 留给 YuFormer 最值钱的不是代码，而是八条工程宪法（全文见 `AGENTS.md` §1）。
逐条一句话 + 我们的落点：

| 宪法 | 一句话 | YuFormer 落点 |
|---|---|---|
| 运行闭包式阶段目录 | 每个一级目录自带 code+configs+docs+launcher，可整体拷贝 | `data-process/`、`pretrain/`、`experiments/e*/` 均按此组织 |
| 自包含配置 = 单一真源 | 一个配置文件声明全部超参，launcher 只做机械翻译 | `pretrain/configs/pretrain_base.yaml`（0.5B×1T WSD 正式合同） |
| 外部资产经环境路径声明 | 数据/tokenizer/checkpoint 不入库，`${YUFORMER_*}` 注入，缺失即 fail-fast | pretrain yaml 全文 `${VAR}` 占位 |
| 确定主义贯穿 | blake2b 稳定哈希 + 显式 seed；数据顺序离线物化、运行时保序消费 | S6 mix_order 顺序表 + `cursor.json` 样本级断点，重启逐位一致（smoke 断言） |
| 三态决策 + 全程审计 | keep/review/drop 三态，可疑打标不静默丢；每条输出带 reasons + record_sha256 | S1 decisions 打标（drop 28 / keep 54,637 / review 944）、全链 manifest |
| 元数据驱动 + fail-fast | 迭代数从数据元数据推导；身份校验拦截错误续训 | 1T 合同推导 `expected_iters=476837`；DCP cursor、release、tokenizer 与运行身份必须同时匹配 |
| 对上游最小侵入 | 框架扩展收敛为独立模块，上游保持可升级 | YuFormer 适配层集中在 `pretrain/backends/torchtitan/yuformer_ext/`，锁定 TorchTitan commit |
| 文档即代码 | 关键决策写进文档而非口头记忆 | `FAILURE_LEDGER.md`（E-1~E-71）、`DECISION_RECORD_20261010.md` |

### 2.2 训练栈选型：自研模型，工业级运行时

模型结构由项目自行实现；正式分布式运行时已从早期 PyTorch 2.5.1 原型迁移到锁定 commit 的
**TorchTitan v0.3.0**。早期自研栈继续承担模型语义、损失与优化器研究对照，生产候选由
TorchTitan 提供 FSDP2、Distributed Checkpoint、容错恢复与可观测性，YuFormer 只维护边界清晰
的模型、数据和配置适配层。这样既保留从零实现的研究价值，也让正式训练的恢复与扩展路径对齐
主流工业训练栈。

| 组件 | 当前实现 | 验收口径 |
|---|---|---|
| 分布式 | TorchTitan FSDP2 `full_shard`，单机 4×H800 | 单卡数值对拍、四卡 smoke、4→2/8 reshard 均须通过 |
| Checkpoint | Distributed Checkpoint；保存拓扑无关 global-sample cursor、模型、优化器、scheduler 与 RNG | train/save/load 三类 SIGKILL 恢复必须与不中断参考轨迹一致 |
| 优化器 | 正式基线 AdamW；Muon 只在 matched-token / matched-FLOP 研究臂启用 | E1 pilot 决定 peak LR 与 global batch，研究臂不得静默覆盖生产配置 |
| 学习率 | 50B linear warmup → 800B stable → 150B cosine decay，`min_lr_ratio=0.1` | 阶段边界按完整 optimizer step 对齐，续训后 scheduler 状态连续 |
| 数据 | mmap 预分词流 + EOD；release、tokenizer gate、stream audit 与消费前缀共同绑定运行身份 | byte size、token count、边界 hash、EOD、dtype、顺序与 cursor 全部 fail-closed |
| 损失与显存 | chunked cross-entropy；4K 基线默认关闭 activation checkpointing | 数值误差与吞吐由资格测试和 production canary 给出，不沿用旧原型 MFU 推断 |

锁定环境为 Python 3.12、PyTorch 2.14.0/cu130、TorchTitan v0.3.0、torchao 0.18.0、
torchvision 0.29.0、Triton 3.8.0。当前主机驱动 535.161.08 不满足 CUDA 13 要求，因此仓库合同
已闭合，但运行时资格测试仍保持 fail-closed。

## 3. 数据工程

这是本项目迄今投入最大的环节：19 源、PB 级中间产物、E-1~E-71 七十一条编号事故。
以下按"源怎么定 → 怎么洗 → 怎么打分 → 怎么选 → 怎么审 → 怎么切"展开。

### 3.1 数据源：19 源目录的诞生与裁撤

源目录在 2026-09-15 按领域锁定（`docs/research/预训练数据源目录.md`，每个源给出精确 HF
路径、体量、许可证、下载命令——这是对 ZGCM-1 黑盒数据源的第一个公开贡献）。09-16 依据
"4×H800 / 176 核 / 2TB 内存 / 329TB 共享存储"的算力底盘做可下载量核查，定稿 19 源
15,269.61 GB（~15.3TB，以 `data/pretrain-raw/download_report.json` 实测为准；
README 的 15.4TB 为早期估算口径，本报告统一用 15.3TB）。

三次剔除与补偿：

- **3 个 gated 源**（本机 HF token 未授权）：the-stack-v2-dedup（代码）→ 补偿
  codeparrot/github-code 300GB 哈希均匀抽样；MathPile（数学）→ proof-pile-2 + automathtext
  已覆盖；CCI3-Data（中文 1TB）→ cci3-hq 提额至 250GB。
- **c4 弃用**（2026-09-22，E-16 处置）：c4 在 S1 历史故障期产物丢失，回补会使 web 域
  S2/S3 去重全链作废；且 CC 覆盖已由 fineweb(10TB)+dclm(600GB) 提供，c4 价值密度最低。
  裁定：弃用。
- **pg19 弃用**（2026-09-27 定案）：28,755 个 .txt 分片与 parquet 摄取框架不兼容全部失败；
  与 dolma-books（OLMo Dolma v1.6 Gutenberg 全集 7.8GB）重叠实证（KJV Bible 等书目命中）。
  修 ingest + 重跑 S1–S3 的成本 > 边际收益，弃用。

最终进管线的 **17 源**（web 3 / code 2 / math 2 / academic 2 / books 1 / encyclopedia 3 /
qa 1 / chinese 2 / multilingual 1），见 `data-process/configs/data_sources.yaml`。

### 3.2 清洗去重管线（S0→S3）与守恒账

| 阶段 | 做什么 | in → out（行） |
|---|---|---|
| S0 摄取 | 格式归一（parquet/gz/zst/txt → 统一 schema），blake2b doc_id | 15.3TB → 6.486B 行 |
| S1 硬过滤 | NFKC 归一、乱码/HTML/敏感串/语言 ID/域规则；keep/review/drop 三态 | 6.486B 行（drop 仅垃圾） |
| S2 精确去重 | sha256（归一化文本）跨源域内两趟扫描 | **drop 2,825,291,008 行 = 43.6%** |
| S3 近重去重 | 小域 MinHash-LSH；web/code 128×4-bit MinHash + 全文精度审计 + direct-edge dominance | S5 顺序账 drop **532,617,350** |
| S7 去污染 | 8-gram 词组对评测基准索引（与 S5 计长同链） | 原始决定 46,044,928；前序过滤后 drop **27,586,581** |
| S5 计长分桶 | 按 yf-64k-bpe 计 token、4K/16K/64K 分桶 | keep **3,105,637,476 行 / 2.531T tokens** |
| S6 配比物化 | 配额 + top-k 确定性选取 + 随机基线顺序 | 正式 100B mix：**94,235,054 行 / 99,444,599,052 tokens** |

**当前守恒链**（`data/processed/stage_s5_global_summary.json`，2026-10-18）：
S5 输入 **6,485,760,234 行**，12,829/12,829 分片，`shard_errors=0`；
2,825,291,008 exact drops + 527,229,390 near-dup drops + 27,602,360 decontam drops +
3,105,637,476 keeps = 输入总数。保留池为 **2,530,570,633,678 tokens**。
正式 100B mix 的组合机器证书为
`data/processed/audits/data_pipeline_acceptance_20261018.json`（status=PASS，绑定 S4/ndv2/
decontam/S5/val/S6、members/order 与 D1–D4 审计哈希）。

**两个关键决策，值得展开**：

**S2 精确去重抓出 43.6%**——超出工业标配。各家的精确去重通常只在自己源内做；我们的池子
混了多个 Common Crawl 派生源（fineweb / fineweb-edu / dclm），它们各自上游去过重、但**互相**
不去重，跨源精确重复高达 2.73B 行（web 域为主）。这个"跨源重叠无人量化"的现象记为研究
发现 R-1。

**S3 web/code 近重止损与逆转（E-17 → E-17b/E-18 → E-31）**——本项目最曲折的一段：

1. **止损**（09-22，E-17）：web 域近重聚类预估 40–55 天——两个正交死因叠加：①验证精度
   （2048 字符前缀 Jaccard 对共享 boilerplate 的 CC 页面失效，模板对全量放行）；②聚类实现
   （纯 Python union-find）。裁定 web/code 以 S2 精度收官，决策记全 keep + 审计标记。
2. **逆转**（09-26，E-17b/E-18）：用户裁定必须工业级精度补齐。复盘发现两个死因各有标准
   解法：①换成全文 5-gram shingle Jaccard + MinHash 128 排列签名预筛（J_est<0.65 ⇒ 真实
   J≥0.8 概率 ~1e-5，验证成本降到每对一次 512B 签名比较）；②union-find 换
   scipy.sparse.csgraph.connected_components。止损时认定的"40 天"实际可压到小时级。
   教训：**止损评审必须把成本爆炸拆成"精度死因 / 实现死因 / 算法杠杆"三栏，拆完再决定**。
3. **二次止损**（10-01，E-31）：ndv2 code 域全文验证实测单桶 20–40h、全域 4–9 天（README
   估算失准 20×——长尾分布必须用 p99 最慢桶外推）。code 域纳入 fast track 豁免；
   **web 域 ndv2 于 10-10 14:06 启动**（48 verify workers，预期 3–5 天墙钟），是 v1.0-full
   S6 点火的硬前置。math 域跨源近重因 policy 字段缺失静默跳过，记为已知风险 R-1 接受。

### 3.3 质量打分：三轨并行与 fasttext 退役

**设计哲学**：打分不做硬丢弃，而是**全量排序**——每个文档都留下分数，谁进训练集由 §3.4 的
top-k 确定性选取决定。打分器本身留作 E3 消融素材。

三轨（S4，`configs/scorer.yaml`）：

| 轨 | 内容 | 角色 |
|---|---|---|
| upstream | 上游自带质量分（fineweb-edu 教育分等）归一化 | 生产轨 |
| fasttext | 自训二分类器（goldset v2：正样本 7 源 631k + 硬负 600k + 软负 200k） | **已退役**（见下） |
| goldrule BERT / `llm_distill` | 规则蒸馏的 BERT 打分器（holdout AUC 0.899） | 生产融合信号（研究子集保留全轨分数、缺失掩码语义与覆盖率） |

- 融合分（10-08 前全语料口径）：`composite_v2 = 0.25·upstream + 0.35·fasttext + 0.40·bert`
- 选取分（fast track 与正式版共用）：`hybrid_v2 = 0.5·llm_distill + 0.3·upstream + 0.2·rule`
  （不依赖 fasttext，fasttext 关死后公式不变）

**E-36：一个静默运行 10 天的配置陷阱**。09-28 配置写 `fasttext_track: off` 意图关闭 fasttext
轨，但 YAML 1.1 把裸 `off` 解析为布尔 `False`，字符串比较 `== "off"` 永不命中——**fasttext
轨从 09-28 起实际一直在跑**，与 E-25"标签语义错位"的原始意图相反。裁定**不改行为**（此刻改
成 off 会让重算分片与其余公式的历史口径不一致，才是真污染）；v1.0-full 全量重打分时统一
关闭。教训：**布尔开关禁用 on/off 字面量，用 true/false 或带引号 "off"**。

**fasttext 退役裁定**（10-10）：修复路径（重训 + 修负样本泄漏 + 跨机复现 + 复检）要 1–2 天，
只换来 AUC~0.85 的二分类器；而 goldrule v2 holdout AUC 0.899，hybrid_v2 里 fasttext 权重
0.35 的边际贡献存疑。裁定：退役转研究 backlog，full 版 score_field = hybrid_v2。
（注：判负样本取自训练池导致 CV 系统性偏乐观——重训任何打分器前必须先修泄漏，否则新 CV
不可信。）

### 3.4 配比：确定性选取 + 课程两段

**M1/M2/M3 三家族**（`docs/research/数据配比方案.md`，09-16 锁定）：

| 家族 | 取法 | 用途 |
|---|---|---|
| M1 | 各域统一 top-30%（零调参，web ≈88%） | E1 对照臂 |
| M2 | 历史均衡配方 1.44T（web 72.9% / code 12.5% / math 4.2% / chinese 5.2% …） | **已被 1T WSD 合同取代**；仅保留历史对照语义 |
| M3 | 各域统一 top-10% ≈ 490B 超质子集 | "更少更高质"验证 + E3 质量档消融 |

**top-k 的确定性实现**：S4 阶段为每域建分位表，S6 选取阈值 = `table[n−⌈top_k·n⌉]`，
同分者按 domain/doc_id 决胜——同一分数、同一选取，重跑一万次结果一致（宪法 4）。

**fast track 交付**（`mix_m3_e1_100b_fast`，S6 summary）：M3 档各域质量 top-10% 候选池
~363.8B tokens，按域配额确定性选取 **99,440,888,820 tokens（99.44B）**，课程两段：

- stage1（25%，24.86B）：upstream edu 低→高排序（NaN 殿后），跨域按 domain/doc_id 决胜
- stage2（75%，74.58B）：doc_id 字节序 = blake2b 均匀哈希序（确定性全局打乱）

每域另确定性切出 ~50M tokens held-out val（9 域，永不进任何训练臂）——E1/E2 的核心评测指标
就是各领域 val loss。

### 3.5 去污染与终训集质量审计

**S7 去污染**：训练文档与 5 个评测基准（MMLU / GSM8K / MATH / HumanEval / C-Eval 的题面 +
选项 + 答案）的词组索引共享 **≥1 个 8-gram** 即 drop。8-gram 比 GPT-3 的 13-gram、LLaMA 的
10-gram 更严格——0.5B 小模型对污染更敏感。索引 2,276,997 个 unique shingles。

**中文误杀实证与方案锁定**（10-10，专项审计 `data/audit_ceval_20261010/` + `DECISION_RECORD`
§A）：CJK 逐字 8-gram ≈ 8 个连续汉字，教学套话（"下列说法正确的是"）即命中——实测
cci3-hq 35.4%、skypile 17.6% 被 ceval 命中 drop，中文池 268M → 126M 腰斩。人工判读 280 篇
抽样的裁决：**277/280 = 98.9% 是误杀**（Wilson 95% CI [96.9%, 99.6%]），真污染仅 3 篇
（真题转载 dump，distinct 命中数 3/9/14）；误杀高频类型是法条/公文/科普/百科中的固定表述。
方案实测后锁定：**distinct 独立 8-gram 命中 ≥2 才 drop**（真污染 3/3 保留，整体救回 33.6%）；
命中计数用 distinct 不用 raw（重复段落虚增，样本最大 raw 185 vs distinct 111）。该方案在
v1.0-full 前对 cci3-hq / skypile / wikipedia-zh 三源重跑。多次命中优于单纯提高 n 的理由：
敏感性锚点不动、利用"成片复制"信号、漏检不对称落在真污染侧、可审计性不变。

**终训集质量门禁**：10-12 的旧审计（`data/audit_data_quality_20261012/report.json`）在 books
样本中发现 16 篇重叠全部位于 50k-shingle 截断窗之后，却错误地建议接受。10-19 复审将其
定性为 release-blocking false negative：全池已有 401,906 篇触发旧截断，不能从 300 篇
样本外推“仅 16 篇”。生产配置已改为全文扫描，查询按 2M shingles 分块控制内存；旧 S5/S6
发布随即撤销，必须在 decontam→S5→S6 重建并重新审计后才能放行。books 的 Project Gutenberg
装饰线仍作为格式噪声单独人审，不与评测污染问题混为一谈。

### 3.6 Tokenizer：自训 4 词表与 E0 评估

自训 byte-level BPE（BBPE，字节基 + 逐对合并的贪心算法，任何 UTF-8 序列可编码、无 OOV）三档
+ Unigram 对照。评估口径：200k 样本 × 25GB 加权语料（17 源抽样），分 8 域测压缩率
（bytes/token）、fertility、OOV、吞吐（`experiments/e0_tokenizer/results/e0_full_metrics.json`）：

| 词表 | 压缩率 (B/tok) | 备注 |
|---|---|---|
| yf-32k-bpe | 3.59（中文 3.91 / code 2.86） | |
| **yf-64k-bpe** | **3.86 = Qwen2.5-151k（3.92）的 98.5%，词表仅其 43%** | **E0 锁定为 pretrain-0.5b 预设** |
| yf-128k-bpe | 4.07（64k→128k 边际 +5.4%，embedding 参数 ×2，tied 下占总参 14%→27%） | 劣化，弃 |
| gpt2-50k（基线） | 2.36，中文崩到 1.41 | 印证词表必须匹配训练分布 |
| qwen2.5-151k（基线） | 3.92，code 3.35 优于 yf-64k 的 3.07 | 大词表的 code 子词优势记录在案 |

⚠️ 存疑：主代理口径 bytes/token 3.75；`experiments/e0_tokenizer/README.md` 记录为 3.86
（3.92 × 98.5% = 3.861，内部自洽）。以 3.86 为准。

Unigram 对照于 09-25 取消：HF tokenizers 的 unigram 训练器在生产规模两次 Rust 级崩溃
（16.7M 样本 TryFromInt 溢出 / 6M 样本 Internal 错误）+ 结构性无字节回退——"训练器工业化
成熟度"本身成为 BBPE 胜出的第三条实证。

**特殊 token 设计决策**（`tokenizer/README.md`）：`<|endoftext|>` / `<|unk|>` /
`<|im_start|>` / `<|im_end|>` / `<|think_start|>` / `<|think_end|>`（thinking 块边界）/
`<|tool_call_start|>` / `<|tool_call_end|>`（工具调用 JSON 边界）。
关键决策：**角色名（system/user/assistant/tool）不设为 special token**——tokenizers 会把
词表内 special token 在任意文本中匹配，预训练语料里的普通词 "user" 会被编成控制 token id，
污染 SFT 控制语义。角色名以纯文本跟在 `<|im_start|>` 之后。

## 4. 训练系统工程

### 4.1 模型实现

`model/` 从零实现 ZGCM-1 式混合注意力解码器（PyTorch 原生 + SDPA），规格逐条对齐
ZGCM-1 官方 Megatron fork（`model/README.md` 有逐项对照表）：门控滑窗 + 全注意力的层分布
（1-indexed，`layer_no % attention_every == 0` 为全注意力层；32 层 → 5 Full + 27 SWA）、
QGKV 融合门控（`x·sigmoid(gate)`，fp32，仅滑窗层）、QK-RMSNorm（eps 1e-6）、RoPE
rotate-half + rot_dim 截断（0.334·head_dim）、深度缩放初始化（o/fc2 用 0.02/√(2L)）。
全部变体开关化：18 个命名预设（tiny ~10.8M 到 large 7.2B 对齐 ZGCM 规格）×
attention_output_gate / qk_norm / rope_every 等布尔开关，外加 YuFormer 保留的 NoPE 研究开关
（rope_every，ZGCM-1 官方没有）。pretrain-0.5b：32L / h1024 / ffn2816 / 16Q8KV /
vocab 65536 / tied，~473M；参数计数有解析式单测断言（容许 2% 误差）。

### 4.2 chunked CE 的数值契约

**问题**：词表 64K 下，CE 前向物化完整 fp32 logits = `seq_len × vocab × 4B`，mbs4 即 4.3GB
（前向 1×mbs + 反向 1×mbs ≈ 2.1GB/每 mbs），micro batch size 被 logits 显存锁死——mbs8/16
直接 OOM。这是 MFU 上不去的第一根因（§5）。

**解法**（E-48，`model/src/yuformer_model/model.py:129` 分块融合交叉熵）：按词表切块
（`ce_chunk=8192`）计算 lm_head 投影并即时消费，从不物化全量 logits。数值契约（用户裁定
验收标准，代码注释为证）：

- **logsumexp / 目标 logit 累积全程 fp32**。bf16 累积在大词表下会溢出出 NaN；禁止
  "logits 转 bf16 存"的捷径。
- 每个样本的 label 只落在一个词表块内，`acc_tgt` 直接加法合并（无需 rescale）；
  `acc_lse` 用首块初始化后再 `logaddexp`，避免 `logaddexp(0, ·)` 引入 e⁰ 幻影项。
- 与 `F.cross_entropy(F.linear(hidden, w).float(), labels)` 数值等价——验收对拍容差
  loss 级 <1e-4（GEMM 分块的 bf16 舍入差异除外）。

**FSDP 根转发坑**：分块 CE 必须经 `forward` 转入（FSDP 根模块内部执行）——`_loss` 从
forward 外直接调会绕过 FSDP 的 forward 钩子路径，参数抓取与自动包装行为不一致。契约写进
docstring："必须由 forward 转入（FSDP 根内执行）"。

### 4.3 断点协议与数据契约

- checkpoint：模型 + 优化器 + 游标 + 四路 RNG（python/numpy/torch/cuda），tmp 目录 +
  `os.replace` 原子生效；`latest.json` 原子写最后落。
- `cursor.json` 存全局 `consumed_samples`（跨 stage 拼接样本序号），prefetch worker 按样本
  序号静态分配——任意断点重启消费序列逐位一致（smoke 断言 61..100 步 vs 连续跑逐位一致）。
- 数据物化（`materialize_tokens.py`）：S6 mix_order 顺序表 + s3 正文 → uint16/uint32 平铺
  token 流 + meta.json（total_tokens / sha256 / edge_sha256 / tokenizer_sha256）。
  `edge_sha256`（size + 首尾各 1MB）是启动期身份守卫。dtype 按词表自适应（≤64K 用 u2，
  128K 用 u4）。

## 5. MFU 工程与实测

### 5.1 实测矩阵（0.5B / 4K / 单卡 H800，2026-10-10 夜探针）

| 配置 | step 时间 | TFLOPS | MFU |
|---|---|---|---|
| SWA mask + Muon, mbs4（臂默认） | 1.80s | 73.8 | **7.5%** |
| 全注意力 (is_causal) + Muon, mbs4 | 1.24s | 102.4 | 10.3% |
| 全注意力 + AdamW, mbs4 | 1.02s | 124.2 | **12.6%** |
| mbs8/16 | OOM | — | fp32 logits 每 mbs 1.07GB（chunked CE 未实现） |
| **mbs8 + sel-mlp + chunked CE（冲刺组合）** | — | — | **11.5%**（臂默认 7.5% → +53%） |

**关键结论：自研栈单卡 MFU 现实值 10–13%，不是规模核算假设的 30–40%。** 分解：
Muon 比 AdamW 贵 ~2.3 点（NS5 正交化在小矩阵上占比高）；SWA mask 路径比 is_causal 贵
~2.8 点（bool mask 走 SDPA 通用路径）；mbs 上不去的根因是 fp32 logits（§4.2）。

### 5.2 瓶颈分析与 Plan C 冲刺

瓶颈排序：**fp32 logits（锁死 mbs）→ 激活显存（限制 mbs 上限）→ checkpoint 税
（full recompute 值不值）**。Plan C（10-11 起，8 卡独占窗口）逐项：

| 项 | 结果 |
|---|---|
| chunked CE（E-48，ce_chunk=8192） | 对拍 <1e-4 + smoke 回归全过；解除 logits 对 mbs 的锁死 |
| selective recompute mlp（E-49） | 省 ~7GB 激活（vs full-ckpt 的 30% FLOPs 税） |
| expandable_segments | 回收 5.7GB 碎片，共享卡上 mbs8 从 OOM 变可跑 |
| mbs8 组合实测 | **11.5%** |

剩余冲刺项（用户裁定顺序）：SWA 感知 flash 路径（4K 下等价全注意力 is_causal，实证 +37%；
正式版考虑 pretrain 预设 attention_every=1，SWA 留给 midtrain 256K 时代）→ bf16
residual/norm 卫生 → torch.compile（最后；数值口径改"同 build 内续训一致"）。

### 5.3 共享 GPU 的现实与门禁定义

**机器不是我们的**：GPU 为多租户共享卡——邻居租户 8 进程 × ~13GB/卡恒定占用、6 卡
100% util，单卡可用 ~66GB。MFU 数字全部带争用背景，microbenchmark 意义上的"干净 MFU"
在本机不存在。

**点火门禁（E-50 协议，10-11 用户裁定）**：

- ≥18% 必须是 **30 分钟真实训练的持续实测**，非 microbenchmark；测量窗口必须**覆盖邻居
  高负载时段**（≥24h 或含已知高峰）——低负载窗口测出的是"无争用上限"，不能做 45 天承诺。
- 每次测量记录邻居 util 基线：trainer 每 log 点把全卡 util/显存写入 metrics.jsonl 的
  `gpu_util_all` / `gpu_mem_mb_all` 字段，事后拟合争用系数。
- 45 天硬顶倒推只需 18%，不追求 35%（增量验收线：≥10% → ≥14% → ≥18% 即停）。

**10-13 时的结论**：门禁未达（冲刺组合 11.5%），当时 E1 尚未启动；1.5T @7.5% 的旧口径
对应 4 卡约 167 天，物理不可行。后续已将正式 base pretrain 预算收敛为 1T，并迁移到
TorchTitan v0.3.0；旧 MFU 数字只作为自研栈诊断证据，不能替代新训练栈的四卡 production
canary。当前硬门禁是升级驱动后完成 TorchTitan 数值一致性、恢复、reshard 与吞吐资格测试。

## 6. 踩坑实录（E-32~E-71 选编）

E-1~E-31 的历史见 `data-process/docs/FAILURE_LEDGER.md` 与 `docs/research/工程经验记录.md`
（单机大数据的内存经济学、静默失败、存储布局错配、标签语义错位、修复引入新 bug 五大元
规律在那里成型）。本章聚焦 10-01 之后收官冲刺期的坑，按四个家族归并。每条给
现象 → 根因 → 修法 → 教训。

### 6.1 规模首达家族：连环三颗雷，同一根因

**E-40 → E-45 → E-51：手工构造 Arrow 表的位宽契约**。mix_m3_e1_100b 选中 ~2 亿行首次
触发 `_doc_id_array` 的 int32 offsets 乘法溢出（6700 万行 × 32B/行 > 2³¹，ArrowInvalid:
Negative offsets）；修成 int64 后紧接着撞 parquet 单列块 2GB 上限（doc_id hex 串 32B/行 ×
1.5–2 亿行 = 4.8–6.4GB，row_group_size=50M 拆块解决）；**第三次撞墙才定位真因**——
`pa.string()` 的 offsets buffer 按 int32 解释，传入 int64 buffer 后 pyarrow 把 int64 当
int32 读，伪 "Negative offsets"（此前观察到的报错值 -32 = int64[0,32,64…] 被按 int32 对读取）。
正确修法：**int64 offsets 必须配 `pa.large_string()`**。

> 教训：Arrow 内存格式的 offsets 位宽是类型契约（string=int32 / large_string=int64），改
> 位宽必须连类型一起改；规模递增时存储层限制是一串连环雷（int32 → 2GB 列块 → 类型错配），
> 每拆一颗要按同口径复查同文件全部写出点。此前史上所有 mix 未达此规模，这一家族只有在
> "规模首达"时刻显形。

### 6.2 NFS/并发家族：共享文件系统上的幽灵

我们的宿主机是多租户共享、跨 pod 进程可见。**10-10 起，一个跑旧代码的"幽灵启动器"
进程持续存在**（启动源至今未查明），三次造成实损：

- **E-41（幽灵并发）**：幽灵拉起第二个 s6 driver，与编排器实例并发写同一日志（字节级
  交错可证），互 rmtree work 目录竞态崩溃循环一上午。修法：driver 重 stage 加 flock
  单实例锁 + 物化前 rmtree 尽力而为。教训：共享存储上的长作业必须有单实例锁，不能假设
  全系统只有一个启动器。
- **E-46（幽灵升级实弹）**：幽灵对标准名 mix_work/<mix> 的 strict rmtree 删掉了我们
  driver 的 collect npz，finalize 两次 FileNotFoundError。修法：work_root 加 `__v2`
  后缀——旧代码幽灵永不使用此名，本 pod 沿用则断点续跑保留。教训：跨 pod 无锁并发下，
  资源命名空间加版本后缀是最后的隔离手段。
- **E-70（幽灵覆写交付物，10-13）**：幽灵的坏代码 32k 物化用 `os.replace` 覆写了官方
  token_streams 流（21GB→0.93GB，meta 同步替换且自洽——**能骗过 size 守卫**，靠"该流
  应 ~21GB"的账目常识识破）。处置：四流 chmod 只读冻结（幽灵 replace → EACCES 弹回，
  物理免疫）+ 32k 私域重制。教训：**共享 FS 上一切耐久交付物落地即 chmod a-w——默认可被
  他进程覆写**；验收产物要有独立的账目核对（tokens 数级、字节数级）。

**E-44（NFS RPC 挂死）**：S6（64 workers）+ ndv2 web（48 workers）同时压 FS，主进程
18 分钟零进展 D 状态、rpc_wait_bit_killable——E-29 的"双 IO 大户互抢双输"在 NFS 饱和时
升级为 RPC 挂死。处置：ndv2 SIGSTOP 让路（3–5 天的活让 3 小时无感），S6 独占跑完
SIGCONT。挂死判定式：主进程 D 状态 + 零进展 + rpc_wait_bit_killable。

**E-56（fork 死锁）**：pyarrow/numpy 在父进程已起内部线程，fork 后子进程继承锁状态，
子进程消亡后主进程 futex 永久等待（18 线程全 futex_wait、1% CPU 假活）。修法：Pool 改
spawn 上下文（慢 ~20s/worker 换无锁遗产）。教训：**"fork 快"只在父进程无线程时成立；任何
import 过 pyarrow/numpy 的长驻进程用 Pool 一律 spawn**。

**E-54（tokenize 单进程瓶颈 + 潜伏越界）**：pass2a 单进程 encode，100B tokens 需 14–28h；
且边界切段只看文件不看行组（同文件跨行组并成一段，取错正文——生产从未跑到所以未爆，
合成端到端测试抓到）。修法：pass2a fork 池并行（后随 E-56 改 spawn）+ 边界
`(diff(sfi)|diff(srg))!=0` + 三级缓存。教训："顺序表→正文→token"链条必须先跑小数据端到端
再上千亿；合并键变更类 bug 在边界数据才显形。

### 6.3 配置语义家族：你以为的配置不是配置

- **E-36（yaml off→False）**：见 §3.3——名义关实际开 10 天，靠审计发现而非行为异常。
- **E-60（编排器是活代码）**：04:31 启动的编排器进程内存里是修复前代码，之后磁盘上的
  E-58/59/60 修复对它一律无效，val 步陷入"重跑整个 stage2→撞旧 bug→崩→重启"死循环
  4.5 小时，每轮残留 part 让 resume 逐轮递减制造虚假进展感。教训：**改了编排器/调度器
  代码必须杀进程重启；排查"修复无效"先查进程启动时间 vs 修复落盘时间**。
- **E-71（调度器 v2 双 bug，10-13）**：①队列项用 `/` 连接而拆分逻辑按 `:`（无冒号时取
  全串）→ group=arm=整条路径，env 路径翻倍、启动静默全败；②arm_done 的 glob 假设了
  不存在的目录层级 → 臂完成后永远不会被判 done。教训：字符串分隔符与拆分逻辑是配对契约，
  改一个必须 grep 全部消费点；**"启动静默失败"的调度器比不跑更坏——首批启动必须人工抽查
  metrics 落盘**。
- **E-34（工件跨环境 unpickle）**：goldrule_v2.joblib 在 numpy 2.x 环境 pickle，本 pod
  numpy 1.x 挂 `ModuleNotFoundError: numpy._core`；S4 闸门 fail-fast 正确拦截。教训：
  **模型工件入库时把训练环境（numpy/sklearn 版本）写进 artifact meta**；pip 二次安装会
  互相降版本，装完必须复核关键包。最终稳定栈：numpy 1.26.4 + scipy 1.15.3 +
  pyarrow 18.1 + sklearn 1.7.2 + pandas 2.2.3。

### 6.4 流程家族：守护、缓存与对账

- **E-32/E-33（分位表 per-worker 私载，七连挂死）**：S4 quantile pass 每 worker 私载全部
  分位表（web 域 22GB×9 域 ≈25GB）×160 workers ≈ 4TB > 2TB 物理内存，OOM killer 杀
  ForkPoolWorker → pool 挂死 → 守卫硬杀循环（10-05~10-08 七次）。修复分两层：先 mmap
  （实测 RSS 仍 16.4GB/进程——clean MAP_PRIVATE 页计入各进程 RSS，随机二分缺页拖尾 5% CPU
  纯 IO 等待），再紧凑化——按 rank 等距采样 2²⁵ 点/域，父进程构建一次（~28s）fork 后
  COW 共享（物理 ~150MB），误差上界 3e-8 ≪ 使用精度 1e-6。实测单分片 105 docs/s →
  16.8K rows/s。教训：**分位表/索引类大只读资产禁止 per-worker 实载；RSS 计账 ≠ 物理共享，
  mmap 不解决 per-process 计账；超大只读查找结构应压缩到 error bound ≪ 使用精度**。
- **E-38/E-39（守护生命周期）**：顺序调用的 health_monitor.py 实为常驻 daemon，永不返回
  → keepalive 循环阻塞在首次 hourly 分支停摆 3h；watchdog 又对已完成阶段的冻结日志持续
  "空杀"，掩盖对当前阶段的真实停滞检测。修法：被监督脚本必须单次执行型；停滞检测作用域
  与存活进程对齐。同日处置进程两次自匹配把自己杀了（E-10 违规者竟是账本作者）。
- **E-37（pyarrow 18 空 parquet 陷阱）**：升级后 `iter_batches()` 对 0-row-group 文件抛错
  （旧版静默空迭代），三层波及（quantile pass 46 空分片 / decontam 合法空分片契约 /
  decontam 自身读空输入），统一 `safe_iter_batches` 判空。教训：**环境升级后必须跑一遍
  空/边界输入冒烟；跨 stage 产物契约要在首次落地时就被下游 gate 验证**。

> 这一家族的总账：10-01 之后新增 28 条编号事故（E-32~E-71，含重号共 40 条），幽灵 pod
> 贯穿 10-10~10-13。其中"拦截型"比例显著上升——S4 闸门拦 E-34、合成端到端测试抓 E-54
> 越界、账目常识抓 E-70——前期的审计基建开始复利。

## 7. 关键决策记录

以下为已锁定的裁定（完整版见 `data-process/docs/DECISION_RECORD_20261010.md` 与
`experiments/EXPERIMENT_IGNITION_PLAN.md` §6）：

**① 中文 decontam：三源限定重跑已完成**——fast track 历史产物保持冻结；正式池对
cci3-hq / skypile / wikipedia-zh 按“distinct ≥2 命中才 drop”增量重算。decontam 救回约
21.557M 个旧误杀，S5 复用已认证 token 缓存并只编码新增文档；当前两阶段均零错误。

**② fasttext 退役**——不重训，full 版 score_field = hybrid_v2，fasttext 转研究 backlog。
理由：修复代价 1–2 天换 AUC~0.85，边际贡献存疑（goldrule v2 已有 0.899）。

**③ ndv2 web/code 已完成**——web 1,013/1,024、code 995/1,024 全文审计通过；决定采用
direct-edge dominance。S5 直接 join 排序 lookup，并用限制性 delta 过滤已认证紧凑表；
`ndv2_refresh` 降为兼容维护，避免无意义重写约 10TB S3。

**④ pretrain 全注意力 + SWA 留给 midtrain（附 fallback）**——E2 补 e2_swaoff 对照臂；
风险在案：pretrain 全注意力 → midtrain 引入 SWA = 架构切换（注意力熵分布突变），无现成
先例。**fallback：pretrain+midtrain 都全注意力**——0.5B@64K 全注意力开销仅 ~9% FLOPs
（256K 才是 ~36%），经济可接受；SWA 降级为纯研究模块。最终选型 = E2 SWA 臂数据 +
一次廉价 midtrain-pilot。

**⑤ full 版 S6 的 S5 血缘规则已实现**——逐分片 processing contract 绑定 lookup、决定表、
精度审计、decontam、tokenizer 和 context 配置。web/code 只删不增时走精确过滤 delta；中文
去污染放宽时只编码新救回文档。当前 12,829 个 S5 分片全部同代，正式 mix 门禁已通过。

## 8. 时间线与资源账目

### 8.1 日级时间线

| 日期 | 事件 |
|---|---|
| 09-15 | 项目骨架；19 源数据目录锁定（全 HF 路径经验证） |
| 09-16 | 初始规模假设：0.5B × 1.5T × 4×H800 ≤45 天；后续被实测吞吐否定并由 1T 合同取代 |
| 09-28 ~ 10-01 | S4 全池打分（fasttext 实际在跑——E-36 当时未知） |
| 10-01 ~ 10-08 | S4 quantile pass **挂死 7 次**（E-32，per-worker 私载 25GB × 160） |
| 10-08 | E-32/E-33 修复（compact 分位表，16.8K rows/s）；**20:14 S4 完成** |
| 10-09 | **23:48 S5 完成**（~3.14T tokens 计长分桶；墙钟 ~27h，有效 ~23h ⚠️ 两口径并存） |
| 10-10 | **02:33 S6 完成**（mix_m3_e1_100b_fast 物化）；14:06 ndv2 web 启动；MFU 夜探针（7.5/10.3/12.6%） |
| 10-10 ~ 10-12 | tokenize 连环排雷（E-51/54/56/60，幽灵并发贯穿）；Plan C 冲刺首夜（chunked CE + sel-mlp → 11.5%） |
| **10-12** | **11:25 DATA_READY**（v0.9-fast 全链收官；⚠️ 存疑：`data/DATA_READY.json` 当前 ts 为 10-13 15:27，疑似调度器轮询重写，原始落账时间以 11:25 为准）；live-fire 判绿 |
| **10-13** | **14:46 全量实验开火**（E0-Q2 四臂 + E2 多臂；step-1 loss 10.6–11.4 正常，无臂完成） |
| 10-20 | 冻结 1T WSD 合同：50B warmup + 800B stable + 150B anneal；context extension 独立记账 |
| 10-21 | pretrain 代码冻结回归 198/198 通过；E3 八臂 tokenization 继续，TorchTitan 资格测试等待驱动升级 |

### 8.2 机器与环节耗时

- **机器**：4–8×H800 80GB（共享租户，邻居 8 进程 × ~13GB/卡）；176 核 CPU；2TB 内存；
  NFS 共享存储（聚合带宽按 0.3–0.5GB/s 规划，E-9）。数据/ckpt 经 `YUFORMER_ROOT` /
  `SHARED_STORAGE_ROOT` 注入，不入库。
- **环节耗时**（fast track）：S4 全池打分 ~4 天（含 7 次挂死排雷）；S5 计长 ~23h
  （160 workers，3.14T tokens）；S6 物化 ~6h；tokenize 100B 并行化后 ~3h/段
  （pass2a spawn 池；此前单进程口径 14–28h）；ndv2 web 预期 3–5 天（48 workers，进行中）。
- 幂等设计的回报：全程 driver 被杀/崩溃 10+ 次，每次重启损失 <10 分钟（分片级断点）。

### 8.3 Token 守恒账（fast track 实验集）

```
15.4TB raw（19 源下载，17 源进管线）
  → 6,483,788,741 行（S1 out = S2 in = S3 in，守恒链闭合）
  → S2 精确去重 −2,825,291,013（43.6%）
  → S3 近重 −7.2M；S7 去污染 −43.2M（fast track 口径，含 ceval 误杀）
  → keep 3,610,096,074 行 / 3,144,594,732,717 tokens（S5 计长落账）
  → 各域质量 top-10% 候选 ~363.8B → 域配额确定性选取
  → mix_m3_e1_100b_fast = 99,440,888,820 tokens（stage1 24.86B 课程 + stage2 74.58B 打乱）
```

正式版采用 **1T final-tokenizer WSD**：50B warmup、800B stable、150B anneal。其精确配方须在
E1/E2/E3 结果与身份审计全部通过后冻结，再依次完成顺序物化、release、分词、全字节流审计和
跨阶段重放审计；当前 E3 tokenization 进行中，正式 1T 流尚未开始物化。4K→16K→64K 的
50B context extension 独立记账，不计入这 1T base budget。

## 9. 附录

### 9.1 关键文件索引

| 类别 | 路径 |
|---|---|
| 工程宪法 / 模块蓝图 | `AGENTS.md` |
| 失败账本（E-1~E-71） | `data-process/docs/FAILURE_LEDGER.md` |
| 事故详录（现象→根因→规则） | `docs/research/工程经验记录.md` |
| 决策记录（三项裁定 + 风险清单） | `data-process/docs/DECISION_RECORD_20261010.md` |
| 管线设计 | `data-process/docs/PIPELINE_DESIGN.md` |
| 全链审计（09-26） | `data-process/docs/AUDIT_20260926.md` |
| 验收清单 | `data-process/docs/ACCEPTANCE.md` |
| 实验点火规划 / MFU 实证 / 门禁 | `experiments/EXPERIMENT_IGNITION_PLAN.md` |
| 规模核算 / 配比方案 / 实验设计 | `docs/research/预训练规模核算.md`、`数据配比方案.md`、`实验设计.md` |
| 数据源目录（HF 路径全表） | `docs/research/预训练数据源目录.md` |
| ceval 误杀专项证据 | `data/audit_ceval_20261010/`（280 条逐篇裁决 jsonl） |
| 终训集质量门禁 | `data/audit_data_quality_20261012/report.json` |
| 管线守恒产物 | `data/processed/stage_s{0..6}_global_summary.json`、`audit_consistency_20260926.json` |
| 去重决策 | `data/processed/s3_decisions/*.summary.json` |
| token 流账目 | `data/token_streams/*.meta.json`（total_tokens / sha256 / tokenizer_sha256） |
| 训练配方 | `pretrain/configs/pretrain_base.yaml` |
| 架构与 ZGCM 规格对照 | `model/README.md` |

### 9.2 复现入口

```bash
# 数据管线（运行闭包，自包含）
cd data-process
PYTHONPATH=src python -m yuformer_data.scale.driver --stage s4 --workers 128 --out-root ../data/processed
PYTHONPATH=src python -m yuformer_data.scale.driver --stage s5 --workers 160 --out-root ../data/processed
PYTHONPATH=src python -m yuformer_data.scale.driver --stage s6 --workers 64  --out-root ../data/processed

# token 流物化（S6 → pretrain）
PYTHONPATH=pretrain/src python pretrain/scripts/materialize_tokens.py \
  --order <s6_root>/mix_order/<mix>_stage1.parquet --s3-root <s6_root>/s3 \
  --tokenizer tokenizer/tokenizers/yf-64k-bpe --out-dir <streams> --name stage1

# 训练（4 卡正式 / smoke / 断点续训）
scripts/launch.sh configs/pretrain_base.env
/usr/bin/python3 scripts/smoke_train.py --gpu 0
RESUME_ITER=10000 scripts/launch.sh configs/pretrain_base.env   # 与 latest.json 不符即 exit 4

# 实验调度（DATA_READY 门自动放行；幂等可重跑）
bash experiments/_common/experiment_scheduler.sh
```

---

*本报告与代码同步生长（宪法 8）。E1/E2/E3 实验结论、pretrain 全程与 posttrain 结果将在
对应阶段完成后回填。*
