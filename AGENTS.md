# YuFormer 开发指南 (AGENTS.md)

> 2026-11-23 当前事实：1.516600064B 正式八卡 pretrain 已于 11 月 15 日启动，4K stable 运行中；总预算 966254395392 labels 包含 WSD、16K 与 64K，上限 33 天。当前状态见 `report/metrics/latest_pretrain_status.json`、`report/TECHNICAL_REPORT.md` 和半日复核。下文早期的“未启动”、四卡和0.5B条目属于历史范围，不能覆盖当前生产事实。
>
> 用户于 2026-11-23 明确授权维护并更新 `git@github.com:yuhaoyan04/YuFormer.git`，包括提交/推送最新源码、轻量实验数据、动态技术报告，以及清理确认过期的本地文件；这项授权覆盖旧交接记录中“不提交/不推送”的限制。已通过本机隔离 GitHub CLI 以 `yuhaoyan04` 登录，首次完整快照 `b41084f` 已推送并核验远端 main；后续每次发布仍须核对实际远端 SHA。聊天连接器未切换，本仓库推送使用独立 HTTPS 凭据。不强推覆盖未知远端变更。
>
> 新任务先读 `docs/engineering/PLAYBOOK.md` 和相关 `INCIDENTS.md`，复用事故反例及验证方法。新异常追加症状、证据、已知/未知根因、修复、验证和迁移规则。完整历史已索引在 `docs/engineering/incidents.json`，不把历史实验或计划中的SFT/RL写成已完成训练。
>
> 用户要求每12小时复核并保持会话；后台守护负责心跳，不持续空转采样。半日复核后可用 `tools/export_pretrain_report.py` 更新轻量指标/README/技术报告，`tools/plot_pretrain_metrics.py` 更新曲线。原训练、租约generation和冻结依赖不得因仓库整理而改动。清理先运行 `tools/clean_stale_artifacts.py` 生成计划，再对已授权且通过保护检查的目标使用 `--apply`；保留小型故障证据。

> 2026-11-13 当前用户要求：使用全部八张 H800 做分布式预训练，目标约 30 天，启动工期上限 33 天。配置为 world_size=8、microbatch=8、grad_accum=16，保持 4,194,304 tokens/update。此前 E1 waiver 继续有效；数据、数值、恢复与八卡吞吐门禁不可跳过。四卡配方与 canary 已归档并标为 STALE。详见 `report/EIGHT_GPU_PRETRAIN_PLAN_20261113.md`。

> 2026-11-14 模型选型更新：按用户“允许接近 30 天，优先选择更大模型”的要求，当前候选已改为专用 `pretrain-final` 配置的 **1.516600064B / 18L / H2560 / FFN6912 / 40Q10KV / attention_every6**，峰值 LR=1e-4，`cudnn_flex_swa`，200 步持续实测预计 32.432922 天（duty=.9）。候选配置已发布，正式训练仍等待完整新门禁。下文 0.5B 是历史研究基线，不得覆盖当前专用模型配置或据此复用旧 GPU 资格。当前事实见 `report/ACTIVE_PRETRAIN_OPERATIONS.md` 和 `report/GPU_MODEL_SIZE_SWEEP_20261114.md`；原始数据 release 保留身份，采用训练专用冻结，最终仅用 `--reuse-prepared-production` 入口点火。


> 本文件是 YuFormer 项目的开发宪法，指导 AI 协作者与人类开发者统一思路。
> 调研基线见 [`docs/research/ZGCM-1-认知框架.md`](docs/research/ZGCM-1-认知框架.md)。
> 当前事实与门禁见 [`report/PROJECT_AUDIT_20261018.md`](report/PROJECT_AUDIT_20261018.md)。

---

## 0. 项目定位

**YuFormer 不是 ZGCM-1 的复现，而是站在其肩膀上的全链路 LLM 工程 + 创新项目。**

- **复用其骨架，不照搬其血肉**：ZGCM-1 的代码组织（运行闭包式阶段目录、自包含配置、env→CLI 翻译、确定主义、三态审计）是我们理想的工程范式，整体继承；具体实现按 YuFormer 的规模、数据与目标重写。
- **全链路闭环**：数据收集 → 数据处理 → 数据配比 → Tokenizer → 模型架构 → Pretrain → Midtrain → SFT → RL → 评测 → 技术报告，每一环都要真正跑通、可回溯、可写进简历。
- **创新是第一公民**：每个模块必须明确"参考做法"与"YuFormer 创新空间"，纯搬运不可接受——要么在方法上改进，要么在工程上更优，要么做出有报告价值的研究结论。
- **配套技术报告**：`report/` 与代码同步生长，最终形成可发表/可展示的技术报告，作为项目产出的佐证。

---

## 1. 工程宪法（不可妥协原则）

以下八条从 ZGCM-1 提炼，是 YuFormer 所有模块的硬约束：

1. **运行闭包式阶段目录**：每个一级目录（data-process/、pretrain/、midtrain/、sft/、rl/ 等）自带 code+configs+docs+launcher，目录间零耦合，可整体拷贝改造。外部依赖一律经环境变量声明，不在代码里硬编码路径。
2. **自包含配置 = 单一真源**：每个训练阶段用一个配置文件（`.env`/`.yaml`）声明全部超参/路径/拓扑，launcher 只做"配置→CLI"的机械翻译。diff 两个配置即得阶段差异，禁止隐藏默认值链。
3. **外部资产经环境路径声明**：tokenizer、数据集、checkpoint、预构建 venv 等不在仓库内，统一经 `YUFORMER_ROOT`/`SHARED_STORAGE_ROOT` 等环境变量定位，仓库只存代码与配置。
4. **确定主义贯穿**：所有随机性用稳定哈希（blake2b）+ 显式 seed 驱动；数据顺序在离线物化时确定，运行时保序消费 + 样本级断点续训；输出 JSON `sort_keys` 序列化。任意断点重启结果可复现。
5. **三态决策 + 全程审计**：数据处理用 keep/review/drop 三态（可疑打标待审而非静默丢弃）；每条输出携带 reasons/flags/metrics 与稳定哈希（record_sha256），manifest 从序列化输出重建，可抽样复核、可做 release 审计。
6. **元数据驱动 + fail-fast**：训练迭代数从数据集元数据（token 数/记录数）推导，配置里的期望值只做身份守卫；跨阶段续训用 marker 校验（RESUME_ITER）拦截"跑错 checkpoint"类灾难；启动前做存在性/import 自检，失败即退出。
7. **对上游最小侵入 / 可 diff**：若 fork 第三方框架（如 Megatron-LM），自定义改动以独立文件 + 运行时注入（patch-on-import）或最小内联 diff 形式收敛，上游树保持可 diff、可升级。改动清单写入各目录 `docs/SOURCE_PROVENANCE.md`。
8. **文档即代码**：每个目录配 `README.md` + `docs/`；训练阶段配阶段说明文档（PHASES/STAGES）+ 框架来源文档（SOURCE_PROVENANCE）；关键决策与实验结论写进文档而非口头记忆。

---

## 2. 仓库布局

```
YuFormer/
├── AGENTS.md                # 本文件（开发宪法 + 模块蓝图）
├── README.md                # 项目门面
├── .gitignore
├── data-process/            # 数据治理流水线（收集/清洗/去重/打分/配比）
├── tokenizer/               # 分词器训练、评估与词表研究
├── model/                   # 模型架构定义（从零实现，非 vendored）
├── pretrain/                # 预训练
├── midtrain/                # 长上下文中期训练
├── sft/                     # 监督微调（通用 + Agentic）
├── rl/                      # 强化学习（GRPO + 可验证奖励）
├── eval/                    # 评测框架与 benchmark 对接
├── experiments/             # 研究实验（按问题组织：e0_tokenizer/e1_scaling/e2_architecture/e3_data_mix/e4_curriculum，每实验=运行闭包）
├── report/                  # 技术报告（LaTeX/Markdown，与代码同步）
└── docs/
    └── research/            # 调研文档（ZGCM-1 认知框架、规模核算、实验设计等）
```

每个阶段目录内部统一遵循：`README.md` + `docs/` + `configs/` + `scripts/` + 源码 + `tests/`。

---

## 3. 模块开发蓝图

> 每个模块按统一结构陈述：**核心思想 / ZGCM-1 参考 / YuFormer 思路 / 创新空间 / 待决问题**。
> 待决问题在逐模块迭代时锁定。

### 3.1 data-process：数据治理流水线

- **核心思想**：数据是模型的天花板。治理流水线锁死输入 schema、输出契约、决策状态与审计字段；数据集专属规则以"adapter/profile 插件"形式挂载，主干保持小而稳定。大规模抽取/分片/token 化/indexed 写入由训练环境侧补齐，本模块是"契约层 + 参考实现"。
- **ZGCM-1 参考**：零依赖纯标准库；三态决策（keep/review/drop）+ Step/Pipeline 流式引擎（DROP 短路）；共享前缀（NFKC 归一/完整性/来源脱敏）+ 类别专属步骤 + 共享后缀（质量→去重→计长分桶）；SimHash 带状索引近重、最大余数法配额、互斥前缀嵌套的课程式长度混合（mix16⊂mix64⊂mix256）。
- **YuFormer 思路**：继承三层抽象（Sample/Decision → Step/Pipeline → Adapter/Profile）与共享前后缀管线骨架；输出治理后 parquet + summary + manifest 契约。当前正式源范围的治理池为 2.447T reference tokens，主候选为约 1T 的 50B warmup + 800B stable + 150B anneal；最终预算仍须由 E1 scaling 裁定。只有 release gate 全部 PASS 后生成的 content-addressed dataset id 才能进入正式训练。配比必须有 source floor/cap，并用 proxy-model mixture ablation 验证；不能把跨源不可比的 upstream score 直接称作 lexical complexity。
- **创新空间**：① 数据质量分类器自研（轻量 fasttext/规则 + 可选 LLM 打分），而非只消费上游分数；② 配比做课程式 + 在线自适应混合的对比研究（产出报告结论）；③ 引入主动学习式数据筛选（按 loss/困惑度挑样本）；④ 多语言数据治理扩展。
- **待决问题**：质量打分器选型与训练方案（fasttext 二分类 vs 回归打分，正样本用 fineweb-edu 高分段）；top-k 选取的排序键（单分 vs 多目标）；跨源近重去重的规模方案（MinHash LSH 外部内存）；课程式配比的具体权重与排序键。

### 3.2 tokenizer：分词器

- **核心思想**：tokenizer 决定信息密度与训练效率上限，是数据与模型之间的契约锚点。词表大小、压缩率、特殊 token 设计（thinking/工具调用边界）需与下游训练目标协同。
- **ZGCM-1 参考**：直接用 GLM-5.1 tokenizer（HF 格式，词表 154,880，untie embedding），不自训；data-process 与训练侧均以"可注入接口"对待 tokenizer，不绑定实现。
- **YuFormer 思路**：**自训 BPE tokenizer**（这是创新点之一，ZGCM-1 直接复用现成的）。用 YuFormer 自有语料训练 byte-level BPE，研究词表大小（32k/64k/128k）对压缩率与训练效率的影响；正式控制 token 已统一为 `<|think_start|>/<|think_end|>` 与 `<|tool_call_start|>/<|tool_call_end|>`，保存时绑定 chat template hash；产出 tokenizer 评估报告（BPB、压缩率、吞吐、各语言覆盖）。
- **创新空间**：① 词表大小消融实验（报告核心卖点）；② 针对代码/数学的子词优化（LaTeX、代码缩进保留）；③ thinking 专用 token 体系设计；④ 与 chat_template.jinja 的协同设计（`enable_thinking` 参数化，复用 ZGCM-1 的前缀差分定位 assistant span 思路）。
- **待决问题**：训练语料覆盖范围；词表大小初选；是否多语言；特殊 token 集合最终定义。

### 3.3 model：模型架构

- **核心思想**：架构是效率与能力的根。ZGCM-1 用"门控滑窗 + 全局混合注意力"把 256K 训练吞吐提到全注意力的 3.94×。YuFormer 应理解并可选实现该高效架构，同时做架构层面的研究性创新。
- **ZGCM-1 参考**：32 层 / hidden 4096 / GQA 32Q+8KV；27 层 128-token 门控滑窗 + 5 层全局（每 6 层一个全局）；门控 `x*sigmoid(gate)` 仅作用于滑窗层，gate 融合在 QKV 投影（QGKV 布局）；RMSNorm + QK-LN；RoPE rotary_percent 0.334；无独立模型文件，=Megatron-Core GPTModel + 两个自定义开关。
- **YuFormer 思路**：**从零实现模型架构**（PyTorch，非 vendored Megatron），这是"从零做 LLM"的核心学习价值。先实现标准 dense Transformer（GQA + SwiGLU + RMSNorm + RoPE）跑通；再作为研究模块实现混合注意力（滑窗+全局）与门控变体，做消融对比。架构定义集中在 `model/`，训练框架侧通过配置选用。
- **创新空间**：① 混合注意力的层间分布研究（全局层位置/比例对长上下文的影响，报告结论）；② 门控机制的变体（如门控施加范围、激活函数选择）；③ 与其他高效注意力（如线性注意力、稀疏注意力）的对比研究；④ 极小规模下验证架构可扩展性（缩放律初步验证）。
- **待决问题**：模型规模（✅ 锁定 `pretrain-0.5b` 预设：32L/h1024/ffn2816/16Q8KV/vocab65536，~473M，**tied embedding**——untied 占 0.5B 档 30% 参数不划算，列为 E3 消融）；门控+滑窗（✅ 已实现并按 ZGCM 官方规格对齐：1-indexed 层分布、QGKV 融合门控仅滑窗层、QK-RMSNorm、rot_dim 截断、depth 缩放初始化，全部开关化可消融）；E3 架构消融（固定 ~0.5B 总参数变 depth/width/MLP 比例，见规模核算文档）；训练框架（pretrain/ 阶段实现 FSDP 训练循环）。

### 3.4 pretrain：预训练

- **核心思想**：预训练结论必须同时绑定数据、代码、模型、优化器和运行环境身份；数据顺序、恢复语义与评测口径都要可复现。
- **ZGCM-1 参考**：4.19T tokens 两阶段（0.99T 课程式 + 3.20T 全局打乱），恒 16K；TP2/PP1/CP2；Muon（5 步 Newton-Schulz）+ Adam 标量参数；恒定 LR 2e-4 无 decay；FP8 hybrid；离线物化数据顺序 + 样本级断点续训 + fail-fast marker 校验跨阶段衔接。
- **YuFormer 思路**：正式后端锁定 TorchTitan v0.3.0 + PyTorch 2.14 + FSDP2/DTensor + DCP，默认 AdamW/BF16 与 token-based WSD；Muon、FP8 是匹配数据与 FLOPs的独立消融。训练 run identity 必须绑定 Git/source tree、完整配置、模型配置、tokenizer、PASS data release 与后端版本；checkpoint 必须覆盖逐 rank RNG、数据游标、scheduler，并通过异步分片保存、SIGKILL 恢复与 2↔4 卡 reshard canary。现有 FSDP1 trainer 只作为兼容与数值对拍后端。
- **创新空间**：① 课程学习排序键研究（词汇复杂度 vs 困惑度 vs 教育等级，报告对比）；② Muon 优化器在小规模的可复现性验证（vs AdamW，报告结论）；③ **数据配比响应实验**（固定小模型算力，扫描领域配比，产出 mixture response 曲线与最优配比表——支撑最终 pretrain 配比决策）；④ **缩放律实验**（isoflop 等算力点拟合 L(N,D)，外推 45 天预算的最优 (N*,D*)——支撑最终规模决策）；⑤ 训练动力学监控与异常检测工程化。
- **待决问题**：WSD stable/decay token 比例、Muon 匹配 FLOPs 的矩阵、FP8 数值门禁，以及 4K→16K→64K 各阶段 token 预算；详细裁定见 [`docs/research/训练栈选型与迁移.md`](docs/research/训练栈选型与迁移.md)。

### 3.5 midtrain：长上下文中期训练

- **核心思想**：长上下文靠"渐进扩展 + 恒定 tokens-per-step"而非一步到位。ZGCM-1 用 16K→64K→256K 三阶段、gbs 768→192→48 补偿 seq 变长保持每步 ~12.6M tokens，仅末阶段提高 RoPE base（5M→10M），不用插值，并有实验支撑"渐进优于直达"。
- **ZGCM-1 参考**：600.51B tokens；每阶段 `--pretrained-checkpoint` 载权重重置优化器（与 pretrain 的 `--load` 续训语义区分）；cosine LR 2e-5→2e-6；重计算从无→selective→full；768 前缀加权 blend（`--data-args-path` 文件传递）；保留大量短样本于后期阶段。
- **YuFormer 思路**：正式阶梯锁定为 4K→16K→64K；机器契约强制恒定 tokens/update、25% 短样本 replay、阶段间 weight-only 加载并重置优化器、FlashAttention 2/3，以及逐阶段长能力与短能力双门禁。base checkpoint 与长文本 dataset identity 未锁定时配置返回 `BLOCKED`。
- **创新空间**：① RoPE base vs RoPE scaling（NTK/线性插值）的对比研究（报告结论）；② 扩展阶梯数与各阶段 token 量的消融；③ 长上下文评测体系的建立（needle-in-haystack、长文档 QA）；④ 重计算策略的成本-收益量化。
- **待决问题**：16K/64K 各阶段 token budget 的 pilot 修正；RoPE base vs YaRN/DCA 消融；长数据配比；是否引入 agentic MDP 轨迹数据。

### 3.6 sft：监督微调

- **核心思想**：通用+Agentic 联合训练，thinking/direct 双模式混合；模板、assistant-only loss mask、变长打包全部离线完成，训练期零开销。
- **ZGCM-1 参考**：19.46B tokens；THD 变长 FA3 打包（每条 ~255K tokens，cu_seqlens 边界拼接、position_ids 逐对话重置）；glm51 模板"前缀差分"定位 assistant span 做 token 级 `IGNORE_INDEX=-100` 掩码；thinking 判定（含 `reasoning_content` 或 `</think>`）；配置三分（runtime.env 机器契约 / zgcm.conf 实验契约 / launch.sh 推导校验）；迭代数从数据元数据推导，smoke 不破坏 LR 视界。
- **YuFormer 思路**：SFT 数据契约、规范化工具轨迹渲染与 `tokens/targets/cu_seqlens` 三路对齐打包器已经落地；assistant-only 掩码直接由规范渲染分段产生，避免字符到 token 的前缀差分歧义。下一步接 TorchTitan block-causal THD、配置与 release manifest；先跑通通用指令，再扩充 Agentic 数据。
- **创新空间**：① 打包效率与变长注意力实现的研究（自研 vs 框架自带）；② thinking/direct 混合比例对推理能力的影响（报告）；③ 多轮工具调用轨迹的格式设计与 loss 策略（工具观察是否参与 loss）；④ 数据质量过滤与去重的 SFT 专用版。
- **待决问题**：SFT 数据来源与规模；是否做 Agentic 工具调用（初期可仅通用指令）；序列长度；打包粒度；thinking 模式是否启用。

### 3.7 rl：强化学习

- **核心思想**：用可验证奖励的 GRPO 让模型从"会答"到"答对"。数学用二元答案正确性、代码用测试通过比例；组内标准化优势 + 动态采样（零方差组替换）+ 参考策略 KL 正则。
- **ZGCM-1 参考**：Megatron-RL（fork 附带）；GRPO + DAPO 式非对称裁剪 + 重要性采样修正；五步协议（难度预筛→分组采样→域奖励→零方差替换→组相对优势更新）；actor LR 2e-6；生成 65,536/上下文 98,304；`rl/` 一级目录是纯文档，实际代码在 sft/ 内。
- **YuFormer 思路**：正式主线已锁定 verl + SGLang，先跑数学 exact-answer GRPO/DAPO；版本 pin 与 DAPO 复现 pin 分开记录。rollout 更新前门禁已覆盖完整分组、零方差组、reward/长度、KL、importance ratio、clip 与截断统计。下一步接模型 adapter、隔离 verifier 与动态采样，再扩展到代码执行奖励。
- **创新空间**：① 奖励 shaping 的消融（纯 0/1 vs 阶梯式 partial/format/negative）；② KL 系数与长度惩罚的影响；③ 动态采样策略的改进；④ GRPO vs 其他 RL 算法（PPO/ReMax）的对比（报告）；⑤ 代码执行沙箱的工程化（安全性、超时、资源限制）。
- **待决问题**：首轮 group size、生成长度、难度预筛阈值，以及数学 verifier 数据与 held-out 评测的隔离方案。

### 3.8 eval：评测

- **核心思想**：无评测则无改进。建立覆盖能力维（数学/代码/知识/指令遵循）与上下文维度（needle、长文档）的评测体系，与训练阶段对齐，支撑报告中的所有结论。
- **ZGCM-1 参考**：14 项推理基准（MATH-500、AIME、HMMT 等）+ Agentic Search（WebWalkerQA、BrowseComp、GAIA）+ Binary Function Search；thinking/direct 双模式评测；温度 1.0、top-p 1.0、mean pass@1 over 32 runs。
- **YuFormer 思路**：正式通用评测锁定 lm-evaluation-harness v0.4.13；发布门禁绑定 backend/model/tokenizer/chat-template、dataset revision、task/prompt、seed、逐样本输出与去污染报告，禁止把 `limit` smoke 分数写入正式表格。长上下文、代码执行和训练期逐域 BPB 使用隔离 runner，但统一结果契约；agentic 评测作为进阶。
- **创新空间**：① 评测自动化的工程化（一键跑全基准、结果聚合报告）；② 训练过程中的动态评测（checkpoint 采样评测，画能力曲线）；③ 评测去污染与数据集重叠检测（与 data-process 的去污染联动）。
- **待决问题**：冻结 benchmark suite 与逐任务 dataset revision；采样参数；评测频率。

### 3.9 report：技术报告

- **核心思想**：报告与代码同步生长，是项目产出的佐证，也是创新的载体。每个模块的研究结论（消融、对比、缩放）都沉淀进报告。
- **ZGCM-1 参考**：arXiv 技术报告，含架构、训练配方、评测结果、资格实验（direct vs staged 长上下文对比）、效率分析。
- **YuFormer 思路**：`report/` 用 Markdown（或 LaTeX）组织，分章节对应模块；每个模块的消融实验与结论直接写入；最终整合成可展示的技术报告。
- **创新空间**：① 报告即项目叙事——把"从零做 LLM"的全过程决策与教训写成有教学价值的文档；② 所有创新点（词表消融、课程排序、架构变体、RL 奖励 shaping）都有报告章节支撑。
- **待决问题**：报告语言（中文/英文）；格式（Markdown vs LaTeX）；发布渠道。

---

## 4. 开发顺序与依赖

模块间依赖（箭头表示"被依赖方需先有雏形"）：

```
data-process ──┐
               ├─→ tokenizer ──┐
model ─────────┤               ├─→ pretrain ──→ midtrain ──→ sft ──→ rl
               │               │                  │          │       │
               └───────────────┴──────────────────┴──────────┴───────→ eval
                                                                        │
                                                                        └─→ report
```

**建议推进顺序**（每步先出可审核的设计，锁定后再实现）：

1. **规模与目标**：0.5B 模型 / pretrain 主候选约 1T tokens / 单机 8×H800 / 4K base + 16K→64K context extension；最终规模由 isoFLOP 拟合、吞吐 canary 和 PASS 数据 release 共同裁定。
2. `data-process` 骨架（Sample/Decision/Pipeline + 共享前后缀 + JSONL/manifest 契约）
3. `tokenizer`（自训 BPE + 评估报告 + 特殊 token 设计）
4. `model`（标准 dense Transformer 跑通，再叠混合注意力/门控研究）
5. `pretrain`（课程式配比 + 配置驱动 launcher + 断点协议）
6. `eval`（早建，每个 checkpoint 可评测）
7. `midtrain`（长上下文阶梯）
8. `sft`（打包器 + assistant-only mask + 通用指令先跑通）
9. `rl`（GRPO + 数学可验证奖励起步）
10. `report` 整合收尾

---

## 5. 工程约定

- **代码风格**：Python，类型注解，无注释除非复杂逻辑（遵循 opencode 约定）；模块用 `pyproject.toml` 声明元数据与依赖。
- **测试**：每个模块配 `tests/`，核心算法（去重、配比、打包、掩码）单测，断言确定性；提供 `scripts/smoke_test.py` 端到端冒烟。
- **配置**：训练阶段用自包含 env/yaml，launcher 机械翻译到 CLI；配置里写 seed 并溯源。
- **提交**：提交信息简明、匹配仓库风格；不提交数据/checkpoint/secrets（`.gitignore` 已覆盖）；不主动 push/commit 除非明确要求。
- **环境**：正式预训练/SFT 环境锁定 Python 3.12、PyTorch 2.14、CUDA 12.6、TorchTitan v0.3.0（并对齐 torchvision 0.29.0 / torchao 0.18.0 / Triton 3.8.0），以兼容本节点 R535；RL 使用独立的 verl 环境，禁止原地混装并须另行通过相同宿主机的 GPU qualification。当前 Python 3.10/PyTorch 2.2 alpha 环境只用于数据处理和兼容测试。依赖按模块 lock/container 固化，运行时版本写入 attempt provenance 与 `docs/SOURCE_PROVENANCE.md`。
- **文档**：目录配 README + docs；阶段配 PHASES/STAGES 文档；框架来源与改动配 SOURCE_PROVENANCE。

---

## 6. 迭代协议

1. **审核本文件**：逐模块审视"核心思想 / YuFormer 思路 / 创新空间 / 待决问题"，标记同意/修改/补充。
2. **逐模块对话**：按 §4 顺序，一次一个模块深入对话，锁定待决问题（规模、选型、参数），形成该模块的详细设计（落到目录结构与接口签名）。
3. **实现**：设计锁定后再动手写代码，每个模块实现后跑 smoke test + 单测，再进入下一模块。
4. **回溯**：git 支持历史回溯；每模块完成后提交一个里程碑，便于回退。

---

## 附：与 ZGCM-1 的关键差异（YuFormer 立场）

| 维度 | ZGCM-1 | YuFormer |
|---|---|---|
| 定位 | 生产级开源大模型 | 全链路学习 + 创新工程 + 报告 |
| 规模 | 7.39B / 4.19T / 192×H100 | 0.5B / 约 1T（三段质量课程候选）/ 8×H800 单机 |
| 模型架构 | vendored Megatron + 2 开关 | **从零实现**（研究导向） |
| Tokenizer | 复用 GLM-5.1 | **自训 BPE + 词表消融** |
| 优化器 | Muon + FP8 | AdamW 起步，Muon 作研究模块 |
| 框架 | Megatron-LM 全套 | 按需自研/轻量接入，优先可读可学 |
| 创新 | 工程效率 + agentic search | 方法消融 + 报告结论（每模块） |
