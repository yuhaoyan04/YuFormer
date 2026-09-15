# ZGCM-1 项目认知框架（YuFormer 复现调研报告）

> 调研对象：`/data2/songxinshuai/yanyh/llm-from-scratch/ZGCM-1`（上游 https://github.com/zgcagi/ZGCM-1.git，HEAD `e9aade6`）
> 技术报告：arXiv 2609.13356；模型 `zgcagi/ZGCM-1-7B`；数据 `zgcagi/ZGCM-1-Data`
> 本文档是 YuFormer 复现工程的认知基线，覆盖组织架构、全链路工作流、核心技术与复现要点。

---

## 1. ZGCM-1 是什么

**7.39B 参数 dense 解码器模型，从零训练，面向数学推理与 Agentic Search**。中关村学院出品。

| 维度 | 数值 |
|---|---|
| MATH-500 / AIME 2026 / HMMT 2025 | 97.13% / 75.00% / 70.42%（7B-8B 档 14 项推理基准综合排名第一） |
| Agentic Search | WebWalkerQA 63.09%、BrowseComp 19.43%、GAIA text-only 42.52% |
| 上下文 | 256K（262,144 tokens），支持 thinking / direct-response 双模式 |
| 训练量 | Pretrain ≈4.19T → Midtrain ≈600.51B → SFT 19.46B tokens |
| 效率 | 混合注意力在 256K 下 3.94× 吞吐提升；架构+FP8+Muon+归一化综合 time-to-loss 提升约 4.2× |
| 硬件 | 192×H100，FP8 hybrid + Muon，实测 585 model TFLOP/s/GPU |

**三大技术支柱**：① 门控滑窗+全局混合注意力；② FP8 + Muon 优化器；③ 课程式数据配比 + 渐进长上下文。

---

## 2. 仓库总体架构与组织哲学

```
ZGCM-1/
├── data-process/   # AI-native 数据治理流水线（纯 Python 标准库，零依赖）
├── pretrain/       # 预训练 4.19T（vendored Megatron-LM + configs + launcher）
├── midtrain/       # 中期训练 600B，16K→64K→256K（与 pretrain 同构）
├── sft/            # 监督微调（fork Megatron-LM，含 Megatron-RL）
├── rl/             # 纯文档（RL 协议描述，实际代码在 sft/ 内）
├── assets/ · README.md · LICENSE(MIT)
```

### 2.1 核心组织原则："运行闭包"式阶段目录

- **每个一级目录自带 code+configs+docs+launcher，目录间零耦合**（各 TRAINING_STAGE.md 明示"本目录的运行闭包独立于其他一级目录"）。可整体拷贝一个阶段目录改造。
- **外部资产一律经 `SHARED_STORAGE_ROOT` 环境声明**：tokenizer、indexed 数据集、checkpoint、预构建 venv（megatron + torchrun + TE + FA3）均不在仓库内。
- **自包含 env 文件 = 单一真源**：一个 `.env` 声明该阶段全部超参/路径/拓扑，launcher 只做"env→CLI"的机械翻译（五段分组：distributed/model/training/parallel/data，逐行可核对）。diff 两个 env 即得阶段差异。
- **vendored 而非 submodule**：Megatron-LM（commit `eba2eaf71d34274c1ca0a59ac5e57a6bf53eb732`，megatron-core 0.18.0rc0，2026-04-07）以纯目录拷贝入库，ZGCM 改动直接内联在树内，docs/SOURCE_PROVENANCE.md 逐项声明改动清单与参考运行时。

### 2.2 ZGCM 对上游 Megatron-LM 的自定义改动（全部收敛为极小集）

1. **`attention_output_gate_only_swa`**（架构核心）：仅滑窗层的 core attention 输出施加 `x * sigmoid(gate)`，gate 由融合 QKV 投影额外输出（Q/G/K/V 布局）。位置：`megatron/core/transformer/transformer_config.py:242-246`、`attention.py:329-340,1289-1524`。
2. **Muon QKV 多候选拆分**（优化器-模型耦合点）：27 层门控（QGKV）+5 层非门控（QKV）形状不同，`emerging_optimizers.py:133-160,275-324` 按梯度实际形状自动匹配布局，分别 Newton-Schulz 正交化后拼接。
3. **GLM5.1 tokenizer/模板契约 + 预打包 SFT 数据集 + THD 变长 FA3**（sft 目录，经 patch-on-import 注入而非硬改上游）。
4. 数据加载类改动（deterministic sequential order / sample offset / prefetch——注意：部分 env 开关在释出代码中无消费点，实际依赖上游 `--no-data-shuffle` 与样本级断点续训）。

其余全部复用上游能力：`--window-size` + `--window-attn-skip-freq` 滑窗调度、Muon 本体（NVIDIA Emerging-Optimizers v0.2.0）、FP8 hybrid delayed scaling、QK-LN、CP a2a 等。

---

## 3. 全链路工作流总览

```
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ data-process │ → │   pretrain   │ → │  midtrain   │ → │     sft     │ → │     rl      │
│  JSONL 治理  │   │  4.19T @16K │   │ 600B 16→64  │   │ 19.46B @256K│   │ GRPO 混合   │
│ 清洗/去重/配比│   │ 课程式数据   │   │   →256K     │   │ think+agent │   │ math+code   │
│ 确定性混合   │   │ Muon+FP8    │   │ MDP轨迹数据  │   │ THD打包     │   │ 可验证奖励   │
└─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘
   输出: JSONL        输出: ckpt        输出: 256K ckpt    输出: SFT ckpt     输出: 最终模型
   +manifest         (--load 续训)   (--pretrained-ckpt   + run_manifest
                       载权重重置状态)
```

**数据格式契约链**：data-process 输出治理后 JSONL（决策/审计字段）→ 训练环境侧完成 tokenize+打包，物化为 Megatron indexed dataset（.idx/.bin）→ pretrain/midtrain 单前缀或 768 前缀加权 blend → SFT 为三路对齐 indexed（tokens/targets/cu_seqlens，`megatron_sft_prepacked_v1`）→ RL 由环境 rollout 在线产数据。

**Tokenizer**：GLM-5.1 tokenizer（HF 格式加载），词表 154,880（=1210×128），untie embedding。仓库外资产。

---

## 4. 模块深度解析

### 4.1 data-process：AI-native 数据治理流水线

**定位**：不是分布式全量处理工程，而是**记录级数据治理的契约层 + 参考实现**（零第三方依赖、纯标准库、单机流式可跑）。大规模抽取/分片/token 化/indexed 写入由训练环境补齐。

**三层核心抽象**：
- `models.py`：`Sample`（全管线统一载体：原始数据+规范字段+decision/reasons/flags/buckets/metrics）+ `Decision` 三态（**keep/review/drop——可疑数据打标待审而非静默丢弃**，决策只降不升）
- `core.py`：`Step/Pipeline` 流式引擎，DROP 短路，内存友好
- `adapters.py`：`DatasetAdapter`（field_map 改名 + hooks 数据集专属修复）——**AI-native 落点：AI 检查新数据集后生成独立 adapter/profile，主干保持小而稳定**

**管线组装**：`类别归一步骤 + COMMON_PREFIX（NFKC 归一/完整性 min16字符/来源与许可证/密钥脱敏）+ 类别专属步骤 + COMMON_SUFFIX（质量策略→精确去重 SHA256→可选 SimHash 近重→token 计长分桶→质量分桶）`。8 类数据：code/web/agentic/instruction/math/reasoning + pdf_ocr/general_text。

**代表性类别逻辑**：
- agentic：工具调用-观察**闭环配对校验**（每个 tool_call 必须有 observation）、轨迹 outcome 分桶（verified/failed/weak，失败降 REVIEW）
- math：多模态缺失检测（引用图片但无 image_text→REVIEW）、verifier 通过分桶
- reasoning：控制伪影检测（`<|system|>` 等外来模板 token）、lineage_hash 保留
- code：FIM triplet 完整性、vendor/min.js 过滤

**核心算法**（每个都独立可测试、可搬运）：
- **SimHash64 + 带状索引近重去重**：5-gram 特征 blake2b 投票指纹，4 band×16 位倒排，距离≤3 无假阴性，免 O(n²)
- **质量策略**：源感知阈值链（source_key→source→dataset→category→default），消费上游分类器分数（quality_mean/edu_score 等），分类器在管线外
- **确定性配比**：最大余数法精确配额 + blake2b 稳定哈希选取与交错
- **课程式长度混合（midtrain 核心）**：互斥前缀嵌套 mix16⊂mix64⊂mix256，短样本对更长混合保持资格、每 ID 只进一个 mix——严格对应 16K→64K→256K 训练课程
- **FFD packing**（保留 cu_seqlens 与 target 边界）、LPT 负载均衡、微分区
- **Agent 数据构造**：遗留文本函数调用经 Python `ast` 解析为结构化 tool_calls；`<think>` 分离到 reasoning_content；MDP 状态转移→(s,a,s') 链的监督格式

**输出契约**：治理后 JSONL（含 record_sha256）+ summary（per-step counters）+ manifest（从序列化输出重建）。防污染：源数据带 `-decontam-v2` 后缀、评测集去污染是管线强制第 6 步。

### 4.2 pretrain：4.19T tokens，两阶段，16K 恒定

| | Stage 1（0.99T） | Stage 2（3.20T） |
|---|---|---|
| 数据 | 课程式：通用语料按词汇复杂度升序，代码与数学独立交错 | 离线全局打乱物化（seed 20260603），运行时保序消费 |
| 顺序 | `NO_DATA_SHUFFLE=1` 保序（课程数据） | 同左（打乱已完成在离线） |
| 衔接 | 全新训练 | `--load` Stage1 目录 + `RESUME_ITER=79473` marker 校验（fail-fast 卫兵，防跑错 ckpt）+ `--override-opt-param-scheduler` |

**统一配置**（两阶段架构/优化段逐行一致）：TP2/PP1/CP2(a2a)+SP → DP48；mbs2/gbs768/seq16384；Muon（momentum 0.9、spectral scaling、5 步 Newton-Schulz、blockwise TP）+ Adam 标量参数（β 0.9/0.95）；**恒定 LR 2e-4**（无 decay）、wd 0.1、clip 1.0、seed 6198；FP8 hybrid + delayed recipe、amax history 1024、`--fp8-param-gather`；无激活重计算；RoPE base 5M、rotary_percent 0.334；`--untie-embeddings-and-output-weights`、cross-entropy fusion、per-token loss。

**课程数据哲学**：数据顺序的确定主义——课程与打乱都在离线物化时完成，运行时顺序消费 + 样本级 consumed offset 续训，任意断点重启样本序列逐条一致。

### 4.3 midtrain：600.51B tokens，三阶段渐进扩上下文

| | 16K 阶段 | 64K 阶段 | 256K 阶段 |
|---|---|---|---|
| tokens | 180.0B | 240.0B | 180.51B |
| 并行 | TP2/PP1/CP2 | TP2/PP1/**CP4** | TP2/**PP2**/CP4 |
| gbs / mbs | 768/2 | 192/1 | 48/1 |
| RoPE base | 5M | 5M | **10M** |
| 重计算 | 无 | selective | full+uniform |
| 阶段初始化 | `--pretrained-checkpoint` 载 pretrain 最终权重（重置优化器与 iteration，与 pretrain 的 `--load` 续训语义有意区分） | 同左，载 16K ckpt | 同左，载 64K ckpt |

**关键设计**：
- **恒定 tokens-per-step ≈12.58M**（gbs 768→192→48 补偿 seq 变长），LR 调度与 loss 跨长度阶段可比
- **无 RoPE scaling/插值**：长度扩展纯靠"提高 RoPE base（仅 256K 阶段 5M→10M）+ 继续训练"
- **数据**：768 前缀加权 blend（`--data-args-path` 文件传递规避 argv 限额）：长样本池(≤64K, 512 shards, 权重2) + 短样本池(≤16K, 256 shards, 权重1)；后期阶段保留大量短样本；在线 shuffle
- **内容**：代码/数学/知识/推理/指令 + **agentic 数据：交互轨迹重组为 MDP 状态-动作转移**，对单步决策施加监督（非整条轨迹 LM）
- **有实验支撑的路线选择**：direct-256K（30B）vs staged（10B@64K→20B@256K）对比实验保留在 docs，staged loss 1.16 < direct 1.19
- LR：cosine 2e-5→2e-6，每阶段 1% warmup（sample-based 调度）

### 4.4 sft：19.46B tokens，通用+Agentic 联合，THD 256K 打包

**框架**：fork Megatron-LM（同 commit `eba2eaf`），ZGCM 修改经 **patch-on-import 运行时注入**（`training_launchers/stable_candidate/pretrain_gpt_force_mcore_save.py` 用 importlib 替换 SFTTokenizer/SFTDataset 类 + runpy 执行 pretrain_gpt.py），上游树保持可 diff、可升级。

**数据格式**（生产配置为索引预打包三路对齐）：`{prefix}_tokens.{bin,idx}` + `{prefix}_targets.{bin,idx}` + `{prefix}_cu_seqlens.{bin,idx}`（`megatron_sft_prepacked_v1`）。**模板、thinking span 定位、loss mask、packing 全部离线完成**，训练期零模板开销。

- **assistant-only loss**：逐 token `IGNORE_INDEX=-100` 掩码；glm51 渲染用"前缀差分"定位 assistant span（先渲染到上一轮的长度，再渲染含本轮的长度，差分即 span）——对任意 jinja 模板通用
- **thinking/direct 混合**：assistant turn 含 `reasoning_content` 或 `</think>` → thinking 模式渲染（`enable_thinking` 传给 chat_template.jinja），同序列内两类混合
- **THD 变长打包**：每条 packed record ≈255K tokens，内部几十条对话以 cu_seqlens 边界拼接、position_ids 逐对话重置、FA3 varlen 注意力使桶内对话互不可见（无 cross-contamination）
- **混合配方**（token 计）：QCORE 通用核心 98.63% + THINK 长思维链 1.37%（1083 条、平均 246K tokens/条）+ TOOL 0（发布配置未开）

**训练配置**：TP8/PP1/CP1、mbs1/gbs48、192 GPU；LR 1e-4→1e-6 cosine **无 warmup**、wd 0.01；FA3 + full recompute + FP8；10 个 token-based epoch（76,325 条 packed record，15,467 iters）。

**启动器工程**（stable_candidate 三文件契约）：`runtime.env`（机器契约）/ `zgcm.conf`（实验契约，唯一真源）/ `launch.sh`（推导+校验）。亮点：
- **迭代数从数据集元数据推导**（train_tokens/train_records），conf 中期望值只做身份守卫——杜绝"配置与数据不符"的静默错误
- **Smoke 测试不破坏 LR 视界**：`SMOKE_TRAIN_ITERS` 只截断步数，LR_DECAY_ITERS 保持全量计划
- run_manifest.json 全量快照、rank0 就绪屏障、零字节 ckpt 检测、启动前 import/函数级自检门、nvidia-smi 遥测

### 4.5 rl：GRPO 混合数学/代码/通用（文档 + sft 内的 Megatron-RL）

**框架**：`rl/` 是纯文档（协议描述）；实际代码是 sft/ 内附带的 **Megatron-RL**（NVIDIA 原生 RL 后训练库），Agent/Environment（`get_prompt/get_reward` 接口）与 Trainer 解耦。

**协议**（rl/RUNNING.md 五步）：① 初始策略 rollout 估计题目难度、剔除高频已解题 → ② 分组采样 → ③ 域特定奖励 → ④ 零方差组动态采样替换 → ⑤ 组相对优势 + GRPO 更新。

- **奖励**：数学二元答案正确性（math_verify，阶梯 shaping：1.0/partial/format/negative）；代码=可执行测试通过比例（**仓库未附实现，需自写 RewardOnlyAgent**）；无效/截断回答不给正奖励
- **正则**：参考策略 KL（k3 估计）+ 轻度长度惩罚
- **GRPO 细节**：DAPO 式上下非对称裁剪、重要性采样修正、组内标准化优势
- **ZGCM 参数**：actor LR 2e-6；组规模 24×16 或 384×8（最多 3072 轨迹）；**生成 65,536 / 总上下文 98,304** tokens；混合域经 WeightedMultiTask YAML 权重 + 最大余数法配额

---

## 5. 模型架构规格汇总（复现的唯一真源）

| 项 | 值 | 配置来源 |
|---|---|---|
| 层/隐藏/FFN | 32 / 4096 / 11008（SwiGLU，无 bias） | pretrain env:32-34 |
| 注意力 | GQA 32Q+8KV，head dim 128 | env:35-37 |
| 混合注意力 | `--window-size 127,0` + `--window-attn-skip-freq 6`：层号%6==0（第 6/12/18/24/30 层）为全局注意力，其余 **27 层为 128-token 滑窗** | env:43-44 |
| 门控 | `--attention-output-gate --attention-output-gate-only-swa`：仅 27 个 SWA 层输出 `x*sigmoid(gate)`，gate 融合在 QKV 投影（QGKV 布局） | env:46-47 |
| 归一化 | RMSNorm eps 1e-6 + QK-LayerNorm | env:42,45 |
| 位置编码 | RoPE，base 5M（pretrain/16K/64K）→10M（256K），rotary_percent 0.334，`--no-rope-fusion` | env:39-41 |
| 词表 | 154,880，untie 输入/输出 embedding | env:38 |
| 参数量 | ≈7.39B（embedding 2×4096×154880 ≈1.27B） | — |

**注意**：架构没有独立"ZGCM 模型文件"——是 Megatron-Core GPTModel + 标准 layer spec + 上述两个自定义开关的组合，完全由 env/CLI 驱动。

## 6. 技术栈与依赖

- **运行时**（SOURCE_PROVENANCE.md）：Python 3.12.3、PyTorch 2.7.0a0（NGC 25.04 线，CUDA 12.9）、TransformerEngine 2.15.0.dev0、FA 2.7.3 + FA3 3.0.0；H100（`NVTE_CUDA_ARCHS=90`）
- **Muon 来源**：外部包 NVIDIA-NeMo/Emerging-Optimizers **v0.2.0**（pyproject git 锁定，版本敏感）
- **sft 包管理**：uv + uv.lock；requires-python >=3.12
- **data-process**：零依赖纯标准库（仅 .zst 可选 zstandard），`PYTHONPATH=src` 即可跑
- 启动：纯 torchrun（无 slurm 依赖），launch_train.sh 218 行做"校验→env→CLI 翻译→exec torchrun tee 日志"

## 7. 工程组织亮点（YuFormer 直接借鉴）

1. **运行闭包式阶段目录** + 自包含 env 单一真源 + env→CLI 机械翻译
2. **三态决策 + 全程审计**（keep/review/drop；reasons/flags/record_sha256；manifest 从输出重建）
3. **AI-native 数据治理**：共享管线锁死契约，AI 按数据集生成 profile/adapter——规则长尾从主干剥离
4. **数据顺序确定主义**：课程/打乱离线物化 + 保序消费 + 样本级断点 + fail-fast marker 卫兵
5. **元数据驱动迭代推导** + 期望值身份守卫；smoke 不破坏 LR 视界
6. **patch-on-import 定制**：上游树可 diff、可升级
7. **离线模板化/打包/掩码**：训练期零开销，数据清洗与训练消费经格式契约解耦
8. **恒定 tokens-per-step** 跨长度阶段；**有实验支撑的路线决策**（direct vs staged 对比保留）
9. **吞吐可解释**：理论 FLOPs 精确计入门控项与逐层 SWA/全局分布
10. **防污染是一等公民**：源名 decontam 标记、lineage、契约强制去污染步骤

## 8. 复现路线建议（小规模）与已知缺口

**已知缺口/坑**：
- `MEGATRON_GPT_DATASET_SEQUENTIAL`/`PREFETCH_FACTOR`/`SAMPLE_OFFSET`/`GLOBAL_SHUFFLE_SEED` 在释出树内无消费点（惰性开关）；Stage2 的 TRAIN_ITERS=333787 是**累计值**（实际新增 254,314）
- 门控使逐层 QKV 形状不同：checkpoint 须 non_homogeneous_layers 保存，导 HF 需按层区分；与 fused_single_qkv_rope 互斥
- 代码执行奖励 agent、通用指令遵循奖励、"Locate" 数据组件、GLM5.1 tokenizer 资产本身均未随仓库发布
- tokenizer/indexed 数据/megatron venv 都是仓库外资产，需自行制备

**小规模复现路径**（YuFormer 规划参考）：
1. data-process 可 1:1 复现（零依赖、契约层设计直接搬）
2. 架构可从零实现（hybrid SWA+global、门控、GQA 都是清晰的小组件）；Muon 用开源实现
3. pretrain/midtrain/sft 可按比例缩：层数/隐藏维/词表/seq 长度可缩，保真项=数据课程思想、恒 tokens-per-step、LR 调度形态、assistant-only mask、THD 打包
4. RL 用 GRPO+可验证数学奖励起步（math_verify 路径完整可搬），代码奖励需自写

## 9. 关键文件速查

| 内容 | 路径（相对 ZGCM-1/） |
|---|---|
| 数据管线入口 | `data-process/src/zgcm_data_pipeline/`（core/models/adapters/pipelines/quality/dedup/training.py） |
| 预训练配置 | `pretrain/configs/stages/{01_pretrain_1t_sequential,02_pretrain_3p2t_global_shuffle}.env` |
| midtrain 配置 | `midtrain/configs/stages/{01_seq16k,02_seq64k,03_seq256k}.env` |
| 启动器（pretrain/midtrain 相同） | `pretrain/scripts/launch_train.sh` |
| midtrain 数据清单 | `midtrain/scripts/build_midtrain_data_args.sh` |
| 门控注意力 | `pretrain/Megatron-LM/megatron/core/transformer/attention.py:329-340,1289-1524` |
| Muon QKV 拆分 | `pretrain/Megatron-LM/megatron/core/optimizer/emerging_optimizers.py:133-160,275-324` |
| SFT tokenizer/glm51 | `sft/megatron/core/tokenizers/text/libraries/sft_tokenizer.py:112-211` |
| SFT 数据集 | `sft/megatron/training/datasets/sft_dataset.py` |
| SFT 启动器 | `sft/training_launchers/stable_candidate/{runtime.env,zgcm.conf,launch.sh}` |
| GRPO loss | `sft/megatron/rl/rl_utils.py:1716-1807` |
| 数学奖励 agent | `sft/examples/rl/environments/math/math_agent.py:49-118` |
| 阶段文档 | `pretrain/docs/PRETRAIN_PHASES.md`、`midtrain/docs/MIDTRAIN_STAGES.md`、`sft/docs/sft/`、`rl/{README,ENVIRONMENT,RUNNING}.md` |
