# YuFormer 全链路技术审计与工业化路线（2026-10-18）

> **HISTORICAL SNAPSHOT / SUPERSEDED（2026-10-20）**：本文记录 10 月 18 日发现的问题及整改起点，
> 其中“正式 1.5T”预算已经失效。当前权威合同是 1T final-tokenizer WSD：50B warmup、
> 800B stable、150B anneal；另计 50B context extension。当前状态与运行时门禁见
> `report/PRETRAIN_STACK_AUDIT_20261020.md`、`pretrain/configs/pretrain_base.yaml` 和
> `experiments/e4_curriculum/configs/e4_1t_wsd_context.yaml`。下文 1.5T 数字仅用于还原审计当时状态。

## 1. 审计结论

YuFormer 已经不是 README 所写的“初始化阶段”。项目已完成大规模数据下载与 S0–S6
处理链、四套 tokenizer、约 0.5B dense Transformer、单卡预训练器、E0/E1/E2 配置与一批
性能/恢复实验；三条 E0-Q2 训练臂已在本次审计中停止并标记失效。数据、模型与训练代码中有不少扎实的工程工作，
尤其是确定性 doc_id、分片 summary、原子 checkpoint marker、token stream 校验、分块 CE、
选择性重计算与运行故障记录。

当前状态仍不满足“正式 1.5T pretrain 点火”或“工业级全链路已闭环”的标准。主要原因并非
模型结构落后，而是存在会改变训练样本或实验结论的正确性问题：

1. 审计时的四卡数据游标会让不同 rank 在相邻 step 重复取样，同时跳过其他样本；新代码已修复，
   但仍需四卡 GPU canary 验收。
2. 审计时的 E0-Q2 验证逻辑仅按 vocab size 过滤，未按 tokenizer 指纹过滤；64K/128K 臂正在解释
   其他 tokenizer 的 token id，32K 臂的唯一可读“验证流”又是自身训练流。当前 val loss
   不能用于选 tokenizer；新 loader 已隔离 tokenizer，但验证资产仍需重新物化。
3. 现存 token stream 将文档裸拼接，没有 EOD/EOS；新物化器已修复，正式流仍需重物化。
4. 99.44B 数据是带 web/code 近重豁免的 fast track；仓库自己的 24 项 ACCEPTANCE 门禁
   尚未签收，不能把这批数据称为最终高质量 pretrain 数据集。
5. 旧 S6 manifest 没有上游指纹；新代码已给 val/S6 增加级联快照。checkpoint 已补逐 rank RNG
   与滚动保留，仍缺分片异步保存和拓扑变更 reshard。

因此，旧 E0-Q2 checkpoint 只保留作吞吐取证和 checkpoint 迁移测试，不能据此宣布
32K/64K/128K tokenizer 胜负，也不得继续恢复。正式多卡实验应在 D0/I0 门禁后重新开始。

### 1.1 本轮整改状态（2026-10-18）

审计后已经完成以下代码整改：

- 多卡样本映射改为 `resume + step × (world × samples_per_rank) + rank × samples_per_rank + local`，
  world=1/2/4 的连续覆盖、rank 间不重叠和 resume 映射已有单测，PrefetchLoader 也覆盖了物理索引测试。
- validation 只读取 `val_*`，并同时检查 vocab 上界与 `tokenizer_sha256`；训练 stages 也必须声明并
  共享同一个 tokenizer 指纹。旧 E0 的跨 tokenizer val 不再会静默进入评估。
- token materializer 默认给每篇文档追加 `<|endoftext|>`，边界策略、token id 与计数写入 meta；
  part cache key 已包含边界策略，避免复用旧的裸拼接缓存。
- checkpoint 保存逐 rank RNG，保留旧 `rng.pt` 读取兼容；新增 latest-N + 周期 milestone retention，
  正式 `pretrain_v1/base` 已配置为最近 2 个 + 每 100k iter，E0 配置未启用自动删除。
- C-Eval 去污染改为至少两个不同 shingle 命中；decontam 的 input/config/index 任一身份变化都会
  级联重算。val/S6 manifest 已绑定逐分片 summary 内容、数据产物身份与全局控制产物 SHA-256。
- chunked CE 已支持标准 `ignore_index=-100`，并与 PyTorch CE 做 loss/gradient 对拍，可用于
  assistant-only SFT masking。
- pretrain 完整测试现为 71 passed；其中包含 2-rank Gloo checkpoint、world=1/2/4 sampler、
  tokenizer 隔离、EOD 物化、run identity 与旧 checkpoint 兼容测试。
- 新增不可变 `run_identity.json`、逐次 attempt provenance 和 metrics step 单调门禁；非空旧目录
  不允许被新任务静默认领。E0 三臂的失效原因、指标哈希、step 回退次数与最后 checkpoint 已写入
  `data/experiments/e0q2/INVALIDATED_20261018.json`。
- 正式训练后端锁定 TorchTitan v0.3.0 / PyTorch 2.14 / FSDP2 / DCP；版本 lock 与 preflight
  已落地。当前环境因 Python 3.10、Torch 2.2 alpha 和 NVIDIA driver 535 未达到门槛而 blocked。
- 30 个 E1/E2 配置已从隐式 FSDP1 + Muon + constant LR 迁移为显式 TorchTitan、PASS
  release/EOD、AdamW + cosine；数据/release 环境值保留不可运行占位符，LR 仍须 pilot 后锁定。
  失效 E0-Q2 与混杂变量的旧 E2 配置已移入 `_holding`；旧 scheduler/autolaunch 固定 exit 4，
  审计时发现的遗留 scheduler 进程也已停止。
- tokenizer 内嵌 Jinja chat template 与 SFT Python 渲染器已对工具调用 JSON 做逐字节对拍；
  中文、引号和 HTML-sensitive 字符不再产生训练/推理 token 漂移。SFT 契约测试已覆盖该不变量。
- 下载器已增加显式 `enabled` 语义和 fail-closed 退出码；C4/PG19 默认 `--all` 不再被误运行，
  任一 `partial/failed/empty` 会返回 2。Dolma Books 三个本地文件已重新全量哈希并与配置一致。
- fastText 的 repeated-fit native 状态会在完整测试进程中触发 NaN；训练/CV 已改为每次拟合使用
  独立子进程并显式 seed，再只读加载模型，避免折间与任务间状态污染。
- S4 正式选择分发现 upstream 缺失重归一实现错误：旧代码把缺失当有效 0 分，抽样中
  1,377/1,761 行受影响。当前代码已恢复缺失轨权重重归一，并用旧分解析变换避免重跑
  65 亿行模型推理；17 源组件级精确复算 1,761 行、0 mismatch。goldrule 的 numpy/sklearn
  运行时已锁版并在模型加载前 fail-fast；修复后的分位表已全量重建并以版本化多文件原子发布。
  新表闭合 12,829 分片、6,485,760,234 行、9 域，build id 为 `8d2ee7cdb1d7620e`。
- RL 新工程轨更新为 verl v0.9.1 + SGLang v0.5.20，并与 DAPO 严格复现 commit、TorchTitan
  预训练环境分开；新增 lm-eval v0.4.13 的评测发布契约，拒绝 partial run、浮动数据 revision、
  缺少逐样本产物或去污染证据的分数。
- 历史 `pretrain/tests/verify/verify_report.json` 因混入 scope 外 V6 OOM 失败已明确标记
  `INVALIDATED`。正式 runner 现在先清空旧分片，并记录 suite scope、pytest exit status 与失败项；
  锁定环境完整重跑前不再把旧 Markdown 的 24/24 当机器 PASS。

仍未闭环的是数据资产重物化与数据质量门禁：现存 token stream 仍是旧版裸拼接，三个 tokenizer
仍需分别生成 held-out val；web/code near-dedup 刷新、DCP/异步分片
checkpoint、拓扑变更恢复也仍是正式点火前置项。

本轮末尾还发现一个正在发生的运行事故：64K 的同一 `out_dir` 同时被本机任务与另一共享
存储客户端写入。`metrics.jsonl` 交错出现约 19.1K step（可见 8 张 GPU）和约 16.6K step
（可见 4 张 GPU）的记录；旧 checkpoint 代码允许较慢任务把 `latest.json` 从高 iteration
写回低 iteration。当前 64K 指标与 checkpoint 目录因此不能视为单一实验产物。本轮已停止
三个本机 torchrun 父进程及其 6 个孤儿 worker，没有删除 checkpoint；32K 记录 4 次 step 回退，
64K/128K 分别记录 603/602 次。新代码已增加跨进程 `run.lock`、checkpoint iteration 单调守卫、
不可变 run identity，以及 `--no-resume` 对已有输出目录的拒绝逻辑。

## 2. 项目真实进度

| 环节 | 已完成 | 尚未完成 / 不能宣称完成 |
|---|---|---|
| 数据获取 | 19 个候选源的下载记录；当前 lineage 明确纳入 17 源并排除 C4/PG19；原始数据约 15.3TB | 数据许可证与可训练用途没有形成逐源、逐版本的机器可读交付门禁 |
| 数据处理 | S1 清洗、S2 exact dedup、S3/ndv2、decontam、S4 打分、S5 计长、S6 配比均已在十亿文档量级运行；中文 decontam、web/code ndv2 与 S6 TOCTOU 修复已生产收敛 | 1.5T 主训练 mix 与 token stream 尚未物化 |
| 数据产物 | 当前 S5 keep 3.106B documents、2.531T tokens，12,829/12,829 分片且零错误；正式 100B mix 为 94,235,054 篇 / 99.4446B tokens，D1–D4 审计通过 | 历史 fast mix 仍绑定旧 S5；1.5T 主训练 mix 尚未物化 |
| Tokenizer | 32K/64K/128K BPE、64K Unigram；unk=0、roundtrip=1、利用率高；新物化器已追加 EOD 并记录边界策略 | E0 下游比较设计无效；各 tokenizer 缺少自己的 held-out val token stream；旧训练流尚未重物化 |
| 模型 | dense Transformer，GQA、RMSNorm、SwiGLU、QK-Norm、RoPE、gated SWA/full attention、tied embedding、chunked CE | 64K SDPA 会物化平方级 mask；可用的 FlashAttention 长上下文环境尚未固化；TorchTitan BaseModel 适配尚未完成 |
| Pretrain | FSDP1 兼容 trainer、Muon/AdamW、rank-safe sampler、逐 rank RNG、run identity/lease；正式后端已锁定 TorchTitan | 新环境未就绪；BaseModel/Grain 适配、DCP async、拓扑变更恢复、四卡 kill-9 canary 尚未过门 |
| Scaling | E1/E2 formal configs 已改为 TorchTitan + PASS release/EOD + AdamW/cosine，并补 LR pilot/isoFLOP 设计 | E0-Q2 已失效；E1/E2 尚无正式结果且被 D0/T0/I0 阻断；E3/E4 只有骨架 |
| Midtrain | 4K→16K→64K 阶梯契约已机器化：恒定 tokens/update、weight-only/reset、短样本 replay、FlashAttention 和双评测门 | 历史原子报告因 scope 污染失效；base/data identity、TorchTitan adapter、锁定环境重跑与正式长训练尚未闭环 |
| SFT / RL / Eval | SFT 数据契约、工具轨迹渲染、assistant-only THD 打包已实现；RL 后端锁版及 rollout 诊断已实现；eval 后端锁版及结果发布门禁已实现 | SFT/RL trainer adapter、数据 release、verifier 隔离、正式 eval suite 与全阶段结果尚未闭环 |
| 文档与版本 | 研究笔记、失败账、决策记录、技术报告草稿较丰富；根 README 已按审计快照更新 | 绝大多数核心实现仍未提交；无统一 lock/container；本次只在当前环境补装了声明的 `tokenizers` 依赖 |

## 3. 当前数据快照与门禁判断

### 3.1 S5 与 S6 的真实数字

`stage_s5_global_summary.json` 显示 12,829 个分片、0 shard error，保留 3,105,637,476
篇文档，共 2,530,570,633,678 tokens。领域 token 数为：

| 领域 | S5 tokens | 正式 100B mix tokens | 主要问题 |
|---|---:|---:|---|
| web | 1.937T | 86.000B | ndv2 direct-edge 删除已生效；正式 mix 已重物化 |
| code | 264.013B | 5.500B | ndv2 direct-edge 删除已生效 |
| chinese | 101.932B | 3.500B | 三中文源 C-Eval 双命中阈值已重算 |
| academic | 88.164B | 1.200B | peS2o 约 68%，arXiv 约 32% |
| math | 81.479B | 1.800B | 两源较均衡 |
| multilingual | 46.120B | 0.300B | 仅 FineWeb-2 |
| encyclopedia | 5.389B | 0.539B | 目标短缺 0.261B |
| books | 4.852B | 0.485B | 目标短缺 0.015B；仅 Dolma Books |
| QA | 1.206B | 0.121B | 目标短缺 0.279B；仅 Stack Exchange |

正式 `mix_m3_e1_100b` 为 99,444,599,052 tokens，低于 100B 的原因主要是 QA、百科与书籍候选不足。
这本身可以接受，但来源坍缩不能只看 domain quota：质量分在全领域内直接 top-k，容易把
“source identity”当成质量代理。正式 mix 应增加 per-source floor/cap 或在 source 内校准分数，
再用小模型消融决定比例。

S6 元数据独立审计现已确认 D1 的配额账闭合：逐域 `selected + shortfall = quota`、逐源 token
之和等于逐域 selected、逐域 selected 之和等于 99.4446B，web shortfall 仅 578 tokens。该审计
同时验证 hybrid_v2 分位表覆盖全部九域、行数与 S5 一致。D3 精确扫描 94,235,054 个 train IDs
与 348,546 个唯一 validation IDs，交集为 0。D4 以有界内存外排扫描 94,235,054 个 member IDs
与 stage2 order IDs：两侧无重复且多重集精确相等。manifest 的 upstream snapshot 已绑定且
在物化前后复核一致；虽然 B4K 文档占 97.8127%，长上下文池仍有 635.3B tokens，E3 守恒与
供给门通过。
证据为 `data/processed/audit_s6_metadata_current_20261018.json`、
`data/processed/audit_train_val_ids_current_20261018.json` 与
`data/processed/audit_mix_identity_current_20261018.json`。

### 3.2 数据门禁状态

当前数据质量报告的 verdict 是 `review`，它抽查的是 fast mix，不是最终全量数据。仓库的
`data-process/docs/ACCEPTANCE.md` 仍有以下硬缺口：

- B1 已通过：web/code 为 direct-edge policy；精度审计、排序 lookup、决定表和全局 dropped
  账同代闭合。S5 直接消费 lookup，因此无需将决定回写 10TB S3。
- B3 已通过：decontam 当前 drop 46,044,928 / 6,485,760,234，约 0.71%。
  旧资产为 68.079M / 6.486B（1.05%）；三中文源把 C-Eval 阈值由一个提升为两个不同 shingle
  后救回约 21.557M 文档。人工审计曾显示 C-Eval 对中文命中 280 篇中 277 篇为误杀；决策文档
  要求至少 2 个 distinct hit；生产决策与 S5 已完成增量重算。
- C4 失败：goldrule research 只有 1,363,129 行，门槛是 10M。
- F2 已通过：正式 S6 manifest 绑定级联 upstream snapshot，且物化结束前再次复算验证无漂移。
- 2026-10-18 已对当前 snapshot 重跑全链一致性审计：S0 共 42,218 个成功分片；17 个纳入源的
  12,829 个分片在 S0→S1→S2→S3→decontam 间逐源分片数一致、行数守恒、范围内 error summary
  为 0。C4 的 634 个与 PG19 的 28,755 个 S0 分片在 S1 失败，被保留为显式范围外警告。审计器
  同时验证所有成功 summary 对应的 parquet 实物、footer 行数与字节数；证据为
  `data/processed/audit_consistency_20261018.json`。据此 A1/A2 已通过机器断言；当前 ndv2、
  decontam 与 S5 全局 summary 均为零错误，完整 release gate 仍须随新 S6 物化重跑。
- 质量抽检在 3,200 篇样本中记录了较多 email/phone/idcard 正则命中；当前 pipeline 只把
  secret pattern 标为 review，`require_license` 又是 false。最终数据需要明确的 PII、secret、
  license、opt-out 处置策略及例外清单。
- 原 22 项门禁没有覆盖来源治理；现已增加 G1/G2，总计 24 项。
  `source_governance.yaml` 逐源要求 immutable revision、license evidence、training/
  redistribution、PII 和 opt-out 签收。Dolma Books 三个 direct-URL 文件已补逐文件 SHA-256
  并得到组合内容身份；其余历史下载仍无法事后证明实际 revision，且全部来源尚未完成用途复核，
  所以继续 fail-closed；证据为
  `data/processed/audit_source_governance_20261018.json`。
- C2 已升级为机器 PASS：全量扫描覆盖 12,829 个 S4 分片、17 来源、6,485,760,234 行及
  203 个语言标签；文件错误与 flags 均为 0，119 个达到 1,000 行门槛的语言全部通过分布形态
  检查。旧 `hybrid_v2` 缺失 upstream 语义经 17 源、1,761 行组件级精确复算验证为 0 mismatch，
  证据为 `data/processed/audits/s4_hybrid_v2_legacy_audit.json`。修复后的全局分位表随后闭合
  12,829 分片和 6,485,760,234 行；其选择合同 manifest SHA-256 为
  `6e49402beeb1b1a279ba6e57382cfd3c9203f2354abf1a98d2a65f701b095f4e`，发布 summary SHA-256
  为 `7cc6d66991e00ef2d5e38faf605b45fa4f7a251171cd0baeb51184b17dc03fb0`。

### 3.3 课程排序语义不成立

S6 将 `upstream_score` 升序称为“lexical complexity 低→高”。但各来源的 upstream score
定义不同，很多来源没有该字段；缺失值会落到 doc_id hash 顺序。对 FineWeb-Edu 而言，
upstream score 更接近教育/质量分，而非词汇复杂度。当前 stage1 因而是“部分来源低质量到
高质量、其余来源随机”，不能宣称复现了 ZGCM curriculum。

ZGCM 的公开实现只对自然语言按 lexical complexity 排序，并将 code/math 单独打乱混入；
报告还说明该方案改善 code/math BPB，却轻微损害通用 BPB，说明它本身也应作为消融而非默认
真理。[ZGCM-1 technical report](https://arxiv.org/html/2609.13356v2)

已将全部正式 mix 配置改为 doc_id hash 确定性全局打乱。旧 `upstream_score`
排序只能在同时声明 `method: upstream_score_legacy` 和 `experimental: true` 时运行；
从两段配置切换到 random baseline 时会删除旧 stage1 顺序表，避免重复消费。现有 S6
产物仍是旧顺序，所以 D2 继续阻断，必须在上游数据门禁修复后重物化。

后续若做课程消融，应新增独立 `curriculum_score`：每种语言/来源分别校准，至少由文档长度、词汇丰富度、句法
复杂度、可读性/年级和质量异常标记组成；去掉极端异常值；code/math 单独 shuffle；用 random
baseline 做 matched-data/matched-FLOP 对照。

## 4. 训练与实验正确性审计

### 4.1 P0：四卡数据游标会重叠和跳样

trainer 只在启动时设置：

```text
local_cursor = global_consumed + rank * samples_per_rank
```

之后每个 rank 都按 `+1` 连续读取，global cursor 每 step 却前进
`samples_per_rank * world_size`。world size > 1 时，rank 1 在 step 0 读到的数据会被 rank 0 在
step 1 再读；同时 global cursor 跨过一批从未读取的数据。正确映射必须把每个 step 的全局
base 与 rank offset 同时纳入：

```text
global_index(step, rank, local_i)
  = resume_consumed + step * (world * samples_per_rank)
  + rank * samples_per_rank + local_i
```

当前所有 E0-Q2 是 world size 1，不受此 bug 影响；任何四卡正式训练都会受影响。必须新增
world=2/4 的样本集合互斥、全局覆盖、断点前后逐位一致测试。

### 4.2 P0：E0-Q2 验证与结论无效

`open_val_streams` 只比较 `meta.vocab_size <= model_vocab`，没有要求
`meta.tokenizer_sha256 == train_tokenizer_sha256`。当前事实是：

- 32K 模型跳过 64K/128K held-out 流，只会把自己的 E0 训练流当成可读验证流。
- 64K BPE 模型会验证 32K BPE、64K Unigram 和 64K BPE 的流；相同整数 id 在不同 tokenizer
  中表示不同 token。
- 128K 模型 vocab 足够大，因此会读取目录中的全部流，包含所有其他 tokenizer 的 token id。
- 真正的 9 个 domain val 流只由 64K BPE tokenizer 物化。

即使修好验证选择，也不能直接比较 raw CE/token：tokenizer 粒度不同。应在同一 held-out 文本
上分别物化各 tokenizer，并报告 bits/byte（或 NLL/byte）、下游 zero/few-shot、吞吐、显存和
wall-clock。ZGCM 也持续监控 BPB、normalized loss、能力评测与优化器统计，而不只看 token CE。
[ZGCM monitoring](https://arxiv.org/html/2609.13356v2)

E0 注释还将实验称为 fixed compute，但三臂 non-embedding 参数相同、总参数分别约 439M、
473M、540M，embedding/output 计算不同；tokens/step 相同不等于 FLOPs 相同。建议同时给出：

1. fixed-token：衡量相同 token 预算的训练成本与最终能力；
2. fixed-FLOP/wall-clock：按真实 FLOPs 或实测 GPU 时间对齐；
3. fixed-text-byte：衡量在相同原始信息量下的压缩与建模质量。

### 4.3 P0：token stream 没有文档边界

materializer 对每篇文档 `add_special_tokens=False` 后直接串联。每个边界会生成
`last_token(doc A) -> first_token(doc B)` 的伪监督。正式流至少应在每篇文档后插入 EOD/EOS，
并把 EOD 数量计入 token budget。进一步可消融 Megatron 风格的 reset-position/reset-attention；
首版可保留跨文档 attention，但必须有 EOD target。

### 4.4 P0：checkpoint 不能称为工业级精确恢复

当前 full model/optimizer 都由 rank 0 集中保存；所有 rank 恢复同一份 rank-0 RNG。没有
checkpoint retention，三条 E0-Q2 曾每 250 step 保存一次，并占用约 929GiB；任务现已停止，
checkpoint 保留作取证和迁移测试。

目标方案应采用 PyTorch Distributed Checkpoint 或 Megatron `torch_dist/fsdp_dtensor`：模型、
optimizer、scheduler、grad scaler、逐 rank RNG、sampler/data cursor、代码/config/data hash 全部
入 checkpoint；支持异步分片保存、不同拓扑 reshard、临时目录+complete marker。保留策略建议：
滚动 `latest-2` + 每 N 千 step milestone + stage-final，另有磁盘余量门禁。Megatron 当前实现
明确收集各 data-parallel rank 的 RNG，并支持 async 与 non-persistent checkpoint。
[Megatron checkpointing](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/checkpointing.py)

### 4.5 长上下文路径尚未就绪

SDPA 后端缓存 `S×S` bool mask，64K 会达到无法接受的显存量；本地 FlashAttention 版本环境又
没有固化。RoPE buffer 能按新 max length 重建，不等于获得了有效的 64K 能力。

工业上普遍采用渐进式 continued pretraining，并在长阶段保留短样本：Qwen3 从 4K 进入 32K，
长阶段使用 75% 16K–32K 与 25% 4K–16K，并结合更高 RoPE base、YaRN/DCA；ZGCM 使用
16K→64K→256K 且持续 replay 短样本；腾讯混元采用 32K→256K，并把自然长文本与普通长度
数据混合。[Qwen3 report](https://arxiv.org/html/2505.09388v1)
[Hunyuan-Large report](https://arxiv.org/html/2411.02265v3)

YuFormer 更合适的目标是 4K general pretrain → 16K/32K → 64K midtrain：采用支持 windowed
attention 的 FlashAttention 2/3 或等价 block-sparse kernel；每阶段保持近似恒定 tokens/update；
使用自然长文档、multi-doc QA、代码仓、长推理/agent trace，并 replay 4K/16K；以 RULER、
Needle、LongBench、长文 perplexity/BPB 和短能力回归共同验收。YaRN/ABF 只能作为候选，需要
小预算对照后选定。

## 5. 与主流公开方案的对照

| 方向 | 主流公开证据 | YuFormer 应采用的做法 |
|---|---|---|
| 数据打标与配比 | Qwen3 对 30T+ tokens 做 educational value/domain/safety 等细粒度标注，并用 proxy model 做 instance-level mixture ablation | 保留多维标签；分数按语言/来源校准；E3 用小模型搜索 mix，而非手填比例后直接放大 |
| model-based filtering | DCLM 的核心结论是 model-based filtering；ByteDance Seed-Coder 使用模型评分 readability/modularity/clarity/reusability；Nemotron-CC 用 classifier ensemble 与受控 synthetic rephrasing | 规则门 + 轻量 classifier + 强模型审计三层；保留 rejected/review 样本用于 hard-negative 与失败挖掘 |
| 多语言 | FineWeb2 对语言自适应过滤/去重，并按 duplication count 与 quality 重平衡 | 中文/多语阈值单独校准；报告每语言 fertility、质量、重复率和下游能力 |
| 课程与退火 | ZGCM：0.99T lexical curriculum + 3.2T global shuffle；Hunyuan：warmup/缓慢衰减/最后 5% 高质量 anneal；OLMo2：大规模阶段后用高质量 mix 退火 | general 阶段做 random vs curriculum 消融；正式训练用 WSD 或 constant+独立 anneal stage，末段高质量 replay |
| 优化与系统 | ZGCM 用 Megatron-Core、Muon 0.9、constant 2e-4、FP8、SWA；百度 ERNIE 开源栈提供 hybrid parallel、FP8、unified checkpoint | Muon 与 AdamW 做 matched-FLOP 试验；正式后端用 TorchTitan/FSDP2/DCP，Megatron-Core 留作多机性能对照 |
| 长上下文 | Qwen3/Qwen2.5-1M、ZGCM、Hunyuan 都使用渐进长度与短样本 replay | 4K→16/32K→64K；自然长数据 + 合成长任务；有效上下文测试决定是否通过 |
| RL | ByteDance DAPO/verl、ZGCM AReaL 都强调动态采样、可验证奖励与大规模 rollout | 0.5B 首先用 verl + GRPO/DAPO 做 math/code RLVR；记录 pass-rate 分布、zero-variance group、KL、clip ratio、长度与 reward hacking |

参考的一手资料：

- [ZGCM-1 论文与开放训练配方](https://arxiv.org/html/2609.13356v2)
- [ZGCM-1 开源仓库](https://github.com/zgcagi/ZGCM-1)
- [Qwen3 Technical Report](https://arxiv.org/html/2505.09388v1)
- [Tencent Hunyuan-Large Technical Report](https://arxiv.org/html/2411.02265v3)
- [Baidu ERNIE 开源预训练说明](https://github.com/PaddlePaddle/ERNIE/blob/release/v1.5/examples/pre-training/README.md)
- [ByteDance Seed-Coder 数据构建](https://seed.bytedance.com/en/blog/seed-coder-open-sourced-llm-based-code-data-building-method-validated)
- [DataComp-LM / DCLM](https://arxiv.org/abs/2406.11794)
- [FineWeb2](https://arxiv.org/abs/2506.20920)
- [Nemotron-CC](https://arxiv.org/abs/2412.02595)
- [OLMo 2 两阶段训练](https://allenai.org/blog/olmo2)
- [verl v0.9.1](https://github.com/verl-project/verl/releases/tag/v0.9.1) 与 [DAPO](https://arxiv.org/abs/2503.14476)
- [lm-evaluation-harness v0.4.13](https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.13)

## 6. 建议的目标技术栈

### 6.1 数据与 tokenizer

1. 建立 immutable raw/source manifest：URI、dataset revision、文件 sha256/etag、下载时间、
   license、允许用途、语言/领域、parser version。
2. 每个阶段输出 content hash、父产物 hash、code commit、config hash、统计与 rejects；用
   content-addressed dataset version 取代“文件存在即跳过”。
3. 去重采用 source 内 exact/near + 必要的 cross-source near dedup；超长文档分块生成 shingle，
   避免只扫前 50K shingles；去污染输出 benchmark、distinct hit、位置和人工裁决样本。
4. scorer 做 source/language calibration；mix 同时施加 domain quota 与 source floor/cap；用
   proxy LM 的 BPB/下游能力选择阈值与比例。
5. tokenizer 先以 64K byte-level BPE 作为生产 baseline；E0 重跑后再决定是否保留 32K/128K。
   每文档追加 EOD；SFT chat template 的 role/think/tool token 现在就冻结，避免后期扩词表。

### 6.2 Scaling 与 pretrain

1. E1 采用 isoFLOP 网格，而不是每个模型只跑一个 token 数。建议 50M/100M/200M/400M/800M，
   每档至少 3 个 token budget，anchor 点 2–3 seeds；拟合 loss = f(N,D)，报告 bootstrap CI 与残差。
2. 同时拟合最佳 peak LR、tokens/update 与 warmup；Qwen3 也是按 stage、架构与数据预测 LR/batch，
   不能把一个 2e-4 配方机械复制到所有规模。
3. 正式 schedule 建议：0.5%–1% linear warmup → stable/slow decay → 最后 5%–10% high-quality
   anneal。当前 constant LR 符合 ZGCM general stage，但必须有 terminal decay 对照，才能得到可交付
   checkpoint。若要支持未知终点的持续训练，可实现 WSD。
4. 0.5B/4×H800 正式采用 TorchTitan v0.3.0 的 FSDP2/DTensor/DCP；自研 FSDP1 trainer
   只保留为兼容与数值对拍后端。Megatron-Core 作为未来多机性能对照，不维护第二条正式训练主线。
5. 监控至少包含 per-domain held-out BPB、normalized loss、grad/param/update norm、activation/logit
   异常、吞吐/MFU、显存、通信占比、data cursor checksum、eval suite 和 checkpoint canary resume。

### 6.3 SFT 与 RL

1. SFT：统一 chat schema；assistant-only loss；sequence packing；长短样本分桶；去污染；规则 +
   strong-model rubric + execution verifier；保留 think/no-think、tool-use、拒答等标签。chunked CE
   已支持并对拍 `ignore_index=-100`；SFT 数据契约、规范渲染和 THD 打包已建立，下一步是数据
   release 与 TorchTitan block-causal trainer adapter。
2. Preference：小项目可先做 DPO/SimPO 形成完整闭环，但不要把它当 RLVR 结果。
3. RLVR：新工程主线固定为 verl v0.9.1 + SGLang v0.5.20，并使用独立 PyTorch 2.13/CUDA 13
   环境；DAPO 严格复现仍使用 recipe 指定的旧 verl commit。从数学 exact-answer 开始，再扩展代码 tests 与指令约束。
   DAPO 严格复现 commit 与较新工程 commit 独立记录，禁止浮动 `main/latest/dev`。
4. GRPO/DAPO 门禁：训练/评测 verifier 隔离、动态采样、全对/全错组比例、reward distribution、
   KL、clip fraction、importance ratio、response length、truncate rate、reward hacking 人审与 held-out pass@k。

## 7. 执行优先级与点火门

### P0：任何四卡或正式实验前

- [x] 修复 rank-strided sampler，并通过 world=1/2/4 + resume 数据序列测试。
- [~] `open_val_streams` 已强制 tokenizer hash；仍需每个 tokenizer 单独物化 9-domain val，E0 改用 BPB。
- [~] token materializer 已插 EOD 并记录 doc-boundary policy；仍需重物化正式数据。
- [x] 中文 decontam distinct-hit 已完成三源生产重跑；全局发布闭合 12,829 分片、
  6,485,760,234 行且 `shard_errors=0`。完成 web/code near-dedup 刷新或形成经过实验支持的
  替代策略与显式风险接受记录。
- [~] val/S6 已加上游快照与级联失效；仍需把同一内容寻址契约扩展到 scorer/quantile 全链。
- [~] 已实现逐 rank RNG 与 retention；仍需 sharded async checkpoint、拓扑 reshard 和 kill -9 canary resume。
- [x] 增加 out-dir 独占 run lease、checkpoint 单调发布与 `--no-resume` 防覆盖；现有 64K
  碰撞目录必须隔离，不能直接作为结果发布。
- [~] TorchTitan/PyTorch/Python/CUDA lock 与机器可读 preflight 已生成；当前 driver/Python/Torch
  不达标，需环境升级后构建容器/lock 并通过 I0。

### P1：正式 0.5B pretrain 前

- 重做 source-balanced mix 与 E3 proxy ablation；goldrule 扩到门槛并完成盲审一致性。
- 完整 license/PII/secret/opt-out 门禁与 dataset card。
- 完成 isoFLOP scaling、LR/batch scaling 和 Muon/AdamW 对照。
- 将 1.5T/45 天承诺按实际稳态吞吐重算。历史文档已测到当前栈曾需约 167 天；即使最新单卡
  探针达到约 21.9% MFU，也必须用 4 卡真实 all-reduce、长时间邻居负载和 checkpoint/eval 开销
  重新估算，不能用单卡峰值外推。

### P2：pretrain 运行期间并行开发

- 16K/32K/64K midtrain 数据与 kernel；短能力回归门。
- SFT data release、TorchTitan block-causal trainer adapter 与 base/instruct eval。
- verl RLVR 模型 adapter、verifier 单测、动态采样与 reward hacking audit。
- 模型卡、数据卡、实验卡、成本/故障统计和可复现 release bundle。

## 8. 本次验证记录

- 模型与 tokenizer 测试：42 passed，1 skipped；新增 chunked CE `ignore_index` loss/gradient 对拍；
  tokenizer 测试不再返回值触发 pytest 警告。
- pretrain 完整测试：71 passed；覆盖 2-rank Gloo checkpoint、world=1/2/4 sampler、tokenizer
  validation 隔离、EOD 物化、checkpoint retention、run lease、run identity、指标/checkpoint
  单调发布、legacy resume 与验证脚本。
- SFT 测试：9 passed；覆盖不可变来源、跨 split 泄漏、工具状态机、合法 JSON arguments、
  assistant-only 标签、Jinja/Python 逐字节渲染、THD 边界与超长拒绝。Midtrain 契约 3 passed；
  RL 测试 3 passed；eval 发布门禁 2 passed。
- data-process 完整回归：136 passed、10 skipped（1000.16s）；此前同进程 repeated-fit 的 6 个
  fastText NaN 均在进程隔离后通过。随后新增的 downloader fail-closed/disabled-source 定向套件
  7 passed，新增 SHA-256 模型身份确定性用例单独复验 1 passed。
- 配置与静态检查：60 份 YAML、11 份仓库 JSON 解析通过；Python compileall、关键 shell
  `bash -n` 与 `git diff --check` 通过。
- 三条本机 E0-Q2 主训练及孤儿 worker 已安全停止；checkpoint 未删除，失效 manifest 已写入。
  这些目录不能恢复或发布结果；重跑必须使用带 EOD、tokenizer 专属 held-out val 和新 run id 的目录。

## 9. 适合作为简历项目的最终叙事

完成上述门禁后，项目亮点应围绕“可验证的端到端系统”组织，而非只列技术名词：

- 构建 15TB 原始语料、十亿文档级 deterministic pipeline，并用数据血缘、去重/去污染、
  model-based quality filtering 与 proxy-model mixture search 交付可审计数据版本。
- 通过 tokenizer BPB/吞吐/下游能力消融、isoFLOP scaling law 与 optimizer/schedule 实验确定
  0.5B dense 模型配方。
- 在 4×H800 上实现 topology-safe exact resume、sharded async checkpoint、FlashAttention、
  compile/recompute/FP8 性能优化，并公开 MFU、故障与成本数据。
- 完成 4K→64K continued pretraining、assistant-only SFT、verifiable-reward RL，并用统一的
  base/instruct/long-context/agent eval 与数据污染审计证明每阶段增益。

这套叙事比“复现 ZGCM 架构”更有区分度，因为它能展示数据科学、分布式正确性、训练系统、
实验设计与后训练的完整证据链。
