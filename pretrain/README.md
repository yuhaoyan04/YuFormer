# YuFormer pretrain

> 2026-11-13 当前用户要求：使用全部八张 H800 做分布式预训练，目标约 30 天，启动工期上限 33 天。配置为 world_size=8、microbatch=8、grad_accum=16，保持 4,194,304 tokens/update。此前 E1 waiver 继续有效；数据、数值、恢复与八卡吞吐门禁不可跳过。四卡配方与 canary 已归档并标为 STALE。详见 [八卡计划](../report/EIGHT_GPU_PRETRAIN_PLAN_20261113.md)。

> 2026-11-14 当前候选：1,516,600,064 参数、18L/H2560/FFN6912、40Q10KV、每六层一个全局注意力，峰值 LR=1e-4，后端 `cudnn_flex_swa`。200 步持续实测按 90% duty 预计约 32.43 天。候选配置已写入，**正式训练尚未启动**，完整数据、新资格、实际模型故障恢复和生产 canary 仍须通过。[实测与研究判断](../report/GPU_MODEL_SIZE_SWEEP_20261114.md)。


预训练运行闭包（宪法 1）：数据契约物化 → 分布式训练 → 实验追踪落盘。
已准备数据的全量执行入口为
`/opt/yuformer-torchtitan/bin/python pretrain/scripts/run_authorized_pretrain.py --reuse-prepared-production`。
启动前必须完成实际十二条流的字节/来源/长度账本审计、三阶段消费配比、生产块排列、
当前八卡资格、最终模型/微批/梯度累积/编译/拓扑的训练/写盘/加载中断恢复、
训练配方冻结与实际生产数据 canary。该入口复用已审计资产，执行最终门禁后启动。
`pretrain-model-order-adoption.json` 的阻塞状态会在任何准备动作之前拦截执行；
只有实际门禁齐全才能改成 `READY_FOR_PRETRAIN`。训练专用冻结通过原始数据身份档案复用
既有 release、token 流和采样集合，不重新运行 S6 或改写数据身份。
实时状态见 `data/health/authorized-pretrain-pipeline.json`；启动前先确认该入口没有实例运行。
执行与门禁顺序见 [`FULL_PRETRAIN_EXECUTION_20261113`](../report/FULL_PRETRAIN_EXECUTION_20261113.md)。
正式主线已锁定 TorchTitan v0.3.0 / FSDP2 / DCP；本目录现有 trainer 标记为
`pytorch-fsdp1-compat`，只用于旧 checkpoint 迁移、数据契约验证和数值对拍。

## 结构

```
pretrain/
├── configs/
│   ├── pretrain_base.yaml   # 主配置：1.52B 候选 × 1T tokens × 8×H800（单一真源）
│   └── pretrain_base.env    # 运行时环境（外部资产路径，宪法 3）
├── src/yuformer_pretrain/
│   ├── config.py       # 配置加载：${ENV} 解析 + 自包含校验 + expected_iters 身份守卫
│   ├── data.py         # TokenStream / DataCursor / PrefetchLoader（双进程预取，保序）
│   ├── muon.py         # Muon（NS5 正交化，~50 行自研）
│   ├── optim.py        # adamw|muon 路由（2D 矩阵走 Muon，embedding/head/scalar 走 AdamW）
│   ├── checkpoint.py   # 原子写 + marker 协议（RESUME_ITER 防跑错）
│   ├── tracking.py     # experiments/_common/tracker.py 适配
│   └── trainer.py      # FSDP 训练循环（BF16 autocast，activation ckpt 可开关）
├── scripts/
│   ├── launch.sh               # env→CLI 翻译（torchrun 入口）
│   ├── materialize_tokens.py   # S6 契约物化：order 表 + s3 正文 → token 流
│   └── smoke_train.py          # 端到端冒烟（tiny + 1MB 合成数据 + 1 GPU 100 步）
├── tests/                # 游标确定性 / Muon / 配置守卫 / 物化契约 / tracker / checkpoint / lr schedule
└── docs/SOURCE_PROVENANCE.md
```

## 用法

正式训练前先进入独立的 Python 3.12 / PyTorch 2.14 cu126 环境并执行门禁。系统自带的
Python 3.10 / Torch 2.2 alpha 不用于正式训练。正在运行的数据物化同样使用
锁定的 `/opt/yuformer-torchtitan/bin/python`（Python 3.12）；其物化器代码、tokenizer
和来源身份单独绑定，不得在编码期间切换环境或代码。
只读 PyArrow 审计可使用独立环境，不应据此推断编码进程的解释器。
driver 535 由 CUDA 12.6 qualification 实测：

```bash
export YUFORMER_TORCHTITAN_ROOT=/opt/torchtitan-v0.3.0
python pretrain/backends/torchtitan/preflight.py \
  --out data/health/torchtitan-preflight.json
```

后端版本、迁移验收项和当前阻塞项见
[`backends/torchtitan/README.md`](backends/torchtitan/README.md)。以下命令属于兼容后端。

### 数据物化（S6 → pretrain，一次性）

```bash
export YUFORMER_S6_ROOT=/data/yuformer/processed        # data-process 输出根
export YUFORMER_TOKEN_STREAMS=/data/yuformer/streams/pretrain-v1
export YUFORMER_DATA_SNAPSHOT=/data/yuformer/processed/releases/<candidate>.release.json
TOK=tokenizer/tokenizers/yf-64k-bpe                     # 或 yf-32k-bpe / yf-128k-bpe

# 原始物化顺序确定性固定；正式消费另外绑定阶段内独立因果块排列
PYTHONPATH=pretrain/src python pretrain/scripts/materialize_tokens.py \
  --order $YUFORMER_S6_ROOT/mix_order/<mix>_stage2.parquet \
  --s3-root $YUFORMER_S6_ROOT/s3 --tokenizer $TOK \
  --out-dir $YUFORMER_TOKEN_STREAMS --name stage2 \
  --val-dir $YUFORMER_S6_ROOT/val_sets \
  --data-snapshot $YUFORMER_DATA_SNAPSHOT --workers 64  # 逐域 held-out val 流
```

产出 `<name>.tokens.bin`（按词表大小选择 uint16/uint32，每篇文档末尾追加 EOD）+
`<name>.meta.json`（`total_tokens / bytes / sha256 / edge_sha256 / tokenizer_sha256 /
document_boundary / data_snapshot_id / source`）。
`edge_sha256`（size + 首尾各 1MB）是启动期身份守卫；全量 `sha256` 在 meta 里供审计。
训练与验证流必须具有相同的 `tokenizer_sha256`；验证只读取 `val_*`，不兼容流会被跳过。
正式配置还要求 `append_eod` 的 count 等于 `n_docs`，并要求训练/验证流的
`data_snapshot_id` 与 PASS release 的 `snapshot_id` 完全一致。物化器还会在读取正文前验证
order/validation parquet 的路径、字节数与 SHA-256 均属于该 snapshot；part 缓存绑定 order
哈希，最终 bin/meta 采用 meta-last 原子发布。

正式 1T 三阶段物化额外使用 `--write-doc-lengths`，发布
`<name>.doc_lengths.u32`。meta 绑定其行数、总和、全文 SHA 和 edge SHA；独立审计要求每个
position 恰有一条长度记录且长度和等于 token 流总量。跨阶段 overlap 门禁据此复原 loader
实际监督的 token 区间，而不是用参考 tokenizer 的完整 headroom 序表替代实际消费量。

冻结配方要求 `data/processed/audits/training_tokenizer_calibration.json`：从已选择的
warmup/stable/anneal 完整已选训练人口分别逐域用独立 BLAKE2b rank 排列抽样，
使用真实 tokenizer 并计入 EOD。schema 2 必须绑定独立采样算法、seed、完整人口计数
和采样器代码；与原 doc_id 选择顺序相关的 hash 前缀校准已被拒绝。
校准绑定原 release、质量策略、源数据
roster、tokenizer、物化器与逐文档长度证据；冻结器重新计算各域比率后才分配 reference
配额。有效证据可复用，过期证据自动重测；没有三阶段选样 release 的首次准备会明确阻塞，
不得用 held-out validation 比率充当训练阶段的校准结果。每个实际阶段流发布后先检查消费
前缀容量与域配比，通过后才编码下一阶段；异常会写入 `BLOCKED_PREFLIGHT` 及错误报告。

### 训练

`pretrain_base.yaml` 是当前 1T 正式合同；旧 1.44T `pretrain_v1` 已移入
`configs/_holding/retired_pretrain_v1/`，且任何后端都会拒绝其 `superseded` 标记。TorchTitan
外部 `BaseModel`、预分词 stateful loader、FSDP2、chunked CE、WSD、
异步 DCP 与 fail-closed launcher 已落代码；正式入口是
`backends/torchtitan/launch.sh`。资格合同与自动 runner 覆盖单卡数值对拍、4 卡 DCP、
训练/保存/加载中断恢复，以及 4→2/8 重分片。当前正式目标是八卡，4→8 必须通过；
历史资格 PASS 在源码和拓扑输入变化后不能继续使用，需要锁定环境重新验收。
2026-10-22 深审发现并修复了 meta 参数初始化和 FSDP 梯度重复除法等缺陷，
历史训练结果已失效，详见 [`PRETRAIN_DEEP_AUDIT_20261022`](../report/PRETRAIN_DEEP_AUDIT_20261022.md)。

`scripts/launch.sh` 现仅运行显式 `backend: pytorch-fsdp1-compat` 的迁移/对拍配置；正式
E1/E2 已全部声明 TorchTitan，因此同样会被拒绝。失效 E0-Q2 与混杂变量的旧 E2 配置已移到
`configs/_holding/`，旧自动调度器固定 exit 4。

正式 launcher 在 torchrun 前校验 tokenizer/data/stream 身份，发布不可变
`run_identity.json`，并持有基于原子 mkdir 的 NFS generation lease；不同身份或并发任务复用同一输出目录
会在创建 GPU 进程前失败。兼容后端保留其原有的同等守卫。

TorchTitan 的九域 held-out 验证按 `domain_count × samples_per_domain / global_batch`
精确推导步数；域集合必须精确等于 `web/code/math/chinese/academic/books/encyclopedia/qa/multilingual`，
且每域样本数必须能被验证 global batch 整除。除周期验证外，训练的最后一个
optimizer step 必须再次验证，结果收集器只接受该步同时具有 aggregate loss 和完整九域 loss
且所有 loss 有限的事件文件。这样 LR、batch、scaling 与配比分析不会把较早或损坏的验证
loss 绑定到最终消费 token 数。

历史 schema 2 验证缓存还须通过九域完整重编码的逐字节对拍，新编码必须验证 source-ID v4。
证据以不可变 CAS manifest 保存，原验证流保持原身份；训练专用冻结通过
`validation_source_proof.py` 将证据与实际完整字节审计的九域 SHA 关联，并校验来源 roster、
原元数据、decoder 与逐文档长度账本。后续报告重新导出不改变已冻结的开训依据。

### 冒烟

```bash
/usr/bin/python3 scripts/smoke_train.py --gpu 0
# 验证：loss 下降 / 断点续训逐位一致（61..100 步 vs 连续跑）/ RESUME_ITER 拒绝 / Muon 有限下降 / checkpoint 完整
/usr/bin/python3 -m pytest tests/ -q
```

## 配置语义（configs/pretrain_base.yaml）

正式 TorchTitan validator 拒绝未知字段、重复 YAML 键与字符串布尔值。
`resume`、`init_weights_from` 等未接入的 YAML 选项会报错；
同一 run 的恢复由 DCP 和 `RESUME_ITER` 协议控制。
正式 `data.sampling_policy: block_permutation` 绑定 schema 2 / `PRODUCTION` manifest：
每八条连续因果样本为一块，在每个 WSD 阶段内独立排列，每个原监督 token 恰好消费一次，
块内上下文和 labels 保持原值，额外 headroom 不进入训练。
checkpoint 绑定排列 manifest 的 SHA 与逻辑样本游标；恢复测试覆盖三阶段边界。
短资格测试仅允许显式 `PREFIX_PROBE`，保留生产人口和原排列的逻辑前缀，不能替代正式 manifest。
三段 WSD 数据边界必须与 LR warmup/decay 精确落在同一 optimizer update。
正式 overlap audit 还会按 loader 实际监督区间统计逐域最终 token 比例，
同时校验计划中的绝对和相对误差上限，超差时禁止点火。
代码变化后，历史 qualification 与 production canary 必须重新生成；
当前 `attention.sdpa_backend: cudnn_flex_swa` 对滑窗使用原生 GQA / Triton FlexAttention block mask，
全局层使用 CuDNN causal；不允许静默数学回退。
独立八卡对拍覆盖真实 4K/微批 8、输出/输入梯度/全部参数梯度及两种主参数 dtype。
`cudnn` 路径保留为显式参考后端；更换后端会使历史资格失效。
`audit_training_progress.py` 只读核对实际 optimizer LR、loss、checkpoint scheduler、token 计数与数据游标；
`PASS_OBSERVED_PREFIX` 只表示已观察部分通过，完整训练通过仍以最终完成标记和门禁为准。
普通 CPU pytest 不启动 `tests/verify` 的 GPU 子任务；该旧兼容集成套件须显式
`YUFORMER_RUN_GPU_VERIFY=1` 并选定空闲卡，不能与正式八卡工作负载共享。
最新审计见 [`PRETRAIN_READINESS_AUDIT_20261113`](../report/PRETRAIN_READINESS_AUDIT_20261113.md)。

| 键 | 语义 |
|---|---|
| `optim.optimizer` | 正式 TorchTitan 路径仅允许 `adamw`；norm WD=0。Muon 仅兼容研究路径，接入正式后端须另行资格测试 |
| `optim.lr` | peak LR 候选；正式值由分规模 LR/tokens-update pilot 决定，不能直接沿用 ZGCM 2e-4 |
| `optim.lr_schedule` | `wsd`（正式默认：warmup→stable→末段 cosine decay）\| `constant` \| `cosine`；`wsd` 由 `decay_steps` 明确末段长度 |
| `train.budget_tokens` | 预算 token 数（1T = 50B warmup + 800B stable + 150B anneal）。**迭代数 = budget / tokens_per_step**（宪法 6 元数据推导） |
| `train.expected_iters` | 身份守卫：必须与推导值相等，否则拒绝启动（防手滑改预算） |
| `train.micro_batch_size / grad_accum` | tokens_per_step = seq_len × mbs × accum × world_size |
| `train.activation_checkpointing` | 显存换速度；当前 4K/1.52B 候选已实测不开，长上下文另行资格验证 |
| `train.keep_last_n / keep_every` | checkpoint 保留策略；保留最近 N 个并可按 iteration 保留长期里程碑，`keep_last_n=0` 禁用自动清理 |
| `data.stages / meta` | 正式 1T 按 warmup/stable/anneal 顺序消费三段；stage_sample_limits 绑定优化器边界，每段内部确定性打乱 |
| `data.epochs` | 全局流回绕次数（容量 = 总样本 × epochs，超容量即 fail-fast） |
| `data.seq_len` | 定长 4096（pretrain 阶段；变长/packing 后置到 midtrain/sft） |

## 断点协议（宪法 6）

```
<out_dir>/checkpoints/
  ckpt-00010000/{model.pt, optim.pt, cursor.json, rng-rank-*.pt, rng.pt, complete.json}
  latest.json   # {"iter", "config_sha256", "run_name", "dir"}——原子写最后落
```

checkpoint iteration 只能单调增加；相同或更小 iteration 的发布会被拒绝，防止并发任务把
`latest.json` 回退。

每个新输出目录还会生成不可变 `run_identity.json`，绑定配置、模型配置、tokenizer、数据、
源码树和 Git 状态；不同身份不能复用同一目录。每次启动另写 `attempts/*.json`，记录主机、
Python/Torch/CUDA 与 world size。TorchTitan launcher 还原子发布 `current_attempt.json`；
`training_complete.json`、结果收集器和重启后领养的 queue job 必须绑定同一个 attempt ID。
新 attempt 会先归档旧 completion/result/TensorBoard event，避免失败重试继承旧 PASS 或最终步
指标。`metrics.jsonl` 只允许 step 单调不减。
实验身份审计还发布其实际读取输入的 SHA-256 快照。冻结 1T 配方会在写配置前重验整份快照；
冻结报告的全部 inputs/outputs 也由 production contract 与最终 stack audit 再次逐文件校验。
production contract 另外绑定正式 env、训练配置、模型配置、tokenizer gate、derived release、
token stream audit 和 tokenizer 文件。任一实验结果或生产输入在审计后发生替换，都会使 READY
失效并要求重新生成下游证据。
若进程在最近 checkpoint 之后继续记录若干步再被杀，恢复时 canonical metrics 会原子回卷到
checkpoint step；未提交后缀保存在 `metrics.orphaned-after-*.jsonl`，避免把两条时间线混写。

- **游标**：`cursor.json` 存全局 `consumed_samples`（跨 stage 拼接的样本序号）。
  每个 step 先划全局连续块，再给每个 rank 分配互不重叠的连续子块；运行时保序消费，
  任意断点重启消费序列逐位一致（smoke 有断言）。
- **RESUME_ITER**：环境变量设置时必须等于 `latest.json` 的 iter，否则 exit 4
  （对齐 ZGCM launch_train.sh，防"跑错 checkpoint"）。
- **config 守卫**：checkpoint 记录配置 sha256，续训时配置被改动即拒绝
  （改配置必须换 `run_name`）。
- 存模型 + 优化器 + 游标 + 每个 rank 的 RNG（python/numpy/torch/cuda），`complete.json`
  与 `latest.json` 在所有 rank 写完后发布；旧版单 `rng.pt` checkpoint 仍可读取。

## 实验追踪（experiments/_common/tracker.py）

每臂 `<out_dir>/results/<run_name>/`：

- `metrics.jsonl`：每 `log_every` 步一行 `{"step", "loss", "lr", "consumed_samples", "step_time"}`；
  `eval_every` 步附加 `val_loss_<domain>`（S6 per-domain held-out）。
- `manifest.json`：config sha256（sort_keys 序列化）、tokenizer sha256、
  数据 sha256 + edge sha256 + source、seed、参数规模、MFU/步时实测。

## MFU 实测

见 docs/SOURCE_PROVENANCE.md 实测记录（smoke tiny 预设 + 0.5B 单卡短程基准）。

## 兼容后端限制

- **CE 分块**：当前按 `train.ce_chunk` 分块计算词表 logits，降低峰值显存；尚未实现
  tensor-parallel vocab shard，扩到更大模型时需替换为分布式 fused CE。
- **FSDP1 full state_dict**：checkpoint 走 full-state-dict 收集（rank0 落盘），不允许用于新的
  scaling 或正式 pretrain 结论；正式后端迁移到 FSDP2 + DCP sharded async checkpoint。
- **prefetch 双进程**为 fork + memmap，CPU 已饱和时预取收益有限，但保序性不受影响。
- **Muon 等效 lr 口径**（0.2×sqrt(max(fan))）来自 Moonshot 公开配方，E0/E1 期间
  与 AdamW 做同图对照后再锁定。
