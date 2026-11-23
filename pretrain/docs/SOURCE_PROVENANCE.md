# SOURCE_PROVENANCE — pretrain/

## 来源与改动

| 组件 | 来源 | 说明 |
|---|---|---|
| FSDP1 兼容循环 | PyTorch 原生 `torch.distributed.fsdp`（运行时 2.2.0a0） | 后端名 `pytorch-fsdp1-compat`；仅用于迁移、契约测试与数值对拍 |
| 正式训练后端 | TorchTitan v0.3.0，commit `086bf6c166ec85c1298eb5596fa9bf95f6a2d840` | 外部依赖，不 vendoring；目标 PyTorch 2.14 / FSDP2 / DTensor / DCP，版本见 `backends/torchtitan/backend.lock.json` |
| 正式 userspace | TorchTitan v0.3.0 / PyTorch 2.14 release validation 组合 | `prepare_env.sh` 原子构建 cu126 隔离环境并保存完整 `pip freeze`；`bootstrap_env.sh` 在 R525+ 主机上复用环境并执行 GPU preflight |
| Muon | 自研实现（`src/yuformer_pretrain/muon.py`，~50 行） | 配方参考 Keller Jordan 公开实现 + Moonshot 等效 lr 口径（0.2×√max(fan)）；Newton-Schulz 系数 3.4445/-4.7750/2.0315，5 步，BF16 迭代 |
| 断点协议 | ZGCM-1 `pretrain/scripts/launch_train.sh` 同款 | RESUME_ITER 校验 + latest marker，exit code 4 |
| 数据契约 | data-process S6（`scale/materialize.py`） | mix_order 顺序表 + val_sets；tokenize 由本目录 `scripts/materialize_tokens.py` 执行（S6 不落 .bin） |
| tracker | `experiments/_common/tracker.py`（本里程碑实现） | 接口见该文件 docstring |

## 兼容后端运行时版本

| 项 | 值 |
|---|---|
| Python | 3.10.12（系统 /usr/bin/python3；原 director conda env 已废弃，见 E-47） |
| torch | 2.2.0a0+6a974be（CUDA 12.3） |
| flash-attn | 2.6.3（2026-10-14 源码构建，nvcc 12.3/sm90；需 >=2.3.0，init 期 guard） |
| GPU | 4×H800 80GB（BF16 dense 峰值 989 TFLOPS/卡） |
| 运行日期 | 2026-09-26（兼容后端历史记录；正式后端另见下表） |

## 正式后端锁定版本

| 项 | 锁定值 / 状态（2026-10-22 UTC） |
|---|---|
| Python | 3.12.15，`/opt/yuformer-python/cpython-3.12-linux-x86_64-gnu` |
| TorchTitan | v0.3.0 / `086bf6c166ec85c1298eb5596fa9bf95f6a2d840`，tracked tree clean |
| PyTorch stack | torch 2.14.0、torchvision 0.29.0、torchao 0.18.0、Triton 3.8.0、torch-checkpointing 0.1.0、cu126 |
| userspace evidence | `data/health/torchtitan-env-prepare.json` = PASS，`torchtitan-userspace.json` = ready，完整依赖见 `torchtitan-env-freeze.txt` |
| host driver | 535.161.08；cu126 门禁要求 ≥525.60.13，并由 qualification 实测 CUDA、Triton、NCCL、FSDP2 与 DCP |
| GPU topology | 节点 8×H800；正式实验默认 4 卡，资格测试额外验证 4→2/8 卡 DCP reshard |
| GPU preflight | CUDA 初始化、BF16、`torch.compile`/Triton 和 NCCL smoke 已通过；完整资格状态见当前 `data/health/torchtitan-qualification.json`，必须校验 source identity |

## 2026-10-22 adapter 修复

完整缺陷、证据和边界见 [`PRETRAIN_DEEP_AUDIT_20261022`](../../report/PRETRAIN_DEEP_AUDIT_20261022.md)。
改动仅位于本仓库，外部 TorchTitan tracked tree 保持 clean：
`yuformer_ext/runtime.py` 保存逐 rank RNG/token/cursor 并验证恢复；`checkpoint.py` 接入保留策略；
`optim.py` 显式 AdamW 分组与非有限梯度阻断；`metrics.py` 隔离 eval 计数及声明 H800 峰值；
`parallelize.py` 使用上游关闭 FSDP 额外梯度除法的同一 helper。
`model/src/yuformer_model/model.py` 修复 RMSNorm 初始化与 tied 参数重复初始化。
旧训练 checkpoint/指标不能作为当前版本的验收证据。

## 2026-11-13 八卡生产验收

本节点重新恢复 Python 3.12.16 / 上述 cu126 锁定环境；当前正式资源为全部八张
H800，目标约 30 天、含预留的启动工期上限 33 天。上游 TorchTitan tracked tree
未因本轮修复改变，改动仍位于仓库 adapter、模型实现及控制脚本中。

`attention.sdpa_backend` 现在经 `YuFormerTitanModel.Config` 传到每层注意力，
cuDNN 使用显式严格上下文与连续 Q/K/V，避免进程级开关被缓存绕过或布局不支持时
静默回退稠密数学实现。`attention_parity.py` 比较 BF16 输出/梯度与数学参考，并编译
真实 4K、微批 8 的门控 GQA，通过 profiler 确认实际 cuDNN 前向和反向算子。

异步 DCP 保留策略仅统计有 `.metadata` 的完整 checkpoint。保存故障测试按具体
后台保存请求识别待完成 step，并核实中断后该 step 尚未发布 metadata；多 rank
重复日志不再被误认成第二次保存。`run_production_recovery.py` 在冻结后额外复验最终
模型、微批、编译、注意力后端和八卡拓扑下的训练/保存/加载中断、loss 与数据游标。

正式 1T 物化子进程使用锁定 Python。旧兼容 GPU 验证改为显式启用并要求空闲卡，
普通 CPU 回归不再隐式覆盖父进程 GPU mask。完整 CPU 回归为 444 PASS、28 SKIP，
证据见 `data/health/pretrain-cpu-regression.{json,xml}`。执行记录和实测边界见
[`FULL_PRETRAIN_EXECUTION_20261113`](../../report/FULL_PRETRAIN_EXECUTION_20261113.md)。

## MFU 实测记录

### 环境 A：smoke（tiny 4L / h256 / vocab32k / tied，~10.8M 参数）

合成 1MB 可学习链数据，seq_len=128，mbs=4 × accum=2，1×H800（去污染同机）：

| 模式 | 实测 | 备注 |
|---|---|---|
| AdamW + fp32 | 见下表 smoke 输出 | 100 步 |
| Muon + bf16 | 见下表 smoke 输出 | 30 步 |

> tiny 模型矩阵过小（256×256 级），MFU 无工程意义，仅验证管线数值正确性。

### 环境 B：pretrain-0.5b 单卡短程基准（BF16，seq 4096）

待测：M=block（0.5B 档 32L/h1024/vocab64k，~473M），mbs 按显存预算，
30 步去掉前 10 步预热取均值。TFLOPS = flops_per_token × tokens_per_step / t，
flops_per_token = 6·(N_nonemb + V·H) + 12·H·sum(context_per_layer)，
其中每层 context 为 full sequence 或 min(window, sequence)，输出词表投影计一次。
H800 SXM BF16 dense 峰值以 backend.lock.json 的 989 TFLOPS 为准。

| mbs | accum | step time | achieved TFLOPS | MFU |
|---|---|---|---|---|
| 待测 | | | | |

（测后回填 README"MFU 实测"节）

## 2026-10-22 夜间调度扩展

`night_research.py` 将 200M batch anchor 与完整的三规模 LR 拟合解耦，绑定完成记录、
实际预算及身份快照；`night_batch_search.py` 与两个 pilot contract 模块支持有界边界补测。
`pretrain_readiness_watch.py` 在任务完成时推进这些分支，沿用 generation lease、
semantic watchdog、checkpoint 恢复和 subreaper。
`prepare_batch_pilots.py` 在任何写入前检查整组配置，拒绝修改已开始的 batch run。
`performance_evidence.py` 支持按当前 qualification 和完整身份审计确认 batch 损失决策，
共享性能数字仍被排除。训练 backend、模型和现有 qualification 身份均未修改。

2026-10-23：静态报告新增配置/env SHA-256，readiness 使用内容绑定，修复相同 PASS
报告不刷新 mtime 导致的阶段衔接误判。实验 queue 的新任务/重试在独立 CPU 网络资格
通过后启用 PyTorch 原生 `DCP_USE_PREFIX_STORE=1`，不修改上游或训练 backend；该选择和
资格文件 hash 写入 execution context。watchdog 同时检测 checkpoint daemon 初始化失败。

2026-10-23 07:00 UTC：readiness 对 pilot 补测沿用 GPU 0/6/7 的既有受控共享策略，
并在 LR 等待时暂停 batch 对这些卡的新任务回填；运行中的 batch 不受影响。资源预留
区分新启动槽位与实际活跃 launcher，队列读取既有 allocation 文件即可生效；训练
backend、上游包及 GPU qualification 源码身份不变。

2026-10-23 10:00 UTC：新增用户排空观察器及持久暂停标记。现有训练不重载；
allocation 将新启动并发置零，guard/readiness 停止。当前 attempt 完成后只停止已捕获的
空闲队列；启动入口遇暂停标记直接退出，防止自动重启。训练 backend 与 qualification
源码身份不变。

2026-10-23 14:40 UTC：新增 `gpu_resource_policy.py`，readiness 与 queue 共用
用户 GPU 白名单；调度、allocation 读写、实际 launch 均限制 GPU 1/4，旧共享关闭。
preflight CUDA 可见设备同样收窄。改动仅在控制面，未改训练 backend、模型或资格
身份依赖；详见资源专项与完整 CPU 回归日志。

两卡恢复同步修复 guard 的启动信号竞态：完成唤醒必须匹配 handler 安装后写入的
watcher PID 注册，再发送 SIGUSR1；不改变训练或模型源码。

受限资源模式沿用 LR 先于 batch 回填的优先级，覆盖用户中断但仍有外层重试预算的
LR 补测；预算耗尽后解除该优先预留。现有训练 launcher 在控制器重载中保持不变。

2026-10-24：修复可续代 RETRY_EXHAUSTED 被启动槽位计数遗漏的衔接死锁；LR 优先级
扣除已分配但尚未开始的空槽，允许第二卡立即填入 batch。guard 增加授权空卡与
待调度任务的五分钟饥饿检测，包括另一卡正常训练时的空闲。仅控制面变化，
训练/模型及既有 GPU qualification 源码身份保持不变。

2026-10-24 scaling-only：新增 `research_scope.py` 与声明式范围配置。静态/身份 roster、
LR resolution、队列 admission、控制器 DAG、候选复验和 1T freezer 按 active scope
处理；E2/E3 保持延期。固定架构/配比以明确非比较证据的 provenance 绑定 scope hash，
最终 production validator 复核该绑定；数据、GPU 恢复、独立性能与产能门禁保持。
训练 backend/模型源码和现有训练 attempt 均未修改。

2026-10-24 四卡扩容：运行时资源策略从 [1,4] 扩为 [0,1,2,4]，现有队列读取热
allocation 后启动边界补测，当前训练不重载。production canary 显式构建四卡
CUDA_VISIBLE_DEVICES，不继承二卡 supervisor 的旧可见设备；记录实际设备列表，
不足四张授权卡直接拒绝。训练 backend/模型及 qualification 源码身份未改变。

2026-10-25：在全部 15 个匹配预算 batch 臂结束后，显式事后声明 fixed_baseline
1M-token batch 政策，仅固定 E1 基线，不作全局 batch 最优化结论。完整模式继续
要求所有 batch 曲线内部最优；固定模式仍校验全部已声明结果、精确训练预算、
身份和选中曲线的双侧 LR 证据，保留未选中 2M/4M 的边界结果及限制。scope SHA
绑定选型与 LR resolution；有界补测不再为未选中的曲线扩张。涉及
`research_scope.py`、`select_batch_pilots.py`、`batch_pilot_contract.py`、
`night_batch_search.py`、`resolve_experiment_lrs.py` 与控制面回归测试。
专项 20 PASS、完整回归 296 PASS / 1 skip；未改变训练 backend、模型或 qualification
源码身份。科学协议修订的依据与限制见 `report/PRETRAIN_1B_EXECUTION_20261022.md`。

2026-10-25 性能短测契约修复：`prepare_performance_canaries.py` 对已启动的 probe
保留原始配置和 provenance 字节；仅 `optim.lr_resolution` 注释变化允许保留原合同，
其余任何配置或环境变化一律 fail-fast，且全部五臂预检后才写入文件。此前 batch
选型报告引用哈希变化导致短测合同被重写、五份 PASS 结果被身份审计归档。本次
逐份验证所有训练字段一致、当前 backend 源码一致及原始 resolved config 精确哈希
后恢复；未修改原 identity、attempt、completion、result。独立身份审计与性能门禁
均 PASS。专项 64 PASS，记录见 `data/health/performance-contract-recovery-20261025.json`。
现有四个 E1 launcher/attempt 不变，训练 backend/模型及 qualification 源码身份不变。

2026-10-26 用户要求实验结束后释放卡：复用持久 pause、零启动额度与独立 drain
observer，停止 guard/readiness，保留当前四组，禁止新任务和重试。发现 E1 的
resolver 仍会因 batch 资格包装刷新重写 provenance，导致三组完成结果归档。
`select_batch_pilots.py` 在基础证据不变时保留原观察时间，仍移除旧包装并重新核验
资格；`resolve_experiment_lrs.py` 保留已启动合同，真实算法/数据参数变化则拒绝，
全部合同预检后才写入。恢复三份完成结果和七份已执行合同的原始 provenance，
逐字节验证 resolved hash、源码身份及完整 attempt/result 绑定；未改训练产物身份，
未中断运行中的四组。完成子集 3/3 身份 PASS，完整 E1 家族仍未完成。CPU 回归
306 PASS、1 skip，60/60 静态合同 PASS；两次真实 batch refresh 后 SHA 不变。
证据：`data/health/e1-identity-recovery-20261026.json` 与
`data/health/user-drain-request-20261026.json`。不自动恢复训练。

2026-11-02 用户要求确保自动并行流水线及 scaling 曲线产出：新增控制面
`render_e1_scaling.py`，readiness 每次分析/身份审计后导出固定 D 的 L(N)、固定 N
的 L(D)、三条 isoFLOP 家族和拟合残差，连同 CSV 点表及内容哈希 manifest。
仅展示身份审计 PASS 且有限的完成点；不完整结果标记阶段状态，24 点、拟合门禁
与完整身份审计 PASS 才允许图表标记分析通过。缓存相同输入，验证导出文件完整性，
保留非单调观测，不让图片成为生产放行依据。分析依赖声明于 analysis extra。
专项 60 PASS、CPU 回归 309 PASS/1 skip，实际三点阶段图已核验。仅重载 guard/readiness，
四个既有训练 launcher/attempt 保留；训练 backend、模型、环境包及 qualification
源代码身份不变。部署证据见 `data/health/scaling-pipeline-verification-20261102.json`。


2026-11-13 全量数据准备前的扩展修复：token materializer 改为按 reference token
与文档数预算分 part、限额批量编码，及时释放 HF Encoding 对象，并限制每个 worker
的 Arrow/BLAS/Rayon 线程数。part cache v4 绑定实现与分区参数；meta 记录 Python
与 native merge 工具 SHA。native merge 在 mmap 后关闭源 FD，并初始化超过 1024
part 后新分配的条目。1050 part、16 FD 上限与内存扰动的回归验证字节顺序和文档账。
冻结配方与 order release 的同内容重发保留原发布时间和文件，控制面恢复复用有效
配方及生产 canary；已有正式 checkpoint 时拒绝静默重冻配方或重建陈旧数据证据。
最新 CPU 回归 457 PASS / 28 skip；13 项公共 GPU 门禁、选定八卡累积梯度对拍、
生产模型八卡 training/saving/loading 中断恢复全部 PASS。独立全累积生产 canary
p10 为 74440.24 tokens/s/device，按 0.9 duty cycle 预估 1T 为 21.59464 天。
完整训练尚未启动，仍等待三段 1T 数据发布及全扫描/实际消费配比/重叠审计。
详见 `report/FULL_PRETRAIN_EXECUTION_20261113.md`。


2026-11-13 S6 运行恢复时发现旧任务遗留的验证集 seed 元数据漂移。验证集数据字节
与原发布一致；10 个 summary 经稳定验证集配置重建后逐字节匹配原发布 SHA，
19 个上游控制哈希全部恢复，随后实际 S6 work generation 的完整上游快照也与
原 100B 发布一致。新增启动前 validation artifact 大小/SHA 检查，防止后续旧任务
改写共享验证集后仍进入重型 S6。专项 E4 28 PASS、S6/lease 26 PASS；新增真实
小池回归验证四段训练 seed 不改变任何 validation 字节或数据 inode。
证据：`data/health/validation-metadata-restoration-20261113.json`、
`data/health/order-startup-regression.json`、`data/health/s6-recovery-regression.json`。


2026-11-13 将精确消费前缀配比检查前移至每段 token 化后，并覆盖缓存复用路径。
检查复用完整 overlap 审计的文档区间提取及缓存，绑定当前冻结 release、order stat、
完整 length ledger SHA，保持原绝对/相对容差。warmup 失败即 BLOCKED_MIXTURE，
阻止继续处理 800B stable；最终完整审计仍不可跳过。静态容量审计的候选 token 数
改为读取基准 manifest，与其中的 effective top-k 配对；不会把后续 mix 更深的
候选带误标为基准 top-10%。测试隔离正在重发的 S6 global summary，使用绑定原始
快照的小型容量账 fixture。专项 E4 34 PASS；完整 pretrain/E4/model CPU 回归
469 PASS / 28 skip，S6/lease 独立 26 PASS。训练 backend/model/冻结配置未改变。


2026-11-13 利用 CPU 数据排序期间的空闲八卡，将生产恢复资格从诊断累积 2 升级为
直接继承正式累积 16。四项真实训练/保存/加载故障及参考运行全部 PASS，逐 rank
游标按 4096 个全局样本/64 次 dataloader 调用核验；固定容差未改变。8 项合同/
漂移回归 PASS，原 A=2 报告已归档。训练 backend 和冻结生产配置未修改。
S6 stable 暴露 domain 字典整列展开的 Arrow 2GB 容量错误；改为 64MiB/100万行预算
分块 decode，schema/标签/选取/顺序不变。27 项 S6/lease 回归及真实 2.16GB/
2.7亿行字典溢出复现、修复后完整 Parquet 往返均 PASS，实际 stable 重跑发布待验。


2026-11-13 数据输入身份修复：S6 对 `deepen_to_quota` 使用实测候选 token 容量，
欠配域独立缓存深扫，再按原质量排序选取；800B stable 已实际发布，尾差 31,730，
配额与 1M 上限不变。doc-id 索引增加完整源文件身份、单写者代际缓存、重复 ID
拒绝和提交前复核；token worker 逐条验证源 doc_id。tokenizer 直接从同一份已哈希
字节构造，worker 拒绝非 parent 绑定 SHA，发布前再查 canonical 文件。缓存身份 v6、
source-document binding schema 2 在逐段/完整流审计与生产门禁中强制要求；原发布
验证集不重写。全量 CPU 回归 494 PASS / 28 skip，专项 70 PASS。八线程索引读取
有界预取且按原文件顺序消费。训练 backend/model/正式冻结配置未改变。


2026-11-13 E-128：order 的 SHA 与 Parquet 解析绑定同一打开文件，检查 fd/path
身份及最终发布身份，避免原子路径回滚隐藏中途错误输入。缓存 v7 / source binding
schema 3 强制 order 字节证明。78 项专项回归通过；原 S6/release/v2 全源索引复用。
测试夹具显式推进小文件 mtime，规避已复现的时间戳粒度造成的偶发假阴性；
生产 S3 仍通过临时文件原子替换发布。训练 backend/model/冻结配方保持不变。


2026-11-13 同类输入复核补齐原文文件 FD 绑定：worker 逐次核对实际打开的 inode/
mtime，再解析该句柄，读取后复核；原逐行 doc_id 与最终路径检查保留。相同 ID
不同正文的原子路径回滚已真实复现并通过修复回归。最终缓存 v8 / schema 4 要求
源文件句柄、order 字节、tokenizer 字节及源 doc_id 四项证明。82 项专项回归 PASS；
完整 CPU 回归重跑。生产核心、S6/release 和 v2 源索引身份未改变。


2026-11-13 TorchTitan reader 输入 mmap 修复：同一 fd 完成发布身份验证、edge
hash 和映射，读后复核 fd/path。正式流 meta 绑定原子发布前最终 staging inode/
device/mtime/size，1T 审计与缓存强制匹配。首尾相同内部替换与路径回滚均有
回归，86 项专项 PASS。核心 SHA 更新为 `b9b68ddc8c963da83c77f45672a71a79039cbe91c13aebe97b56b8a22b1bf1d1`，需重新进行
全部八卡数值、恢复、吞吐资格；旧 `97d928b9…` 证据保留为历史。


2026-11-13 E-131 remediation in progress: complete warmup measured 49,667,693,072 actual tokens versus 50,000,297,985 required causal-stream tokens; all-stream web weight differs by 0.0044234082 from its frozen target (unchanged absolute tolerance 0.002). Validation-domain calibration is not representative of every selected training quality band. Added phase/domain hash-prefix calibration with at least 100M reference tokens per domain in this run, actual EOD-inclusive length ledgers, immutable parent release and quality-policy binding. The freezer now requires independently remeasured phase-specific ratios and rejects stale or incomplete evidence. Focused regressions: 22 PASS. Current measurement remains in progress; production recipe/canary/data evidence must be regenerated, and full pretrain has not launched. Training core, optimizer schedule and all acceptance tolerances remain unchanged.


E-131 update (~21:25 UTC): phase-specific training calibration is now part of the authorized coordinator before freezing. It reuses valid evidence, remeasures stale evidence, and revalidates the result before a production recipe can be published. A missing selected three-phase order release blocks bootstrap explicitly. Calibration/freezer/coordinator regression: 38 PASS; early capacity/ledger/mixture blocking regression: 10 PASS. Calibration measurements hash and parse each order/length ledger through the same open file descriptor; path rollback tests cover both. The complete original warmup ledger matches the independent part measurement, while its actual capacity fails the fixed production prefix. Warmup pilot ratios crosscheck to the complete stream with maximum domain-weight error 0.000586738246. Stable and anneal measurements remain pending; full pretrain remains unlaunched.


At ~21:53 UTC E-131 implementation and full CPU regression passed: **532 PASS / 28 skip**, no failures/errors. Phase-specific calibration measured actual EOD-inclusive totals warmup=878,896,128, stable=879,678,165, anneal=874,670,399 over at least 100M reference tokens/domain/phase. The final frozen recipe SHA is `11326994dbdd0b1fc2d3cfdb1e39bb580468c1994320c8c6f8f06a6b826f7692`; reference selection budgets are warmup=52,092,016,778, stable=820,447,760,213, anneal=155,550,771,759. Actual training budgets remain 50B/800B/150B, 2% headroom, WSD steps 11,921/190,735/35,762 and 4,194,304 tokens/update. Base YAML/env are byte-identical to the previous authorized recipe. System PyArrow 12 compatibility was validated over all real measurement ledgers after using Array.to_numpy following combine_chunks. Old release/plan/canary evidence was archived rather than rebinding its hashes; the first full-suite run correctly rejected those stale active artifacts. The original insufficient full stream is retained under data/token_streams/pretrain-1t/_history/e131-original-8069ab2b. Authorized coordinator PID 186272 resumed; common GPU qualification and production recovery independently validate against current inputs. A genuinely fresh eight-GPU 200-update production canary is RUNNING; all eight GPUs showed 100% utilization. Corrected full orders/streams and final production gates remain pending, so formal pretrain has not launched.


At ~22:17 UTC the fresh recipe-bound eight-GPU canary completed **200 updates PASS**. Independent validation confirmed current source qualification identity, frozen recipe `11326994…`, runner/config/env hashes and collected result. p10 throughput **73,897 tokens/s/device**, median **73,933.3125**, MFU **25.152820587158203%**, planned duty 0.9; projected 1T training **21.75339036856494 days**, below the unchanged 33-day ceiling and 30-day target. The step-100 checkpoint was committed and training continued to step 200. Coordinator PID 186272 advanced to corrected S6 order generation; warmup holds source lease generation 14, PID 188592, and is validating upstream gates. Prior stable/anneal job PASS entries are historical and have not yet been revalidated for the new quotas; the external live view labels them accordingly. Read-only actual capacity-plan preview also PASS_STATIC with zero errors; full-tokenizer closure remains pending. No formal training has launched.


At ~23:04 UTC corrected warmup S6 exited PASS after commit/cleanup. Verified 43,669,607 docs and 52,091,983,950 reference tokens; quota shortfall 32,828 < 1M, upstream snapshot unchanged `8a243b08…`. Stable began as PID 190540 at 23:04:55 under the unchanged production S6 source `3d94d17d…`; by ~23:22 all 12,391 collect tasks completed and measured multilingual deep fill automatically rescanned 29 shards (preferred candidates 18,815,681,098 versus quota 19,789,601,438 reference tokens).

Cleanup diagnosis exposed E-132: final-domain NPZ archives were never closed, leaving thousands of live handles and NFS .nfs silly-renamed candidates until function exit. Staged fix source SHA `b44ab9098e67160781b94cca65ecafe24399695dda2d2e10f95c9a9f8f7b076a` preallocates arrays and closes each archive immediately after copying. Bitwise array comparison and a real 32-FD/96-part reproduction pass; full S6 source/caching/lease/resource regression 38 PASS. The live stable producer remains untouched/running. Only parent order controller PID 188527 is held with SIGSTOP (it owns no lease); coordinator PID 186272 and stable source lease heartbeat remain live. Automatic boundary service PID 192511 will wait for stable commit and normal lease release, verify the reviewed source hash, apply/copy tests, run actual-source 38 cases plus the full CPU suite, and SIGCONT the parent before anneal imports source. Pending hold/resume state is persisted in data/health/s6-fd-fix-phase-switch.json. Production source is still `3d94d17d…` until that boundary; no in-flight producer source bytes were replaced. No formal pretrain has launched.


At ~23:32 UTC the E-132 staged source also passed a **real NFS** canary on the same filesystem device 186 as production data. Deleting a deliberately open legacy NPZ reproduced .nfs silly rename. The bounded loader merged 64 parts under RLIMIT_NOFILE=32; process FD count remained exactly 4 before/after, and all 64 candidates plus the legacy test file were removable with zero .nfs residues. Evidence: data/health/s6-fd-real-nfs-canary.json. Source remains staged until stable producer exits; boundary service 192511 is active. Stable sorting is progressing at ~70GiB anonymous memory, with OOM counters zero.


2026-11-14 ~00:33 UTC E-132 safe-boundary update completed PASS. Stable producer exited normally and released the stage lease; reviewed source `b44ab9098e67160781b94cca65ecafe24399695dda2d2e10f95c9a9f8f7b076a` was atomically installed only after exit. Actual-source S6/caching/lease/FD regression **38 PASS**, full CPU suite **532 PASS / 28 skip**, no failures/errors. Parent controller 188527 received SIGCONT automatically, hold state is RESUMED_AFTER_VERIFIED_FIX, and anneal started as PID 197547. Current frozen recipe and eight-GPU canary bindings independently revalidated PASS; model/training core remains unchanged. Real NFS canary proves exact FD count 4 before/after under limit 32 and no residual .nfs files. Original source and CPU evidence preserved in the boundary service history. Source fix is applied and regression-verified; production anneal merge/cleanup observation and all actual M3 token-stream gates remain pending. No formal pretrain has launched.


2026-11-14 ~01:02 UTC: E-132 actual anneal producer using S6 `b44ab909…` completed merge/publication and normal exit with 12–14 observed FDs (old stable: 8,738). All three corrected reference orders PASS; derived release `yuformer-data-86f5860c47e6162e6ba0` binds frozen `11326994…`. Warmup actual-M3 encoding started; full-stream capacity/mixture gates remain pending. No training-core source changes or formal pretrain launch occurred.


## Update 2026-11-14 08:18 UTC — canonical E134/E136 integration and structured SWA breakthrough

All original B9 width/LR jobs finished. 1.38B/1.75B width cases fail hard33d at34.075/41.321; previous1.2B200steps1e-4 30.54d remains historical reference, not adopted. Main data controller199465/producer214752 continues stable~434B/6840parts by08:14, noOOM, anneal pending. Uniform diagnostic238135 warmup+stable done, anneal sample680085docs global-index lookup active; canonical caltool/report/frozen113/releaseDD28/materializer9a STILL PROTECTED AND UNCHANGED.

Canonical production code changed after all B9 research completed: E134 data.py +block_permutation.py copied from tested schema2 prototypes; config_registry now forwards explicitpolicy/manifest. Backend validator allows onlysequential/block_permutation, validatesallblockassets and adds orderidentity to run_identity. Formalproductionvalidator calls require_production_order=True, rejectssequential and PREFIX_PROBE. E136 qualification_identity now bindsactualproductionmodel/env/metas/orderassets usingproduction_dependency_identity helper, andbindsnewproduction_probe_contracts/derive helperfiles. Recoveryconstructors NOWcopyactualBASE/ENV and3stageWSD8/8/8prefixprobes(actualmodel,AdamW,seed,micro/A16); identity bindsactualprobecontentsandallhelpercode, readonlyvalidationdoesnotpublish. Canaryconstructor NOWderives actualproductionwarmup200stepprefixprobe andretains1Trelease/metas/audit/env instead oflegacy100Bsequential. No actualrecovery/canarylaunchedyet, base/finalmodelunchanged. CurrentalloldGPUreportsSTALEbysource; donotrebadge.

training_data_origin helper remains experiments/e4/scripts; backendproductiondata origin check andfullphysicalauditor nowcall resolve_data_origin ratherthanrequiringnewtrainingfreezeSHA==oldreleaseSHA. Directoriginalreleasevalidationstillstrict; candidateTRAINING_ONLY_WITH_EXACT_DATA_REUSE requiresall47oldarchiveobjects/unchangeddataquotas+consumedsets. Live tokenize_1t_orders.py UNCHANGED (199465 memorycodeactive). Fullphysicalauditoutputstillbindsrelease.DATAorigin113 only (avoidsnewtrainingfreeze circularity). FrozenADOPTpublisher stillTODO.

Full CPU regression on actual integrated code: PID253417 COMPLETE **617PASS/28skip** (645collected),170.52s, XMLdata/health/pretrain-cpu-regression.xml/logreport/logs/pretrain-cpu-regression-e134-e136.log. Extra tests includeproductionprefixprobe4, actualcanonical+prototype loader12, validator36 (formalrejectseq/probe andsameboundarraybytes), recovery8 actualWSD/modelseed. No GPUqualification waiver implied.

IndependentbuiltinTorch2.14FlexSWA8GPUoperatorpilot PASSall8:4heads24/32/40/52×2layouts, BF16nativeGQA, head64/window128; lengths65/128/129/257 againstFP32MATHoutputs+dq/dk/dv, unchangedrelL2.02/maxabsref.05tolerances. 4K B8FWD+BWD speedups9.1–10.85x (CuDNN13.25–28.45ms vsFlex1.22–2.81ms); operatoronlynotmodel. Scriptbenchmark_flex_swa.py, resultdirflex_swa_20261114, healthflex-swa-pilot.json. FirstAUTOdecodeattempt failedshortGQAblocktile128vsBLOCK_M512; alloriginalfailureevidence+exactverifiedoldsource archived `_history/flex_swa_auto_decode_failure_20261114` viaartifact_archive NFSfallback. Explicitkernel_options BACKENDTRITON resolves it; no package/env change.

Optional hybrid backend **cudnn_flex_swa** addedcanonicalattention.py andTitan model/enumvalidators. SWAusesnativeGQAblockmaskdiff<window, globalCuDNNnativecausalunchanged. Masks precomputed/shared oncorrectGPU inTitan init_states, plainnoncheckpointattributes; exactseqguard/noattentionpadding allowed, explicitautocastdtype retained, compile required. Defaults/base stillcudnn. Newactualcompiledattentionmoduleparity script pretrain/backends/torchtitan/flex_module_parity.py PASSall8independentGPU cases:actualQKnorm/RoPE/gate/QKV/o_proj output+inputgrad+ALLparametergrads againstCuDNN, heads24/32/40/52 BF16masterorFP32master/autocast, SWA129/global129/SWA4096B8. Source-boundreporthealthflex-module-parity.json, resultdirflex_module_parity_20261114; all24modulecasesPASS. These sourcebytes MUSTstayimmutablewhilecurrentnewmodelbenchmarkpinsqid.

CurrentNEWGPUfrontier: originalsupervisor252952failed aftercompletedbaselinebecauseunexpectedhostGPU0PID964467brieflybusybeforesecondcase. No unknownprocesskilled; itexitedby08:15, allGPUsidle. Baseline0.530107904B hybrid p10**114669.60156/device,14.018626d**, vsB9old74767~21.5d, firstrealend-to-endspeedup1.53x. Actualall8GPUstartupand200? THISONLY24UPDATES; not200. SourceB9no longerapplies.

Immutableplan/configs/results/runs `flex_frontier_20261114`. Scriptbenchmark_flex_model_frontier.py planpinsbench/analyzer/wrapper+actualmoduleparity+currentqid;6boundedshapesallpreserveattention_every6:baseline16L1536,18L2048/5504,14L2560/6912,18L2560/6912,20L2560/6912,24L2560/6912; B8/A16fallback4/32,2/64onlyactualOOMorallocatorpressure. NoPP. Current **resume supervisor256928** `resume_flex_model_frontier.py` safelycontinuesimmutableplanafterGPUidlefailure, archivesprevioushealthinpre-resume-supervisor.json, verifiesALLoriginalsource/plan/model/config/env/completedresultSHAandaccounting, reusesmeasuredbaselineonly. Bounded600sidlewaitbeforeeachnewcase; nosharing/unknowngpukills. Healthmodel-flex-frontier.json/logmodel-flex-frontier-resume.log, poll /usr/bin/python /opt/yuformer-build/poll_model_size_sweep.py model-flex-frontier.json. Itmaycurrentlystarting18L2048. DO NOTeditcanonicalbackend/model/qid/probehelpers/base/finalmodel/bench/analyzer whilethissweepruns; experimentdataadopter/calibrationstaging/docs/testsonlysafe. Benchbaseline_053bguardnowrequires530107904paramsbeforeprepare, preventingmislabelpostmodeladoption. NewbenchSHA differsfromoldfae; useplan.

Next: monitornewfrontier+data+uniform; finishdataorigintraining-onlyADOPTfreezer (no genericS6rebuild), stageindependentE133calbuilder/loaderpatch butDONOTapplyuntiluniform238135ends. Mainmaterializer/controller/frozen/release protecteduntilsafeexplicitadoptionboundary. Needchoosebestdepth/powerNnear30basedactualfullmodel+sustainedquality, runnewnumerics13stages/recovery24/canary200+finaldata12byteaudit/publishPRODUCTIONblockmanifest/exactmixture/overlap, thennewauthorizedcoordinatorreuse-onlylaunch. Formaltrainingnotstarted, nofinalreply, continuemonitoruserinstruction.


## Update 2026-11-14 10:33 UTC — recovery parity requires complete trajectories

Read-only review found that `compare_loss` accepted any nonempty intersection of reference/resumed loss steps. Missing middle/end records could therefore produce a false recovery PASS. It now requires every expected optimizer step `(checkpoint_step, through_step]` in both trajectories, rejects nonfinite values, and records missing steps; the original absolute-loss tolerance .02 is unchanged. Empty post-checkpoint intervals fail. Added six regression cases (missing middle, both truncated, single common step, NaN, infinity, empty interval); targeted qualification/recovery/selection suite **48 PASS**. Existing complete-trajectory test remains PASS. Runner SHA `323961b61491922897519dad0a1b1888b6c6d491764c4e5bff8f8d7ed0cb0b49`; this changes qualification identity and requires fresh actual GPU recovery, without changing training-core source `2a7e15593e5173f87c15423ae93900208b3d0098b679f0cbb291852b1c03805e`. Actual selected model/source/LR/200-step measurement independently revalidated PASS after the checker change. Historical control measurements stay scientific evidence, not current production qualification.

Own waiting post-encoding supervisor 272268 was stopped before editing its pinned runner; replacement **272679** pins the stronger checker. It is WAITING_REAL_TOKENIZATION_PASS, never launches formal pretrain, and will run full-byte audit → all-three exact temporal order recount → PRODUCTION manifest → fresh13GPU gates → actual1.52B24-step three-stage recovery. Live raw token controller199465/materializer214752 and data frozen113/releaseDD28 remain unchanged.

2026-11-14T12:16:06.219272+00:00 FULL CURRENT CPU REGRESSION COMPLETE **698PASS/28skip/1FAIL**,727collected,105.75sec. XML`data/health/pretrain-cpu-source-gated-remediation-regression.xml`, log`report/logs/pretrain-cpu-source-gated-remediation-regression.log`. SoleFAILsame`test_repository_has_no_active_config_defects`, exactly10old113freeze/plan/model/base/env/auth/cal/E1reportdrifts—notnewcodefailure; noexclusions/waivers. FinalFULLsuiteMUSTpassafterADOPT+postfreezeplan/currentgates. Newcoverageincludes6full-recovery-trajectorycasesand9immutablevalidation-source-proofcases; targeted40PASS+actualreal43-objectproofagainstgenuineexisting100BphysicalvalreportPASS, main1TphysicalauditstillNOTPASS. Mainstablepass2brawunpack1754/12993around12:15,notstalled; nohotsource/cache/scratch/producermigration. Postencoding276552WAITING, alltokensjobs3+9needrealPASS beforeautoaudits. AllnineindependentvalproofPASSimmutablearchive037365ae... acceptedbyADOPTnewgate. Currentmodelbaseenv/auth/selectionhashesunchangedfrompublished1.52B1e-4; currentcore2aunchanged. UserconversationNOFINALcontinuefullauthorizedpretraingoal.

Validation proof gate: `experiments/e4_curriculum/scripts/validation_source_proof.py` is called by training-only ADOPT and post-encoding qualification. It reads immutable CAS evidence for all nine independent source-ID-v4 full re-encodes and checks exact equality against current original validation metadata and actual admitted stream SHA, current decoder, source roster and all new document-length ledgers. Canonical validation assets remain unchanged. Production frozen inputs bind43archivedobjects+manifest+helpercode, and training_adoption.validation_source_proof records the archive. Test cases reject missing domain/ledger, false source binding, wrong roster/admitted bytes, changed canonical old metadata/decoder and corrupt archive; re-exporting a live report alias leaves archived proof valid. Actual9-domainproof169438531tokens/142332docs PASS;40targetedtests andfullCPU698PASS/28skip/1knownoldfreezeFAIL. Full 1T physical admission and actual-model GPU qualification remain separately mandatory.


## 2026-11-15：checkpoint 目录持久化与实际生产数据确认

当前完整训练源码为 `a1383a95f1e3a323fab652fc1c1be69487521e13486de2d118b789fba53acd37`。
该版本在原 f127 模型数学实现之上，仅修改 `yuformer_ext/checkpoint.py` 的 DCP 写入路径，
不修改上游 TorchTitan/PyTorch 安装。新增 writer 在元数据发布之前持久化分片目录和父目录，
在元数据原子重命名之后再次同步目录；保留原分片和元数据文件 fsync。
同步、线程异步及进程异步均显式使用该 writer，错误经原 DCP 完成通道传播。
最终 checkpoint 保留完整模型、优化器、调度器、游标和 RNG 状态。

当前后端文件 SHA 为 `6f36db7f28fc7d5eb6d72ff27f8d67d5b47d3c84081b97552e7273923211ea2e`。
`pretrain/scripts/qualify_checkpoint_durability.py` 在实际存储上完成七项检查，覆盖精确状态恢复、
发布前后目录同步顺序、同步故障传播、禁止关闭文件同步，以及两进程三种保存模式。
真实证明为 `data/health/checkpoint-directory-durability-qualification.json`，训练专用 ADOPT
校验其全部用例、当前 producer/backend SHA 及完整训练源码身份。首次检查入口因结果路径衔接
错误失败的日志和源码另行保留；修正入口后重新真实执行，未修改原失败记录。

编译 artifact cache 已进行完整归档、逐文件恢复及同配置路径的 GPU 完整状态恢复。
恢复原 step200 模型且不执行 optimizer update 时，九类原指标最大差值为 4.76837158203125e-7，
通过未改变的 1e-5 阈值；原工作 cache 目录已恢复。该检查仅证明当前环境下的恢复复现，
不能替代 GPU 故障恢复资格。

旧 100B 研究流前 838860800 个目标 token 实测含 92.5262% web 和 1.5764% 中文；
正式 warmup 的相同目标 token 数在实际 block8 顺序下含 60.0406% web 和 7.9374% 中文。
因此最终选型协议固定使用当前后端、实际生产 warmup 相同样本顺序的大小模型配对确认；
原研究数据的所有质量失败保留。协议及源码变化在正式生产数据对照开始前声明，见
`experiments/e2_architecture/results/production_matched_controls_20261115/protocol.json`。

16K 当前源码 24 步训练、验证及最终 DCP 已通过，48 条实际 LR、25 个调度参考点与最终游标
独立核验通过。64K 整层重计算仍被非有限梯度守卫拒绝；受支持的系统配置为 MLP 选择性重计算、
32 路 loss 分块和指定 GC 分配器策略，当前执行其 24 步确认。系统预检不代表长上下文能力合格。
完整当前源码八卡 GPU 资格、实际 1.52B WSD 故障恢复、训练专用冻结及最终生产 canary 仍为
正式点火的必要条件。当前事实与实际运行状态以 `report/PRETRAIN_CURRENT_GATE_STATUS_20261114.md`
及 `report/ACTIVE_PRETRAIN_OPERATIONS.md` 为准。


## 2026-11-15T04:03:52.935496+00:00: CPU-only checkpoint writer and explicit deterministic hybrid diagnostic

Actual1.5166B restart failed at its next PROCESS DCP save: the explicit TorchTitan DefaultStager bypassed FileSystemWriter.stage, leaving GPU copy-ahead enabled for already-CPU tensors. The spawned writer entered CUDA current_stream on defaultGPU0 and exhausted memory. GPU-visible CPU-state causal probes with the exact explicit stager reproduced worker CUDA initialization at copyahead10000000 and eliminated it at0, with exact DCP roundtrips. DurableFileSystemWriter now defaults copyahead0 and manager passes checkpoint_id explicitly for sync/thread/process operations; existing shard/metadata file and directory fsync ordering remains. Source after this fix plus explicit typed train.deterministic support was4a17907ba421fccabf2a913bb385768e0862ca7d32b1b7e0353f4a1a5cb2e5e6.

Two actual unbroken1.52B short-WSD runs on olda138 differed by0.9375067 loss despite no restart. This demonstrated that original random-init short-WSD loss-parity failures could not be attributed only to recovery. No.02 threshold was widened. Global deterministic mode with cuDNN SDPA failed with No available kernel, so a new explicit flash_flex_swa backend is under qualification: native PyTorch Flash SDPA for full causal layers, existing Triton Flex for SWA. Both causal/GQA/RoPE/gating semantics remain under actual output/gradient reference qualification. Backend enum, strict config validation and adapter mask preparation explicitly include this path.

Current complete source37dcb674dbf5c500f2c9e33c60f0e5943b2ab9434a2ce5ae40271cad9c73ec82. Production model/base/env/authorization are unchanged; no production numerical/kernel policy has been promoted. Actual production-width40Q/10KV FP32-master compiled module pilot passed all4 shapes with original.02/.05 tolerances and bitwise repeated forward/backward. Remaining seven real module workers and then full actual-model repeated/fault runs are executing. Canonical CPU7 current-source proof passed. Old numerical, source and quality failures are preserved; current-source full GPU/recovery and final formal canary remain required.
