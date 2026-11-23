# 1B 初筛流水线执行记录

> 2026-10-24 11:17 UTC 资源更新：用户批准再加两张空卡，当前只使用 GPU 0、1、2、4，最多四组并发。先补齐 LR/batch，选择门禁通过后四卡运行 E1 scaling；E2/E3 继续延期。下文两卡记录为历史状态。


## 当前有效范围：2026-10-24 scaling-only

用户将研究收敛为 E1 scaling；E2 架构与 E3 配比消融及两者的三 seed 候选复验延期。
`pretrain/configs/research_scope.json` 决定有效实验族与固定基线，`research_budget.yaml`
仍决定 active arm 的 token 预算。后续顺序为 LR/batch → E1（24 臂）及独立性能测试
→ 固定基线配方冻结 → 生产可行性/数据/恢复门禁。E2/E3 不再是冻结前置条件。

有效身份验收共 61 臂，含 5 个独立性能短测；当前有配置的静态验收为 56/56 PASS。
配置训练预算（不含性能短测）为 **74,910,966,339 tokens**。原方案的 30B E2、8B E3，
以及 18–36B 候选复验不再开跑，节省合计 56–74B tokens；已完成数据/实验不删除。
既有有界 LR/batch 补测仍须满足选择门禁，不把边界最优点直接宣布为已完成选型。

架构固定为现有 `pretrain-0.5b` dense 设计家族，只按 E1 的规模结论调整目标参数量；
配比固定为现有 stable 宽覆盖基线（web/code/math/chinese/academic/books/encyclopedia/
qa/multilingual = .740/.120/.040/.035/.035/.004/.005/.001/.020），沿用
50B warmup / 800B stable / 150B anneal 的领域倍率与 tokenizer 校准，不宣称有 E3
消融支持。冻结输出明确记录 `FIXED_BASELINE` 与 comparative_evidence=false；延期
实验保持 `DEFERRED`，不伪造 PASS 结果或统计优势。

scope 文件 hash 纳入 LR resolution、默认身份审计与冻结输入；冻结和最终生产 validator
均核对当前 scope。E1 获得四卡全部可用槽位，不再为延期 E2 保留一半并发。训练/模型
源码身份保持不变；当前两个活跃 attempt 在重载时保留。只允许 GPU 0、1、2、4，其余四卡
不纳入调度。四卡 production canary/必要重资格需在授权卡全部可用时执行，不因缩减研究而豁免；
本轮未启动全量 1T pretrain。

验证：完整 pretrain CPU 回归 289 PASS、1 skip；scope 与 e4 冻结专项 28 PASS，包含
无 E2/E3 结果的完整冻结、真实 E1/LR/身份/独立性能仍必须 PASS、full 模式仍要求完整
消融证据，以及延期组不能启动。专项使用隔离测试证据，不是实际生产门禁通过。
部署证据见 `data/health/scaling-only-deployment-20261024.json`。

以下保留原始排期和后续执行历史；与本节冲突的实验范围/数量以本节和机器策略为准。

2026-10-22 UTC，用户批准后已切换配置并重新启动八卡 LR pilot。目标是最终 production
validator PASS；正式约 1T pretrain 的训练时间另计。唯一预算策略为
`pretrain/configs/research_budget.yaml`。

## 执行预算与排期

| 阶段 | 训练规模 |
|---|---|
| LR / batch pilot | 原 15×0.5B + 200M 边界补测 3×0.5B；9×约 2B |
| E1 | 24 臂；L(N) 各 1B，L(D) 为 0.1/0.3/1/3/9.846B，14 个 isoFLOP 点 |
| E2 | 10 架构×3 seeds，每臂 1B |
| E3 | 8 臂，每臂约 1B；复用已有审计数据流 |
| 独立性能 | 五档模型各 32 updates，严格独占采样 |
| 候选复验 | E2、E3 各比较入选项与基线，3 seeds×3B；相同项去重 |

当前 E1–E3 初筛合计约 **80.45B** tokens；含 LR/batch pilot 与边界补测为 **107.42B**，另加
独立性能短测及 **18–36B** 候选复验。复验有 6–12 臂，整体为 100–106 臂；94 臂验收
范围覆盖补测，复验另做严格身份验收。

2026-10-22 14:57 UTC：200M 的五点搜索最优在上界（1.2e-3，held-out loss
4.116907；6e-4 为 4.120089），尚不能据此认定优势显著。在空闲 GPU 2/3/4 并行启动
1.8e-3、2.4e-3、4.8e-3，每臂 500M tokens，沿用相同 seed、数据顺序、batch、warmup、
cosine 和评测。新增配置由 `lr_pilot_search.json` 声明，独立 extension 队列负责有界重试、
训练进度 watchdog、GPU lease 与进程回收；原五个 800M launcher PID 保持不变。
选型、batch 准备、E1/E2 LR 冻结、静态合同和身份审计已改为识别声明后的 18 臂，
不会把未完成的补测略过，也不会在最优点仍位于实际 LR 上界时放行。补测约 2.5 小时，
预计 17:20–17:45 UTC 收尾；不延长当前由 800M pilot 决定的关键路径。

E1 等算力预算也已修正为按模型真实形状计算，包括输出词表投影和 attention；不能再用名义
模型大小的 `6N` 近似去宣称各臂严格等算力。三个算法 FLOPs 目标为 4.8e17/1.4e18/4.8e18，
不包含 activation recomputation；静态校验强制消费算力误差小于一次更新。

短实验 tokens/update 上限为 262,144，至少 128 次更新，并根据 batch 换算 LR。生产 batch
的选择另存，最终由四卡 canary 验证。warmup、日志、评测、保存频率随实验长度重算。
下一批队列按估计算力从大到小启动，保持现有任务，E1/E2 默认各四卡并动态释放空余资源。

八卡持续可用、门禁无需补实验时，首轮含 pilots 约 6–10 天（10 月 28 日至 11 月 1 日），
候选复验另预留 1–3 天。最终 1T 物化、分词、审计和四卡 canary 之后，生产入口目标约
2–4 周（11 月 5–19 日），慢情形预留至 11 月 26 日。CPU/I/O 吞吐和统计结果仍需实测
校准；日期不代替门禁。最终 1T 数据量不会随着实验预算缩小。

## 本次发现并修复的真实故障

原模型把 `pad_token_id=0` 传给 Embedding，导致 row 0 初始化为零并屏蔽输入梯度。
本项目 tokenizer 以 token 0 表示 EOD，同时允许 PAD 别名为 EOD。在数据流的第三个
4096-token 窗口，首 token 恰为 EOD；800M、64 次梯度累积在首次 optimizer update 前
出现非有限梯度。关闭 activation checkpointing 也复现，两步 GA=1 短测未覆盖该失败窗口。

修复为 packed LM 的所有 embedding rows 均可学习，padding 由 attention/loss masks
处理；BOS/EOS 默认语义同步到本项目 tokenizer。启动合同另校验模型 EOS 与流 EOD id。
旧 pilot 结果和 checkpoints 全部归档于 `data/experiments/_invalidated_token0_20261022/`，
不能复用于修复后的选型。原数据 release、tokenizer、E3 物化矩阵与 binaries 无需重建。

GPU 资格已重新达到 **12/12 PASS**，新增真实 800M/full-AC/64 次梯度累积的双更新检查，
其余包括数值对拍、分布式梯度对拍、DCP、训练/保存/加载 SIGKILL 和 4→2/4→8 reshard。

缩数据范围还暴露了原 scaling 拟合初值的局部解问题。现已用中心化 log 坐标及多初值求解，
已知曲线测试恢复正常；原拟合质量门禁保持不变。

## 停滞、僵尸和故障恢复

- queue 每分钟检查实际训练 step；普通日志、CPU 心跳无法掩盖训练不前进。默认启动宽限
  60 分钟、训练停滞 30 分钟，并按实际日志频率自适应增大超时，容纳大 batch 和评测/保存。
- 超时终止经过归属验证的进程树；异常退出后按 attempt id 与输出路径共同匹配，清理遗留
  子进程，防止旧训练继续占 GPU。新 supervisor 使用 Linux subreaper 并回收其子进程。
- 恢复沿用经过资格验证的 DCP、逐 rank RNG、scheduler 和数据游标协议；队列有退避和
  次数上限。重复非有限梯度最多尝试两次，随后保留阻断，不进行无限外层重试。
- `pipeline_guard.py` 每分钟执行轻量脚本检查。queue 状态超过十分钟不更新时，只重启
  coordinator，保留并接管训练；watcher 超出自身调度窗口仍无快照时有界重启。
- guard 对已结束的资格、数据与队列动作唤醒 watcher；队列完成也发送 SIGUSR1。
  稳定阶段的全面检查仍为两小时，切换阶段十五分钟，无需持续对话轮询。
- 非 GPU 后台阶段的进程树连续两小时无 CPU/I/O 进展时清理并进入有界重试。无法安全恢复
  或统计门禁失败会公开保留 FAIL/BLOCKED，不能伪装为 PASS。

CPU 故障注入已覆盖：真实子进程在启动或训练 step 之后卡住、超时清理、三次重试耗尽、
跨 session 子进程回收、无僵尸遗留、以及不触碰同 attempt 但不同输出目录的进程。

完整回归与最终恢复专项的准确计数记录在 `report/logs/research_1b_eod_final_regression.log`
和 `report/logs/research_1b_recovery_final.log`。最终状态以这些执行日志为准。

## 后续自动依赖

200M LR（含补测）与身份门禁通过即可并行启动 batch；其余 LR 锚点继续训练。
完整 LR + batch → 并行 E1/E2 与独立性能测量 → E2 分析 → E3；E2/E3 各自触发候选复验。
默认 94 臂身份审计覆盖初筛、学习率补测与性能组；新增复验另做精确 roster/代码/配置/数据/attempt 身份
审计、实际训练量和 paired-seed 九域非劣校验，全部证据 SHA-256 绑定到配方冻结。

最终数据物化、release、tokenization、overlap 审计及持续四卡 canary 按既有 DAG 接续。
实时事实见 `data/health/pretrain-readiness-latest.json`、`pipeline-guard.json` 和各组
`*-queue.json`，本记录不代替这些运行状态。

## 2026-10-22 夜间无人值守调度

用户授权夜间动态补充实验并尽量利用八卡。`night_research_policy.json` 是新增计算预算
的唯一策略文件；17:00 UTC 前已加载到守护进程，八个原有训练 launcher 均持续运行。

- 200M 独立选型绑定全部八臂的配置、数据、实际训练预算、attempt、完成记录、结果与当前
  训练源码。只有选型在实际 LR 范围内被夹住且八臂身份全部 PASS，才生成
  `lr_200m_batch_anchor.json`，启动九个 matched ~2B batch/LR 实验。
  800M 未完成不会阻塞它；800M 搜索变化不会使这个 anchor 漂移。
- LR 最优仍在边界时，自动按 1.5/2/4 倍向上或倒数向下增补。要求对应锚点的当前
  roster 全量完成并通过身份审计；正在运行的 extension 队列不修改 roster。
  自动最多新增九臂，每臂 500M，共不超过 4.5B tokens，每个规模最多九个补测点，
  peak LR 限制在 1e-6–0.02。原三个人工批准的 200M 补测不计入这个新增上限。
- 任一 batch 档的三点曲线完整后即可判断边界，并在空闲卡上自动补 2/4 倍或
  0.5/0.25 倍 LR。每臂仍训练 1,996,488,704 tokens，最多新增六臂、
  每档最多四个补测点，peak LR 限制在 1e-6–0.04。
  这些结果属于 `batch_pilot_extension`，纳入静态、选择器及身份门禁。
- 两种扩展最多额外配置 16,478,932,224 tokens；含已声明实验的配置预算上限为
  123,896,921,155 tokens，另加独立性能短测和原定 18–36B 候选复验。
  上限不是预先开跑的任务量，只在真实边界需要补证时增加。
- batch loss-only 选择必须先通过当前 qualification 和全部 batch 身份/文件快照校验，
  才能成为正式损失选型证据。共享吞吐、MFU 和耗时仍排除；E1 的独立空卡性能测量
  必须另行通过，正式配方冻结不接受未经资格验证的 provisional 结果。
- 每张卡一个训练 lease；后台每分钟检测进度与守护健康，重检查为阶段切换 15 分钟、
  稳态两小时，任务完成通过信号立即唤醒。已知异常执行有界重试并使用可用的已提交
  checkpoint；新故障或搜索预算耗尽保留失败证据并阻断对应门禁。

预期 17:20–17:45 UTC 的 200M 收尾后立即填入 batch；原五个 800M 预计在当晚至
次日凌晨陆续释放 GPU，空卡继续扩展 batch 并发。新补测会改变该预估，机器状态以
`night-research-status.json` 和队列文件为准。完整 LR/batch 通过后才并行启动 E1/E2，
E3 继续等待 E2 与 E1 校正的选择结论。

验证：夜间关键门禁 73 项 PASS；全 pretrain CPU 回归 259 项 PASS、1 项跳过，
静态契约 81/81 PASS；已有 GPU qualification 与训练源码身份保持一致。
日志：`report/logs/night_research_gates_20261022.log`、
`report/logs/night_research_regression_20261022.log`；部署与训练 PID 保留证据：
`data/health/night-research-deployment-20261022.json`。

## 2026-10-23 02:30 UTC 夜间结果与修复

原十五臂 LR 与三臂 200M 补测全部 PASS。200M 的 1.8e-3/2.4e-3/4.8e-3 最终验证 loss
为 4.126101/4.154641/4.202057，最佳仍为 1.2e-3（4.116907）；已形成范围内部的最优点。
原 800M 最佳为 6e-4（4.126462），位于原上界，系统在 23:18 UTC 自动增加
9e-4/1.2e-3/2.4e-3 三臂，分别在 GPU 2/0/1 运行，预算各为相同 500M。
五个 batch 实验正在 GPU 3–7 运行，另外四个等待；尚未进入正式 E1/E2/E3。

夜间实际出现了两项缺陷，不能把该时段视为连续八卡满负荷：

1. 200M anchor 在 17:22 UTC 已 PASS，但 batch 到 23:18 UTC 才启动，延迟约六小时。
   `validate_formal_contracts.py` 原先只在每行保存 PASS/run_name，没有配置哈希；配置改变但
   验证仍全部 PASS 时，报告内容不变且不会重写。门禁却依赖报告 mtime 晚于 resolution，
   因而一直误判，直到新增 800M 配置改变报告 roster 才偶然放行。
   已为每行绑定配置与 env SHA-256，并将 batch 与 E1/E2 readiness 改为核对完整内容。
2. `batch_1m_high` 的 CUDA launch failure 已从 step-500 成功恢复；800M 2.4e-3 的
   checkpoint daemon 在 step-250 因 EADDRINUSE 初始化失败，但直到 step-500 的下一次保存
   才传播异常，且此前无有效 checkpoint，导致约 500 步重跑。
   queue watchdog 已识别后台初始化失败日志，避免继续计算到下一保存点。
   后续新任务/重试使用 PyTorch 原生 `DCP_USE_PREFIX_STORE=1`，复用各任务现有 rendezvous
   store，避免随机空端口先查后绑定的竞争。已有训练继续使用启动时的 transport。

prefix-store transport 通过八组并发单 rank、以及一组双 rank CPU/Gloo canary；各做两次
异步 process save 与精确 load，并故意令 free-port 探测返回已占用 rendezvous 端口。
这是通信与 CPU checkpoint 资格，原 GPU 数学/恢复/reshard 资格另行保留且身份未改变。
证据为 `data/health/dcp-transport-qualification.json` 及其绑定的两个 canary report。

修复回归：263 PASS、1 skip；专项 63 PASS，静态合同 84/84 PASS。
队列 coordinator 重新加载并接管活跃 launcher，训练 PID 与源代码身份保持不变；
记录见 `data/health/pretrain-repair-handoff-20261023.json`。
原 10–14 UTC 的完整 batch 完成预估已失效；按当前吞吐，九个基础 batch 实验预计
18–20 UTC 收尾，不含尚未触发的边界补测。所有 LR/batch 选型通过后才启动正式 E1/E2。


## 2026-10-23 07:00 UTC 资源变化与恢复优先级

当前原 LR 十五臂和 200M 三臂补测共十八臂 PASS；800M 补测一臂 RUNNING、两臂
RETRY_WAIT。batch 九臂中四臂 RUNNING、一臂 RETRY_WAIT、四臂 PENDING，尚无完成的
batch 验收结果。正式 E1/E2/E3 尚未开始。五个活跃 launcher 均保留，不强制中断训练。

800M 9e-4/2.4e-3 和 batch_2m_low 新遇 CUDA unspecified launch failure；可分别使用
step-1000、step-500、step-476 的已提交 DCP 恢复。batch_1m_high 的重试已在 GPU 7
运行并继续推进。节点出现本容器 PID namespace 外的八卡任务，GPU 1–3 已有外部占用；
无法确定作业归属或结束时间，亦不能据此证明 CUDA 故障根因。检查时 ECC 计数为零。

补测分支现在可使用既有受控共享策略，仍仅允许 GPU 0/6/7，训练源码与现有 GPU
qualification 身份不变。有 LR 补测等待时，batch 的新启动槽位临时排除这些受控卡；
当前在卡上的 batch 继续计算，完成释放后由 LR 优先接续。资源预留同时追踪实际活跃
launcher，防止过早分配卡；完成后释放非启动槽位，防止空卡被旧队列长期占用。

专项 53 PASS；完整 pretrain CPU 回归 264 PASS、1 skip。
日志为 `report/logs/retry_resource_priority_20261023.log` 和
`report/logs/retry_priority_regression_20261023.log`；部署证据为
`data/health/retry-priority-deployment-20261023.json`。

前文 12–14 UTC LR / 18–20 UTC batch 完成预估失效，不再作为执行承诺；新故障和外部
占用使资源等待时间未知。重排时间以真实恢复时刻、可用槽位和最新吞吐为条件，门禁
仍要求全部配置预算完成、身份一致及合理的 LR/batch 范围内部最优点。


## 2026-10-23 10:00 UTC 用户要求排空后暂停

用户要求当前运行的实验全部结束后停止，移交集群。09:56 UTC 捕获八个正在运行的
attempt；三组较早 batch（1M low/mid、2M high）已完成，队列此前已接续其他实验。
捕获清单见 `data/health/user-drain-request-20261023.json`。

已将全部 allocation 的 max_parallel 置零，停止 guard/readiness 自动推进，保留八个
活跃训练 attempt。不会启动剩余实验，不做失败重试、不扩展搜索，也不进入 E1/E2/E3。
现有队列仅继续监测、收集结果和清理本 attempt 的进程；故障退出的任务留待恢复时处理。
独立 `drain_current_experiments.py` 每分钟检查，待捕获训练全部退出且队列完成收集，
停止空闲 queue coordinator 后自行退出。持久标记 `data/health/pretrain-user-pause.json`
从 DRAINING 变为 PAUSED；guard、readiness、queue 的新启动均检查此标记并拒绝自动恢复。
恢复必须获得用户明确指示，清除暂停标记并重新分配资源；未改动训练 backend/模型源码。

当前八组包括两组刚启动的约 2B-token batch，按观察吞吐最后一批约需 8–10 小时，
为条件估计，不是资源已释放。项目以外的进程不受操作影响。

2026-10-23 10:06 UTC：用户进一步要求两组 >90% 实验（800M LR 1.2e-3、
batch_1m_high）完成后停，<50% 的四组立即停止。已核验 launcher/env 身份并终止
800M LR 2.4e-3、batch_2m_mid、batch_4m_high、batch_4m_low 的进程树，
保留既有 checkpoint；这是用户资源交接中断，不作为算法失败或完整实验结果。
两组约 56% 的任务（800M LR 9e-4、batch_2m_low）暂保留，等待用户明确是否停止；
所有新启动及失败重试仍禁用。停止清单见 `data/health/user-selected-stop-20261023.json`。

2026-10-23 10:11 UTC：用户确认两组约 56% 的任务继续跑完，避免重算；最终保留
800M LR 1.2e-3 / 9e-4、batch_1m_high / batch_2m_low 共四个当前 attempt。
四组完成或退出后关闭空闲队列，持续 PAUSED；此前停止的四组不恢复，禁止新任务
和失败重试。已确认 drain observer 正常、两队列 max_parallel=0。


## 2026-10-23 14:40 UTC 仅 GPU 1、4 恢复研究流水线

此前保留的四个 attempt 均已 PASS；最后两组 batch_2m_low、800M LR 9e-4
分别在 14:11、14:29 UTC 完成。排空观察器于 14:30 UTC 停止队列并退出。
用户随后明确要求使用这两组原来的 GPU 1、4 继续剩余 pretrain 前实验，其他六卡
留给另一用户。此指令替代暂停要求，不扩大为全量 pretrain 的启动授权。

`data/health/gpu-resource-policy.json` 为当前资源真源：allowed_gpu_indices=[1,4]，
max_concurrent_jobs=2，allow_busy_gpu_sharing=false。readiness 在读取 GPU 快照后立即
过滤；空卡选择、旧共享回退、allocation 写入、queue 的 CLI/热更新和每次新启动
均检查这份限制。旧 GPU 0/6/7 共享策略已禁用，其他六卡即使空闲也不会纳入本项目。
preflight 的 CUDA_VISIBLE_DEVICES 同样限制为 1,4，不在其他卡建立 CUDA context。

后续先恢复用户中断的 800M LR 2.4e-3（有效 step-500 DCP）及尚未完成的 batch；
已完成 PASS 结果不重跑。有界 LR/batch 补测、E1/E2、E3 与三 seed 复验按既有内容/身份
门禁推进，两卡共享同一个 GPU-slot lease，任意时刻至多两个训练任务。单卡性能
canary 可在这两卡顺序运行；四卡 production throughput canary 和新的四/八卡
qualification 不具备资源条件，保持等待，不借用另外六卡，也不降低验收口径。
没有自动全量 pretrain 启动动作。

验证：资源限制专项 61 PASS，完整 pretrain CPU 回归 272 PASS、1 skip，静态契约
84/84 PASS。只修改控制面，不改变训练 backend/模型与既有 qualification 源码身份。
部署记录：`data/health/two-gpu-pipeline-deployment-20261023.json`。
旧暂停标记已归档为 `pretrain-user-pause.completed-20261023.json`；新两卡 guard 接续监控，
延续每分钟进度检查、稳态两小时/切换十五分钟重审、完成信号唤醒和有界恢复。

恢复时发现 guard 在创建新 watcher 的同一轮，可能因旧队列退出而在 handler 注册前
发送 SIGUSR1；默认处理会终止新 watcher。已改为只唤醒 PID 与 wakeup 注册记录一致
且命令身份匹配的 watcher，注册写入本来就在信号 handler 安装之后。未注册时延后
通知，下一轮仍可补发。专项 25 PASS，guard 已重新加载此修复。

两卡恢复的实际观测：GPU 4 已启动 batch_2m_mid；GPU 1 出现非本项目占用，
LR 等待该卡空闲，未借用其他六卡。受限模式下，batch 保留当前运行任务但暂停
新回填，把下一个释放槽位优先给待补齐 LR；LR 外层重试预算已耗尽时不会继续
占用这份优先权。单卡资源暂不可用时不会把“两卡上限”误报为实际两卡并发。

最终验证：275 PASS、1 skip；资源/衔接专项 60 PASS；当前静态合同 86/86 PASS。
1M batch 三点完成后最优仍处上边界，既有有界策略已声明 9.6e-3、1.92e-2 两臂
补测（每臂相同约 1.996B tokens），尚未启动；配置总量现为 99 臂、
112,910,966,339 tokens，不含性能短测和候选复验；仍在既定最多六臂 batch 扩展上限内。
控制器重新加载已确认保留 GPU 4 的活跃 launcher PID/attempt 不变。


## 2026-10-24 01:51 UTC 两卡衔接死锁修复

夜间并未实现两卡持续运行。batch_2m_mid 于 23 日 23:55:38 UTC 完成并 PASS，
基础 batch 共 6/9 PASS；LR 共 20/21 PASS，未完成的是人工中断的 800M 2.4e-3。
GPU 1 在 23 日 14:47 UTC 起的全部后续快照均空闲；batch_2m_mid 完成后，GPU 1、4
同时空闲到本次修复约两小时。旧报告关于 GPU 1 外部占用仅适用于 14:43 的观测，
不能解释此后的空闲；这是两卡调度实现遗漏，不能把该时段记为稳态运行。

根因是 publish_queue_allocation 将 RETRY_EXHAUSTED 从 remaining 中剔除，即便
该任务仍有外层重试预算、且用户已授权继续。LR 因 limit=0 永远无法重启自己的
coordinator；batch 优先级又为同一任务让出启动额度，形成无训练进程但队列存活的
死锁。现有进程存活/写状态检查无法覆盖这类异常。

修复将尚可续代的任务计入可分配工作量，外层预算耗尽和 nonfinite_gradient 仍阻断；
并在 LR coordinator 已获得槽位、但 job 状态尚未更新的窗口扣除已有空槽，允许第二卡
立即跑 batch。新增两卡衔接回归覆盖：LR RETRY_EXHAUSTED → coordinator 重启分配
GPU 1 → 同一轮 batch 分配 GPU 4，以及外层重试耗尽后不再发卡。

guard 新增每分钟的低成本槽位检测：授权卡空闲且有任务等待五分钟，就唤醒已注册
的 watcher；另一张卡正在训练也不会掩盖此信号。正常训练继续维持低频重审，唤醒
带五分钟冷却。此机制不扩大 GPU 白名单，不终止其他进程，也不重置重试预算。

01:51 UTC 已在 GPU 1 恢复 800M 2.4e-3，实际加载 step-500 DCP 后从 step 501
训练；GPU 4 启动 batch_4m_high，首步损失为有限值。正式 E1/E2/E3 仍未开始。
专项 65 PASS；部署与恢复证据见 `data/health/two-gpu-handoff-repair-20261024.json`。


## 2026-10-24 11:17 UTC 四卡加速基础实验，再进入 E1

用户批准增加两张空卡，随后明确先四卡跑完 scaling 前的基础实验，再四卡运行 E1。
已通过 nvidia-smi 确认新增 GPU 0、2 无计算进程、显存占用零；GPU 1、4 的现有训练
保留。资源白名单扩为 [0,1,2,4]、max_concurrent_jobs=4，禁止忙卡共享；其他四卡
[3,5,6,7] 不分配。基础 batch 现为 7/9 PASS，4M low/mid 留在 GPU 1/4；两组已声明的
1M LR 边界补测由原 extension coordinator 自动接续到 GPU 0/2，不额外增加实验臂。

E1 仍等待全部匹配 LR/batch 证据与选型门禁通过，随后可用四卡分别运行独立实验臂；
不把同一单卡科学契约擅自改为四卡 DDP。E2/E3 和复验继续 DEFERRED。
四卡 production canary 在科学与数据门禁齐备、四张授权卡均空闲时具有资源条件；
修正 canary 启动环境，显式绑定当前四个授权物理 GPU（含非连续编号），避免继承旧
二卡 CUDA_VISIBLE_DEVICES 使 world_size=4 启动失败或误用未授权卡。设备列表进入
canary 报告，训练源码/已有 qualification 身份不变，全量 pretrain 仍是后续阶段。

资源扩容/非连续 GPU canary 映射专项 25 PASS；原有两个训练 PID/attempt 保留。
部署记录：`data/health/four-gpu-expansion-20261024.json`。

四卡扩容最终回归：292 PASS、1 skip。已确认 GPU 0/1/2/4 均有本项目活跃训练，
新增两组边界补测已输出有限损失；原 low/mid 的 PID 与 attempt 未改变。

## 2026-10-25 06:43 UTC：固定 batch 基线的事后协议修订

LR 21/21、基础 batch 9/9、已声明 batch 补测 6/6 均已完成。全部 batch 臂实际
训练 1,996,488,704 tokens。此前要求三个 batch 曲线全部找到内部最优点；既定六臂
补测预算耗尽后，2M 最优仍在 LR 上边界，4M 最优仍在下边界，导致 E1 无法接续。
最后一个补测于 06:04:45 UTC 完成，06:36 快照仍未启动 E1。

为落实只保留 scaling 并尽快进入该阶段的范围要求，现显式将 1M tokens/update
（精确 1,048,576）固定为 E1 的 batch 基线。这是看过 pilot 结果后的协议修订，
不是预注册的 batch 最优化结论。1M 曲线 LR=0.0096 的验证 loss=3.574015，
两侧 0.0048、0.0192 的 loss 分别为 3.584412、3.580525；保留完整五点曲线。
2M/4M 的未收敛搜索与边界最优继续出现在结果中，不声称全局最优 batch，
也不把单次运行的损失差称为统计显著优势。

1M 是 pilot 选型和 LR 迁移的参考 batch。既有 1B screen 策略仍把 E1 实际
tokens/update 限为 262,144，按既定平方根规则迁移 LR，保证短预算实验有足够
更新步数；`lr_resolution.json` 同时记录参考与实际 batch，不把它们混写。

`research_scope.json` 声明 fixed_baseline 政策，选型及 LR resolution 绑定其 SHA-256。
仍要求所有 15 个匹配预算结果齐备、身份一致、选定 1M 曲线有双侧证据，以及当前
GPU qualification；独立性能短测、E1 拟合、数据及生产 canary 门禁保持。默认完整
batch 搜索模式仍对任意曲线的边界最优返回 BLOCKED；本次仅在声明的固定基线
模式下解除未选中曲线的阻塞，不删除结果、不强制改写 PASS。

控制面专项 20 PASS，完整 pretrain CPU 回归 296 PASS、1 skip；日志为
`report/logs/fixed_batch_baseline_regression_20261025.log`。06:47 UTC 发出控制器
重载请求，部署证据见 `data/health/fixed-batch-baseline-deployment-20261025.json`。
训练 backend、模型源码及已有 qualification 身份均未修改。

06:47:59 UTC 新控制器确认 LR 21/21、batch 15/15 正式 PASS，LR resolution 当前有效，
60/60 静态契约 PASS；五组独立性能短测开始。06:51:46 UTC 已有 100M、200M、50M
短测 PASS，E1 队列随空卡自动启动：GPU 2 为 `e1_ld_300m_9p85b`，GPU 4 为
`e1_ln_800m`。GPU 0/1 继续 800M/400M 短测，完成后可回填 E1；未使用其他四卡。

## 2026-10-25 08:51 UTC：四路 E1 与短测证据恢复

GPU 0/1/2/4 均已运行 E1，四个 launcher/attempt 从启动至本次检查未重启。
当前为 300M/3B 数据量点、800M isoFLOP 点、300M/9.85B 数据量点、800M/1B
模型规模点，日志进度分别约 9.35%、11.24%、3.12%、9.96%。300M 已写出 step-1000
DCP checkpoint，损失与梯度均为有限值。

五个性能短测训练均完成 PASS，但此前后续 batch 选型报告改变引用 SHA，生成器
重写 `optim.lr_resolution.batch_selection_sha256`，使身份审计正确拒绝了变更后的
合同并归档原始结果。此次修复生成器的已启动合同不可变性：保持原始 provenance，
训练参数/环境变化则拒绝，五臂全部预检后才写文件。逐份核对除 provenance 外
所有字段相同、原 resolved config 字节 SHA 完全一致、训练源码与资格仍有效后，
恢复原始结果，没有改写 run identity、attempt、completion 或 result。

独立身份审计 5/5 PASS，独立性能证据 PASS，队列由正常入口重新汇总为 PASS，
不重跑短测、不打断 E1。恢复证据见 `data/health/performance-contract-recovery-20261025.json`，
性能报告见 `report/independent_performance_20261025.json`。

修复后 CPU 回归 301 PASS、1 skip；新增测试覆盖 provenance 变化保持原合同、
LR/batch/长度/重计算变化拒绝，以及整组预检失败时不部分改写。原有 scaling
合成测试显式隔离真实性能文件，避免本机短测完成后改变预期 provisional 状态。
日志：`report/logs/performance_contract_cpu_regression_20261025.log`。08:53 UTC 四卡
利用率 99%～100%，E1 为 4 RUNNING、20 PENDING、0 完成，五组性能门禁均 PASS。

## 2026-10-26 04:04 UTC 用户要求完成后释放 GPU

本次检查 E1 已有 3 PASS：`e1_iso_c2_800m`、`e1_ld_300m_3b`、`e1_ln_800m`，
分别于 00:11、02:35、02:52 UTC 完成；原流水线已在对应卡上接续新任务。
当前仍有 4 RUNNING、17 PENDING。用户要求释放跑完的卡，已立即写入持久暂停
标记，所有 allocation max_parallel=0，停止 guard/readiness，禁止继续回填、扩展
和失败重试。四个正在运行的 launcher/attempt 保留，由既有 queue 继续监测和
收集结果；每个任务结束即释放对应 GPU，不启动其他实验。

独立 drain observer 每分钟检查，待全部捕获任务退出且结果收集结束，停止空闲
coordinator 后自行退出。request 与进度见 `data/health/user-drain-request-20261026.json`
及 `data/health/pretrain-user-pause.json`。17 个待运行实验保留为 PENDING；恢复需要
用户新指示。当前资源尚未全部释放，不将已完成但随后接续训练的卡称为空闲卡；
不干预项目外进程，不改训练代码与已完成结果。

排空时补充修复 E1 provenance 重写：已完成三组的训练产物在归档中完整保留。
逐字节验证配置身份、当前训练源码和 attempt/result 后恢复，完成子集身份复核
3/3 PASS；为四组正在运行实验同步恢复原执行合同的 provenance，不改变训练。
resolver 增加已启动合同不可变性，batch selector 的资格刷新保留基础证据的原
观察时间；实际连续两轮刷新 SHA 不变。CPU 回归 306 PASS、1 skip，静态 60/60
PASS。当前四组继续由原 queue 监控；自动推进保持关闭，未消耗 GPU 重跑完成项。

## 2026-10-26 05:01 UTC 用户要求立即终止并释放卡

用户明确改为立即停止四组当前实验，优先让出 GPU。已停止 drain observer，通过
既有 queue 的 SIGTERM 收尾终止四个精确归属的 launcher 子树，随后确认捕获的
训练进程无存活者，GPU 0/1/2/4 利用率均 0%、显存仅 5～7 MiB，未操作其他用户
进程。持久 pause 保留，guard、watcher、queue 和 observer 均已退出，不自动重启。

三组已完成 E1 结果保留；四组中断任务标记为人工停止、关闭自动重试。
保留现有日志和已提交 checkpoint：100M isoC2 为 step-7000，200M isoC2 为
step-1000，300M/9.85B 为 step-13000；400M isoC2 尚无已提交 checkpoint，
不得宣称四组均可从中断点恢复。未等待下一次保存而继续占卡。
停止清单与 GPU 释放证据：`data/health/user-stop-request-20261026.json`。

## 2026-11-02 11:49 UTC 恢复 scaling 自动队列

用户明确恢复快速并行 scaling。重新验证 60/60 静态合同、当前 GPU qualification、
LR/batch resolution 与五份独立性能证据；身份审计 44 个已完成臂 PASS，21 个 E1
未完成臂仍阻塞整体门禁。三组已完成 E1 不重跑；100M/200M isoC2 与 300M/9.85B
可分别从 step-7000/1000/13000 续训，400M isoC2 尚无已提交 checkpoint，需重跑。

已归档旧 pause 标记，恢复 guard/readiness 与有界重试，保留原 GPU [0,1,2,4]
四卡白名单、空卡准入和禁止共享策略；E2/E3 继续延期，不启动全量 pretrain。
本次四张授权卡被其他用户评测进程占用。最初 GPU 6/7 尚空闲，资源切换确认
期间该评测已占用全部八卡，未擅自借用或清理外部进程。11:49:50 UTC 新 snapshot
确认 E1 为 WAITING_EXTERNAL / experiment_gpu_allocation，监控正常；授权卡空闲
后自动准入，最多四路。当前没有把恢复队列报告为已经恢复 GPU 训练。

恢复证据：`data/health/scaling-resume-20261102.json`；旧暂停标记归档于
`data/health/pretrain-user-pause.completed-20261102.json`。

## 2026-11-02 自动流水线及 scaling 结果导出核验

资源释放后，原四卡队列于 12:36 UTC 自动接续。100M/200M isoC2 与 300M/9.85B
按原 checkpoint 续训，400M isoC2 从头重跑。四路独立任务沿用空卡准入、GPU/run
generation lease、有界失败重试、semantic stall 检测、DCP 恢复与完成后自动补位。
guard 每分钟进行轻量检查；稳定阶段完整审计间隔两小时，完成事件触发唤醒，不靠
LLM 高频轮询。只使用 [0,1,2,4]，最多四路；其他卡保持预留。

检查发现原分析器只有机器可读拟合结果，没有自动导出图表。新增 renderer 并挂接
readiness 的分析/身份审计阶段：自动输出 `curves/scaling_curves.png`、SVG、
`endpoints.csv` 与 manifest，覆盖 L(N)、L(D)、三条 isoFLOP 和联合拟合残差。
当前仅三个已审计终点，图表明确为 EXPLORATORY_PROGRESS；不会拿运行中 loss 冒充
最终验证点，不删非单调数据，不提前绘完整拟合。24 点与既定拟合/身份门禁均 PASS
才标记分析通过；失败拟合不冻结生产配方。bootstrap 是跨点稳定性估计，不冒充
重复 seed 的统计方差，1T 预测仍属于对观测 token 范围以外的外推。

专项测试 60 PASS，完整 CPU 回归 309 PASS/1 skip。已实际生成并检查三点阶段图；
输入相同时不重绘，文件损坏时重建。控制面重载保留四组训练进程与 attempt，未
改变训练/模型/环境包或已有 qualification 身份。

## 2026-11-07 用户授权四卡运行 E1 三 seed 复验

用户要求先扎实完成 scaling，再进入全量预训练，并授权 GPU 0–3 用于 scaling。
当前 scope 仍为 scaling-only，资源策略最多四个并发、禁止 GPU 共享，
`full_pretrain_launch_allowed=false`。E2/E3 继续延期。

复验覆盖原 E1 的全部 24 个设计点：保留 seed 20261001 初筛结果，并对每点补跑
20261002、20261003 两个配对 seed，共 48 个新增 run。冻结合同核对了 paired-seed
名册、每点完整配置与源点除 run 名/seed 外一致，以及 84,899,181,190 个新增
目标预算 token；分析器另强制该总量与 `research_budget.yaml` 预注册上限一致。
按完整 optimizer step 取整后，预计新增实际训练量为
84,893,761,536 tokens，目标与实际之间的短差逐臂小于一个 step；scale fit 使用
实际训练 token，目标预算与短差分别留档。

复核修正了 seed-aware 分析路径的缺口：基线与复验共 72 个 run 的身份审计、
逐点三 seed 配对及完整配置配对、目标预算与实际训练 token 的口径。拟合前任一
审计不通过都会阻断；成功报告记录身份/预算/seed 设计审计。合成 24 点×3 seed
端到端拟合通过。随后发现原 bootstrap 在每个设计点独立抽 seed，会破坏已配对的
跨点协方差；现改为每次拟合对所有 24 点共用一组 seed 抽样索引，并同步更新
预注册方法声明、图表 claim limit 和回归用例。确定性回归验证所有设计点共用一次
seed 索引抽样；分析器与 renderer 相关回归 13 项通过，预算短差及 bootstrap 方法
断言均通过。分析/渲染
工具变更不在训练源码 identity hash 覆盖范围内；四个活跃 run 的 source hash
已与当前训练源码核对一致。

截至 2026-11-07 10:12 UTC，队列为 4 RUNNING、44 PENDING、0 完成/失败；GPU 0–3
利用率 99–100%，显存占用约 19–35 GiB/卡。两个 300M/9.85B 复验任务到 step 2620，
两个 800M/1B 任务到 step 850，均为首次 attempt。队列约每分钟更新状态，训练日志
持续前进。readiness watcher 未运行；全部任务结束后需手动触发分析、身份审计和曲线
导出。09:56–10:12 的进程快照未发现 readiness watcher 或全量预训练 launcher，1T
预训练仍未启动。

当前可用的 screen-only 拟合 `scaling_fit_screening_20261107.json` 为 FAIL：R²=0.9807、
LOO 中位数误差=2.77%、p95=8.15% 均过线，但 normalized RMSE=0.03638，高于 0.03
门槛。它不是三 seed 最终拟合；不得覆盖或放宽该门槛来宣称通过，待 48 个复验结果齐备后
用三 seed 均值重新拟合并按完整门禁裁定。对主要残差点作只读抽查：800M/约 252M-token
isoFLOP 臂及 300M/100M-token 臂均为 PASS，实际 step/token 与配置一致，九域验证值齐全，
最终 loss 对应最终训练 step；目前未发现训练计数或陈旧验证值异常，仍需最终三 seed 结果
区分随机波动与拟合形式不足。

09:20 UTC 修复配对 seed bootstrap：每次 bootstrap 对 24 个点同步抽取相同的 seed
索引，保留跨点协方差；更新分析配置、图表 claim limit 和测试。分析与导出回归
共 13 项通过，其中包括固定 seed 索引验证和合成 24×3 seed 分析 fixture。该修正只涉及分析方法，不改训练
配置和训练源码身份。

09:35 UTC 将 seed-aware bootstrap 的固定 RNG seed（20261107）写入拟合 JSON，便于
对参数区间完全复算；同一分析/导出回归重跑 13 项全部通过。

09:37 UTC 补齐 E1 研究设计要求的 compute-optimal 结论：拟合报告按三条 isoFLOP
家族计算 `C=6ND` 下的解析 `(N*,D*)`、`D*/N*` 与 paired-seed bootstrap 区间；renderer
新增单独的 PNG/SVG 前沿图，明确标注它是拟合模型的解析结果而非直接观测。相关分析与
renderer 回归共 13 项通过，含合成端到端前沿计算和图表导出断言。

09:56 UTC 复核训练 coordinator PID 64527 仍存活（运行约 4h13m）；队列为 4 RUNNING、
44 PENDING。两个 300M/9.85B 臂到 step 2470，两个 800M/1B 臂到 step 800，全部 attempt 1；
四卡利用率 99%，2×300M 臂显存约 34.6 GiB/卡，2×800M 臂约 19.2 GiB/卡。

09:42 UTC 对 compute-optimal 解析式的 float64 下溢路径补保护：不可表示的 bootstrap
最优点返回无效样本，避免 `N*=0` 后 `D*=C/(6N*)` 除零。分析与 renderer 回归 14 项通过，
其中含该极端参数边界用例。

09:45 UTC 增加 compute-optimal 解析式直接验证：合成参数下核对闭式 `N*`、`6N*D*=C`
及 `D*/N*`；分析/renderer 专项套件 15 项全部通过。

09:47 UTC 修复 renderer 的结论展示门禁：即使 bootstrap 前沿数值有效，只要 24 点拟合质量
或身份审计未通过，compute-optimal 图仍显示 pending，并在 manifest claim limit 明确说明
门禁未通过。专项回归扩展到 16 项全部通过。

09:49 UTC 将前沿图完整性再收紧为 c0/c1/c2 三档全部存在且 bootstrap 区间均有效；缺任一档
就展示 pending 并在 claim limit 说明。manifest 新增机器可读的前沿状态和有效家族列表；
分析/renderer 专项回归共 17 项全部通过，随后 renderer 六项回归针对该 manifest 再通过。

09:56 UTC 固定 roster 审计检查发现显式传入 24 个当前配置名可能绕过身份审计器的规范 E1
名册。新增共享精确名册校验并接入复验准备器和 seed-aware 拟合；当前 24 个配置与标准
名册精确相同。更新测试夹具后，分析与 renderer 专项回归 18 项全部通过。

09:59 UTC 将同一精确 roster 校验下沉到身份审计器的公共 `audit(..., groups=...)` 入口，
使其他显式分组调用也不能替换 E1 点名册；定向 API 用例及完整 18 项专项回归通过。

10:00 UTC 对冻结合同做只读重核：标准 base roster 24/24，replication 48/48 且每点恰好
包含 seed 20261002/20261003；声明、契约、配置与 policy 上限均为 84,899,181,190 tokens。
按整步训练预计实际复验量为 84,893,761,536 tokens，累计取整短差 5,419,654 tokens。

10:02 UTC 发现上述 bootstrap 方法更新造成冻结 contracts 中 `research_budget.yaml` 输入哈希
漂移。该更新发生在复验队列启动之后，故作为显式 analysis-method amendment 留档，不再称为
预注册：记录旧/新方法、修订理由、时间及两版 SHA-256，并更新合同对应输入哈希。修订依据
是固定配对 seed 设计；没有查看复验最终验证结果。训练臂配置、seed、预算和已运行任务均未改。

10:05 UTC seed-aware analyzer 增加 policy/实现一致性检查，若 `research_budget.yaml` 的
bootstrap 声明与代码方法不同则 BLOCKED；报告同时保留该方法修订在 contracts 中的哈希证据。
更新后的分析/renderer 专项回归 18 项全部通过；新增 policy mismatch 阻断用例后，完整
专项回归 19 项全部通过。
最终 seed-aware fit JSON 也将内嵌该 `analysis_method_amendments` 记录；合成端到端用例通过。
准备器、身份审计器、seed-aware 分析器和 renderer 的 `py_compile` 均通过。
10:09 UTC 新增冻结 policy 哈希过期阻断用例，确认输入哈希漂移不会进入拟合；分析与 renderer
专项回归共 20 项全部通过。
10:11 UTC 复验准备器增加冻结预算 policy 绑定保护，policy 哈希变化时拒绝静默重建合同；
端到端回归验证该拒绝路径，专项套件扩展到 21 项全部通过。

10:16 UTC seed-aware metric collector 增加对总 loss 与九域 loss 的 `last_step == expected_iters`
检查，防止拟合使用陈旧 held-out 指标。24 个现有 base 结果逐项只读核对全部是最终 step；
陈旧 step 用例、合成三 seed 端到端用例及分析/renderer 专项套件共 22 项通过。

10:17 UTC 复核当前 analysis-method amendment：旧/新 SHA-256 均保留，新哈希与 policy 文件及
合同输入一致；声明方法与 analyzer 实现一致，且 amendment 明确记录训练预算/config 未改。

10:12 UTC coordinator PID 64527 仍存活（运行约 4h28m）；4 RUNNING、44 PENDING，全部
attempt 1。两个 300M/9.85B 臂到 step 2620，两个 800M/1B 臂到 step 850；四张卡利用率
99–100%，训练持续前进。当前没有 seedrep 完成结果，readiness watcher/全量预训练 launcher 均未运行。

10:27 UTC coordinator PID 64527 仍存活（运行约 4h44m）；4 RUNNING、44 PENDING，全部
attempt 1。结构化训练日志的最新 step 为 300M/9.85B 两臂 2777/2779、800M/1B 两臂
899/902；GPU 0–3 利用率 99–100%，显存占用分别约 34.6/34.6/19.2/19.2 GiB。队列持续更新，
尚无复验完成结果；readiness watcher 与全量预训练 launcher 均未运行。

10:43 UTC coordinator PID 64527 仍存活（运行约 4h59m）；4 RUNNING、44 PENDING，全部
attempt 1。结构化训练日志的最新 step 为 300M/9.85B 两臂 2926/2928、800M/1B 两臂
947/950；GPU 0–3 利用率 99–100%，显存占用约 34.6/34.6/19.2/19.2 GiB。四项均在前进，
尚无复验完成结果；未发现 readiness watcher 或全量预训练 launcher。

10:45 UTC 静态复核 `run_experiment_queue.py` 的接续路径：活跃臂结束后先收集并验收结果，
每 5 秒检查空槽，并按配置估算计算量从大到小启动可运行任务；当前四卡均已分配，未发现
需要为保持 GPU 满载而重启或改写调度器的理由。

10:58 UTC coordinator PID 64527 仍存活（运行约 5h15m）；4 RUNNING、44 PENDING，全部
attempt 1。结构化训练日志的最新 step 为 300M/9.85B 两臂 3076/3078、800M/1B 两臂
996/1000；GPU 0–3 利用率 98–100%，显存占用约 34.6/34.6/19.8/19.8 GiB。任务仍持续前进，
尚无复验完成结果；未发现 readiness watcher 或全量预训练 launcher。

11:15 UTC coordinator PID 64527 仍存活（运行约 5h31m）；4 RUNNING、44 PENDING，全部
attempt 1。结构化训练日志最新 step 为 300M/9.85B 两臂 3236/3239、800M/1B 两臂
1046/1050；GPU 0–3 利用率 93–100%，显存占用约 34.6/34.6/19.8/19.8 GiB。四个任务仍前进，
尚无复验完成结果；未发现 readiness watcher 或全量预训练 launcher。

11:16 UTC 收尾 runbook 复核发现它仍调用单 seed `analyze_e1_scaling.py`，并将 screen-only 拟合
当作最终结果路径；这会漏掉本次 24×3 seed 复验目标。已修正为 `analyze_e1_seedrep.py`、
`seed_replication_fit.json`，并在 shell 流程启用 `set -euo pipefail`，让分析失败时不继续渲染；
E1 README 同步说明 screen fit 不能替代 seed-aware 最终 gate。当前 queue 仍未到收尾条件，未运行分析。

11:30 UTC coordinator PID 64527 仍存活（运行约 5h47m）；4 RUNNING、44 PENDING，全部
attempt 1。结构化训练日志最新 step 为 300M/9.85B 两臂 3389/3392、800M/1B 两臂
1095/1099；GPU 0–3 利用率 99–100%，显存占用约 34.6/34.6/19.8/19.8 GiB。任务持续前进，
尚无复验完成结果；未发现 readiness watcher 或全量预训练 launcher。

11:45 UTC coordinator PID 64527 仍存活（运行约 6h02m）；4 RUNNING、44 PENDING，全部
attempt 1。结构化训练日志最新 step 为 300M/9.85B 两臂 3537/3539、800M/1B 两臂
1144/1148；GPU 0–3 利用率 99–100%，显存占用约 34.6/34.6/19.8/19.8 GiB。四个任务持续前进，
尚无复验完成结果；未发现 readiness watcher 或全量预训练 launcher。

12:01 UTC coordinator PID 64527 仍存活（运行约 6h17m）；4 RUNNING、44 PENDING，全部
attempt 1。结构化训练日志最新 step 为 300M/9.85B 两臂 3689/3691、800M/1B 两臂
1193/1198；GPU 0–3 利用率 96–100%，显存占用约 34.6/34.6/19.8/19.8 GiB。任务持续前进，
尚无复验完成结果；未发现 readiness watcher 或全量预训练 launcher。

12:03 UTC 以 48 个复验配置及 scheduler 同一 FLOPs/token 公式估算总工作量约 1.35e20 FLOPs
（84.90B configured tokens、323,844 optimizer steps）。11:45–12:01 的活跃 300M 与 800M
任务 step 增速折算约 84/77 TFLOP/s/GPU；若其余规模平均效率相近，完整复验约需 4.9 天，
从当前时点约剩 4.6 天。由于尚未观测其他模型规模的持续 MFU/step 吞吐，此值仅作规划估计，
并预留到 5–7 天量级；以队列真实完成速度更新，不据此触发全量预训练。

12:04 UTC 核对当前 `gpu-resource-policy.json`：ACTIVE，仅允许 GPU 0–3，最多四并发，禁止共享，
`full_pretrain_launch_allowed=false`，授权范围为 scaling-only。磁盘剩余约 333T，inode 使用率
59%；当前四个复验任务各自结构化日志约 237–288 MiB，存储空间不是当前队列瓶颈。

12:16 UTC coordinator PID 64527 仍存活（运行约 6h33m）；4 RUNNING、44 PENDING，全部
attempt 1。结构化训练日志最新 step 为 300M/9.85B 两臂 3841/3843、800M/1B 两臂
1242/1247；GPU 0–3 利用率 98–100%，显存占用约 34.6/34.6/19.8/19.8 GiB。四个任务持续前进，
尚无复验完成结果；未发现 readiness watcher 或全量预训练 launcher。

12:31 UTC coordinator PID 64527 仍存活（运行约 6h48m）；4 RUNNING、44 PENDING，全部
attempt 1。结构化训练日志最新 step 为 300M/9.85B 两臂 3990/3992、800M/1B 两臂
1291/1295；GPU 0–3 利用率 99–100%，显存占用约 34.6/34.6/19.8/19.8 GiB。所有任务持续推进，
尚无复验完成结果；未发现 readiness watcher 或全量预训练 launcher。

12:47 UTC coordinator PID 64527 仍存活（运行约 7h03m）；4 RUNNING、44 PENDING，全部
attempt 1。结构化训练日志最新 step 为 300M/9.85B 两臂 4138/4140、800M/1B 两臂
1339/1344；GPU 0–3 利用率 99–100%，显存占用约 34.6/34.6/19.8/19.8 GiB。所有任务持续推进，
尚无复验完成结果；未发现 readiness watcher 或全量预训练 launcher。

13:02 UTC coordinator PID 64527 仍存活（运行约 7h19m）；4 RUNNING、44 PENDING，全部
attempt 1。结构化训练日志最新 step 为 300M/9.85B 两臂 4291/4293、800M/1B 两臂
1389/1394；GPU 0–3 利用率 99–100%，显存占用约 34.6/34.6/19.8/19.8 GiB。所有任务持续推进，
尚无复验完成结果；未发现 readiness watcher 或全量预训练 launcher。

13:18 UTC coordinator PID 64527 仍存活（运行约 7h34m）；4 RUNNING、44 PENDING，全部
attempt 1。结构化训练日志最新 step 为 300M/9.85B 两臂 4442/4444、800M/1B 两臂
1438/1443；GPU 0–3 利用率均约 99%，显存占用约 34.6/34.6/19.8/19.8 GiB。所有任务持续推进，
尚无复验完成结果；未发现 readiness watcher 或全量预训练 launcher。

13:33 UTC coordinator PID 64527 仍存活（运行约 7h50m）；4 RUNNING、44 PENDING，全部
attempt 1。结构化训练日志最新 step 为 300M/9.85B 两臂 4591/4593、800M/1B 两臂
1487/1492；GPU 0–3 利用率 99–100%，显存占用约 34.6/34.6/19.8/19.8 GiB。没有空闲卡，
队列仍自动占满四卡；尚无复验完成结果，也未发现全量预训练 launcher。

13:48 UTC coordinator PID 64527 仍存活（运行约 8h05m）；4 RUNNING、44 PENDING，全部
attempt 1。结构化训练日志最新 step 为 300M/9.85B 两臂 4742/4745、800M/1B 两臂
1533/1539；GPU 0–3 利用率均约 99%，显存占用约 34.6/34.6/19.8/19.8 GiB。队列仍满载四卡，
尚无复验完成结果；未发现全量预训练 launcher。

14:04 UTC coordinator PID 64527 仍存活（运行约 8h20m）；4 RUNNING、44 PENDING，全部
attempt 1。结构化训练日志最新 step 为 300M/9.85B 两臂 4894/4896、800M/1B 两臂
1582/1588；GPU 0–3 利用率 99–100%，显存占用约 34.6/34.6/19.8/19.8 GiB。所有任务持续前进，
尚无复验完成结果；未发现全量预训练 launcher。

14:19 UTC coordinator PID 64527 仍存活（运行约 8h36m）；4 RUNNING、44 PENDING，全部
attempt 1。结构化训练日志最新 step 为 300M/9.85B 两臂 5042/5045、800M/1B 两臂
1631/1637；GPU 0–3 利用率 99–100%，显存占用约 34.6/34.6/19.8/19.8 GiB。所有任务持续前进，
尚无复验完成结果；未发现全量预训练 launcher。

14:35 UTC coordinator PID 64527 仍存活；初次进程快照短暂显示 D 状态，但 30 秒后复查为
`S`/`hrtimer_nanosleep`，queue `updated_at` 与文件 mtime 均推进至 14:35:48，属于已恢复的
瞬时 I/O 等待，未重启。该次快照队列为 4 RUNNING、44 PENDING；结构化日志 step 为
300M/9.85B 两臂 5194/5198、800M/1B 两臂 1681/1686；GPU 0–3 利用率 95–100%，显存约
34.6/34.6/19.8/19.8 GiB。未发现全量预训练 launcher。

14:50 UTC coordinator PID 64527 仍存活（运行约 9h07m）；4 RUNNING、44 PENDING，全部
attempt 1。结构化训练日志最新 step 为 300M/9.85B 两臂 5346/5349、800M/1B 两臂
1730/1736；GPU 0–3 利用率 99–100%，显存占用约 34.6/34.6/19.8/19.8 GiB。任务持续推进，
尚无复验完成结果；未发现全量预训练 launcher。

15:05 UTC coordinator PID 64527 仍存活（运行约 9h22m）；4 RUNNING、44 PENDING，全部
attempt 1。结构化训练日志最新 step 为 300M/9.85B 两臂 5497/5500、800M/1B 两臂
1778/1785；GPU 0–3 利用率 99–100%，显存占用约 34.6/34.6/19.8/19.8 GiB。队列与训练持续推进，
尚无复验完成结果；未发现全量预训练 launcher。

15:21 UTC coordinator PID 64527 仍存活（运行约 9h37m）；4 RUNNING、44 PENDING，全部
attempt 1。结构化训练日志最新 step 为 300M/9.85B 两臂 5645/5647、800M/1B 两臂
1827/1833；GPU 0–3 利用率 99–100%，显存占用约 34.6/34.6/19.8/19.8 GiB。任务持续推进，
尚无复验完成结果；未发现全量预训练 launcher。

15:36 UTC coordinator PID 64527 仍存活（运行约 9h53m）；4 RUNNING、44 PENDING，全部
attempt 1。结构化训练日志最新 step 为 300M/9.85B 两臂 5796/5799、800M/1B 两臂
1876/1882；GPU 0–3 利用率 99%，显存占用约 34.6/34.6/19.8/19.8 GiB。队列持续推进，
尚无复验完成结果；未发现全量预训练 launcher。

15:51 UTC coordinator PID 64527 仍存活（运行约 10h08m）；4 RUNNING、44 PENDING，全部
attempt 1。结构化训练日志最新 step 为 300M/9.85B 两臂 5948/5951、800M/1B 两臂
1925/1932；GPU 0–3 利用率 97–99%，显存占用约 34.6/34.6/19.8/19.8 GiB。所有任务持续前进，
尚无复验完成结果；未发现全量预训练 launcher。

16:07 UTC coordinator PID 64527 仍存活（运行约 10h24m）；4 RUNNING、44 PENDING，全部
attempt 1。结构化训练日志最新 step 为 300M/9.85B 两臂 6095/6099、800M/1B 两臂
1974/1981；GPU 0–3 利用率 96–100%，显存占用约 34.6/34.6/19.8/19.8 GiB。任务持续推进，
尚无复验完成结果；未发现全量预训练 launcher。

16:13 UTC 针对用户提出的 Chinchilla 式 isoFLOP 曲线预期，直接只读复核了 E1 screen 的 14 个
isoFLOP 原始结果：全部 `status=PASS`，global 与九域 held-out 指标均对应最终训练 step。
单 seed screen 的最低 loss 点依次为 c0(4.8e17 FLOPs) 的 50M 边界、c1(1.4e18) 的 100M、
c2(4.8e18) 的 200M；c1/c2 在已测参数范围内呈先降后升，最优点的 D/N 约为 34.6/26.0，
c0 在当前范围持续随 N 增大而变差、最低点落在 50M 左边界。这与定性 isoFLOP 谷底及计算量增加时
最优规模/数据分配移动的方向相符，但 c0 最优点未被双侧 bracket。上述只是单 seed screen 证据；
当前联合拟合仍因 normalized RMSE=0.03638 高于 0.03 门槛而 FAIL。需等待 3-seed 复验后检验
谷底位置和区间，不据此宣称完整 Chinchilla 复现或放行全量训练。

16:37 UTC 按用户要求将资源巡检间隔调整为 30 分钟后的首个快照：coordinator PID 64527 仍存活，
队列 4 RUNNING、44 PENDING，全部 attempt 1；300M/9.85B 两臂为 step 6396/6399，800M/1B
两臂为 2069/2076。GPU 0–3 利用率 99–100%，显存约 34.6/34.6/19.8/19.8 GiB。实验族正常
持续运行，无完成/失败状态变化；未发现全量预训练 launcher。

17:08 UTC 按 30 分钟间隔复核：coordinator PID 64527 仍存活（运行约 11h24m），队列
4 RUNNING、44 PENDING，全部 attempt 1；300M/9.85B 两臂到 step 6692/6696，800M/1B 两臂
到 2166/2173。四卡利用率 99–100%，显存约 34.6/34.6/19.8/19.8 GiB，队列正常推进，
无完成/失败状态变化；未发现全量预训练 launcher。

17:38 UTC 按 30 分钟间隔复核：coordinator PID 64527 仍存活（运行约 11h55m），队列
4 RUNNING、44 PENDING，全部 attempt 1；300M/9.85B 两臂到 step 6992/6995，800M/1B 两臂
到 2262/2270。GPU 0–3 利用率 98–99%，显存约 34.6/34.6/19.8/19.8 GiB。四个任务正常推进，
无完成/失败状态变化；未发现全量预训练 launcher。

18:09 UTC 半小时快照显示 coordinator PID 64527 仍存活（运行约 12h25m），队列 4 RUNNING、
44 PENDING，全部 attempt 1；300M/9.85B 两臂到 step 7288/7291，800M/1B 两臂到 2359/2367。
GPU 0–3 利用率 97–100%，显存约 34.6/34.6/19.8/19.8 GiB。首次 `ps` 显示 D 状态；30 秒后同一
PID 恢复为 `S`/`hrtimer_nanosleep`，队列 mtime 更新至 18:09:36，属于瞬时 I/O 等待，未重启。
任务持续推进，无完成/失败状态变化；未发现全量预训练 launcher。

18:39 UTC 半小时快照显示 coordinator PID 64527 仍存活（运行约 12h56m），队列 4 RUNNING、
44 PENDING，全部 attempt 1；300M/9.85B 两臂到 step 7584/7588，800M/1B 两臂到 2456/2464。
GPU 0–3 利用率均约 99%，显存约 34.6/34.6/19.8/19.8 GiB。任务持续前进，无完成/失败变化；
未发现全量预训练 launcher。

19:09 UTC 半小时快照显示 coordinator PID 64527 仍存活（运行约 13h26m），队列 4 RUNNING、
44 PENDING，全部 attempt 1；300M/9.85B 两臂到 step 7883/7887，800M/1B 两臂到 2550/2559。
GPU 0–3 利用率 98–99%，显存约 34.6/34.6/19.8/19.8 GiB。任务正常推进，无完成/失败变化；
未发现全量预训练 launcher。

19:40 UTC 半小时快照显示 coordinator PID 64527 仍存活（运行约 13h57m），队列 4 RUNNING、
44 PENDING，全部 attempt 1；300M/9.85B 两臂到 step 8179/8184，800M/1B 两臂到 2647/2656。
GPU 0–3 利用率均约 99%，显存约 34.6/34.6/19.8/19.8 GiB。队列持续推进，无完成/失败变化；
未发现全量预训练 launcher。

20:10 UTC 半小时快照显示 coordinator PID 64527 仍存活（运行约 14h27m），队列 4 RUNNING、
44 PENDING，全部 attempt 1；300M/9.85B 两臂到 step 8478/8483，800M/1B 两臂到 2744/2752。
GPU 0–3 利用率 99–100%，显存约 34.6/34.6/19.8/19.8 GiB。队列持续推进，无完成/失败变化；
未发现全量预训练 launcher。

20:41 UTC 半小时快照显示 coordinator PID 64527 仍存活（运行约 14h57m），队列 4 RUNNING、
44 PENDING，全部 attempt 1；300M/9.85B 两臂到 step 8774/8780，800M/1B 两臂到 2840/2850。
GPU 0–2 利用率 99%，GPU 3 首次采样为 0% 但仍占约 19.8 GiB；针对性复查约 30 秒后 GPU 3
恢复 99%，其训练日志 mtime/step 更新至 20:41:18/2850，训练进程正常存活，因此判定为瞬时采样/负载
间隙，不重启。未发现全量预训练 launcher。

21:11 UTC 半小时快照显示 coordinator PID 64527 仍存活（运行约 15h28m），队列 4 RUNNING、
44 PENDING，全部 attempt 1；300M/9.85B 两臂到 step 9070/9076，800M/1B 两臂到 2937/2946。
GPU 0–3 利用率均约 99%，显存约 34.6/34.6/19.8/19.8 GiB。所有任务持续推进，无完成/失败变化；
未发现全量预训练 launcher。

21:41 UTC 半小时快照显示 coordinator PID 64527 仍存活（运行约 15h58m），队列 4 RUNNING、
44 PENDING，全部 attempt 1；300M/9.85B 两臂到 step 9368/9376，800M/1B 两臂到 3031/3041。
GPU 0–3 利用率 99–100%，显存约 34.6/34.6/19.8/19.8 GiB。任务持续推进，无完成/失败变化；
未发现全量预训练 launcher。

22:12 UTC 半小时快照显示 coordinator PID 64527 仍存活（运行约 16h28m），队列 4 RUNNING、
44 PENDING，全部 attempt 1；300M/9.85B 两臂到 step 9664/9672，800M/1B 两臂到 3128/3138。
GPU 0–3 利用率 99–100%，显存约 34.6/34.6/19.8/19.8 GiB。任务持续推进，无完成/失败变化；
未发现全量预训练 launcher。

22:42 UTC 半小时快照显示 coordinator PID 64527 仍存活（运行约 16h59m），队列 4 RUNNING、
44 PENDING，全部 attempt 1；300M/9.85B 两臂到 step 9964/9972，800M/1B 两臂到 3225/3235。
GPU 0–3 利用率 99–100%，显存约 34.6/34.6/19.8/19.8 GiB。队列持续推进，无完成/失败变化；
未发现全量预训练 launcher。

23:12 UTC 半小时快照显示 coordinator PID 64527 仍存活（运行约 17h29m），队列 4 RUNNING、
44 PENDING，全部 attempt 1；300M/9.85B 两臂到 step 10260/10268，800M/1B 两臂到 3321/3332。
GPU 0–3 利用率 99–100%，显存约 34.6/34.6/19.8/19.8 GiB。所有任务持续推进，无完成/失败变化；
未发现全量预训练 launcher。

23:43 UTC 半小时快照显示 coordinator PID 64527 仍存活（运行约 18h00m），队列 4 RUNNING、
44 PENDING，全部 attempt 1；300M/9.85B 两臂到 step 10560/10568，800M/1B 两臂到 3419/3430。
GPU 0–3 利用率 99–100%，显存约 34.6/34.6/19.8/19.8 GiB。首次进程快照显示 D 状态；30 秒后同一
PID 为 `S`/`hrtimer_nanosleep`，queue mtime 更新到 23:44:20，属瞬时 I/O 等待，无需重启。未发现
全量预训练 launcher。

2026-11-08 00:14 UTC 半小时快照显示 coordinator PID 64527 仍存活（运行约 18h30m），队列
4 RUNNING、44 PENDING，全部 attempt 1；300M/9.85B 两臂到 step 10857/10866，800M/1B 两臂到
3513/3525。GPU 0–3 利用率均约 99%，显存约 34.6/34.6/19.8/19.8 GiB。四个任务持续推进，
无完成/失败变化；未发现全量预训练 launcher。

2026-11-08 00:44 UTC 半小时快照显示 coordinator PID 64527 仍存活（运行约 19h01m），队列
4 RUNNING、44 PENDING，全部 attempt 1；300M/9.85B 两臂到 step 11156/11166，800M/1B 两臂到
3611/3623。GPU 0–3 利用率 99–100%，显存约 34.6/34.6/19.8/19.8 GiB。队列持续推进，
无完成/失败变化；未发现全量预训练 launcher。

2026-11-08 01:15 UTC 半小时快照显示 coordinator PID 64527 仍存活（运行约 19h31m），队列
4 RUNNING、44 PENDING，全部 attempt 1；300M/9.85B 两臂到 step 11455/11464，800M/1B 两臂到
3707/3719。GPU 0–3 利用率 99–100%，显存约 34.6/34.6/19.8/19.8 GiB。所有任务持续推进，
无完成/失败变化；未发现全量预训练 launcher。

若 watcher 仍未恢复，需在 YuFormer 仓库根目录确认 seedrep queue PASS 且 48/48
结果为 PASS 后，依次运行以下 CPU 收尾流程；分析和身份均 PASS 后再渲染图表：

```bash
set -euo pipefail
python3 pretrain/scripts/audit_experiment_identities.py \
  --output report/experiment_identity_audit_latest.json
python3 pretrain/scripts/analyze_e1_seedrep.py \
  --root . \
  --output experiments/e1_scaling/results/seed_replication_fit.json
python3 pretrain/scripts/render_e1_scaling.py \
  --analysis experiments/e1_scaling/results/seed_replication_fit.json \
  --identity-audit report/experiment_identity_audit_latest.json \
  --output-dir experiments/e1_scaling/results/curves
```

此前 runbook 错误地指向单 seed screen analyzer；它不能作为最终复验拟合。上述命令已改为
seed-aware analyzer，非零退出会由 `set -e` 阻止后续绘图。拟合报告至少应确认
`status=PASS`、拟合门禁 PASS、72/72 身份、24×3 seed 设计、
目标/实际 token 预算审计 PASS、`seed_bootstrap_rng_seed=20261107` 及跨点共享索引的
配对 bootstrap 方法，并包含 c0/c1/c2 三档 `compute_optimal_frontier` 及其区间状态；
曲线 manifest 应为 `ANALYSIS_AND_IDENTITY_PASS`，且包含 `compute_optimal_frontier.png`
和 `.svg`。该收尾流程不启动 1T 训练。

2026-11-08 01:47 UTC 针对 GPU 3 的短复查确认：前一任务 `e1_seedrep_e1_ln_800m_s3` 在 step 3814
正常结束并通过，队列随即在 GPU 3 启动下一项 `e1_seedrep_e1_iso_c2_800m_s2`
（attempt 1），不是任务卡死或异常空转。队列更新到 01:47:18 UTC，状态为 4 RUNNING、43 PENDING、
1 PASS；四张卡的最新利用率采样均为 98–100%。继续按约 30 分钟间隔做常规快照，不启动全量预训练。

2026-11-08 01:51 UTC：复验队列完成一项额外任务并自动补位。权威队列状态为 4 RUNNING、42 PENDING、
2 PASS；PASS 项是 `e1_seedrep_e1_ln_800m_s2` 与 `_s3`，其余活动任务均为 attempt 1。
静态核对整份 48-job 队列：恰有 24 个设计点，每个均含 seed 2/3；所有 env 和 YAML 配置存在，
输出目录均精确映射到各自 job ID，没有非法命名、缺失配置或重试项。GPU 0–3 利用率 99–100%，
队列 coordinator PID 64527 存活；资源策略仍为 `full_pretrain_launch_allowed=false`。
此队列与原始 24 个 seed-1 screening 点合并后才构成 24×3 最终分析输入；当前不可提前拟合最终曲线。

补充核验 01:55 UTC 已 PASS 的两项 800M/1B seed 复验结果：两者 result、current_attempt 与 training_complete
的 attempt ID 一致；全局及九域共 10 项 held-out 指标均来自最终 step 3814 且为正有限值；实际训练
999,817,216 tokens，低于配置预算 1,000,000,000 共 182,784 tokens（小于单步 token 数），记账一致。

2026-11-08 02:22 UTC 半小时快照：队列更新时间 02:22:16，状态 4 RUNNING、42 PENDING、2 PASS；
四个活动项 attempt 均为 1，30 分钟内日志持续更新。GPU0/1 的 300M/9.85B 两个 seed 分别推进到
step 12111/12122；GPU2/3 的 isoFLOP c2 800M 两个 seed 推进到 step 101/110。四卡利用率
98–100%，显存约 34.6/34.6/19.2/19.2 GiB。协调器 PID 64527 存活，未见重试；资源策略仍
`full_pretrain_launch_allowed=false`。队列正常占满授权四卡；seed-aware 全量拟合仍等待 48 项复验结果齐备。

2026-11-08 02:52 UTC 半小时快照：队列更新时间 02:52:17，仍为 4 RUNNING、42 PENDING、2 PASS，
无重试；活动 attempt 均为 1。300M/9.85B 两个 seed 分别推进到 step 12406/12417，isoFLOP c2
800M 两个 seed 到 step 196/202，所有日志更新时间至 02:52:20。GPU0–3 利用率 99–100%，显存
34.6/34.6/19.2/19.2 GiB；coordinator PID 64527 存活。`full_pretrain_launch_allowed=false` 仍有效，
全量预训练未启动；48 项复验剩余 42 pending，最终拟合门禁未到可执行阶段。

2026-11-08 03:22 UTC 半小时快照：队列更新时间 03:22:15，4 RUNNING、42 PENDING、2 PASS，当前活动项
均为 attempt 1、无重试。两个 300M/9.85B seed 到 step 12698/12709；isoFLOP c2 800M 两个 seed
到 step 291/295；日志更新时间均为 03:22:16。GPU0–3 利用率 98–99%，显存约
34.6/34.6/19.2/19.2 GiB。队列 coordinator PID 64527 存活，授权四卡均在使用；资源策略继续
设置 `full_pretrain_launch_allowed=false`。复验队列正常推进，最终 24×3 拟合仍等待其余 42 项。

2026-11-08 03:52 UTC 半小时快照：队列更新时间 03:52:19，状态仍为 4 RUNNING、42 PENDING、2 PASS；
全部活动任务 attempt 1 且无重试。300M/9.85B 两 seed 到 step 12994/13003，isoFLOP c2 800M 两
seed 到 step 387/388，四份训练日志 mtime 均更新至 03:52:24。GPU0–3 利用率 99–100%，显存约
34.6/34.6/19.2/19.2 GiB；coordinator PID 64527 存活。资源策略仍禁止全量预训练，队列继续占满
授权四卡并正常推进；最终 24×3 研究结论需等待剩余 42 项复验任务完成。

2026-11-08 04:22 UTC 半小时快照：队列更新时间 04:22:46，仍为 4 RUNNING、42 PENDING、2 PASS；
无重试，活动任务 attempt 均为 1。300M/9.85B 两 seed 到 step 13291/13303，isoFLOP c2 800M
两 seed 到 step 482/484，日志 mtime 均为 04:22:49。GPU0–3 利用率均 99%，显存约
34.6/34.6/19.2/19.2 GiB。coordinator PID 64527 存活；四卡持续被 scaling 队列占满，
`full_pretrain_launch_allowed=false` 仍有效。

2026-11-08 04:52 UTC 半小时快照：队列更新时间 04:52:17，4 RUNNING、42 PENDING、2 PASS；无重试，
所有活动任务为 attempt 1。300M/9.85B 两 seed 到 step 13579/13590，isoFLOP c2 800M 两 seed 到
step 571/575，日志均更新至 04:52:22。GPU0–3 利用率 98–100%，显存约 34.6/34.6/19.2/19.2 GiB；
coordinator PID 64527 存活（采样时为 R 状态）。授权四卡持续满载，`full_pretrain_launch_allowed=false`。

2026-11-08 05:22 UTC 半小时快照：队列更新时间 05:22:21，4 RUNNING、42 PENDING、2 PASS；没有重试，
活动任务均为 attempt 1。300M/9.85B 两 seed 到 step 13874/13885，isoFLOP c2 800M 两 seed 到
step 664/671，日志更新时间 05:22:22。GPU0–3 利用率 93–100%，显存约 34.6/34.6/19.2/19.2 GiB；
coordinator PID 64527 存活。资源策略仍关闭全量预训练，四卡用于 scaling 队列。

2026-11-08 05:52 UTC 半小时快照：队列更新时间 05:52:34，4 RUNNING、42 PENDING、2 PASS；活动项无重试，
均为 attempt 1。300M/9.85B 两 seed 到 step 14168/14178，isoFLOP c2 800M 两 seed 到 step
758/767；四份日志均更新至 05:52:38。GPU0–3 利用率 99–100%，显存约 34.6/34.6/19.2/19.2 GiB；
coordinator PID 64527 存活，运行约 1 天 8 分钟。全量预训练开关仍为 false，授权四卡继续全部用于
scaling 实验。

2026-11-08 06:22 UTC 半小时快照：队列更新时间 06:22:45，4 RUNNING、42 PENDING、2 PASS；无重试，
活动任务均为 attempt 1。300M/9.85B 两 seed 到 step 14465/14475，isoFLOP c2 800M 两 seed 到
step 851/862，日志 mtime 均为 06:22:51。GPU0–3 利用率均 99%，显存约 34.6/34.6/19.8/19.8 GiB；
coordinator PID 64527 存活（采样时 R 状态）。全量预训练策略开关 false，四卡仍全部用于 scaling。

2026-11-08 06:52 UTC 半小时快照：队列更新时间 06:52:58，4 RUNNING、42 PENDING、2 PASS；无重试，
所有活动任务 attempt 1。300M/9.85B 两 seed 到 step 14759/14769，isoFLOP c2 800M 两 seed 到
step 944/958，日志 mtime 均为 06:52:59。GPU0–3 利用率 99–100%，显存约 34.6/34.6/19.8/19.8 GiB；
coordinator PID 64527 存活，运行约 1 天 1 小时 9 分。`full_pretrain_launch_allowed=false` 仍生效。

2026-11-08 07:23 UTC 半小时快照：队列更新时间 07:22:58，4 RUNNING、42 PENDING、2 PASS，活动项仍为
attempt 1 且无重试。300M/9.85B 两 seed 到 step 15051/15062，isoFLOP c2 800M 两 seed 到
step 1035/1051，日志均更新至 07:23:00。GPU0–3 利用率 97–100%，显存约 34.6/34.6/19.8/19.8 GiB；
coordinator PID 64527 存活。资源策略继续禁止全量预训练，四卡仍由 scaling 队列使用。

2026-11-08 07:32 UTC 用户将人工监控间隔改为每 2 小时；下一次全面检查约 09:32 UTC。另按用户要求更新
seedrep ETA：此刻队列为 4 RUNNING、42 PENDING、2 PASS，即 46/48 个 seedrep 作业尚未完成。估算对每个
source point 读取已完成 seed-1 screen 的 `trained_tokens`、`trained_steps`、`throughput(tps).last/first`，
结合复验队列当前 RUNNING 的最新 step 估剩余比例，并按 4 个 GPU worker、队列顺序做 list-scheduling；
不假设更高并行度。以各点 screen `last tps` 作为吞吐代理，剩余关键路径约 90.5 小时（3.77 天）；
加入约 10% 吞吐/作业启动验证余量后约 100 小时（4.17 天）；较保守地改用 first/last 吞吐几何均值并
每作业加 5 分钟开销约 123 小时（5.12 天），压力情形约 135 小时（5.63 天）。因此实用 ETA 约为还需
4–5.5 天，从 07:32 UTC 算约 2026-11-12 中午至 2026-11-13 深夜 UTC；中心估计约 4.5–5 天。
此为调度近似，不含长时间节点中断或额外重试；实测复验吞吐若持续偏离 seed-1 screen 代理，应在两小时
状态检查时重估。最终 24×3 分析仍须等 48 项复验全部 PASS，随后身份/预算审计和 seed-aware 拟合通过。

## 2026-11-12 E1 seed replication 收尾

2026-11-12 02:43 UTC，`e1_seedrep` 队列终态 `PASS`：48/48 复验通过、0 失败；GPU 0–3 已空闲。
队列 supervisor 曾在用户暂停目标时退出，但四个训练子进程仍存活。恢复时使用原生队列恢复逻辑接管原
attempt（未增加 attempt、未重复训练），随后队列正常完成。

收尾审计结果：`audit_experiment_identities.py` 为 113/113 PASS；`analyze_e1_seedrep.py` 的 seed 设计审计
和 token 预算审计均 PASS。24 点 × 3 seeds 的复验实际新增 `84,893,761,536` tokens，合同预算为
`84,899,181,190`，差额 `5,419,654` tokens，属完整 optimizer update 取整范围。

三 seed 联合拟合总体 `FAIL`：NRMSE `3.6635%` 高于固定 `3%` 门槛；R² `0.98017`、LOO 中位误差
`3.13%`、LOO p95 `7.86%`、吞吐拟合检查通过。拟合门禁未通过时 renderer 输出 `EXPLORATORY_PROGRESS`，
seed-bootstrap NRMSE 95% 区间为 `3.554%–4.120%`，区间下界也高于门槛。对同一预设 loss law 做 200 组
随机初值的 CPU 优化，最优 RMSE 与正式拟合一致（NRMSE `3.6635%`），未发现局部最优造成门槛失败。
renderer 隐藏 compute-optimal frontier。三档 IsoFLOP 观测最低点依次为 c0：25.4M 参数（搜索下边界），c1：
59.4M，c2：142.8M，均值 loss 为 `3.7743 / 3.4984 / 3.2391`；方向符合随算力增大而使用更大模型，
但 c0 最优点未 bracket，且全局拟合残差不达门槛。

产物：`experiments/e1_scaling/results/seed_replication_fit.json`；探索曲线及端点：
`experiments/e1_scaling/results/curves_seedrep/`；身份审计：`report/experiment_identity_audit_latest.json`。
资源策略 `full_pretrain_launch_allowed=false` 保持不变；不得以探索曲线作为 1T 选型依据。下一步先诊断
联合 loss-law 残差；若需新增训练，必须先修订并审计研究预算合同，再启动新实验。

渲染器收尾修复：之前全局拟合质量门禁 FAIL 时，把已存在的 24 个端点残差错误地显示成“fit pending”。
现改为在 `EXPLORATORY_PROGRESS` 图中显示诊断残差、显式标记 `fit gate FAIL`，同时继续隐藏 frontier；
renderer 重新运行结果为 24/24 audited endpoints，manifest 的 frontier 状态为
`WITHHELD_FIT_GATE_FAIL`。这只改进失败诊断展示，不更改数据、拟合、阈值或放行决策。

门禁产物修复：seed-aware analyzer 原先在 fit gate FAIL 时仍填入 `production_recommendation`，且把可计算的
compute-optimal 点标为 `PASS`。现显式改为 `WITHHELD_FIT_GATE_FAIL`，候选只保留诊断数值并带 withheld 状态；
重跑 1,000 次配对 seed bootstrap 后，参数、loss、NRMSE 与区间不变，整体仍 FAIL。输出不再暴露可被误读的
生产推荐。
