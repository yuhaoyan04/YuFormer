# 异常案例库

`data/` 与 `report/history/` 路径是外部运行证据；公开源码、轻量指标和对应SHA见 `ops/` 与 `report/metrics/`。保留证据引用不表示原始权重或日志已上传。

已索引 44 条历史记录与 18 条近期案例。原始 E 编号存在重复，H 编号按原文顺序消歧；历史状态和适用版本不改写。机器数据见 [incidents.json](incidents.json)。

| ID | 阶段 | 案例 | 状态 |
|---|---|---|---|
| P001 | data | 校准样本与筛选哈希相关 | FIXED_AND_REVALIDATED |
| P002 | dataloader | 全量配比正确但时间切片漂移 | FIXED_AND_REVALIDATED |
| P003 | qualification | 旧前缀不能证明生产数据质量 | FIXED_AND_REQUALIFIED |
| P004 | attention | Flex自动decode tile不兼容GQA | FIXED_IN_QUALIFICATION |
| P005 | long_context | 64K重计算与保存策略失效 | SUPPORTED_PATH_QUALIFIED |
| P006 | checkpoint | CPU保存线程在GPU0建多余context | FIXED_FOR_LONG_STAGES |
| P007 | long_data | 短replay的books供给不足 | FIXED_RELEASE_V2 |
| P008 | lr | 下一步LR与调度器虚拟终点差一 | FIXED_OBSERVER_CONTRACT |
| P009 | monitoring | TensorBoard事件快照名称未被识别 | FIXED_BEFORE_FORMAL_ADMISSION |
| P010 | monitoring | step1验证和scalar reservoir边界 | FIXED_OBSERVERS |
| P011 | evaluation | RULER资产未展开LFS或上下文不足 | FIXED_CASE_COHORT |
| P012 | evaluation | RepoQA gold不能被原评分器正确识别 | FIXED_CASE_COHORT |
| P013 | evaluation | CRLF被审计器误归一化 | FIXED_AUDITOR_FALSE_ALARM |
| P014 | storage | 共享存储rename或hardlink被拒绝 | FIXED_FALLBACK |
| P015 | operations | 外围SIGKILL而八卡仍在训练 | RECOVERED_SAME_LIVE_ATTEMPT |
| P016 | validation | books短窗小幅回升 | OPEN_OBSERVATION |
| P017 | repository | 旧归档进入待提交集合和远端写权限不足 | RESOLVED_CLEANUP_AND_REMOTE_SNAPSHOT_VERIFIED |
| P018 | validation | 多域短窗回升与比较基准不一致 | OPEN_OBSERVATION |

## P001 校准样本与筛选哈希相关

**症状：** warmup全量web中fineweb-edu约19.59%，校准前缀33.60%；全阶段配比掩盖偏差。

**根因/未知项：** 筛选、tie-break、截断和校准复用doc_id顺序，样本非独立。

**处置：** 用独立文档rank覆盖完整已选集合，核对实际域计数并绑定同一打开FD。

**验证与范围：** 独立校准及完整实际消费配比验收通过；pilot区间不替代全流审计。

**可复用规则：** 校准抽样与训练排序使用独立随机化，估算和实际消费分别对账。

证据：`report/E133_E134_DATA_ORDER_REMEDIATION_20261114.md`。

## P002 全量配比正确但时间切片漂移

**症状：** warmup连续十段web由72.70%降至49.90%，目标60%。

**根因/未知项：** 质量选择与训练顺序相关，总计数无法发现时间漂移。

**处置：** 保留token/ledger和因果目标集合，独立B8块置换；绑定完整双射及manifest SHA。

**验证与范围：** 首末样本、阶段转换、拓扑恢复、换索引拒绝和实际时间切片审计通过。

**可复用规则：** 配比要同时检查全程守恒、实际前缀与时间切片；恢复必须绑定排序身份。

证据：`report/E133_E134_DATA_ORDER_REMEDIATION_20261114.md`, `pretrain/backends/torchtitan/yuformer_ext/block_permutation.py`。

## P003 旧前缀不能证明生产数据质量

**症状：** 旧研究前缀约92.53% web、1.58%中文，不能代表生产warmup。

**根因/未知项：** 研究配置、样本前缀与当前正式配方身份不相同。

**处置：** 从真实生产配置派生数值/恢复/持续控制，绑定最终模型、seed、batch、数据和源码。

**验证与范围：** 最终同样本0.530B/1.5166B控制及扩大验证通过；旧实验保持原范围。

**可复用规则：** 资格必须覆盖实际生产执行闭包，不能按名称复用旧PASS。

证据：`experiments/e2_architecture/results/model-scale-scientific-assessment.json`, `pretrain/scripts/qualification_identity.py`。

## P004 Flex自动decode tile不兼容GQA

**症状：** 短GQA路径AUTO decode在tile128与BLOCK_M512之间失败。

**根因/未知项：** 自动kernel选择与实际GQA布局不匹配。

**处置：** 显式采用经过验证的TRITON路径，对实际head/layout执行输出与梯度对拍。

**验证与范围：** 八卡operator及集成模型资格通过；这不替代最终模型能力。

**可复用规则：** attention优化同时验证语义、梯度、短长序列和真实布局。

证据：`experiments/e2_architecture/results/flex_swa_20261114`, `pretrain/backends/torchtitan/flex_module_parity.py`。

## P005 64K重计算与保存策略失效

**症状：** 整层重计算出现非有限梯度，另一MLP配置在最终保存出现NCCL/CUDA错误。

**根因/未知项：** 64K显存、重计算与异步保存组合超出已验证执行路径。

**处置：** 采用实际通过的MLP重计算、32路loss chunk、GC策略及rank-bound stager。

**验证与范围：** 真实16K/64K训练、验证、最终保存和两阶段全状态SIGKILL恢复通过。

**可复用规则：** 必须测到最终保存与恢复，4K smoke不能外推64K。

证据：`midtrain/README.md`, `data/health/midtrain-actual-eight-gpu-stage-qualification.json`。

## P006 CPU保存线程在GPU0建多余context

**症状：** NVML观察15个context，GPU0多占约3.55GiB。

**根因/未知项：** 异步DCP CPU进程反序列化CUDA状态，默认设备未按rank绑定。

**处置：** 长阶段使用显式rank-bound eager stager；不热改正在跑的4K源码。

**验证与范围：** 真实HOST PID/UUID核对15→8，长阶段保存和恢复通过。4K额外context作为已知且容得下的限制保留。

**可复用规则：** CPU路径也检查CUDA反序列化和rank设备映射；用NVML核对实际进程。

证据：`midtrain/docs/SOURCE_PROVENANCE.md`, `midtrain/src/yuformer_midtrain/checkpoint.py`。

## P007 短replay的books供给不足

**症状：** V1完整窗口准入发现短books可用文档不足。

**根因/未知项：** 将短replay源限制在不够的anneal集合。

**处置：** V2采用有界稳定短文档和已记录的配比调整，不动主4K数据。

**验证与范围：** 两长阶段完整窗口release独立审计PASS；V1失败保留。

**可复用规则：** 先以唯一文档/窗口容量证明供给，再发布阶段配方。

证据：`data/health/long-production-window-release-v2.json`。

## P008 下一步LR与调度器虚拟终点差一

**症状：** 按表面cosine公式推导的保存LR可能与上游实际next-update不符。

**根因/未知项：** 已完成step、下一次更新、total+1虚拟终点与日志时机混用。

**处置：** 用独立公式与锁定TorchTitan实际调度器边界、真实标量及原生LR对拍。

**验证与范围：** warmup/stable/decay边界与四个实际长阶段资格轨迹通过；正式LR持续校验。

**可复用规则：** 恢复验收分别对账scheduler step和保存的next optimizer LR。

证据：`pretrain/backends/torchtitan/lr_schedule_contract.py`, `report/PRETRAIN_HALF_DAY_REVIEW_20261117T0046.md`。

## P009 TensorBoard事件快照名称未被识别

**症状：** 已复制的events-0前缀不能被EventAccumulator识别。

**根因/未知项：** 读取器依赖events.out.tfevents命名约定。

**处置：** 用标准命名保存不可变前缀并独立重读。

**验证与范围：** 第2000步原始及复制事件十个tag逐值匹配；旧失败证据归档。

**可复用规则：** 文件格式契约也包含发现/命名规则，复制成功不等于可读取。

证据：`report/FIRST_FORMAL_VALIDATION_2000_20261115.md`。

## P010 step1验证和scalar reservoir边界

**症状：** 验证观察器误拒绝实际step1；有限reservoir可能使最近100点P10口径失真。

**根因/未知项：** 未完整对应真实验证日程，默认scalar缓存上限不能代表全部记录。

**处置：** 保留step1/final边界并完整加载scalars；以真实轨迹验证观察器。

**验证与范围：** 完整日程检查和实际多运行事件回放通过；reservoir问题在正式超过上限前修复。

**可复用规则：** 监控也要用真实事件和终点/默认缓存反例验证，不能只测训练器。

证据：`report/history/operations-consolidation-20261123/ACTIVE_PRETRAIN_OPERATIONS.md`。

## P011 RULER资产未展开LFS或上下文不足

**症状：** LFS pointer被当词库，原始案例存在低于75%context长度的输入。

**根因/未知项：** 下载成功不等于真实资产，tokenizer改变目标长度。

**处置：** 核对真实LFS对象、原始文件和tokenizer，固定候选顺序取满足长度的案例。

**验证与范围：** 13任务×32案例×3长度的1248条诊断案例审计通过，非官方完整benchmark。

**可复用规则：** 评测资产验证内容和几何尺寸，不能把smoke子集包装成公开成绩。

证据：`eval/docs/CONTEXT_STAGE_PIPELINE.md`, `eval/results/frozen_context_stage_cases_20261115.json`。

## P012 RepoQA gold不能被原评分器正确识别

**症状：** C++ nested-function等gold自测可仅约0.5757 BLEU，未过0.8标准。

**根因/未知项：** 提取器和真实函数边界/注释格式相互影响。

**处置：** 保留原评分器阈值，对gold做自一致性eligibility检查后冻结案例。

**验证与范围：** 48个gold实际自测和288个带/不带context案例审计通过；模型能力尚待测。

**可复用规则：** 先证明评分器能识别gold，再解释模型分数；不靠降低阈值过门禁。

证据：`eval/docs/CONTEXT_STAGE_PIPELINE.md`, `eval/src/yuformer_eval/stage_scores.py`。

## P013 CRLF被审计器误归一化

**症状：** 长文原始正文核对出现字节不一致告警。

**根因/未知项：** 审计脚本read_text转换CRLF，而冻结文本保留原字节。

**处置：** 审计使用read_bytes().decode，不修改冻结案例。

**验证与范围：** 独立raw重编码、UTF8字节分母与目标排除EOD检查通过。

**可复用规则：** 查差异先区分生产资产变化和审计器自身转换。

证据：`eval/docs/CONTEXT_STAGE_PIPELINE.md`, `report/history/operations-consolidation-20261123/ACTIVE_PRETRAIN_OPERATIONS.md`。

## P014 共享存储rename或hardlink被拒绝

**症状：** 目录rename或跨目录硬链接可能EPERM，首检查点归档不能假定POSIX支持。

**根因/未知项：** 实际共享存储权限/操作语义与本地盘不同。

**处置：** 按实际错误使用受支持的逐文件复制、fsync与SHA核对。

**验证与范围：** 首250步9文件独立复制及完整字节校验通过；原失败路径保留。

**可复用规则：** 存储原语先在目标文件系统验证，完成标记必须最后持久化。

证据：`docs/research/工程经验记录.md`, `data/health/first-formal-checkpoint-250-admission-20261115.json`。

## P015 外围SIGKILL而八卡仍在训练

**症状：** 2026-11-18主协调、租约owner和辅助守护成为zombie，旧JSON仍为RUNNING。

**根因/未知项：** SIGKILL已确认，发送来源未知；cgroup oom_kill=0，不能归因为OOM。

**处置：** 确认原bash/torchrun/八worker身份，归档证据，显式接管原generation/token并恢复守护。

**验证与范围：** 八卡未重启，92份历史控制状态、实际LR及验证重读通过，后续半日复核持续运行。

**可复用规则：** 检查实际进程和启动时间；活训练不能被新租约generation误fence。

证据：`report/PRETRAIN_OUTER_SUPERVISOR_RECOVERY_20261118.md`, `data/health/live-pretrain-supervisor-recovery-admission-20261118.json`。

## P016 books短窗小幅回升

**症状：** 第56000→60000步books 2.617864→2.621027，+0.120791%；其他八域及总体改善。

**根因/未知项：** 尚未确定原因；固定64样本点估计不能证明是噪声。

**处置：** 保留关注状态，核对配方、source、游标、LR与恢复计数，继续冻结训练并观察。

**验证与范围：** 243份检查点控制记录、6080个LR标量通过，stable游标正确；不声明该质量项已恢复。

**可复用规则：** 小幅单域取舍要量化并留痕，不通过改数据或抹告警制造单调曲线。

后续测量（2026-11-23T12:51:56.040119+00:00）：62000/64000 books loss为2.613039732/2.610648870，均低于56000的2.617864370；原回升事实保留，根因未定，不称为代码修复。

证据：`data/health/books-domain-regression-followup-20261123T0046.json`, `report/PRETRAIN_HALF_DAY_REVIEW_20261123T0046.md`。

## P017 旧归档进入待提交集合和远端写权限不足

**症状：** 初次整理时约13.65GB旧检查点/归档进入待提交集合，远端仅早期代码；当时SSH和聊天连接器账号均无写权限。

**根因/未知项：** 重型运行档案未完全排除，远端账号与仓库写权限不匹配。

**处置：** 排除重型运行资产并核验391份冻结依赖未变，保留本地提交与bundle。用户完成GitHub官方设备授权，以隔离CLI登录yuhaoyan04；仅配置本仓库HTTPS pushurl及credential helper，非强制推送后核对远端main。

**验证与范围：** 清理961文件/13,661,559,135字节；实际账号yuhaoyan04、push=true；2026-11-23T04:46:22.980598+00:00远端main核验等于b41084f4d14565b1449f59f482d63c507ea12b8b。聊天连接器与原SSH身份未改；后续提交需单独核验。

**可复用规则：** 分离重型资产与轻量结果；提交身份、SSH身份、连接器身份和CLI身份分别核验，仓库级隔离凭据避免改全局环境；以实际远端SHA确认发布。

证据：`docs/engineering/REPOSITORY_MAINTENANCE.md`, `report/REPOSITORY_MAINTENANCE_20261123.md`, `report/metrics/repository_publication_20261123.json`。

## P018 多域短窗回升与比较基准不一致

**症状：** 64000相对60000的code/math/multilingual分别回升0.179529%/0.112315%/0.046690%；聚合相对60000下降0.106300%，却相对62000上升0.123285%。

**根因/未知项：** 质量变化根因未定。不同窗口参照导致方向不同；已检查source pins、LR、游标及断点控制状态，未找到对应配置错误，不能凭此排除真实质量变化或宣称只是噪声。

**处置：** 保留原attention状态，同时报告前一次点和三点窗口参照；展示所有九域近期曲线，继续冻结训练并在下一12小时复核重新评估。

**验证与范围：** 259份原生控制记录、6482个实际LR标量和最新三次固定heldout事件前缀通过；九域64000均低于54000，books已低于56000，但短窗域回升仍为开放观察。固定64样本无此处置信区间，最终能力未测。

**可复用规则：** 趋势报告必须写明参照step；总均值改善不能消除单域退步，后续点变好不等于根因已修复。

证据：`report/PRETRAIN_HALF_DAY_REVIEW_20261123T1246.md`, `report/PRETRAIN_DOMAIN_VALIDATION_FOLLOWUP_20261123T1246.md`, `report/metrics/domain_validation_followup_20261123T1246.json`。

## 历史案例索引

完整症状、根因和处置保留在 [原始工程经验记录](../research/工程经验记录.md) 与 JSON 的 `record` 字段；按原文行号查阅。

| 唯一 ID | 原始标题 | 原文行 |
|---|---|---:|
| H001 | E-1 多进程 Pool 的静默挂死 | 8 |
| H002 | E-2 停滞守卫的生命周期 bug（修复者引入的新 bug） | 14 |
| H003 | E-3 流式写出器的隐形内存积累 | 20 |
| H004 | E-4 数据静默丢失的字段回退语义 | 24 |
| H005 | E-5 内存护栏三级撞顶：阈值问题 vs 病理实现 | 30 |
| H006 | E-6 防护阈值设定规则（由 E-5 总结） | 37 |
| H007 | E-11 全局物化操作的隐藏内存翻倍（2026-09-20 OOM killer 实录） | 41 |
| H008 | E-7 断点幂等是所有激进修复的前提 | 49 |
| H009 | E-8 防御性守卫反向抓实现 bug | 53 |
| H010 | E-9 共享存储是全文扫描的隐形瓶颈 | 57 |
| H011 | E-10 工具命令的自我保护 | 61 |
| H012 | E-16 静默失败比崩溃更危险：S1 的 76% 分片失败潜伏两天（2026-09-22，本项目最严重流程事故） | 65 |
| H013 | E-17 知道什么时候该止损：web 近重聚类的 40 天陷阱（2026-09-22） | 71 |
| H014 | E-17b/E-18 止损被逆转之后：把 40 天的坑改成一天内的活（2026-09-26，ndv2） | 76 |
| H015 | E-20 行组粒度放大：布局决定点查成本（2026-09-26，ndv2 实测） | 82 |
| H016 | E-19 fasttext 训练 NaN 三连击的根因链（2026-09-26 全解密） | 88 |
| H017 | E-20 打分器的生产/研究分工裁定（2026-09-26，用户标准驱动） | 92 |
| H018 | E-21 goldset 采样单线程瓶颈（2026-09-26） | 97 |
| H019 | E-21 RSS 高水位 ≠ 活动集：分配器驻留在 cgroup 口径下是计费内存（2026-09-26，ndv2 实测） | 101 |
| H020 | E-22 词缓存 clear 时序 bug：decontam 4,689 分片确定性失败（本日最大事故） | 109 |
| H021 | E-23 S5 长文域批处理病态：行数驱动 batch 的 33 倍内存爆炸（压测抓出） | 115 |
| H022 | E-24 顺序敏感的派生物：S5/S6 对 ndv2 的隐性过期（护栏补齐） | 121 |
| H023 | E-25 打分器质量门的三个连环坑（2026-09-27，S4 收尾复盘） | 127 |
| H024 | E-26 问答域的隐形 HTML 残留（2026-09-27 质量抽检） | 133 |
| H025 | E-27 分位表与打分补齐的竞态（2026-09-30 审计发现，未爆发即修复） | 138 |
| H026 | E-28 tokenize 物化的随机序 IO 灾难（2026-09-30 审计发现，未爆发即修复） | 143 |
| H027 | E-23 GPU 显存僵尸泄漏锁死训练窗口（2026-10-15） | 148 |
| H028 | E-24 ndv2 阶段性止损（2026-10-15；已于 2026-10-18 撤销） | 153 |
| H029 | E-30 ndv2 的可运行终局：压缩验证、精度门与直接边决策（2026-10-18） | 158 |
| H030 | E-23b 假死诊断与险些双开（2026-10-15 续） | 164 |
| H031 | E-25 SFS 目录 rename 禁令：8 连崩的终极根因（2026-10-16，本项目最昂贵的一行代码） | 168 |
| H032 | E-75 E0-Q2 点火实录：调度器同卡堆叠竞态 + E-23 残配置 9 天工期（2026-10-16 接手首跑） | 174 |
| H033 | E-76 unigram 臂数据性 NaN：前向有限、反向全员 NaN 的确定性样本中毒（2026-10-16，E0-Q2 首跑实抓） | 185 |
| H034 | E-77 val 评估的显存永久驻留：每轮 +14~28GB、四轮 OOM（2026-10-16 16:12 实抓） | 192 |
| H035 | R-1 web 域近重候选密度超预期 | 204 |
| H036 | R-2 fasttext 小数据退化 | 208 |
| H037 | E-12 纯 Python 归并 × 全局物化 = 慢死 + OOM（2026-09-21 实录，S3 内存病理 round 3） | 211 |
| H038 | E-13 聚合内存 = 任务上限 × 并发数（2026-09-22 实录） | 216 |
| H039 | E-14 page cache 是 cgroup OOM 的隐形杀手（2026-09-22 实录，S3 第四次） | 220 |
| H040 | E-15 S3 内存病理全景（存档，教训密度最高的一个阶段） /  爆点  /  根因模式  /  修复模式  /  | 225 |
| H041 | E-29 双 IO 大户互抢是双输：S4 打分 × ndv2 verify 的调度账（2026-10-01） | 234 |
| H042 | E-30 账本作者踩了账本里的规则：b_cache 扩容忘了乘并发数（2026-10-01，未爆发即回滚） | 239 |
| H043 | E-31 code verify 的二次止损：预算估算必须用"最慢桶 × 总对数"而非平均（2026-10-01） | 243 |
| H044 | E-33 监控者的第一课：处置动作必须有双判据 + 最小间隔（2026-10-04，watchdog 首跑误杀 S4） | 248 |
