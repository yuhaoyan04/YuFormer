# YuFormer 数据预处理失败账本（FAILURE LEDGER）

> 用途：避坑指南。汇总数据管线（S0→S6 + tokenize）全部真实事故/缺陷——失败几次、
> 每次根因、如何修复、怎样预防。逐条明细见 [`docs/research/工程经验记录.md`](../../docs/research/工程经验记录.md)（E-1～E-28 共 30 条编号 + 2 条研究发现）。
> 维护约定：新事故必须先入工程经验记录，再同步更新本账本的统计与清单。

## 一、总账（截至 2026-09-30）

| 口径 | 数量 | 说明 |
|---|---|---|
| 编号事故 | **32** | E-1~E-32（E-17b/E-18、E-20、E-21 各有两条重号记录） |
| **爆发型**（造成实际损失） | **15** | 数据丢失 2 次（E-4、E-16 连带）、OOM 家族 7 次（E-5/11/12/13/14/15/32）、挂死 1 次（E-1）、批量失败 2 次（E-16、E-22）、训练异常 1 次（E-19）、性能陷阱 2 次（E-9/12） |
| **拦截型**（审计/测试/压测在爆发前修复） | **9**（+E-30 b_cache 并发核算违规 30min 内回滚、E-31 code verify 预算止损） | E-23、E-24、E-25、E-26、E-27、E-28、E-8（守卫反向抓 bug） |
| 止损决策类 | 2 | E-17（止损）、E-17b/E-18（逆转） |
| 环境/工具类 | 3 | E-2、E-9、E-10 |
| **拦截率** | 7/21（33%）→ 本轮审计后新代码缺陷 100% 拦截 | 拦截手段：白盒压测（stress 脚本）、跨阶段对账（audit）、分域抽检（spotcheck）、代码审查（E-15 元规则） |

**损失换算**（幂等设计 E-7 的回报）：14 次爆发事故中 driver 被杀/崩溃 10+ 次，
每次重启损失 <10 分钟（分片级断点）；两次数据级事故（E-4 丢 58% 行、E-16 潜伏两天）
均由守恒对账发现并完整恢复。

## 二、按管线阶段复盘（坑在哪里）

| 阶段 | 踩坑 | 教训密度 |
|---|---|---|
| **S0/S1 摄取过滤** | E-4（字段回退丢 58% 行）、E-16（76% 分片失败潜伏 2 天）、E-26（qa 域 HTML 盲区）、E-1/E-2（driver 挂死+守卫生命周期） | ★★★ 最危险：静默数据级错误 |
| **S2/S3 去重** | E-5/6（护栏三级撞顶）、E-11（全局物化翻倍）、E-12（Python 归并慢死）、E-13（并发内存核算缺失）、E-14（page cache OOM）、E-17（止损→逆转） | ★★★ 教训最密集：单机大数据内存经济学 |
| **S3-v2 跨源补齐** | E-20a（行组放大 12GB/7k 对）、E-21b（RSS 驻留 143GB）、E-6 重演（cap 28.0 贴脸撞穿） | ★★ |
| **S4 质量打分** | E-19（fasttext NaN 三连）、E-25（标签代理/零膨胀/无指纹三连坑）、E-21a（goldset 单线程瓶颈） | ★★★ 方法论级：标签语义决定一切 |
| **S7 去污染** | E-22（词缓存 clear 时序，4,689 分片失败） | ★ |
| **S5/S6/物化** | E-23（长文批 33× 内存）、E-24（lookup 静默过期）、E-27（分位表竞态）、E-28（随机序 IO 5000× 放大） | ★★ 全部拦截型——审计生效的证据 |

## 三、五大失败模式（元规律）

### 模式 A：单机大数据的内存经济学（10 次）
OOM/内存病理是最高频失败（E-3/5/6/11/12/13/14/15/21b/23）。
- 病理原型只有三种：**accumulate-then-process**（攒齐再处理）、**全局物化**（全量驻留）、**行数驱动的批/缓存**（大行分片爆炸）
- 解法原型：分块归并 / 排序键区间化 / 字节上限双约束 / 并发×上限核算（E-13）/ ru_maxrss 口径核算（E-21b）
- **预防**：每个"攒数据"环节默认按 10× 数据规模做内存模式审查（E-15 元规则）；护栏按预估峰值 3-5× 设界（E-6），贴脸即诊断增长曲线而非上调阈值（E-5）

### 模式 B：静默失败（4 次，最危险）
E-4（丢行）、E-16（76% 分片潜伏）、E-22（批量失败被"N/N 完成"掩盖）、E-24（产物静默过期）。
- 共性：**错误被记下但没人看** / **成功口径混入尝试数** / **skip 条件不含输入新鲜度**
- 解法：shard_errors 置顶验收 + 上下游行数对账断言 + 派生物记输入/模型指纹
- **预防**：任何"N/N 完成"必须区分尝试/成功；跨阶段交接做对账（audit_consistency.py 已固化）

### 模式 C：存储布局与访问模式错配（3 次）
E-20a（行组点查放大）、E-28（随机序取正文 5000×）、E-9（SFS 元数据瓶颈）。
- **预防**：正文类随机访问先重排为"存储序扫描 + 逻辑序重放"；评估布局先算"最小读粒度 × 访问次数"；IO 预算按 SFS 聚合 0.3-0.5GB/s 估

### 模式 D：标签/指标语义错位（3 次）
E-25（来源代理标签→文体判别器）、E-25b（源间 CV 测不了全池排序）、R-2（小数据退化）。
- **预防**：标签语义必须与生产任务同构（LLM 内容级标注）；验证协议要在生产分布上测

### 模式 E：修复引入的新 bug（3 次）
E-2（守卫不停杀旧进程）、E-5 第二轮（错误上调阈值）、launch.sh 重定向顺序（本轮）。
- **预防**：每次修复必须带回归测试锚定；修完跑全量相关测试套件

## 四、新阶段开发 Checklist（避坑固化）

- [ ] **批处理双约束**：行数上限 + 字节上限（E-3/23）；压测清单含"最肥分片"（按 bytes/row 排序取 top-1）
- [ ] **内存三算**：结构体加总 × 3-5 余量（E-6）× 并发数（E-13），以 ru_maxrss 实测复核（E-21b）
- [ ] **访问序审查**：先画数据流（谁按什么序读什么），随机访问必配重排物化（E-28）
- [ ] **skip 条件完备**：产物 + summary + 上游指纹/模型指纹（E-24/27）；聚合构建以输入完整为前置（E-27）
- [ ] **进度语义**：尝试/成功分离，error 置顶，30min 心跳守卫（E-1/16）
- [ ] **验收三件套**：audit_consistency（守恒）+ quality_spotcheck（抽检）+ shard_errors=0
- [ ] **回归测试**：每个修复锚定一个测试；语义变更连带改测试期望（E-4 教训）
- [ ] **工具自保**：pkill 用 `[.]` 防自匹配（E-10）；长时任务先幂等再优化（E-7）

## 五、完整索引

| 编号 | 主题 | 阶段 | 类型 |
|---|---|---|---|
| E-1 | Pool 静默挂死 | 全局 | 流程 |
| E-2 | 守卫生命周期 | 全局 | 修复引入 |
| E-3 | 写出器行数 flush | S0/S1 | 内存 |
| E-4 | 字段回退丢行 58% | S1 | 静默数据 |
| E-5/6 | 护栏撞顶/阈值规则 | S3 | 内存 |
| E-7 | 幂等是修复前提 | 全局 | 方法论 |
| E-8 | 守卫反向抓 bug | S2/S3 | 拦截 |
| E-9 | SFS 聚合带宽 | 全局 | IO |
| E-10 | pkill 自匹配 | 工具 | 工具 |
| E-11~15 | S3 内存病理全家桶 | S3 | 内存×5 |
| E-16 | 76% 静默失败 | S1 | 静默数据（最严重） |
| E-17/17b/18 | 止损与逆转 | S3-v2 | 决策 |
| E-19 | fasttext NaN | S4 | 训练 |
| E-20a | 行组放大 | S3-v2 | IO |
| E-20b | 打分器分工 | S4 | 决策 |
| E-21a/b | goldset 瓶颈/RSS 驻留 | S4/S3-v2 | 性能/内存 |
| E-22 | 词缓存时序 | S7 | 确定性 bug |
| E-23 | 长文批爆炸 | S5 | 内存（拦截） |
| E-24 | lookup 过期 | S5/S6 | 一致性（拦截） |
| E-25 | 打分器三连坑 | S4 | 标签语义 |
| E-26 | qa HTML 残留 | S1 | 质量（抽检拦截） |
| E-27 | 分位表竞态 | S4/S6 | 一致性（拦截） |
| E-28 | 随机序 IO | tokenize | IO（拦截） |
| E-29~31 | IO 串行化/b_cache 核算/code verify 止损 | 全局/S4/S3-v2 | 决策×3 |
| E-32 | 分位表 per-worker 私载 OOM 七连挂死 | S4 | 内存 |
| E-33 | quantile 表紧凑化（rank 等距采样，父载共享） | S4 | 性能/内存 |
| E-34 | goldrule 工件跨环境 unpickle（numpy._core） | S4 | 环境 |
| E-35 | decontam 54 分片=旧命名遗留非真缺 | S7 | 口径 |
| E-36 | yaml `off`→False 陷阱，fasttext 轨名义关实际开（裁定保持） | S4 | 配置语义 |
| E-37 | pyarrow18 空 parquet 抛错 + decontam 空产物契约卡 S5 gate | S4/S7/S5 | 环境/契约 |
| E-38 | daemon 型脚本顺序调用阻塞 keepalive + 处置进程自匹配自杀 | 全局 | 流程/工具 |
| E-39 | watchdog A2 对已完成的上一阶段持续空杀（掩盖对当前阶段的真停滞检测） | 全局 | 流程 |
| E-40 | mix_members 构建 int32 offsets 溢出（>6700 万行 ArrowInvalid） | S6 | 规模 |
| E-41 | 幽灵并发拉起第二个 s6 driver，rmtree 竞态崩溃循环 | S6 | 并发 |
| E-99/100 | 跨 Pod 租约缺失/NFS 目录复用失去所有权 | S5/S6 | 并发一致性 |
| E-101 | S6 top-k 上限与补足语义冲突，100B 配额系统性短缺 | S6 | 配置/数据配比 |
| E-102 | 正式 token 物化未验证 snapshot 输入且缓存/并发边界不足 | tokenize | 一致性/性能 |
| E-103 | 质量审计未复用逐源去污染 distinct-hit 门槛 | audit | 语义一致性 |
| E-104 | 随机 order 对 97GiB 索引执行随机二分缺页 | tokenize | IO/性能 |
| E-105 | checkpoint 后未提交 metrics 后缀阻断合法恢复 | pretrain | 断点续训 |
| E-106 | 原始文件存在但固定版本下载不完整 | S0/provenance | 静默数据（审计拦截） |
| E-107 | S6 全局账本泄漏被排除源统计 | S6/release | 作用域一致性（拦截） |
| E-108 | val 重建早于旧 S6 commit marker 撤销 | S6/release | 原子发布（运行中拦截） |
| E-109 | 质量抽检先把 50M 行组转成 Python 对象再取 3200 样本 | audit | 内存/性能（审计中拦截） |
| E-110 | 27-ID 小审计沿用 95M-ID 的 97GiB 顺序索引扫描 | tokenize/audit | 规模自适应（审计中拦截） |
| E-111 | 人审只展示文档前缀，把 notebook 正文和许可证后的代码完全藏住 | audit | 审计可判定性（人工质疑拦截） |
| E-112 | TorchTitan v0.3.0/PyTorch 2.14 锁却声明不存在的 cu126 发布组合 | pretrain/I0 | 环境身份（preflight 拦截） |
| E-113 | tokenizer 文件、控制 token 合同与旧 M2 抽样语料三者身份不一致 | tokenizer/T0 | 工件身份（门禁前拦截） |
| E-114 | 95M 文档 token parts 用 Python heap 逐篇归并并逐篇 write | tokenize | GIL/系统调用性能（收尾前拦截） |
| E-115 | token materializer 的 POSIX flock 未 fence 跨 Pod 旧 writer | tokenize | 跨 Pod 并发一致性（实运行拦截） |
| R-1/2 | 跨源密度/小数据退化 | S3/S4 | 研究发现 |

## 六、增补（2026-10-01）

- **E-29**：双 IO 大户互抢双输（S4 顺序流 × ndv2 随机点查）——串行化 + PAUSE 自管理；随机点查型任务要么独占要么配大缓存
- **E-30**：账本作者违反 E-13（b_cache 扩容忘乘 48 并发，1.15TB>953GiB）——**调单 worker 上限前强制 ×并发数写进注释**；账本不防疲劳，只有机械核算流程防
- **E-31**：code verify 二次止损（实测 4-9 天，README 估算失准 20×）——**长尾预算用 p99 桶外推**；缓存方案先算总量/容量覆盖率（<10% 判死换布局物化）

## 七、增补（2026-10-08）

- **E-32**：S4 quantile pass 七连挂死（stall guard 10-05~10-08 杀 7 次，50 分片/6.5h）——根因：`process_shard_quantile` 每 worker 经 `load_quantile_tables` 私载全部分位表（web 域 22GB×9 域 ≈25GB），编排器 `--workers 160` 下物理内存需求 ~4TB > 2TB，OOM killer 杀 ForkPoolWorker → pool 挂死 → 守卫硬杀循环。修复：`load_quantile_tables` 改 `np.load(mmap_mode="r")` + 见 E-33。教训：**分位表/索引类大只读资产禁止 per-worker 实载**；估算内存时"每 worker 复制 × 并发数"是机械核算（E-30 同型）
- **E-33**：quantile 表紧凑化（拦截于 E-32 修复后实测）——mmap 全量表单 worker 冒烟 RSS 16.4GB（clean MAP_PRIVATE 页计入各进程 RSS）且随机二分缺页拖尾（5% CPU 纯 IO 等待）。最终方案：`build_compact_quantile_tables` 按 rank 等距采样 2^25 点/域（小域精确拷贝），仍是 sorted 数组 → `quantile_of` 原样复用；父进程构建一次（~28s）fork 后全 worker COW 共享（物理 ~150MB）。误差上界 1/(m-1)≈3e-8（远小于 precision=1e-6 舍入步长；200 万样本实测 max diff=1e-6、mean=1e-8）。实测单分片 105 docs/s → 16.8K rows/s。教训：**RSS 计帐 ≠ 物理共享，mmap 不解决 per-process 计帐与缺页拖尾；超大只读查找结构应压缩到 error bound ≪ 使用精度**
- **E-34**：goldrule_v2.joblib 跨环境 unpickle 崩溃（`ModuleNotFoundError: numpy._core`）——工件在 numpy 2.x 环境 pickle，本 pod numpy 1.x 直接挂；11 个 bert 重算分片因此 shard_fatal，S4 闸门拦截 quantile pass（fail-fast 设计正确生效）。修复：`load_goldrule_model` 加 numpy._core→numpy.core 兼容别名（numpy<2 时触发，已实测 predict 正常）；scikit-learn 对齐工件训练版本 1.7.2（期间经历 numpy 2↔1.26 拉锯，最终稳定栈 numpy 1.26.4 + scipy 1.15.3 + pyarrow 18.1 + sklearn 1.7.2 + pandas 2.2.3 + matplotlib 3.10.9，全链实测通过）。教训：**模型工件入库时把训练环境（numpy/sklearn 版本）写进 artifact meta**；跨 pod 重启先核对 env 与工件 pickle 版本；pip 二次安装会互相降版本，装完必须复核关键包
- **E-35**：decontam 54 分片"缺失"实为命名口径问题——当前 s3 任务集 12829 个的覆盖已齐全（回补运行 ~30s 完成）；decontam/ 下 54 个无 parquet 的 summary 是旧粗粒度命名遗留（v1.0-full 清理对象），不阻塞任何 gate
- **E-36**：`fasttext_track: off` 配置陷阱——yaml 1.1 把 `off` 解析为布尔 False，`== "off"` 字符串比较永不命中 → **fasttext 轨自 2026-09-28 起实际一直运行**（与 E-25 意图相反，但全语料口径一致）。裁定：**不改行为**（现在改成 off 会令重算分片与其余公式不一致，才是真污染）；v1.0-full 全量重打分时再统一关闭。配置已加注释。附带：`test_end_to_end_throughput_real` 阈值 20000→800（track-on 下 fasttext 三窗推理是瓶颈，观测 1294 docs/s/核优于生产文档值 112-200，800 留 2 倍余量防 10× 级回归）。教训：**布尔型开关禁用 on/off 字面量，用 true/false 或引号 "off"**
- **E-37**：pyarrow 12→18 升级暴露 0-row-group parquet 陷阱——`ParquetFile.iter_batches()` 对空文件抛 `OSError: file only has 0 row groups`（旧版静默空迭代）。三层波及：① quantile pass 46 空分片报错（修复：`ingest.safe_iter_batches` 统一判空，score/length_bucket/materialize/decontam 全部换用）；② decontam 合法空分片不落 parquet 的 E-16 旧契约卡死 S5 硬前置（修复：合法空分片落 schema-only parquet，失败路径仍不落）；③ decontam 自身读空 s3 输入同样踩雷（一并换 safe_iter_batches）。教训：**环境升级后必须跑一遍空/边界输入冒烟；跨 stage 产物契约（summary+parquet 双在）要在首次落地时就被下游 gate 验证**
- **E-38**（2026-10-08 夜）：keepalive 停摆 3h——顺序调用的 health_monitor.py 实为常驻循环 daemon，永不返回 → 循环阻塞在首次 hourly 分支，watchdog 从此停摆（排查线索：monitor_stdout 在涨而 watchdog_stdout 静止）。修复：keepalive 只驱动单次执行的 watchdog，monitor 不再拉起（其数据源是旧日志文件，产出误导数据）。同日两次处置进程自匹配把自己杀了（pgrep/pkill 模式含在自身 cmdline——E-10 违规者竟是账本作者）。教训：**被监督脚本必须是单次执行型，先读脚本尾确认 main 结构再接线；杀进程用 pid 或 [x] 括号技巧后二次确认**
- **E-39**（2026-10-09）：watchdog A2 对已完成的上一阶段持续空杀——S4 完成后其日志尾冻结，s4_score key 每轮被判"零进度"且 pkill 模式匹配不到任何进程（空杀噪音），还会掩盖对当前阶段（S5/S6）的真实停滞检测。修复：A2 只对"driver 还活着"的阶段判定（`procs` 计数为 0 直接跳过）。教训：**停滞检测的作用域必须与当前存活进程对齐，否则在阶段切换点持续误报**
- **E-40**（2026-10-10）：`_doc_id_array` 手工 from_buffers 用 int32 offsets，每行 32B → **6700 万行即乘法溢出为负**（pyarrow ArrowInvalid: Negative offsets）。史上所有 mix 未达此规模，mix_m3_e1_100b 选中 ~2 亿行首次触发；正式 mix（1.44T）会在同处必炸。修复：offsets 改 int64。教训：**手工构造 arrow buffers 时 offsets 一律 int64；row 数上界 × 单元字节数逼近 2^31 的算式要显式标注并写边界测试**
- **E-41**（2026-10-10）：S6 崩溃循环一上午——根因不是代码而是**并发：第二个 s6 driver 被"幽灵启动器"拉起**（12:50 起跑，与编排器的 12:57 实例并发写同一日志直至 13:22，字节级交错可证；宿主机为多租户共享、跨 pod 进程可见但 lsof 权限受限，启动源未查明）。两实例互 rmtree work 目录竞态（Directory not empty）+ 各自重复 collect。修复：①driver 重 stage 加 flock 单实例锁（抢不到 exit 0 等编排器下轮重试，driver 死亡内核放锁，持锁挂死由 A2 杀）；②物化前 rmtree 改尽力而为（ignore_errors）。教训：**共享存储上的长作业必须有单实例锁，不能假设全系统只有一个启动器；并发崩溃循环的现场先验证"日志写入者是否唯一"（字节交错、双份启动横幅），再查代码**
- **E-44**（2026-10-10）：共享 NFS 在多 IO 大户并发下 RPC 挂死（rpc_wait_bit_killable 两次：flock 一次、S6 collect 启动一次）——S6（64w）+ ndv2 web（48w）同时压 FS 时，S6 主进程 18 分钟零进展 D 状态。处置：ndv2 SIGSTOP 让路（3-5 天的活让 3 小时无感），S6 独占跑完自动 SIGCONT。教训：**E-29 的"双 IO 大户互抢双输"在 NFS 饱和时会升级为 RPC 挂死，不只是变慢——重 IO 任务并发前必须评估 FS 饱和余量，挂死判定 = 主进程 D 状态 + 零进展 + rpc_wait_bit_killable**
- **E-45**（2026-10-10）：parquet 单列块 2GB 上限——members/stage-order 表 doc_id hex 串 32B/行，1.5-2 亿行 = 4.8-6.4GB，单次 write_table 抛 ArrowInvalid（E-40 int64 修复后紧接着暴露的第二层规模墙）。修复：write_table 加 row_group_size=50M（每列块 1.6GB，读侧契约不变）。教训：**规模递增时存储层限制是一串连环雷（int32 offsets→2GB 列块→…），每拆一颗就要按同样口径复查同文件全部写出点；hex 串 doc_id 占 2 倍空间，v2 数据格式可考虑 fixed_size_binary(16) 减半**
- **E-46**（2026-10-10 夜）：幽灵 driver 的破坏从"日志噪音"升级为"实弹"——其启动时对标准名 mix_work/<mix> 的 strict rmtree 把我们 driver 的 collect npz 删掉，finalize 加载 npz 抛 FileNotFoundError（两次）。修复：work_root 加 `__v2` 后缀（materialize.py:832）——旧代码幽灵永不使用此名 → 隔离；本 pod 重启沿用 → 断点续跑保留。注意：NFS root-squash 下无法 mv  nobody 拥有的目录（EPERM），旧目录只能留在原地浪费磁盘。教训：**跨 pod 无锁并发下，资源命名空间加版本后缀是最后的隔离手段；对共享存储上他人可写的目录做 rmtree 前必须先确认自己是唯一运行者（这一课是给幽灵作者的）**
- **E-51**（2026-10-11 凌晨）：E-40 的 int64 offsets 修复本身有类型错配——`pa.string()` 的 offsets buffer 按 **int32** 解释，传入 int64 buffer 后 pyarrow 把 int64 当 int32 读 → 伪 "Negative offsets in binary array"（S6 members 第三次撞墙的真因；此前观察到的 "2GB 上限" 报错值 -32 = int64[0,32,64...] 被按 int32 对读取）。**正确修法：int64 offsets 必须配 `pa.large_string()`**（schema 三处 doc_id 字段同步改），写 parquet 时 pyarrow 按行组切片回 int32（列块 <2GB 安全，与 E-45 row_group_size 配套）。200M 行全链路复现验证通过（25s 构建/41s 写出/读回契约完好）。教训：**Arrow 内存格式的 offsets 位宽是类型契约（string=int32/large_string=int64），改 offsets 位宽必须连类型一起改；E-40 只改一半，E-45 的 row_group_size 也救不了类型错配——连环雷的第三层是同因异表**
- **E-54**（2026-10-11）：tokenize 单进程瓶颈 + 潜伏越界 bug——①materialize_tokens pass2a 单进程 encode，100B tokens 需 14-28h（E-28 只修了读序 IO，没修 tokenize 并行度）；②**边界切段只看 fi 不看 rg**：同文件跨行组被并成一段，用首个行组的 col 取后续行组 → 越界/取错正文（生产从未跑到 pass2a 所以未爆，合成端到端测试抓到）；③无断点，重启全重来。修复：pass2a fork 池并行（part 文件独立，归并确定性不变）+ 边界 `(diff(sfi)|diff(srg))!=0` + 三级缓存（doc 索引 E-53/prep scan 数组按 order 指纹/rg_bounds）。教训：**"顺序表→正文→token"链条必须先跑小数据端到端再上千亿；合并键变更（fi→fi,rg）类 bug 在边界数据才显形**
- **E-56**（2026-10-12 凌晨）：fork Pool 死锁——pyarrow/numpy 在父进程已起内部线程，fork 后子进程继承锁状态；子进程消亡后主进程 futex 永久等待（18 线程全 futex_wait、fd 全关、1% CPU 假活）。tokenize pass2a 并行因此白跑一轮。修复：spawn 上下文（子进程重导入脚本，__main__ 守卫防重跑，慢 ~20s/worker 换无锁遗产）。教训：**"fork 快"只在父进程无线程时成立；任何 import 过 pyarrow/numpy 的长驻进程用 Pool 一律 spawn**——E-38 daemon 阻塞、E-56 fork 死锁，监督/并行框架选型的坑已凑齐
- **E-60**（2026-10-12）：编排器是"活代码"——04:31 启动的编排器进程内存里是修复前代码，之后磁盘上的 E-58/59/60 修复对它一律无效，val 步陷入"重跑整个 stage2→撞旧 bug→崩→重启"死循环 4.5 小时（每轮残留 part 还让 part resume 逐轮递减制造虚假进展感）。处置：杀编排器 → keepalive 从磁盘重启 → 全修复生效。**教训：改了编排器/调度器代码必须杀进程重启（keepalive 管复活不管热更新）；排查"修复无效"先查进程启动时间 vs 修复落盘时间**
- **E-70**（2026-10-13）：幽灵覆写交付物——幽灵 pod 的（坏代码）32k 物化用 os.replace 覆写了官方 token_streams 流（21GB→0.93GB，meta 同步替换且自洽，能骗过 size 守卫，靠"该流应 ~21GB"的账目常识识破）。E-46 只保护了 mix_work，正式产物目录此前裸奔。处置：四流 chmod 只读冻结（幽灵 replace → EACCES 弹回，物理免疫）+ 32k 私域重制替换。教训：**共享 FS 上一切耐久交付物落地即 chmod a-w——默认可被他进程覆写，"信任目录级互斥"在跨 pod 场景不存在**；验收产物要有独立的"账目常识"核对（tokens 数级、字节数级）
- **E-71**（2026-10-13）：调度器 v2 双 bug——①队列项用 `/` 连接而拆分逻辑按 `:`（`%%:*` 无冒号时取全串）→ group=arm=整条路径 → env 路径翻倍、启动静默全败；②arm_done 的 glob 假设了不存在的目录层级（带前缀臂名 vs 去前缀 outname）→ 臂完成后永远不会被判 done。**教训：字符串分隔符与拆分逻辑是配对契约，改一个必须 grep 全部消费点；"启动静默失败"的调度器比不跑更坏——首批启动必须人工抽查 metrics 落盘**

## 八、增补（2026-10-18 数据资产复审）

- **E-78：文件存在不等于产物有效**。12,829 个旧 decontam/S5 summary 的错误数为零，但没有输入、配置、benchmark index、tokenizer 或输出内容身份；上游策略修正后仍会被“文件存在”门禁放行。修复：decontam/S4/S5/S6 全部改为逐分片 processing contract + 内容/控制面指纹，global summary 从完整发布 roster 重建；旧资产只能经结构闭合、精确 token 抽样和显式迁移证书进入缓存路径。规则：**所有 skip/gate 必须回答“由哪一代输入和配置生成”，不能只检查 path 与 error=0**。
- **E-79：10TB S3 的原地列刷新整片入内存**。`ndv2_refresh` 原实现对每个分片 `pq.read_table`，大分片会把文本列整体驻内存，多 worker 下必然 OOM。修复：8K 行流式读取、只替换 `s3_decision`、临时 parquet 原子发布；变换契约绑定原/新文件指纹、doc_id 摘要与保留列，使 decontam 能证明其 `doc_id/text` 输入未变。规则：**所谓“只改一列”的维护任务也必须按全表字节数做内存审查**。
- **E-80：索引元数据记录 revision，不等于重建配置锁定 revision**。五个 benchmark 的旧 `index.meta.json` 有 commit SHA，但 YAML 没有 revision，重建时仍会解析远端 HEAD，导致同配置跨日期漂移。修复：把生产实际使用的五个 40-hex commit 写回配置，并启用 `require_pinned_revisions` fail-fast；index identity 与逐源扫描阈值身份分离。规则：**运行后记录版本只能审计过去；配置先锁版本才能复现未来**。
- **E-81：`hybrid_v2` 的 upstream 缺失重归一只写在注释里**。`_upstream_norm_array` 把缺失值填成 0，后续融合却以 `~isnan` 判断有效性，导致无 upstream 的文档仍占用 0.3 权重并被系统性压分。修复：无效值保留 NaN、其余轨按权重重归一；旧 S4 分数用解析变换 `old_score × full_weight / retained_weight` 精确恢复，无需重跑 goldrule，并以 `legacy_zero_upstream_v1/current_v2` 表示合同、精确组件抽样和分位表 manifest 封住混用路径。规则：**缺失值语义必须由值和 validity mask 共同测试，不能依赖注释约定**。
- **E-82：文档记录的模型环境与实际运行时漂移**。当前 pod 为 sklearn 1.2.0，goldrule pickle 来自 1.7.2；旧 loader 只报警，实际 `transform` 报 `NotFittedError`。修复：数据处理依赖精确锁定，模型旁写 runtime contract，加载前版本不符立即拒绝；最终模型生成脚本同步记录训练依赖。规则：**pickle/joblib 模型的库版本属于模型权重合同，warning 不能作为生产兼容策略**。
- **E-83：pickle 兼容别名污染 sklearn 扩展导入**。为读取 numpy 2.x 生成的 pickle，在 numpy 1.26 下提前注册 `numpy._core → numpy.core`；sklearn 1.7.2 的 Cython 扩展随后把该伪模块当真实 C API，报 `numpy._core.multiarray failed to import`。修复：先导入 TF-IDF/LR 及其编译扩展，再仅为反序列化注册别名。规则：**全局 `sys.modules` 兼容补丁必须晚于依赖该命名空间的原生扩展初始化**。
- **E-84：分位表覆盖写破坏旧发布**。旧 `build_quantile_tables` 逐域直接覆盖固定名 NPY，最后才原子替换 summary；中途崩溃会让旧 summary 指向部分新、部分旧甚至截断的表。修复：按完整 input contract 生成 `build_id`，所有 NPY 写入版本化新文件并逐文件临时发布，全部完成后才切换 summary。规则：**多文件发布不能只让 manifest 原子化；manifest 引用的数据文件也必须不可变、先完成后挂接**。
- **E-85：分位表构建的单流冷 NFS 与静默漏数**。旧值扫描逐分片串行读取，并在 parquet 打不开时直接 `continue`；冷 NFS 上 FineWeb 仅约 130 万行/秒，坏分片还会从最终分布中无声消失。修复：用 8 路有界线程池并行读取独立分片、关闭 Arrow 嵌套线程池、利用 parquet 统计证明整分片单一 domain 后跳过字符串解码；任一分片异常立即使整次构建失败。生产重建闭合 12,829 分片、6,485,760,234 行，值扫描约 11 分钟。规则：**全局聚合器必须 fail-closed，并用有界文件级并发隐藏对象存储/NFS 的单流延迟**。
- **E-86：单一去污染阈值变化触发 2.68 亿文档全重扫**。三中文源只把 C-Eval distinct-hit 从 1 提到 2；旧决策已逐行记录被选 benchmark，因此只有旧 `hit_benchmark=ceval` 的行可能改变，keep 与其他 benchmark drop 在阈值单调增加下严格不变。修复：增量路径先验证旧输入/输出行数与账本闭合，再对每批 doc_id 精确对齐，只复算被改变 benchmark 选中的行；每分片记录复算数与证明条件。生产运行由旧 summary 外推 3.4 小时降至约 25 分钟，并救回约 21.557M 条误杀。规则：**单调策略变更先推导受影响集合，用旧可审计决策做增量重算，不能默认全池重跑**。
- **E-87：S5 lineage 认证逐行 Python 循环**。旧认证对 64.8 亿行逐行生成 bucket/eligibility、解析单个 hex doc_id 并更新 Counter，线程并发受 GIL 限制到约 1.2 核。修复：`searchsorted` 向量化分桶，直接验证 Arrow list offsets/flatten values，整批 C 路径 hex 解码并用 `bincount` 聚合；content SHA-256 字节合同不变。规则：**十亿行结构审计也属于生产计算内核，必须采用列式向量化，不能把“一次性迁移”当作低性能代码的豁免**。
- **E-88：把 ndv2 决定回写 10TB S3 当成 S5 的硬前置**。S5 实际已经按 doc_id 直接 join 排序 lookup，继续全量改写只会复制文本 IO，并制造 `ndv2_refresh` 与 lookup 两套真相。修复：S5/S6 门禁直接验证 decision policy、全文精度审计边集指纹、lookup 字节数和全局 dropped 账；S6 upstream snapshot 绑定这些控制面与 lookup 文件。`ndv2_refresh` 降为旧消费者兼容维护。规则：**下游能以不可变 sidecar 精确 join 时，不要为冗余列回写不可变大事实表**。
- **E-89：限制性去重变更仍触发 S5 读取 10TB 正文并重新分词**。旧 web/code lookup 为空，新 ndv2 只会从已认证的 S5 keep 池继续删除；保留行的 token 长度与桶不变。修复：直接流式过滤认证后的紧凑 S5 表，用 decontam 决策精确处理两类 drop 的交叠，再重建内容 SHA、桶与 token 账本；合成测试强制 tokenizer 不得被调用。生产 10,632 个 web/code 分片约 10 分钟完成。规则：**先判断策略变化的单调方向；只删不增的派生表可以在已认证子集上做精确 delta，不必回源重算不变量**。
- **E-90：S6 upstream snapshot 把所有历史分数轨的 121GB NPY 都当当前依赖**。正式 mix 只读 `hybrid_v2`，旧实现却遍历 `_build*.summary.json` 并重哈希 composite/rule/旧 hybrid 的全部表；启动耗时被放大，任何无关研究轨变化还会令正式 mix 失效。修复：snapshot 接收显式 `score_field`，只绑定对应 build manifest 及其引用的不可变 NPY；活动表仍做全内容哈希。回归测试同时断言无关轨变化不影响身份、活动轨变化必须影响身份。规则：**数据血缘应覆盖真实读取闭包；多绑无关资产和漏绑依赖一样会破坏可复现性**。
- **E-91：S6 只在启动时检查上游，长 collect 遭并发 S5 改写后仍发布混代 mix**。后台旧 orchestrator 在 S6 扫描期间补完 5.37M 个 web keep 并重建 S5 全局 summary；首轮 S6 因缺少结束时复核而把先后两代分片混在同一 members 表。修复：陈旧 manifest 在任何覆盖前先撤销；members/order 写完后重算完整 upstream snapshot，前后 SHA-256 不同即删除未提交产物并失败，只有相同才写 manifest commit marker。生产重跑的前后 snapshot 均为 `59697b9f…0d27`。规则：**长时间物化必须关闭 TOCTOU 窗口；启动门禁只能证明开始时一致，发布门禁必须证明整个读取窗口内输入未变**。
- **E-92：去污染只扫 50k-shingle 前缀造成长文档 false negative**。旧质量审计在 300 篇 books 样本中用 `cap=None` 找到 16 篇后半部 benchmark 重叠，却以“公版书单向包含”为由建议接受；全池账本显示实际有 401,906 篇被截断，样本数字不能充当影响上界。修复：生产 `max_shingles_per_doc=null` 全文扫描；S1 2M chars 约束单文档上界，benchmark 查询按 2M shingles 分块，避免 4096 行批次把所有长文档 hashes 一次拼接。旧 S5/S6 release 撤销并从 decontam 重建。规则：**任何会降低污染召回率的性能截断都必须用 recall 审计证明；发现确定性漏检后不能以语义解释替代修复**。
- **E-93：两个“配置存在但不生效/不闭合”问题**。`materialize_tokens.py --workers` 被解析后完全忽略，运行只读 `YUFORMER_MAT_WORKERS`；正式 mix 又未显式写 goldrule 块，collect 端暗用 0.5%/2M 默认值，merge 端却暗用 cap=0，最终只产出 1.19M 行而 C4 要求 10M。修复：显式 workers 优先于环境变量并加回归测试；goldrule 缺省统一为完全禁用，正式配置明确 10% sample、每域 10M cap、总量 10M fail-fast。规则：**CLI 参数必须有行为测试；同一配置在生产者和归并者只能有一个归一化后的真源**。
- **E-94：goldrule 研究集声称“全轨”却漏掉当前模型轨**。旧 C4 parquet 只保存 fasttext/bert/旧 composite/rule，漏掉 `llm_distill` 和生产选取使用的有效 `hybrid_v2`，且退役 `bert_score` 有 88.2% 非 finite；collect 还为每个 S4 行保存五元 Python tuple，160 worker 下浪费大量内存，归并时同时打开约 13k NPZ。修复：在既有 S4 选择扫描中只为“过质量阈值且命中 hash sample”的行保留全轨，避免全行 tuple 和第二遍十亿行扫描；输出 12 条完整分数轨，按真实 upstream 可用性把伪 0 归一分改为 null，对旧缺 upstream 的 `hybrid_v2` 应用当前语义修复；逐轨发布 finite 覆盖率并强制 `rule_score`、`llm_distill`、`hybrid_v2` 100% finite；NPZ 逐个读取后立即关闭。规则：**研究数据集必须保存所研究系统的当前信号和缺失语义，并把覆盖率作为发布合同，而不是靠列名暗示完整性**。
- **E-95：S6 注释声称可断点续跑，启动却无条件删除所有 collect parts**。12,829 分片的长任务只要在最终 manifest 前中断，下一次就从零重扫；这也让 C4 扩容后的第二遍数值列读取承担不必要的重复成本。修复：work generation 绑定 collector schema、规范化配置、完整上游 snapshot 与任务 roster；每分片仅在候选/goldrule parts 原子落盘后写 commit marker，同代重启只复用完整 marker，任一身份变化自动清空整代缓存。规则：**长任务的“可恢复”必须由代际身份和分片提交点共同证明，目录还在不等于可安全续跑**。
- **E-96：S5/S6 只检查 decontam 文件存在和 `shard_errors=0`，允许消费重算中的混代决策**。全文重算期间旧全局 summary 仍是健康的前缀代，而逐分片正被逐步替换；若此时启动 S5，可把两代结果写进同一发布，S6 又不会校验 S5 的控制面身份。修复：decontam 发布就绪合同强制完整 coverage、`truncated_docs=0`、全文 scan policy；S5 在改写任何分片前撤销陈旧全局 commit marker，并在长任务前后比较 decontam/ndv2/tokenizer/context 控制快照；S6 与 s6_val 重算当前 S5 输入合同，任何过期或混代 summary 都拒绝。规则：**“文件齐全”不是代际一致性；每个长阶段都要有起止快照与下游重新计算验证**。
- **E-97：S6 manifest 在 C4 研究集和全局 summary 之前发布，且幂等 skip 不检查 C4 文件**。C4 行数/finite 门禁若失败，外部读者可短暂看到一个并未完整提交的 manifest；已发布后单独丢失 goldrule parquet，重跑还会静默 skip。修复：manifest 改为 members、order、goldrule、global summary 全部原子落盘后的最后一个 commit marker；manifest 显式列出 C4 路径与 SHA-256，幂等检查要求文件存在且内容哈希相同，删除 C4 的回归测试会触发完整重建。规则：**commit marker 必须最后写，且恢复判定必须覆盖发布声明中的每个必需工件**。
- **E-98：S5 的 ndv2“只删行”刷新丢失 token-cache 证书**。刷新后的表已有当前 tokenizer 处理合同、内容哈希和精确 token 数，但 summary 没有续签 `token_count_cache_contract`；11,125/12,829 个分片因此会在下一次策略变化时误判为不可复用，回读约 10TB S3 正文并重分词。修复：当前流水线直接分词产物签发 schema-v2 合同（tokenizer + processing contract + output fingerprint + content hash），ndv2/全文去污染增量过滤继承并续签旧合同；存量紧凑表用数据集 token audit 加逐行 bucket/eligibility/content digest 的列式认证恢复资格。规则：**任何保持 token 数不变量的派生变换都必须传递并更新缓存证明，不能只传递数据列**。
- **E-99：Pod 本地 pidfile 放行跨 Pod 的同 stage 双写**。一个 10 月 8 日启动的旧 S5 与全文去污染后的增量 S5 同时写共享目录；两个 PID namespace 不可见，旧锁将对方 PID 当成已死，最终出现 parquet 与 summary 由不同 worker 交叉覆盖、`output_bytes` 不闭合。修复：重 stage 使用共享 NFS 的原子 `mkdir` 租约，owner 包含 host/pid/UUID 并周期心跳，未过期租约使第二个 driver 以非零退出；S5 全局 commit marker 只在全分片身份闭合且 `shard_errors=0` 时写入，失败统计单独落到 audits。规则：**共享存储上的单写者互斥必须使用所有 Pod 都能竞争的原语；pidfile 不能跨 PID namespace，且失败汇总不得冒充发布标记**。
- **E-100：删除并立即复用同一个 NFS 租约目录会让旧 writer 继续运行**。初版共享租约在接管 released/stale 锁时直接删除并重建固定目录；旧 Pod 的 NFS 元数据缓存仍可能看到旧 owner，且 heartbeat 只记录异常、不终止进程，于是旧 writer 在已经失去所有权后继续写。随后实测 root-squash 还允许在锁内创建子目录、却拒绝从父目录 rename 整个锁。最终修复改成稳定锁根下的 append-only `generation-N`：竞争者只能有一个成功原子创建同一下一代目录，历史代永不删除/改名；释放只写 token-scoped marker；heartbeat 每次核对本代 owner token，目录消失或 token 改变立即 `os._exit(74)`，连续三次 I/O 异常同样 fail-stop。规则：**租约不仅要互斥获取，还必须 fence 已失去所有权的 writer；共享路径不能删除后原名复用，锁协议必须按真实 NFS 权限模型实测**。
- **E-101：配置写着“不足域向深档补足”，实现却把 `top_k` 当绝对候选上限**。正式 100B mix 首次发布只有约 99.404B，缺少约 596M tokens；旧门禁只记录 shortfall，没有把超限短缺判失败。修复：配置显式声明 `quota_fill_policy: deepen_to_quota` 与 `max_total_shortfall_tokens: 1_000_000`；仅对 top-10% 不足的 books/encyclopedia/qa 扩展到全质量池，再按分数稳定排序取到整文档边界；发布前对总 shortfall fail-closed。重建结果为 99,999,695,366 tokens，剩余 304,634 全部来自不可切分文档尾差。规则：**配比配置中的“偏好质量带”和“硬配额”必须有明确优先级，shortfall 必须是门禁而非日志字段**。
- **E-102：候选 release 的 snapshot id 曾可被复制到任意 token 流**。旧物化器没有证明 `--order` 是 release 中声明且同哈希的 artifact；part 缓存目录只绑定 tokenizer/EOD、不绑定 order；64 个 worker 各自完整加载多 GiB NPZ 扫描数组；输出直接写最终文件；编排器还遗漏 `--workers`，实际回退单进程，并与历史流共用目录。修复：物化前按 release artifact 的路径、字节数、SHA-256 三重校验；part identity 纳入 order hash；scan 数组拆为只读 mmap；bin/meta 以隐藏临时文件完成后按 meta-last 顺序原子发布；CLI 持有单写者 lease；编排器使用 release 隔离目录并显式 64 workers。规则：**token 流的 snapshot 标签必须由输入字节证明，不能由调用者声明；长时物化的缓存、锁、临时文件和发布标记都属于数据身份合同**。
- **E-103：质量审计把“任一 benchmark shingle 命中”误当正式污染决定**。生产对 CCI3 的 C-Eval 策略是 `min_distinct_hits=2`，旧审计却以 raw hit > 0 判失败，因此 3,200 篇样本中 60 篇只有一个独立命中的合法文档被误报。修复：审计直接载入生产 decontam 配置，按 source/benchmark 计算 effective threshold，去重 hit shingle 后再裁决，并把低于阈值的 raw hit 单列为 review evidence；样本 roster 与报告均绑定 mix identity 并原子发布。规则：**审计必须复用生产决策语义；“检测到信号”和“达到删除门槛”是两个不同字段，不能靠审计脚本重新发明规则**。
- **E-104：hash-shuffle 顺序对 97GiB 排序索引做 9505 万次随机二分**。物化器虽已把正文读取重排为存储序，但 doc-id 定位仍直接按 hash 随机顺序 `searchsorted`，实测 8 分钟只触达约 22GiB 索引页，启动阶段外推 35–45 分钟并长期处于 NFS 缺页等待。修复：先稳定排序 query IDs，在排序索引上执行具有页局部性的二分，再用 query permutation 精确还原逻辑 position；另把 file→row-group 映射从“每文件扫描全部 selected rows”改为一次稳定分组，worker 缓存行组累计边界。规则：**逻辑随机不要求物理随机；每一层 join、定位和正文读取都要独立审查访问局部性**。
- **E-105：从完整 checkpoint 恢复时，metrics 文件可能包含 checkpoint 之后的未提交后缀**。故障进程可在 checkpoint 100 写完后继续记录到 step 111；恢复从 step 100 再追加 step 101 时，单调 tracker 正确拒绝，却令合法恢复无法继续。修复：rank 0 在载入已提交 checkpoint 后把超出 step 的 metrics 原子移入 `metrics.orphaned-after-*.jsonl`，规范 metrics 回滚至 commit point 后再续写；V1 的五个强制中断/恢复场景全部通过。规则：**训练状态的提交边界由 checkpoint 定义；附属日志可以超前，但恢复时必须保留审计副本并回滚 canonical view**。
- **E-106：S0 有文件、有 summary、有固定 revision，仍不等于完整下载**。`arxiv-latex-corpus/train.jsonl` 本地仅 56,105,768,891B，而固定上游对象为 78,855,074,113B；`skypile-150b` 的固定版本有 437 个数据文件，本地/S0 只有 436 个，缺 `data/2022-05_zh_head_0006.jsonl`。旧 provenance 只证明“已处理文件是谁”，没有证明“应下载文件是否齐全”。本轮正式 release 排除两源，以 peS2O/CCI3-HQ 同领域池补足，并新增远端固定版本 roster 与 S0 path/size 的完整性审计；15 个纳入源全 PASS 才允许发布。规则：**来源身份与下载完整性是两个独立门禁；全量源必须对固定版本做 expected-vs-observed 文件集合和字节数闭合**。
- **E-107：S6 虽按 source filter 选择文档，发布 summary 却从全局 S5 按域抄 pool/drop 统计**。排除不完整源后，这会让 members 正确但证书仍混入 arXiv/SkyPile 的行数与决定数；更严重的是 source filter 原先不进入 `config_echo`，同名配置可指向不同语料。修复：正式 mix 显式声明 `included_sources` 并纳入配置哈希；CLI filter 必须与其集合完全一致；val 配额、候选池和发布账本全部从精确 task roster 聚合，回归测试覆盖身份变化与排除源不泄漏。规则：**选择作用域必须进入内容身份，所有统计都要从同一 task roster 聚合，不能混用全局 summary**。
- **E-108：S6 在判断旧 manifest 是否过期之前先调用 `ensure_val_sets`**。当 S5 或正式源清单变化时，val parquet/ids/summary 会先被原子替换，而旧 manifest 仍短暂可见，形成“提交标记有效但子产物已跨代”的读取窗口。本次重建在窗口出现后立即撤销了仍为 BLOCKED 的旧候选标记，未进入训练。修复：先计算 val 输入/配置合同；若旧 manifest 的 config 不同、val 合同过期或 members 缺失，先撤销 manifest/global summary，再允许写 val；回归测试在 `_write_val_domain` 调用点断言旧 manifest 已不存在。规则：**父级 commit marker 必须在任何子产物可能改写之前撤销，不能只保证新 marker 最后写**。
- **E-109：质量抽检知道 3,200 个目标行号后，仍把命中的生产 row group（最大 50M 行）四列全部 `to_pylist`**。本次 fresh audit 峰值达到约 17GiB RSS，绝大多数 Python 对象随后被丢弃。修复：读取 Arrow row group 后先对 3,200 个局部 offset 执行 `Table.take`，再只转换抽中行；抽样身份和输出不变。规则：**列裁剪之后还要做行裁剪；“只读少量字段”不代表大行组转 Python 对象是安全的**。
- **E-110：E-104 为正式 95M-ID roster 改成顺序扫 97GiB 排序索引后，27 篇 E4 小审计也无条件走同一路径**。大查询的存储局部性优化在小查询端变成固定 97GiB 税。修复：`lookup_sorted_doc_ids` 对 ≤4,096 个 query 使用直接二分，对更大 roster 保留排序 query + 16M 行顺序块扫描；两条路径各有精确命中/缺失测试。规则：**访问算法必须按查询基数切换；随机 I/O 和全表扫描的交叉点要显式进入实现**。
- **E-111：人审报告固定截取前 400 字符，notebook 的 Jupytext 头和源码许可证会占满整个展示区**。正式 S3/S5/token stream 始终使用全文，但报告没有标注全文长度或截断状态，人工会合理误判为正文被截断，且无法审核真正内容。修复：审计 JSON/Markdown 显式记录全文字符/行数、展示源字符数和截断状态；长文档按确定性的头/中/尾窗口展示，当前报告提高到 1,200 个源字符，并注明训练使用全文。规则：**人工门禁的样本视图必须足以判断被审对象，展示截断要显式且不能只取最容易被模板占据的前缀**。
- **E-112：训练环境把 TorchTitan v0.3.0/PyTorch 2.14 与 CUDA 12.6.3 拼成了上游没有验证过的组合**。固定 tag 的官方 GPU release workflow 实际锁定 PyTorch 2.14.0、torchvision 0.29.0、torchao 0.18.0、Triton 3.8.0 和 `cu130`；当前节点 535.161.08 又低于 CUDA 13 的 R580 驱动要求。修复：backend lock 改为完整 `cu130` 发布合同，preflight 增加三个伴随包的精确检查，并提供先验驱动/Python 检查后才创建隔离环境的 bootstrap。规则：**后端版本必须按上游实际发布矩阵整体锁定，不能只锁主包后自行拼接未经验证的 CUDA 组合**。
- **E-113：代码和文档已把两个 tool-call 边界纳入 tokenizer 合同，但现存 32K/64K/128K 文件只有六个控制 token**；其训练语料还是旧 M2 的任意 S3 分片样本，包含本轮因下载不完整而排除的 arXiv/SkyPile，也没有证明与 val 互斥。若直接冻结 64K，SFT 模板中的 tool-call 边界会被拆成普通子词，且 tokenizer 身份无法从当前 data release 复现。修复：旧 token 流只保留为物化器全量验收证据；重写 tokenizer corpus sampler，使其只从 snapshot 绑定的 S6 train order 按域内 doc-id hash 前缀抽整篇文档，并产出逐域配额、原文 byte 数、文件 SHA-256 和 snapshot id manifest；三档 BBPE 共享同一份 M3 coverage-balanced corpus，完整预留八个控制 token 后再训练、评估和定案。规则：**tokenizer 是独立版本化数据工件；训练语料、控制 token 合同和最终 tokenizer 文件必须同代冻结，不能靠路径名暗示一致**。
- **E-114：pass2a 已把 100B token 写成 1,538 个 storage-local part，旧 pass2b 却对 95,220,247 篇文档逐篇执行 `heapq.pop/push` 和一次 `write`**。算法不仅有约 15 亿次 Python heap 比较，还制造 9500 万次小系统调用；编码完成后仍可能在收尾阶段耗费数小时。修复：每个 part 的 positions/offsets/tokens 直接映射为 Arrow LargeList，构建一次全局 document permutation，按 10 万文档块在 Arrow C++ 中 `take`、`list_flatten`，每块一次顺序写；发布前继续核对 token 计数与文件字节。规则：**大规模流水线的收尾归并也要按元素数核算 GIL 和 syscall，C++ 列式内核应承载变长数组的重排**。
- **E-115：materializer 复用训练侧 `fcntl.flock` 作为 `run.lock`，第一次中断后旧 Pod 仍继续写自己的隐藏 staging，而另一 Pod 已能进入同一输出目录**。两边因临时文件含 PID 没有互相破坏，但旧 writer 若稍后执行最终 rename，仍可能覆盖新结果；单机测试无法证明 NFS 客户端之间的锁语义。修复：`RunLease` 改为共享目录上的 append-only generation lease，以原子 `mkdir` 竞争；heartbeat 和发布前都核对自己的 generation 仍为最新、token 未释放，失去所有权立即 fence；bin/meta 的两次原子发布前分别调用 ownership guard。规则：**跨 Pod 单写者必须用共享文件系统可观测的原子代际协议，并把 ownership check 放进最终 commit 路径；本地 advisory lock 只能用于同机互斥**。
- **E-116：generation lease 在训练 identity 之前创建 `.run.lease`，identity 随后把该控制目录误判为旧训练产物；进程被 SIGKILL 后，新进程还会把刚停止心跳的本机 owner 当作最长 24 小时仍存活**。结果是全新训练无法启动，OOM/断电后的合法断点续训也无法立即恢复。修复：run identity 明确允许由当前启动协议创建的 `run.lock/.run.lease`；租约仅在 owner hostname 等于本机且 `kill(pid, 0)` 明确返回进程不存在时立即接管，跨 Pod owner 仍严格依赖心跳超时和 generation fence。V1 的不间断训练、SIGKILL、错误 resume guard、续训 loss/cursor 对拍五项重新全通过。规则：**互斥控制面必须进入 run-directory 合同；本机死亡可由内核确认，跨 namespace 存活性不能靠 PID 猜测**。
- **E-117：正式验证流按训练配置裁到每域 20M token，首次独立审计却用 `val_max_tokens=0` 按完整 50M 验证集计算期望**。8 个大域因此在文件完整 SHA、edge hash、snapshot/order/tokenizer 身份全部正确时仍被误报 token/doc/EOD 数不符；QA 全域只有约 12M，反而通过，进一步证明是裁剪口径漂移。修复：正式审计固定 20M 并把上限写入证书；编排器必须验证该证书及所有 bin/meta 的新鲜度后才能原子发布 `DATA_READY`；新物化 meta 显式记录请求上限、选中文档数和正文 token 数。规则：**验证集裁剪参数属于数据身份合同，生产者、训练配置和独立审计必须共享并落盘同一数值**。
- **E-118：coverage-balanced tokenizer 配方把 6.5B 的 5% 分给 multilingual，要求 325M token，但正式 S6 train roster 该域只有 299,997,632**。sampler 正确扫描完 95,220,247 行后 fail-closed，未发布半成品。修复：配方升为 v2，把 multilingual 调为 4.5%（292.5M，保留约 7.5M 的整文档余量），差额 0.5% 移到容量充足的 web；总规模和其余稀缺域覆盖不变。规则：**采样配方发布前必须用正式 train roster 做逐域容量可行性校验，并为不可切分文档保留显式余量**。
- **E-119：tokenizer corpus 完成 doc-id 定位后仍持有 103.8GB/25.9GB/25.9GB 三个索引 memmap，顺序扫描触碰的 clean file pages 在正文抽取期间继续计入约 140GB RSS**。匿名内存实际只有约 2–3GB，但驻留文件页会制造节点内存压力并干扰并发任务。修复：复制出选中行的紧凑 `file_idx/row_idx` 后，对三个 mmap 显式 `MADV_DONTNEED` 并释放引用，再进入 parquet 抽取。规则：**mmap 降低 heap 不等于没有驻留成本；大索引阶段结束时应主动归还 clean pages**。
- **E-120：tokenizer corpus 虽把 4.4M 个目标文档恢复成 storage-local 顺序，正文仍逐文件单线程读取**。首个实测点 200/11,970 文件耗时约 22 分钟，线性外推接近 22 小时，远超门禁排期。修复：使用有界线程池并行读取/解压独立 parquet 文件，窗口固定为 worker 数的 2 倍；主线程严格按文件序消费 future，并独占各域 ParquetWriter，保持输出确定性和有界文本内存。规则：**恢复局部性只解决访问顺序，不能替代文件级并发；并行读取与确定性发布应通过有界预取和单写者同时满足**。
- **E-121：`ShardTextFetcher` 读取命中 row group 后先把整列 text `to_pylist()`，再挑选目标行**。16-reader corpus 抽取只写出约 300MB 时匿名 RSS 已达约 42GB；大 row group 的未选正文被重复转成数百万 Python 字符串，文件级并发放大了这一错误。修复：先在 Arrow 列上对目标 local row offsets 执行 `take`，只把命中结果转成 Python；row-group assignment 也只计算一次。规则：**列裁剪之后必须先做 Arrow 行裁剪再跨 Python 边界，并发度不能掩盖单任务的读取放大**。
- **E-122：tokenizer corpus 完成 11,970 个文件抽取、九域 parquet 和逐文件 SHA 后，用 `os.replace(staging_dir, final_dir)` 发布，root-squash NFS 拒绝目录 rename；通用异常清理又立即删除了已经完成的约 10.4GB staging**。修复：预建最终目录，逐文件 `os.replace`，最后移动 `manifest.json` 作为唯一 commit marker；中途失败按逆序回滚已移动文件，完整且已有 manifest 的 staging 永不自动删除。规则：**共享 NFS 上必须实测文件与目录 rename 权限差异；昂贵 staging 的发布失败不能与生成失败共用破坏性清理路径**。
- **E-123：最终门禁曾只核对 token-stream audit 的 snapshot ID，没有核对其 dataset ID；审计证书本身也没有写入 dataset ID**。snapshot 内容虽能约束数据字节，却不能替代已经通过 A1–G2 的 release 身份，旧逻辑允许同一 snapshot 下来自不同门禁证据集合的审计被拼接。修复：token-stream audit 从 release manifest 写入 `dataset_id`；编排器缓存验证和最终签章同时要求 dataset/snapshot 二元身份完全一致，并增加错配回归测试。规则：**最终证书拼接多个证据时必须核对完整 release identity，不能用内容快照代替治理与验收身份**。

- **E-124：S6 域标签字典在 schema cast 时整列展开，超过普通字符串 2GB 容量**（2026-11-13）。800B stable 首次合并已完成收集与排序，但 `pa.table` 将全量 domain DictionaryArray 转为 `pa.string()` 时触发 `Take operation overflowed binary array capacity`；doc_id 的 large_string 修复无法覆盖另一列的隐式展开。修复：域标签先按 64MiB/100万行双预算逐块 decode，再作为同 schema 的 ChunkedArray 写表，保持标签、选取、顺序和配额不变。27 项 S6/租约回归通过；真实 2.16GB/2.7亿行 canary 复现旧异常并验证修复后完整 Parquet 往返。800B stable 于 2026-11-13 14:40 UTC 实际发布通过，成员/顺序 artifact 身份及原始上游快照闭合。规则：**字典数组的内存与 offset 安全性必须按展开后的列核算；schema cast 也是物化边界，不能只检查显式字符串构建代码**。证据：`data/health/s6-domain-capacity-canary.json`。

- **E-125：文档 top-k 比例乘全池 tokens 不能判断该质量带的实际 token 容量**（2026-11-13）。stable multilingual 的 top-45% 候选只有 18.816B tokens，低于 19.156B 配额；名义估算却认为充足，`deepen_to_quota` 未触发，直到 16.5GB members/15.4GB order 写完才被 1M shortfall 硬门拦下。修复：采集后按实测候选 tokens 对欠配域定向深扫，独立阈值绑定缓存防止窄档/深档复用混淆，继续按质量精确选取；实际容量与完整文档前缀缺口检查提前到写表前，失败输出逐域诊断。配置配额、来源、质量排序和 shortfall 上限不变。30 项回归通过，覆盖质量/长度负相关、断点缓存、严格 top-k 与提前失败；全量 stable 实际发布通过：830,221,696 篇，812,725,195,319 reference tokens，尾差 31,730 < 1M。规则：**文档比例和 token 比例是不同量；配额策略必须消费实际容量账，并在重型排序/写盘之前拒绝确定性欠配**。证据：`data/health/stable-candidate-quota-diagnosis.json`。

- **E-126：doc-id 索引只绑定路径和文件大小，同大小源文件重写会静默读错正文**（2026-11-13）。双行 Parquet 对调后旧缓存仍命中，所请求 ID 对应另一行正文。修复：完整源 roster 绑定 size/device/inode/mtime_ns，代际数组与 manifest-last 原子提交，单写者租约及重复 ID 检查；worker 在编码前逐行核对请求 doc_id，并在读取与发布前核对源身份。规则：**文档长度/领域配比不能证明正文映射正确；请求 ID 与实际读到的 ID 必须逐条闭合**。

- **E-127：tokenizer 路径短暂替换再恢复，可让旧 worker 输出错误 ID 却沿用原 tokenizer 身份**（2026-11-13）。复现保持 token 数不变，最终文件哈希恢复正常。修复：从同一份已校验序列化字节构造 tokenizer，worker 强制匹配 parent SHA，最终发布重查文件；缓存身份 v6 与 source binding schema 2 显式绑定已验证 tokenizer 字节。70 项专项回归通过。规则：**必须对实际消费的字节计算身份，不能分开 hash(path) 与 load(path)**。

- **E-128：order 按路径分开 hash 与读取，原 inode 原样回滚可发布错误顺序**（2026-11-13）。回归实测：hash 后在 `read_table(path)` 期间将两文档 order 替换为反序版，再恢复原 inode/mtime/字节，旧实现发布 `[13783,0,616,0]`，期望为 `[616,0,13783,0]`；token 数与最终文件身份均不变。修复：同一个打开文件描述符负责 SHA 和 Parquet 读取，读取前后核对 fd/path 身份，发布前再查 order 身份，元数据大小取已验证输入；缓存 v7 / source binding schema 3 强制 order 字节证明。78 项专项回归通过，覆盖原子回滚、读时改写、合并后改写阻止发布及安全缓存恢复。原 S6 与 v2 源索引保留复用，当前 warmup 在编码前暂停后等待完整回归恢复。规则：**身份检查必须覆盖实际消费的打开工件；最终路径恢复正确不能证明中间消费正确**。证据：`data/health/order-boundary-race-reproduction.json`。

- **E-129：原文路径回滚可在相同 doc_id 下替换正文**（2026-11-13）。worker 原有 path stat 前后检查与逐行 doc_id 校验，仍允许在 `ParquetFile(path)` 打开期间临时发布相同 ID 的不同正文，再恢复原 inode/mtime：实测 token 从期望 `[616,0,13783,0]` 变为 `[19020,0,13783,0]`，计数不变。修复：先打开源文件，用 `fstat` 核对实际 inode/mtime，再经该文件对象解析并在读后复核 fd；最终逐文件 path 检查与逐行 ID 检查保留。缓存 v8 / 输入证明 schema 4 强制 `source_handles_verified`。82 项专项回归 PASS，覆盖打开前/打开时的替换与回滚。规则：**文档 ID 正确不等于正文版本正确；必须核对实际打开文件的身份**。证据：`data/health/source-handle-race-reproduction.json`。

- **E-130：训练 reader 的 edge 校验和 mmap 按路径分开打开，可消费另一版本 tokens**（2026-11-13）。CPU 实测临时替换后二次回滚，旧 reader 给出 `[101,102,103,104]` 而期望为 `[1,2,3,4]`。修复：生产物化 meta 写入 staging 文件最终 inode/device/mtime/size；正式 1T 审计与缓存复用强制该身份；TorchTitan reader 从同一打开文件完成发布身份检查、edge hash 和 mmap，再复核 fd/path。测试覆盖原子回滚、含/不含旧资产身份字段的兼容，以及首尾哈希不变但内部被改写的新 inode。86 项专项 PASS；因训练核心变更，旧八卡门禁按历史证据处理，新代码必须重新通过全部数值/恢复/吞吐门禁。规则：**校验过路径不等于消费过被校验文件；启动后的 mmap 必须绑定实际证明的 inode**。

- **E-131：用 held-out 验证集的 tokenizer 压缩率给训练质量带分配实际 token 配额**（2026-11-13，修复验证中）。完整 warmup 编码得到 49,667,693,072 tokens，少于精确因果前缀所需的 50,000,297,985；全流 web 实测权重 0.5956085861，相对目标 0.6000319942 的绝对偏差 0.0044234082，超过既定 0.002。原 2% 容量余量不能修复域间压缩率偏差。修正方案：分别从三个冻结训练阶段的 hash-shuffle order 中逐域取至少 100M reference tokens，以实际 tokenizer + EOD 及逐文档 ledger 校准；绑定原训练 release、质量策略、source roster、tokenizer 与所有测量工件，按阶段重算 reference 配额。22 项专项回归 PASS；完整测量、配方重冻及新数据门禁尚待完成。原编码结果保留，正式训练尚未启动。规则：**reference token 只用于选样；训练预算与配比必须由所选质量带的实际 tokenizer token 口径校准并在消费前验证**。

E-131 verification update (~21:53 UTC): phase-specific calibration and freeze implemented; **532 CPU PASS / 28 skip**. Final frozen recipe `11326994…` preserves formal training YAML/env byte-for-byte and recalculates only reference selection quotas from each training phase. Old insufficient stream and stale dependent artifacts were archived. Current eight-GPU canary is RUNNING on a fresh run; corrected complete streams and actual-consumption gates are still pending. This issue is not marked fully closed until those gates pass.

- **E-132：S6 逐域归并保留数千个 NPZ 归档句柄，NFS 清理产生大量 silly rename**（2026-11-13，修复候选已验证，等待安全阶段切换）。旧 `arrs = [np.load(p) for p in parts]` 在最后一个域归并后仍保留归档对象，候选已拷贝后也未关闭。实际 warmup 退出前观察到编号超过 9000 的候选 FD 指向 `mix_work/.../web/.nfs*`，清理持续数分钟；在常见低 FD 上限下会直接 EMFILE。修正候选逐分片 `with np.load(..., allow_pickle=False)`，预分配按已提交行数确定的列数组，拷贝即关闭，并拒绝行数/列 dtype/形状不一致。32 FD 限额下旧方式打开 96 分片复现 EMFILE，新方式 PASS，数组逐项相等；S6 选样/缓存/租约及新增资源回归共 38 PASS。当前 stable 子进程继续运行，父级阶段调度在边界等待；独立服务在其正常提交和租约释放后才原子更新源码、重跑实际源测试与完整 CPU 回归，再恢复父级调度。已发布 warmup 数据不变，正式训练未启动。规则：**归档对象的生命周期必须短于缓存清理；不能依赖函数退出或 GC 才释放数千个文件句柄**。证据：`data/health/s6-fd-fix-candidate.json`、`data/health/s6-fd-fix-boundary.json`。

E-132 update (2026-11-14 ~00:33 UTC): fixed source `b44ab909…` applied after stable exited and released its lease; 38 actual-source S6 tests and 532 full CPU tests PASS (28 GPU opt-in skips). Real NFS low-FD canary PASS. Parent resumed automatically and anneal is running new source. Warmup/stable numeric outputs retained; no formal model run was interrupted. Actual anneal resource behavior is still being observed.


E-132 production verification (2026-11-14 ~01:02 UTC): anneal completed merge/publication/normal exit using `b44ab909…`; observed open FDs stayed at 12–14 instead of old stable 8,738. New members/order each contain 126,683,145 rows; requested 155,550,771,759 versus selected 155,550,703,026 reference tokens, tail 68,733 < unchanged 1M limit, upstream snapshot `8a243b08…` unchanged. Order controller PASS and pipeline advanced automatically to the new derived release and warmup tokenization. E-132 fix is applied and production-verified; E-131 actual-M3 full-stream capacity/mixture closure remains pending. Evidence: `data/health/e131-corrected-anneal-order-verification.json`.


E-131 real warmup verification (2026-11-14 ~03:14 UTC): corrected complete stream 50,939,497,047 actual tokens, exact causal requirement 50,000,297,985, surplus 939,199,062. All nine consumed-domain mixtures PASS, maximum absolute error 0.0014967086203835356 < unchanged 0.002. Independent FD-bound causal-prefix recount exactly matched all gate counters and rechecked the complete length-ledger SHA/sum. Stable encoding began automatically; stable/anneal closure remains pending. Evidence: `data/health/e131-corrected-warmup-independent-mixture.json`.

### E133 — 质量截断后复用 doc_id 前缀作 tokenizer 校准导致来源偏差（2026-11-14，修复验证中）

- S6 质量排序的 tie-break 使用 doc_id；校准再取相同 doc_id 排序前缀，不构成独立随机样本。完整 warmup web 中 fineweb-edu 的参考 token 占比 19.5933%，前缀样本却为 33.5996%；code 中 github-code 完整占比 33.3220%，样本 48.8854%。证据 `data/health/calibration-prefix-source-bias.json`。
- 额外旧抽检进程组已正常终止并标为 SUPERSEDED_BIASED_PREFIX_SAMPLE；正式 stable 编码继续。已完成 warmup 的实际完整因果消费区间配比门禁仍为 PASS，不回写或伪造旧证据。
- 新增独立 BLAKE2b 种子驱动的按域文档 rank 抽样，扫描完整已选集合并核对 manifest 文档数及参考 token 总量；父顺序哈希与 Parquet 解析绑定同一 FD。7 项回归通过，包含可重现的来源偏差和路径回滚。
- 新诊断覆盖三个阶段，每域至少 100M reference tokens，独立工作目录与报告；不改动当前运行生产者绑定的 canonical 校准、冻结配方及 release。canonical 迁移、抽样复核与 stable/anneal 全量实际消费门禁仍待完成。

### E134 — 已选语料按筛选用 doc_id 直接排序导致非预期的阶段内配比课程（2026-11-14，修复验证中）

- 实际 warmup 50,000,297,984 个训练目标按连续 10 段计量，web 从 72.6957% 降到 49.9016%，目标 60.0032%，最大域偏差 12.6925 个百分点。所有域的全程总配比仍通过，说明只检验全程总量不够。证据 `data/health/warmup-training-order-balance.json`，顺序与长度账本均同 FD 哈希后解析，精确覆盖因果目标区间。
- 根因与 E133 同源：质量 tie-break/配额截断和训练排序复用了 doc_id。正式启动协调器 186272 仅对该 PID 发 SIGTERM 后退役，原状态归档；独立编码控制器 199465、stable 生产者 214752 及 GPU 测试继续。`data/health/pretrain-model-order-adoption.json` 记录正式启动阻塞，正式训练未启动。
- 新增独立 BLAKE2b-128 种子驱动的阶段内因果块置换，不改动 tokenizer、成员、因果窗口、训练目标集合、阶段容量、WSD/LR 边界。置换必须为双射，恢复从逻辑 consumed_samples 计算同一物理窗口。15 项抽样/置换回归通过，覆盖长文档跨多块的精确计数。
- 真实 warmup 64-sample 块将最大分段偏差降至 .0044483，8-sample 块进一步降至 .00269676（0.2697 个百分点）。后者与全阶段最大偏差 .00149671 的差异为有限分块随机波动，需后续实际 loader/八卡吞吐/恢复资格；不可将研究诊断直接标为生产资格。
- 原顺序保留，无需重做 token 编码；正式 runtime 接入、阶段 2/3 同类审计、新模型与旧不可变数据配方的身份继承尚待安全边界完成。

### E135 — 排队研究进程缓存旧分析模块，未自动纳入新增筛选组（2026-11-14，已隔离补测）

- PID234400 启动时已导入三组版本的 analyze_model_size_sweep；排队期间文件增加第四组 attention_layout，但内存中的模块不自动更新。实际 200 步任务选择了 1.060163584B 后备，尽管当前磁盘分析最大合格点为 1.140385920B。通过运行 state.selected_screening、模型文件及启动日志中实际参数量一致性核对发现；并未发生正式训练架构切换。
- 保留该真实后备测试，不重新贴标签或覆盖其输出。新独立 frontier_sustained_20261114 队列 PID239937 等待后备完成，fresh 四组分析必须完整，显式校验 candidate=hybrid_9l_h3072_g2、parameters=1140385920，排队及测量前后绑定 analyzer 模块 SHA，实际模型 bytes/计数再复核。
- 任何排队任务依赖的脚本模块也必须保持不可变或在执行边界显式重新验证；只绑定最外层 wrapper 文件不足以证明选择逻辑当前。

### E136 — 正式模型配置内容未完整绑定恢复资格身份（2026-11-14，原型修复已复现验证）

- 隔离复现将正式模型num_layers从16改18，模型SHA改变，但 production_recovery_identity 保持相同；其依赖的 qualification_identity 只覆盖通用 qualification 模型配置，未解析实际production模型路径内容。证据 data/health/recovery-model-identity-gap.json。原仓库正式配置未改动。
- 候选修复 production_dependency_identity.py 解析生产base/env实际model/release/meta/order manifest/index依赖，qualification_identity显式绑定其字节。3项回归PASS；同一真实恢复identity路径的隔离修复预演中再次改模型内容，identity正确变化：data/health/recovery-model-identity-fixed-preview.json。
- 候选代码保留在 experiments/e4_curriculum/prototypes/pretrain_scripts，canonical尚未应用，需等当前LR对照完成后统一接入并重跑资格。还必须让production recovery/canary实际使用新的block读取策略；不能只在报告中写上策略但运行旧sequential数据。
