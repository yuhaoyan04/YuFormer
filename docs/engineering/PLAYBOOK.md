# LLM 全流程工程经验手册

依据本项目实际故障与验证整理。历史原文见 [工程经验记录](../research/工程经验记录.md)，案例索引见 [异常案例库](INCIDENTS.md)。迁移到新模型、数据和硬件后，必须重新做资格验证。

## 数据治理与预处理

先对账来源、分片、成功/失败和实际字节，再看总 token。任务尝试数不能充当成功数，上游缺分片时拒绝聚合。`sampled=0`、空文件和缺域必须有明确语义。幂等 skip 绑定输入、模型与策略指纹。

读批、编码、缓存、写批和归并逐层设置字节预算。用最肥的 books/code 分片压测，按实测 RSS × 并发核算 cgroup；区分匿名内存、共享内存和可回收文件缓存，`memory.failcnt` 不能单独证明 OOM kill。

均匀随机点查不依赖 FIFO 缓存。先按物理 file/row-group 扫描，再按冻结逻辑顺序重放；候选集使用外部分桶、块归并与有界句柄。prefix-only 相似度可能误判 boilerplate；HTML 清洗不能破坏 code/math；质量分类器须核对语言结构。

## Tokenizer 与数据加载

用最终 tokenizer 重算实际消费前缀，reference-token 估算不能代替训练 labels。EOS=0、特殊控制 token、chat template 和 tokenizer SHA 都是契约，不能随意把 0 当 padding 丢弃。

检查 S+1 causal shift、阶段首末样本、global sample cursor、rank 分片、block permutation 和 resume 后下一批。按实际消费集合审计课程配比，同时约束重复与 held-out 重叠；抽样点查不能冒充全流审计。

## 模型、吞吐与八卡训练

规模筛查先统一 global batch、真实数据、步骤与评测口径，再比较稳态 P10、显存、重试和质量。GPU utilization 不等于有效 MFU，短时峰值不能替代持续 canary。

锁定依赖后先做数值/梯度对拍，再跑八卡持续测试。核对 rank 到物理 UUID 的映射；CPU checkpoint writer 也可能因反序列化 CUDA tensor 在错误 GPU 建 context。长序列必须实际测 16K/64K，不能从 4K 显存线性外推。

总工期包含 WSD、长阶段、验证、保存与 duty factor。模型增大是否更充分利用算力，由实际每卡 labels/s 与同口径质量共同决定。

## 学习率、课程与恢复

分别对账已完成 step、下一次 optimizer LR 和日志时机。检查 warmup 最后一步、stable 首步、decay 首末步和非 log 对齐的终点；类似余弦公式可能遗漏上游虚拟终点。

同阶段恢复对齐六计数、Adam step/LR、全 rank RNG 和游标，并比较逐步 loss。跨 context 只加载权重并重置状态。保存成功还要验证 shard、控制状态及必要的全浮点有限性。

不只看 RUNNING 或 `kill(pid,0)`：核对 worker 启动时间、父子关系、日志进展、GPU 活动和 zombie。外围 owner 已死但训练存活时，经证据确认后显式接管原 generation/token 的续租；创建新 generation 会 fence 活 worker。接管必须保留原证据与独立验收。

## 验证与阶段准入

预先冻结 held-out 位置、样本数、目标 token 与计分规则。保存逐样本输出和可重读事件前缀。逐域短窗回升保留观察，不能未经证据称为噪声，也不能改配方后抹掉记录。

域 loss 连续回升时，把质量观测和输入计数分别核对。先验证同 attempt、LR、阶段与游标，再用真实 runtime permutation 对相邻已消费区间计数。全局 sample cursor 先扣除前序阶段样本数；物理 block id 再扣除已选择 source window 的 offset，最后索引经 SHA 绑定的域计数矩阵。直接把物理 id 当矩阵下标会数错数据。`tools/audit_live_domain_windows.py` 提供只读实现和边界反例；缓存矩阵必须重新绑定实际数据、顺序和几何，不能继承旧配方 PASS。份额接近全阶段均值只能说明粗域输入未明显失衡，不能证明域内语言/质量分布或验证波动的根因；不临时改阈值签发混合门禁。

区分 smoke、系统资格、训练期验证和最终能力。子集评测明确案例数与污染边界；gold 自测失败不能签发可靠分数。长文 BPB 以真实 UTF-8 字节为分母，EOD 不虚增目标。准入由实际 baseline/candidate 与冻结规则重算，拒绝手填 PASS。

## SFT / RL：待实际训练验证的迁移要求

SFT 检查对话状态机、tool-call id、arguments JSON、reasoning/content 分离、assistant-only loss、packed 对话 attention 隔离和 position reset。存在 packing 模块不代表训练闭环完成。

RL 使用独立环境验证模型/推理 adapter。更新前对账每题 G 个 rollout、重复 ID、reward 有限性、零方差组、截断、KL、importance ratio 和 clip。verifier 与 held-out evaluator 隔离；代码奖励使用受限执行环境与对抗测试排查 reward hacking。当前已有契约，尚无正式 SFT/RL 训练结论。

## 事故结束条件

记录“症状 → 证据 → 根因或未知项 → 最小修复 → 实际验证 → 复用规则”。保留失败与范围，添加能阻止同族回归的测试或审计。清理重型产物时保留来源清单，避免后续项目重复猜测。
