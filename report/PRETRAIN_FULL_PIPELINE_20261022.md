# E1–E3 全量流水线（2026-10-22 UTC）

> 以下为当日早先的大预算历史记录。当前已切换到 1B 初筛及候选复验，且修复 EOD embedding
> 错误并重跑 GPU 资格；以 [PRETRAIN_1B_EXECUTION_20261022.md](PRETRAIN_1B_EXECUTION_20261022.md) 为准。

已启用完整实验族的自动接续，目标是正式 pretrain 生产合同 PASS。GPU 数值与恢复资格仍为当前代码
11/11 PASS。节点外部任务结束后，LR pilot 已从 3 路扩至 8 路；两次调度进程交接均保留全部
已有训练 launcher PID 和 attempt，不重置正在运行的优化器或数据游标。

## 完整矩阵

| 阶段 | 实验数 | 配置训练预算 | 依赖 |
|---|---:|---:|---|
| LR pilot | 15 | 7.5B tokens | 当前数据和后端资格 |
| batch pilot | 9 | 约 17.97B | LR 选择 |
| E1 scaling | 24 | 约 453.21B | LR/batch 选型 |
| E2 architecture | 30 | 300B | 训练只依赖 LR/batch；分析另等 E1 loss fit |
| E3 mixture | 8 | 80B | 当前 E2 选择与 PASS E3 流；配置届时生成 |
| 独立性能补测 | 5 | 每档 32 updates | LR/batch 选定与空闲 GPU |

全矩阵共 91 组，独立性能短测之外的配置训练预算约 **858.67B tokens**。
这是接近 0.86T 的实际训练工作量，不能按短 smoke 的耗时估计整个排期。
该 token 总量包含各实验重复使用的训练前缀，不代表额外新增 0.86T 独立语料。

## 调度规则

1. 最多使用节点 8 卡，单个实验一张卡；空闲卡可以动态加入，无需中断现有训练。
2. E1/E2 都有任务时，八卡默认各分四卡；可用卡较少时分配对应份额，一族收尾后另一族可扩到八卡。
   已启动的训练优先保留，调度变化只控制后续任务。
3. 各组从 `data/health/gpu-allocations/<group>.json` 读取在线分配。
   GPU supervisor 持有独立 generation lease，训练再持有自己的 run lease，防止并发调度重复占用。
4. 空闲卡启动前检查 compute PID；只有原授权 GPU 0/6/7 允许走共享 loss lane。
   所有受控研究队列的吞吐均排除出正式选型，五个独立性能补测提供性能证据。
5. 每个任务结束后以 SIGUSR1 唤醒已校验身份的 watcher，即时更新分析、资源分配和下一阶段。
   无事件时仍按阶段切换 15 分钟、稳定长任务 2 小时检查，减少轮询开销。
6. 训练失败有界退避重试；数据、代码和结果身份门禁保持强制。性能补测发现争用时只终止、归档本项目短测。

## 前往 pretrain 的剩余门禁

LR/batch → 并行 E1/E2 与独立性能补测 → E2 分析 → E3 → 91 组身份验收 → 冻结配方 →
约 1T 三段数据物化、tokenization、overlap 审计，与选定模型/batch 的持续四卡吞吐 canary →
生产 validator PASS。

E1 的性能/拟合质量、E2/E3 的统计/域回归门禁失败时会保留阻断，不能仅以“跑完”晋级。
本轮调度扩容不修改模型架构、学习率、tokens/update 或单组训练预算。

机器可读执行范围见 `data/health/pretrain-full-pipeline-20261022.json`；
实时进度见 `data/health/pretrain-readiness-latest.json` 与各组 `*-queue.json`。

## 本轮验证

完整快速回归 273 passed / 2 skipped；最终调度专项 55 passed。专项包含独立 GPU/run lease 的
真实子进程验证、三卡按 2+1 分配、在线扩容与错误分配拒绝、争用中断归档及 watcher 身份核验。
11 项 GPU 资格仍与当前训练代码匹配。八个训练 launcher 在调度进程交接前后 PID 均保持一致。
