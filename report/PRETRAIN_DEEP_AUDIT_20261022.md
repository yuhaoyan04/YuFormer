# Pretrain 实现审计与修复（2026-10-22 UTC）

本轮检查以真实 TorchTitan v0.3.0 / PyTorch 2.14 cu126 执行为依据。数据 release 与 tokenizer 不变。
正式目标仍为约 1T：50B warmup、800B stable、150B anneal；长度扩展在独立 midtrain 阶段执行。
不能依据历史有限 loss 或旧的静态 PASS 宣布训练栈已经合格。

## 已确认的缺陷与处理

| 领域 | 证据与影响 | 修复 |
|---|---|---|
| 参数初始化 | meta → to_empty 后 RMSNorm 权重没有重置；将其预填 NaN 可稳定重现非有限输出。历史 loss 即使有限也不能证明初始化正确 | 初始化全部 RMSNorm 为 1；共享参数按 identity 去重，避免 tied embedding/head 重复初始化；meta 对拍先注入 NaN |
| FSDP 梯度 | loss 已除以 global valid tokens，FSDP 默认再除 DP world，导致多卡梯度缩小 | 调用上游 `disable_fsdp_gradient_division`；真实两卡分别覆盖 tied/untied、GA=2 和不均匀 loss mask |
| 优化器 | 缺少更新前非有限梯度阻断；norm 使用普通 weight decay | 所有 rank 合并有限性检查后再更新；norm WD=0，其余 AdamW，显式 betas/epsilon |
| Checkpoint 状态 | 上游 trainer 默认状态不足以证明 Python/numpy/CPU/CUDA RNG 全部恢复 | 扩展 trainer state，DCP 保存逐 rank RNG、全局 token 数、样本游标；同拓扑按 rank 恢复 |
| 恢复守卫 | 缺少 runtime RESUME_ITER、scheduler 和数据游标的交叉验证 | 首次训练 batch 前写逐 rank resume receipt，拒绝步数/游标不一致；最终 accounting 与 attempt 绑定 |
| 假完成 | 上游 loader 耗尽可能正常结束；从最终 DCP 恢复会跳过最终验证 | 结束时必须达到目标 step 和样本数；最终 checkpoint 恢复时补做九域验证 |
| 日志恢复 | resume 归档全部 event 后，最后 checkpoint 恢复可能失去训练指标；简单合并又会混入被回滚的指标 | 记录 attempt ancestry 和恢复步，只继承 checkpoint 以内的祖先 event；最终完成要求实际 token/sample accounting |
| 验证指标 | eval 使用相同 metrics processor，重置未记录的训练 token/计时 | 保存、隔离并恢复训练计数；排除验证墙钟时间 |
| MFU/FLOPs | H800 被上游 fallback 当作 A100；SWA 的 FLOPs 与全注意力混计 | 硬件 lock 声明 H800 SXM BF16 dense 989 TFLOPS；dense projection 包含输出 head 一次，attention 按每层有效窗口计数 |
| Scaling 参数数 | 多个分析脚本一致漏计每层一个 RMSNorm；关闭 QK norm 仍计其参数 | 修正四处分析/冻结公式，使用 meta 模型实际参数数对拍 |
| 保存策略 | YAML keep_every 未接入；旧生产保存间隔 10000 steps 太长 | 保留最近 N 和长期 milestone；异步保存 N=0 或 N≥2；生产每 250 steps 保存 |
| 资格测试入口 | report 参数名与证据字段冲突；injection_mode 重复关键字；旧 source 归档误移动 runner lease | 修复入口与发布，保留 lease 控制文件；增加模拟完整故障/reshard 流程测试 |
| 故障终止 | killpg 最外层进程组无法覆盖 lease/torchrun 的独立 session | 按 /proc PPID 树终止所有自有后代，并复核 PID starttime；真实跨 session 子进程测试 |
| NFS 归档 | 本机禁止部分目录 rename；失效结果归档可能直接抛 EPERM | 优先原子 rename；受限目录逐文件归档；保留证据 |
| 队列与门禁 | 同组并发 writer、GPU 分配竞态、旧 runner 命令匹配遗漏、CUDA graphs/未支持 sharding 静默接受 | 单组 generation lease；跨动作 GPU reservation；相对/绝对命令匹配；未资格化选项 fail-fast |
| 生产入口与代码冻结 | 正式入口未要求当前完整资格报告；code-freeze candidate 漏掉 model/src | 生产 validator 强制资格身份匹配；禁止 EXTRA_ARGS 覆盖；冻结包含模型源码与测试 |

## 运行证据

本轮最终 11 项资格全部 PASS（报告更新时间 2026-10-22T08:29:03.867048+00:00）。
最终资格报告以 `data/health/torchtitan-qualification.json` 为准，必须 `status=PASS` 且
`qualification_identity.sha256` 与当前代码一致。它包括 LR 边界、单卡数值、两卡梯度、
单卡训练、四卡异步 DCP、未中断参考、训练/保存/加载三处 SIGKILL，以及 4→2/8 reshard。

两卡 FP32 梯度对拍首轮：tied 最大绝对误差 1.86e-8、relative L2 1.72e-7；
untied 最大绝对误差 1.30e-8、relative L2 1.62e-7。最终值见
`data/health/torchtitan-distributed-gradient-parity.json`。
另外执行反向对照：只移除关闭梯度除法的 helper，两种 embedding 情况的梯度相对误差
均为 0.500000，测试按预期 FAIL，确认该门禁能抓住原缺陷。

额外的大模型路径短测在 `data/experiments/pretrain_audit_20261022/runtime_800m_fullac/`：
800M preset、4K、BF16、FullAC，完成优化器更新、DCP 和最终 checkpoint 恢复，
峰值 reserved 约 16.9 GiB。它是正确性短测，不能用于吞吐或收敛结论。

测试日志位于 `report/logs/pretrain_deep_audit_*`。旧环境完整回归首次为
276 passed / 2 skipped / 1 failed；失败来自新增资格门禁的测试夹具，已补齐。
修复并补齐性能证据链后的快速完整回归为 268 passed / 2 skipped；新增 guard 定向回归 76 passed；
正式环境 runtime adapter 单测 6 passed。全部 78 份正式合同静态校验 PASS。
日志分别为 `pretrain_deep_audit_final_regression.log`、`pretrain_deep_audit_additional_guards.log`、
`pretrain_deep_runtime_regression.log` 和 `pretrain_deep_formal_contracts.log`。

## 历史结果失效

`data/health/pretrain-invalidation-20261022.json` 列出归档的旧训练目录。
旧 checkpoint、loss、LR 选择、memory probe 不能作为当前正式训练证据。
原始数据、治理 release、tokenizer、E3 已物化流不在失效范围内。
归档只移动本项目证据，不终止外部用户训练。

## 验收边界与剩余工作

1. 数值/故障资格可以在共享剩余资源上运行；其吞吐/MFU 不具有独占性能意义。
2. 共享 loss 与吞吐证据分开验收。E1 在五档独立性能短测全部通过前维持 `PASS_PROVISIONAL`；
   补测模型、精度和 tokens/update 与已选 E1 合同一致，绑定当前代码、配置、attempt、完整资格。
   E2/E3 的 loss 选择在当前资格与对应组完整身份审计通过后可 PASS，原始共享吞吐仍排除。
   最终身份审计现覆盖 91 组（原 86 组 + 5 个性能补测），配方冻结仍需 E1 的性能可行性与后续持续四卡 canary。
3. 正式 1T 开跑还需：E1/E2/E3 结论、当前实验身份审计、冻结配方、按最终 tokenizer 校准的
   三段物化和精确监督 token overlap 审计、选定模型/batch 的持续四卡吞吐 canary，以及生产 validator PASS。
4. 4→2/8 测试验证 DCP 可加载及样本连续性。测试中 global batch 随拓扑改变，
   不声称跨拓扑 bitwise 相同，也不将它当作生产无损改卡数的训练协议。
5. 通过短测不能保证数周训练绝无故障；后续异常必须由 checkpoint、有限性守卫、
   有界重试与可复核的日志处理。禁止删除门禁或使用旧 checkpoint 绕过已确认的数值缺陷。

## 自动推进状态

11 项资格 PASS 后已恢复 watcher（阶段切换 900 秒，稳定长任务 7200 秒）与
GPU 0/6/7 的三路 LR pilot。队列从新 checkpoint 开始，按有界退避重试。
机器可读汇总为 `data/health/pretrain-deep-audit-20261022.json`；
即时状态以 `data/health/pretrain-readiness-latest.json` 和 `e1_pilot-queue.json` 为准。

## 独立性能补测与证据晋级

`prepare_performance_canaries.py` 从选定 E1 合同生成五档、各 32 updates 的短测；
它们并行于 E1，由 watcher 自动调度到空闲卡（有一张即可），不占用共享 loss lane。
队列要求点火时 GPU compute PID 集为空，运行时每 5 秒检查，出现额外 PID 或查询失败便
终止、归档本项目短测，退避重试；外部进程不被终止。调度器重启或短测重试时从头测量，
禁止继承未监控区间。该检查是采样争用检测，不等价于集群调度器的硬件独占租约。

`performance_evidence.py` 校验五档补测的配置、身份、监控、稳态点数及当前资格。
E1 使用独立补测拟合 throughput，保留共享 loss 原始标签；冻结时重验补测 artifact 快照。
E2/E3 的 loss 资格晋级不依赖全局审计报告的可变文件 hash，避免其他组完成后改写既有 E3 数据合同。
补测链路已完成 CPU 集成回归（含真实自有子进程被争用检测中断并归档）；GPU 性能实测仍等待 LR/batch 选定与空闲卡。
