# YuFormer Long-Context Midtraining

正式阶梯是 4K pretrain → 16K → 64K。每个阶段使用前一阶段的 **weight-only** 初始化并重置
优化器；`tokens_per_update` 保持恒定，25% 短样本 replay 用于控制短能力回退。64K 路径必须
使用通过系统门禁的线性内存 attention，SDPA 的平方级显式 attention mask 不可进入正式运行。

当前两个对照臂均显式要求八卡。16K 使用 microbatch=2、grad_accum=16，64K 使用
microbatch=1、grad_accum=8，均保持 4,194,304 tokens/update；显存与 kernel 仍须独立实测。
按用户最新选择，16K/64K 两阶段约 50B tokens **计入总约 1T 与整套 33 天预算**，
不能再使用仅 4K/1T 的速度估算整套工期。预算真源为
`data/health/user-total-pretrain-and-length-budget.json`；全阶段实测总耗时仍须通过门禁。

1.52B 八卡系统预检已跑通 16K 和 64K 各 24 步，包含验证与最终 checkpoint。
原 cuDNN 测试保留为历史证据；当前候选使用确定性 Flash 全局 attention 与编译 Flex 滑窗。
该新后端也已完成各 24 步训练/验证/最终保存，系统 p10 约 40895/24934 token/s/卡；
证据为 `data/health/deterministic-flash-long-profiles.json`。64K 必须使用 MLP 选择性重计算、32 路 loss 分块及
`expandable_segments:True,garbage_collection_threshold:0.7` 分配器策略。整层重计算在当前
64K 测试中出现非有限梯度，未配置回收策略的 MLP 方案在最终 checkpoint 出现 NCCL/CUDA
错误；契约拒绝这些未通过的组合。系统结果见 `data/health/long64k-supported-config-gc-confirmation.json`。
这仅是运行路径和显存筛查，长能力与短能力回归仍须在锁定基础模型、长文本数据后单独验收。

`configs/ladder_4k_16k_64k.yaml` 目前按设计返回 `BLOCKED`，因为 base checkpoint hash 与长文本
PASS dataset id 仍是占位符：

```bash
PYTHONPATH=midtrain/src python midtrain/scripts/audit_ladder.py \
  --config midtrain/configs/ladder_4k_16k_64k.yaml \
  --out data/health/midtrain-ladder.json
```

`configs/direct_4k_64k.yaml` 是 matched-token 对照臂：从同一个 4K checkpoint 直接进入
64K，消费相同的约 50B tokens，并保持相同的 4,194,304 tokens/update、数据配比、短样本
replay 和 peak LR。它与渐进臂共同回答长度阶梯本身是否带来收益。
这是保留的研究对照配置，不在本轮全量阶梯主线或其 33 天预算内启动。

每阶段必须同时通过长上下文能力和短能力回归。正式评测至少覆盖 RULER、长文 BPB、长文 QA/
代码仓任务，以及与 4K base 相同的冻结 core suite。RoPE base、YaRN/DCA 属于 matched-token
消融变量；没有 RULER/长文 BPB 增益时不能仅凭 needle 测试宣称 64K 有效上下文。

`src/yuformer_midtrain/window_records.py` 和 `window_release.py` 已实现精确 S+1 源窗口、
固定 25% replay、全局数据游标及 source/heldout/mixture/overlap/durability 审计绑定；
`data.py` 与 `registry.py` 接入显式声明的共享 TorchTitan 后端。`scripts/launch.py` 已实现
租约保护、不可变运行身份、阶段 preflight、实际父权重核对与同阶段全状态恢复。

真实 CPU DCP 检查已证明 weight-only 初始化重置优化器/调度器/游标/RNG，而同阶段续训在保留
`initial_load_model_only=True` 参数时仍恢复全部状态，下一次更新与不中断一致；证据为
`data/health/midtrain-actual-cpu-stage-transition-20261115-v3/report.json`。
实际八卡 4K→16K→64K 阶段切换与两阶段 SIGKILL 恢复也已全部通过，逐步 loss 差为 0，
见 `data/health/midtrain-actual-eight-gpu-stage-qualification.json`。64K 使用独立的
rank-bound checkpoint stager，避免保存线程在 GPU 0 创建其他 rank 的 CUDA context；
来源及失败留档见 [SOURCE_PROVENANCE.md](docs/SOURCE_PROVENANCE.md)。

完整生产窗口的数据 admission 与长短能力验收仍是独立门禁，不能用上述 24 步系统资格替代。
生产预算、LR、warmup、保存和验证频率已经锁定于
`configs/production_total966b_stage_plan.yaml`。运行顺序和正式配置生成方式见
[PHASES.md](docs/PHASES.md)。历史 ladder 配置保留占位符，正式启动不修改已冻结的历史文件。
