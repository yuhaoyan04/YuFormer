# YuFormer RLVR

正式 RL 主线是锁定版本的 **verl v0.9.1 + SGLang v0.5.20**，先做数学 exact-answer
GRPO/DAPO，再扩展到
隔离执行的代码测试奖励。版本见 [`backend.lock.json`](backend.lock.json)，来源与升级边界见
[`docs/SOURCE_PROVENANCE.md`](docs/SOURCE_PROVENANCE.md)。当前目录已建立 rollout 审计契约，
尚未把 YuFormer 模型接入 verl，因此不能声明 RL 训练闭环完成。

RL 使用独立的 PyTorch 2.13/CUDA 12.6 环境以兼容本节点 R535；不得把预训练的
TorchTitan/PyTorch 2.14 环境原地改装为 verl 环境。严格 DAPO 复现另用 recipe 锁定的旧
verl commit，结果必须分开命名，并在启用前独立完成 GPU qualification。

每个 rollout JSONL 行至少包含 `prompt_id`、全局唯一 `rollout_id`、有限数值 `reward` 和
`response_tokens`；可选记录 `truncated`、`kl`、`importance_ratio`、`clip_low`、`clip_high`。
优化器更新前必须验证每题恰有 G 个 rollout，并剔除组内 reward 零方差的题目：

```bash
PYTHONPATH=rl/src python rl/scripts/audit_rollouts.py \
  --input data/rl/attempt/rollouts.jsonl \
  --out data/rl/attempt/rollouts.audit.json \
  --group-size 8 --max-response-tokens 8192
```

审计产物报告全对/全错与零方差组比例、reward/长度分布、截断率、KL、重要性比率和上下裁剪率。
训练还必须满足：训练 verifier 与 held-out eval verifier 隔离；verifier 有对抗与变形测试；
prompt、reference answer、grader 和执行镜像全部内容寻址；定期人工复核 reward hacking；报告
held-out pass@1/pass@k、长度归一化结果和 base/SFT 能力回归。
