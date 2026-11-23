# YuFormer Evaluation

正式通用评测主线锁定 `lm-evaluation-harness` v0.4.13。该版本修复了 few-shot 样本把当前
评测题泄漏进 demonstrations 的问题，因此旧版本分数不与新结果混报。长上下文的 RULER/
LongBench、代码执行的 EvalPlus 与训练期逐域 BPB 使用各自隔离环境，但必须输出同一结果契约。

任何可发布分数都必须保存逐样本输出，并通过 `yuformer_eval.audit`：锁定 harness commit、
checkpoint/tokenizer/chat-template 内容哈希、随机种子、完整数据集运行、任务配置和 prompt 哈希、
不可变 dataset revision、样本数、指标方向以及逐任务去污染报告。`limit` 运行只作 smoke，不能
进入论文表格或 scaling 拟合。

```bash
PYTHONPATH=eval/src python eval/scripts/audit_results.py \
  --manifest data/eval/run/result-manifest.json \
  --root data/eval/run \
  --out data/eval/run/audit.json
```

发布表格同时报告 accuracy/EM/pass@k 等主指标、标准误或 paired bootstrap CI、tokens/s、峰值
显存和失败率；base、midtrain、SFT、RL checkpoint 使用同一冻结 suite。训练 verifier、开发集
与 held-out evaluator 必须隔离，污染命中样本既给原始分数也给剔除后的敏感性结果。

上游：<https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.13>

本次 1.5166B 生产训练的阶段评测另见 [CONTEXT_STAGE_PIPELINE.md](docs/CONTEXT_STAGE_PIPELINE.md)。
其四组固定诊断案例、实际 native DCP 测量、八卡分片重建与量化准入服务于 4K→16K→64K
阶段切换，不能作为完整公开 benchmark 或无污染评测的发布分数。完整案例和规则已在
最终模型能力测量前冻结；当前全量 4K 正在运行，控制器保持等待，长能力尚未签发 PASS。
