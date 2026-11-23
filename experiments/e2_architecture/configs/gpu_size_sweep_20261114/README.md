# 八卡模型规模吞吐筛选

用户要求：在数据预处理继续运行时，实测约 1B–5B 模型是否能提高八卡利用率，同时保持约 30 天完成 1T 的工期。

入口为 `experiments/e2_architecture/scripts/benchmark_model_sizes.py`。先用正式 TorchTitan 环境运行 `--prepare-only`，核对候选参数与生成的独立配置，再运行同一脚本启动扫描。当前生产模型、冻结配方、tokenizer 和数据生产代码均不修改。

所有候选沿用 BF16、AdamW、4K、八卡 HSDP 2×4、cuDNN SDPA、模型编译、相同数据和每步 4,194,304 tokens。每个候选运行 24 次优化器更新，排除前 8 步，使用剩余 16 个逐步吞吐点的 p10 和中位数；1T 工期按八卡合计吞吐及 90% 有效训练时间计算。结束时保存 checkpoint 并执行九域验证；训练指标在最终保存和验证前记录。该短跑仅用于性能筛选，不能证明模型质量或替代正式训练资格。

每个候选优先 microbatch 8 / accumulation 16；仅在 CUDA OOM 时尝试 4/32、2/64，保持全局批量和 token 预算。配置、模型、代码和结果均绑定哈希，输出在独立 `data/experiments/gpu_size_sweep_20261114`，健康状态为 `data/health/model-size-sweep.json`。对比应同时报告相对并行数据处理条件下的 0.530B 基线降速、30/33 天工期、显存及数值状态。

胜出候选还须通过持续吞吐 canary、学习率与数值检查、真实恢复测试及正式模型身份重冻结；当前扫描不得直接签发 production PASS。
