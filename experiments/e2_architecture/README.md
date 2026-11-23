# E2 架构消融

E2 在 E1 选定的模型规模附近比较深度/宽度、FFN 比例、QK-Norm、partial RoPE、gated SWA、
全注意力与 tied embedding。当前等待 LR/batch pilot 裁定；实验数据流、tokenizer 和当前
TorchTitan 12 项 GPU 资格均已 PASS。E2 训练与 E1 训练并行，E2 分析另等 E1 拟合。

`experiments/_common/gen_e1e2_arms.py` 生成 10 个唯一架构、每个 3 seeds，共 30 个 formal
configs。共同的 2.75× FFN / 32 层基线跨问题复用，避免同 seed 重复训练。每个臂必须使用相同的
随机数据顺序、token budget、AdamW/cosine 定义、评测集和 backend commit；主表同时报告
参数量与实际 FLOPs，不能把不同 FLOPs 的 raw loss 直接解释为架构收益。所有正式架构均运行
3 seeds；报告先聚合均值/标准差，再用 E1 scaling law 折算到共同 non-embedding 参数量。最终
选择使用 paired-seed 95% CI，并在与经验最优无法区分的架构中选择估算 FLOPs 最低者。

当前每臂初筛预算为 1B tokens。入选架构与 `e2_l32` 基线另从头进行 3B-token、三个
seed 的复验；若入选即基线则去重为三组。复验检查参数归一后的 aggregate loss 与九域
非劣性，失败保持阻断。复验结果及其输入 SHA-256 必须绑定到最终配方，不能只凭短跑排名冻结。

旧 `e2_l16_lf` 同时改变 micro-batch、gradient accumulation、recompute 与优化器，无法归因于
架构，已移入 `pretrain/configs/_holding/legacy_e2/`，只保留为性能历史记录。正式环境文件已
绑定 2026-10-19 候选 snapshot；加载器仍要求 release PASS，因而不会误用旧 E0 token stream。
