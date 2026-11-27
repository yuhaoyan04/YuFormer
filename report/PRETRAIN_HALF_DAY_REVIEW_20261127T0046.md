# 半日预训练运行检查

时间：2026-11-27T00:46:29.572923+00:00，同attempt c5bc01ad619700958f8fdac0967605f2/sourceAF54。正式训练与后台控制器运行中。

最新92740步/388,979,752,960labels，phase stable，train loss2.26414，TPS48055每卡，CUDA峰值67.46GiB。最新原生游标映射到stable.tokens.bin，数据reader和冻结配方未改。

370份250..92500原生结构/完整控制状态验收独立重读：六计数、164Adam计数/LR、8rank RNG一致，所有source pins及最新metadata SHA通过。实际9276个LR标量匹配；本项不是重复全浮点finite扫描，首250全数值扫描及归档另有证据。

| 域 | 88000 | 90000 | 92000 |
|---|---:|---:|---:|
| academic | 2.297122 | 2.294762 | 2.292990 |
| books | 2.582395 | 2.579949 | 2.575122 |
| chinese | 2.866369 | 2.860457 | 2.860868 |
| code | 1.080573 | 1.087673 | 1.082257 |
| encyclopedia | 2.323006 | 2.322515 | 2.322015 |
| math | 1.837129 | 1.839507 | 1.836974 |
| multilingual | 2.479354 | 2.484279 | 2.479459 |
| qa | 2.283394 | 2.285208 | 2.282831 |
| web | 2.650606 | 2.650728 | 2.647005 |
| aggregate | 2.266661 | 2.267231 | 2.264391 |

固定heldout事件prefix逐一SHA及TensorBoard重读通过。最新相对首列aggregate变化-0.10%；域回退观测：{"validation_metrics/domain/code/loss": 0.15587173881770333, "validation_metrics/domain/multilingual/loss": 0.0042214927833184035}。这是早期点估计进展，不代表最终能力。

八卡util范围100–100%，温度60–73°C，ECC0，memory.failcnt历史4未增。全链路966254395392labels包含4K WSD/16K/64K，90%duty预测总工期31.568天，剩余19.969天，预计2026-12-17T00:01:45.174664+00:00。长阶段等待真实完整4K。

机器证据 `data/health/assistant-half-day-pretrain-review-20261127T0046.json`，SHA256 `bade74656697faa0800161ef66355a5e088f1f657f1aba62bc68368422f6406c`。下一检查2026-11-27T12:46:12.619001+00:00，期间无对话轮询。
