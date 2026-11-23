# 半日预训练运行检查

时间：2026-11-23T12:51:56.040119+00:00，同attempt c5bc01ad619700958f8fdac0967605f2/sourceAF54。正式训练与后台控制器运行中。

最新64800步/271,790,899,200labels，phase stable，train loss2.29423，TPS48624每卡，CUDA峰值67.46GiB。最新原生游标映射到stable.tokens.bin，数据reader和冻结配方未改。

259份250..64750原生结构/完整控制状态验收独立重读：六计数、164Adam计数/LR、8rank RNG一致，所有source pins及最新metadata SHA通过。实际6482个LR标量匹配；本项不是重复全浮点finite扫描，首250全数值扫描及归档另有证据。

| 域 | 60000 | 62000 | 64000 |
|---|---:|---:|---:|
| academic | 2.325037 | 2.324862 | 2.323309 |
| books | 2.621027 | 2.613040 | 2.610649 |
| chinese | 2.917930 | 2.916584 | 2.915952 |
| code | 1.108368 | 1.105272 | 1.110358 |
| encyclopedia | 2.364923 | 2.358422 | 2.358484 |
| math | 1.869418 | 1.866692 | 1.871518 |
| multilingual | 2.556772 | 2.535598 | 2.557965 |
| qa | 2.317971 | 2.314280 | 2.316814 |
| web | 2.691922 | 2.690984 | 2.686236 |
| aggregate | 2.308152 | 2.302859 | 2.305698 |

固定heldout事件prefix逐一SHA及TensorBoard重读通过。最新相对首列aggregate变化-0.11%；域回退观测：{"validation_metrics/domain/code/loss": 0.17952900417330664, "validation_metrics/domain/math/loss": 0.11231479107300935, "validation_metrics/domain/multilingual/loss": 0.04669020355232423}。这是早期点估计进展，不代表最终能力。

八卡util范围99–100%，温度60–74°C，ECC0，memory.failcnt历史4未增。全链路966254395392labels包含4K WSD/16K/64K，90%duty预测总工期31.940天，剩余23.838天，预计2026-12-17T08:57:58.586868+00:00。长阶段等待真实完整4K。

机器证据 `data/health/assistant-half-day-pretrain-review-20261123T1246.json`，SHA256 `97153ff0415437cd95f34e3b2d879f0796e1d97ae7d56da29ef4806d5c04edaa`。下一检查2026-11-24T00:46:12.619001+00:00，期间无对话轮询。
