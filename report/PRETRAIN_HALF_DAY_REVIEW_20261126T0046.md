# 半日预训练运行检查

时间：2026-11-26T00:46:28.827005+00:00，同attempt c5bc01ad619700958f8fdac0967605f2/sourceAF54。正式训练与后台控制器运行中。

最新84760步/355,509,207,040labels，phase stable，train loss2.28012，TPS47894每卡，CUDA峰值67.46GiB。最新原生游标映射到stable.tokens.bin，数据reader和冻结配方未改。

339份250..84750原生结构/完整控制状态验收独立重读：六计数、164Adam计数/LR、8rank RNG一致，所有source pins及最新metadata SHA通过。实际8478个LR标量匹配；本项不是重复全浮点finite扫描，首250全数值扫描及归档另有证据。

| 域 | 80000 | 82000 | 84000 |
|---|---:|---:|---:|
| academic | 2.304487 | 2.303025 | 2.299024 |
| books | 2.586743 | 2.588208 | 2.588283 |
| chinese | 2.874503 | 2.875205 | 2.869481 |
| code | 1.089023 | 1.087913 | 1.092968 |
| encyclopedia | 2.331493 | 2.332376 | 2.324644 |
| math | 1.844474 | 1.845373 | 1.841763 |
| multilingual | 2.524358 | 2.527535 | 2.497805 |
| qa | 2.290869 | 2.289948 | 2.287726 |
| web | 2.660680 | 2.657028 | 2.656810 |
| aggregate | 2.278514 | 2.278512 | 2.273167 |

固定heldout事件prefix逐一SHA及TensorBoard重读通过。最新相对首列aggregate变化-0.23%；域回退观测：{"validation_metrics/domain/books/loss": 0.0595506593523476, "validation_metrics/domain/code/loss": 0.3622287703848759}。这是早期点估计进展，不代表最终能力。

八卡util范围100–100%，温度60–75°C，ECC0，memory.failcnt历史4未增。全链路966254395392labels包含4K WSD/16K/64K，90%duty预测总工期31.687天，剩余21.088天，预计2026-12-17T02:52:55.187946+00:00。长阶段等待真实完整4K。

机器证据 `data/health/assistant-half-day-pretrain-review-20261126T0046.json`，SHA256 `2df09b9b44d38c4b9bfeaee2bac0c5294664efd9f4d87bad72d768e36ecad871`。下一检查2026-11-26T12:46:12.619001+00:00，期间无对话轮询。
