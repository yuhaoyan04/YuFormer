# YuFormer 项目状态与研究审计

> 审计日期：2026-11-07。此文依据仓库文档、实验身份/结果产物、当前节点设备清单与路径实查。历史运行产物用于记录已完成的实验，不自动代表当前节点具备同一训练环境。

## 项目目标与当前有效范围

项目目标是基于 ZGCM-1 的工程经验，自行实现并训练 YuFormer，建立数据、tokenizer、模型、预训练、后训练与评测的可追溯闭环。当前先完成 E1 scaling 统计复验：保持 24 个设计点、架构、tokenizer、release、优化器和各点预算不变，补跑两个配对 seed。E2 架构和 E3 配比仍延期。seed-aware `L(N,D)` 拟合通过门禁并完成资源/吞吐审计后，才进入全量 1T 预训练。

## 进度盘点

| 环节 | 可确认状态 | 下一道门 |
|---|---|---|
| 数据治理 | `data/PRETRAIN_DATA_READY.json` 为 PASS；约 2.53T 清洗池、100B 训练/九域验证 release 有审计。 | E1 结果确定配方后，仍需独立生成并审计正式 1T 训练流。 |
| Tokenizer | `data/TOKENIZER_READY.json` 为 PASS；64K tokenizer 和九域 held-out 评估是当前实验基线。 | 冻结的 1T streams 必须重新绑定、校验 tokenizer 与 overlap。 |
| 模型 | 自研 dense Transformer/GQA/SWA 等配置已落地；`model/tests` 当前节点 35 PASS、1 skip。 | 规模结论确定后做匹配目标架构的四卡训练 canary。 |
| TorchTitan 运行时 | 已在当前节点重建锁定 userspace：Python 3.12.16、PyTorch 2.14.0+cu126、TorchTitan v0.3.0 锁定 commit，userspace validator 与 fresh preflight 均 PASS。单卡 BF16 数值对拍、双卡 FSDP2/chunked-loss gradient parity 均 PASS。 | 四卡 recovery/reshard qualification 尚未重跑；四张 H800 现已由用户明确授权用于 scaling 复验。 |
| LR / batch pilot | LR resolution 与 batch selection 产物均为 PASS；状态记录 21 个 LR、15 个 batch 结果。batch 采用 1M token/update 固定基线，不声称为全局最优。 | 结果输入/身份需在 E1 与正式配方冻结时重新校验。 |
| E1 scaling | 24/24 原始单 seed 结果齐备，身份/预算门禁通过，独立吞吐测量已记录。screen 拟合 `R²=0.981`、LOO 中位/95 分位误差 2.77%/8.15%，NRMSE 3.64% 超过 3% 门槛。 | 新增两个配对 seed 的 48 个复制任务已在 GPU 0–3 启动；每点总计 3 seeds、额外预算 84,899,181,190 tokens。门槛不变，完成后做 seed-aware 拟合与 bootstrap。 |
| 1T 正式预训练 | 尚未开始。E1 fit、配方冻结、生产 canary、1T 容量/数据流与 overlap 门禁均未完成。 | 按自动化 DAG：E1 PASS → 配方冻结 → 1T 计划/容量 → 四卡 canary → 1T 物化/审计 → production contract。 |
| Midtrain / SFT / RL / Eval | 阶段契约和部分数据/诊断模块已实现；对应 trainer/adapters、正式数据 revision 或评测 suite 尚未全部闭环。`midtrain` 3 项、`sft` 9 项、`rl` 3 项、`eval` 2 项测试通过。 | 等 base checkpoint 后分别完成框架接入、数据版本锁定与端到端验收。 |

## 本次确认的问题与修复

1. **GPU 白名单与当前硬件错位。** 旧策略声明 GPU `[0,1,2,4]`，但当前 `nvidia-smi` 报告 `[0,1,2,3]`。`pretrain/scripts/gpu_resource_policy.py` 过去只校验索引小于 8，没有核对实时设备库存；现已增加实时清单校验，缺失设备或无法读取清单时 fail-closed。用户随后明确授权全部四张当前可见 H800 仅用于 scaling，运行策略已更新为 `[0,1,2,3]`、四作业并行、禁止 GPU sharing；全量预训练仍禁止启动。
2. **本节点原先缺少锁定环境。** 已用项目脚本恢复 `/opt/yuformer-python` 与 `/opt/yuformer-torchtitan`，source commit 为 `086bf6c166ec85c1298eb5596fa9bf95f6a2d840`；`pip check`、userspace validator 和 fresh driver/CUDA preflight 都 PASS。单卡和双卡数值对拍通过。当前四卡 seed 复验正在真实 E1 配置上运行；恢复/重分片专项资格仍待复跑。
3. **E1 有结果但统计门禁不通过。** 24 点已齐备，拟合并非因漏跑或身份失败而阻断，而是 3.64% 的 normalized RMSE 超过 3%。最大的残差来自 `e1_iso_c1_800m`（观测 loss 5.438，拟合 4.999）；其次为 `e1_ld_300m_100m`。单 seed 设计无法将优化随机性和曲线失配分开，不能把现有 bootstrap CI 当作 seed 不确定性。
4. **健康快照过时。** `data/health/pretrain-readiness-latest.json` 是 11 月 6 日快照，仍引用已消失的 GPU 4，不能作为当前资源 PASS。当前 README、AGENTS 与本审计已更新；后续自动化应以实时 GPU 清单和新合同为准。

## 检查范围与验证结果

- 当前节点：4×H800（索引 0–3），535.161.08 驱动；系统 Python 为 3.10.12 / PyTorch 2.2.0a0，正式运行使用隔离 Python 3.12.16 / PyTorch 2.14.0+cu126。复验启动前四卡均空闲；随后四卡各启动一个单卡 seed 任务。
- `pretrain/tests`：全量运行 343 PASS、29 skip；一次人为中断信号打断了两条真实进程队列用例，随后两项单独重跑均 PASS。受资源策略和共享槽位修复影响的定向回归共 105 PASS。
- 锁定 TorchTitan userspace validator：PASS；当前节点 fresh preflight：PASS；TorchTitan 单卡 BF16 数值对拍：PASS；双卡分布式梯度对拍：PASS。
- `model/tests`：35 PASS、1 skip；`midtrain/tests`：3 PASS；`sft/tests`：9 PASS；`rl/tests`：3 PASS；`eval/tests`：2 PASS。
- `data-process/tests -m 'not real'`：208 PASS、13 skip；需外部下载的真实集成用例未执行。`tokenizer/tests`：12 PASS。`model/tests`：35 PASS、1 skip；`midtrain/tests`：3 PASS；`sft/tests`：9 PASS；`rl/tests`：3 PASS；`eval/tests`：2 PASS。
- 全仓一次性 pytest collection 不适合作为总验证入口：多目录收集时有同名 `test_materialize` 模块冲突；初次系统 Python collection 也缺少 `huggingface_hub`、`tokenizers`。改为各模块使用匹配环境分组运行后，除明确标记为 real 的外部下载集成测试外均通过。
- 当前节点锁定 preflight 已重跑并 PASS；完整四卡恢复/reshard qualification 未重跑，因当前白名单只含三张卡。

## 并行执行 DAG

1. **P0 已闭合：** 锁定 Python/TorchTitan 环境、preflight、实时 GPU 清单和 4 卡独占资源策略均已通过。
2. **P1 正在运行：** 48 个配对 seed 复验合同已冻结、108/108 正式合同校验通过；四卡队列按 FLOPs 优先并行运行。完成后自动审计结果身份与 token accounting，seed-aware analyzer 计算平均 loss、seed 区间、NRMSE、LOO 和 R²。
3. **P2 结果判定：** 只使用原门槛判断 scaling 是否可用，不下调门槛。若仍 FAIL，则根据 seed 方差诊断模型失配或预算不足，再预注册下一轮；若 PASS，则冻结规模推荐并运行当前设备上的四卡 recovery/reshard 与持续吞吐 canary。
4. **P3 正式 1T 数据：** CPU/IO worker 并行做最终 order、tokenization、独立 stream audit、数据容量及跨阶段 overlap 审计；每个发布绑定内容哈希和最终 recipe。E2/E3 仍按当前 scope 延期。
5. **P4 训练与后续阶段：** 只有 P0–P3 对应门禁都 PASS 后启动 1T；checkpoint、恢复、loss、吞吐和实际消耗 token 持续写入身份绑定的 run。base checkpoint 完成后再启动 midtrain/SFT/RL/eval 闭环。

当前最重要的结论是：项目已远超“pretrain 框架刚搭好”。原始 24 点 scaling 的质量门禁未通过，所以已按完整设计而不是挑点复跑，把每点扩展到三个配对 seed。四卡复验正在运行；在 seed-aware 结果和后续资格门禁通过前，不启动 1T 全量预训练。
