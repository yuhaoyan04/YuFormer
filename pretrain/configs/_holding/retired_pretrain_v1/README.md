# Retired pretrain_v1

该目录只保存 2026-10-13 的 1.44T 单阶段历史合同，用于事故复盘和配置迁移测试。它不是正式
训练入口；YAML 的 `experiment_status: superseded` 会被 TorchTitan contract validator 和配置
registry 拒绝。当前唯一生产候选是 `pretrain/configs/pretrain_base.yaml` 的 1T
50B/800B/150B WSD 合同。
