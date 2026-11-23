# 当前 pretrain 门禁与状态

更新：2026-11-23T00:46:28.253923+00:00。正式八卡1.5166B预训练运行中，phase=stable，第60780步，254,929,797,120labels。当前复核状态：**HALF_DAY_REVIEW_DOMAIN_REGRESSION_REQUIRES_ATTENTION**。完整逐域数据与观察见 [最新机器快照](metrics/latest_pretrain_status.json) 和 [半日复核](PRETRAIN_HALF_DAY_REVIEW_20261123T0046.md)。

4K WSD已启动，16K/64K数据和实际八卡系统/恢复资格通过；未来生产长阶段及最终能力尚未完成。预算966254395392labels包括全部三阶段，预计总工期32.010天、剩余24.411天。Books短窗+0.120791%保留观察，不改配方或冻结门禁。

外围SIGKILL恢复、租约及实际PID见 [运行交接](ACTIVE_PRETRAIN_OPERATIONS.md)；当前主状态PID为guardian539905，不能用旧430805判断训练。仓库清理未动391份冻结依赖。完整工程/实验说明见 [技术报告](TECHNICAL_REPORT.md)，维护状态见 [维护记录](REPOSITORY_MAINTENANCE_20261123.md)。历史重复状态已并入外部 `report/history/operations-consolidation-20261123/`，本页不再堆叠旧准备状态。
