当前运行状态（2026-11-24T12:47:05.469873+00:00）：**正式八卡1.5166B pretrain半日检查PASS_HALF_DAY_OPERATIONAL_REVIEW。** 最新72780步/loss2.3427，累计305,261,445,120labels，phase=stable，TPS48506每卡，CUDA峰值67.46GiB。原生数据游标对应stable.tokens.bin。

291份250..72750原生结构/完整控制状态验收独立复核，7280个实际LR标量匹配，无数值/源/控制告警。最新72000固定九域aggregate2.287684441，相对68000变化-0.22%；域回退观测{}。事件prefix独立SHA及重读PASS。报告PRETRAIN_HALF_DAY_REVIEW_20261124T1246.md，机器证据assistant-half-day-pretrain-review-20261124T1246.json，SHAab8927c6f9bbc254b78c8d6bf8fade41f6cd2901acdd8e5ec0874c08f1e6d7a1。当前是早期heldout进展，不是最终能力完成。

八卡ECC0、显存读数详见本次机器证据，历史memory.failcnt4未增，同c5bc attempt/sourceAF54。总预算966254395392labels含4K WSD/16K/64K，90%duty保守总工期31.843天、剩余22.744天，预计2026-12-17T06:38:16.764258+00:00完成，目标33天。114冻结输入coordinator仍等待完整4K，长阶段未开始；严禁热改执行源或在active run调用finalcollector。按用户每12h汇总，下次2026-11-25T00:46:12.619001+00:00。

以下为历史时点记录；以上及实时health为准。

---

当前运行状态（2026-11-24T00:46:27.565310+00:00）：**正式八卡1.5166B pretrain半日检查PASS_HALF_DAY_OPERATIONAL_REVIEW。** 最新68780步/loss2.32859，累计288,484,229,120labels，phase=stable，TPS48621每卡，CUDA峰值67.46GiB。原生数据游标对应stable.tokens.bin。

275份250..68750原生结构/完整控制状态验收独立复核，6880个实际LR标量匹配，无数值/源/控制告警。最新68000固定九域aggregate2.292660475，相对64000变化-0.57%；域回退观测{}。事件prefix独立SHA及重读PASS。报告PRETRAIN_HALF_DAY_REVIEW_20261124T0046.md，机器证据assistant-half-day-pretrain-review-20261124T0046.json，SHA40b51f5b767a96286e4bdf7d6ecd49f0d5019b4c96667a5e7671f8f4f3737447。当前是早期heldout进展，不是最终能力完成。

八卡ECC0、显存读数详见本次机器证据，历史memory.failcnt4未增，同c5bc attempt/sourceAF54。总预算966254395392labels含4K WSD/16K/64K，90%duty保守总工期31.886天、剩余23.288天，预计2026-12-17T07:40:32.666476+00:00完成，目标33天。114冻结输入coordinator仍等待完整4K，长阶段未开始；严禁热改执行源或在active run调用finalcollector。按用户每12h汇总，下次2026-11-24T12:46:12.619001+00:00。

以下为历史时点记录；以上及实时health为准。

---

当前运行状态（2026-11-23T12:51:56.040119+00:00）：**正式八卡1.5166B pretrain半日检查HALF_DAY_REVIEW_DOMAIN_REGRESSION_REQUIRES_ATTENTION。** 最新64800步/loss2.29423，累计271,790,899,200labels，phase=stable，TPS48624每卡，CUDA峰值67.46GiB。原生数据游标对应stable.tokens.bin。

259份250..64750原生结构/完整控制状态验收独立复核，6482个实际LR标量匹配，无数值/源/控制告警。最新64000固定九域aggregate2.305698395，相对60000变化-0.11%；域回退观测{"validation_metrics/domain/code/loss": 0.17952900417330664, "validation_metrics/domain/math/loss": 0.11231479107300935, "validation_metrics/domain/multilingual/loss": 0.04669020355232423}。事件prefix独立SHA及重读PASS。报告PRETRAIN_HALF_DAY_REVIEW_20261123T1246.md，机器证据assistant-half-day-pretrain-review-20261123T1246.json，SHA97153ff0415437cd95f34e3b2d879f0796e1d97ae7d56da29ef4806d5c04edaa。当前是早期heldout进展，不是最终能力完成。

八卡ECC0、显存读数详见本次机器证据，历史memory.failcnt4未增，同c5bc attempt/sourceAF54。总预算966254395392labels含4K WSD/16K/64K，90%duty保守总工期31.940天、剩余23.838天，预计2026-12-17T08:57:58.586868+00:00完成，目标33天。114冻结输入coordinator仍等待完整4K，长阶段未开始；严禁热改执行源或在active run调用finalcollector。按用户每12h汇总，下次2026-11-24T00:46:12.619001+00:00。

以下为历史时点记录；以上及实时health为准。

---

# 当前 pretrain 门禁与状态

更新：2026-11-23T00:46:28.253923+00:00。正式八卡1.5166B预训练运行中，phase=stable，第60780步，254,929,797,120labels。当前复核状态：**HALF_DAY_REVIEW_DOMAIN_REGRESSION_REQUIRES_ATTENTION**。完整逐域数据与观察见 [最新机器快照](metrics/latest_pretrain_status.json) 和 [半日复核](PRETRAIN_HALF_DAY_REVIEW_20261123T0046.md)。

4K WSD已启动，16K/64K数据和实际八卡系统/恢复资格通过；未来生产长阶段及最终能力尚未完成。预算966254395392labels包括全部三阶段，预计总工期32.010天、剩余24.411天。Books短窗+0.120791%保留观察，不改配方或冻结门禁。

外围SIGKILL恢复、租约及实际PID见 [运行交接](ACTIVE_PRETRAIN_OPERATIONS.md)；当前主状态PID为guardian539905，不能用旧430805判断训练。仓库清理未动391份冻结依赖。完整工程/实验说明见 [技术报告](TECHNICAL_REPORT.md)，维护状态见 [维护记录](REPOSITORY_MAINTENANCE_20261123.md)。历史重复状态已并入外部 `report/history/operations-consolidation-20261123/`，本页不再堆叠旧准备状态。
