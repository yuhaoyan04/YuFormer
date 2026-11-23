# E4 课程学习

**研究问题**：课程式 vs 全局打乱 / 长上下文渐进外推
**设计权威**：[`../../docs/research/实验设计.md E4 节`](../../docs/research/实验设计.md)（问题/假设/矩阵/图表编号以此为准）

## 结构

- `configs/` — 一臂一配置（自包含：模型/数据/优化器/预算/种子）
- `scripts/` — launcher（配置→CLI）+ 收集/拟合/画图
- `results/` — 指标 JSON / 拟合参数 / 图表 PNG（大数据走外部存储）

## 臂清单（跑前填写配置名，跑后填结果链接）

| 臂 | 配置 | 状态 | 关键结果 |
|---|---|---|---|
| 1T WSD + 渐进 context | `configs/e4_1t_wsd_context.yaml` | `PASS_STATIC` | 50B/800B/150B；4K base + 16K/64K extra 50B |
| 全局打乱对照 | 待 proxy 数据配比实验后冻结 | PLANNED | 同数据、tokens、LR 面积 |
| 4K→64K 直达对照 | 待长文本 release 后冻结 | BLOCKED | 与渐进臂 matched token/data/LR |

静态审计：`python experiments/e4_curriculum/scripts/audit_1t_plan.py`。报告写入
`results/e4_1t_plan_static_audit.json`；只有三段最终 tokenizer stream 均完成闭合后，状态才能从 `PASS_STATIC` 升为正式 release PASS。

正式物化由 `tokenize_1t_orders.py` 统一维护 12 个可恢复 job：`warmup/stable/anneal`
三条训练流和九条 held-out validation 流。恢复时先核对输入 release、order、tokenizer、文件长度
和首尾哈希；随后 `audit_1t_token_streams.py` 独立读取全部字节，要求 binary/meta 两侧 roster
分别精确相等并验证全文 SHA、EOD、容量及血缘。全文审计失败会生成新的 retry generation，
只重建失败或缺失的流；多余产物移入 `_quarantine` 后重新审计。生产合同只接受当前审计证据。

启动 TB 级 tokenization 前会执行 fail-closed 存储门禁。估算使用 tokenizer 实际 token 宽度，
并覆盖全部未完成最终流以及最大未完成阶段在 external merge 时同时存在的 NPZ part、解包 raw
part 和 staging binary，再增加 10% 安全余量与 100 GiB 保留空间。证据写入
`data/health/pretrain-1t-tokenization.json` 的 `storage_preflight`；不足时状态为
`BLOCKED_STORAGE`，不会先生成大型部分产物。释放空间后可由同一可恢复队列重新进入门禁。

derived order release 在启动 tokenization 前要求 held-out 文件精确等于规范九域，并验证 base
S6 manifest 的发布哈希。warmup/stable/anneal 三个 manifest 必须闭合内部
`upstream_snapshot_sha256`，且与 base release 完全一致；新生成阶段若仍发生 snapshot 漂移会在
第一次物化后立即停止。训练流另外发布 position 序逐文档 token 长度，以最终 tokenizer 的实际
消费区间审计跨阶段 replay。

`frozen_1t_recipe.json` 同时记录冻结所用 E1/E2/E3、LR resolution、token stream calibration、
experiment identity audit 以及所有生成配置的内容哈希。冻结入口会验证 identity audit 的逐文件
证据快照，production contract 和最终 stack audit 会再次验证 frozen inputs/outputs，避免历史
PASS 报告与后来被替换的实验结果或正式配置混用。

所有正式派生证据同时绑定生成工具源码：freeze、post-freeze plan audit、derived release、
token-stream audit（含共享全文审计器）、overlap audit（含两个 C merge 程序）、production canary
和 production contract。watcher 的 retry generation 也包含这些工具哈希；统计或审计逻辑变化时，
旧 PASS 会自动失效并重建。大文件首次审计使用全文 SHA，之后的低频 READY 检查使用已验证的
device/inode/size/mtime 身份，避免每轮重复读取 1T stream 相关账本。


E-131 update (~21:25 UTC): phase-specific training calibration is now part of the authorized coordinator before freezing. It reuses valid evidence, remeasures stale evidence, and revalidates the result before a production recipe can be published. A missing selected three-phase order release blocks bootstrap explicitly. Calibration/freezer/coordinator regression: 38 PASS; early capacity/ledger/mixture blocking regression: 10 PASS. Calibration measurements hash and parse each order/length ledger through the same open file descriptor; path rollback tests cover both. The complete original warmup ledger matches the independent part measurement, while its actual capacity fails the fixed production prefix. Warmup pilot ratios crosscheck to the complete stream with maximum domain-weight error 0.000586738246. Stable and anneal measurements remain pending; full pretrain remains unlaunched.
