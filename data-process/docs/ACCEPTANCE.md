# 工业级 pretrain 数据集验收清单（ACCEPTANCE）

> 数据集交付（S6 物化完成）时的验收标准。每项必须逐条打勾并留产物路径，
> 全部通过才算"彻底拿到经过完整 pipeline 的高质量 pretrain 数据集"。
> 配套工具：`scripts/audit_consistency.py`、`scripts/quality_spotcheck.py`、
> `scripts/release_gate.py`。清单共 24 项；实际状态以 release manifest 中的机器断言或带 reviewer
> 的人工签收为准，Markdown 勾选本身不构成放行。
>
> **2026-10-19 状态**：固定版本完整性复审发现 arxiv-latex-corpus 的主文件被截断、
> skypile-150b 缺一个分片，两者已从本轮正式范围排除；15 个纳入源的远端 roster、S0
> 文件身份和许可证据机器门 G1 已通过。旧 S6 commit marker 已撤销，100B source-scoped
> S6 与独立机器审计已重建通过；E2 人审已于 2026-10-19 由用户逐样本验收通过。
> A1–G2 已全部通过；正式 release 为 `yuformer-data-86f5860c47e6162e6ba0`。100B train、
> 九域 validation token stream 与 6.5B tokenizer corpus 的独立审计也已通过；最终
> pretrain 数据入口状态以 `data/PRETRAIN_DATA_READY.json` 为准。

## A. 守恒与完整性（硬门）

- [x] **A1 跨阶段行数守恒**：`audit_consistency_20261019.json` PASS；15 个纳入源的分片与成功 parquet footer 全部对拍
- [x] **A2 s0→s1 守恒**：15 个纳入源的行数和 artifact error 均闭合
- [x] **A3 全链 shard_errors=0**：当前发布所依赖的 S5/S6 与各阶段 summary 无错误
- [x] **A4 重刷源联动完成**：stack-exchange 与 dolma-books 的当前 lineage 已贯通全文 decontam

## B. 去重与去污染（硬门）

- [x] **B1 ndv2 web/code 决定已发布**：`s3_decisions/<domain>.summary.json` 为 direct-edge policy；全文精度审计与当前边集同代；排序 lookup 字节数、逐域 dropped 和 `stage_ndv2_global_summary` 闭合。S5 直接消费 lookup，无需重写 10TB S3
- [x] **B2 去重账对账**：S5 exact/near/decontam/keep 顺序账闭合；ndv2 raw lookup IDs/codes、决定 summary 与全局 dropped 一致，exact-overlap 解释 raw/effective 差额
- [x] **B3 decontam 全覆盖**：12,829/12,829 分片按全文策略完成，`truncated_docs=0`；相对旧版新增删除 12,363 篇

## C. 质量分数轨（二选一后核验）

- [x] **C1 score_field 决策记录**：候选轨必须有独立 holdout/CV 或显式裁定；当前生产裁定为 `hybrid_v2`（goldrule v2 holdout AUC 0.899），决策依据写入 S6 manifest
- [x] **C2 分数分布健康**：全量扫描各源/语言 `hybrid_v2` 与域内 rank 分位（p1/p25/p50/p75/p99）——12,829 分片、6,485,760,234 行机器审计通过
- [x] **C3 分位表覆盖**：`hybrid_v2` 表覆盖 mix 全部九域，逐域 mmap 行数与 S5 输入一致
- [x] **C4 goldrule 研究子集**：当前产物 16,406,403 行，schema/SHA-256 已验证，`rule_score` / `llm_distill` / `hybrid_v2` 均 100% finite

## D. 采样与课程（S6 产物）

- [x] **D1 配额守恒**：deep-fill 后 Σ selected_tokens = 99,999,708,923；总 shortfall 291,077，小于声明的 1M 不可切分文档尾差上限
- [x] **D2 顺序语义与守恒**：正式 baseline 为单一 doc_id hash 确定性全局打乱；order tokens == selected
- [x] **D3 val 互斥**：95,220,247 train IDs 与 324,092 unique val IDs 精确对拍，交集为 0；九域 val 配额满足
- [x] **D4 mix_members/order 表完整**：两侧各 95,220,247 行、各自无重复，256 路外排多重集完全相等

## E. 抽检与审计（软门，报告论据）

- [x] **E1 分域质量抽检**：当前 members 哈希绑定的 3,200 篇九域审计 PASS；正文/分数 3,200/3,200 对齐，编码、HTML、语言和正式阈值下 benchmark overlap 全绿。60 个 CCI3 单 C-Eval hit 低于逐源 distinct-hit=2 门槛并单列留痕
- [x] **E2 人审样本**：当前正式 mix 的九域 p25/p50/p75 共 27 篇已生成于 `data/processed/audits/human_review_mix_m3_e1_100b_20261019.{json,md}`；27/27 重编码一致、机器门通过，并由 `user-decision-20261019` 全部签收为 PASS
- [x] **E3 长度分桶分布**：文档/token 桶账精确闭合；B4K 文档 97.8130%，长上下文池 633.96B tokens，研究供给门通过
- [x] **E4 tokenizer 口径一致**：九域 p25/p50/p75 共 27 篇逐篇重编码 27/27 一致；真实 `materialize_tokens.py` 输出 442,795 tokens（含 27 EOD）并与独立期望逐 token 相等。证书：`data/processed/audits/e4_tokenizer_crosscheck_mix_m3_e1_100b_20261019.json`

## F. 可复现性（宪法 4）

- [x] **F1 幂等复验**：第二次 S6 invocation 在 409 秒后命中 content-addressed skip；members/order/manifest/global summary/C4 五类产物前后 SHA-256 完全一致
- [x] **F2 决策可追溯**：S6 manifest 记录级联 upstream snapshot、thresholds/quotas/selected stats，且发布前后 snapshot SHA-256 一致
- [x] **F3 审计报告归档**：失败账本、验收清单与机器证书已随项目归档

## G. 来源治理与删除请求（硬门）

- [x] **G1 不可变来源与许可证据**：15 个纳入源均绑定固定 revision、逐文件 provenance 和远端 expected-vs-S0 observed 完整性；arXiv/SkyPile 的不完整事实与排除理由已进入 release scope；不将数据集层 license 自动解读为底层每篇文档的权利授予
- [x] **G2 用途/PII/opt-out 签收**：`user-decision-20261019-g2` 接受仅限内部非生产研究的五项控制；逐源 training use、redistribution、PII 处置与 opt-out/delete 记录由机器审计确认 PASS

---

## 交付物清单（数据集 = 以下全部）

| 交付物 | 路径 | 消费方 |
|---|---|---|
| mix 成员表 | `mix_members/<mix>.parquet` | 审计/追溯 |
| 训练顺序表 | baseline 为 `mix_order/<mix>_stage2.parquet`；显式课程消融可增加 stage1 | pretrain materialize_tokens |
| per-domain val | `val_sets/<domain>.*` | pretrain/E1 held-out |
| goldrule 研究子集 | `goldrule_research/<mix>.parquet` | scorer 研究轨/E3-Q3 |
| S6 manifest + global summary | `mix_order/…manifest.json`、`stage_s6_global_summary.json` | 验收 F2 |
| 验收记录（本清单打勾版） | `docs/ACCEPTANCE_<mix>_<date>.md` | 交付凭证 |
| 正式 token streams + 全量审计 | `data/token_streams/mix_m3_e1_100b-20261019/`、`data/processed/audits/formal_token_stream_release_mix_m3_e1_100b_20261019.json` | pretrain loader |
| snapshot-bound tokenizer corpus | `data/tokenizer_corpus_m3_20261019/` | tokenizer trainer |
| tokenizer corpus 独立审计 | `data/processed/audits/tokenizer_corpus_m3_20261019.json` | tokenizer 数据门禁 |
| pretrain 数据终态证书 | `data/PRETRAIN_DATA_READY.json` | tokenizer / scaling / pretrain 点火门禁 |

历史机器证书：`data/processed/audits/data_pipeline_acceptance_20261018.json`
（当时 status=PASS，SHA-256 `51252aea1d65d9a0697557b5eb6bec8da5c9c2ba8b818ee148e62da123529438`；
已被 2026-10-19 的全文去污染策略变更取代，不得用于当前 release）。
