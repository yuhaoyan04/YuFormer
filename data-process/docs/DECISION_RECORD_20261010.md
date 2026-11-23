# 决策记录：v1.0-full 前数据侧三项裁定（2026-10-10）

> 性质：与 FAILURE_LEDGER 同级的决策文档（仿 E-31 格式）。本文锁定的事项在 v1.0-full
> 执行时照单执行，不再临时决策。触发背景：并行代码审查（阻断级 B1-B6 + 高优先级 6 项）。

## ① 中文 decontam 过杀——fast track 不动，full 前限定重跑

**症状**：CJK 逐字 8-gram ≈ 8 连续汉字，教学套话（"下列说法正确的是"）即命中 ceval 索引。
实测 cci3-hq 35.4%、skypile 17.6% 被 ceval 命中 drop，中文池 268M → 126M 腰斩。

**裁定**：fast track（mix_m3_e1_100b_fast）不重跑——E1 实验集中文仅占 3.5% 且 top-10%
质量选取，池子腰斩不改变配额，对 scaling 结论是噪声级影响。

**full 前重跑（锁定参数，待抽样复核数字回填 §A）**：
- 范围限 `--sources cci3-hq skypile wikipedia-zh` 三源（非全量，半天内）；
- 修复方案：**CJK 文档要求 ≥2 次独立 8-gram 命中才 drop**（优于单纯提 n：套话单条命中
  不误杀，真转载真题多条命中跑不掉）；
- 执行链：改 decontam → 三源重跑 → **手工清三源 s5 产物 → 重跑三源 s5**（S5 无模型
  指纹守卫，不会自动失效——E-24 家族缺口，执行时按指纹清单核对）；
- 判负样本泄漏（审查高优 6）： scorer/bert 训练负样本取自训练池，CV 数字系统性偏乐观——
  重训任何打分器前必须先修泄漏，否则新 CV 不可信。

## ② fasttext 打分器——退役，不重训

**裁定**：fasttext 轨关闭（`fasttext_track: false`，E-36 收官）；**full 版 score_field
= hybrid_v2**（mix_m3.yaml / mix_m3_e1_100b.yaml 已锁定，与 fast 轨一致）；
fasttext 转研究 backlog。

**理由**：修复路径（重训+修泄漏+跨机复现+复检）1-2 天，换来 AUC~0.85 的二分类器；
goldrule v2 holdout AUC 0.899，hybrid_v2 里 fasttext 权重 0.35 的边际贡献存疑。

**口径注意**：10-08 前全语料的 composite_v2 是 fasttext 参与打分的；开关生效后任何
重算分片走 fasttext=0 公式。**选取轨 hybrid_v2 不受影响**（三轨融合不含 fasttext）。

## ③ ndv2 web/code 近重去重——2026-10-18 已完成

**结果**：web/code 全量决定与全文精度审计已发布。web 1,013/1,024、code 995/1,024
抽样边通过精确 5-gram Jaccard 门；删除采用 direct-edge dominance，不做传递闭包。

**full S6 物化点火条件（写入门禁）**：决定 policy、精度审计边集指纹、lookup 文件行数和
全局 dropped 账全部一致。S5 直接消费 lookup；`ndv2_refresh` 只保留作旧消费者兼容维护。

**伴随修复（已落）**：E-43 `_sig_prefilter` a 侧 miss 负索引误杀（dedup_v2.py）；
verify→cluster 间 fail-fast 经核实**已存在**（E-16 门禁，dedup_v2.py:1192）。
**code 域**：已与 web 同批完成；**math 域**：s3 域内近重已完成，
跨源近重未跑（policy 字段缺失）——计入下方已知风险清单，不单独排期（权重 1.8%）。

## 已知风险清单（v1.0-full 执行前逐项销项）

| # | 风险 | 处置 |
|---|---|---|
| R-1 | math 域无跨源近重去重（policy 字段缺失的静默跳过） | 接受，报告注明 |
| R-2 | S5 产物无模型/上游指纹守卫，decontam 重跑后须手工清 s5 | ①号重跑时按清单执行 |
| R-3 | stage_s1/s2/s3_global_summary 被局部重跑覆盖成子集口径 | 审计以 audit_consistency_20260926.json 为准 |
| R-4 | stage_s4_bert_global_summary 停在全失败快照（实际 4127 分片成功） | 刷新 summary；BERT 无指纹，重训后旧分静默沿用（级联刷新缺失） |
| R-5 | 中文 decontam 误杀率未验证 | §A 抽样复核（2h，进行中）后回填锁定参数 |
| R-6 | ndv2 web 未完成期间，S6 fast track 产物不含 web 近重去重 | E1 实验集豁免（已知偏差，报告注明）；full 必须等③完成 |

## §A 抽样复核回填区（ceval 误杀率实证，2026-10-10 完成）

- 样本：cci3-hq 150 + skypile 100 + wikipedia-zh 30（seed=42，decontam hit=ceval 的 drop 文档；
  逐篇人工判读 + 生产 ceval 索引同构实测 distinct 命中数）
- **误杀率：277/280 = 98.9%**（Wilson 95% CI [96.9%, 99.6%]；cci3 98.7% / skypile 99.0% / wiki-zh 100%）
- **真污染仅 3 篇**（真题转载 dump，命中数 3/9/14）；误杀高频类型：法条/公文/政策原文引用、
  科普、新闻通稿、百科条目——命中证据为"的说法，正确的是"等套话与固定表述
- 方案实测（真污染保留 / 整体误杀救回）：≥1（现状）3/3 / 0% ｜ **≥2：3/3（100%）/ 33.6%** ｜
  ≥3：3/3（100%，安全余量已归零——真污染最小命中数=3）/ 55.6% ｜ ≥4：**2/3 放走真污染，禁用**
- **锁定参数（v1.0-full 照此执行，不再变更）**：三源重跑统一 **distinct 独立 8-gram 命中 ≥2 才
  drop**；ngram_n=8、max_shingles_per_doc=50000、tokenizer/归一化不动；命中计数用 distinct
  （codes==ceval 后 np.unique），禁用 raw 计数（重复段落虚增，样本最大 raw 185 vs distinct 111）
- 路线对比结论：多次命中 **优于** 提 n（敏感性锚点不动/利用"成片复制"信号/漏检不对称地落在
  真污染侧/可审计性不变）；若日后长尾误杀仍有压力，叠加"索引剔除 explanation/题干模板二阈值"，
  而不是动 n
- 外推（doc 级线性）：≥2 下 cci3-hq 救回 ~4.2M/13.2M、skypile ~13.1M/40.7M、wiki-zh ~38K/81.7K
- 遗留：样本取自各源前部 2-3 分片（真污染占比或有 ±数点波动，98.9% 误杀率结论稳健）；
  token 级回收量待重跑后按 summary 统计
