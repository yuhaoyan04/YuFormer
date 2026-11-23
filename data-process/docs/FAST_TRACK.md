# Fast Track：3T raw 级 pretrain 数据集的最快工业级交付方案（2026-09-27）

> **历史计划，已失效。** 2026-10-18 审计确认该 fast snapshot 的 web/code ndv2、中文去污染、
> EOD token 流与上游血缘均未过正式门禁，旧 E0-Q2 也已作废。本页只保留为决策历史；当前状态
> 以 `report/PROJECT_AUDIT_20261018.md` 和 release manifest 为准。编排器不得据此写出
> `DATA_READY`。

> 目标：以工业界验证过的管线配置，最快速交付可用数据集支撑 E0/E1 实验启动，
> 同时不牺牲最终数据质量（双版本策略）。

## 一、我们的管线 vs 工业界（配置对照）

| 环节 | 工业界代表做法 | YuFormer 现状 | 定位 |
|---|---|---|---|
| 抽取+规则过滤 | FineWeb/CCNet：语言ID/长度/符号比/重复行 | S0/S1（NFKC/乱码/HTML/敏感串/LID/域规则）✅ 完成 | 与工业标配一致 |
| 精确去重 | URL/doc-hash exact（各家的第一步） | S2（sha256 归一文本，跨源域内）✅ 完成 | **超出标配**（跨 CC 派生源互去，抓出 43.6%） |
| fuzzy 去重 | **FineWeb：仅源内 MinHash**；LLaMA3：跨源 | S3（源内 MinHash 全域）✅ + ndv2（跨源全文验证，超出标配） | 源内 ✅ 完成；跨源为增强项 |
| 质量打分 | LLaMA3/FineWeb-Edu：**fasttext 分类器**；DCLM：DCLM 分 | S4 三轨（fasttext/BERT/rule）+ CV 门禁 | fasttext=工业标配；BERT 已按 E-20 转消融；**rule 兜底**为加速项 |
| 去污染 | GPT-3 13-gram / LLaMA 10-gram | S7 8-gram（更严格）✅ 收尾中 | 超出标配 |
| 配比+课程 | LLaMA3 配比、课程文献少见 | S6（域权重+最大余数法+课程两段）| 实现完待跑 |

**结论：管线配置本身已是工业级（多环超配）**——瓶颈不在方法在执行顺序。加速来自
**交付策略**而非砍质量。

## 二、加速决策：双版本交付（fast track 核心）

**依据 FineWeb 先例**：工业主流的 fuzzy dedup 以**源内**为主，跨源近重去重是增强项
（FineWeb 未做跨源；我们的 S2 exact 已跨源抓掉 43.6% 重复，跨源近重的边际增量预计
5-15% 数据量）。让它阻塞全部下游不值得。

| 版本 | 内容 | 关键路径 | 交付时点 |
|---|---|---|---|
| **v0.9-fast** | web=源内去重（E-17 占位）；**code ndv2 完成**（今晚）；score_field=rule_composite（已基线校准） | decontam(~24h) → S4(~14h) → S5(~8-15h) → S6(~15-20h) | **~10-01** |
| **v1.0-full** | ndv2 web 完成 → lookup 刷新 → **S5 lookup 指纹驱动自动重算（E-24）** → S6 重物化 | ndv2 web（3-4 天，后台并行） | ~10-03 |

机制保证（已实现）：
- S5 gate：`--allow-e17-web`（web 豁免、code 仍强制）；分片 summary 记 lookup 指纹，
  full 版 ndv2 完成后**自动过期重算**，无静默一致性风险
- S6 gate：mix 配置 `allow_e17_web: true`（`mix_m3_e1_100b_fast.yaml`）
- E0/E1 用 v0.9-fast 开跑（实验结论不依赖最后 5-15% 去重增量）；正式 1.5T pretrain
  用 v1.0-full

## 三、执行时间线（v0.9-fast 关键路径）

```
9-27 今  code ndv2 收尾（剩 251 桶，~8-12h @48w）│ decontam 主线 │ CV 收尾（goldrule 数据点）
9-28     decontam 完成 → 补跑 dolma-books/stack-exchange（+1h）→ S4 全量打分启动（14h，128w 独占）
         ndv2 web verify 后台并行（不动关键路径）
9-29     S4 完成 + 双分位表（composite+rule）→ ndv2 code decide+refresh（2h）
9-29 晚  S5 全量（--allow-e17-web，8-15h）
9-30     S6 物化 mix_m3_e1_100b_fast（15-20h，含 val 两遍切出 + goldrule 子集）
10-01    ★ v0.9-fast 交付（验收清单 ACCEPTANCE.md 逐项过）→ materialize_tokens → E0-Q2 开跑
10-03    v1.0-full：ndv2 web → S5 自动重算 → S6 重物化（正式训练用）
```

## 四、监控与健康护栏（已上线）

- `scripts/health_monitor.py` 守护进程（15min 周期）：进度/速率/ETA（日志尾解析）、
  STALL 检测（30min 无进展）、进程存活、error 增量置顶、资源水位；历史落
  `data/health/health_history.jsonl`
- 验收：`docs/ACCEPTANCE.md`（24 项硬/软门 + 交付物清单）
- 事故响应：停滞/死进程 → 查 health flags → driver 幂等重启（断点无损，E-7）

## 五、与"不加速"的对比

| | fast track | 等全量 ndv2 |
|---|---|---|
| E0/E1 开跑 | 10-01 | 10-03~10-04 |
| 数据质量 | web 少 5-15% 近重剔除（S2 exact 已在） | 完整 |
| 正式 pretrain | v1.0-full（不变） | 同 |
| 风险 | 双物化多一次 S6（~20h 机器时，非关键路径） | ndv2 web 超预估直接顺延一切 |
