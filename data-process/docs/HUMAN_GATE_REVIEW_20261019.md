# mix_m3_e1_100b 人工门禁签收

候选快照：`yuformer-snapshot-b27bb2dc051c09a9c9f5`

本文件只用于 E2/G2 人工判断。机器门 A1–G1（除 E2）已经通过；最终状态仍以
`data/processed/releases/mix_m3_e1_100b-20261019.release.json` 为准。

## E2：27 篇内容样本

请审阅 `data/processed/audits/human_review_mix_m3_e1_100b_20261019.md`。样本按九个领域各取
质量分位 p25/p50/p75，27/27 文档身份、S5 token 数与重分词均一致，机器判定 PASS。

自动规则标出 4 篇，均需人工确认：

| 样本 | 领域/来源 | 标记 | 当前解释 |
|---|---|---|---|
| 4、5、6 | books / dolma-books | `char_run` | Gutenberg 转录说明、星号/表格边框等排版字符，不是程序报错或乱码；正式 mix 的 300 篇 books 随机样本中 110 篇（36.7%）在前 300 字符出现同类说明，属于系统性出版元数据。样本 4 的说明约占全文 0.12%，正文完整，建议 PASS |
| 19 | multilingual / fineweb-2 | `pii` | 正文开头的 `13.12.2006` 被宽松规则命中；上下文是专利/科研发布日期 |

签收结果：**PASS**（`user-decision-20261019`）。项目负责人审阅九领域样本并进一步核验
Jupytext notebook 与 Gutenberg books 边界案例后，确认当前数据质量通过验收；无要求删除的
`doc_id`。

## G2：用途、再分发、PII 与退出机制

签收 `PASS` 表示接受以下受限研究合同，不表示取得底层内容的额外权利：

1. 只用于内部、非生产的 LLM 研究和 scaling/architecture 实验。
2. 不对外分发 raw text、token stream 或可重建逐行内容的产物；模型发布另做独立审查。
3. 3,200 篇抽检中有 62 篇被宽松 PII 正则标记：55 个 email、5 个 18 位数字、2 个中国手机号形状数字。抽查上下文以论文作者邮箱、开源许可证联系邮箱、Gutenberg 转录邮箱和数字形状误报为主；该检查不能证明全量语料已去标识化，残余个人信息风险仍存在。
4. 收到来源退出或删除请求时，记录 `source/doc_id` tombstone；后续训练前重建受影响的内容寻址快照和 token stream。当前签收不承诺对已训练模型完成机器遗忘。
5. 高风险来源事实已明确保留：CCI3-HQ 没有 blanket license；FineWeb/FineWeb2/FineWeb-Edu/Dolma 的数据库许可证不替代底层网页权利；ProofPile2、GitHub Code 和 StarCoderData 含混合的逐文档/逐仓库许可证。

逐源证据与固定 revision 位于 `data-process/configs/source_governance.yaml`；远端完整性、逐文件
provenance 和 G1 结果分别位于：

- `data/processed/audits/raw_source_completeness_20261019.json`
- `data/processed/source_provenance/recovered_source_provenance_20261019.json`
- `data/processed/audits/source_governance_20261019.json`

签收结果：**PASS for internal non-production research under the five controls above**
（`user-decision-20261019-g2`，2026-10-19）。项目负责人接受上述五项控制；本签收仅授权当前
候选用于内部非生产研究，不授权原始文本、token stream、可重建逐行内容或模型的对外发布。

若后续要求全量 PII redaction，本候选应撤销并在 S1/S5/token 计数链路重建后重新签发。
