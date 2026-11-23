# YuFormer 数据治理管线设计（S0–S7）

> 2026-10-20 当前版。历史输入为 19 个候选源；正式 15 源范围为 2.447T reference-token 治理池，主候选输出为 1T 三阶段训练消费计划。
> 执行原则：文件级分片、176 核并行、逐步落盘（parquet 中间格式）、幂等断点续跑、全程可审计。

## 总览

```
S0 摄取归一 → S1 硬过滤 → S2 精确去重 → S3 近重去重 → S4 质量打分 → S5 计长分桶 → S6 配比物化 → S7 审计验收
   ~1天        ~2天         ~1天          ~2-3天         ~2天         ~1天        ~2天         穿插
```

## S0 摄取归一（~1 天）

- 每个下载文件 = 一个 shard 任务；pyarrow 原生读 parquet，zstandard 流式解压 zst/gz。
- Adapter 字段映射 → canonical record：`{id, text, domain, source, license, lang, upstream_score, meta}`。
- 坏行容错跳过并计数；输出 parquet 分片 + 每源摄取统计。

> ✅ **已实现**（2026-09-17）：`src/yuformer_data/scale/`（`ingest.py` / `schema.py` /
> `profiles.py` / `driver.py`），逐源配置 `configs/ingest_profiles.yaml`。
> 文件级幂等断点续跑、shard 级失败容错、zstd 压缩 parquet 流式写出、
> meta = 原始行 JSON（剔除正文字段、截断 4KB）。用法见 `README.md`「生产管线（S0/S1）」。

## S1 硬过滤（~2 天，drop 率目标 10-15%）——✅ 已实现（2026-09-17 审查修复后 v2）

哲学：**严格清洗、低丢弃**。只有明确的垃圾才 drop；质量平庸的留给 S4 打分排序。

**实现与修复记录**：
- 实现：`scale/detectors.py`（纯函数规则）+ `scale/filter.py`（执行）+ `tests/test_scale.py`
- 2026-09-17 代码审查（报告见会话记录）修复：B1 worker 死亡 Pool 静默挂死 → driver 停滞守卫（30 分钟无分片完成硬退出，断点无损重启）；H1 BatchWriter 无字节上限 → 256MB 字符硬顶 flush；H2 空串/非字符串字段不再阻断候选回退；实体规则改字符占比（与标签规则单位一致）；base64 规则 DNA 单字母表豁免 + 非 web 域降级 review；secrets 修死代码 + 占位符豁免 + 高熵/引号要求 + AKIA/ghp_/sk-/PEM 具体模式；乱码签名补 emoji/标点双编码形态（ðŸ/â€ 系）
- **规则版本**：全量 S1 分两段执行——前 6,600 分片为 v1 规则，其后为 v2；差异占比 <0.5%（实体/base64/secrets 放宽 + 乱码收紧），审计可 per-shard 区分
- **H3 决议**：S1 产物保留原文（normalize 仅用于检测），S2/S3/S4 统一 `from .detectors import normalize_text` 复用同一规范化，禁止各自实现
- **已知缺口 v2 补**（下轮增强，源自审查的工业界对齐项）：URL 残留行、cookie/GDPR 横幅、"Enable JavaScript" 模板页、lorem ipsum、404/停放页、词级 n-gram 重复率（FineWeb 核心信号）、行级 dup 扩到 code/books、邮箱/电话密度、`<!-- -->` 注释残留、中文全角/GBK 误读签名、math 答案信号与 code generated/LICENSE 规则

**通用层**（全部领域）：
- NFKC 归一、控制字符清理、空白折叠
- 完整性：空/过短(<16字符) drop；超长(>2M字符) review
- 敏感串扫描（API key/密码/token 正则）→ review
- 语言检测（fasttext langid）→ lang 字段 + 低置信 review

**乱码过滤（新增，用户明确要求）**：
- U+FFFD 替换符占比 > 1e-4 → drop
- Latin-1 误解码签名（Ã/Â/å/ç 等在非对应语言文本中高频出现）→ drop
- 控制字符占比异常、UTF-8 非法序列残留 → drop
- 规则命中打 `reasons: [garbled]`，进审计统计

**HTML 源码/无用数据过滤（新增，用户明确要求）**：
- 标签字符占比（`<...>` 内字符 / 全文）> 5% → drop（大段 HTML 源码）
- `<script>/<style>` 残留块 → drop
- HTML 实体洪泛（`&lt; &gt; &amp;` 密度超阈值）→ drop
- base64 / data-URI 大 blob → drop

**类别层**（复用现有 Step 逻辑）：web boilerplate/行重复率；code vendor/generated/弱信号；math 答案信号；中文乱码变体规则。

> ✅ **已实现**（2026-09-17）：`src/yuformer_data/scale/detectors.py`（纯函数规则集）+
> `filter.py`（执行器）。三态 keep/review/drop；drop 不写正文、只落
> `s1_audit/<source>/<relpath>.parquet`（doc_id+reasons+text 前 200 字符）供抽查；
> keep/review 带 decision/reasons/lang/metrics 列落 `s1/`。
>
> 实测（2026-09-17，director env，单核；数字与样本明细见 `README.md`「生产管线（S0/S1)」）：
> S0 parquet 37–45 MB/s/核（目标 ≥20 ✅）、gz 流式 15–27 MB/s/核；
> S1 检测器本体 7.5 万行/s（短行）/ 5.1 千行/s（fineweb ~2KB 行，≈15MB/s 文本）；
> 5 shard 样本 drop 率 0.31%（符合"严格清洗、低丢弃"哲学）。
> 测试 `tests/test_scale.py`（规则正负用例 + 真实 3 文件端到端 smoke）。

## S2 精确去重（~1 天）——选型已定

- **方法**：归一化文本（NFKC + casefold + 空白折叠）sha256，按哈希前缀分 256 桶落盘两趟扫描（与 Dolma / RedPajama 生产实践一致）。
- 跨文件、跨源生效（fineweb 与 dclm/c4 的 CC 重叠在此消除）；桶内保留规则确定性（来源优先级 > 哈希决胜）。
- 预计瘦身 5-10%。

> ✅ **已实现**（2026-09-17）：`src/yuformer_data/scale/dedup_exact.py` +
> 共享库 `dedup_common.py`，配置 `configs/dedup_config.yaml`。
> 三趟：pass1 每 shard 算 sha256 写 `s2_hash_parts/<domain>/`（桶列 + 哈希 +
> doc_id + source，排序落盘）；pass2 按 (domain, 桶范围) 任务读入内存分组
> （内存上限超限报错，不许 OOM），组内 survivor = (source 优先级表, doc_id 字典序)
> 最优，drop 决定写 `s2_decisions/<domain>/`（仅 drop：doc_id, decision, dup_of）；
> drops 构建排序 bin 查找表（memmap + searchsorted）后 pass3 流式合并回
> `s2/<source>/<relpath>.parquet`（s1 全列 + `s2_decision`）。
> 文件级幂等（summary 齐在即 skip，重跑输出字节一致）、tmp+os.replace、
> 部分源重跑有 domain_sources 守卫。归一化 import 复用 S1 `normalize_text`
> （+ casefold + 空白折叠）。测试 `tests/test_dedup.py`。

## S3 近重去重（~2-3 天，最重一步）——选型已定，**2026-09-22 止损收官**

**选用 MinHash-LSH（CCNet / Dolma / FineWeb 系的工业标准），放弃现有 SimHash-64 实现。**

| 候选 | 判定 | 理由 |
|---|---|---|
| SimHash-64 带状（现有代码） | ❌ 弃用 | 64-bit 签名在 50 亿文档规模召回不足；纯内存索引无法跨分片 |
| **MinHash-LSH** | ✅ **选定** | 5-gram 词组 shingles + 128 排列；LSH 分 16 band × 8 行（候选阈值 t≈0.71）；候选对计算真实 Jaccard ≥0.8 判重。text-dedup/Dolma 同款配置 |
| 后缀数组精确近似 | ❌ 成本过高 | O(n²) 级，5T tokens 单机不可行 |
| Embedding 语义去重 | ⏳ 研究选项 | GPU 成本高，列为 round-2 消融（与 SimHash 对比写报告） |

- 工程：两趟扫描 + 磁盘 KV（band→doc_ids 分片）；签名生成 176 核并行；先去重池内跨源（web 三源互去）。
- 风险：最重一步，先拿 100GB fineweb 原型压测定吞吐再全量跑。

**2026-09-22 止损记录（E-17）**：web 域近重聚类因候选边爆炸（~0.5B 边/字节桶，预估 40+ 天）跳过，决策记全 keep；S3 实际执行 = 6 小域完整 MinHash-LSH 近重 + web 域 S2 精度（精确去重已去 45.7%，含全部跨源精确重复）。依据：CC 派生源的跨源"近重"以共享 boilerplate 为主（前缀 Jaccard 无法区分模板相似与内容重复），工业先例仅做源内去重。候选爆炸的量化分析进报告（发现 R-1）。

**历史设计：2026-09-26 E-17 逆转（已被 2026-10-18 方案替代）**：`scale/dedup_v2.py`，driver `--stage ndv2 | ndv2_refresh`。当时采用全文 5-gram shingle 重验证与 connected-components；它解决了前缀误判，但随机正文点查和传递闭包仍不适合十亿文档生产规模。保留本段作为事故演进记录，当前合同以下方 2026-10-18 设计为准。

**历史设计：2026-09-25 逆转（已被 2026-10-18 方案替代）**：E-17 止损撤销，首次引入全文 shingle 重验证。新增的评测去污染仍为有效要求；ndv2 的执行合同以后文当前正式设计为准。

**2026-10-15 的 ndv2 放弃决定已于 2026-10-18 撤销。** 当时的全文随机点查实现确实不可运行，但它不是算法边界。

**2026-10-18 当前正式设计**：候选专用的 128×4-bit MinHash 索引把验证载荷降为 64B/doc；每域以 1,024 对全文 5-gram Jaccard 审计约束近似边精度（点估计 ≥97%，95% Wilson 下界 ≥95%）。删除决定采用 16 路外排 direct-edge dominance，不做近似边的传递闭包。code 实测 103,099,673 边、39,220,532 drops、峰值 3.3GB；全文审计 995/1,024 通过（97.17%，Wilson lower 95.96%）；web 全文审计 1,013/1,024 通过（98.93%）。S5 直接消费排序 lookup；门禁绑定决定表、边集精度审计和全局 dropped 账。`ndv2_refresh` 仅为读取旧 S3 decision 列的外部消费者保留，不是正式 S5/S6 前置。

> ✅ **已实现**（2026-09-17）：`src/yuformer_data/scale/dedup_minhash.py`。
> 全链六步（全部磁盘分片、单任务内存上限保护，超限报错不许 OOM）：
> ① **sig**（shard 并行）：5-gram 词组（空白切词，原始文本 span 直接 blake2b-16→uint64）
> + 128 排列 `h_i(x)=(a_i·x+b_i) mod 2^64`（a_i/b_i 由 blake2b("perm", i) 派生，零随机数）
> → 每 doc 512B 签名，长文截断 100k 字符（计数入 summary）；写 `s3_sig/<domain>/<doc_id 前缀桶>/`
> 与 16 个 band part `s3_lsh/<domain>/b<b>/`（band_key = blake2b-16(sig[8b:8b+8])，按
> `band_key % 2048` 桶列排序）。
> ② **band reduce**（domain×band×桶范围任务）：numpy 向量化排序分组，同 band_key ≥2 doc
> 发候选对（组 >50 星型归约），写 `s3_pairs/`。
> ③ **候选集**：全局唯一候选 doc_id 集合（`s3_cand/*.npy`）+ 候选对按 doc_id 首字节重分区
> （`s3_pairs_bydoc/`）。
> ④ **collect + docstore**：shard 并行收集候选正文为 runs（每 run 内按 doc_id 排序）。
> **run 区间化 docstore（2026-09-21 修复，替换原 k-way merge 全局物化）**：web 域 12 亿候选 /
> 1.2 万 runs / 2TB 文本下，全局归并实测 ~17k 行/s 且 2TB texts.bin 写 page cache 触发 cgroup
> OOM（RSS 仅 7.8GB）。改为 runs + 每 run doc_id 列副本（V16 平铺 sidecar，memmap）：
> doc_id 首字节 = k 的文档在每个 run 内是连续区间 ⇒ verify 按桶取文本 = per-run
> searchsorted + 行组切片，无全局归并、无大 bin 写出。`DocStore` 双模式兼容存量四 bin
> 小域。旧格式（`s3_store/<domain>`：ids/srcs/offs/texts.bin）仍被 legacy reader 读取。
> ⑤ **verify（小域：精确 Jaccard）/ UF-direct（web）**（2026-09-22 实战修正）：
> 小域（候选对 < `uf_direct_pair_threshold`=1e9）：逐对重算真实 Jaccard（唯一 5-gram 集合交并，
> 前缀截断 verify_text_chars=2048，≥0.8 判重）→ `s3_verified/`。**web 域 84 亿候选对下逐对验证
> IO 不可行**（伙伴正文/签名散点读 ≈ 400TB 量级，实测任一路径均不可接受），改为
> **候选对直并查集聚类**（CCNet 做法）：签名按 doc_id 首字节重排为 flat bin
> （`s3_csig/<domain>/_idx/<byte>.{ids,sig}.bin`，s3_sig 已按首字节分区故无需搬移正文），
> 候选对全量（8.4B 边）经全局 doc 索引（bin 偏移 + searchsorted）入 numpy 并查集
> （路径压缩到不动点），簇 survivor = (source 优先级, doc_id) → drops。LSH 16 band(t≈0.71)
> 候选即边 ⇒ 去重略激进于 0.8 严格阈值，簇级 survivor 规则兜底；确定性不变。
> ⑥ **decide + materialize**：小域读 `s3_verified/` 聚类；web 用 ⑤ 的 UF 结果；
> drops 决定写 `s3_decisions/<domain>.parquet` + drops 查找表；shard 并行物化
> `s3/<source>/<relpath>.parquet`（s2 全列 + `s3_decision`）。
> 作用域为 domain 内跨源（web 三源互去、代码两源互去）。与规格的偏差：s2/s3 输出保留全部
> 行（drops 打标不过滤），与 S1 三态审计哲学一致，S6 物化时再过滤；签名大表（512B/doc）
> 仅作中间产物。实测与测试见 `README.md` 与 `tests/test_dedup.py`。

## S4 质量打分（~2 天，**本项目核心贡献**）——选型已定

**三源打分 ensemble（2026-09-20 用户裁定：算力允许下效果最优方案），不用本地 LLM 打标：**

**时序锁定（2026-09-17 用户裁定）：打分器 gold-set 与 tokenizer 训练样本都从 S3 去重后的池子抽取**——去重消除重复 boilerplate 对词频/样本分布的污染，顺序为 S1→S2→S3→(tokenizer 抽样训练 ∥ S4 gold-set 训练)→S4 全池推理。

| 轨 | 方法 | 载体 | 工业锚点 |
|---|---|---|---|
| 规则/上游分 | 上游分透传 + S1 决策特征 | CPU | — |
| **自训 fasttext** | gold-set 二分类（硬负为主 67:33） | 160 CPU 核 | DCLM/OLMo 配方 |
| **BERT 回归器（2026-09-20 新增）** | 同 gold-set 训练，mBERT-base 回归打分 | 4×H800（数据阶段闲置算力，与 fasttext 并行互不抢占） | QuRating 配方 |
| **composite_v2** | 三轨可配置加权融合（**2026-09-26 E-20 裁定**：生产配比选取走 fasttext+上游双轨——BERT 全池实测 4.5-6k docs/s 需 2-3 周不可接受；BERT 转消融角色只对 E3 实验子集打分。bert 缺轨自动降级双轨，已预置） | | |

- BERT 档规格：`bert-base-multilingual-cased`（中文覆盖必需），FP16 批量推理，目标聚合 ≥6 万篇/秒（全池 ≤16 小时墙钟）；GPU 数值不确定性弱于 CPU，分数量化到 1e-4 并文档化
- **截断策略（2026-09-20 修订：拒绝一刀切头部截断）**：短文档（≤1024 字符）头部直打；**长文档（books/academic/code 等 >4K 字符）用 head/mid/tail 三窗 × 256 token 取均值**——代码文件头部全是 license/import、书籍头部是扉页，头部截断对这些域系统性失真。fasttext 轨同此策略（当前 1024 字符截断有同样问题）
- **截断校准门（S4 预检门新增，数据驱动决策）**：抽 5,000 篇（短/中/长分层），比较 head-only vs 三窗 vs full-doc(2048) 三种打分的 Spearman 相关。若 head-only 与 full ≥0.9 → 全池用 head-only（省算力）；0.75-0.9 → 长文域启用三窗（开销 ~1.3×，长文只占 ~10-15%）；<0.75 → BERT 档 max_len 升 512 并全池三窗。校准结果连同相关性数据进报告（打分器章节的实验点）
- **全量推理输入 = 去重后池**（`s3_decision == keep` 的行）
- **正样本质量门**：gold-set 内部先做交叉验证（留一源验证，防"分类器只认识某领域文风"），fasttext 与 BERT 两档都要过门；AUC/PR 不达标走预检门预案。
- **效率硬指标（验收标准）**：fasttext 160 核聚合 ≥5 万篇/秒（全池 ≤1 小时墙钟）；BERT 4 GPU 聚合 ≥6 万篇/秒；领域内分位数用 per-domain 离线排序（更精确、同值同分）。全部分数轨落盘支持消融（上游分/fasttext/bert/composite_v2——E3-Q3 四臂）。

### S4 预检门：打分器数据审计（2026-09-19 用户关切立规，**S5/S6 之前强制通过**）

用户关切：负样本不足/结构失衡会导致打分器不可用。以下审计全部通过才允许进入 S5/S6：

1. **负样本结构**：硬负样本（平庸但正常：dclm/edu 低分尾）占比 ≥50%（当前 400k:200k = 67:33 ✓ 2026-09-19 已加固）。显而易见垃圾只提供判别下限。
2. **留一源 CV 门禁**：mean AUC ≥0.85 **且** 每个 holdout ≥0.75 **且** 中文 holdout 必须通过（防语言偏置回归——开发版 0.617 被拒的教训）。
3. **覆盖盲区声明**：正样本含 en/zh 百科/书/学术/edu/math；**code 域为已知盲区**（无正亦无负），打分器对 code 域分数仅作参考，code 配额选取以规则+上游元数据为准；全池推理后检查 per-domain 分分布，code/math 域不得整体塌缩到 0 或 1（ Sanity 检查）。
4. **门禁失败的分级预案**：① 负样本扩充 v2——用 v2 规则未覆盖的 boilerplate 模式（URL 残留行/cookie 横幅/lorem ipsum，见 S1 已知缺口）合成增强负样本；② dclm 全量上游 fasttext 分回归预训练 → gold-set 微调；③ 最终降级——top-k 选取退回"上游分（web 域）+ 规则（其他域）"，自训分仅作消融臂（三轨落盘架构原生支持，配比实验 E3-Q3 正好量化打分器的实际贡献）。

> ✅ **已实现**（2026-09-17）：`src/yuformer_data/scale/scorer_train.py`（gold-set 构造 +
> 留一源交叉验证 + fasttext 训练）与 `scale/score.py`（双轨打分 + 领域内分位数），配置
> [`configs/scorer_goldset.yaml`](../configs/scorer_goldset.yaml) /
> [`configs/scorer.yaml`](../configs/scorer.yaml)，driver `--stage s4|score`。
>
> 与本草稿的两处实现差异（记录于 scorer_goldset.yaml 头注释）：
> ① 分位数用「两遍法 + per-domain 离线排序数组（numpy）+ searchsorted 线性插值」，
> 替代 t-digest——全量排序更精确、同值同分（决胜无关）、pass2 查询 O(log n)，
> 全量内存 ~40GB（本机 2TB，符合规格"内存够"的前提）；
> ② 负样本"fineweb 上游 ≤1 尾部"因 fineweb 无上游分数字段，改用 dclm-baseline
> 0-1 分尾部（`upstream_max: 0.02`）；s1_audit 负样本正文限 text_head ≤200 字符。
>
> 实测（director env，2026-09-17，单核）：见 `README.md`「生产管线（S4）」——端到端
> ~0.6–1.7 千篇/s/核（高负载机实测），瓶颈为 fasttext 本体 O(tokens×dim)，
> ≥2 万篇/秒/核 未达标及量化分析/可调配置项见 README；分位数标注 pass ~3.2 千篇/s/核。
> `tests/test_score.py`：gold-set 比例 / 3 微源 CV 结构 / 训练 md5 确定性 /
> composite+分位数数值断言 / 幂等 / 真实分片吞吐断言（当前机器上不达阈值，见 README）。
>
> ✅ **gold-set 构造并行化**（2026-09-26，S6 前关键路径瓶颈解除）：规则内 shard 列表按
> 字节数均分到 spawn 子进程（worker 数=min(32, 核数)，`YUFORMER_GOLDSET_WORKERS` 可覆盖），
> 每 worker 流式水库抽样（均匀随机 key 取最小 quota 个 = 无放回等概率；key 由
> blake2b(seed|规则名|w{i}) 子流驱动），结果落盘 parquet 后按 (worker, 行序) 确定性合并——
> 管道回传大对象曾触发 SimpleQueue._wlock 死锁（2026-09-26 实测），落盘方案根除；
> 无 reasons_any 规则用 arrow compute 向量化过滤（只物化合格行），GC 全程禁用
> （树形对象无环，避免巨大常驻堆 × 高分配率触发 gen2 全量扫描，实测 ~100× 降速）。
> 确定性：同 seed 重跑逐行一致（`tests/test_goldset_parallel.py`：两次全量构建
> md5 相同 + 与单线程参考实现 χ² 分布对拍 + 真实数据 wikipedia-zh 双跑对拍）；
> 输出契约（gold.parquet / gold.fasttext.txt / goldset.summary.json 字段、
> 2026-09-25 去重逻辑）不变。实测全量 ~1.13M 条：采样 4984s + 落盘 ~230s，共 5213s
> （decontam 128 核同机满载竞争下的墙钟；edu-high 84GB/185s ≈ 470MB/s 聚合可证
> 瓶颈为外部竞争，空闲机按单 worker 28–47MB/s × 32 估算 ≤ 20 分钟达标）。
> ⚠️ 附带发现：`edu-low-tail` 规则（upstream_max=1.0）在当前 s1 数据下合格行为 0
> （fineweb-edu 各分片 upstream_score 实测最小值 2.516），`satisfied=false`——
> 配置注释"0-5 分底部 ~20%"的前提已不成立，待数据侧复核（edu 上游分是否被改写）。
>
> ✅ **BERT 回归档已实现**（2026-09-20）：`scale/bert_train.py`（gold-set 复用
> scorer_train 同一构造与留一源 CV 划分；mBERT-base 二分类 CE，sigmoid 概率作分；
> bs64 / lr 2e-5 / 2 epoch / warmup 5% / AMP fp16，单卡即可）与 `scale/score_bert.py`
> （GPU 推理：s3 keep 池 → `s4_bert/<source>/<relpath>.parquet`，4 进程每进程一卡，
> FP16 batch 512，shard 级幂等，bert_score 量化 1e-4——GPU 数值不确定，重跑 ±1e-4
> 抖动属预期）。融合升级：`composite_v2 = 0.25·upstream_norm + 0.35·fasttext +
> 0.40·bert`（configs/scorer.yaml，按行重归一；bert 整轨缺失自动退回双轨 composite），
> 四列全落盘（upstream_norm/fasttext_score/bert_score/composite_v2），分位数排序键
> 切换为 composite_v2；driver `--stage s4_bert`（GPU worker，复用 stall-guard/进度日志）。
> CV 门禁同 fasttext 档（mean ≥0.85 且每源 ≥0.75），结果写
> `data/models/bert_scorer_cv_metrics.json`。dev 规模验证（scorer_goldset_dev 子采样
> 6k 行）：mean AUC 0.905 / min fold 0.883 ✅ 过门；正式 gold-set（S3 池）CV 与全量
> 推理待 S3 完成后统一排期。实测吞吐（4×H800，~2KB/篇，dedup 同机高负载）：7.4k 篇/s
> 聚合端到端，纯 GPU forward 上限 ~7.9k/GPU；**≥6 万篇/秒聚合指标经 FLOP 量化分析
> 物理不可达**（mBERT-base fwd ~56 GFLOP/篇，4 卡 100% MFU 上限 ~48k），降级路径
> （domain_allowlist / max_len 128 / INT8-TensorRT / 加卡）待裁定，详见 README
> 「生产管线（S4）」实测吞吐节。
>
> ✅ **截断策略修订**（2026-09-20 用户裁定）：`window_scoring: true` /
> `long_doc_chars: 4096`（scorer.yaml 根级，fasttext/BERT 两轨共用）——短文档（归一化
> ≤1024 字符）头部 max_len 直打；长文档（>4096，books/academic/code）head/mid/tail
> 三窗各 max_chars 字符 / 256 token 均分。新模块 `scale/truncation_calibration.py`：
> 确定性分层抽 5,000 篇（短/中/长 40/40/20%）算 head-only / 三窗 / full-doc(512)
> 三种 bert_score，Spearman 相关矩阵 + 判定建议（head vs full ≥0.9 全池 head-only；
> 0.75-0.9 长文三窗；<0.75 全池 512 三窗）写 `data/models/truncation_calibration.json`。
> 三窗版实测开销 ~1.3×（15% 长文混合语料单卡 2038 vs 2725 篇/s），与预估一致。

## S5 计长分桶（~1 天，依赖 tokenizer 模块）

- 注入自训 BPE 精确计长；B4K/B16K/B64K/gt_64K 互斥前缀嵌套（mix4K⊂mix16K⊂mix64K）。
- 未就绪前用 chars/4 估算，tokenizer 就绪后重算。

> ✅ **已实现**（2026-09-26）：`scale/length_bucket.py`，driver `--stage s5`。
> 三路决策 join（s2 列 / s3_decisions lookup bin memmap / decontam 决策表）+ yf-64k-bpe
> `encode_batch` 精确计长（add_special_tokens=False）+ 互斥桶与 mix_eligibility 资格
> （training.py 前缀嵌套语义）。硬前置 = decontam 全覆盖（fail-fast）。
> 实测与 ≥2 万篇/秒/核 未达标量化分析见 README「生产管线（S5）」；测试
> `tests/test_length_bucket.py`。真实 fineweb 分片抽查 n_tokens 逐篇精确一致。

## S6 配比物化（~2 天）

- **前置实验子集**：从 M3 档（top-10% 质量）确定性切出 ~100B tokens 单一分布子集，供 E1 缩放律 / E2 架构消融全部前置实验共用（保证可比性；D 轴由其子采样实现）。
- **留出验证集**：打分池内按领域确定性切出 held-out val set（各 50-100M tokens，永不进训练）——配比实验与训练监控依赖它。
- **多配比支持**：mix 配置 = 领域权重 + top-k 比例 + 课程参数；同一打分池可低成本物化任意多组训练集（E1 实验的基础）。
- 默认候选：50B top-10% warmup + 800B quality-prefix stable + 150B top-10% anneal。每段内部确定性全局打乱，跨段重复显式记账；未经验证的 upstream/edu 分数不再冒充难度课程。三套实验画像与 E3 八臂集合见 [`docs/research/数据配比方案.md`](../../docs/research/数据配比方案.md)。
- tokenize 写训练二进制分片，分片 manifest 带 token 数（pretrain 迭代数推导元数据）。

> ✅ **已实现**（2026-09-26，E1 前最后一块）：`scale/materialize.py`，driver
> `--stage s6 --mix-config <yaml>`（`s6_val` 单独切 val）。**硬门 fail-fast 四+一项**
> （2026-09-25 清单落地，`gate_errors`/`assert_gates`）：① s4_scored 覆盖 + s4 全局
> 无错误 + 分位表构建；② decontam 全部完成无 error shard；③ ndv2 的 web/code
> s3_decisions policy ≠ near_dup_clustering_skipped + refresh 全局无错误；
> ④ 每领域 val 已切出（ensure_val_sets：blake2b 确定性抽样 50M tokens/域，
> 两遍流式 4096 桶直方图，val_sets/ 落盘，mix_members 之外）；⑤ s5 覆盖（读入源
> 完整性）。配比：分位表阈值（composite_v2，同值同分决胜无关）→ 候选收集（排除
> val，parts 落盘）→ (composite 降序, doc_id 字节序决胜) 全向量化 lexsort →
> token 前缀到最大余数法配额（training.allocate_mix_quotas）。课程两段：
> stage1 = 各领域 upstream edu 低→高前 25% tokens；stage2 = 余 75% 按 doc_id
> 字节序（= blake2b 均匀哈希序，确定性全局打乱）。**物化 doc_id 顺序表**
> （mix_order/<mix>_stage{1,2}.parquet + manifest），tokenize 放 pretrain 阶段。
> mix 配置：configs/mix_{m1,m2,m3}.yaml + mix_m3_e1_100b.yaml（首次物化目标）。
> 全局 summary：per-domain/per-source 入选 tokens + drop 原因（s2/s3/去污染剔除、
> 质量分位数截断量、配额缺口）。测试 `tests/test_materialize.py`（硬门 fail-fast /
> 最大余数法数值 / top-k 双跑字节一致 / val 互斥 / 合成端到端 / driver 接线）。

## S7 审计验收（穿插）

- **去污染**：物化前对评测集（MMLU/GSM8K/MATH/HumanEval/C-Eval）做 n-gram 重叠检查，命中剔除。
- Review 队列工作流：S1-S4 review 样本抽样复核，修正规则。
- 分布体检报告：每源 token 量、领域占比、长度/分数分布 vs S0 预期。
- 全链路 record 级 manifest。

### S7-decontam 评测去污染（2026-09-25 实现，S6 物化硬前置门）

**方法**（GPT-3/LLaMA 同款 n-gram 重叠法）：基准测试集题面+选项+答案全部纳入 →
归一化（NFKC+casefold+空白折叠，复用 S1 `normalize_for_hash`，H3 决议不另实现）→
8-gram 词组哈希 → 去重集合（排序 uint64 bin，memmap + searchsorted 判定）。
全池文档流式过一遍，共享 ≥1 个 8-gram 即 drop，决策表记命中基准名。

**阈值依据（工业锚点）**：GPT-3 用 13-gram、LLaMA 用 10-gram。两者均为百亿/千亿参数
模型；YuFormer 0.5B  memorization 下限更低、评测集更小（GSM8K 仅 1.3k 题），同量污染
对评测效度的破坏更大——取 **8-gram 更严格**（任何 8 词连续重叠即剔除，比 LLaMA 的
10-gram 多捕"局部模板/公式片段"级重叠）。

**关键设计**：
- **哈希方案**（与"blake2b-8"规格的偏差记录在案）：词→id 用 FNV-1a-64、词组用多项式
  rolling hash（mod 2^64），numpy 全向量化——blake2b 无法向量化，达不到全池扫描吞吐
  目标（E-12）。64-bit 空间下全池 ~1e12 次 shingle 查询，假命中期望 <1e-4；决策表带
  `hit_benchmark` 列供审计抽样复核。
- **中文 CJK 逐字 token**：中文文本无空格分词，整句=1 个"词"会让 8-gram 永远构不出
  （C-Eval 整体漏检）。CJK 表意文字逐字成 token、拉丁连续段成词，中英文同一管线。
- **全文扫描**：生产配置为 `max_shingles_per_doc=null`。2026-10-19 的正式 mix 审计在
  dolma-books 的 50k 前缀之外发现 benchmark 重叠，否定了“尾部概率可忽略”的旧假设。
  S1 `max_chars=2M` 限制单文档规模；`_decide_batch` 按固定 2M shingles 预算分批查询
  benchmark memmap，使峰值内存与 parquet batch 中长文档总长度解耦。
- **索引与扫描同源**：基准索引构建与池扫描共用同一 `doc_shingle_hashes`（同一归一化、
  同一 tokenizer、同一哈希），不存在两侧实现漂移面。
- **内存模型（E-13/E-14）**：索引为 memmap 排序 bin（2.3M shingle ≈ 18MB），160 worker
  共享 page cache，per-worker RSS 极小；文档流式逐行，无任何 accumulate-then-process。

> ✅ **已实现**（2026-09-25）：`scale/decontam.py` + 配置 [`configs/decontam.yaml`](../configs/decontam.yaml)，
> driver `--stage decontam`。基准源 5 个全部经 hf-mirror 逐个验证（HTTP 206 + parquet
> schema 实读：mmlu all/test 14,042 题、gsm8k main/test 1,319、hendrycks_math 7 学科
> test 5,000、openai_humaneval test 164、ceval-exam 52 学科 test 12,342；hendrycks/
> competition_math 为脚本数据集无静态 parquet，按规格用 EleutherAI 镜像）。实际索引
> 2,276,997 unique 8-gram（低于预估值 5-15M：选择题选项短、CJK 逐字后 shingle 数仍有限）。
> 产物：`decontam/<source>/<relpath>.parquet`（doc_id, decision, hit_benchmark——S6 join
> 用决策表）+ per-shard summary + `stage_decontam_global_summary.json`（shard_errors 置顶、
> per-benchmark 命中分布）。文件级幂等（parquet+summary 齐在即 skip）、失败分片只落
> error summary 不落空产物（E-16）、tmp+os.replace、停滞守卫复用 driver 同款。
> 实测（director env，单核，机器 BERT 推理+fasttext 训练高负载 ~130 核被占）：
> 短文档 ~0.7KB 1.9 万行/s/核；典型 ~2KB 文档 8.5-6.9k 行/s/核（2.6M shingles/s/核）。
> 全池 36.5 亿篇瓶颈在共享存储 IO（E-9：0.3-0.5GB/s 聚合）而非计算：13.9TB 按 0.4GB/s
> ≈ 10 小时墙钟，160 核计算裕量 >10×。测试 `tests/test_decontam.py`（合成基准注入 +
> 端到端命中/归一化/幂等/截断语义/跨进程确定性 + 真实下载 skip-if-不可达）。

## 下游接口

- 训练集分片：`tokenized shards + manifest(token 数/领域/bucket)`
- 配比实验：`mix config → 物化脚本 → K 组训练集`
- 缩放实验：同池不同 (N, D) 采样物化
