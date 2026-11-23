# YuFormer Data Process

可审计的数据治理流水线——下载、清洗、去重、去污染、打分、配比、长度分桶和不可变 release。

## 数据领域与数据集

当前 release lineage 按 **9 个领域、17 个有效源**组织。C4 与 PG19 的历史失败摘要保留作
审计证据，但二者已经显式退出 release scope；受权限限制而未获取的候选源也不计入当前数据集。

| 领域 | 数据集 | HF 路径 | 体量 | 用途 |
|---|---|---|---|---|
| **通用网页** | FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | 1.3T tokens | 教育质量分数，课程式排序 |
| | FineWeb | `HuggingFaceFW/fineweb` | 15T tokens | 通用网页主干 |
| | DCLM-Baseline | `mlfoundations/dclm-baseline-1.0` | 4T tokens | 跨源网页补充 |
| **代码** | StarCoderData | `bigcode/starcoderdata` | 783GB | 多语言代码 |
| | GitHub-Code | `codeparrot/github-code` | 300GB 配额 | 代码补充源 |
| **数学** | Proof-Pile-2 | `EleutherAI/proof-pile-2` | 55B tokens | arXiv+OpenWebMath+AlgebraicStack |
| | AutoMathText | `math-ai/AutoMathText` | 14B tokens | LLM 筛选数学文本 |
| **学术** | peS2o | `allenai/peS2o` | 40M 论文 | S2ORC 清洗版 |
| | arXiv LaTeX Corpus | `KiteFishAI/arxiv-tex-corpus-full` | 80GB | arXiv LaTeX 源码 |
| **文学** | Dolma Books | `olmo-data.org/dolma-v1_6/books` | 3 shards | Gutenberg 书籍 |
| **百科** | Wikipedia (EN) | `wikimedia/wikipedia` | 4B tokens | 事实知识 |
| | Wikipedia (ZH) | `wikimedia/wikipedia` | 1.7B tokens | 中文百科 |
| | WikiText-103 | `Salesforce/wikitext` | 103M tokens | 精选文章 |
| **问答** | Stack Exchange | `HuggingFaceH4/stack-exchange-preferences` | 10M Q&A | 专家问答 |
| **中文** | SkyPile-150B | `Skywork/SkyPile-150B` | 150B tokens | 中文网页语料 |
| | CCI3-Data | `BAAI/CCI3-Data` | 1TB | 智源中文互联网 |
| | CCI3-HQ | `BAAI/CCI3-HQ` | 518GB | CCI3 高质量子集 |
| **多语言** | FineWeb-2 | `HuggingFaceFW/fineweb-2` | 4.3TB | 1000+ 语言 |

下载方式见 [`scripts/download_pretrain_data.py`](scripts/download_pretrain_data.py)（支持按 `max_size_gb` 限制下载量），或直接运行 shell 包装脚本（hf-mirror 镜像已内置）：

```bash
# 全量下载（断点续传，中断后重跑即可）
bash data-process/scripts/download_pretrain_data.sh

# 后台运行 + 进度查看
nohup bash data-process/scripts/download_pretrain_data.sh > data/download.log 2>&1 &
bash data-process/scripts/download_status.sh

# 预检（不下载）
bash data-process/scripts/download_pretrain_data.sh --dry-run   # 列出全部文件与大小
bash data-process/scripts/download_pretrain_data.sh --test      # 每源下载 1 个文件验证可访问性
```

## 体量规划（2026-10-20 当前口径）

最初共获取 19 个候选源、约 15.3TB；当前 release scope 为其中 17 个源。2026-10-18
完成 ndv2 与中文去污染修复后的 S5 实测保留 3,105,637,476 篇、
2,530,570,633,678 tokens；正式 100B S6 物化 99,444,599,052 tokens。正式训练数据量只能从
通过 24 项 release gate 的 manifest 读取，不能再用早期容量估算代替产物统计。逐源配额见
[`configs/data_sources.yaml`](configs/data_sources.yaml)；受限源按 blake2b 路径哈希做确定性均匀抽样。

实际可下载量核查修正（2026-09-16）：①公开代码语料上限低于调研预期，fineweb 定稿 10TB（用户裁定，2.5T tokens）；②peS2o 取最新 v3（zst 格式）；③PG19 正文经 [`scripts/download_pg19.sh`](scripts/download_pg19.sh) 从 GCS 直采；④3 个 gated 源（the-stack-v2-dedup / MathPile / CCI3-Data）本机 token 未授权已移除，分别用 github-code / 数学余量 / cci3-hq 提额补偿——如需启用 gated 源，见 `data_sources.yaml` 文件头说明；⑤书籍缺口（PG19 仅 2.8B）由 dolma-books（OLMo Dolma v1.6 Gutenberg 全集 7.8GB，~9B tokens，olmo-data.org 直连）补齐，下载器支持 `urls:` 直连源。

当前正式 15 源 S6 范围包含 **2.447T reference tokens**。100B release 收集了 287.104B candidates，但其中 44.366B 来自四个 full-pool fallback 域，不能整体标为 top-10%；五个保持精确 top-10% 的域合计 242.738B candidates。主训练候选为 **1T final-tokenizer consumption**：50B warmup、800B stable、150B high-quality anneal；允许首尾阶段有审计过的高质量重放。清洗哲学保持严格门禁、低误杀，质量全量打分后按域取最高质量前缀。最终模型/数据预算仍由 E1 scaling 实验裁定。

## 流水线架构

```
JSONL 输入
  → DatasetAdapter 字段映射
  → 类别首步归一化
  → COMMON_PREFIX（NFKC 归一 / 完整性 / 来源 / 脱敏）
  → 类别专属步骤（code/web/agentic/instruction/math/reasoning）
  → COMMON_SUFFIX（质量策略 → 精确去重 → 近似去重 → 计长分桶 → 质量分桶）
  → 清洗后 JSONL + summary + manifest
```

### 三层核心抽象

1. **Sample + Decision**：全管线统一记录载体，三态决策 keep/review/drop（只降不升）
2. **Step + Pipeline**：流式执行引擎，DROP 即短路
3. **Adapter + Profile**：数据集专属规则以插件挂载，主干保持小而稳定

### Context Length 分桶（阶段性学习支持）

YuFormer 的创新点之一：**数据处理阶段就按可配置的上下文长度阶梯对样本分桶**，支撑 pretrain/midtrain 的渐进式长上下文训练。

在 `configs/cleaning_config.yaml` 中配置：
```yaml
context_lengths: [4096, 16384, 65536]  # 对应 pretrain stage 1/2/3
```

处理时每个样本被分配到 `B4K` / `B16K` / `B64K` / `gt_64K` 桶。配比阶段用**互斥前缀嵌套**（mix4K ⊂ mix16K ⊂ mix64K）构造各阶段的训练数据——短样本对更长阶段保持资格，但每条 ID 只进一个最终 mix。

这与 ZGCM-1 的 `mix16⊂mix64⊂mix256` 机制一致，但边界完全可配置。

## 使用

```bash
# 安装锁定的数据处理运行时；goldrule 工件会在加载时校验 numpy/sklearn 合同
python -m pip install -e data-process

# 清洗单个数据集
PYTHONPATH=src python -m yuformer_data --input data.jsonl --output cleaned.jsonl \
  --stage pretrain --category code --dataset my-code-corpus

# 带质量策略与去重配置
PYTHONPATH=src python -m yuformer_data --input data.jsonl --output cleaned.jsonl \
  --stage pretrain --category web --dataset my-web-corpus \
  --config configs/quality_policies.json --near-dedup

# 冒烟测试
PYTHONPATH=src python scripts/smoke_test.py
```

## 输出格式

每条输出记录（JSONL，`sort_keys=True`）：
```json
{
  "buckets": {"length": "1_4096", "quality": "main", ...},
  "category": "code",
  "data": {...},
  "dataset": "my-corpus",
  "decision": "keep",
  "flags": [],
  "metrics": {"estimated_tokens": 1024.0},
  "reasons": [],
  "record_sha256": "a1b2...",
  "source_id": "doc-001",
  "stage": "pretrain"
}
```

另输出 `*.summary.json`（per-step 计数器）和 `*.manifest.json`（从输出重建的决策计数）。

## 生产管线（S0/S1）

面向 19 源 ~15.4TB 原始数据的规模化摄取与硬过滤，实现于
[`src/yuformer_data/scale/`](src/yuformer_data/scale/)，逐源配置在
[`configs/ingest_profiles.yaml`](configs/ingest_profiles.yaml)（列名经真实文件抽样验证）。

### 源处置记录（2026-09-27 审计定案）

| 源 | 处置 | 依据 |
|---|---|---|
| c4 | **弃用**（E-16，2026-09-22） | S1 634 分片时代性失败；web 覆盖已由 fineweb+dclm 承担，重跑会作废 web 域 S2/S3 |
| pg19 | **弃用**（2026-09-27，scope 决策） | 纯 .txt 单书文件不兼容 ingest 框架（28,755 分片 S1 全 ValueError）；抽样书目（KJV Bible 等）在 dolma-books 中命中——内容高度重叠，修 ingest + 重跑 S1-S3 的成本 > 边际收益。books 域由 dolma-books 承担 |
| fineweb-2 | 保留（低影响记录） | 实测为 akp 等小语言圣经译本（低多样性宗教文本），仅 29 分片 3 千万行，multilingual 域贡献有限 |
| stack-exchange | **S1 清洗修复后重跑**（2026-09-27） | 正文残留 inline HTML（`<p>`/`<a>`/实体）；`detectors.clean_for_domain`（qa 域 strip + unescape，检测与写出同源），趁 decontam 未跑到该源完成 S1→S2→S3 重刷 |

### Release 门禁

正式训练不能直接引用某个 parquet 路径，必须引用状态为 `PASS` 且带 `dataset_id` 的 release
manifest。release spec 显式声明纳入与排除的 source lineage，并实现
[`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md) 的 24 项 A1–G2 门禁。

2026-10-19 的正式 100B release、100B train + 九域 validation token streams、snapshot-bound
6.5B tokenizer corpus 及两项独立全量审计均已 PASS。下游以
`data/PRETRAIN_DATA_READY.json` 为唯一终态入口，并要求其中四份证据的 dataset/snapshot
身份完全一致；旧 fast-track 和旧 E0-Q2 不属于本次正式交付。

```bash
# 当前 snapshot：逐源覆盖/行数守恒/error summary，并核对成功产物的 parquet footer
python3 data-process/scripts/audit_consistency.py \
  --processed-root data/processed \
  --scope data-process/configs/releases/mix_m3_e1_100b_fast.yaml \
  --out data/processed/audit_consistency_20261018.json

# 全量 C2 分源/分语言分数分布（Arrow fragment readahead + 固定直方图内存）
python3 data-process/scripts/audit_score_distributions.py \
  --s4-root data/processed/s4 \
  --scope data-process/configs/releases/mix_m3_e1_100b_fast.yaml \
  --workers 4 --files-per-task 64 \
  --out data/processed/audit_score_distributions_20261018.json

# G1/G2 来源版本、许可、PII 与 opt-out 签收（默认 fail-closed）
python3 data-process/scripts/audit_raw_source_completeness.py \
  --sources data-process/configs/data_sources.yaml \
  --scope data-process/configs/releases/mix_m3_e1_100b.yaml \
  --s0-root data/processed/s0 \
  --out data/processed/audits/raw_source_completeness_20261019.json

python3 data-process/scripts/recover_source_provenance.py \
  --sources data-process/configs/data_sources.yaml \
  --scope data-process/configs/releases/mix_m3_e1_100b.yaml \
  --raw-root data/pretrain-raw --s0-root data/processed/s0 \
  --out-dir data/processed/source_provenance/inventories \
  --out data/processed/source_provenance/recovered_source_provenance_20261019.json

python3 data-process/scripts/audit_source_governance.py \
  --governance data-process/configs/source_governance.yaml \
  --scope data-process/configs/releases/mix_m3_e1_100b_fast.yaml \
  --out data/processed/audit_source_governance_20261018.json

# 汇总 24 项证据；任一项非 PASS 时退出码为 2，且不生成 dataset_id
python3 data-process/scripts/release_gate.py \
  --spec data-process/configs/releases/mix_m3_e1_100b_fast.yaml \
  --out data/processed/releases/mix_m3_e1_100b_fast-20261018.release.json
```

C2 当前证据全量覆盖 12,829 个 S4 分片、6,485,760,234 行、17 来源和 203 个语言标签；
119 个达到 1,000 行门槛的语言全部健康，中文/英文域内 rank 中位数为 0.5040/0.4994。
每个 task 只物化 64 个文件的三列投影，避免把十亿行数据或全文列装入内存。

`--skip-output-footers` 只用于快速诊断，不得作为 release 证据。即使被排除源存在历史失败，
审计报告仍会记录其数量与排除理由；纳入源的任一失败、缺阶段、分片不一致、行数不守恒、
产物缺失/损坏都会使审计非零退出。

`release_gate.py --hash-blocked-artifacts` 可在门禁尚未全过时生成稳定的 `snapshot_id`，用于
E4 token 物化与对拍；它不会生成 `dataset_id`。`snapshot_id` 只绑定 source lineage 与数据
artifact 内容，门禁状态改变时保持稳定；全部 24 项 PASS 后才生成同时绑定 snapshot 与证据的
`dataset_id`。这样避免了 E4 需要先物化 token，而 token 又需要绑定数据身份的循环依赖。

### 用法

`download_pretrain_data.py` 现在默认要求 HF 源在 `data_sources.yaml` 中声明不可变
commit revision，直连 URL 必须声明逐文件 SHA-256。报告保留 resolved commit、确定性
选中文件清单摘要与逐文件状态；direct-URL 文件还记录实测字节数与内容 SHA-256。
`max_size_gb` 是声明过的确定性抽样边界，成功取得全部选中项时状态为 `ok`，并另以
`selection_limited=true` 表明没有选择仓库全集；`partial` 只表示选中项存在下载失败。
命令只有在所有选中源均为 `ok` 时返回 0；任一 `partial/failed/empty` 都返回 2，供编排器
fail-closed 停止下游处理。
`--allow-floating-revision` 只能用于访问性探测，
该模式产物不得进入 release。`--all` 默认跳过 `enabled: false` 的退役源；只有显式传入
`--include-disabled` 才能运行它们，该 override 同样不得自动进入正式 release。现有
2026-09-16 报告没有 revision，不能事后填写今日 commit
冒充历史版本。

来源门禁分成两个独立证明：`recover_source_provenance.py` 证明实际进入 S0 的每个文件
来自哪个固定 commit/etag（直连源用内容 SHA-256）；`audit_raw_source_completeness.py`
重新查询该固定 revision 的预期选择清单，并与 S0 的 path/bytes 闭合，证明下载没有静默
截断或漏分片。正式 release 的纳入源必须两者都通过；已知不完整源只能带明确理由进入
`audit_scope.excluded_sources`，不能由“本地文件存在”放行。

```bash
cd data-process
PYTHONPATH=src /usr/bin/python3 -m yuformer_data.scale.driver \
  --stage s0 --workers 160 --data-root ../data/pretrain-raw --out-root ../data/processed
# S1（依赖 S0 产物）：
... --stage s1
# 全链路 / 冒烟：
... --stage all --sources wikitext dolma-books proof-pile-2 --limit 2
```

- 语言检测模型（可选）：`export YUFORMER_LID_MODEL=/path/to/lid.176.bin`；未设置时 `lang='unk'` 且 per-shard summary 记 `skipped_steps: ["lang"]`。
- 启动 fail-fast 自检：pyarrow/zstandard 可导入、data-root 存在、profile 与磁盘源目录对得上（`optional: true` 的源除外）。

### 输出布局

```
data/processed/
├── s0/<source>/<原相对路径>.parquet        # canonical schema: doc_id,text,domain,source,lang,upstream_score,meta
├── s0/<source>/<原相对路径>.summary.json
├── s1/<source>/<原相对路径>.parquet        # 同上 + decision,reasons,metrics（keep/review 正文）
├── s1_audit/<source>/<原相对路径>.parquet  # 仅 drop：doc_id,reasons,text_head(200字符)
├── s1/<source>/<原相对路径>.summary.json
└── stage_<s0|s1>_global_summary.json       # 全 shard 聚合 + per_source/per_domain/reason 计数
```

### 断点续跑语义（文件级幂等）

每输入文件 = 一个 shard 任务。shard 的 parquet + summary.json 同时存在即跳过（重跑零成本、输出字节一致）；shard 级失败只写 error summary 不中断全局，且因输出缺失会在下次运行自动重试。输出先写 `*.tmp-<pid>` 再 `os.replace` 原子落盘，并发写 shard 级隔离、无共享状态。shard 列表按 (source, relpath) 排序，doc_id = blake2b-16(source+relpath+row_idx)，全程确定性。

### S1 硬过滤规则（`scale/detectors.py`，纯函数）

NFKC 归一/控制字符清理/空白折叠/空行压缩 → 完整性（<16 字符 drop、>2M 字符 review）→ 乱码（U+FFFD>1e-4、Latin-1 mojibake 双字符签名、控制字符残留>1e-3 → drop）→ HTML 源码（标签占比>5%、script/style 块、实体密度>1%、base64/data-URI blob>8KB → drop）→ 敏感串（api_key/secret/password/access_token → review）→ fasttext 语言检测 → 领域规则（web 行级重复率>0.55 drop；code 路径含 node_modules/vendor/.min.js/.map drop）。三态 keep/review/drop，drop 不进正文输出只进 audit。

### 实测吞吐（director env，2026-09-17，单机 176 核中的一核，下载任务同机并行有干扰）

| 阶段/负载 | 实测 | 目标 |
|---|---|---|
| S0 parquet（fineweb 2.1GB shard） | 37–45 MB/s/核 | ≥20 MB/s/核 ✅ |
| S0 gzip 流式（dolma-books） | 15–27 MB/s/核 | — |
| S0 zst 流式（proof-pile-2） | 12.5 MB/s/核 | — |
| S1 短行（wikitext，~0.5KB/行） | 2.6–3.9 万行/s/核端到端（检测器本体 7.5 万行/s） | 5 万行/s/核（检测器本体已达标；端到端受 parquet 读写与 JSON 序列化限制） |
| S1 网页行（fineweb，~2KB/行） | ~5,100 行/s/核 ≈ 15 MB/s 文本 | 160 并发 ≈ 80 万行/s 聚合 |
| S1 长文书籍行（dolma-books，~70KB+/行） | ~22 行/s/核 ≈ 2 MB/s 文本 | 长文按文本量计费，聚合足够 |

S1 三态实测（5 shard 样本）：keep 2,197,837 / drop 6,885（0.31%，其中 fineweb 网页 0.32%）/ review 1,793。drop 主因：`empty_or_short`、`html_tag_ratio`、`garbled_*`；review 主因：`too_long`（>2M 字符）、`secret_pattern`。逐 shard 明细见 `data/processed/*/​*.summary.json` 与 `stage_s*_global_summary.json`。

### 测试

```bash
PYTHONPATH=src /usr/bin/python3 -m pytest tests/test_scale.py -q
```

detectors 正负用例、doc_id 确定性、合成 parquet/gz/txt 映射、幂等性（二次全 skip 且字节一致）、3 个真实文件端到端 smoke。

## 生产管线（S2/S3，去重）

S2 精确去重（sha256 归一化文本）+ S3 MinHash-LSH 近重去重（5-gram + 128 排列 + 16 band×8 行 +
真实 Jaccard≥0.8 验证），实现于 `scale/dedup_exact.py` / `scale/dedup_minhash.py`，
共享库 `scale/dedup_common.py`，参数在 [`configs/dedup_config.yaml`](configs/dedup_config.yaml)
（源优先级表、桶数、阈值、内存上限全配置化）。域内跨源（web 三源互去、代码两源互去）。

### 用法

```bash
cd data-process
# S2（依赖 s1/；隐式依赖检查，缺产物 fail-fast 提示）：
PYTHONPATH=src /usr/bin/python3 -m yuformer_data.scale.driver \
  --stage s2 --workers 160 --out-root ../data/processed
# S3（依赖 s2/）：
... --stage s3
# s2+s3 链：
... --stage dedup
# 指定源冒烟：
... --stage dedup --sources fineweb fineweb-edu dclm-baseline --limit 5
```

### 输出布局

```
data/processed/
├── s2_hash_parts/<domain>/<source>/<relpath>.parquet   # S2 pass1: bucket,h0..h3(sha256),d0d1(doc_id),source（排序）
├── s2_decisions/<domain>/<bucket4>.parquet             # S2 pass2: 仅 drop（doc_id, decision=drop_exact_dup, dup_of）
├── s2_lookup/<domain>/{lookup.ids.bin,lookup.codes.bin}# 排序 drops 查找表（memmap+searchsorted）
├── s2/<source>/<relpath>.parquet                       # s1 全列 + s2_decision（keep/drop_exact_dup 打标）
├── s3_sig/<domain>/<doc_id前缀2hex>/<source>/<relpath>.parquet   # (doc_id, sig 512B)
├── s3_lsh/<domain>/b<band>/<source>/<relpath>.parquet  # band part：bucket,k0,k1(k0k1=band_key),d0d1
├── s3_pairs/<domain>/b<band>/<bucket>.parquet          # 候选对（a0a1,b0b1 = doc_id uint64 对）
├── s3_cand/<domain>.npy                                # 全局唯一候选 doc_id 集合（V16 排序）
├── s3_pairs_bydoc/<domain>/<byte>/chunk-*.parquet      # 候选对按 doc_id 首字节重分区
├── s3_cand_parts/<domain>/<source>/<relpath>.parquet   # 候选正文 (doc_id, source, text)
├── s3_store/<domain>/{ids,srcs,offs,texts}.bin         # 候选正文外存 docstore（memmap 随机访问）
├── s3_verified/<domain>/<byte>.parquet                 # Jaccard≥0.8 验证通过的对
├── s3_decisions/<domain>.parquet                       # 仅 drop（doc_id, decision=drop_near_dup, dup_of）
├── s3/<source>/<relpath>.parquet                       # s2 全列 + s3_decision
├── s3_v2/packed_csig4/<domain>/<byte>.sig4.bin         # 候选文档 128×4-bit MinHash 索引
├── s3_v2/verified/<domain>/<byte>.parquet              # 经校准签名阈值通过的边（发布前另做全文精度门）
├── s3_v2/ctext/<domain>/<byte>/{ids,srcs,offs,texts}.bin  # ndv2 flat byte bin（ids 排序，a 侧顺序切片 / b 侧 memmap 点查，零行组放大 E-20）
├── s3_decisions/<domain>.e17_backup.parquet            # ndv2 覆盖前 E-17 全 keep 占位备份
└── stage_<s2|s3|ndv2|ndv2_refresh>_global_summary.json # 从完整发布 roster 重建的全局健康与计数
```

### 断点续跑语义

与 S0/S1 同构（每任务 parquet+summary 齐在即 skip、tmp+os.replace、任务失败容错计数、
50 任务进度日志、sort_keys JSON summary）。S2/S3 的决定文件带 `domain_sources` 守卫：
部分源重跑后扩全量时自动重算决定，不会误用旧决定。内存安全：所有"单分区加载"都有
`memory_cap_gb` 上限，超限抛 `MemoryCapExceeded` 落 error summary（下次自动重试），不许 OOM。

### 实测吞吐（director env，2026-09-17，**与 S1 全量运行同机，测量全程 160 核被占满**，数字偏保守）

| 指标 | 实测 | 说明 |
|---|---|---|
| S3 pass1 签名（dclm web ~2.6KB/doc，CPU 时间） | 2.8 MB/s/核文本（458k shingles/s/核，494 docs/s/核） | 内存带宽被 S1 全量抢占；隔离微基准单 shingle 哈希 1.8M/s/核（≈12 MB/s/核当量） |
| S3 pass1 签名（dolma-books ~89KB/doc，CPU 时间） | 2.1–3.3 MB/s/核文本 | 长文 shingles/doc 多，固定开销摊薄 |
| S2 pass1 哈希 | sha256 流式，>100 MB/s/核 | 非瓶颈 |
| 真实分片端到端 | cci3-hq 565MB + dclm-baseline 244MB + fineweb 445MB，s2→s3 计数闭合（见 `tests/test_dedup.py`） | 满载下 14 min |

目标 ≥5 MB/s/核文本：隔离单核可达（shingle 哈希微基准 1.8M/s/核），同机满载端到端实测
2.1–2.8 MB/s/核（CPU 时间口径）。签名阶段 shard 级 embarrassingly parallel（160 worker
线性聚合 ~450 MB/s 文本），全量 5B 文档签名预估数小时级；**全量跑前建议在空闲核上复测**。

### 测试

```bash
PYTHONPATH=src /usr/bin/python3 -m pytest tests/test_dedup.py -q
```

六项：①跨 shard 精确去重（规范化等价 + survivor 确定性两次一致）；②MinHash J≈0.85 判重 /
J≤0.55 不判重 / 128 排列字节级确定；③并查集链式 A~B~C 聚一簇；④全链幂等（重跑全 skip 且
输出字节一致）；⑤3 个真实 s1 分片端到端 s2→s3 计数闭合；⑥内存上限保护（超限报错非 OOM）。

## 生产管线（S3-v2，web/code 跨源近重）

```bash
PYTHONPATH=src python -m yuformer_data.scale.driver \
  --stage ndv2 --out-root ../data/processed --workers 48
# 可选兼容维护：仅当外部消费者仍直接读取 S3 的 s3_decision 列时才需要
PYTHONPATH=src python -m yuformer_data.scale.driver \
  --stage ndv2_refresh --out-root ../data/processed --workers 64
```

正式路径复用 S3 的 128-permutation MinHash 候选。候选文档签名压成 4-bit/permutation
（64B/doc）的按首字节索引；候选对按 256 桶去重后，以有限样本修正后的签名一致数验证。
当前阈值为 108/128（配置点估计 0.83）。该路径避免旧全文验证对网络文件系统做海量稀疏
正文读取；精确全文 5-gram Jaccard 保留为质量门和小规模审计实现。

108/128 是精度优先的强近重复门，并不等价于“Jaccard≥0.8 全召回”。按 b-bit MinHash
碰撞概率精确计算，边界 J=0.80 的理论通过率为 21.66%，J=0.85 为 74.34%，J=0.90 为
99.18%。发布门在精度审计之外还要求 J=0.85 理论通过率 ≥70%；因此本阶段的准确表述是
稳定删除强近重复，不能把未通过的候选解释为非重复样本。

近似验证不得直接发布删除结果。每个域会从已通过边做 1,024 对确定性全局系统抽样，
回取全文并计算精确 5-gram Jaccard；点估计精度必须 ≥97%，95% Wilson 下界必须 ≥95%。
审计合同绑定边集指纹、阈值和采样方法，失败或过期都会阻止 decide/S5。code 生产实测为
995/1,024 精确阳性，precision=97.17%，Wilson lower=95.96%；web 为 1,013/1,024，
precision=98.93%，Wilson lower=98.09%。

决定阶段使用 16 路外排 `edge_dominance_external`：每条通过边按
`(source_priority, doc_id)` 选择较差端点，文档只有直接输掉一条验证边才会被删除。这里有意
不做连通分量传递闭包，避免少量近似假阳性桥把两个本不相似的大簇级联误删。每个分区
归并同一 loser 的最佳直接 winner，再流式写 decision parquet 和排序 lookup；内存与全域
边数解耦。code 生产对照：103,099,673 条通过边，直接边策略删除 39,220,532 篇；旧
connected-components 删除 40,214,080 篇，多删 993,548 篇。新路径峰值 3.3GB（旧路径
18.4GB），411 秒完成。

S5 直接 join 已排序的 `s3_lookup/<domain>/lookup.ids.bin`；S6 门禁同时核对 lookup 字节数、
decision summary、全文精度审计的边集指纹与 ndv2 全局 dropped 数。因此正式路径不再要求把
相同决定重写进约 10TB 的 S3 文本分片。`ndv2_refresh` 保留为兼容维护命令：它以 8K 行 batch
流式刷新 `s3_decision` 列，供仍读取该列的外部消费者使用。S5 另绑定 decontam、tokenizer 与
context 配置，任何上游换代都会自动失效或进入带证明条件的精确增量路径。

测试：

```bash
PYTHONPATH=src pytest -q tests/test_dedup_v2.py
YUFORMER_RUN_REAL_DATA_TESTS=1 PYTHONPATH=src pytest -q tests/test_dedup.py tests/test_scale.py tests/test_score.py
```

多 GB 真实数据集成/吞吐测试显式 opt-in；常规单元回归不会扫描生产分片。

## 生产管线（S4，质量打分）

S4 = 全池质量打分：自训 fasttext 质量分 + 上游分双轨复合 + 领域内分位数，实现于
`scale/scorer_train.py` / `scale/score.py`，配置
[`configs/scorer_goldset.yaml`](configs/scorer_goldset.yaml)（gold-set 正负样本配额）与
[`configs/scorer.yaml`](configs/scorer.yaml)（复合权重、per-source 上游分归一表、分位数精度）。
**2026-09-20 新增 BERT 回归档（ensemble 第二 learned 轨）**：`scale/bert_train.py` /
`scale/score_bert.py`，三轨融合 `composite_v2 = 0.25·upstream_norm + 0.35·fasttext +
0.40·bert`（权重与降级开关在 scorer.yaml `composite_v2` / `bert` 节）。

### 用法

```bash
cd data-process
# 1) 训练打分器：gold-set 构造（确定性水库抽样）→ 留一源交叉验证 → 训练 + 量化
PYTHONPATH=src /usr/bin/python3 -m yuformer_data.scale.scorer_train \
  --config configs/scorer_goldset.yaml --out-root ../data/processed --model-dir ../data/models
#    --step goldset|cv|train|all 可单独执行；产物 gold.parquet / cv_metrics.json /
#    yuformer_quality_scorer.{bin,ftz} + 各 summary（sort_keys JSON）
# 1b) BERT 回归档：同一 gold-set 与留一源 CV 划分（门禁 mean≥0.85 且每源≥0.75），
#     产物 data/models/yuformer_quality_bert/ + data/models/bert_scorer_cv_metrics.json
HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src /usr/bin/python3 \
  -m yuformer_data.scale.bert_train --config configs/scorer_goldset.yaml \
  --out-root ../data/processed --model-dir ../data/models
#    --step goldset|cv|train|all；单卡即可（数据 ~1M×256 token 很小）
# 2) BERT 全池 GPU 推理（前置：--stage dedup 已产出 s3/；每进程一卡，默认全部可见 GPU）
PYTHONPATH=src /usr/bin/python3 -m yuformer_data.scale.driver \
  --stage s4_bert --out-root ../data/processed --gpus 0 1 2 3
# 3) 推理（s1 → s4_scored → 分位数表 → s4 全列）
PYTHONPATH=src /usr/bin/python3 -m yuformer_data.scale.driver \
  --stage s4 --workers 160 --out-root ../data/processed
#    --stage score = 只跑打分 pass（不落 domain_quantile）；--sources/--limit 同 S0/S1
#    s4 打分会自动 join s4_bert 产物：bert 产物出现后同一 shard 自动重跑
#    （summary 的 bert_joined 标记驱动）；bert 整轨缺失时 composite_v2 退化为双轨。
# 已有分片只需重建正式选择分布时，不重跑任何模型推理：
PYTHONPATH=src /usr/bin/python3 -m yuformer_data.scale.driver \
  --stage s4_build --score-field hybrid_v2 --force --out-root ../data/processed
# 默认用 8 个有界读取线程隐藏冷 NFS 延迟；可用 YUFORMER_S4_BUILD_WORKERS 调整。
# 任一不可读分片会使构建失败，不会静默跳过；summary 只在全部 NPY 发布后原子切换。
```

### 输出布局

```
data/processed/
├── s4_bert/<source>/<relpath>.parquet          # BERT 轨: doc_id + bert_score（量化 1e-4）
├── s4_scored/<source>/<relpath>.parquet        # s1 列 + self_score, composite,
│                                               #   upstream_norm, fasttext_score, bert_score,
│                                               #   composite_v2（float32）
├── s4_quantiles/pairs/<source>/<relpath>.parquet  # pass1: (domain, composite_v2) 全量
├── s4_quantiles/tables/<domain>.<build_id>.npy # pass1 聚合: 不可变 per-domain 排序数组（float32）
├── s4/<source>/<relpath>.parquet               # 最终: 上 + domain_quantile（0-1，线性插值，量化 1e-6）
└── stage_s4_global_summary.json                # 分布直方图/源、三轨分 Spearman 相关、吞吐、模型 md5
```

### 关键语义

- **复合分**：`composite = 0.5·normalize(upstream) + 0.5·self_score`；上游分缺失时
  `composite = self_score`；normalize = per-source min/max 线性归一（fineweb-edu 0-5、
  dclm 0-1，见 scorer.yaml）。
- **正式选择分 `hybrid_v2` 的缺失语义**：upstream 缺失时移除其权重，并在
  `llm_distill + rule_score` 上重新归一。2026-10-18 复审发现旧实现把缺失错误编码为
  有效 0 分；旧分片通过 `legacy_zero_upstream_v1` 合同做解析修复，并且必须先通过
  `s4_hybrid_v2_legacy_audit.json` 的组件级精确复算。分位表和 S6 都消费修复后的有效分，
  禁止直接读取旧列值作为正式选择分。
- **三轨融合 composite_v2**（2026-09-20）：`0.25·upstream_norm + 0.35·fasttext_score +
  0.40·bert_score`，按行重归一（某轨缺失其权重置 0，其余轨按原比例归一）；bert 整轨
  缺失自动退回双轨 composite（数值与 legacy composite 一致）。分位数排序键 = composite_v2。
  **GPU 数值不确定**：bert_score 量化到 1e-4，重跑 ±1e-4 抖动属预期（权重字节级 md5
  亦不稳定，质量门以 CV 指标为准，不以权重哈希为准）。
- **截断策略（2026-09-20 裁定，`window_scoring`/`long_doc_chars`，两轨共用）**：短文档
  （归一化 ≤1024 字符）头部 max_len 直打；长文档（> long_doc_chars，books/academic/code）
  head/mid/tail 三窗各 max_chars 字符（fasttext）/ 256 token（BERT）均分。BERT 长文
  归一化上限 32k 字符、token 上限 8192（超长尾文档 mid 窗为前 8192 token 中点，确定性
  近似）；三窗均值与 head-only 存在 ~1e-3 级系统性差（窗尾无 SEP），校准脚本量化裁定。
- **截断校准**：`PYTHONPATH=src python -m yuformer_data.scale.truncation_calibration \
  --out-root ../data/processed --model-dir ../data/models/yuformer_quality_bert`——确定性
  分层抽 5,000 篇（短/中/长 40/40/20%），算 head-only / 三窗 / full-doc(max_len 512)
  三种 bert_score，Spearman 相关矩阵 + 判定建议（head vs full ≥0.9 → 全池 head-only；
  0.75-0.9 → 长文三窗；<0.75 → 全池 max_len 512 三窗）写
  `data/models/truncation_calibration.json`。
- **分位数**：两遍法。pass1 打分时顺手落 (domain, composite) 对；聚合后离线排序成
  per-domain 数组；pass2 每 shard 用 searchsorted + 线性插值标注，同值同分（决胜无关），
  量化到 1e-6。分位表是快照——s1 增量产出后需重建表再跑 pass2（`build_quantile_tables(force=True)`）。
- **质量门**：留一源 CV（每次 holdout 一个正样本源，其余源训练、holdout 上测 AUC/PR），
  目标 mean AUC ≥0.85；不达标时 `cv_metrics.json` 写入负样本扩充建议，不静默接受。
- **训练隔离**：fastText 0.9.x 在同一 Python 进程反复 fit 会出现 native 状态相关 NaN；
  final fit 与每个 CV fold 都在带显式 seed 的全新子进程训练、保存，再由 driver 只读加载。
  summary 记录 `training_process_isolation`，不得用降低 LR 的偶然绿灯替代进程隔离。
- **效率**：fasttext 批量 predict（batch 1000）、列式 parquet 读写、模型 .ftz 量化；
  推理 pass 与量化标注 pass 均文件级幂等（parquet+summary 齐在即 skip，tmp+os.replace）。
- driver fail-fast：s4/score 需要 s1 产出与已训练模型（缺模型提示先跑 scorer_train）。

### 实测吞吐（director env，2026-09-17，单机 176 核中 pinning 单核，S1 全量 driver 同机并行有干扰）

| 负载 | 实测 | 备注 |
|---|---|---|
| 打分 pass 端到端（dclm-baseline，~5KB/篇，截断 4096 字符） | ~0.6–1.7 千篇/s/核 | predict 占 ~80% |
| 分解：parquet 读 / 预处理 / fasttext predict / 写 | 8.1k / 4.8k / 0.82k / — 篇/s | dclm 74k 篇 shard |
| 量化标注 pass（s4_scored → s4） | ~3.2 千篇/s/核 | searchsorted 插值，纯 CPU |
| **BERT 推理（2026-09-20，4×H800，合成 ~2KB/篇英文 = 256 token 顶格）** | **7.4k 篇/s 聚合（1.85k/GPU 端到端）** | 同机 160 核 dedup 高负载下实测 |
| BERT 三窗版（同上，15% 长文混合语料，单卡对比） | 2038 vs 2725 篇/s（**开销 ~1.34×**） | 与 1.3× 预估一致；真实池长文占比 ~10-15% |
| 分解：纯 GPU forward（fp16，bs512，预 tokenized） | ~7.9k 篇/s/GPU | 上限参考 |
| 分解：单线程 tokenize | ~4.2k 篇/s | 已用 Rayon 8 线程 + 预取流水重叠 |

**关于 BERT 档 ≥6 万篇/秒聚合硬指标（未达标，量化分析）**：瓶颈是 FLOP 墙而非工程开销。
mBERT-base 全池 forward ≈ 2×110M 参数×256 token ≈ 56 GFLOP/篇；4×H800 实测 matmul
682 TFLOPS、forward 有效 MFU ~40-50%，单卡上限 ~12k 篇/s（100% MFU 理论 12.1k），
**4 卡物理上限 ~48k < 60k**。已做的优化：纯 fp16 加载（比 AMP autocast 快 33%）、
tokenizer Rayon 多线程 + encode 预取流水与 GPU 重叠、按 batch 最长动态 padding
（真实池短文档多，预计再省 1.3-1.8× FLOP）、`torch.compile` 开关（实测无增益，留作
TensorRT 前置）。高负载机端到端 7.4k 聚合推算全池（~35 亿 keep 篇）≈ 130 墙钟小时，
即使空机 GPU 打满（~30k 聚合）也需 ~32 小时，均超 16 小时预算——**需用户裁定的降级
路径**（配置均已就绪）：① `bert.domain_allowlist` 只扫 web 域；② 缩小 max_len
256→128（FLOP ~2×，信号损失待评估）；③ INT8/TensorRT（再 ~2×）；④ 加卡。
bert_score 重跑 ±1e-4 抖动属预期（量化步长，见「关键语义」）。

**关于 ≥2 万篇/秒/核 硬指标（未达标，量化分析）**：瓶颈是 fasttext 推理本体
O(tokens×dim)——dim=100 + wordNgrams=2 对 ~700 token 的英文网页篇，健康单核预测上限
~2.5–4k 篇/s，预处理+IO 再加 ~30%。规格给的优化路径（ftz 量化 / 更长 batch / 多线程）
只削固定开销，改变不了 token×dim 主项（实测 ftz 反略慢于 bin：2540 vs 3533 篇/s；
线程扩展在 64 核限额+高负载下 1→2 线程仅 1.6x）。要达硬指标只能动规格参数：
`max_score_chars` 4096→1024（~2.5x）与/或 `dim` 100→50（~2x），两者皆为配置项，
但属规格变更需另行裁定。中文篇（无空格分词，token 少）实测快 3-4x。
160 核聚合 5B 篇的墙钟预算（8.7k 篇/s/核）同上受限。

### 测试

```bash
PYTHONPATH=src /usr/bin/python3 -m pytest tests/test_score.py tests/test_score_bert.py -q
```

`test_score.py`：gold-set 比例与确定性、3 微源 CV 结构、fasttext thread=1 训练 md5 一致、
composite/分位数手工数值断言、幂等（二次全 skip 字节一致）、真实 s1 分片端到端吞吐
断言 ≥2 万篇/秒/核（当前高负载机上未达标，见上表分析）。
`test_score_bert.py`（8 项；GPU 不可用自动 skip，CPU 只剩 composite_v2 融合语义测试）：
冒烟训练 loss 下降 + 保存/重载往返、final 训练 summary、推理数值（bert_score ∈[0,1]
且量化 1e-4 + 好文>垃圾分）、shard 级幂等（二次全 skip 字节一致）、缺 s3_decision
列报错提示先跑 dedup、domain_allowlist 过滤、score pass 三轨融合 + bert 产物出现后
自动重算。

### 当前模型状态（2026-09-17  interim）

`data/models/yuformer_quality_scorer.{bin,ftz}` 由**开发版** gold-set 训练
（正样本 dolma-books 20k + cci3-hq 高分段 20k；fineweb-edu/wikipedia/pes2o 的 S1
尚未产出，配 `max_shards` 无偏选 shard）。留一源 CV：holdout dolma-books AUC 0.80 /
holdout cci3-hq AUC 0.43（**未达 0.85 目标**）——根因是英语单源正样本导致语言偏置
（分类器学到"英文=hq"，中文篇系统低分），质量门正确拒收。in-sample AUC 0.9995 仅供参考。
**待 S1 全量（含 wikipedia-zh/fineweb-edu/pes2o）后按 configs/scorer_goldset.yaml
全量配额重训并复检 CV，通过前不要用于正式配比。**

BERT 档（2026-09-20 interim）：dev 规模留一源 CV（scorer_goldset_dev 确定性子采样
6k 行，2 正源）mean AUC 0.905 / min fold 0.883 ✅ 过门（仅验证流程，无中文源、
非正式 gold-set）。正式 CV（写 `data/models/bert_scorer_cv_metrics.json`）与全量
训练/推理按 2026-09-17 时序锁定等 S3 完成后统一排期；GPU 训练权重 md5 不稳定属预期
（质量门以 CV 指标为准）。

## 生产管线（S7-decontam，评测去污染）

GPT-3/LLaMA 同类 n-gram 重叠去污染（S6 物化硬前置门）：训练文档与基准测试项
（MMLU/GSM8K/MATH/HumanEval/C-Eval 的题面+选项+答案）共享 **8-gram**。默认一个不同
shingle 即 drop；C-Eval 的双 shingle 门只对 `cci3-hq`、`skypile-150b`、
`wikipedia-zh` 三个已人工审计的中文源生效，其他源保持单命中以免降低召回。
实现于 `scale/decontam.py`，配置 [`configs/decontam.yaml`](configs/decontam.yaml)，
方法/阈值依据/实测明细见 [docs/PIPELINE_DESIGN.md](docs/PIPELINE_DESIGN.md) S7-decontam 节。

### 用法

```bash
cd data-process
# 一键：构建基准索引（幂等，下载 5 基准 test 集 ~15MB，一次即可）→ 分片并行扫描：
HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src /usr/bin/python3 \
  -m yuformer_data.scale.driver --stage decontam --workers 160 --out-root ../data/processed
# 冒烟（前 N 个分片）：
... --stage decontam --limit 5
# 只扫部分源：
... --stage decontam --sources fineweb github-code
```

前置：input_stage（默认 `s3/`，配置项 `input_stage`）产物已存在。索引位置由配置
`index_dir` 决定（默认 `../data/decontam_index`，外部资产不入库）；基准源 2026-09-25
逐个经 hf-mirror 验证（5/5 可用，69/69 路径 OK），构建日志与 revision sha 记录在
`index.meta.json`。

### 输出布局

```
data/processed/
├── decontam/<source>/<relpath>.parquet   # 决策表（S6 join 用）: doc_id, decision, hit_benchmark
├── decontam/<source>/<relpath>.summary.json  # per-shard: 命中数/各基准分布/截断数/吞吐
└── stage_decontam_global_summary.json    # shard_errors 置顶 + per-benchmark 命中统计 + per_source

data/decontam_index/
├── index.hashes.bin / index.codes.bin    # 排序去重 8-gram 哈希（<u8/uint8，memmap 共享 page cache）
├── index.meta.json                       # 基准名/revision/逐路径可用性/per-benchmark shingle 数
├── index.summary.json                    # config echo（配置变更自动重建）
└── repo_cache/                           # 基准 parquet 缓存（hf_hub_download）
```

### 关键语义

- **达到各基准 `min_distinct_hits` 才 drop，决策打标不过滤**：`decision ∈ {keep, drop_contaminated}`，命中基准名
  落 `hit_benchmark`（多基准命中取配置序最前者）；S6 物化时 join 本表过滤。
- **归一化与 S2/S3 同源**（`normalize_for_hash`，H3）；中文 CJK 逐字 token（否则中文
  整句=1 词，8-gram 永不命中，C-Eval 会整体漏检）；ASCII 纯文本走等价快速路径
  （NFKC 对 ASCII 恒等，测试断言逐字节一致）。
- **哈希**：词→FNV-1a-64 id、词组→多项式 rolling（mod 2^64），numpy 向量化
  （blake2b 无法向量化达不到全池吞吐目标；64-bit 空间假命中期望 <1e-4，记录在案）。
- **全文扫描**：`max_shingles_per_doc: null`；旧版只扫前 50,000 个词组的前缀策略已于
  2026-10-19 撤销。正式 mix 抽样在长书尾部发现 benchmark 重叠，证明前缀截断会产生
  系统性 false negative。S1 的 `max_chars=2M` 提供单文档内存上界，索引查询再按 2M
  shingles 分块，避免 parquet 批次内长文档累计造成内存峰值。
- **断点续跑与事故账本**：input fingerprint、config SHA 与 index SHA 全部一致才 skip；失败
  分片只落 error summary、空产物不落盘、下次自动重试（E-16）；停滞守卫（E-1）、
  索引 memmap 全 worker 共享 page cache（E-13/E-14）；全局 summary 的 `shard_errors`
  置顶，验收第一条检查是否为 0。
- **索引/扫描合同分离**：benchmark 文本、归一化与 n-gram 参数决定索引身份；逐源命中阈值
  只决定扫描身份。阈值修正不会无意义重建相同索引，但会精确失效受影响来源的决策分片。
- **单调阈值增量重算**：当某个 benchmark 的 `min_distinct_hits` 只增不减时，旧 keep 与其他
  benchmark 的 drop 决策保持不变，只对旧 `hit_benchmark` 等于该 benchmark 的行重新计算。
  每批仍按 doc_id 精确对齐，并闭合输入、决策与输出账本。三中文源的 C-Eval 阈值修复因此在
  约 25 分钟完成，当前全局为 12,829 分片、6,485,760,234 行、46,044,928 个原始污染决定、
  `shard_errors=0`，并救回约 21.557M 个旧误杀文档。

### 实测吞吐（director env，2026-09-25，单核；测量时机器 BERT 推理+fasttext 训练
占 ~130 核，数字偏保守）

| 负载 | 实测 | 备注 |
|---|---|---|
| 短文档（~0.7KB/篇，~100 词） | ~1.9 万行/s/核 |  shingles ~1.9M/s/核 |
| 典型文档（~2KB/篇，~300 词） | 6.9k–8.5k 行/s/核 | 2.1–2.6M shingles/s/核 |
| 全池外推（36.5 亿篇 / 13.9TB） | **瓶颈是共享存储 IO（E-9：0.3–0.5GB/s 聚合）** | 按 0.4GB/s ≈ **10 小时墙钟**；160 核计算裕量 >10× |

### 测试

```bash
HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=src /usr/bin/python3 \
  -m pytest tests/test_decontam.py -q
```

合成基准注入端到端：已知重叠必命中且基准名正确、7-gram 边界不命中、归一化等价
（大小写/多空格）、幂等（二次 skip 字节一致）、超长文档截断语义（截断内命中 drop /
截断外 keep）、失败分片空产物不落盘、跨进程/清缓存确定性、`_fast_normalize` 与
`normalize_for_hash` 逐字节一致；真实基准下载测试（gsm8k+humaneval 全链路）在网络
不可达时自动 skip。


## 生产管线（S5，计长分桶，2026-09-26 实现）

S3 keep 池 → 自训 yf-64k-bpe 精确计长 → B4K/B16K/B64K/gt_64K 互斥分桶 +
前缀嵌套资格标记。实现于 `scale/length_bucket.py`，driver `--stage s5`。

### 用法

```bash
cd data-process
PYTHONPATH=src /usr/bin/python3 -m yuformer_data.scale.driver \
  --stage s5 --workers 160 --out-root ../data/processed
# 冒烟 / 单源 / 自定义 tokenizer（默认 YUFORMER_TOKENIZER 或仓库 tokenizers/yf-64k-bpe）：
... --stage s5 --limit 5
... --stage s5 --sources fineweb github-code
... --stage s5 --tokenizer /path/to/yf-64k-bpe
```

### 输入与决策 join（三路过滤，全部落审计统计）

输入 = `s3/<src>/<rel>.parquet` 全行，按 doc_id 左连：
1. `s2_decision` 列：`drop_exact_dup` 剔除（S2 决策已物化在分片）；
2. s3 近重：`s3_lookup/<domain>/lookup.ids.bin`（memmap + searchsorted，全 worker
   共享 page cache）或其回退 `s3_decisions/<domain>.parquet`，`drop_near_dup` 剔除；
3. 去污染：`decontam/<src>/<rel>.parquet` 决策表，`drop_contaminated` 剔除。

**硬前置**：decontam 必须覆盖全部 s3 输入分片（driver 启动即 fail-fast 校验，
缺一片拒跑）。计长用 `tokenizers` 库直读 tokenizer.json，`encode_batch` 批量编码、
`add_special_tokens=False`（裸文本 token 数 = pretrain packing 口径）；每 worker
进程加载一次 tokenizer。

历史 S5 只能通过 `scripts/migrate_legacy_data_lineage.py` 进入增量路径：先在各来源均匀抽取
分片并用当前 tokenizer 精确重编码，再逐分片扫描紧凑 S5 parquet，闭合行数、token 总数、
桶、资格标记并计算内容 SHA-256。审计证书、tokenizer SHA 和旧输出指纹全部匹配后，旧
`n_tokens` 才能作为流式缓存；重算时每分片仍精确复核缓存样本，新救回的文档一律重新
tokenize。这样策略修正无需重复编码数十亿个未变化文档，也不会把无血缘旧结果直接放行。

web/code 的 ndv2 属于“旧 lookup 为空、新 lookup 只增加删除”的限制性变更。通过 lineage
认证的 S5 紧凑表可直接按当前 lookup 过滤，并与 decontam 决策表精确计算交叠；随后重建内容
SHA、长度桶与 token 账本，无需读取约 10TB S3 文本或重新分词。中文去污染放宽则复用旧
`n_tokens`，只对新救回文档编码。2026-10-18 两条增量路径完成后，全局 summary 为
12,829/12,829 分片、`shard_errors=0`、3,105,637,476 篇、2,530,570,633,678 tokens；正式
`mix_m3_e1_100b` 物化为 94,235,054 篇 / 99,444,599,052 tokens，S6 元数据、train/val
互斥和 members/order 多重集审计均通过。

### 输出布局

```
data/processed/
├── s5/<source>/<relpath>.parquet   # doc_id, n_tokens(uint32), length_bucket,
│                                   #   mix_eligibility(list<string>：n_tokens<=L 的各 context length)
├── s5/<source>/<relpath>.summary.json  # in/out_rows, total_tokens, decisions(三类剔除量), buckets, 吞吐
└── stage_s5_global_summary.json    # shard_errors 置顶 + totals + per_source/per_domain（含 decisions.*）
```

**互斥前缀嵌套资格**（training.py prefix mix 语义）：doc 落唯一互斥桶
（`assign_context_bucket`），`mix_eligibility` = 有资格进入的 mix 阶段列表
（短桶有资格进长阶段；每条最终只进一个 mix——最终选取物化在 S6）。
gt_64K doc 资格表为空（midtrain 截断消费，可审计的有意识取舍）。

### 实测吞吐（director env，2026-09-26，decontam 128 核同机竞争下）

| 负载 | 实测 | 备注 |
|---|---|---|
| fineweb 分片（292MB，218k 行，~1.3KB/行） | ~0.9k 行/s/核（波动区间 0.6k–2.5k） | 端到端含三路决策 join + zstd 读写 |
| yf-64k-bpe encode_batch 本体（~2KB/篇） | 0.6k–2.3k 篇/s/核 | 同负载波动 3-4×；空闲机需复测 |

**≥2 万篇/秒/核 指标未达标**（与 S4 fasttext 档同一结论：纯 CPU tokenizer 推理的
物理上限，ByteLevel pre-tokenizer 逐字节扫描是主导成本；线程化实测无增益——
8 线程聚合 751 docs/s 反低于单线程 2.5k，encode_batch 不向外扩展）。聚合口径：
160 worker × ~2k 篇/s ≈ 32 万篇/s ⇒ 全池 36.5 亿篇 ≈ 3 小时墙钟（IO 与 S4 同量级），
满足 S5 一天预算；复测与提速路径（截断编码/ faster pre-tokenizer）记报告 backlog。
真实分片抽查：n_tokens 与 tokenizer 逐篇精确一致；S2 剔除率 43.1%（与 E-17 记录
45.7% 同量级）、decontam 命中 220/21.8 万。

### 测试

```bash
PYTHONPATH=src /usr/bin/python3 -m pytest tests/test_length_bucket.py -q
```

分桶/资格标记边界、三路决策 join 计数（s2/s3/decontam 各一）、n_tokens 与
yf-64k-bpe 精确一致、幂等（二次 skip）、decontam 缺片落 error summary（E-16）、
driver 硬前置（decontam 覆盖）fail-fast。

## 生产管线（S6，配比物化，2026-09-26 实现——E1 前最后一块）

打分池 + 分位表 → 领域 top-k 质量序选取（token 配额最大余数法）→ 课程两段
doc_id 顺序表物化（tokenize 放 pretrain 阶段）。实现于 `scale/materialize.py`，
driver `--stage s6 --mix-config <yaml>`（`--stage s6_val` 可单独切 val 集）。

### 用法

```bash
cd data-process
# 首次物化 = M3 档 100B 实验子集（E1/E2 共用，硬门全过后执行）：
PYTHONPATH=src /usr/bin/python3 -m yuformer_data.scale.driver \
  --stage s6 --mix-config configs/mix_m3_e1_100b.yaml --workers 64 --out-root ../data/processed
# 正式配比（M1/M2/M3 三套画像，configs/mix_m{1,2,3}.yaml）/ 只切 val / 部分源冒烟：
... --stage s6 --mix-config configs/mix_m2.yaml
... --stage s6_val --mix-config configs/mix_m2.yaml
... --stage s6 --mix-config configs/mix_m3_e1_100b.yaml --sources dolma-books proof-pile-2
```

### 前置硬门（fail-fast，缺一即拒绝物化，全部缺失一次性报告）

`materialize.gate_errors` + `assert_gates`（SystemExit，报告按门分组）：
1. **① s4_scored 覆盖**：逐分片产物+summary 齐 + `stage_s4_global_summary`
   shard_errors=0 + 分位表已构建且覆盖 mix 全部领域（score_field 仅支持
   composite_v2——阈值来自 s4 分位表 `table[n-⌈top_k·n⌉]`，同值同分决胜无关）；
2. **② decontam 完成**：`stage_decontam_global_summary` shard_errors=0 +
   逐分片产物覆盖 s3 输入集；
3. **③ ndv2 lookup 已发布**：web/code 的 `s3_decisions/<domain>.summary.json` 使用 direct-edge
   policy；全文精度审计通过且边集指纹一致；排序 lookup 文件大小与 dropped 行数闭合；
   `stage_ndv2_global_summary` 无错误且逐域 dropped 与决定表一致；
4. **④ held-out val 已切出**：`ensure_val_sets`（本模块执行，文件级幂等）——
   每领域 min(50M tokens, 池 1%)，**确定性 blake2b 抽样**（两遍流式：pass1
   4096 桶 token 直方图定边界桶，pass2 边界桶内全哈希排序取前缀 = 无放回等概率），
   永不进训练；`val_sets/<domain>.parquet` + `ids.bin`（memmap join 用）；
5. **⑤ s5 覆盖**：逐分片产物齐 + `stage_s5_global_summary` shard_errors=0。

S6 manifest 的 upstream snapshot 绑定实际读取的 S4/S5 分片、当前 val、ndv2 lookup 和所选
`score_field` 的分位表。历史 composite/rule/其他 hybrid 表不属于当前 mix 的读取闭包，不会
误触发失效；所选分位 NPY 仍做全内容哈希，以捕获同尺寸损坏（E-90）。
长时间 collect 完成后会再次计算同一快照，只有前后 SHA-256 完全一致才写 manifest；如果
并发上游任务在 S6 期间改写任何输入，未提交的 members/order 会被删除并要求重跑（E-91）。

### 配比与顺序物化

- **配额**：`total_tokens × 权重`（最大余数法 = `training.allocate_mix_quotas`，
  与 ZGCM 同实现），域内 top_k 为取样深度上限（cap = top_k × 域池 tokens）。
- **选取**：候选 = composite ≥ 域阈值 且 ∉ val；`sort_key_quality` =
  (composite 降序, doc_id 字节序决胜) 全向量化 lexsort（无 Python 逐行排序）；
  token 前缀截到 min(配额, cap)。候选 parts 落盘 `mix_work/<mix>/`（driver 归并，
  成功即清），失败分片落 error summary 且**不物化任何产物**（E-16）。
- **正式 baseline 顺序**（`mix_order/<mix>_stage2.parquet` = position/doc_id/domain/
  n_tokens 顺序表 + token 统计；**tokenize 由 pretrain 阶段执行**，本模块不落 `.bin`）：
  全部正式配置使用 doc_id 的 blake2b 均匀哈希顺序，得到可复现的全局随机流。
  `upstream_score` 在不同来源间语义不可比，不能充当 lexical complexity；旧两阶段路径只有
  同时声明 `method: upstream_score_legacy` 与 `experimental: true` 才可用于显式消融。
  从旧配置切回 random 时物化器主动删除残留 stage1，防止重复消费。
- **产物**：`mix_members/<mix>.parquet`（doc_id, domain, composite, n_tokens，
  domain+doc_id 总序）、`mix_order/<mix>.manifest.json`（config echo/上游 snapshot/阈值/配额/
  选取统计/课程统计——config 与上游身份同时命中才幂等 skip）、`stage_s6_global_summary.json`
  （per_domain：配额/池量/三类剔除量/候选量/**质量分位数截断量**/配额缺口/
  val 量/per-source 入选 tokens；顺序表 token 数与模式）。

### mix 配置

`configs/mix_{m1,m2,m3}.yaml`（池原生 200B / stable 800B / 极化 top-10% 200B）、
`configs/mix_warmup_50b.yaml` 与 `configs/mix_anneal.yaml`
与 `configs/mix_m3_e1_100b.yaml`（E1 首次物化目标，选取规则 = M3、总量 100B）。
配置自包含：领域权重 + top_k + 总量目标 + 顺序策略 + val 配额 + score_field +
seed + mix_name；diff 两个配置即得配比差异。

### 测试

```bash
PYTHONPATH=src /usr/bin/python3 -m pytest tests/test_materialize.py -q
```

硬门 fail-fast（①缺分片/①分位表缺域/②error shard/③E-17 占位（web）/⑤缺分片 +
多门聚合报告）、最大余数法数值用例（含权重配额和=总量 +1 余数分配
与参考实现一致）、top-k 选取确定性（双池全产物字节一致：members/stage1/stage2/
val/summary）、val 切出互斥性（val ∩ train = ∅）与配额界、合成端到端（质量序
正确性/课程两段 token 守恒/25% 比例/stage1 edu 升序/幂等 skip/driver.main 接线）。
