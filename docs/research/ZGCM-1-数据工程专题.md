# ZGCM-1 数据工程专题报告

> 本报告聚焦"用了什么数据、如何配比、怎么清洗/去重/过滤/打标、metadata 格式"的数据工程细节，作为 YuFormer 数据模块设计的精确依据。所有数值带 `file:line`（相对 `ZGCM-1/data-process/` 或 `ZGCM-1/`）。仓库内未公开的内容明确标注。

---

## 一、用了什么数据

### 1.1 Pretrain 4.19T（web + 学术OCR + 代码 + 数学 + LaTeX）

两阶段构成（`pretrain/docs/PRETRAIN_PHASES.md:11-12`）：

| Stage | Token | 数据策略 |
|---|---:|---|
| Stage 1 | ~0.99T | 通用语料按**词汇复杂度**升序排序（剔除极端离群值）；代码与数学**独立交错**（因词汇复杂度不反映推理难度） |
| Stage 2 | ~3.20T | 离线全局打乱（seed `20260603`）的全量混合，**增加代码/数学比例**并加入专门推理成分；跨阶段去重排除 Stage-1 已用内容 |

**公开的 profile 与数据源**（仓库只给了 3 个 pretrain source family 的 profile，web/数学/LaTeX 无独立 profile，走 `general_text` 通用通道）：

| source family | profile | 描述 | file:line |
|---|---|---|---|
| `code` | `gharchive` | GitHub release zip 源码（默认 source=`gharchive_release_zip`） | `stages/pretrain/datasets/gharchive.py:1,274` |
| `pdf_ocr` | `finepdfs` | FinePDFs 中英学术 PDF | `stages/pretrain/datasets/finepdfs.py:1` |
| `pdf_ocr` | `olmocr` | OLMOCR 科学 PDF OCR | `stages/pretrain/datasets/olmocr.py:1` |
| `general_text` | （无 profile） | web/数学/LaTeX 等通用文本 | `stages/pretrain/source_cleaning.py:51-62` |

**未公开**：pretrain 各类的精确 HF 数据集名、各类 token 量/占比、词汇复杂度计算公式（文档描述，代码内无实现）。

### 1.2 Midtrain 600.51B（code + math + knowledge + reasoning + instruction + agentic）

从 **2.86T 去重候选池**采样 600.51B（`midtrain/docs/MIDTRAIN_STAGES.md:9-10`），三阶段：

| Stage | Token | gbs | seq | 并行 |
|---|---:|---:|---:|---|
| 16K | 180.0B | 768 | 16,384 | TP2/PP1/CP2 |
| 64K | 240.0B | 192 | 65,536 | TP2/PP1/CP4 |
| 256K | 180.51B | 48 | 262,144 | TP2/PP2/CP4 |

每步恒 ~12.6M tokens（`MIDTRAIN_STAGES.md:18`）。各阶段保留短样本（`MIDTRAIN_STAGES.md:21-24`）：

| 阶段 | ≤16K | 16K-64K | >64K |
|---|---:|---:|---:|
| 64K | 180.89B | 59.11B | — |
| 256K | 127.81B | 21.72B | 30.98B |

### 1.3 Posttrain（SFT）——唯一公开真实 HF 数据集名的 profile

`dolci_think` profile（`stages/posttrain/datasets/dolci_think.py:13-22`）公开 8 个源：

| HF/源名 | 内部 label | ability |
|---|---|---|
| `saumyamalik/OpenThoughts3-full-filtered-math-decontam-v2` | openthoughts_math | math_reasoning |
| `saumyamalik/correct-python-sft-187k-x16-thoughts-filtered-decontam-v2` | dolci_python_code | code_reasoning |
| `allenai/persona-precise-if-r1-final-content-filtered-chinese-filtered` | persona_precise_if | precise_if |
| `saumyamalik/if_qwq_reasoning_verified_filtered_decontam-v2` | qwq_verified_if | precise_if |
| `allenai/nemotron-post-training-dataset-subset-ngram-filtered-no-tool-calls` | nemotron_code_no_tool | code_reasoning |
| `allenai/SYNTHETIC-2-SFT-cn-fltrd-final-ngram-filtered-chinese-filtered` | synthetic2_verified | code_basic_reasoning |
| `saumyamalik/OpenThoughts3-full-filtered-science-decontam-v2` | openthoughts_science | science_reasoning |
| `saumyamalik/OpenThoughts3-full-filtered-code-subsampled-decontam-v2` | openthoughts_code | code_reasoning |

其余 profile（`nemotron_math`/`web_knowledge`/`agent_coding`/`dolci_tooluse`/`ultradata`）的真实 HF 数据集名未公开。全部源名带 `-decontam-v2`/`-ngram-filtered` 后缀——**评测去污染是一等公民**。

---

## 二、如何配比

### 2.1 两种配比模式（核心思想）

| 模式 | 含义 | ZGCM 用途 | 原因 |
|---|---|---|---|
| **离线物化** | 配比/排序/去重/分词全部固化进单一 indexed dataset，loader 保序消费 | **Pretrain 全程** | 课程顺序必须固化；断点续训样本序列逐条一致 |
| **在线 blend** | `--data-args-path` 传多个加权前缀，loader 在线跨 shard 采样 | **Midtrain 全程** | 跨上下文复用同一组 shard；长短样本稳定配比 |

**Pretrain 离线物化**（`01_pretrain_1t_sequential.env:13,94,95`）：单 `REAL_DATA_PREFIX` + `DATA_ARGS_PATH=""` + `NO_DATA_SHUFFLE=1`，无 blend 脚本。

**Midtrain 在线 blend**（`midtrain/scripts/build_midtrain_data_args.sh`）：768 前缀加权 blend，三阶段共用同一 `midtrain_data_args.txt`：

| 池 | shard 数 | 权重 | 命名 | 候选来源 |
|---|---:|---:|---|---|
| 长样本池 | 512 (0-511) | **2** | `mix64_v3_glm51_shuffle512_shard_%05d_text_document` | B16+B64（排除 mix16 已选） |
| 短样本池 | 256 (0-255) | **1** | `mix16_v4_glm51_shuffle256_shard_%05d_text_document` | B16 |
| 合计 | 768 | — | — | 长:短 = 2:1 |

每行格式 `<权重> <prefix>`（`build_midtrain_data_args.sh:21,27`）。

### 2.2 课程式长度混合（互斥前缀嵌套）—— midtrain 配比核心

`stages/midtrain/mix.py:14-18` 的 `DEFAULT_ELIGIBLE`：

| 最终混合 | 候选 base 桶 | 排除规则 |
|---|---|---|
| `mix16` | B16 | — |
| `mix64` | B16 + B64 | 排除 mix16 已选 ID |
| `mix256` | B16 + B64 + B256 | 排除 mix16/mix64 已选 ID |

`prefix_sample_mix`（`mix.py:25-59`）：处理顺序固定 `mix16→mix64→mix256`，每个 mix 用 `blake2b(f"{seed}:{final_name}\0{id}", 16字节)` 排序取前 target 个，已选集 `selected` 单调增长保证互斥。**seed 默认 `"zgcm-midtrain"`**（`mix.py:29`）。

长度桶边界（`stages/midtrain/buckets.py:6`）：`assign_length_bucket(token_count, boundaries=(16384, 65536, 262144))`：

| token 范围 | 桶标签 |
|---|---|
| ≤16,384 | B16 |
| 16,384-65,536 | B64 |
| 65,536-262,144 | B256 |
| >262,144 | gt_256 |

### 2.3 配额分配——最大余数法

`training.py:45-59` `allocate_mix_quotas(weights, total_samples)`：
1. 过滤 `weight<=0`（`training.py:49`）
2. 归一化 → 精确配额 `exact = total × weight / sum`（`training.py:53`）
3. `math.floor` 取整（`training.py:54`）
4. 余数降序（tie-break 按 name 字典序升序）补齐剩余（`training.py:55-58`）
5. 保证 `sum(quotas) == total_samples`（测试 `tests/test_pipelines.py:169-170`）

权重由调用方传入，函数无硬编码默认。

### 2.4 确定性选取与交错

`select_mixed_sample_ids`（`training.py:62-82`）：每源用 `blake2b(f"{seed}:{source}\0{id}")` 排序取配额，再用 `blake2b(f"{seed}:mix\0{id}")` 跨源交错排序（非按源分块）。**seed 默认 `"zgcm"`**（`training.py:66`）。源样本不足时 raise。

---

## 三、怎么清洗

### 3.1 流水线组装

`pipelines.py:33`：`steps = category_steps[:1] + COMMON_PREFIX + category_steps[1:] + COMMON_SUFFIX`

即：**类别首步归一 → COMMON_PREFIX(4步) → 其余类别步 → COMMON_SUFFIX(5步)**。执行时 DROP 立即 break（`core.py:46-62`）。

### 3.2 COMMON_PREFIX 四步（`steps/common.py:168-173`）

**Step 1 `normalize_schema_text`**（对 text_fields 每 str 字段，`common.py:27-31`）：
1. CRLF→LF：`replace("\r\n","\n").replace("\r","\n")`
2. **NFKC** 归一：`unicodedata.normalize("NFKC", ...)`
3. 删控制符：正则 `[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]`（删 C0+DEL，**保留** \t \n \r）（`common.py:17`）
4. 行内空白压缩：`[\t\f\v ]+`→单空格，逐行 strip（`common.py:15`）
5. 空行压缩：`\n{3,}`→`\n\n`（≥3 换行压成 2，即最多 1 空行），整体 strip（`common.py:16`）

**Step 2 `validate_basic_integrity`**（`common.py:42-53`）：
- 空文本 → DROP `empty_training_text`
- `len < min_chars`(默认**16**) → DROP `too_short`
- `len > max_chars`(默认**2,000,000**) → REVIEW `too_long`（非 DROP）

**Step 3 `validate_provenance`**（`common.py:70-82`）：
- `require_license`(默认 False) 开启且无 license → REVIEW `missing_license`
- 无 source（**无条件检查**）→ REVIEW `missing_source`

**Step 4 `scan_sensitive_content`**（`common.py:56-67`）：
- 密钥检测正则（`common.py:18-20`）：`(?i)(?:api[_-]?key|secret|password|access[_-]?token)\s*[:=]\s*["']?[A-Za-z0-9_\-/.]{12,}`（值≥12字符）→ REVIEW `possible_secret`
- 敏感词表（`common.py:61-66`）：纯外部配置 `sensitive_terms`（默认空元组，无内置词表），casefold 子串匹配 → REVIEW `sensitive_term`

### 3.3 六类专属步骤（每类的首步在 COMMON_PREFIX 前，其余在后）

**code（`steps/code.py`）**：
- 正则 `_GENERATED`=`(?i)(?:generated file|do not edit|auto[- ]generated)`，`_CODE_SIGNAL`=`[{}();]|\b(?:def|class|function|import|package|SELECT|FROM)\b`
- 拒绝 vendor/node_modules/.min.js/.map（默认 `code_denied_path_parts` 4 项，`code.py:30-37`）→ DROP `non_core_or_vendored_file`
- 前 4096 字符搜 `_GENERATED` → REVIEW `possibly_generated_code`
- 代码信号弱（无 `_CODE_SIGNAL` 且 `<3` 行）→ REVIEW `weak_code_signal`
- FIM triplet（prefix/middle/suffix）不完整 → DROP `incomplete_fim_triplet`
- 分桶 subtype：code/fim/patch/code_qa；verified：verified/unverified

**web（`steps/web.py`）**：
- 样板正则 `_BOILERPLATE`=`(?i)(cookie policy|accept all cookies|privacy policy|sign in|subscribe now)`（5关键词），命中**≥4** → REVIEW `web_boilerplate`
- 行重复率：唯一非空行/总非空行 **<0.45** → DROP `repetitive_web_text`
- QA 对：question **<4** 字符或无 answer → DROP；答案匹配 `(?i)^\s*(?:unknown|n/?a|i don'?t know|无法回答|不知道)\s*[.!。！]?$` → DROP `unknown_answer`；grounded=False → REVIEW
- 质量分桶：score≥0.8→high，≥0.5→medium，<0.5→low

**agentic（`steps/agentic.py`）**：
- 无 messages 且无 pre_rendered → REVIEW `unstructured_agent_trace`
- 首角色非 system/user → REVIEW `invalid_first_role`；任一空角色 → DROP `missing_message_role`
- 工具调用闭环（pending 计数器）：`pending=0`，遇 tool_call `+1`，遇 tool/observation `max(0,pending-1)`；最终有未配对 → REVIEW `unpaired_tool_call`；无 tool_call → REVIEW `no_tool_call`
- outcome 关键词（casefold）：failed=`exit due to`/`permission denied`/`tool error`/`timed out`；verified=`submitted`/`tests passed`/`task completed`/`success`；分桶 verified/failed/weak；failed 且非 verified → REVIEW

**instruction（`steps/instruction.py`）**：
- 字段归一：instruction→prompt，output→response
- messages 缺 user 或 assistant → DROP；prompt/response 缺失 → DROP；prompt==response（casefold）→ DROP
- AI 套话正则 `_TEMPLATE`=`(?i)(?:as an ai language model|i cannot assist with|here is the requested response)`（搜 response）→ REVIEW `generic_model_template`
- 分桶 task/risk/turns（messages>2→multi）

**math（`steps/math.py`）**：
- 多模态缺失正则 `_IMAGE_DEP`=`(?i)(?:<img|\.(?:png|jpg|jpeg|gif|svg)\b|\\includegraphics|see (?:the )?figure)`，命中且无 image_text → REVIEW `missing_multimodal_context`
- solution **<32** 字符 → REVIEW `solution_too_short`
- 答案信号正则 `_ANSWER_SIGNAL`=`(?i)(?:final answer|answer\s*:|therefore|thus|\\boxed\{|答案|所以)`，无且无 answer 字段 → REVIEW `missing_final_answer_signal`
- verified=False → REVIEW；分桶 form/difficulty/domain

**reasoning（`steps/reasoning.py`）**：
- 控制伪影正则 `_CONTROL_ARTIFACT`=`(?is)<\|(?:system|assistant|user|endoftext)[^>]*\|>|\[system prompt\]|api[_ ]error`
- reasoning **<64** 字符 **或** 无推理信号（`_REASON_SIGNAL`=`(?i)(?:because|therefore|first,|step \d+|we need to|由于|因此|首先|步骤)`）→ REVIEW `weak_reasoning_signal`
- lineage_hash 机制：`source_row_hash`→`lineage_hash`（setdefault，`reasoning.py:49`）

---

## 四、怎么去重

### 4.1 精确去重（COMMON_SUFFIX Step 2，`common.py:85-93`）

- 哈希 = `sha256( normalize_text_value(text).casefold() )`（重新归一+casefold，`common.py:86`）
- 存 `data["text_sha256"]`（setdefault）
- 已见 → DROP `exact_duplicate`；否则加入 `context.seen_hashes`（set）

### 4.2 近似去重（COMMON_SUFFIX Step 3，`common.py:96-114`，**默认禁用**）

启用条件：`config["near_dedup"]=True`（默认 False）。配置：
- `near_dedup_threshold` 默认 **3**（Hamming 距离阈值，≤3 判近重）
- `near_dedup_bands` 默认 **4**（4 band × 16 bit）
- `near_dedup_ngram` 默认 **5**（word-level 5-gram）

**SimHash64**（`dedup.py:12-28`）：
- token = `\w+`（UNICODE，casefold）
- 5-gram（空格连接）每个经 `blake2b(digest_size=8)` 得 64-bit
- 64 位加权投票：`bit` 置 1 当 `weight>=0`

**NearDeduper**（`dedup.py:35-73`）：
- 约束：`64 % bands == 0` 且 `threshold < bands`（保证 LSH 无假阴性）
- 4 band × 16 bit 倒排索引，候选集取交集后 Hamming 验证
- `hamming_distance` = `(a^b).bit_count()`

> MinHash 不在包内计算，FinePDFs profile 消费上游已算的 `minhash_count`（`finepdfs.py:43,54-57`，默认 `>2` 判过度重叠 DROP）。

---

## 五、怎么过滤（质量策略）

### 5.1 质量策略机制（COMMON_SUFFIX Step 1，`quality.py:53-90`）

`QualityPolicy` 是**源感知的三段阈值**，分类器在管线外（分数由上游打标）：
- `score_field`：读哪个字段（如 `quality_mean`/`quality_score`/`edu_score`）
- `review_min`：**实为 DROP 阈值**——`score < review_min` → DROP `quality_score_below_drop_threshold`
- `keep_min`：**实为 REVIEW 阈值**——`review_min <= score < keep_min` → REVIEW `quality_score_below_keep_threshold`
- `score >= keep_min` → KEEP
- `hard_error_field` + `hard_error_max`：`hard_error > hard_error_max` → DROP `hard_error_score_exceeded`
- `require_score`：score 缺失 → REVIEW

策略解析链（`quality.py:53-63`）：`source_key → source → dataset → category → default`。

### 5.2 实际配置（`examples/quality-policies.json` 完整内容）

```json
{
  "near_dedup": true,
  "near_dedup_threshold": 3,
  "quality_policies": {
    "web": {"score_field": "quality_mean", "keep_min": 4.0, "review_min": 3.0, "require_score": false},
    "math": {"score_field": "quality_score", "keep_min": 0.8, "review_min": 0.5, "hard_error_field": "hard_error_score", "hard_error_max": 3}
  }
}
```

| source | score_field | DROP 线 | REVIEW 线 | 硬错误 |
|---|---|---|---|---|
| web | quality_mean | <3.0 | [3.0,4.0) | — |
| math | quality_score | <0.5 | [0.5,0.8) | hard_error_score>3 → DROP |

无 `default` 策略；code/agentic/instruction/reasoning 无条目 → 不施加分值策略。

### 5.3 数据集 profile 的专属过滤阈值

| profile | 参数 | 默认值 | 含义 | file:line |
|---|---|---|---|---|
| finepdfs | `finepdfs_lid_threshold` | 0.85 | 语言分<则 DROP | finepdfs.py:46 |
| finepdfs | `finepdfs_minhash_max` | 2.0 | MinHash 重叠>则 DROP | finepdfs.py:47 |
| finepdfs | `finepdfs_primary_threshold` | 0.8 | 主质量分<则 DROP | finepdfs.py:48 |
| olmocr | `olmocr_min_compression_ratio` | 0.08 | zlib 压缩比<则 DROP（OCR垃圾） | olmocr.py:11 |
| olmocr | `olmocr_min_edu_score` | 0.6 | 教育分<则 DROP | olmocr.py:30 |
| gharchive | `gharchive_max_file_bytes` | 0(禁用) | 文件大小>则 DROP | gharchive.py:195 |
| web_knowledge | `web_knowledge_min_quality_band` | "lt3" | 最低质量带（lt3<ge3<ge4） | web_knowledge.py:68 |
| nemotron_math | `nemotron_math_hard_error_threshold` | 4.0 | 硬错误风险分≥则 DROP | nemotron_math.py:75 |
| nemotron_math | `nemotron_math_drop_keys` | () | (source,row) 丢弃键集合 | nemotron_math.py:73 |
| dolci_think | `dolci_think_strict` | True | 严格模式（模板token→DROP而非REVIEW） | dolci_think.py:87 |
| dolci_think | `dolci_think_max_rough_tokens` | 32768 | 粗略token>则 DROP | dolci_think.py:109 |
| dolci_think | `dolci_think_review_rough_tokens` | 16000 | 粗略token>则 REVIEW | dolci_think.py:112 |

---

## 六、怎么打标（决策、flags、buckets、reasons）

### 6.1 Decision 三态（`models.py:38-41,73-79`）

KEEP/REVIEW/DROP。`mark()` 只降不升：
- DROP：无条件覆盖
- REVIEW：仅当当前是 KEEP 时覆盖
- KEEP：无 `mark(KEEP,...)` 调用

状态机：允许 `KEEP→REVIEW`、`KEEP→DROP`、`REVIEW→DROP`；禁止逆向。

### 6.2 reasons（list[str]，有序，去重）

DROP reasons（17 条）：`empty_training_text`/`too_short`/`exact_duplicate`/`near_duplicate`/`non_core_or_vendored_file`/`incomplete_fim_triplet`/`repetitive_web_text`/`incomplete_qa_pair`/`unknown_answer`/`missing_message_role`/`incomplete_instruction_messages`/`incomplete_instruction_pair`/`prompt_response_identical`/`missing_problem_or_solution`/`missing_reasoning`/`quality_score_below_drop_threshold`/`hard_error_score_exceeded`

REVIEW reasons（24 条）：`too_long`/`possible_secret`/`sensitive_term`/`missing_license`/`missing_source`/`possibly_generated_code`/`weak_code_signal`/`web_boilerplate`/`ungrounded_answer`/`unstructured_agent_trace`/`invalid_first_role`/`no_tool_call`/`unpaired_tool_call`/`failed_or_environment_trajectory`/`generic_model_template`/`missing_multimodal_context`/`solution_too_short`/`missing_final_answer_signal`/`verifier_failed`/`control_or_prompt_artifact`/`weak_reasoning_signal`/`missing_final_answer`/`missing_quality_score`/`quality_score_below_keep_threshold`

### 6.3 flags（set[str]）

`possible_secret`/`sensitive_term`/`missing_license`/`missing_source`/`missing_quality_score`（+ stages 下数据集专属 flag）

### 6.4 buckets（dict[str,str]，17 种键）

共享：`length`（1_2048/2049_8192/8193_32768/32769_65536/gt_65536）、`quality`（main/review/drop）
code：`subtype`（code/fim/patch/code_qa）、`language`、`verified`
web：`subtype`（web_qa/web_text）、`topic`、`model_quality`（high/medium/low）
agentic：`subtype`、`outcome`（verified/failed/weak）、`tool_family`
instruction：`task`、`risk`、`turns`（multi/single）
math：`form`、`difficulty`、`domain`、`verified`
reasoning：`origin`、`domain`、`verified`

### 6.5 metrics（dict[str,float]）

`estimated_tokens`/`quality_score`/`hard_error_score`/`turns`

---

## 七、metadata 格式

### 7.1 Sample schema（`models.py:44-93`）

| 字段 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `dataset` | str | 必填 | 数据集名 |
| `source_id` | str | 必填 | 记录 id（缺失回退 `{dataset}:{index}`） |
| `stage` | Stage | 必填 | pretrain/midtrain/posttrain/sft |
| `category` | DataCategory | 必填 | 8 类 |
| `data` | dict | 必填 | 原始记录+规范化字段 |
| `text_fields` | tuple[str,...] | ("text",) | 构成训练文本的字段 |
| `decision` | Decision | KEEP | 清洗决策 |
| `reasons` | list[str] | [] | 决策原因（有序去重） |
| `flags` | set[str] | set() | 审计标注 |
| `buckets` | dict[str,str] | {} | 采样/分桶标签 |
| `metrics` | dict[str,float] | {} | 数值指标 |

### 7.2 输出 JSONL 记录（11 字段，`steps/common.py:159-165`）

`as_dict()` 的 10 字段 + `record_sha256`。写出时 `sort_keys=True`，on-disk 键序（字母序）：
`buckets, category, data, dataset, decision, flags, metrics, reasons, record_sha256, source_id, stage`

`record_sha256` = `sha256( json.dumps(data, ensure_ascii=False, sort_keys=True, default=str) )`——**仅哈希 data 子对象**。

序列化参数（`io.py:43`）：`ensure_ascii=False, sort_keys=True, default=str`，一行一条，紧凑无 indent，写出不支持压缩。

### 7.3 summary.json（`cli.py:56-63`）

```json
{
  "stage": "...", "category": "...", "dataset": "...", "source": "...",
  "dataset_profile": "...", "written": 123,
  "pipeline": ["step1", "step2", ...],
  "counters": {"input": 100, "step.X.seen": 100, "step.X.drop": 5, "output.keep": 90, ...},
  "manifest": "output.summary.json"
}
```

`counters` 键命名：`input`、`step.<name>.seen`、`step.<name>.<keep|review|drop>`、`output.<keep|review|drop>`。

### 7.4 manifest（`common/manifest.py:21-32`）

从**回读输出文件**重建（保证与下游 loader 一致）：
```json
{"stage": "...", "records": 90, "decisions": {"keep": 85, "review": 5, "drop": 0}, "output": "out.jsonl"}
```

`records` == 输出文件行数（默认不含 dropped，除非 `--keep-dropped`）。无 `file_sha256` 字段。

---

## 八、关键阈值速查表

| 维度 | 阈值 | file:line |
|---|---|---|
| 文本最小字符 | **16** | common.py:47 |
| 文本最大字符 | **2,000,000**（超→REVIEW） | common.py:48 |
| 空行压缩 | **≥3 连续 \n** → \n\n | common.py:16 |
| 密钥值最小长度 | **≥12** 字符 | common.py:19 |
| 样板命中阈值 | **≥4** | web.py:27 |
| 行重复率 DROP | unique/total **<0.45** | web.py:30 |
| QA 问题最小长度 | **4** | web.py:40 |
| math solution 最短 | **32** 字符 | math.py:35 |
| reasoning 最短 | **64** 字符 | reasoning.py:36 |
| instruction multi-turn | messages **>2** | instruction.py:57 |
| SimHash n-gram | **5**（word） | common.py:112 |
| SimHash 位宽 | **64-bit**（blake2b digest_size=8） | dedup.py:21 |
| SimHash bands | **4**×16bit | common.py:107 |
| SimHash Hamming 阈值 | **3**（≤3 判近重） | common.py:106 |
| token 估算 | **(len+3)//4**（≈4字符/token） | common.py:123 |
| 通用长度桶边界 | **2048/8192/32768/65536** | common.py:126 |
| midtrain 长度桶边界 | **16384/65536/262144** | buckets.py:6 |
| web quality DROP/REVIEW | **<3.0** / **[3.0,4.0)** | quality-policies.json |
| math quality DROP/REVIEW | **<0.5** / **[0.5,0.8)** | quality-policies.json |
| math hard_error DROP | **>3** | quality-policies.json |
| finepdfs lid/minhash/primary | **0.85/2.0/0.8** | finepdfs.py:46-48 |
| olmocr compression/edu | **0.08/0.6** | olmocr.py:11,30 |
| nemotron_math hard_error | **4.0** | nemotron_math.py:75 |
| dolci_think max/review tokens | **32768/16000** | dolci_think.py:109,112 |
| 长池:短池权重 | **2:1** | build_midtrain_data_args.sh |
| midtrain seed | **zgcm-midtrain** | mix.py:29 |
| 通用 seed | **zgcm** | training.py:66 |

---

## 九、对 YuFormer 数据模块的启示

1. **契约层 + 参考实现**的定位值得继承：data-process 不做分布式全量处理，而是锁死 schema/决策/审计的契约层，大规模能力以 injectable 接口留白。YuFormer 可在单机上完整跑通治理逻辑，对接训练环境时补齐 tokenizer/indexed writer。
2. **三态决策 + 全程审计**是数据可追溯的根基：keep/review/drop + reasons/flags/record_sha256 + manifest 回读重建——这套机制复现成本极低（`mark()` 仅 7 行），但工程价值极高。
3. **配比两种模式都值得做**：pretrain 的离线物化（课程顺序固化）与 midtrain 的在线 blend（长短池加权）是两种不同诉求的解法，YuFormer 可在小规模上各做一遍并对比。
4. **质量分类器需自研**：ZGCM 只消费上游分数做阈值裁决，分类器在管线外。YuFormer 的创新空间在此——轻量 fasttext/规则 + 可选 LLM 打分。
5. **去重 SimHash 带状索引**的 LSH 无假阴性设计是教科书级小算法，可 1:1 复刻；MinHash 留给大规模环境。
6. **清洗规则的中英双语覆盖**（`答案/所以/由于/因此/首先/步骤` 等）提示 YuFormer 若做中文需补齐双语正则。
7. **未公开的关键信息**（词汇复杂度公式、pretrain 各类占比、midtrain 各 shard 数据集来源）——YuFormer 需自行设计这些，反而是创新空间的所在。
