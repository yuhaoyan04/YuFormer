# YuFormer Tokenizer

分词器训练、评估与全量数据 tokenize 工具。支持多组 tokenizer 实验对比，为数据配比实验提供基础。

## 设计目标

1. **自训 BPE**：不直接复用 GLM/Qwen 等现成 tokenizer，从 YuFormer 自有语料训练
2. **多组实验**：支持不同词表大小（32k/64k/128k）、不同模型类型（BPE/Unigram）、不同特殊 token 体系的对比
3. **全量 tokenize**：流式处理已下载数据集，输出 token_ids JSONL 供训练消费
4. **评估体系**：压缩率、OOV 率、语言覆盖、token 分布——产出 tokenizer 评估报告

## 特殊 Token

| Token | 用途 |
|---|---|
| `<\|endoftext\|>` | 文档结束 / padding / EOS |
| `<\|unk\|>` | 未知 token |
| `<\|im_start\|>` | 消息块开始（后接纯文本角色名 + `\n`） |
| `<\|im_end\|>` | 消息块结束 |
| `<\|think_start\|>` / `<\|think_end\|>` | thinking 块边界 |
| `<\|tool_call_start\|>` / `<\|tool_call_end\|>` | assistant 工具调用的规范化 JSON 边界 |

注意：角色名（system/user/assistant/tool）**不是** special token。tokenizers 会把词表内
special token 当作 added token 在任意输入文本中匹配，若将 "user" 等设为 special token，
预训练语料中这些普通词会被编码为控制 token id，污染 SFT 控制语义。角色名以纯文本形式
出现在 `<|im_start|>` 之后（见 `special_tokens.build_chat_template()`）。
保存 tokenizer 时同时写出 `chat_template`、模板 SHA-256 与
`special_tokens_map.json`；SFT release 必须绑定模板和 chat contract hash。

## 使用

`tokenizers/yf-{32k,64k,128k}-bpe` 是 2026-09-26 的 M2 pilot 工件，缺少当前合同中的两个
tool-call token，不能作为正式 T0。新一代训练语料必须由 snapshot-bound S6 train order 构建：

```bash
python data-process/scripts/sample_tokenizer_corpus.py \
  --order data/processed/mix_order/mix_m3_e1_100b_stage2.parquet \
  --s3-root data/processed/s3 \
  --snapshot data/processed/releases/mix_m3_e1_100b-20261019.release.json \
  --recipe tokenizer/configs/corpus_m3_coverage_balanced.yaml \
  --out-root data/tokenizer_corpus_m3_20261019

python data-process/scripts/audit_tokenizer_corpus.py \
  --corpus data/tokenizer_corpus_m3_20261019 \
  --snapshot data/processed/releases/mix_m3_e1_100b-20261019.release.json \
  --order data/processed/mix_order/mix_m3_e1_100b_stage2.parquet \
  --recipe tokenizer/configs/corpus_m3_coverage_balanced.yaml \
  --val-dir data/processed/val_sets \
  --out data/processed/audits/tokenizer_corpus_m3_20261019.json
```

三档 BBPE 共享该 manifest 中完全相同的文档。2026-10-20 的正式工件位于
`tokenizers/m3_20261020/`，三档 artifact audit 均 PASS。release-bound 九域 held-out E0 最终
选择 `yf-64k-bpe`：相对 32K 少 6.55% token，训练 FLOPs/原始字节代理仅高 0.58%；128K
相对 64K 少 5.15% token，却多 8.31% FLOPs/字节。机器门见 `data/TOKENIZER_READY.json`。

正式训练使用 `--formal`，它会先全量复核 corpus manifest 中每个 parquet 的大小和 SHA-256，
将 wall time、环境版本、实际样本/byte 数、特殊 token ID 与全部输出哈希写进
`train_meta.json`，并以目录原子发布。完成后运行：

```bash
python tokenizer/scripts/audit_tokenizer_artifact.py \
  --tokenizer tokenizer/tokenizers/candidates/m3-20261019/yf-64k-bpe \
  --out tokenizer/reports/yf-64k-bpe-m3.artifact-audit.json
```

只有该审计、独立 held-out E0 指标和数据 release 均 PASS 后才签发 `tokenizer_id`。正式 E0
语料和结果分别位于 `data/tokenizer_eval/e0_heldout_m3_20261020/` 与
`experiments/e0_tokenizer/results/e0_m3_heldout_metrics.json`。

```bash
# 训练
PYTHONPATH=src python scripts/train_tokenizer.py \
  --corpus data/fineweb-edu/ data/proof-pile-2/ \
  --vocab-size 65536 --model bpe --output tokenizers/yf-64k-bpe

# 评估
PYTHONPATH=src python scripts/evaluate_tokenizer.py \
  --tokenizer tokenizers/yf-64k-bpe --eval-corpus data/wikipedia-en/

# 全量 tokenize
PYTHONPATH=src python scripts/tokenize_dataset.py \
  --tokenizer tokenizers/yf-64k-bpe \
  --input data/fineweb-edu/ --output data-tokenized/fineweb-edu/

# 对比
PYTHONPATH=src python scripts/compare_tokenizers.py \
  --tokenizers tokenizers/yf-32k-bpe tokenizers/yf-64k-bpe \
  --eval-corpus data/wikipedia-en/ data/proof-pile-2/
```
