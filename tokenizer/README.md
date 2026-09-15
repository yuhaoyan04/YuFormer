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
| `<\|endoftext\|>` | 文档结束 / padding |
| `<\|im_start\|>` | 消息块开始 |
| `<\|im_end\|>` | 消息块结束 |
| `system` / `user` / `assistant` / `tool` | 角色标识 |
| ` thinking_start ` / ` thinking_end ` | thinking 块边界 |

## 使用

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
