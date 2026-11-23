# YuFormer SFT

SFT 与 pretrain 共用已锁定的 TorchTitan v0.3.0 后端。训练入口在 YuFormer BaseModel adapter、
正式 tokenizer 与 base checkpoint 通过门禁后开放；当前先固定数据契约和审计器。

正式样本采用 JSONL，每行必须包含：

- `sample_id`：全数据集唯一稳定 ID；
- `source`、`source_revision`、`license`：来源与不可变版本；
- `split`：`train|validation|test`；
- `messages`：OpenAI 风格多轮消息，允许 `system/user/assistant/tool`；
- assistant 的 `reasoning_content` 可选且与最终 `content` 分开；tool call 必须有唯一 id，
  `function.arguments` 必须是 JSON object 或编码 object 的合法 JSON 字符串，每个 tool
  response 必须引用尚未消费的 call id。

```bash
python sft/scripts/audit_dataset.py \
  --input data/sft/corpus.jsonl \
  --out data/sft/corpus.audit.json
```

审计要求 ID 唯一、消息状态机合法、至少有一个非空 assistant 监督目标、来源 revision 为
40–64 位十六进制 commit/content hash，并拒绝跨 split 的规范化对话重复。正式训练还必须把
tokenizer/chat-template hash、逐 split 内容 hash、去污染结果和人工签收写入 SFT release manifest。

`yuformer_sft.packing` 是规范渲染与打包实现：tool call 使用与 tokenizer 内嵌 Jinja
`tojson` 模板逐字节一致的排序、ASCII/HTML-safe JSON；tool response
保留 `tool_call_id`；`reasoning_content`、最终内容、tool call 与 assistant 结束边界参与监督，
system/user/tool 全部为 `IGNORE_INDEX=-100`。它按完整对话做 greedy packing，并输出
`tokens/targets/cu_seqlens`；超长对话直接报错，不静默截断。新增的
`<|tool_call_start|>/<|tool_call_end|>` 改变了 tokenizer 契约，所以历史 tokenizer 不兼容，
E0 重跑时必须重新训练并记录 chat contract hash。

TorchTitan SFT 使用 assistant-only loss mask 与 packed block-causal attention；每个
`cu_seqlens` 区间重置 position id 且禁止跨对话注意。训练验收同时报告每来源/任务 loss、
有效 assistant token 比例、packing 利用率、长度截断率和 base 能力回归。
