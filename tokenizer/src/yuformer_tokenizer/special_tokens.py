from __future__ import annotations

# 角色名（system/user/assistant/tool）刻意不作为 special token：
# tokenizers 会把词表中的 special token 当作 added token 在输入文本中匹配，
# 普通语料里的 "user" 等词会被编码成控制 token id，污染 SFT 控制语义。
# 角色以纯文本形式出现在 <|im_start|> 之后（见 build_chat_template）。
DEFAULT_SPECIAL_TOKENS: list[str] = [
    "<|endoftext|>",
    "<|unk|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|think_start|>",
    "<|think_end|>",
    "<|tool_call_start|>",
    "<|tool_call_end|>",
]

MINIMAL_SPECIAL_TOKENS: list[str] = [
    "<|endoftext|>",
    "<|unk|>",
]

# 角色名在模板中以纯文本出现（非 special token，见上方注释）。
ROLE_TOKENS: dict[str, str] = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
}

THINK_START = "<|think_start|>"
THINK_END = "<|think_end|>"
TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|endoftext|>"
UNK_TOKEN = "<|unk|>"


def build_chat_template() -> str:
    return (
        "{% for message in messages %}"
        "<|im_start|>{{ message.role }}\n"
        "{% if message.role == 'tool' and message.tool_call_id %}"
        "tool_call_id={{ message.tool_call_id | tojson }}\n"
        "{% endif %}"
        "{% if message.role == 'assistant' and message.reasoning_content %}"
        "<|think_start|>{{ message.reasoning_content }}<|think_end|>\n"
        "{% endif %}"
        "{{ message.content }}"
        "{% if message.role == 'assistant' and message.tool_calls %}"
        "{% for tool_call in message.tool_calls %}"
        "<|tool_call_start|>{{ tool_call | tojson }}<|tool_call_end|>"
        "{% endfor %}"
        "{% endif %}"
        "<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
