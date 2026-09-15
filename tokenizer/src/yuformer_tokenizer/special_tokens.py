from __future__ import annotations

DEFAULT_SPECIAL_TOKENS: list[str] = [
    "<|endoftext|>",
    "<|unk|>",
    "<|im_start|>",
    "<|im_end|>",
    "system",
    "user",
    "assistant",
    "tool",
    "<|think_start|>",
    "<|think_end|>",
]

MINIMAL_SPECIAL_TOKENS: list[str] = [
    "<|endoftext|>",
    "<|unk|>",
]

ROLE_TOKENS: dict[str, str] = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
}

THINK_START = "<|think_start|>"
THINK_END = "<|think_end|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
EOS_TOKEN = "<|endoftext|>"
PAD_TOKEN = "<|endoftext|>"
UNK_TOKEN = "<|unk|>"


def build_chat_template() -> str:
    return (
        "{% for message in messages %}"
        "<|im_start|>{{ message.role }}\n"
        "{% if message.role == 'assistant' and message.reasoning_content %}"
        "<|think_start|>{{ message.reasoning_content }}<|think_end|>\n"
        "{% endif %}"
        "{{ message.content }}"
        "<|im_end|>\n"
        "{% endfor %}"
    )
