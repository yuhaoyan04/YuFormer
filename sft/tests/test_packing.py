import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[2] / "tokenizer" / "src"))

from jinja2 import Environment
from yuformer_tokenizer.special_tokens import build_chat_template

from yuformer_sft.packing import (
    IGNORE_INDEX,
    REQUIRED_SPECIAL_TOKENS,
    encode_chat,
    pack_conversations,
    render_chat,
)


class Encoding:
    def __init__(self, ids):
        self.ids = ids


class FixtureTokenizer:
    def __init__(self):
        self.vocab = {token: index for index, token in enumerate(REQUIRED_SPECIAL_TOKENS)}
        self.next_id = len(self.vocab)
        self.pattern = re.compile("(" + "|".join(map(re.escape, REQUIRED_SPECIAL_TOKENS)) + ")")

    def token_to_id(self, token):
        return self.vocab.get(token)

    def encode(self, text, add_special_tokens=False):
        ids = []
        for piece in filter(None, self.pattern.split(text)):
            if piece in self.vocab:
                ids.append(self.vocab[piece])
            else:
                for character in piece:
                    if character not in self.vocab:
                        self.vocab[character] = self.next_id
                        self.next_id += 1
                    ids.append(self.vocab[character])
        return Encoding(ids)


def test_tool_trace_is_rendered_and_only_assistant_is_supervised():
    messages = [
        {"role": "user", "content": "weather"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "need tool",
            "tool_calls": [
                {"id": "call-1", "function": {"name": "weather", "arguments": {"city": "北京"}}}
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "sunny"},
        {"role": "assistant", "content": "It is sunny."},
    ]
    tokenizer = FixtureTokenizer()
    rendered = render_chat(messages)
    assert 'tool_call_id="call-1"' in rendered
    assert '<|tool_call_start|>{"function"' in rendered
    encoded = encode_chat(messages, tokenizer)
    assert len(encoded.input_ids) == len(encoded.labels)
    assert any(token != IGNORE_INDEX for token in encoded.labels)
    user_ids = tokenizer.encode("weather").ids
    first_user_position = next(
        index
        for index in range(len(encoded.input_ids) - len(user_ids) + 1)
        if encoded.input_ids[index : index + len(user_ids)] == user_ids
    )
    assert encoded.labels[first_user_position : first_user_position + len(user_ids)] == [
        IGNORE_INDEX
    ] * len(user_ids)


def test_python_renderer_matches_saved_jinja_template_byte_for_byte():
    messages = [
        {"role": "system", "content": "Use tools safely."},
        {"role": "user", "content": "北京 weather <now>?"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "check & report",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call-'1'",
                    "function": {
                        "name": "weather",
                        "arguments": {"city": "北京", "filter": "<current>"},
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-'1'", "content": "sunny"},
        {"role": "assistant", "content": "It is sunny."},
    ]
    jinja_rendered = Environment().from_string(build_chat_template()).render(
        messages=messages,
        add_generation_prompt=False,
    )
    assert render_chat(messages) == jinja_rendered


def test_packing_preserves_boundaries_and_rejects_oversize():
    tokenizer = FixtureTokenizer()
    first = encode_chat(
        [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
        tokenizer,
    )
    second = encode_chat(
        [{"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}],
        tokenizer,
    )
    max_tokens = len(first.input_ids) + len(second.input_ids)
    packs = pack_conversations([first, second], max_tokens=max_tokens)
    assert len(packs) == 1
    assert packs[0]["cu_seqlens"] == [
        0,
        len(first.input_ids),
        max_tokens,
    ]
    assert len(packs[0]["tokens"]) == len(packs[0]["targets"])

    try:
        pack_conversations([first], max_tokens=len(first.input_ids) - 1)
    except ValueError as exc:
        assert "exceeds" in str(exc)
    else:
        raise AssertionError("oversize conversation was not rejected")
