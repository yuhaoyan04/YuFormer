"""Canonical chat rendering and assistant-only THD packing."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Protocol

IGNORE_INDEX = -100
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
THINK_START = "<|think_start|>"
THINK_END = "<|think_end|>"
TOOL_CALL_START = "<|tool_call_start|>"
TOOL_CALL_END = "<|tool_call_end|>"
EOS_TOKEN = "<|endoftext|>"
REQUIRED_SPECIAL_TOKENS = (
    EOS_TOKEN,
    IM_START,
    IM_END,
    THINK_START,
    THINK_END,
    TOOL_CALL_START,
    TOOL_CALL_END,
)


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> Any: ...

    def token_to_id(self, token: str) -> int | None: ...


@dataclass(frozen=True)
class EncodedConversation:
    input_ids: list[int]
    labels: list[int]


def _canonical_json(value: Any) -> str:
    """Match Jinja's ``tojson`` filter byte-for-byte.

    Hugging Face renders the saved chat template with Jinja.  Keeping this
    implementation explicit prevents SFT preprocessing and inference from
    tokenizing the same tool trace differently.  Jinja's default JSON policy
    sorts keys, uses ``json.dumps`` defaults, and HTML-safes four characters.
    """
    rendered = json.dumps(value, sort_keys=True)
    return (
        rendered.replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("'", "\\u0027")
    )


def _payload(message: dict[str, Any]) -> str:
    role = message["role"]
    pieces: list[str] = []
    if role == "tool" and message.get("tool_call_id"):
        pieces.append(f"tool_call_id={_canonical_json(message['tool_call_id'])}\n")
    if role == "assistant" and message.get("reasoning_content"):
        pieces.extend(
            [THINK_START, str(message["reasoning_content"]), THINK_END, "\n"]
        )
    pieces.append(str(message.get("content") or ""))
    if role == "assistant":
        for tool_call in message.get("tool_calls") or []:
            pieces.extend(
                [TOOL_CALL_START, _canonical_json(tool_call), TOOL_CALL_END]
            )
    return "".join(pieces)


def render_chat(messages: list[dict[str, Any]]) -> str:
    """Render the exact human-readable form used by ``encode_chat``."""
    return "".join(
        f"{IM_START}{message['role']}\n{_payload(message)}{IM_END}\n"
        for message in messages
    )


def chat_contract_sha256() -> str:
    contract = {
        "schema": "yuformer_chat_v1",
        "special_tokens": REQUIRED_SPECIAL_TOKENS,
        "tool_json": "jinja-tojson-sort-keys-ascii-htmlsafe",
        "tool_id_prefix": "tool_call_id=<canonical-json>\\n",
        "assistant_loss": "payload+im_end+newline+eos",
        "other_loss": "ignore",
        "segmented_encoding": True,
    }
    return hashlib.sha256(_canonical_json(contract).encode("utf-8")).hexdigest()


def _ids(tokenizer: TokenizerLike, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    values = encoded.ids if hasattr(encoded, "ids") else encoded
    return [int(value) for value in values]


def validate_tokenizer_contract(tokenizer: TokenizerLike) -> dict[str, int]:
    ids: dict[str, int] = {}
    for token in REQUIRED_SPECIAL_TOKENS:
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            raise ValueError(f"tokenizer is missing required special token: {token}")
        encoded = _ids(tokenizer, token)
        if encoded != [token_id]:
            raise ValueError(f"special token is not atomic: {token} -> {encoded}")
        ids[token] = int(token_id)
    if len(set(ids.values())) != len(ids):
        raise ValueError("required special tokens do not have distinct IDs")
    return ids


def encode_chat(
    messages: list[dict[str, Any]],
    tokenizer: TokenizerLike,
    *,
    append_eos: bool = True,
) -> EncodedConversation:
    """Encode with stable segment boundaries and assistant-only supervision.

    The returned labels align with input IDs. A causal trainer shifts logits and
    labels by one position when computing cross entropy.
    """
    special_ids = validate_tokenizer_contract(tokenizer)
    input_ids: list[int] = []
    labels: list[int] = []
    last_role = None
    for message in messages:
        role = message["role"]
        header_ids = _ids(tokenizer, f"{IM_START}{role}\n")
        payload = _payload(message)
        payload_ids = _ids(tokenizer, payload) if payload else []
        end_ids = _ids(tokenizer, f"{IM_END}\n")
        input_ids.extend(header_ids)
        labels.extend([IGNORE_INDEX] * len(header_ids))
        supervised = role == "assistant"
        input_ids.extend(payload_ids)
        labels.extend(payload_ids if supervised else [IGNORE_INDEX] * len(payload_ids))
        input_ids.extend(end_ids)
        labels.extend(end_ids if supervised else [IGNORE_INDEX] * len(end_ids))
        last_role = role
    if append_eos:
        eos_id = special_ids[EOS_TOKEN]
        input_ids.append(eos_id)
        labels.append(eos_id if last_role == "assistant" else IGNORE_INDEX)
    if len(input_ids) != len(labels):
        raise AssertionError("input and label lengths diverged")
    return EncodedConversation(input_ids=input_ids, labels=labels)


def pack_conversations(
    conversations: Iterable[EncodedConversation], *, max_tokens: int
) -> list[dict[str, list[int]]]:
    """Greedily pack whole conversations and retain THD sequence boundaries."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    packs: list[dict[str, list[int]]] = []
    tokens: list[int] = []
    labels: list[int] = []
    cu_seqlens = [0]

    def flush() -> None:
        nonlocal tokens, labels, cu_seqlens
        if tokens:
            packs.append(
                {"tokens": tokens, "targets": labels, "cu_seqlens": cu_seqlens}
            )
        tokens, labels, cu_seqlens = [], [], [0]

    for index, conversation in enumerate(conversations):
        if len(conversation.input_ids) != len(conversation.labels):
            raise ValueError(f"conversation {index} input/label lengths differ")
        length = len(conversation.input_ids)
        if length == 0:
            raise ValueError(f"conversation {index} is empty")
        if length > max_tokens:
            raise ValueError(
                f"conversation {index} has {length} tokens and exceeds max_tokens={max_tokens}"
            )
        if tokens and len(tokens) + length > max_tokens:
            flush()
        tokens.extend(conversation.input_ids)
        labels.extend(conversation.labels)
        cu_seqlens.append(len(tokens))
    flush()
    return packs
