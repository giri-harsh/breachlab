"""Guard prompt, label <-> verbalizer mapping and head+tail truncation (PRD 7.2)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from breachlab.data.schema import Row

LABELS = ("safe", "injection", "jailbreak", "exfiltration")
Kind = Literal["input", "output"]

# <= 60 tokens: the fine-tune carries the knowledge, not the prompt.
GUARD_SYSTEM = {
    "input": (
        "You are a security guard for an AI vault. Classify the user message with one word: "
        "{words}."
    ),
    "output": (
        "You are a security guard for an AI vault. Classify the assistant reply with one word: "
        "{words}."
    ),
}
# Qwen3 non-thinking mode: an empty think block precedes the answer.
THINK_PREFIX = "<think>\n\n</think>\n\n"


def label_index(row: Row) -> int:
    """Index into LABELS for a row (safe, or its attack category)."""
    return 0 if row.label == "SAFE" else LABELS.index(row.category)


def render_prompt(text: str, words: Sequence[str], kind: Kind = "input") -> str:
    """ChatML prompt ending where the answer token is predicted (one forward pass)."""
    system = GUARD_SYSTEM[kind].format(words=", ".join(words))
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{text}<|im_end|>\n"
        f"<|im_start|>assistant\n{THINK_PREFIX}"
    )


def prompt_parts(words: Sequence[str], kind: Kind = "input") -> tuple[str, str]:
    """(prefix, suffix) around the message; train and serve tokenize the three pieces separately."""
    full = render_prompt("\0", words, kind)
    prefix, suffix = full.split("\0")
    return prefix, suffix


def encode_prompt(
    encode: Callable[[str], list[int]],
    text: str,
    words: Sequence[str],
    kind: Kind = "input",
    limit: int = 512,
) -> list[int]:
    """prefix ids + head/tail-truncated message ids + suffix ids (identical in training and serving)."""
    prefix, suffix = prompt_parts(words, kind)
    return encode(prefix) + truncate_ids(encode(text), limit) + encode(suffix)


def truncate_ids(ids: list[int], limit: int) -> list[int]:
    """Keep head and tail when longer than ``limit`` (PRD 7.2)."""
    if len(ids) <= limit:
        return ids
    head = limit // 2
    return ids[:head] + ids[len(ids) - (limit - head) :]
