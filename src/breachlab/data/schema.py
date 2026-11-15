"""Row schema (PRD 8.2): pydantic model and pandera frame schema."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

Label = Literal["SAFE", "UNSAFE"]
Category = Literal["injection", "jailbreak", "exfiltration", "none"]
Split = Literal["train", "val", "mining_pool", "gold_test", "ood_test", "unassigned"]
Language = Literal["en", "hi", "hinglish"]
Encoding = Literal["none", "base64", "hex", "leet", "rot13", "split"]
LengthBucket = Literal["short", "medium", "long"]

COLUMNS = [
    "id",
    "text",
    "label",
    "category",
    "source",
    "attack_family",
    "language",
    "encoding",
    "length_bucket",
    "split",
    "created_at",
    "parent_id",
]


def length_bucket(text: str) -> LengthBucket:
    n = len(text)
    return "short" if n < 80 else "medium" if n < 250 else "long"


def make_id(text: str, source: str) -> str:
    return hashlib.sha256(f"{source}\x00{text}".encode()).hexdigest()[:16]


_DEVANAGARI = range(0x0900, 0x0980)


def detect_language(text: str, default: Language = "en") -> Language:
    if any(ord(c) in _DEVANAGARI for c in text):
        return "hi"
    return default


class Row(BaseModel):
    id: str
    text: str = Field(min_length=1)
    label: Label
    category: Category
    source: str
    attack_family: str = "none"
    language: Language = "en"
    encoding: Encoding = "none"
    length_bucket: LengthBucket = "short"
    split: Split = "unassigned"
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    parent_id: str | None = None

    @classmethod
    def new(
        cls,
        text: str,
        label: Label,
        category: Category,
        source: str,
        *,
        attack_family: str = "none",
        language: Language = "en",
        encoding: Encoding = "none",
        parent_id: str | None = None,
    ) -> Row:
        if label == "SAFE":
            category = "none"
        return cls(
            id=make_id(text, source),
            text=text,
            label=label,
            category=category,
            source=source,
            attack_family=attack_family,
            language=detect_language(text, language),
            encoding=encoding,
            length_bucket=length_bucket(text),
            parent_id=parent_id,
        )

    def with_(self, **kw: Any) -> Row:
        return self.model_copy(update=kw)


def jsonl_lines(text: str) -> list[str]:
    """Split JSONL on \\n only: str.splitlines() also breaks on U+2028/U+0085 inside JSON strings."""
    return [x for x in text.split("\n") if x.strip()]


def validate_rows(rows: list[Row]) -> None:
    """Raise pandera SchemaError (or ValueError) if rows violate the PRD 8.2 schema."""
    import pandas as pd
    import pandera.pandas as pa

    schema = pa.DataFrameSchema(
        {
            "id": pa.Column(str, unique=True),
            "text": pa.Column(str, pa.Check.str_length(min_value=1)),
            "label": pa.Column(str, pa.Check.isin(["SAFE", "UNSAFE"])),
            "category": pa.Column(
                str, pa.Check.isin(["injection", "jailbreak", "exfiltration", "none"])
            ),
            "source": pa.Column(str),
            "attack_family": pa.Column(str),
            "language": pa.Column(str, pa.Check.isin(["en", "hi", "hinglish"])),
            "encoding": pa.Column(
                str, pa.Check.isin(["none", "base64", "hex", "leet", "rot13", "split"])
            ),
            "length_bucket": pa.Column(str, pa.Check.isin(["short", "medium", "long"])),
            "split": pa.Column(
                str,
                pa.Check.isin(
                    ["train", "val", "mining_pool", "gold_test", "ood_test", "unassigned"]
                ),
            ),
            "created_at": pa.Column(str),
            "parent_id": pa.Column(str, nullable=True),
        },
        checks=pa.Check(
            lambda df: (df["label"] == "SAFE") == (df["category"] == "none"),
            error="SAFE iff category none",
        ),
        strict=True,
    )
    df = pd.DataFrame([r.model_dump() for r in rows], columns=COLUMNS)
    schema.validate(df)
