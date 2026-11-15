"""CPU guard runtime: tokenizer + ONNX session -> 4 verbalizer logits -> verdict (PRD 7.2).

The ONNX/tokenizer objects are injected, so this module is unit-tested with fakes and reused unchanged by
the guard service and the latency harness. Prefix-KV reuse lives in the ONNX backend (``onnx_backend``), not here.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from breachlab.train.prompts import Kind, encode_prompt
from breachlab.train.scoring import Verdict, decide

Alias = Literal["champion", "challenger"]


Encoder = Callable[[str], list[int]]


LogitsFn = Callable[[list[int]], NDArray[np.float64]]


@dataclass(frozen=True)
class GuardParams:
    version: str
    temperature: float = 1.0
    threshold: float = 0.5


@dataclass(frozen=True)
class GuardResult:
    verdict: Verdict
    version: str
    latency_ms: float
    stage: str | None = None  # cascade only: student | teacher | failed_closed


class GuardModel:
    def __init__(
        self,
        encode: Encoder,
        logits_fn: LogitsFn,
        words: list[str],
        params: GuardParams,
        max_tokens: dict[str, int],
    ) -> None:
        self.encode, self.logits_fn, self.words = encode, logits_fn, words
        self.params, self.max_tokens = params, max_tokens

    def score(self, text: str, kind: Kind = "input") -> GuardResult:
        t0 = time.perf_counter()
        limit = self.max_tokens["input_guard" if kind == "input" else "output_filter"]
        ids = encode_prompt(self.encode, text, self.words, kind, limit)
        v = decide(self.logits_fn(ids), self.params.temperature, self.params.threshold)
        return GuardResult(v, self.params.version, (time.perf_counter() - t0) * 1000)


class GuardService:
    """Champion and challenger side by side; unknown alias falls back to the champion."""

    def __init__(self, models: dict[str, GuardModel]) -> None:
        if "champion" not in models:
            raise ValueError("a champion model is required")
        self.models = models

    def score(self, text: str, kind: Kind = "input", alias: str = "champion") -> GuardResult:
        return self.models.get(alias, self.models["champion"]).score(text, kind)

    def shadow(self, text: str, kind: Kind = "input") -> tuple[GuardResult, GuardResult | None]:
        champ = self.models["champion"].score(text, kind)
        ch = self.models.get("challenger")
        return champ, (ch.score(text, kind) if ch else None)
