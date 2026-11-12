"""Typed loader for configs/levels.yaml (PRD 5.1 levels, v1.1 game config: CONTRACTS 7)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "configs" / "levels.yaml"
MAIN_LEVELS = (1, 2, 3, 4, 5)  # the boss ladder; level 6 is the daily boss (CONTRACTS 7.7)


class LevelConfig(BaseModel):
    id: int = Field(ge=1, le=6)
    name: str
    boss: str = ""
    defense: str
    input_guard: str
    output_filter: bool
    demonstrates: str
    system_prompt: str | None = None  # {secret} is replaced; None = agent default
    guard_threshold: float | None = Field(None, ge=0.0, le=1.0)  # blocked iff p_unsafe >= this
    max_messages: int | None = None  # None = limits.max_messages_per_level
    daily: bool = False
    hints: list[str] = Field(default_factory=list)


class Limits(BaseModel):
    max_message_chars: int
    max_messages_per_level: int
    rate_limit_per_session_per_min: int
    rate_limit_per_ip_per_min: int
    hint_rate_per_session_per_min: int = 10
    skip_rate_per_session_per_min: int = 10
    admin_rate_per_ip_per_min: int = 5


class Scoring(BaseModel):
    base: list[int] = Field(default_factory=lambda: [100, 200, 400, 800, 1600])
    per_message_left: int = 10
    hint_cost: int = 100
    streak_step: float = 0.1
    streak_cap: float = 1.5

    def points(self, level: int, messages_left: int, hints_used: int, streak: int) -> int:
        """CONTRACTS 7.3: max(0, base + per_msg * left - hint_cost * hints) * min(1 + step * streak, cap)."""
        raw = max(
            0,
            self.base[level - 1]
            + self.per_message_left * messages_left
            - self.hint_cost * hints_used,
        )
        return int(round(raw * min(1 + self.streak_step * streak, self.streak_cap)))


class LevelsFile(BaseModel):
    levels: list[LevelConfig]
    limits: Limits
    scoring: Scoring = Field(default_factory=Scoring)
    keyword_wall: dict[str, list[str]] | None = None

    def level(self, level: int) -> LevelConfig:
        return next(lv for lv in self.levels if lv.id == level)

    def cap(self, level: int) -> int:
        cfg = self.level(level)
        return (
            cfg.max_messages if cfg.max_messages is not None else self.limits.max_messages_per_level
        )


def load_levels(path: Path = DEFAULT_PATH) -> LevelsFile:
    return LevelsFile.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@lru_cache(maxsize=1)
def default_levels() -> LevelsFile:
    """configs/levels.yaml, read once per process (the agent's prompts, the keyword wall)."""
    return load_levels()
