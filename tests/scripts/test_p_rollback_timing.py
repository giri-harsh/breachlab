"""Pure parts of the rollback timing script (PRD 10.8) with a fake registry. No network."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "p_rollback_timing.py"
spec = importlib.util.spec_from_file_location("p_rollback_timing", SCRIPT)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["p_rollback_timing"] = mod
spec.loader.exec_module(mod)


class Fake:
    def __init__(self) -> None:
        self.aliases: dict[tuple[str, str], str] = {}

    def get_model_version_by_alias(self, name: str, alias: str) -> Any:
        if (name, alias) not in self.aliases:
            raise KeyError(alias)
        return SimpleNamespace(version=self.aliases[(name, alias)])

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.aliases[(name, alias)] = version

    def delete_registered_model_alias(self, name: str, alias: str) -> None:
        self.aliases.pop((name, alias), None)


def test_refuses_the_production_model() -> None:
    with pytest.raises(ValueError):
        mod.check_name("breachlab-guard")
    with pytest.raises(ValueError):
        mod.check_name("other-model")
    assert mod.check_name(mod.TEST_MODEL) == mod.TEST_MODEL


def test_cycle_returns_to_base_and_summarizes() -> None:
    f = Fake()
    name = mod.TEST_MODEL
    f.set_registered_model_alias(name, "champion", "1")
    cycles = [mod.time_cycle(f, name, "2", "1") for _ in range(3)]
    assert f.aliases[(name, "champion")] == "1"
    assert f.aliases[(name, "previous")] == "2"
    s = mod.summarize(cycles)
    assert s["alias_move_within_budget"] is True
    assert set(s["rollback_s"]) == {"min", "median", "max"}
    assert mod.summarize(cycles, budget_s=0.0)["alias_move_within_budget"] is False


def test_cycle_detects_a_wrong_readback() -> None:
    f = Fake()
    name = mod.TEST_MODEL
    f.set_registered_model_alias(name, "champion", "1")
    with pytest.raises(RuntimeError):
        mod.time_cycle(f, name, "2", "3")
