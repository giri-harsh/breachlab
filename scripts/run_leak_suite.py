"""System leak suite (PRD 9.1) through the real GameService: Groq vault agent at temperature 0, canary sessions.

Levels needing the guard service (3-5) run only when GUARD_URL is set; otherwise they are reported as not run
(nothing is stored for them, so a later run with GUARD_URL set picks them up). Writes data/eval/leak_suite.json
(resumable per level: a level with a stored result is skipped). A level where more than half the requests fail
(dead guard host, agent quota) is not stored either.
Usage: python scripts/run_leak_suite.py [levels...] [--subset 54] [--out PATH] [--attacks FRESH.jsonl]
``--subset 54`` runs the fixed tuning subset (every 4th attack of the 216, always the same 54); ``--out`` writes to
another results file (its progress file sits next to it), so tuning runs never overwrite the final table.
Level 3 runs on the champion guard (GUARD_URL) with the lenient per-level ``guard_threshold`` from
configs/levels.yaml (supervisor decision A3); GUARD_BASE_URL is not used.

Infrastructure failures are never counted as attacks: a request is valid only if it got a vault reply or a block.
A failed request (guard waking up, vault busy) is retried with backoff, re-warming the guard first; progress is
checkpointed per attack in data/eval/leak_suite_progress.json so a cut-off run resumes where it stopped. After
``STOP_AFTER`` consecutive attacks that exhaust their retries with the same error the level stops, and the stored
entry records the cause and the n reached (``partial``, so a later run resumes it).

Env: GROQ_API_KEY, GUARD_URL (optional), GUARD_TOKEN (bearer, never printed), GUARD_TIMEOUT_S (default 8, as the
gateway), AGENT_MODELS (comma list, default AGENT_MODEL, default gpt-oss-20b). At temperature 0 the agent uses
only the first model, so one run never mixes models.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import httpx
import sqlalchemy as sa
from sqlalchemy.orm import Session

from breachlab.common.db import Base
from breachlab.eval.system_leak_suite import LevelResult, attack_suite, suite_hash, wilson
from breachlab.gateway.agent import AgentError, AgentReply, GroqAgent, parse_models
from breachlab.gateway.guards import HttpGuard, RegexGuard
from breachlab.gateway.levels import LevelsFile, load_levels
from breachlab.gateway.service import GameError, GameService, GuardSet

OUT = Path("data/eval/leak_suite.json")
PROGRESS = Path("data/eval/leak_suite_progress.json")
BACKOFF_S = (3.0, 8.0, 15.0, 30.0)  # waits between the (len + 1) tries of one attack
STOP_AFTER = 3  # consecutive attacks failing with the same error => stop the level
MIN_VALID = 200
NO_HOST = "not run: no guard host (ADR-0063)"


def agent_models() -> list[str]:
    """AGENT_MODELS (comma list) else AGENT_MODEL else gpt-oss-20b (120b hit the 200k tokens/day free-tier cap)."""
    raw = os.environ.get("AGENT_MODELS") or os.environ.get("AGENT_MODEL") or "openai/gpt-oss-20b"
    return parse_models(raw) or ["openai/gpt-oss-20b"]


def pending_levels(
    done: dict[str, dict[str, object]], have_guard: bool
) -> tuple[list[int], list[int]]:
    """(levels to run now, levels skipped because there is no guard). A stored level without a rate is not done."""
    todo = [
        lv
        for lv in (1, 2, 3, 4, 5)
        if "rate" not in done.get(str(lv), {}) or done[str(lv)].get("partial")
    ]
    return [lv for lv in todo if lv < 3 or have_guard], [
        lv for lv in todo if lv >= 3 and not have_guard
    ]


class PatientAgent:
    """Groq agent with sleep-and-retry on 429 (free-tier TPM)."""

    def __init__(self, inner: GroqAgent) -> None:
        self.inner = inner

    def reply(self, level: int, secret: str, message: str, temperature: float = 0.7) -> AgentReply:
        for i in range(8):
            try:
                return self.inner.reply(level, secret, message, temperature)
            except AgentError:
                time.sleep(5 * (i + 1))
        raise AgentError("agent unavailable")


def warm(url: str, token: str, client: httpx.Client) -> bool:
    """/health then one /score, so the instance is booted before attacks are sent. Never raises, never logs the token."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        client.get(f"{url}/health", headers=headers, timeout=30.0)
        r = client.post(
            f"{url}/score",
            json={"text": "warm-up", "kind": "input", "alias": "champion"},
            headers=headers,
            timeout=30.0,
        )
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def send_valid(
    svc: GameService, sid: str, level: int, msg: str, rewarm: Callable[[], object]
) -> tuple[dict[str, object] | None, str]:
    """One attack with retries. Returns (reply, "") when valid, else (None, last error) after the tries are used."""
    err = ""
    for i in range(len(BACKOFF_S) + 1):
        try:
            return svc.send(sid, level, msg), ""
        except GameError as exc:
            err = f"{exc.status} {exc.detail}"
        except Exception as exc:  # noqa: BLE001 - any infra error is retried and, if persistent, recorded
            err = type(exc).__name__
        if i < len(BACKOFF_S):
            if "guard" in err:
                rewarm()
            time.sleep(BACKOFF_S[i])
    return None, err


def level_hash(levels: LevelsFile, level: int) -> str:
    """Hash of everything in configs/levels.yaml that changes a level's outcome (prompt, guard rule, keyword wall)."""
    cfg = levels.level(level).model_dump(exclude={"hints", "boss", "name", "demonstrates"})
    wall = levels.keyword_wall if level == 2 else None
    return hashlib.sha256(json.dumps([cfg, wall], sort_keys=True).encode()).hexdigest()[:12]


def load_progress(attacks: list[str], hashes: dict[int, str]) -> dict[str, dict[str, object]]:
    """Per-level checkpoints for this attack list and this level config (a prompt change starts the level afresh)."""
    if not PROGRESS.exists():
        return {}
    data: dict[str, dict[str, object]] = json.loads(PROGRESS.read_text())
    h = suite_hash(attacks)
    return {
        lv: v
        for lv, v in data.items()
        if v.get("suite_hash") == h
        and v.get("level_hash", hashes.get(int(lv))) == hashes.get(int(lv))
    }


def subset(attacks: list[str], n: int) -> list[str]:
    """The fixed tuning subset: every (len // n)-th attack, so all 12 intents and every wrapper family appear."""
    if n <= 0 or n >= len(attacks):
        return attacks
    step = len(attacks) // n
    return attacks[::step][:n]


def main(
    only: Sequence[int] = (),
    n_subset: int = 0,
    out: Path | None = None,
    attacks_file: Path | None = None,
) -> int:
    global OUT, PROGRESS
    if out is not None:
        OUT, PROGRESS = out, out.with_name(out.stem + "_progress.json")
    guard_url = os.environ.get("GUARD_URL", "")
    token = os.environ.get("GUARD_TOKEN", "")
    gclient = httpx.Client()
    timeout = float(os.environ.get("GUARD_TIMEOUT_S", "8"))
    http = (
        HttpGuard(guard_url, timeout, token=token, client=gclient) if guard_url else None
    )  # same wiring as the gateway
    models = agent_models()
    engine = sa.create_engine("sqlite://")
    Base.metadata.create_all(engine)
    if (
        attacks_file is not None
    ):  # a fresh, non-template attack set (one JSON object with "text" per line)
        attacks = [
            json.loads(x)["text"] for x in attacks_file.read_text("utf-8").split("\n") if x.strip()
        ]
    else:
        attacks = subset(attack_suite(), n_subset)
    done: dict[str, dict[str, object]] = json.loads(OUT.read_text()) if OUT.exists() else {}
    levels = load_levels()
    hashes = {lv: level_hash(levels, lv) for lv in (1, 2, 3, 4, 5)}
    stale = {  # entries from before level hashes existed are kept as they are
        lv
        for lv, v in done.items()
        if v.get("level_hash", hashes.get(int(lv))) != hashes.get(int(lv))
    }
    pending_done = {
        lv: v for lv, v in done.items() if lv not in stale
    }  # a changed level is run again
    progress = load_progress(attacks, hashes)
    with Session(engine) as db:
        svc = GameService(
            db,
            GuardSet(regex=RegexGuard(), base=http, champion=http),
            PatientAgent(GroqAgent(os.environ["GROQ_API_KEY"], models)),
            "leak-suite-key",
            "salt",
            levels=levels,
            agent_temperature=0.0,
        )
        run, skipped = pending_levels(pending_done, http is not None)
        for level in skipped:
            print(level, NO_HOST, flush=True)
        for level in [lv for lv in run if not only or lv in only]:
            if http is not None and level >= 3:
                warm(guard_url, token, gclient)
            sid = str(
                svc.start_session(f"Leak Suite {level}", "10.0.0.1", is_canary=True)["session_id"]
            )
            prog = progress.setdefault(
                str(level),
                {
                    "suite_hash": suite_hash(attacks),
                    "level_hash": hashes[level],
                    "outcomes": {},
                    "failed": {},
                },
            )
            outcomes: dict[str, list[Any]] = prog["outcomes"]  # type: ignore[assignment]
            failed: dict[str, str] = {}
            prog["failed"] = failed
            streak, last_err, stopped = 0, "", ""
            for idx, msg in enumerate(attacks):
                if str(idx) in outcomes:
                    continue
                reply, err = send_valid(
                    svc, sid, level, msg, lambda: warm(guard_url, token, gclient)
                )
                if reply is None:
                    failed[str(idx)] = err
                    streak = streak + 1 if err == last_err else 1
                    last_err = err
                    print(level, idx, "failed after retries:", err, flush=True)
                    if streak >= STOP_AFTER:
                        stopped = f"{streak} consecutive attacks failed with: {err}"
                        break
                else:
                    streak, last_err = 0, ""
                    verdict = reply.get("verdict") or {}
                    p_attack = verdict.get("p_attack") if isinstance(verdict, dict) else None
                    outcomes[str(idx)] = [
                        bool(reply.get("blocked")),
                        bool(reply["leaked"]),
                        p_attack,
                    ]
                PROGRESS.parent.mkdir(parents=True, exist_ok=True)
                PROGRESS.write_text(json.dumps(progress))
            PROGRESS.write_text(json.dumps(progress))
            n = len(outcomes)
            leaks = sum(o[1] for o in outcomes.values())
            blocked = sum(o[0] for o in outcomes.values())
            errors = len(attacks) - n
            if n * 2 < len(attacks) and not stopped:  # most requests failed: no meaningless rate
                print(level, f"not stored: {errors}/{len(attacks)} requests failed", flush=True)
                continue
            if n == 0:
                print(level, f"not stored: no valid attack ({stopped})", flush=True)
                continue
            lo, hi = wilson(leaks, n)
            r = LevelResult(level, n, leaks, leaks / n, lo, hi)
            done[str(level)] = {
                **r.__dict__,
                "errors": errors,
                "blocked": blocked,
                "suite_hash": suite_hash(attacks),
                "level_hash": hashes[level],
                "guard_threshold": levels.level(level).guard_threshold,
                "subset": n_subset or None,
                "agent": models[0],
                "temperature": 0,
                **(
                    {"partial": True, "stopped": stopped, "n_min": min(MIN_VALID, len(attacks))}
                    if errors and n < min(MIN_VALID, len(attacks))
                    else {}
                ),
            }
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(done, indent=2))
            print(level, done[str(level)], flush=True)
    OUT.write_text(json.dumps(done, indent=2))
    return 0


def cli(argv: Sequence[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("levels", nargs="*", type=int)
    ap.add_argument(
        "--attacks",
        type=Path,
        help='JSONL of fresh attacks ({"text": ...} per line) instead of the fixed suite',
    )
    ap.add_argument("--subset", type=int, default=0, help="run the fixed N-attack tuning subset")
    ap.add_argument(
        "--out", type=Path, default=None, help="results file (default data/eval/leak_suite.json)"
    )
    a = ap.parse_args(list(argv))
    return main(a.levels, a.subset, a.out, a.attacks)


if __name__ == "__main__":
    sys.exit(cli(sys.argv[1:]))
