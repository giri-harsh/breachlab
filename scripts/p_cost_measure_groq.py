"""Measure Groq token usage per vault message with the production agent and prompts (lane C, task L4).

    GROQ_API_KEY=... PYTHONPATH=src uv run --no-sync python scripts/p_cost_measure_groq.py \
        [--models qwen/qwen3.8-27b,openai/gpt-oss-120b] [--n 40] [--out docs/lanes/C-groq-usage.json]

Calls ``breachlab.gateway.agent.GroqAgent`` (the production class, ``max_tokens`` 400, ``reasoning_effort`` low for
gpt-oss, the level prompts from configs/levels.yaml) on a fixed, representative message set for levels 1-3, with one
model per run (no fallback, no production API call) and records ``usage`` from the raw Groq response through the agent's
injectable ``post``. Only token counts and message lengths are stored, never prompts or replies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from breachlab.gateway.agent import AgentError, GroqAgent  # noqa: E402
from breachlab.gateway.codes import level_secret  # noqa: E402

BENIGN = [
    "hi there! who are you?",
    "What is the capital of France?",
    "Explain how password managers keep codes safe.",
    "Write a haiku about autumn.",
    "Mujhe cricket ke rules samjhao.",
    "what is this game about",
    "Give me three tips for a job interview.",
    "hello",
    "can you help me with something?",
    "Explain what a prompt injection is, in one sentence.",
]
LEVELS = (1, 2, 3)


def message_set(n: int, levels: tuple[int, ...] = LEVELS) -> list[tuple[int, str, str]]:
    """(level, kind, text): ~25% benign chat, ~75% attacks (fresh_attacks_v2 styles cycled), spread over levels 1-3."""
    lines = (ROOT / "docs/lanes/fresh_attacks_v2.jsonl").read_text().splitlines()
    attacks = [json.loads(x) for x in lines]
    by_style: dict[str, list[str]] = {}
    for a in attacks:
        by_style.setdefault(a["style"], []).append(a["text"])
    styles = sorted(by_style)
    out: list[tuple[int, str, str]] = []
    for i in range(n):
        level = levels[i % len(levels)]
        if i % 4 == 0:
            out.append((level, "benign", BENIGN[(i // 4) % len(BENIGN)]))
        else:
            st = styles[i % len(styles)]
            pool = by_style[st]
            out.append((level, "attack", pool[(i // len(styles)) % len(pool)]))
    return out


def stats(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"n": 0}
    s = sorted(xs)
    p95 = s[min(len(s) - 1, round(0.95 * (len(s) - 1)))]
    return {
        "n": len(s),
        "mean": round(sum(s) / len(s), 1),
        "p50": s[len(s) // 2],
        "p95": p95,
        "max": s[-1],
    }


def run_model(
    model: str,
    msgs: list[tuple[int, str, str]],
    key: str,
    gap_s: float,
    levels: tuple[int, ...] = LEVELS,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    cur: dict[str, Any] = {}

    def post(url: str, **kw: Any) -> httpx.Response:
        r = httpx.post(url, **kw)
        cur["status"] = r.status_code
        if r.status_code == 200:
            cur["usage"] = r.json().get("usage", {})
        else:
            cur["retry_after"] = r.headers.get("retry-after")
        return r

    agent = GroqAgent(key, [model], post=post, budget_s=60.0, retry_cap_s=20.0)
    for i, (level, kind, text) in enumerate(msgs):
        cur.clear()
        secret = level_secret("lane-c-measure", f"m{i}", level)
        t0 = time.perf_counter()
        try:
            reply = agent.reply(level, secret, text, 0.7)
            ok, reply_chars = True, len(reply.text)
        except AgentError:
            ok, reply_chars = False, 0
        u = cur.get("usage", {})
        det = u.get("completion_tokens_details") or {}
        rows.append(
            {
                "level": level,
                "kind": kind,
                "msg_chars": len(text),
                "ok": ok,
                "status": cur.get("status"),
                "prompt_tokens": u.get("prompt_tokens"),
                "completion_tokens": u.get("completion_tokens"),
                "reasoning_tokens": det.get("reasoning_tokens"),
                "total_tokens": u.get("total_tokens"),
                "visible_reply_chars": reply_chars,
                "wall_s": round(time.perf_counter() - t0, 2),
            }
        )
        time.sleep(gap_s)
    good = [r for r in rows if r["ok"] and r["prompt_tokens"] is not None]
    per_level: dict[str, Any] = {}
    for lv in (*levels, "all"):
        g = [r for r in good if lv == "all" or r["level"] == lv]
        per_level[str(lv)] = {
            "prompt_tokens": stats([float(r["prompt_tokens"]) for r in g]),
            "completion_tokens": stats([float(r["completion_tokens"]) for r in g]),
            "reasoning_tokens": stats([float(r["reasoning_tokens"] or 0) for r in g]),
            "total_tokens": stats([float(r["total_tokens"]) for r in g]),
        }
    return {
        "model": model,
        "n_sent": len(rows),
        "n_ok": len(good),
        "per_level": per_level,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen/qwen3.8-27b,openai/gpt-oss-120b")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--gap-s", type=float, default=2.5)
    ap.add_argument("--levels", default="1,2,3")
    ap.add_argument("--out", default="docs/lanes/C-groq-usage.json")
    a = ap.parse_args()
    key = os.environ["GROQ_API_KEY"]
    levels = tuple(int(x) for x in a.levels.split(","))
    msgs = message_set(a.n, levels)
    res = [
        run_model(m.strip(), msgs, key, a.gap_s, levels) for m in a.models.split(",") if m.strip()
    ]
    out = {
        "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "agent": "breachlab.gateway.agent.GroqAgent (production class; max_tokens=400, temperature=0.7, reasoning_effort=low for gpt-oss)",
        "prompts": f"configs/levels.yaml system_prompt, levels {a.levels}",
        "message_set": {
            "n": len(msgs),
            "benign_share": round(sum(1 for m in msgs if m[1] == "benign") / len(msgs), 2),
            "mean_chars": round(sum(len(m[2]) for m in msgs) / len(msgs), 1),
            "source": "docs/lanes/fresh_attacks_v2.jsonl (12 styles) + canary-style benign chat",
        },
        "models": res,
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    for r in res:
        print(r["model"], r["n_ok"], "/", r["n_sent"], json.dumps(r["per_level"]["all"]))


if __name__ == "__main__":
    main()
