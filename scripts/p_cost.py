"""Cost per 1,000 requests and per 1,000 games from MEASURED usage (lane C, task L4; supersedes docs/lanes/P-cost.json).

    PYTHONPATH=src uv run --no-sync python scripts/p_cost.py [--guard-factor 1.0] [--no-shadow]
        [--out docs/lanes/C-cost.json] [--md docs/lanes/C-cost.md]

Reads (all produced by this repo, nothing production is queried here):
  docs/lanes/C-groq-usage.json, C-groq-usage-l45.json   Groq ``usage`` per vault call (scripts/p_cost_measure_groq.py)
  docs/lanes/C-gateway-cpu.json                           gateway CPU per request (scripts/p_cost_gateway_cpu.py)
  docs/lanes/B3-vercel-bench-warm.json, T-v2-latency.json guard model time on Vercel (1 vCPU)
  docs/lanes/G-difficulty-prod.json, P-full-request.json  guard block rates per level, benign block rate
and the list prices below (fetched from the providers' official pages on 2026-11-13, URLs in PRICING).

Definitions. A *request* is one ``POST /api/message``. A *game* is one player who clears or abandons the five
bosses (levels 1-5); messages per boss are an ASSUMPTION (SCENARIOS) because production is not queried; the hard cap is
20 messages per boss (configs/levels.yaml ``limits.max_messages_per_level``). 75% of messages are attacks and 25% are
benign chat (the mix of the Groq measurement set). The Gemini API is not in the serving path (offline judge only), so it
has no per-request cost. All figures are list prices for usage above the free tiers; the actual bill is $0.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LANES = ROOT / "docs/lanes"

PRICING: dict[str, Any] = {
    "fetched": "2026-11-13",
    "groq_usd_per_1m_tokens": {
        "openai/gpt-oss-20b": {"in": 0.075, "out": 0.30},
        "openai/gpt-oss-120b": {"in": 0.15, "out": 0.60},
        "qwen/qwen3.8-27b": {"in": 0.80, "out": 4.00},
        "source": "https://console.groq.com/docs/models (server-rendered table 'price per 1M tokens'; groq.com/pricing is client-rendered)",
    },
    "groq_limits": {
        "openai/gpt-oss-20b": {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000},
        "openai/gpt-oss-120b": {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000},
        "qwen/qwen3.8-27b": {"rpm": 30, "rpd": 1000, "tpm": 8000, "tpd": 200000},
        "source": "https://console.groq.com/docs/rate-limits shows these as the base (Developer-plan) limits; the Free-plan tab did not render, CLAUDE.md records the observed 200k tokens/day cap",
    },
    "vercel_fluid_iad1": {
        "active_cpu_usd_per_h": 0.128,
        "memory_usd_per_gb_h": 0.0106,
        "invocations_usd_per_m": 0.60,
        "source": "https://vercel.com/docs/functions/usage-and-pricing (regional table, Washington D.C. iad1; invocations: the page's $0.60 per million Pro example)",
    },
    "vercel_hobby_included": {
        "active_cpu_h": 4,
        "memory_gb_h": 360,
        "invocations": 1_000_000,
        "note": "per month, per team (all projects); exceeding Hobby limits pauses the feature for 30 days, no overage billing",
        "source": "https://vercel.com/docs/plans/hobby and https://vercel.com/docs/pricing",
    },
    "neon": {
        "launch_usd_per_cu_h": 0.106,
        "storage_usd_per_gb_month": 0.35,
        "free_cu_h_per_project_month": 100,
        "free_storage_gb_per_project": 1,
        "source": "https://neon.com/pricing",
    },
    "huggingface": {
        "public_repos": "free, best-effort storage; private repos 100 GB free",
        "source": "https://huggingface.co/docs/hub/storage-limits",
    },
}

# ASSUMPTIONS (explicit; change here or on the command line).
SCENARIOS = {  # messages per boss for levels 1-5
    "quick": [1, 2, 3, 5, 5],
    "typical": [2, 3, 5, 10, 10],
    "full_cap": [20, 20, 20, 20, 20],
}
HEADLINE = "typical"
ATTACK_SHARE = 0.75
# Fresh-attack block rates per level on production (docs/lanes/G-difficulty-prod.json fresh110); L1 has no guard.
ATTACK_BLOCKED = {1: 0.0, 2: 27 / 110, 3: 46 / 110, 4: 103 / 110, 5: 105 / 110}
BENIGN_BLOCKED = {
    1: 0.0,
    2: 0.0,
    3: 0.0,
    4: 0.10,
    5: 0.10,
}  # 5/50 benign blocked in P-full-request.json (guard v1)
GUARD_MEM_GB = 2.0  # breachlab-guard: 2 GB Hobby Fluid function (ADR-0080; project resourceConfig fluid=true, standard memory)
GATEWAY_MEM_GB = 2.0
GUARD_OVERHEAD_S = 0.03  # per /score call on top of model time (HTTP + JSON)
NET_S = 0.15  # gateway<->guard roundtrip minus model time (T-v2-latency: 872 ms roundtrip vs 722 ms model)
DB_WALL_S = 0.15  # assumed ~15 queries x 10 ms per message (not measured on Neon)
CPU_FACTOR = (
    3.0  # in-process CPU x3 for Postgres driver, TLS and JSON on real network I/O (not measured)
)
EXTRA_GATEWAY_CALLS = 1.0  # GET /api/session or similar per message (short)
EXTRA_CALL_WALL_S = 0.10
COLD = {
    "wall_s": 12.8,
    "cpu_s": 10.0,
}  # B3-vercel-bench-cold: 12.8 s first request, boot 8.6-9.7 s mostly CPU
# Neon row sizes (bytes), from src/breachlab/common/db.py: tuple header + fixed columns + text, plus PK index entry.
ATTEMPT_FIXED_B = (
    24 + 4 + 37 + 4 + 8 + 12 + 14 + 8 + 12 + 8 + 1 + 12 + 8 + 8 + 8 + 4 + 8 + 4
)  # = 189 incl. varlena heads
GAME_FIXED_B = 90 + 60 + 5 * 110 + 5 * 80  # session + player + 5 levels rows + <=5 leak labels
INDEX_B = 32


def load(name: str) -> Any:
    return json.loads((LANES / name).read_text())


def groq_tokens(model: str) -> dict[int, dict[str, float]]:
    """Per level mean prompt/completion tokens and wall seconds for one model (levels 1-5).

    gpt-oss-20b was not measured (its daily cap was spent): tokens are proxied by gpt-oss-120b (same family, tokenizer
    and reasoning_effort=low) and priced at the 20b list price.
    """
    model = "openai/gpt-oss-120b" if model == "openai/gpt-oss-20b" else model
    out: dict[int, dict[str, float]] = {}
    for fname in ("C-groq-usage.json", "C-groq-usage-l45.json"):
        for m in load(fname)["models"]:
            if m["model"] != model:
                continue
            for lv, v in m["per_level"].items():
                if lv == "all":
                    continue
                rows = [r for r in m["rows"] if r["ok"] and r["level"] == int(lv)]
                out[int(lv)] = {
                    "prompt": v["prompt_tokens"]["mean"],
                    "completion": v["completion_tokens"]["mean"],
                    "completion_p95": v["completion_tokens"]["p95"],
                    "prompt_p95": v["prompt_tokens"]["p95"],
                    "wall_s": sum(r["wall_s"] for r in rows) / len(rows),
                    "reply_chars": sum(r["visible_reply_chars"] for r in rows) / len(rows),
                    "n": float(len(rows)),
                }
    if 4 in out and 5 not in out:
        out[5] = out[4]
    return out


def vault_share(level: int) -> float:
    return ATTACK_SHARE * (1 - ATTACK_BLOCKED[level]) + (1 - ATTACK_SHARE) * (
        1 - BENIGN_BLOCKED[level]
    )


def guard_calls(level: int, shadow: bool) -> float:
    if level <= 2:
        return 0.0  # L1 no guard, L2 regex wall in the gateway process
    n = 1.0
    if level >= 4 and shadow:
        n += 1.0  # shadow challenger call to the same host (service.py), not used at L3 (guard_threshold set)
    if level == 5:
        n += vault_share(5)  # output filter scores the vault reply
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--guard-factor",
        type=float,
        default=1.0,
        help="guard CPU and memory time multiplier (cascade ~0.1-0.2)",
    )
    ap.add_argument("--no-shadow", action="store_true", help="drop the shadow guard call at L4-L5")
    ap.add_argument("--out", default="docs/lanes/C-cost.json")
    ap.add_argument("--md", default="docs/lanes/C-cost.md")
    a = ap.parse_args()

    gw_cpu = load("C-gateway-cpu.json")
    b3 = load("B3-vercel-bench-warm.json")["warm"]["c1"]
    t2 = load("T-v2-latency.json")
    guard_model_mean_s = b3["model_mean_ms"] / 1000  # 500 requests, c1, includes tail
    guard_model_p50_s = t2["model_ms_p50"] / 1000

    def run(models: tuple[str, ...], factor: float, shadow: bool) -> dict[int, dict[str, Any]]:
        v = PRICING["vercel_fluid_iad1"]
        cpu_h, mem_gbh, inv_m = (
            v["active_cpu_usd_per_h"],
            v["memory_usd_per_gb_h"],
            v["invocations_usd_per_m"],
        )
        g_cpu_s = guard_model_mean_s * factor
        g_wall_s = (guard_model_mean_s + GUARD_OVERHEAD_S) * factor
        per_level: dict[int, dict[str, Any]] = {}
        for lv in range(1, 6):
            share, gc = vault_share(lv), guard_calls(lv, shadow)
            tok = {m: groq_tokens(m) for m in models}
            vault_usd: dict[str, float] = {}
            for m in models:
                t = tok[m].get(lv)
                if t is None:
                    continue
                p = PRICING["groq_usd_per_1m_tokens"][m]
                vault_usd[m] = share * (t["prompt"] * p["in"] + t["completion"] * p["out"]) / 1e6
            ref = tok[models[-1]][lv]
            gw_wall = DB_WALL_S + gc * (g_wall_s + NET_S) + share * ref["wall_s"]
            gw_cpu_s = gw_cpu["post_message_cpu_ms"]["mean"] / 1000 * CPU_FACTOR
            extra_cpu = gw_cpu["get_state_cpu_ms"]["mean"] / 1000 * CPU_FACTOR * EXTRA_GATEWAY_CALLS
            gw_gbs = GATEWAY_MEM_GB * (gw_wall + EXTRA_GATEWAY_CALLS * EXTRA_CALL_WALL_S)
            gd_gbs = GUARD_MEM_GB * g_wall_s * gc
            gd_cpu = g_cpu_s * gc
            inv_gw, inv_gd = 1 + EXTRA_GATEWAY_CALLS, gc
            msg_chars = 76.0
            row_b = ATTEMPT_FIXED_B + msg_chars + share * ref["reply_chars"] + INDEX_B
            per_level[lv] = {
                "vault_call_share": round(share, 3),
                "guard_calls": round(gc, 3),
                "vault_usd": {m: u for m, u in vault_usd.items()},
                "gateway": {
                    "cpu_s": gw_cpu_s + extra_cpu,
                    "gb_s": gw_gbs,
                    "invocations": inv_gw,
                    "usd": (gw_cpu_s + extra_cpu) / 3600 * cpu_h
                    + gw_gbs / 3600 * mem_gbh
                    + inv_gw * inv_m / 1e6,
                },
                "guard": {
                    "cpu_s": gd_cpu,
                    "gb_s": gd_gbs,
                    "invocations": inv_gd,
                    "usd": gd_cpu / 3600 * cpu_h + gd_gbs / 3600 * mem_gbh + inv_gd * inv_m / 1e6,
                },
                "neon": {
                    "row_bytes": round(row_b),
                    "storage_usd_per_month_retained": row_b
                    / 1e9
                    * PRICING["neon"]["storage_usd_per_gb_month"],
                    "compute_cu_s_marginal": 0.25
                    * (DB_WALL_S + EXTRA_GATEWAY_CALLS * EXTRA_CALL_WALL_S),
                },
            }
        return per_level

    def totals(
        per_level: dict[int, dict[str, Any]],
        msgs: list[int],
        model: str,
        models_used: tuple[str, ...],
    ) -> dict[str, Any]:
        n = sum(msgs)
        vault = sum(msgs[i] * per_level[i + 1]["vault_usd"][model] for i in range(5))
        gw = sum(msgs[i] * per_level[i + 1]["gateway"]["usd"] for i in range(5))
        gd = sum(msgs[i] * per_level[i + 1]["guard"]["usd"] for i in range(5))
        cu_s = sum(msgs[i] * per_level[i + 1]["neon"]["compute_cu_s_marginal"] for i in range(5))
        neon_compute = cu_s / 3600 * PRICING["neon"]["launch_usd_per_cu_h"]
        game_bytes = GAME_FIXED_B + sum(
            msgs[i] * per_level[i + 1]["neon"]["row_bytes"] for i in range(5)
        )
        neon_storage = game_bytes / 1e9 * PRICING["neon"]["storage_usd_per_gb_month"]
        neon = neon_compute + neon_storage
        game_usd = vault + gw + gd + neon
        res = {
            "messages": n,
            "usd_per_game": game_usd,
            "usd_per_1k_games": game_usd * 1000,
            "usd_per_1k_requests": game_usd / n * 1000,
            "breakdown_usd_per_1k_requests": {
                "vault_llm": vault / n * 1000,
                "vercel_gateway": gw / n * 1000,
                "vercel_guard": gd / n * 1000,
                "neon": neon / n * 1000,
            },
            "breakdown_usd_per_1k_games": {
                "vault_llm": vault * 1000,
                "vercel_gateway": gw * 1000,
                "vercel_guard": gd * 1000,
                "neon": neon * 1000,
            },
            "neon_bytes_per_game": round(game_bytes),
            "vault_tokens_per_game": sum(
                msgs[i]
                * per_level[i + 1]["vault_call_share"]
                * (groq_tokens(model)[i + 1]["prompt"] + groq_tokens(model)[i + 1]["completion"])
                for i in range(5)
            ),
            "vault_calls_per_game": sum(
                msgs[i] * per_level[i + 1]["vault_call_share"] for i in range(5)
            ),
            "guard_calls_per_game": sum(
                msgs[i] * per_level[i + 1]["guard_calls"] for i in range(5)
            ),
            "guard_cpu_s_per_game": sum(
                msgs[i] * per_level[i + 1]["guard"]["cpu_s"] for i in range(5)
            ),
            "guard_gb_s_per_game": sum(
                msgs[i] * per_level[i + 1]["guard"]["gb_s"] for i in range(5)
            ),
            "gateway_cpu_s_per_game": sum(
                msgs[i] * per_level[i + 1]["gateway"]["cpu_s"] for i in range(5)
            ),
            "gateway_gb_s_per_game": sum(
                msgs[i] * per_level[i + 1]["gateway"]["gb_s"] for i in range(5)
            ),
            "gateway_invocations_per_game": sum(
                msgs[i] * per_level[i + 1]["gateway"]["invocations"] for i in range(5)
            ),
            "guard_invocations_per_game": sum(
                msgs[i] * per_level[i + 1]["guard"]["invocations"] for i in range(5)
            ),
        }
        return res

    def ceilings(t: dict[str, Any], model: str) -> dict[str, Any]:
        hob = PRICING["vercel_hobby_included"]
        lim = PRICING["groq_limits"][model]
        cpu_s = t["guard_cpu_s_per_game"] + t["gateway_cpu_s_per_game"]
        gbs = t["guard_gb_s_per_game"] + t["gateway_gb_s_per_game"]
        inv = t["guard_invocations_per_game"] + t["gateway_invocations_per_game"]
        out = {
            "vercel_active_cpu_4h": hob["active_cpu_h"] * 3600 / cpu_s,
            "vercel_memory_360gbh": hob["memory_gb_h"] * 3600 / gbs,
            "vercel_invocations_1m": hob["invocations"] / inv,
            "groq_tpd_200k_per_month_one_model": 30 * lim["tpd"] / t["vault_tokens_per_game"],
            "groq_rpd_1000_per_month_one_model": 30
            * lim["rpd"]
            / max(t["vault_calls_per_game"], 1e-9),
            "neon_storage_1gb": 1e9 / t["neon_bytes_per_game"],
        }
        return {k: round(v) for k, v in out.items()}

    all_models = ("openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b")
    result: dict[str, Any] = {
        "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "supersedes": "docs/lanes/P-cost.json (chars/4 lower bound)",
        "actual_bill_usd": 0.0,
        "actual_bill_note": "Groq free tier, Vercel Hobby, Neon free, Hugging Face free; $0 budget (ADR-0037). Figures below are LIST prices for usage above the free tiers.",
        "pricing": PRICING,
        "assumptions": {
            "scenarios_messages_per_boss": SCENARIOS,
            "headline_scenario": HEADLINE,
            "attack_share": ATTACK_SHARE,
            "attack_block_rate_by_level": {k: round(v, 3) for k, v in ATTACK_BLOCKED.items()},
            "benign_block_rate_by_level": BENIGN_BLOCKED,
            "guard_model_time_mean_s": guard_model_mean_s,
            "guard_model_time_p50_s_v2": guard_model_p50_s,
            "guard_memory_gb": GUARD_MEM_GB,
            "gateway_memory_gb": GATEWAY_MEM_GB,
            "gateway_cpu_ms_in_process_mean": gw_cpu["post_message_cpu_ms"]["mean"],
            "gateway_cpu_factor_for_network_io": CPU_FACTOR,
            "db_wall_s_per_message": DB_WALL_S,
            "gateway_guard_net_s": NET_S,
            "extra_gateway_calls_per_message": EXTRA_GATEWAY_CALLS,
            "guard_factor": a.guard_factor,
            "shadow_guard_call_at_l4_l5": not a.no_shadow,
            "vault_model_for_gpt_oss_20b": "tokens proxied from the gpt-oss-120b measurement (20b daily cap exhausted), priced at the 20b rate",
            "memory_billing": "no sharing across concurrent requests (upper bound for the gateway; the guard is CPU-bound on 1 vCPU)",
            "gemini": "not in the serving path (offline judge only): $0 per request",
            "hf_hub": "model/dataset hosting and the guard download at cold start: free",
        },
        "measured": {
            "groq_tokens_per_vault_call": {
                m: {
                    str(k): {kk: round(vv, 1) for kk, vv in v.items()}
                    for k, v in groq_tokens(m).items()
                }
                for m in all_models
            },
            "gateway_cpu": gw_cpu,
            "guard_cold_start": COLD,
        },
        "scenarios": {},
    }
    base = run(all_models, a.guard_factor, not a.no_shadow)
    result["per_level_usd_per_request"] = {
        str(lv): {
            "vault_call_share": d["vault_call_share"],
            "guard_calls_per_request": d["guard_calls"],
            "vault_llm": d["vault_usd"],
            "vercel_gateway": d["gateway"]["usd"],
            "vercel_guard": d["guard"]["usd"],
            "neon_storage_per_month_retained": d["neon"]["storage_usd_per_month_retained"],
        }
        for lv, d in base.items()
    }
    for sc, msgs in SCENARIOS.items():
        result["scenarios"][sc] = {"messages_per_boss": msgs}
        for m in all_models:
            t = totals(base, msgs, m, all_models)
            t["free_tier_ceiling_games_per_month"] = ceilings(t, m)
            t["binding_limit"] = min(
                t["free_tier_ceiling_games_per_month"],
                key=lambda k: t["free_tier_ceiling_games_per_month"][k],
            )
            result["scenarios"][sc][m] = t
    # sensitivity on the headline scenario and the configured-primary vault model
    sens: list[dict[str, Any]] = []
    head_msgs = SCENARIOS[HEADLINE]
    for factor, shadow in [
        (1.0, True),
        (1.0, False),
        (0.2, True),
        (0.2, False),
        (0.1, True),
        (0.1, False),
    ]:
        per = run(all_models, factor, shadow)
        t = totals(per, head_msgs, "openai/gpt-oss-20b", all_models)
        t["free_tier_ceiling_games_per_month"] = ceilings(t, "openai/gpt-oss-20b")
        sens.append(
            {
                "guard_factor": factor,
                "shadow_call": shadow,
                "usd_per_1k_requests": t["usd_per_1k_requests"],
                "usd_per_1k_games": t["usd_per_1k_games"],
                "vercel_guard_usd_per_1k_games": t["breakdown_usd_per_1k_games"]["vercel_guard"],
                "vercel_active_cpu_ceiling_games_per_month": t["free_tier_ceiling_games_per_month"][
                    "vercel_active_cpu_4h"
                ],
                "binding_limit": min(
                    t["free_tier_ceiling_games_per_month"],
                    key=lambda k: t["free_tier_ceiling_games_per_month"][k],
                ),
            }
        )
    result["sensitivity_guard_compute"] = {
        "scenario": HEADLINE,
        "vault_model": "openai/gpt-oss-20b",
        "rows": sens,
    }
    cold_usd = (
        COLD["cpu_s"] / 3600 * PRICING["vercel_fluid_iad1"]["active_cpu_usd_per_h"]
        + GUARD_MEM_GB * COLD["wall_s"] / 3600 * PRICING["vercel_fluid_iad1"]["memory_usd_per_gb_h"]
        + PRICING["vercel_fluid_iad1"]["invocations_usd_per_m"] / 1e6
    )
    ht = result["scenarios"][HEADLINE]["openai/gpt-oss-20b"]
    ratio = guard_model_p50_s / guard_model_mean_s
    result["active_cpu_ceiling_if_guard_time_is_p50"] = {
        "note": "guard CPU per call = v2 median model time (0.72 s) instead of the 1.12 s warm mean (which includes the slow tail on a shared 1 vCPU)",
        "games_per_month": round(
            PRICING["vercel_hobby_included"]["active_cpu_h"]
            * 3600
            / (ht["guard_cpu_s_per_game"] * ratio + ht["gateway_cpu_s_per_game"])
        ),
    }
    result["cold_start"] = {
        "usd_per_cold_start": cold_usd,
        "usd_per_1k_games_if_one_cold_start_per_game": cold_usd * 1000,
        "note": "sparse traffic: the guard instance is idle between players; B3-vercel-bench-cold.json measured 12.8 s and 8.6-9.7 s boot",
    }
    Path(a.out).write_text(json.dumps(result, indent=1))
    Path(a.md).write_text(render_md(result))
    h = result["scenarios"][HEADLINE]["openai/gpt-oss-20b"]
    print(
        json.dumps(
            {
                "usd_per_1k_requests": round(h["usd_per_1k_requests"], 4),
                "usd_per_1k_games": round(h["usd_per_1k_games"], 2),
                "breakdown_per_1k_games": {
                    k: round(v, 3) for k, v in h["breakdown_usd_per_1k_games"].items()
                },
                "ceilings": h["free_tier_ceiling_games_per_month"],
            }
        )
    )


def usd(x: float) -> str:
    return f"${x:,.4f}" if x < 1 else f"${x:,.2f}"


def render_md(r: dict[str, Any]) -> str:
    h = r["scenarios"][HEADLINE]
    pm = "openai/gpt-oss-20b"
    L: list[str] = []
    L.append("# Lane C: cost per 1,000 requests and per 1,000 games\n")
    L.append(
        f"Generated by `scripts/p_cost.py` on {r['measured_at']}; supersedes `docs/lanes/P-cost.json` (chars/4 lower bound). "
        "List prices fetched 2026-11-13 from the providers' official pages (URLs in `C-cost.json`).\n"
    )
    L.append("## Answer\n")
    L.append(
        "**Actual bill today: $0** (Groq free tier, Vercel Hobby, Neon free, Hugging Face free, Gemini not in the serving path).\n"
    )
    t = h[pm]
    L.append(
        f"If usage exceeded the free tiers (list prices, typical game of {sum(h['messages_per_boss'])} messages, "
        f"vault = gpt-oss-20b, the configured first model): **{usd(t['usd_per_1k_requests'])} per 1,000 requests** and "
        f"**{usd(t['usd_per_1k_games'])} per 1,000 games**.\n"
    )
    L.append("## Breakdown (typical game, USD per 1,000 games / per 1,000 requests)\n")
    L.append(
        "| Vault model | vault LLM | Vercel gateway | Vercel guard | Neon | total / 1k games | total / 1k requests |"
    )
    L.append("|---|---|---|---|---|---|---|")
    for m in (pm, "openai/gpt-oss-120b", "qwen/qwen3.8-27b"):
        x = h[m]
        b = x["breakdown_usd_per_1k_games"]
        L.append(
            f"| {m} | {usd(b['vault_llm'])} | {usd(b['vercel_gateway'])} | {usd(b['vercel_guard'])} | {usd(b['neon'])} | "
            f"{usd(x['usd_per_1k_games'])} | {usd(x['usd_per_1k_requests'])} |"
        )
    L.append("")
    L.append(
        "Same breakdown per 1,000 requests (one request = one `POST /api/message`, typical-game mix):\n"
    )
    L.append("| Vault model | vault LLM | Vercel gateway | Vercel guard | Neon | total |")
    L.append("|---|---|---|---|---|---|")
    for m in (pm, "openai/gpt-oss-120b", "qwen/qwen3.8-27b"):
        x = h[m]
        q = x["breakdown_usd_per_1k_requests"]
        L.append(
            f"| {m} | {usd(q['vault_llm'])} | {usd(q['vercel_gateway'])} | {usd(q['vercel_guard'])} | {usd(q['neon'])} | {usd(x['usd_per_1k_requests'])} |"
        )
    L.append("")
    L.append(
        "The production chain is gpt-oss-20b, then qwen3.8-27b, then gpt-oss-120b (fallback only on a rate limit); when the 20b daily cap is spent, qwen3.8-27b serves, which is the most expensive row.\n"
    )
    L.append("## Scenarios (messages per boss L1-L5)\n")
    L.append(
        "| Scenario | messages per boss | messages/game | $ per 1k games (20b / 120b / qwen) | $ per 1k requests (20b) |"
    )
    L.append("|---|---|---|---|---|")
    for sc, d in r["scenarios"].items():
        L.append(
            f"| {sc} | {d['messages_per_boss']} | {d[pm]['messages']} | "
            f"{usd(d[pm]['usd_per_1k_games'])} / {usd(d['openai/gpt-oss-120b']['usd_per_1k_games'])} / {usd(d['qwen/qwen3.8-27b']['usd_per_1k_games'])} | {usd(d[pm]['usd_per_1k_requests'])} |"
        )
    L.append("")
    L.append("## Per request by level (gpt-oss-20b, USD)\n")
    L.append(
        "| Level | vault call share | guard calls | vault LLM | Vercel gateway | Vercel guard |"
    )
    L.append("|---|---|---|---|---|---|")
    for lv, d in r["per_level_usd_per_request"].items():
        L.append(
            f"| L{lv} | {d['vault_call_share']} | {d['guard_calls_per_request']} | {d['vault_llm'][pm]:.6f} | {d['vercel_gateway']:.6f} | {d['vercel_guard']:.6f} |"
        )
    L.append("")
    L.append("## Which free-tier limit binds first (typical game, per month)\n")
    L.append("| Limit | games / month before it is hit |")
    L.append("|---|---|")
    ce = h[pm]["free_tier_ceiling_games_per_month"]
    names = {
        "vercel_active_cpu_4h": "Vercel Hobby Active CPU, 4 h (team-wide; guard inference is almost all of it)",
        "vercel_memory_360gbh": "Vercel Hobby provisioned memory, 360 GB-h",
        "vercel_invocations_1m": "Vercel Hobby invocations, 1,000,000",
        "groq_tpd_200k_per_month_one_model": "Groq 200k tokens/day on gpt-oss-20b alone (x30 days)",
        "groq_rpd_1000_per_month_one_model": "Groq 1,000 requests/day on gpt-oss-20b alone (x30 days)",
        "neon_storage_1gb": "Neon free 1 GB storage per project (cumulative, not monthly)",
    }
    for k, v in sorted(ce.items(), key=lambda kv: kv[1]):
        L.append(f"| {names[k]} | {v:,} |")
    L.append("")
    L.append(
        f"The first to bind is **{names[h[pm]['binding_limit']]}**. Groq is a per-day cap (about 200k tokens per model per day, three models in the chain), "
        "so it throttles bursts; the Vercel Active CPU allowance is a per-month cap and exceeding it pauses the Hobby team's functions for 30 days (Hobby has no overage billing).\n"
    )
    L.append("## Sensitivity: guard compute (cascade replacing the guard, shadow call removed)\n")
    L.append(
        "`guard_factor` multiplies guard CPU and memory time (1.0 = measured champion, 0.2 to 0.1 = a cascade with 5-10x lower guard compute). `shadow` is the second `/score` call the gateway makes at L4-L5 for the challenger (same host, same graph).\n"
    )
    L.append(
        "| guard_factor | shadow call | $ per 1k requests | $ per 1k games | guard $ per 1k games | Active-CPU ceiling (games/month) | binds first |"
    )
    L.append("|---|---|---|---|---|---|---|")
    for s in r["sensitivity_guard_compute"]["rows"]:
        L.append(
            f"| {s['guard_factor']} | {'yes' if s['shadow_call'] else 'no'} | {usd(s['usd_per_1k_requests'])} | {usd(s['usd_per_1k_games'])} | "
            f"{usd(s['vercel_guard_usd_per_1k_games'])} | {s['vercel_active_cpu_ceiling_games_per_month']:,} | {s['binding_limit']} |"
        )
    L.append("")
    L.append(
        f"If guard CPU per call were the v2 median (0.72 s) instead of the 1.12 s warm mean, the Active CPU ceiling would be {r['active_cpu_ceiling_if_guard_time_is_p50']['games_per_month']:,} games/month.\n"
    )
    L.append(
        f"Cold starts (sparse traffic): one guard cold start costs {usd(r['cold_start']['usd_per_cold_start'])} at list price "
        f"({usd(r['cold_start']['usd_per_1k_games_if_one_cold_start_per_game'])} per 1,000 games if every game starts cold).\n"
    )
    L.append("## Measured inputs\n")
    L.append(
        "Groq `usage` from the raw API response through the production `GroqAgent` (max_tokens 400, reasoning_effort low, level prompts from `configs/levels.yaml`), 40 messages for levels 1-3 and 12 for levels 4-5, per model. Mean tokens per vault call (prompt / completion, completion includes reasoning):\n"
    )
    L.append("| Model | L1 | L2 | L3 | L4-L5 |")
    L.append("|---|---|---|---|---|")
    for m in ("openai/gpt-oss-120b", "qwen/qwen3.8-27b"):
        g = r["measured"]["groq_tokens_per_vault_call"][m]
        L.append(
            f"| {m} | "
            + " | ".join(
                f"{g[k]['prompt']:.0f} / {g[k]['completion']:.0f}" for k in ("1", "2", "3", "4")
            )
            + " |"
        )
    L.append("")
    L.append("## Prices used (fetched 2026-11-13)\n")
    gp = r["pricing"]["groq_usd_per_1m_tokens"]
    L.append(
        f"- Groq per 1M tokens in/out: gpt-oss-20b ${gp['openai/gpt-oss-20b']['in']}/${gp['openai/gpt-oss-20b']['out']}, "
        f"gpt-oss-120b ${gp['openai/gpt-oss-120b']['in']}/${gp['openai/gpt-oss-120b']['out']}, "
        f"qwen3.8-27b ${gp['qwen/qwen3.8-27b']['in']}/${gp['qwen/qwen3.8-27b']['out']} ({gp['source']}). No cached-input price is listed."
    )
    v = r["pricing"]["vercel_fluid_iad1"]
    L.append(
        f"- Vercel Fluid, Washington D.C. (iad1, both projects' region): Active CPU ${v['active_cpu_usd_per_h']}/h, provisioned memory ${v['memory_usd_per_gb_h']}/GB-h, "
        f"invocations ${v['invocations_usd_per_m']}/M ({v['source']}). Hobby includes 4 h Active CPU, 360 GB-h, 1M invocations per month; past that Hobby is paused 30 days (https://vercel.com/docs/plans/hobby). "
        "Active CPU is billed only while code runs (I/O waits are free); provisioned memory is billed for the whole time a request is in flight."
    )
    n = r["pricing"]["neon"]
    L.append(
        f"- Neon Launch: ${n['launch_usd_per_cu_h']}/CU-h, ${n['storage_usd_per_gb_month']}/GB-month; Free: {n['free_cu_h_per_project_month']} CU-h per project per month, "
        f"{n['free_storage_gb_per_project']} GB per project, 5 GB egress ({n['source']})."
    )
    L.append(
        "- Hugging Face Hub: public model and dataset hosting is free (best-effort; https://huggingface.co/docs/hub/storage-limits), so the guard weights, dataset and model card cost nothing; the Hub download at guard cold start is not billed."
    )
    L.append(
        "- Groq limits: https://console.groq.com/docs/rate-limits renders the Developer-plan base limits (30 RPM, 1K RPD, 8K TPM, 200K TPD for the three models); its Free-plan tab did not render in a plain fetch. The repo's own observation (CLAUDE.md, G-difficulty-prod.json) is the same 200k tokens/day cap, with gpt-oss-20b already exhausted. The Vercel usage API (`/v1/usage`) needs a Pro team (HTTP plan_upgrade_required), so current Hobby consumption could not be read; billing charges for 2026-11 are empty ($0).\n"
    )
    L.append("## Assumptions and limits of this estimate\n")
    for k, v in r["assumptions"].items():
        L.append(f"- `{k}`: {v}")
    L.append(
        "- Nothing here queries production data; messages per boss are assumptions, not observed medians. Neon DB wall time (0.15 s per message) and the 3x CPU allowance for Postgres/TLS were not measured on Neon."
    )
    L.append(
        "- Vercel Pro would add a $20/month seat and a monthly usage credit; Neon Launch has no base fee in the pricing table. Both are excluded: the figures are marginal unit prices."
    )
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
