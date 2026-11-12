"""Pure parts of the full-request latency / cost script (PRD 1.2, 9.1). No network."""

from __future__ import annotations

import hashlib
import hmac
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "p_full_request_latency.py"
spec = importlib.util.spec_from_file_location("p_full_request_latency", SCRIPT)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules["p_full_request_latency"] = mod
spec.loader.exec_module(mod)


def test_canary_header_and_key() -> None:
    h = mod.canary_header("k", now=1000.0)
    ts, sig = h.split(".")
    assert ts == "1000"
    assert sig == hmac.new(b"k", b"1000", hashlib.sha256).hexdigest()
    assert mod.gateway_key({"GATEWAY_SECRET_KEY": "g", "ADMIN_PASSWORD": "p"}) == "g"
    want = hashlib.sha256(b"breachlab-gateway-v1|p").hexdigest()
    assert mod.gateway_key({"ADMIN_PASSWORD": "p"}) == want


def test_plan_is_fixed_and_interleaved() -> None:
    plan = mod.build_plan(100, ["extra-a", "extra-b"])
    assert len(plan) == 100
    assert sum(a for _, a in plan) == 50
    assert [a for _, a in plan[:4]] == [False, True, False, True]
    assert plan[0][0] == mod.BENIGN[0]
    assert plan[1][0] == mod.ATTACKS[0]
    assert plan[21][0] == "extra-a"
    assert mod.build_plan(100, ["extra-a", "extra-b"]) == plan


def test_percentile() -> None:
    assert mod.percentile([], 50) is None
    assert mod.percentile([5.0], 95) == 5.0
    xs = [float(i) for i in range(1, 101)]
    assert mod.percentile(xs, 50) == pytest.approx(50.5)
    assert mod.percentile(xs, 95) == pytest.approx(95.05)


RECS = [
    {"i": 0, "is_attack": False, "status": 200, "rtt_ms": 1000.0, "server_ms": 900.0,
     "blocked": False, "prompt_chars": 40, "reply_chars": 400, "guard_version": "v1"},
    {"i": 1, "is_attack": True, "status": 200, "rtt_ms": 600.0, "server_ms": 500.0,
     "blocked": True, "prompt_chars": 60, "reply_chars": 0, "guard_version": "v1"},
    {"i": 2, "is_attack": False, "status": 202, "rtt_ms": 300.0, "prompt_chars": 10},
    {"i": 3, "is_attack": True, "status": "ReadError", "rtt_ms": 60000.0, "prompt_chars": 10},
]  # fmt: skip


def test_summarize_warm_only_and_errors() -> None:
    s = mod.summarize(RECS)
    assert s["n_requests"] == 4 and s["n_ok"] == 2
    assert s["errors_by_status"] == {"202": 1, "ReadError": 1}
    assert s["client_rtt"]["max_ms"] == 1000.0
    assert s["by_kind"]["attack"]["blocked"] == 1
    assert s["by_path"]["guard_only (blocked)"]["n"] == 1
    assert s["meets_target"] is True
    assert mod.summarize(RECS, target_p95_ms=500.0)["meets_target"] is False
    assert mod.summarize([])["meets_target"] is False


def test_cost_summary_estimates_and_unpriced() -> None:
    c = mod.cost_summary(RECS, system_prompt_chars=360, measured_at="t")
    assert c["requests"] == 2 and c["vault_calls"] == 1
    assert c["vault_tokens"]["input_per_vault_call_mean"] == 100.0
    assert c["vault_tokens"]["output_per_vault_call_mean"] == 100.0
    assert c["vault_tokens"]["per_1k_requests"] == {"input": 50000, "output": 50000}
    assert c["guard_cpu_seconds"]["per_request_mean"] == 0.5
    assert c["guard_cpu_seconds"]["per_1k_requests"] == 500.0
    assert c["actual_bill_usd"] == 0.0
    assert c["usd_list_price"] == "not priced"
    assert mod.est_tokens(0) == 0 and mod.est_tokens(5) == 2
