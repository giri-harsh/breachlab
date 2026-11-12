"""Measure the guard function on Vercel the way the gateway sees it (ADR-0080, ADR-0064 style, nothing rounded).

    GUARD_URL=https://breachlab-guard.vercel.app GUARD_TOKEN=... python scripts/bench_vercel_guard.py \
        --gold gold_test.jsonl --out bench.json [--requests 500] [--cold-gaps 120 600 1800]

* ``warm``: 500 requests at concurrency 1 and at concurrency 4, texts drawn from the gold_test length distribution
  and clipped to 500 characters (the gateway's message limit). Per run: client round trip and the service's own
  ``latency_ms`` (p50 / p95 / mean), the share of requests that hit a cold instance (``x-guard-cold``), and errors by
  HTTP status (failures are counted, never dropped).
* ``cold``: after each idle gap the first ``POST /score`` is timed (round trip, header boot seconds, status), then
  ``/vercel-info`` says whether the instance was a fresh one. Shows how long an instance really stays warm.
Secrets come from the environment and are never printed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from breachlab.eval.latency import percentile, sample_texts

CLIP = 500


def headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def one(client: httpx.Client, url: str, text: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        r = client.post(f"{url}/score", json={"text": text, "kind": "input", "alias": "champion"})
    except httpx.HTTPError as exc:
        return {"status": type(exc).__name__, "rtt_ms": (time.perf_counter() - t0) * 1000.0}
    rec: dict[str, Any] = {
        "status": r.status_code,
        "rtt_ms": (time.perf_counter() - t0) * 1000.0,
        "cold": r.headers.get("x-guard-cold") == "1",
        "boot_s": float(r.headers.get("x-guard-boot-s", "nan")),
        "server_wall_ms": float(r.headers.get("x-guard-wall-ms", "nan")),
    }
    if r.status_code == 200:
        body = r.json()
        rec["model_ms"] = float(body["latency_ms"])
        rec["label"] = body["label"]
    return rec


def summarize(recs: list[dict[str, Any]], wall_s: float, concurrency: int) -> dict[str, Any]:
    ok = [r for r in recs if r["status"] == 200]
    rtt = [r["rtt_ms"] for r in ok]
    model = [r["model_ms"] for r in ok]
    return {
        "n": len(recs),
        "concurrency": concurrency,
        "errors_by_status": dict(Counter(str(r["status"]) for r in recs if r["status"] != 200)),
        "cold_responses": sum(1 for r in recs if r.get("cold")),
        "rtt_p50_ms": percentile(rtt, 50) if rtt else None,
        "rtt_p95_ms": percentile(rtt, 95) if rtt else None,
        "rtt_mean_ms": sum(rtt) / len(rtt) if rtt else None,
        "rtt_max_ms": max(rtt) if rtt else None,
        "model_p50_ms": percentile(model, 50) if model else None,
        "model_p95_ms": percentile(model, 95) if model else None,
        "model_mean_ms": sum(model) / len(model) if model else None,
        "wall_s": wall_s,
        "requests_per_s": len(recs) / wall_s,
    }


def warm(
    client: httpx.Client, url: str, texts: list[str], n: int, concurrency: int
) -> dict[str, Any]:
    batch = sample_texts(texts, n, 42)
    for t in batch[:5]:  # warm-up: the first request of a fresh instance pays the cold start
        one(client, url, t)
    t0 = time.perf_counter()
    if concurrency == 1:
        recs = [one(client, url, t) for t in batch]
    else:
        with ThreadPoolExecutor(concurrency) as ex:
            recs = list(ex.map(lambda t: one(client, url, t), batch))
    out = summarize(recs, time.perf_counter() - t0, concurrency)
    out["raw_rtt_ms"] = [r["rtt_ms"] for r in recs]
    return out


def cold(client: httpx.Client, url: str, text: str, gaps: list[int]) -> list[dict[str, Any]]:
    out = []
    for gap in gaps:
        print(f"idle {gap}s ...", flush=True)
        time.sleep(gap)
        rec = one(client, url, text)
        try:
            info = client.get(f"{url}/vercel-info", timeout=30.0).json()
        except (httpx.HTTPError, ValueError):
            info = {}
        rec.update(
            {
                "idle_s": gap,
                "instance_age_s": info.get("instance_age_s"),
                "served": info.get("served"),
            }
        )
        rec["fetch"] = info.get("fetch")
        out.append(rec)
        print(json.dumps(rec), flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default=os.environ.get("GUARD_URL", ""))
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--requests", type=int, default=500)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--cold-gaps", type=int, nargs="*", default=[])
    ap.add_argument("--skip-warm", action="store_true")
    a = ap.parse_args(argv)
    url = a.url.rstrip("/")
    texts = [
        json.loads(line)["text"][:CLIP]
        for line in a.gold.read_text("utf-8").splitlines()
        if line.strip()
    ]
    token = os.environ.get("GUARD_TOKEN", "")
    client = httpx.Client(
        headers=headers(token),
        timeout=httpx.Timeout(120.0),
        limits=httpx.Limits(max_connections=16),
    )
    res: dict[str, Any] = {
        "url": url,
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if not a.skip_warm:
        res["warm"] = {f"c{c}": warm(client, url, texts, a.requests, c) for c in a.concurrency}
        a.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    if a.cold_gaps:
        res["cold"] = cold(client, url, texts[0], a.cold_gaps)
    a.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    for k, v in res.get("warm", {}).items():
        print(k, {x: v[x] for x in v if x != "raw_rtt_ms"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
