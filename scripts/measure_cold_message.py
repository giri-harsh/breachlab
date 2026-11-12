"""Time the first message after the guard went idle, as the player sees it (ADR-0085, S14c).

    ADMIN_PASSWORD=... python scripts/measure_cold_message.py [--base-url URL] [--level 4]
                       [--warm-first --typing-s 8] [--out result.json]

Run it only after the guard has been idle for 10+ minutes (an instance is gone by then). It opens a canary session,
optionally fires ``GET /api/warm`` without waiting (what the web app does on load) and "types" for ``--typing-s``
seconds, then sends one benign message and keeps retrying while the gateway answers 202 (warming) exactly like the UI
does. Prints the wall time from the moment the player presses Send, every status seen, and whether it ended in a reply.
The secret comes from the environment and is never printed.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import hmac
import json
import os
import threading
import time
from typing import Any

import httpx

MESSAGE = "What is the capital of France?"


def canary_header(key: str, now: float | None = None) -> str:
    ts = str(int(now if now is not None else time.time()))
    return f"{ts}.{hmac.new(key.encode(), ts.encode(), hashlib.sha256).hexdigest()}"


def gateway_key(env: dict[str, str]) -> str:
    if env.get("GATEWAY_SECRET_KEY"):
        return env["GATEWAY_SECRET_KEY"]
    return hashlib.sha256(f"breachlab-gateway-v1|{env['ADMIN_PASSWORD']}".encode()).hexdigest()


def run(
    base: str, level: int, warm_first: bool, typing_s: float, key: str, max_wait_s: float = 90.0
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "base": base,
        "level": level,
        "warm_first": warm_first,
        "typing_s": typing_s,
    }
    with httpx.Client(timeout=60.0) as c:
        h = lambda: {"x-canary": canary_header(key)}  # noqa: E731
        t0 = time.perf_counter()
        r = c.post(
            f"{base}/api/session",
            json={"nickname": f"Cold {int(time.time()) % 100000}"},
            headers=h(),
        )
        r.raise_for_status()
        sid = r.json()["session_id"]
        out["session_s"] = round(time.perf_counter() - t0, 2)
        if warm_first:

            def fire_warm() -> (
                None
            ):  # like the page: a non-blocking fetch that keeps running while the player types
                with contextlib.suppress(httpx.HTTPError), httpx.Client(timeout=60.0) as wc:
                    r = wc.get(f"{base}/api/warm", headers=h())
                    out["warm_answer"] = {"status": r.status_code, "body": r.json()}

            threading.Thread(target=fire_warm, daemon=True).start()
            time.sleep(typing_s)
        statuses: list[int] = []
        press = time.perf_counter()
        while True:
            r = c.post(
                f"{base}/api/message",
                json={"session_id": sid, "level": level, "message": MESSAGE},
                headers=h(),
            )
            statuses.append(r.status_code)
            if r.status_code == 202:
                if time.perf_counter() - press > max_wait_s:
                    break
                time.sleep(float(r.json().get("retry_after_s", 5)))
                continue
            break
        out["send_to_answer_s"] = round(time.perf_counter() - press, 2)
        out["statuses"] = statuses
        out["final_status"] = statuses[-1]
        if r.status_code == 200:
            body = r.json()
            out["blocked"], out["guard_version"] = body.get("blocked"), body.get("guard_version")
        else:
            out["detail"] = (
                str(r.json().get("detail", ""))[:80]
                if r.headers.get("content-type", "").startswith("application/json")
                else ""
            )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="https://breachlab-api.vercel.app")
    ap.add_argument("--level", type=int, default=4)
    ap.add_argument("--warm-first", action="store_true")
    ap.add_argument("--typing-s", type=float, default=8.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    res = run(
        a.base_url.rstrip("/"), a.level, a.warm_first, a.typing_s, gateway_key(dict(os.environ))
    )
    res["measured_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
