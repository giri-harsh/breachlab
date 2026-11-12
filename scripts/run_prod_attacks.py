"""Run an attack file against the PRODUCTION gateway with canary sessions (D5 prod re-run).

    python scripts/run_prod_attacks.py --attacks docs/lanes/fresh_attacks_v2.jsonl --levels 1 2 3 4 5 --out FILE.json

Canary sessions (FR-10) skip level gating, caps and rate limits and are excluded from the board and training data. One
session per level; each attack is one message. Leak = the response's ``leaked`` flag. The vault is whatever the
production chain answers with. Resumable: results are written per level to ``--out`` after every attack.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

BASE = "https://breachlab-api.vercel.app"


def gateway_key() -> str:
    key = os.environ.get("GATEWAY_SECRET_KEY")
    if key:
        return key
    return hashlib.sha256(
        f"breachlab-gateway-v1|{os.environ['ADMIN_PASSWORD']}".encode()
    ).hexdigest()


def header(key: str) -> dict[str, str]:
    ts = str(int(time.time()))
    return {"X-Canary": f"{ts}.{hmac.new(key.encode(), ts.encode(), hashlib.sha256).hexdigest()}"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacks", type=Path, required=True)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default=BASE)
    a = ap.parse_args(argv)
    texts = [
        json.loads(x)["text"] for x in a.attacks.read_text("utf-8").split("\n") if x.strip()
    ]
    key = gateway_key()
    res: dict[str, Any] = json.loads(a.out.read_text("utf-8")) if a.out.exists() else {}
    with httpx.Client(timeout=90.0) as c:
        for level in a.levels:
            done: dict[str, Any] = res.setdefault(str(level), {})
            sid = None
            for i, text in enumerate(texts):
                if str(i) in done and done[str(i)].get("status") == 200:
                    continue
                for attempt in range(6):
                    if sid is None:
                        r = c.post(
                            f"{a.base}/api/session",
                            json={"nickname": f"Prod Fresh L{level}"},
                            headers=header(key),
                        )
                        r.raise_for_status()
                        sid = r.json()["session_id"]
                    r = c.post(
                        f"{a.base}/api/message",
                        json={"session_id": sid, "level": level, "message": text},
                        headers=header(key),
                    )
                    if r.status_code in (202, 503, 429):
                        time.sleep(min(5 * (attempt + 1), 30))
                        continue
                    break
                body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
                done[str(i)] = {
                    "status": r.status_code,
                    "blocked": bool(body.get("blocked")),
                    "leaked": bool(body.get("leaked")),
                    "layer": (body.get("verdict") or {}).get("layer"),
                }
                a.out.write_text(json.dumps(res), "utf-8")
            ok = [v for v in done.values() if v["status"] == 200]
            print(
                f"L{level}: {len(ok)}/{len(texts)} answered, blocked {sum(v['blocked'] for v in ok)}, leaked {sum(v['leaked'] for v in ok)}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
