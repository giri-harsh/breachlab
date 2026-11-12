"""Latency ladder harness for the deployed guard service (ADR-0039) and the torch-vs-ONNX quality parity check.

``bench`` drives ``POST /score`` with texts sampled from the gold_test length distribution (clipped to 500
characters like the gateway) and reports two views per run, unrounded: client round trip (what the gateway
sees: network + queue + model) and the service's own ``latency_ms`` (model time only). ``parity`` compares the
4-way logits of an ONNX variant with the torch logits on the same rows: AUROC and FPR at the tuned threshold.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from breachlab.data.schema import Row
from breachlab.eval.latency import percentile, sample_texts
from breachlab.eval.metrics import summarize
from breachlab.train.calibrate import to_preds
from breachlab.train.scoring import decide

CLIP_CHARS = 500

Post = Callable[[str], dict[str, Any]]


@dataclass(frozen=True)
class BenchResult:
    n: int
    concurrency: int
    errors: int
    rtt_p50_ms: float
    rtt_p95_ms: float
    rtt_mean_ms: float
    model_p50_ms: float
    model_p95_ms: float
    model_mean_ms: float
    wall_s: float


def clip(texts: Sequence[str], chars: int = CLIP_CHARS) -> list[str]:
    return [t[:chars] for t in texts]


def bench(
    post: Post,
    texts: Sequence[str],
    *,
    requests: int = 500,
    concurrency: int = 1,
    warmup: int = 10,
    seed: int = 42,
) -> BenchResult:
    """``post(text)`` must return the /score JSON; failures raise and are counted, not hidden."""
    batch = sample_texts(texts, requests, seed)
    for t in batch[:warmup]:
        post(t)

    def timed(t: str) -> tuple[float, float] | None:
        t0 = time.perf_counter()
        try:
            body = post(t)
        except Exception:  # noqa: BLE001 - counted in `errors`
            return None
        return (time.perf_counter() - t0) * 1000, float(body["latency_ms"])

    w0 = time.perf_counter()
    if concurrency == 1:
        out = [timed(t) for t in batch]
    else:
        with ThreadPoolExecutor(concurrency) as ex:
            out = list(ex.map(timed, batch))
    wall = time.perf_counter() - w0
    ok = [o for o in out if o is not None]
    if not ok:
        raise RuntimeError("every request failed")
    rtt, model = [o[0] for o in ok], [o[1] for o in ok]
    return BenchResult(
        len(batch),
        concurrency,
        len(out) - len(ok),
        percentile(rtt, 50),
        percentile(rtt, 95),
        float(np.mean(rtt)),
        percentile(model, 50),
        percentile(model, 95),
        float(np.mean(model)),
        wall,
    )


def http_post(
    url: str, alias: str = "champion", kind: str = "input", token: str | None = None
) -> Post:  # pragma: no cover
    """POST /score client; ``token`` (default env ``GUARD_TOKEN``) is sent as a bearer when set, never printed."""
    import os

    import httpx

    tok = token if token is not None else os.environ.get("GUARD_TOKEN", "")
    client = httpx.Client(
        timeout=180.0,
        limits=httpx.Limits(max_connections=16),
        headers={"Authorization": f"Bearer {tok}"} if tok else {},
    )

    def post(text: str) -> dict[str, Any]:
        r = client.post(
            f"{url.rstrip('/')}/score", json={"text": text, "kind": kind, "alias": alias}
        )
        r.raise_for_status()
        body: dict[str, Any] = r.json()
        return body

    return post


def parity(
    rows: list[Row],
    torch_logits: NDArray[np.float64],
    ort_logits: NDArray[np.float64],
    temperature: float,
    threshold: float,
) -> dict[str, float]:
    """Quality of an ONNX variant relative to torch on identical rows (gold_test) at the tuned threshold."""
    if not (len(rows) == len(torch_logits) == len(ort_logits)):
        raise ValueError("rows and logits must have the same length")
    a = summarize(to_preds(rows, torch_logits, temperature, threshold))
    b = summarize(to_preds(rows, ort_logits, temperature, threshold))
    agree = np.mean(
        [
            decide(x, temperature, threshold).label == decide(y, temperature, threshold).label
            for x, y in zip(torch_logits, ort_logits, strict=True)
        ]
    )
    return {
        "n": float(len(rows)),
        "auroc_torch": a["auroc"],
        "auroc_onnx": b["auroc"],
        "auroc_delta": b["auroc"] - a["auroc"],
        "fpr_torch": a["fpr"],
        "fpr_onnx": b["fpr"],
        "fpr_delta_pp": (b["fpr"] - a["fpr"]) * 100,
        "asr_torch": a["asr"],
        "asr_onnx": b["asr"],
        "label_agreement": float(agree),
        "max_abs_logit_diff": float(np.max(np.abs(torch_logits - ort_logits))),
    }


def read_gold(path: Path) -> list[Row]:
    return [
        Row.model_validate(json.loads(line))
        for line in path.read_text("utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:  # pragma: no cover - network
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--alias", default="champion")
    ap.add_argument("--gold", type=Path, default=Path("data/processed/v1/gold_test.jsonl"))
    ap.add_argument("--requests", type=int, default=500)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    texts = clip([r.text for r in read_gold(a.gold)])
    post = http_post(a.url, a.alias)
    res = {
        f"c{c}": asdict(bench(post, texts, requests=a.requests, concurrency=c))
        for c in a.concurrency
    }
    text = json.dumps(res, indent=2)
    print(text)
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
