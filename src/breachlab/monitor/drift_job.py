"""Weekly drift job (PRD 11, ADR-0041): pull the last 7 days of non-canary attempts from Neon, embed, report.

Neon is reached over its HTTPS SQL endpoint (the Actions runner and the dev sandbox are HTTPS-only friendly, see
``scripts/queue_gold_review.py``). Embeddings are a free-tier CPU substitute for a sentence-embedding model:
character n-gram TF-IDF reduced with truncated SVD, fitted on the baseline texts only. The report contains
counts and shares, never message text. Skips cleanly (exit 0, a short note instead of a report) when there is
no database secret or too little data; the clustering itself lives in ``breachlab.monitor.drift``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np
from numpy.typing import NDArray

from breachlab.monitor.canary_otlp import OtlpPusher, pusher_from_env
from breachlab.monitor.drift import DriftReport, build_report

MIN_KNOWN = 50
MIN_NEW = 20
KNOWN_LIMIT = 5000
NEW_LIMIT = 20000
SVD_DIMS = 64

DRIFT_METRICS = ("drift_share_outside_known", "drift_new_attempts", "drift_flagged")

Query = Callable[[str, list[Any]], list[dict[str, Any]]]

NEW_SQL = (
    "select a.message from attempts a join sessions s on s.id = a.session_id "
    "where s.is_canary = false and a.level > 0 and a.created_at >= now() - make_interval(days => $1::int) "
    "order by a.id desc limit " + str(NEW_LIMIT)
)
KNOWN_SQL = (
    "select a.message from attempts a join sessions s on s.id = a.session_id "
    "where s.is_canary = false and a.created_at < now() - make_interval(days => $1::int) "
    "order by a.id desc limit " + str(KNOWN_LIMIT)
)


def neon_query(url: str) -> Query:  # pragma: no cover - network
    import httpx

    ep = f"https://{urlsplit(url).hostname}/sql"
    headers = {"Neon-Connection-String": url, "Content-Type": "application/json"}

    def q(sql: str, params: list[Any]) -> list[dict[str, Any]]:
        r = httpx.post(ep, headers=headers, json={"query": sql, "params": params}, timeout=60)
        r.raise_for_status()
        rows = r.json()["rows"]
        return list(rows)

    return q


def iso_week(now: datetime) -> str:
    y, w, _ = now.isocalendar()
    return f"{y}-{w:02d}"


def fetch_texts(query: Query, days: int) -> tuple[list[str], list[str]]:
    """(known baseline older than ``days``, new attempts within ``days``), message text only."""
    new = [str(r["message"]) for r in query(NEW_SQL, [str(days)])]
    known = [str(r["message"]) for r in query(KNOWN_SQL, [str(days)])]
    return known, new


def make_embedder(
    known: Sequence[str], dims: int = SVD_DIMS, seed: int = 0
) -> Callable[[Sequence[str]], NDArray[np.float64]]:
    """TF-IDF (char 3-5 grams) + truncated SVD fitted on ``known``; the returned callable maps any texts."""
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    x = vec.fit_transform(list(known))
    k = max(2, min(dims, x.shape[1] - 1, x.shape[0] - 1))
    svd = TruncatedSVD(n_components=k, random_state=seed).fit(x)

    def embed(texts: Sequence[str]) -> NDArray[np.float64]:
        out = svd.transform(vec.transform(list(texts)))
        return np.asarray(normalize(out), dtype=np.float64)

    return embed


def skipped_note(week: str, reason: str) -> str:
    return f"# Drift report {week}\n\nSkipped: {reason}.\n"


def run(
    env: Mapping[str, str],
    *,
    query: Query | None = None,
    now: datetime | None = None,
    out_dir: Path = Path("reports"),
    days: int = 7,
    pusher: OtlpPusher | None = None,
) -> tuple[Path | None, DriftReport | None]:
    """Write ``reports/drift-YYYY-WW.md``. Returns (path, report); (None, None) when skipped for lack of secrets."""
    when = now or datetime.now(UTC)
    week = iso_week(when)
    url = env.get("NEON_DATABASE_URL", "")
    if query is None:
        if not url:
            print("skip: NEON_DATABASE_URL is not set")
            return None, None
        query = neon_query(url)
    known, new = fetch_texts(query, days)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"drift-{week}.md"
    if len(known) < MIN_KNOWN or len(new) < MIN_NEW:
        reason = f"not enough data (baseline {len(known)} < {MIN_KNOWN} or new {len(new)} < {MIN_NEW} attempts)"
        path.write_text(skipped_note(week, reason), encoding="utf-8")
        print("skip:", reason)
        return path, None
    rep, md = build_report(known, new, make_embedder(known), week)
    md += f"- baseline attempts: {len(known)}\n- embedding: char n-gram TF-IDF + SVD (CPU substitute, ADR draft)\n"
    path.write_text(md, encoding="utf-8")
    if pusher is not None:
        pusher.gauge("drift_share_outside_known", rep.share_outside_known)
        pusher.gauge("drift_new_attempts", float(rep.n_new))
        pusher.gauge("drift_flagged", 1.0 if rep.flagged else 0.0)
        res = pusher.flush()
        print(f"otlp push: ok={res.ok} http={res.status}")
    print(md)
    return path, rep


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m breachlab.monitor.drift_job")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = ap.parse_args(argv)
    e = os.environ if env is None else env
    t0 = time.perf_counter()
    run(
        e,
        days=args.days,
        out_dir=args.out_dir,
        pusher=pusher_from_env(e, service="breachlab-drift"),
    )
    print(f"drift job finished in {time.perf_counter() - t0:.1f}s")
    return 0  # a drift flag is information, not a failure (the report is the artifact)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
