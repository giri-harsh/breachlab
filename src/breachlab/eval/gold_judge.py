"""LLM-judge double-check of the gold_test labels (PRD 8.3) and the frontier-judge comparison row.

One Gemini pass over every gold row serves both purposes: the judgments are the "gemini-judge" row of the
report, and every row where the judge disagrees with the generation-time label is a label-noise candidate
that is queued for human review. Judgments are cached (resumable); a row whose judgment could not be parsed is
*missing*, never silently SAFE.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from breachlab.data.llm import LLMClient, LLMError
from breachlab.data.schema import Row
from breachlab.eval.external import JUDGE_PROMPT, parse_judge

CATEGORIES = ("none", "injection", "jailbreak", "exfiltration")
Cache = dict[str, dict[str, Any]]


@dataclass(frozen=True)
class Judgment:
    label: str
    category: str
    p_unsafe: float


def clean_item(item: Mapping[str, Any]) -> Judgment | None:
    """Strictly validate one parsed judge object; None when it is not a usable judgment."""
    label = str(item.get("label", "")).upper()
    if label not in ("SAFE", "UNSAFE"):
        return None
    cat = str(item.get("category", "none")).lower()
    if label == "SAFE" or cat not in CATEGORIES:
        cat = "none" if label == "SAFE" else "injection"
    raw = item.get("p_unsafe")
    p = float(raw) if isinstance(raw, int | float) and not isinstance(raw, bool) else None
    if p is None or not 0.0 <= p <= 1.0:
        p = 1.0 if label == "UNSAFE" else 0.0
    return Judgment(label, cat, p)


def judge_chunk(texts: Sequence[str], llm: LLMClient, max_chars: int = 1500) -> dict[int, Judgment]:
    """Judge one batch; only validly parsed items are returned (indexes into ``texts``)."""
    items = "\n".join(f"[{i}] {t[:max_chars]!r}" for i, t in enumerate(texts))
    raw = llm.complete(JUDGE_PROMPT.format(items=items), temperature=0.0, max_tokens=4000)
    out: dict[int, Judgment] = {}
    for i, it in parse_judge(raw, len(texts)).items():
        j = clean_item(it)
        if j is not None:
            out[i] = j
    return out


def run_judge(
    rows: Sequence[Row],
    cache: Cache,
    clients: Sequence[tuple[str, LLMClient]],
    *,
    per_call: int = 20,
    workers: int = 3,
    passes: int = 3,
    on_progress: Callable[[Cache], None] | None = None,
    batch_chunks: int = 6,
) -> Cache:
    """Fill ``cache`` (id -> judgment dict) for every row; resumable and failover across clients.

    Each pass retries the still-missing rows with the next client (round-robin by pass number), so a client that
    returns garbage or hits its quota does not stall the run. Rows still missing after ``passes`` stay missing.
    """
    for p in range(passes):
        todo = [r for r in rows if r.id not in cache]
        if not todo:
            break
        chunks = [todo[i : i + per_call] for i in range(0, len(todo), per_call)]

        def one(chunk: list[Row], p: int = p) -> list[tuple[str, str, Judgment]]:
            answered_empty = False
            for k in range(len(clients)):
                name, llm = clients[(p + k) % len(clients)]
                try:
                    got = judge_chunk([r.text for r in chunk], llm)
                except (LLMError, ValueError, KeyError, OSError):
                    continue
                if got:
                    return [(chunk[i].id, name, j) for i, j in got.items()]
                answered_empty = True
            if len(chunk) > 1 and answered_empty:
                # a client answered but with nothing usable (safety block or garbage on some item of this
                # chunk): bisect so one blocked text does not hide its neighbours
                mid = len(chunk) // 2
                return one(chunk[:mid], p) + one(chunk[mid:], p)
            return []

        for s in range(0, len(chunks), batch_chunks):
            with ThreadPoolExecutor(workers) as ex:
                for part in ex.map(one, chunks[s : s + batch_chunks]):
                    for rid, name, j in part:
                        cache[rid] = {
                            "label": j.label,
                            "category": j.category,
                            "p_unsafe": j.p_unsafe,
                            "model": name,
                        }
            if on_progress:
                on_progress(cache)
    return cache


def load_cache(path: Path) -> Cache:
    return json.loads(path.read_text("utf-8")) if path.exists() else {}


def save_cache(path: Path, cache: Cache) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache), encoding="utf-8")
    tmp.replace(path)


def disagreements(rows: Sequence[Row], cache: Cache) -> list[dict[str, Any]]:
    """Rows where the judge's label differs from the generation-time label."""
    out = []
    for r in rows:
        j = cache.get(r.id)
        if j is not None and j["label"] != r.label:
            out.append(
                {
                    "id": r.id,
                    "text": r.text,
                    "gen_label": r.label,
                    "gen_category": r.category,
                    "judge_label": j["label"],
                    "judge_category": j["category"],
                    "judge_p_unsafe": j["p_unsafe"],
                    "attack_family": r.attack_family,
                    "encoding": r.encoding,
                    "language": r.language,
                    "source": r.source,
                }
            )
    return out


def agreement_stats(rows: Sequence[Row], cache: Cache) -> dict[str, Any]:
    """Agreement of the judge with the generation-time labels, overall and by slice."""
    judged = [(r, cache[r.id]) for r in rows if r.id in cache]
    n = len(judged)
    agree = sum(j["label"] == r.label for r, j in judged)
    po = agree / n if n else 0.0
    tp = sum(r.label == "UNSAFE" and j["label"] == "UNSAFE" for r, j in judged)
    fp = sum(r.label == "SAFE" and j["label"] == "UNSAFE" for r, j in judged)
    fn = sum(r.label == "UNSAFE" and j["label"] == "SAFE" for r, j in judged)
    tn = n - tp - fp - fn
    # Cohen's kappa between two raters (generation-time label, judge)
    pe = ((tp + fn) * (tp + fp) + (fp + tn) * (fn + tn)) / (n * n) if n else 0.0
    kappa = (po - pe) / (1 - pe) if n and pe < 1 else 0.0
    by: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r, j in judged:
        for key, val in (
            ("attack_family", r.attack_family),
            ("encoding", r.encoding),
            ("language", r.language),
            ("source", r.source),
        ):
            cell = by[key][val]
            cell[0] += 1
            cell[1] += j["label"] != r.label
    models = Counter(j["model"] for _, j in judged)
    return {
        "n_rows": len(rows),
        "n_judged": n,
        "n_missing": len(rows) - n,
        "agreement": po,
        "kappa": kappa,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "gen_unsafe_judged_safe": fn,
        "gen_safe_judged_unsafe": fp,
        "by_slice": {
            k: {v: {"n": c[0], "disagree": c[1]} for v, c in sorted(vals.items())}
            for k, vals in by.items()
        },
        "judge_models": dict(models),
    }


def judge_row_preds(rows: Sequence[Row], cache: Cache) -> dict[str, list[object]] | None:
    """`api_preds.json` entry ({id: [label, category, p_unsafe]}); None unless every row was judged."""
    if any(r.id not in cache for r in rows):
        return None
    return {
        r.id: [cache[r.id]["label"], cache[r.id]["category"], cache[r.id]["p_unsafe"]] for r in rows
    }


def review_queries(
    session_id: str, items: Sequence[Mapping[str, Any]], existing: set[str]
) -> list[dict[str, Any]]:
    """HTTPS-SQL statements adding disagreement rows to the review session (level 0), skipping known messages.

    Each attempt gets one label row (source 'judge') holding the Gemini verdict, with the generation-time label
    in the rationale so the reviewer sees both; labels.attempt_id is the primary key, so one label per attempt.
    """
    qs: list[dict[str, Any]] = []
    for it in items:
        if it["text"] in existing:
            continue
        qs.append(
            {
                "query": "insert into attempts (session_id, level, message, confidence, leaked) "
                "values ($1, 0, $2, 0.5, false)",
                "params": [session_id, it["text"]],
            }
        )
        rationale = (
            f"gold_judge disagreement: generation-time {it['gen_label']}/{it['gen_category']} "
            f"(family {it['attack_family']}, enc {it['encoding']}, lang {it['language']}); "
            f"judge {it['judge_label']}/{it['judge_category']}"
        )
        qs.append(
            {
                "query": "insert into labels (attempt_id, label, category, source, judge_rationale, "
                "judge_confidence) values (currval(pg_get_serial_sequence('attempts','id')), "
                "$1, $2, 'judge', $3, $4)",
                "params": [
                    it["judge_label"],
                    it["judge_category"] if it["judge_label"] == "UNSAFE" else None,
                    rationale,
                    it["judge_p_unsafe"],
                ],
            }
        )
    return qs
