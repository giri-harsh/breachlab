"""Orchestrates a dataset build: load -> mutate -> scrub -> dedup -> split -> validate -> leakage -> card."""

from __future__ import annotations

import json
import random
from pathlib import Path

from breachlab.data import hard_benign, mutator
from breachlab.data.card import render_card, stats
from breachlab.data.dedup import dedup
from breachlab.data.leakage import check_leakage
from breachlab.data.llm import LLMClient
from breachlab.data.loaders import SPECS
from breachlab.data.pii import scrub
from breachlab.data.schema import Row, jsonl_lines, validate_rows
from breachlab.data.splits import assign_splits


def _cached_hard_benign(llm: LLMClient, calls: int, cache: Path, workers: int = 1) -> list[Row]:
    """Resumable: hard-benign LLM rows are generated once and cached next to the shards."""
    if cache.exists():
        return [Row.model_validate_json(x) for x in jsonl_lines(cache.read_text("utf-8"))]
    rows = hard_benign.generate_llm_batch(llm, calls, workers=workers)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(r.model_dump_json() for r in rows) + "\n", encoding="utf-8")
    return rows


def balance_safe(rows: list[Row], rng: random.Random) -> list[Row]:
    """Downsample plain-SAFE rows so SAFE ~= UNSAFE; hard-benign rows are always kept (>=15% target)."""
    unsafe = sum(r.label == "UNSAFE" for r in rows)
    hard = [r for r in rows if r.label == "SAFE" and r.attack_family == "hard_benign"]
    plain = [r for r in rows if r.label == "SAFE" and r.attack_family != "hard_benign"]
    keep = max(0, unsafe - len(hard))
    if len(plain) > keep:
        plain = rng.sample(plain, keep)
    kept = {r.id for r in hard + plain}
    return [r for r in rows if r.label == "UNSAFE" or r.id in kept]


def build(
    seed_rows: list[Row],
    llm: LLMClient,
    out_dir: Path,
    *,
    version: str = "v1",
    n_hard_benign: int = 100,
    n_extraction: int = 50,
    gold_n: int = 1500,
    mining_n: int = 5000,
    seed: int = 42,
    shard_size: int = 50,
    batch: bool = False,
    k_det: int | None = None,
    workers: int = 1,
    llm_parent_limit: int | None = None,
    hb_llm_calls: int = 0,
    hb_llm: LLMClient | None = None,
    n_combinatorial: int = 0,
    n_quoted: int = 0,
    extra_det: int = 0,
    balance: bool = False,
    max_llm_shards: int | None = None,
) -> list[Row]:
    rng = random.Random(seed)
    attacks = [r for r in seed_rows if r.label == "UNSAFE"]
    llm_parents = None
    if llm_parent_limit is not None:
        llm_parents = {
            r.id for r in random.Random(seed).sample(attacks, min(llm_parent_limit, len(attacks)))
        }
    synthetic = mutator.run_sharded(
        attacks,
        llm,
        out_dir / "shards",
        shard_size=shard_size,
        seed=seed,
        batch=batch,
        k_det=k_det,
        workers=workers,
        llm_parents=llm_parents,
        max_llm_shards=max_llm_shards,
    )
    if extra_det:
        have = {r.text for r in synthetic}
        erng = random.Random(seed + 1)
        for atk_row in attacks:
            fresh = [v for v in mutator.mutate_deterministic(atk_row, erng) if v.text not in have]
            for v in erng.sample(fresh, min(extra_det, len(fresh))):
                have.add(v.text)
                synthetic.append(v)
    synthetic += mutator.secret_extraction_rows(n_extraction, rng)
    hb = hard_benign.seed_templates(n_hard_benign, rng)
    if n_quoted:
        hb += hard_benign.quoted_seed_rows(attacks, n_quoted, rng)
    if n_combinatorial:
        hb += hard_benign.combinatorial(n_combinatorial, rng)
    if hb_llm_calls:
        hb += _cached_hard_benign(
            hb_llm or llm, hb_llm_calls, out_dir / "hard_benign.jsonl", workers=workers
        )
    allrows = seed_rows + synthetic + hb
    if balance:
        allrows = balance_safe(allrows, rng)
    allrows = [r.with_(text=scrub(r.text)) for r in allrows]
    ids = {r.id for r in allrows}
    allrows = [
        r if r.parent_id in ids or r.parent_id is None else r.with_(parent_id=None) for r in allrows
    ]
    ids_after: dict[str, str] = {}
    deduped = dedup(allrows)
    kept = {r.id for r in deduped}
    # a dropped parent must not orphan children: re-point to nearest kept ancestor or None
    parent_of = {r.id: r.parent_id for r in allrows}
    for r in deduped:
        p = r.parent_id
        while p is not None and p not in kept:
            p = parent_of.get(p)
        ids_after[r.id] = p or ""
    final = [r.with_(parent_id=ids_after[r.id] or None) for r in deduped]
    final = assign_splits(final, gold_n=gold_n, mining_n=mining_n, seed=seed)
    validate_rows(final)
    check_leakage(final)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "mining_pool", "gold_test"):
        lines = [r.model_dump_json() for r in final if r.split == split]
        (out_dir / f"{split}.jsonl").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
    licenses = {s.repo_id: s.license for s in SPECS.values()}
    (out_dir / "DATASET_CARD.md").write_text(
        render_card(final, version, licenses), encoding="utf-8"
    )
    (out_dir / "stats.json").write_text(json.dumps(stats(final), indent=2), encoding="utf-8")
    return final
