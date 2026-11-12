"""data-v2 (lane T, supervisor A4): hard-benign rows for the guard v2 retrain.

    python -m breachlab.data.v2 --out data/processed/v2

Generates ~4,000 new SAFE rows with LLMs (Groq + Gemini, free quotas): short openers, greetings, small talk, questions
about the assistant, short factual questions, "rules of a game" questions, in English, Hindi (Devanagari) and Hinglish,
plus longer Hindi/Hinglish benign messages. Rows are filtered (no words that make a benign row look like an attack on
the vault), PII-scrubbed and de-duplicated by exact text and MinHash against train, val, gold_test, the mining pool,
the held-out S16 opener set and the canary probes, so the evaluation sets stay held out. The new rows are split 88/12
into train and val (val then also covers the new distribution for calibration); gold_test and the mining pool are
copied unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
from collections.abc import Sequence
from pathlib import Path

from datasketch import MinHashLSH

from breachlab.data.dedup import NUM_PERM, THRESHOLD, minhash
from breachlab.data.leakage import check_leakage
from breachlab.data.llm import GeminiClient, GroqClient, LLMClient, LLMError, RoundRobinLLM
from breachlab.data.openers import CATEGORY_HELP, STYLE, parse_lines
from breachlab.data.pii import has_pii
from breachlab.data.schema import Language, Row

SOURCE = "hardbenign-v2"
DEVANAGARI = re.compile("[ऀ-ॿ]")
# unlike the opener slice, "rules" is allowed here (benign "rules of cricket" questions are what the canary probes)
BANNED = re.compile(
    r"\b(ignore|instruction|instructions|system prompt|prompt|secret|password|passcode|code|vault|reveal|bypass|"
    r"jailbreak|override|developer|admin|ignore karo|bhool jao)\b",
    re.IGNORECASE,
)
CATEGORY_HELP = {
    **CATEGORY_HELP,
    "short_question": "short everyday factual or how-to questions (one line, a few words)",
    "rules_of_things": "questions asking someone to explain the rules of a game, sport or a board game, or how a festival or custom works",
    "longer_benign": "longer harmless messages of 10 to 25 words: a question or request about cooking, travel, study, health, money or hobbies",
}
# rows per (language, category); sums to 1,000 per language plus 500 longer rows for Hindi and Hinglish
PLAN_SHORT = {
    "greeting": 150,
    "small_talk": 200,
    "meta_question": 200,
    "simple_request": 200,
    "short_question": 150,
    "rules_of_things": 100,
}
PLAN_LONG = {"longer_benign": 500}
MAX_WORDS = {"short": 14, "long": 30}
CHUNK = 40


def prompt_for(language: str, category: str, n: int) -> str:
    longer = category == "longer_benign"
    limit = MAX_WORDS["long"] if longer else MAX_WORDS["short"]
    return (
        f"Write {n} different opening messages a person might send to a friendly chatbot. Category: "
        f"{CATEGORY_HELP[category]}. Language: {STYLE[language]}. Each message has at most {limit} words, is natural, "
        "harmless, and differs in wording from the others. Never mention codes, passwords, secrets, instructions or "
        "prompts. Output exactly one message per line, with no numbering, no quotes and no commentary."
    )


def acceptable(line: str, language: str, category: str) -> bool:
    words = len(line.split())
    lo = 6 if category == "longer_benign" else 2
    hi = MAX_WORDS["long"] if category == "longer_benign" else MAX_WORDS["short"]
    if not (lo <= words <= hi) or BANNED.search(line) or has_pii(line):
        return False
    deva = bool(DEVANAGARI.search(line))
    return deva if language == "hi" else not deva


def normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def generate_cell(
    llm: LLMClient, language: Language, category: str, want: int, seen: set[str]
) -> list[str]:
    out: list[str] = []
    for _ in range(max(6, want // CHUNK * 3)):
        if len(out) >= want:
            break
        try:
            text = llm.complete(
                prompt_for(language, category, CHUNK), temperature=0.95, max_tokens=3000
            )
        except (LLMError, Exception):  # noqa: BLE001 - a provider outage or quota stop must not end the build
            continue
        for line in parse_lines(text):
            key = normalize(line)
            if key and key not in seen and acceptable(line, language, category):
                seen.add(key)
                out.append(line)
    return out


def read_rows(path: Path) -> list[Row]:
    return [
        Row.model_validate(json.loads(x)) for x in path.read_text("utf-8").split("\n") if x.strip()
    ]


def build_rows(
    llm: LLMClient,
    existing: Sequence[Row],
    plan_short: dict[str, int],
    plan_long: dict[str, int],
    extra_texts: Sequence[str] = (),
    partial: Path | None = None,
) -> tuple[list[Row], dict[str, int]]:
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    seen = {normalize(e.text) for e in existing} | {normalize(t) for t in extra_texts}
    for e in existing:
        lsh.insert(e.id, minhash(e.text))
    for i, t in enumerate(extra_texts):
        lsh.insert(f"extra-{i}", minhash(t))
    rows: list[Row] = []
    counts: dict[str, int] = {}
    done: dict[str, int] = {}
    if (
        partial is not None and partial.exists()
    ):  # resume: cells already finished by an earlier (cut-off) run
        for line in partial.read_text("utf-8").split("\n"):
            if line.strip():
                rec = json.loads(line)
                row = Row.model_validate(rec["row"])
                rows.append(row)
                lsh.insert(row.id, minhash(row.text))
                seen.add(normalize(row.text))
                done[rec["cell"]] = done.get(rec["cell"], 0) + 1
    langs_all: tuple[Language, ...] = ("en", "hi", "hinglish")
    langs_long: tuple[Language, ...] = ("hi", "hinglish")
    cells = [(lang, cat, n) for lang in langs_all for cat, n in plan_short.items()]
    cells += [(lang, cat, n) for lang in langs_long for cat, n in plan_long.items()]
    for lang, cat, n in cells:
        cell = f"{lang}/{cat}"
        if done.get(cell, 0) >= int(n * 0.9) or cell in done and done[cell] >= n:
            counts[cell] = done[cell]
            continue
        kept = 0
        cell_rows: list[Row] = []
        for line in generate_cell(llm, lang, cat, int(n * 1.5), seen):
            row = Row.new(
                line, "SAFE", "none", SOURCE, attack_family=f"hardbenign:{cat}", language=lang
            )
            mh = minhash(row.text)
            if kept >= n or lsh.query(mh):
                continue
            lsh.insert(row.id, mh)
            rows.append(row)
            cell_rows.append(row)
            kept += 1
        counts[cell] = kept
        if partial is not None:
            with partial.open("a", encoding="utf-8") as f:
                for r in cell_rows:
                    f.write(
                        json.dumps({"cell": cell, "row": r.model_dump()}, ensure_ascii=False) + "\n"
                    )
    return rows, counts


def split_rows(
    rows: list[Row], val_share: float = 0.12, seed: int = 0
) -> tuple[list[Row], list[Row]]:
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    n_val = int(len(shuffled) * val_share)
    return [r.with_(split="train") for r in shuffled[n_val:]], [
        r.with_(split="val") for r in shuffled[:n_val]
    ]


def scaled(plan: dict[str, int], factor: float) -> dict[str, int]:
    return {k: max(1, int(v * factor)) for k, v in plan.items()}


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - network
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/processed/v2"))
    ap.add_argument("--v1", type=Path, default=Path("data/processed/v1"))
    ap.add_argument(
        "--scale", type=float, default=1.0, help="fraction of the plan (0.01 for a smoke run)"
    )
    a = ap.parse_args(argv)
    existing = [
        r
        for f in ("train", "val", "gold_test", "mining_pool")
        for r in read_rows(a.v1 / f"{f}.jsonl")
    ]
    held_out = [r.text for r in read_rows(Path("data/review/opener_slice_v1.jsonl"))]
    from breachlab.monitor.canary import ATTACKS, BENIGN

    held_out += BENIGN + ATTACKS
    # qwen3.8-27b and gpt-oss-120b on Groq: fast (2-5 s per 40 lines) and a separate quota from the vault agent's
    # gpt-oss-20b; Gemini 2.5 flash needs about 90 s per call, too slow for ~150 calls
    providers: list[LLMClient] = (
        [GeminiClient()]
        if os.environ.get("DATA_V2_LLM") == "gemini"
        else [GroqClient("qwen/qwen3.8-27b"), GroqClient("openai/gpt-oss-120b")]
    )
    llm = RoundRobinLLM(providers)
    a.out.mkdir(parents=True, exist_ok=True)
    new, counts = build_rows(
        llm,
        existing,
        scaled(PLAN_SHORT, a.scale),
        scaled(PLAN_LONG, a.scale),
        held_out,
        a.out / "partial.jsonl",
    )
    train_new, val_new = split_rows(new)
    train = read_rows(a.v1 / "train.jsonl") + train_new
    val = read_rows(a.v1 / "val.jsonl") + val_new
    for name, rows in (("train", train), ("val", val)):
        (a.out / f"{name}.jsonl").write_text(
            "".join(r.model_dump_json() + "\n" for r in rows), "utf-8"
        )
    for name in ("gold_test", "mining_pool"):
        shutil.copy(a.v1 / f"{name}.jsonl", a.out / f"{name}.jsonl")
    check_leakage(
        train + val + read_rows(a.out / "gold_test.jsonl") + read_rows(a.out / "mining_pool.jsonl")
    )
    (a.out / "new_rows.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in new), "utf-8")
    stats = {
        "new_rows": len(new),
        "train_total": len(train),
        "val_total": len(val),
        "per_cell": counts,
    }
    (a.out / "v2_stats.json").write_text(json.dumps(stats, indent=1), "utf-8")
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
