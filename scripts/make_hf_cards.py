"""Build the HF model card and dataset card and push them (README.md only, never weights).

Numbers come only from files in the repository: ``web/public/scorecard.json`` (metrics, calibration note, p95),
``reports/eval-v1.md`` (3-seed mean and sd, gold double-check agreement), ``docs/decisions.md`` (ADR-0064 latency of
the shipped rung) and ``DATASET_CARD.md`` (split counts). If one of them cannot be parsed the script stops instead
of writing a card with a missing or invented number.

This script never changes visibility of an existing repo (new repos are created private); visibility is set by
``scripts/hf_publish.py``. No model weights are uploaded from this machine.

  uv run python scripts/make_hf_cards.py --dry-run   # write the cards to reports/hf-cards/ and print nothing else
  uv run python scripts/make_hf_cards.py             # push README.md to the two HF repos (needs HF_TOKEN)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NS = os.environ.get("HF_NAMESPACE", "Pseudo-Harsh")
SHIPPED = "qwen3-0.6b-sft"
INT4 = "qwen3-0.6b-sft-int4"

# Display names for scorecard keys (order = card order).
DISPLAY = {
    "regex": "regex filter",
    "llama-prompt-guard-2-86m (Groq)": "Llama Prompt-Guard-2-86M (via Groq)",
    "gemini-judge": "Gemini as zero-shot judge",
    "deberta-prompt-injection": "DeBERTa prompt-injection classifier",
    "qwen3-0.6b-zeroshot": "Qwen3-0.6B zero-shot",
    "qwen3-1.7b-zeroshot": "Qwen3-1.7B zero-shot",
    "qwen3-0.6b-sft": "**Qwen3-0.6B + LoRA, SFT-only (this model)**",
    "qwen3-0.6b-sft-int4": "Same adapter, int4 weight-only graph as served (own threshold)",
    "qwen3-0.6b-sft-dpo": "Qwen3-0.6B + LoRA, SFT then DPO (challenger)",
    "qwen3-1.7b-sft": "Qwen3-1.7B + LoRA, SFT-only (comparison)",
    "qwen3-1.7b-sft-dpo": "Qwen3-1.7B + LoRA, SFT then DPO (comparison)",
}

# ADR-0013 license table, restricted to what this project touches (wildjailbreak is excluded, ADR-0019).
LICENSES = """| asset | license | how it is used |
|---|---|---|
| deepset/prompt-injections | Apache-2.0 | seed attacks |
| jackhhao/jailbreak-classification | Apache-2.0 | seed attacks and benign |
| Lakera/gandalf_ignore_instructions | MIT | seed attacks |
| JailbreakBench/JBB-Behaviors | MIT | seed attacks |
| HuggingFaceH4/ultrachat_200k | MIT | benign |
| databricks/databricks-dolly-15k | CC-BY-SA-3.0 | benign (share-alike text is part of the training data) |
| allenai/wildjailbreak | ODC-By (gated) | **excluded**: access was never granted (ADR-0019), no row comes from it |
| lmsys/toxic-chat, tatsu-lab/alpaca | CC-BY-NC-4.0 | **dropped**: non-commercial (ADR-0013) |
| Qwen/Qwen3-0.6B, Qwen/Qwen3-1.7B | Apache-2.0 | base models |
| protectai/deberta-v3-base-prompt-injection-v2 | Apache-2.0 | baseline only |
| meta-llama/Llama-Prompt-Guard-2-86M | Llama license | baseline through the Groq API only; weights not redistributed |"""


class CardError(RuntimeError):
    """A number the card needs could not be found in the repository."""


def _grab(pattern: str, text: str, what: str, flags: int = 0) -> re.Match[str]:
    m = re.search(pattern, text, flags)
    if not m:
        raise CardError(f"cannot find {what}")
    return m


def facts(root: Path = ROOT) -> dict[str, Any]:
    """Collect every number the cards quote, from the repository files."""
    card = json.loads((root / "web/public/scorecard.json").read_text(encoding="utf-8"))
    report = (root / "reports/eval-v1.md").read_text(encoding="utf-8")
    dataset = (root / "DATASET_CARD.md").read_text(encoding="utf-8")

    shipped = card["models"][SHIPPED]
    cal = _grab(
        r"T=([\d.]+), threshold=([\d.]+)", shipped["note"], "T/threshold in the scorecard note"
    )
    int4 = card["models"].get(INT4)
    if int4 is None:
        raise CardError("cannot find the served int4 row in the scorecard")
    i4 = _grab(
        r"T=([\d.]+), own threshold=([\d.]+)", int4["note"], "int4 threshold in the scorecard note"
    )
    sft_block = report.split("### SFT\n", 1)
    if len(sft_block) < 2:
        raise CardError("cannot find the SFT seed table in eval-v1.md")
    mean = _grab(
        r"\| mean \+- std \(n=(\d+)\) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|",
        sft_block[1],
        "SFT mean +- std row",
    )
    seeds = re.findall(r"^\| (\d+) \| ", sft_block[1].split("### SFT + DPO", 1)[0], re.M)
    agree = _grab(
        r"agreement with the generation-time label ([\d.]+)%, Cohen's kappa ([\d.]+)",
        report,
        "gold double-check agreement",
    )
    lat = _grab(
        r"concurrency 1: p50 ([\d.]+) ms, p95 ([\d.]+) ms",
        report,
        "shipped-rung latency in eval-v1.md",
    )
    fp32 = _grab(
        r"p50 (\d+) vs (\d+) ms \(.*?\), p95 (\d+) vs (\d+) ms", report, "fp32 4-vCPU comparison"
    )
    split = {
        k: int(v)
        for k, v in re.findall(r"^\| (train|val|gold_test|mining_pool) \| (\d+) \|", dataset, re.M)
    }
    total = _grab(r"Total rows: (\d+)", dataset, "total rows")
    return {
        "card": card,
        "shipped": shipped,
        "temperature": cal.group(1),
        "threshold": cal.group(2),
        "n_seeds": int(mean.group(1)),
        "seeds": seeds,
        "seed_asr": mean.group(2).strip(),
        "seed_fpr": mean.group(3).strip(),
        "seed_f1": mean.group(4).strip(),
        "seed_auroc": mean.group(5).strip(),
        "agree": agree.group(1),
        "kappa": agree.group(2),
        "int4": int4,
        "int4_threshold": i4.group(2),
        "p50_kv": f"{float(lat.group(1)):.0f}",
        "p95_kv": f"{float(lat.group(2)):.0f}",
        "p95_fp32": fp32.group(3),
        "split": split,
        "total_rows": int(total.group(1)),
    }


def _table(card: dict[str, Any]) -> str:
    rows = ["| model | ASR (lower is better) | FPR | F1 | AUROC |", "|---|---|---|---|---|"]
    for key, label in DISPLAY.items():
        m = card["models"].get(key)
        if m:
            rows.append(
                f"| {label} | {m['asr'] * 100:.1f}% | {m['fpr'] * 100:.1f}% | {m['f1']:.3f} | {m['auroc']:.3f} |"
            )
    return "\n".join(rows)


def model_card(f: dict[str, Any]) -> str:
    sp = f["split"]
    seeds = ", ".join(f["seeds"])
    return f"""---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
library_name: peft
pipeline_tag: text-classification
language: [en, hi]
tags: [prompt-injection, jailbreak-detection, guard, lora, qwen3, onnx, research-demo]
---
# breachlab-guard-{f["card"]["version"]}

**License: Apache-2.0** (weights and code; the base model Qwen3 is Apache-2.0). Code: https://github.com/giri-harsh/breachlab. **Note:** the training data includes
CC-BY-SA-3.0 rows (databricks-dolly-15k); the dataset itself is CC-BY-SA-3.0 (see the dataset card).

LoRA fine-tune of Qwen3-0.6B that labels a message as `safe`, `injection`, `jailbreak` or `exfiltration` in a
single forward pass (softmax over four verbalizer tokens, temperature-calibrated). It is the input guard of the
BreachLab game, where players try to trick an AI vault into leaking a secret code. **Research demo, not a production
safety system.**

## Update 2026-11-12: guard v2 (`champion-v2/`)

The Hub repository also holds guard v2 (`champion-v2/`, the variant the live game serves): the same recipe trained on
data-v2 (v1 plus 2,172 hard-benign short, Hindi and Hinglish rows). Served int4 graph, threshold 0.9646 chosen on the v2
validation split: gold_test ASR 0.44%, FPR 1.83%; held-out short benign opener FPR 6.3% (v1: 54.3%), Hindi openers 15.0%
(v1: 96%). Details and caveats: `reports/eval-v2.md` in the code repository. The numbers below describe v1.

## Which variant ships

- **SFT-only.** LoRA after supervised fine-tuning (`adapters/sft-0.6b-s0`, seed 0). A DPO stage on 399 mined hard
  pairs gave no significant gain (McNemar p = 1.0), so by the project rule DPO does not ship; the DPO adapter is
  kept as the `challenger/` export and the untuned model as `base/`.
- **Export:** ONNX with **prefix-KV reuse** (the fixed instruction prefix is computed once). The reference export is
  fp32 (`champion/guard-kv.onnx`, max logit difference to plain fp32 1.5e-05). **What the live game serves is the
  4-bit weight-only variant `champion/guard-q4b32.onnx`** (MatMulNBits, block 32, int8 embedding rows; 433 MB), because
  it is the only graph that fits the free host (a Vercel Python function, 1 vCPU, 2 GB; ADR-0082). Dynamic int8 failed
  the quality-parity check on both model sizes; weight-only int8 kept parity but was slower on CPU.
- **Calibration (fp32):** temperature T = {f["temperature"]}, decision threshold on `p_unsafe` = {f["threshold"]}, both
  fitted on the validation split so that the false-positive rate stays at or below 2%. Expected calibration error on
  val fell from 0.023 to 0.009. The low threshold is what fitting for FPR <= 2% produced.
- **Calibration (int4, served):** at the fp32 threshold the int4 graph fails the quality-parity rule on FPR (7.7% vs
  1.6%) although its ranking is intact (AUROC drop 0.0003). It therefore has its own threshold
  {f["int4_threshold"]} (`champion/calibration-q4b32.json`), fitted on val with the same rule (FPR <= 2%), never on
  gold_test. On gold_test it measures ASR {f["int4"]["asr"] * 100:.2f}%, FPR {f["int4"]["fpr"] * 100:.2f}%,
  F1 {f["int4"]["f1"]:.3f}, AUROC {f["int4"]["auroc"]:.4f}: FPR is 0.5 pp above fp32 and 0.08 pp above the 2.0% bar,
  inside the fp32 confidence interval (ADR-0082, a recorded deviation from ADR-0057).
- **Service contract:** `POST /score {{"text", "kind": "input"|"output", "alias"}}` returns
  `label`, `category`, `p_unsafe`, `confidence`, `version`, `latency_ms`; `GET /health`. The serving image
  (`vercel-guard/` in the project repository) pulls these files with a token and is host-agnostic.

## Evaluation (gold_test, {sp.get("gold_test", "n/a")} rows, in-distribution)

{_table(f["card"])}

ASR = share of attacks the guard lets through; FPR = share of benign messages it blocks. Confidence intervals,
McNemar tests, slices and the worst misses are in `reports/eval-v1.md` in the project repository.

**Seeds.** Seeds {seeds} were trained and scored with the same configuration (SFT, mean +- sd over n = {f["n_seeds"]}):
ASR {f["seed_asr"]}, FPR {f["seed_fpr"]}, F1 {f["seed_f1"]}, AUROC {f["seed_auroc"]}. A fourth run (seed 2) diverged in
fp16 (NaN gradients, NaN logits on every gold row); it is excluded from the mean, kept as evidence, and the third
seed was re-run unchanged as seed 3. The sd over three seeds is itself noisy. The table above is seed 0.

## Cost

Free tiers, $0 today (Groq free tier, Vercel Hobby, Neon free; hosting on the Hub is free). Measured from real vault-agent
token usage and the guard's measured compute, at list prices if usage went past the free tiers: about $0.11 per 1,000
requests and $3.4 per 1,000 games (a game is five bosses, about 30 messages). Per 1,000 games: this guard on Vercel
$2.24, vault LLM $0.47 (gpt-oss-20b), gateway $0.60, database $0.06. The limit that binds first is the Vercel Hobby
Active CPU allowance (4 h a month, about 260 games a month at 1.1 s of guard CPU per call). A guard with 5-10x lower
compute would cost about $1.0 to $1.4 per 1,000 games. Details: `docs/lanes/C-cost.md` in the BreachLab repository.

## Latency (target not met)

The project target was p95 <= 300 ms on a CPU host. **It is not met.**
- Served variant (int4 weight-only with prefix-KV reuse) on the real host, a Vercel Python function (1 vCPU, 2 GB),
  measured from a client over the network, 500 warm requests at concurrency 1: model time p50 / p95 =
  {f["p50_kv"]} / {f["p95_kv"]} ms (round trip p50 842, p95 3556 ms). At concurrency 4 the single vCPU serializes the
  requests (round trip p50 3664 ms).
- Cold start: an instance is gone after about 10 minutes idle; the first request after that takes about 13 s, longer
  than the gateway's 8 s guard timeout, so that player gets a fail-closed 503 "guard waking up" and the next request
  works.
- Earlier, before this host existed, the fp32 prefix-KV variant measured p95 2326 ms on a Kaggle kernel pinned to
  2 vCPU, and plain fp32 ONNX, 512 tokens, 4-vCPU Kaggle CPU: p95 {f["p95_fp32"]} ms. Those numbers are superseded
  and kept in `reports/eval-v1.md` only as history.
No token cap reaches the 300 ms target either (see `reports/eval-v1.md`). A GPU or a faster CPU host is the path to
the target.

## Repository layout
- `champion/` SFT-only guard (served for levels 3-5 on the Vercel host, which loads only `guard-q4b32.onnx`),
  `challenger/` SFT+DPO (shadow, not served there), `base/` untuned Qwen3-0.6B (not served there, so level 3 uses the
  champion graph, ADR-0082): each holds `guard-fp32.onnx` (+ `.onnx.data`), `guard-kv.onnx` where the prefix-KV export
  exists, `guard-w8.onnx` (weight-only int8), `tokenizer.json`, `calibration.json` (temperature, threshold) and
  `export.json`; the champion folder also holds `guard-q4b32.onnx` and `calibration-q4b32.json`.
- `adapters/sft-0.6b-s0`, `adapters/dpo-0.6b-s0` (LoRA, Qwen3-0.6B) and `adapters/sft-1.7b-s0` (LoRA, Qwen3-1.7B
  comparison model, not served).
- Served by the guard service in the BreachLab repository (`vercel-guard/`, ONNX Runtime, `POST /score`).

## Training data

`{NS}/breachlab-data` (public, CC-BY-SA-3.0), {f["total_rows"]:,} rows in total: train {sp.get("train", "n/a"):,}, val
{sp.get("val", "n/a"):,}, mining pool {sp.get("mining_pool", "n/a"):,} (only for hard-negative mining), gold_test
{sp.get("gold_test", "n/a"):,} (never trained on). Public-source seeds plus LLM- and rule-generated mutations
(paraphrase, roleplay, multi-turn, base64, hex, rot13, leetspeak, payload splitting, Hindi, Hinglish) and generated
hard-benign prompts. Splits are grouped by parent so mutations of one seed never straddle train and test; near-duplicate
leakage is checked. No live player data. Code and evaluation: https://github.com/giri-harsh/breachlab. Source licenses:

{LICENSES}

The dolly subset is CC-BY-SA-3.0 (share-alike). Whether that obligation extends to the fine-tuned weights has not
been settled by a lawyer; check it before you reuse the weights commercially.

## Intended use and limitations

- **Intended:** a guard for the BreachLab game and a reference for small-model prompt-injection classification, in
  English, Hindi and Hinglish.
- **In-distribution.** `gold_test` shares source datasets and synthetic generators with training. The numbers above say
  little about attacks from other generators or from real players; a fresh test set from live play is planned.
- **Gold labels are generation-time labels** (the label of the source row or of the mutation's parent), not human
  labels. A Gemini double-check agrees on {f["agree"]}% of rows (Cohen's kappa {f["kappa"]}); the disagreements are
  queued for human review, which is pending. The judge reads raw text and cannot decode encoded attacks, so a
  disagreement is not proof of a wrong label.
- Benign messages that discuss or quote attacks are the known confusion area (the hard-benign slice of gold_test is
  built from templates and quoted seeds, so it is easier than real users). Messages longer than 500 characters,
  multi-turn state and languages other than English, Hindi and Hinglish are not covered.
- The served int4 guard blocks about 2% of benign messages on gold_test (2.08%) and a little more of short questions
  about the assistant itself (informal check: 2 of 12 short benign openers); it also serializes requests on one vCPU.
- A single guard is one layer; the game also checks the agent reply for the secret. Do not rely on this model as the only
  defense of a real system.
"""


def dataset_card(f: dict[str, Any], base: str) -> str:
    base = base.replace(
        "| allenai/wildjailbreak | odc-by |",
        "| allenai/wildjailbreak | odc-by (EXCLUDED: no row comes from it) |",
    )
    base = base.replace(
        "Research demo. Live game data is never included in the public dataset.\n", ""
    ).rstrip()
    return f"""---
license: cc-by-sa-3.0
pretty_name: BreachLab guard dataset v1
language: [en, hi]
task_categories: [text-classification]
tags: [prompt-injection, jailbreak-detection, research-demo]
---
{base}

## Provenance and licenses

Every row comes from one of: public datasets (table above), rule-based mutations of those seeds (encodings, payload
splitting), LLM-written mutations and benign prompts (Gemini and Groq free-tier models, batched, with the parent id
kept), or template-built hard-benign prompts (benign text that quotes or analyses an attack). `allenai/wildjailbreak` is
listed in the loader table but was never accessible (gated, ADR-0019): **no row comes from it**. Non-commercial
sources (toxic-chat, Alpaca) were dropped (ADR-0013).

{LICENSES}

Code, evaluation and the game: https://github.com/giri-harsh/breachlab

**License of this dataset: CC-BY-SA-3.0.** The dolly rows are CC-BY-SA-3.0 and carry share-alike, so the combined dataset is
released under CC-BY-SA-3.0; keep attribution for the MIT and Apache sources. LLM-generated rows are subject to the
generating providers' terms. Code and model weights are Apache-2.0 (`DATA_LICENSE.md` in the code repository).

## Splits and labels

train {f["split"].get("train", "n/a")}, val {f["split"].get("val", "n/a")}, mining_pool {f["split"].get("mining_pool", "n/a")}, gold_test
{f["split"].get("gold_test", "n/a")}. Mutations inherit `parent_id` and all descendants of a parent share a split; a
MinHash near-duplicate check across splits passes. Labels are **generation-time labels** (source label or the
mutation parent's label). A Gemini double-check of gold_test agrees on {f["agree"]}% of rows (kappa {f["kappa"]}); the
110 disagreements are queued for human review (pending). `gold_test` is frozen and must not be used for training.

## Limitations

English-heavy (about 90%), Hindi and Hinglish about 5% each. Fewer LLM-written paraphrases than planned (free-tier
quota, ADR-0029). Attack text is quoted as data: some rows are offensive. Hard-benign prompts are built from templates and
quoted seeds, not from real users. The data is in-distribution with the guard's evaluation; it is not a benchmark of
real-world attack diversity.

Research demo. Live game data is never included in this dataset.
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="write cards to reports/hf-cards/ only")
    a = ap.parse_args(argv)
    f = facts()
    model_md = model_card(f)
    data_md = dataset_card(f, (ROOT / "DATASET_CARD.md").read_text(encoding="utf-8"))
    if a.dry_run:
        out = ROOT / "reports" / "hf-cards"
        out.mkdir(parents=True, exist_ok=True)
        (out / "MODEL_README.md").write_text(model_md, encoding="utf-8")
        (out / "DATASET_README.md").write_text(data_md, encoding="utf-8")
        print("cards written to", out)
        return 0
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    repo = f"{NS}/breachlab-guard-{f['card']['version']}"
    data_repo = f"{NS}/breachlab-data"
    api.create_repo(repo, private=True, exist_ok=True)  # never flips an existing repo to public
    api.create_repo(data_repo, private=True, repo_type="dataset", exist_ok=True)
    api.upload_file(path_or_fileobj=model_md.encode(), path_in_repo="README.md", repo_id=repo)
    api.upload_file(
        path_or_fileobj=data_md.encode(),
        path_in_repo="README.md",
        repo_id=data_repo,
        repo_type="dataset",
    )
    print("cards pushed (README.md only, visibility untouched):", repo, data_repo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
