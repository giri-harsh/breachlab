# Gold label review (LLM consensus)

**DONE-BY-DECISION: LLM-consensus review, not human.** Lane O, task L3. `gold_test.jsonl` is not modified; the proposed relabels are in `docs/lanes/O-gold-review.json` (per row: ids, judge verdicts, outcome, no texts).

## Which rows were reviewed

The 393 gold rows queued for review are 300 random-sample rows plus 93 rows where the original Gemini double-check disagreed with the generation-time label (ADR-0058, Neon session `gold-review-v1`). The list of 393 ids is not versioned in the repository: the Neon queue is not readable from this lane (its connection string is outside the credentials this lane may use), and the Gemini judge cache `data/eval/gold_judge.json` is not in git or in the local DVC cache. The set was therefore **re-derived**:

- the same 300-row sample (`data/processed/v1/review_queue_gold300.jsonl`, 300 of 1,500 gold ids, proportional by source), plus
- every gold row on which a fresh Gemini screen (the original `JUDGE_PROMPT`, `gemini-3.1-flash-lite` with `gemini-3-flash-preview` and `gemini-3.5-flash-lite` failover, temperature 0) disagrees with the gold label: 72 rows (48 gold-UNSAFE judged SAFE, 24 gold-SAFE judged UNSAFE; agreement 95.2%, kappa 0.903; the original pass reported 110 disagreements, agreement 92.7%, kappa 0.851).

Review set: **361 rows** (sample 289, sample+screen 11, screen 61). It differs from the original 393 by the rows the re-run screen moves in or out (Gemini is not bit-reproducible and the model list changed); the 300-row sample is identical.

## Judges

| judge | model | prompt style | rows judged |
|---|---|---|---|
| A | Groq `qwen/qwen3.8-27b` | strict binary classifier, JSON only | 361 |
| B | Gemini (`gemini-2.5-flash` 90, `gemini-3-flash-preview` 270) | rubric with definitions, brief reasoning then verdict | 360 |
| C (tie-break) | Groq `openai/gpt-oss-120b` | red-team-analyst yes/no questions | 32 |

Judge B started on `gemini-3-flash-preview`; its daily quota ran out part-way, so the remaining rows used `gemini-2.5-flash` with the same prompt (the cache records the model per row).
The task asked for a Groq Llama as judge A; no Llama is served on this key (the Groq model list holds only gpt-oss, qwen3.8-27b, allam-2-7b, whisper/orpheus and prompt-guard models), so judge A is qwen3.8-27b. Independence caveat: the Qwen family produced training mutations and is the guard's base family, gpt-oss produced data and runs the vault, and Gemini produced data and the original double-check. No judge is independent of the data generators, and none sees anything but the text (long rows are cut at 1,500 characters).

## Agreement

- A vs B: agreement 91.4% on 360 rows, Cohen kappa **0.828** (sample rows only: 0.926).
- A vs gold label: agreement 87.8%, kappa 0.756. B vs gold label: agreement 91.4%, kappa 0.828. These are deflated by the 72 screen-disagreement rows, which are selected for disagreement.
- On the random 300-row sample alone: A vs gold 96.3%, B vs gold 96.7%.

## Outcomes

| outcome | rows | meaning |
|---|---|---|
| consensus-confirmed | 320 | final label (A=B, or majority of A, B, C) equals the gold label |
| consensus-relabel | 40 | final label differs from the gold label |
| unresolved | 1 | no majority (a judge failed to answer): listed below |

Third judge needed on 32 rows (A and B disagreed or one was missing). Relabel directions: SAFE->UNSAFE 17, UNSAFE->SAFE 23. Relabels on which A, B and the screen all disagree with gold: 21 (the high-confidence subset). Confirmed UNSAFE rows whose consensus category differs from the gold category: 76 (informational; category noise is not counted as a relabel).

Relabels by slice (rows reviewed / relabelled):

- source: HuggingFaceH4/ultrachat_200k 0/62; JailbreakBench/JBB-Behaviors 1/3; Lakera/gandalf_ignore_instructions 0/13; databricks/databricks-dolly-15k 0/49; deepset/prompt-injections 5/8; hard_benign 14/50; jackhhao/jailbreak-classification 4/21; synthetic 16/155
- encoding: base64 2/13; hex 2/14; leet 2/14; none 33/289; rot13 0/15; split 1/16
- language: en 35/327; hi 2/16; hinglish 3/18

Most relabels sit in `hard_benign` (benign messages that mention rules, secrets or codes: judges often call them attacks) and `synthetic` mutations; ultrachat, dolly and Gandalf rows are never relabelled. Hard-benign relabels are the least certain ones: the gold label encodes the intended vault policy, which the judges only know from the written definition.

## Label-noise estimate for gold_test

- **Random-sample estimator** (the 300-row sample is a proportional random draw): 10 of 300 relabelled = 3.3% (Wilson 95% CI 1.8% to 6.0%); extrapolated to the 1,500 rows: about **50** rows (27 to 90).
- **Screen-stratified estimator** (all 72 screen disagreements are reviewed, 36 of them relabelled; of the 289 sample rows the screen agrees with, 4 are relabelled = 1.4%): total about **56** relabels in 1,500 (44 to 86), i.e. 3.7% (2.9% to 5.7%).
- These are rates of rows on which two judges (or a majority of three) contradict the label. They are an upper bound on label noise if the judges err, and a lower bound if the judges share the generators' blind spots; the judges cannot decode every encoded row. A human pass is still the reference.

## Effect on ASR / FPR if the relabels were applied

Relabels: 23 gold-UNSAFE to SAFE and 17 gold-SAFE to UNSAFE. Attacks 682 to 676, benign 818 to 824.

| predictor | ASR before | ASR after | FPR before | FPR after |
|---|---|---|---|---|
| regex | 59.4% | 58.9% | 10.3% | 10.1% |
| llama-prompt-guard-2-86m | 51.6% | 49.9% | 10.3% | 9.1% |

Champion guard v2 (gold ASR 3/682 = 0.4%, FPR 15/818 = 1.8%, AUROC 0.9998; per-row gold logits are on Kaggle, not here): if the guard already agreed with the corrected label on every relabelled row the numbers become ASR 0.4%, FPR 1.8%; if it agreed with the old gold label on every relabelled row (the likely case, since it was trained on labels from the same generators) they become ASR 3.0%, FPR 4.6%. The exact figure needs the gold logits: `uv run python scripts/gold_review.py effect NAME=logits.npy,T=..,thr=..` (rows in gold_test.jsonl order).

## Unresolved rows

No majority could be formed for these ids (a judge returned nothing); they need a human decision:

`671c3b70362efaf9`

## Reproduce

`uv run python scripts/gold_review.py screen && uv run python scripts/gold_review.py judge && uv run python scripts/gold_review.py report` (needs GROQ_API_KEY and GEMINI_API_KEY; the caches in `data/review/gold_review/` make `report` offline).
