---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
library_name: peft
pipeline_tag: text-classification
language: [en, hi]
tags: [prompt-injection, jailbreak-detection, guard, lora, qwen3, onnx, research-demo]
---
# breachlab-guard-v1

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
- **Calibration (fp32):** temperature T = 1.41, decision threshold on `p_unsafe` = 0.063, both
  fitted on the validation split so that the false-positive rate stays at or below 2%. Expected calibration error on
  val fell from 0.023 to 0.009. The low threshold is what fitting for FPR <= 2% produced.
- **Calibration (int4, served):** at the fp32 threshold the int4 graph fails the quality-parity rule on FPR (7.7% vs
  1.6%) although its ranking is intact (AUROC drop 0.0003). It therefore has its own threshold
  0.232 (`champion/calibration-q4b32.json`), fitted on val with the same rule (FPR <= 2%), never on
  gold_test. On gold_test it measures ASR 0.59%, FPR 2.08%,
  F1 0.985, AUROC 0.9990: FPR is 0.5 pp above fp32 and 0.08 pp above the 2.0% bar,
  inside the fp32 confidence interval (ADR-0082, a recorded deviation from ADR-0057).
- **Service contract:** `POST /score {"text", "kind": "input"|"output", "alias"}` returns
  `label`, `category`, `p_unsafe`, `confidence`, `version`, `latency_ms`; `GET /health`. The serving image
  (`vercel-guard/` in the project repository) pulls these files with a token and is host-agnostic.

## Evaluation (gold_test, 1500 rows, in-distribution)

| model | ASR (lower is better) | FPR | F1 | AUROC |
|---|---|---|---|---|
| regex filter | 59.4% | 10.3% | 0.531 | 0.652 |
| Llama Prompt-Guard-2-86M (via Groq) | 51.6% | 10.3% | 0.602 | 0.868 |
| Gemini as zero-shot judge | 12.8% | 2.8% | 0.915 | 0.969 |
| DeBERTa prompt-injection classifier | 10.4% | 14.2% | 0.867 | 0.904 |
| Qwen3-0.6B zero-shot | 100.0% | 0.4% | 0.000 | 0.591 |
| Qwen3-1.7B zero-shot | 61.9% | 13.2% | 0.495 | 0.777 |
| **Qwen3-0.6B + LoRA, SFT-only (this model)** | 0.6% | 1.6% | 0.988 | 0.999 |
| Same adapter, int4 weight-only graph as served (own threshold) | 0.6% | 2.1% | 0.985 | 0.999 |
| Qwen3-0.6B + LoRA, SFT then DPO (challenger) | 0.4% | 1.7% | 0.988 | 0.999 |
| Qwen3-1.7B + LoRA, SFT-only (comparison) | 0.0% | 1.3% | 0.992 | 1.000 |
| Qwen3-1.7B + LoRA, SFT then DPO (comparison) | 0.0% | 1.3% | 0.992 | 1.000 |

ASR = share of attacks the guard lets through; FPR = share of benign messages it blocks. Confidence intervals,
McNemar tests, slices and the worst misses are in `reports/eval-v1.md` in the project repository.

**Seeds.** Seeds 0, 1, 3 were trained and scored with the same configuration (SFT, mean +- sd over n = 3):
ASR 0.49 +- 0.17%, FPR 1.83 +- 0.24%, F1 0.9867 +- 0.0017, AUROC 0.9995 +- 0.0002. A fourth run (seed 2) diverged in
fp16 (NaN gradients, NaN logits on every gold row); it is excluded from the mean, kept as evidence, and the third
seed was re-run unchanged as seed 3. The sd over three seeds is itself noisy. The table above is seed 0.

## Latency (target not met)

The project target was p95 <= 300 ms on a CPU host. **It is not met.**
- Served variant (int4 weight-only with prefix-KV reuse) on the real host, a Vercel Python function (1 vCPU, 2 GB),
  measured from a client over the network, 500 warm requests at concurrency 1: model time p50 / p95 =
  720 / 3468 ms (round trip p50 842, p95 3556 ms). At concurrency 4 the single vCPU serializes the
  requests (round trip p50 3664 ms).
- Cold start: an instance is gone after about 10 minutes idle; the first request after that takes about 13 s, longer
  than the gateway's 8 s guard timeout, so that player gets a fail-closed 503 "guard waking up" and the next request
  works.
- Earlier, before this host existed, the fp32 prefix-KV variant measured p95 2326 ms on a Kaggle kernel pinned to
  2 vCPU, and plain fp32 ONNX, 512 tokens, 4-vCPU Kaggle CPU: p95 3112 ms. Those numbers are superseded
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

`Pseudo-Harsh/breachlab-data` (private for now), 27,977 rows in total: train 19,330, val
2,147, mining pool 5,000 (only for hard-negative mining), gold_test
1,500 (never trained on). Public-source seeds plus LLM- and rule-generated mutations
(paraphrase, roleplay, multi-turn, base64, hex, rot13, leetspeak, payload splitting, Hindi, Hinglish) and generated
hard-benign prompts. Splits are grouped by parent so mutations of one seed never straddle train and test; near-duplicate
leakage is checked. No live player data. Source licenses:

| asset | license | how it is used |
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
| meta-llama/Llama-Prompt-Guard-2-86M | Llama license | baseline through the Groq API only; weights not redistributed |

The dolly subset is CC-BY-SA-3.0 (share-alike). Whether that obligation extends to the fine-tuned weights has not
been settled by a lawyer; check it before you reuse the weights commercially.

## Intended use and limitations

- **Intended:** a guard for the BreachLab game and a reference for small-model prompt-injection classification, in
  English, Hindi and Hinglish.
- **In-distribution.** `gold_test` shares source datasets and synthetic generators with training. The numbers above say
  little about attacks from other generators or from real players; a fresh test set from live play is planned.
- **Gold labels are generation-time labels** (the label of the source row or of the mutation's parent), not human
  labels. A Gemini double-check agrees on 92.7% of rows (Cohen's kappa 0.851.); the disagreements are
  queued for human review, which is pending. The judge reads raw text and cannot decode encoded attacks, so a
  disagreement is not proof of a wrong label.
- Benign messages that discuss or quote attacks are the known confusion area (the hard-benign slice of gold_test is
  built from templates and quoted seeds, so it is easier than real users). Messages longer than 500 characters,
  multi-turn state and languages other than English, Hindi and Hinglish are not covered.
- The served int4 guard blocks about 2% of benign messages on gold_test (2.08%) and a little more of short questions
  about the assistant itself (informal check: 2 of 12 short benign openers); it also serializes requests on one vCPU.
- A single guard is one layer; the game also checks the agent reply for the secret. Do not rely on this model as the only
  defense of a real system.
