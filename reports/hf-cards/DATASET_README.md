---
license: cc-by-sa-3.0
pretty_name: BreachLab guard dataset v1
language: [en, hi]
task_categories: [text-classification]
tags: [prompt-injection, jailbreak-detection, research-demo]
---
# BreachLab dataset v1

## License

**CC-BY-SA-3.0.** The data includes share-alike rows from `databricks/databricks-dolly-15k` (CC-BY-SA-3.0), so the dataset is released under CC-BY-SA-3.0; per-source licences are in the table below and `DATA_LICENSE.md`. Code and model weights are Apache-2.0.

Total rows: 27977

## By label

| value | count |
|---|---|
| SAFE | 14136 |
| UNSAFE | 13841 |

## By split

| value | count |
|---|---|
| gold_test | 1500 |
| mining_pool | 5000 |
| train | 19330 |
| val | 2147 |

## By source

| value | count |
|---|---|
| HuggingFaceH4/ultrachat_200k | 4959 |
| JailbreakBench/JBB-Behaviors | 100 |
| Lakera/gandalf_ignore_instructions | 771 |
| databricks/databricks-dolly-15k | 4860 |
| deepset/prompt-injections | 573 |
| hard_benign | 2901 |
| jackhhao/jailbreak-classification | 1607 |
| synthetic | 12206 |

## By attack_family

| value | count |
|---|---|
| base64 | 1070 |
| benign | 11235 |
| deepset | 245 |
| fake_system | 778 |
| gandalf | 771 |
| hard_benign | 2901 |
| hex | 847 |
| hindi | 1346 |
| hinglish | 1285 |
| jackhhao | 519 |
| jbb | 100 |
| leet | 1035 |
| multi_turn | 1376 |
| paraphrase | 1349 |
| payload_split | 749 |
| roleplay | 1371 |
| rot13 | 991 |
| secret_extraction | 9 |

## By language

| value | count |
|---|---|
| en | 25293 |
| hi | 1400 |
| hinglish | 1284 |

## By encoding

| value | count |
|---|---|
| base64 | 1070 |
| hex | 847 |
| leet | 1035 |
| none | 23285 |
| rot13 | 991 |
| split | 749 |

## Source licenses

| source | license |
|---|---|
| HuggingFaceH4/ultrachat_200k | mit |
| JailbreakBench/JBB-Behaviors | mit |
| Lakera/gandalf_ignore_instructions | mit |
| allenai/wildjailbreak | odc-by (EXCLUDED: no row comes from it) |
| databricks/databricks-dolly-15k | cc-by-sa-3.0 |
| deepset/prompt-injections | apache-2.0 |
| jackhhao/jailbreak-classification | apache-2.0 |

## Provenance and licenses

Every row comes from one of: public datasets (table above), rule-based mutations of those seeds (encodings, payload
splitting), LLM-written mutations and benign prompts (Gemini and Groq free-tier models, batched, with the parent id
kept), or template-built hard-benign prompts (benign text that quotes or analyses an attack). `allenai/wildjailbreak` is
listed in the loader table but was never accessible (gated, ADR-0019): **no row comes from it**. Non-commercial
sources (toxic-chat, Alpaca) were dropped (ADR-0013).

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

**License of this dataset: CC-BY-SA-3.0.** The dolly rows are CC-BY-SA-3.0 and carry share-alike, so the combined dataset is
released under CC-BY-SA-3.0; keep attribution for the MIT and Apache sources. LLM-generated rows are subject to the
generating providers' terms. Code and model weights are Apache-2.0 (`DATA_LICENSE.md` in the code repository).

## Splits and labels

train 19330, val 2147, mining_pool 5000, gold_test
1500. Mutations inherit `parent_id` and all descendants of a parent share a split; a
MinHash near-duplicate check across splits passes. Labels are **generation-time labels** (source label or the
mutation parent's label). A Gemini double-check of gold_test agrees on 92.7% of rows (kappa 0.851.); the
110 disagreements are queued for human review (pending). `gold_test` is frozen and must not be used for training.

## Limitations

English-heavy (about 90%), Hindi and Hinglish about 5% each. Fewer LLM-written paraphrases than planned (free-tier
quota, ADR-0029). Attack text is quoted as data: some rows are offensive. Hard-benign prompts are built from templates and
quoted seeds, not from real users. The data is in-distribution with the guard's evaluation; it is not a benchmark of
real-world attack diversity.

Research demo. Live game data is never included in this dataset.
