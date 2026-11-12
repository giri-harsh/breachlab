# BreachLab dataset v1

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
| allenai/wildjailbreak | odc-by |
| databricks/databricks-dolly-15k | cc-by-sa-3.0 |
| deepset/prompt-injections | apache-2.0 |
| jackhhao/jailbreak-classification | apache-2.0 |

Research demo. Live game data is never included in the public dataset.

## v2 addendum (2026-11-12)

v2 = v1 plus 2,172 new benign rows aimed at the guard's measured weakness (short benign openers and Hindi/Hinglish text were blocked far too often):
- Hard-benign short rows per language (en, hi, hinglish): greetings, small talk, meta questions, simple requests, short questions, "rules of things" (e.g. game or grammar rules).
- Longer benign rows in hi and hinglish (250 each).
- Generated with free-tier Groq models; dedup (MinHash) against train, val, gold_test, mining pool, the held-out opener slice and canary probes; PII filter; banned-term filter so no row contains attack vocabulary.
- New rows split 88/12 into train/val. `gold_test` (1,500) and `mining_pool` (5,000) are byte-identical to v1.
- Counts: train 21,242, val 2,407. Per-cell counts in `v2_stats.json`.
- Scale: built at half the planned volume (about 2.2k of about 4k rows) to stay inside free-tier quotas.

**License:** CC-BY-SA-3.0, as v1 (`DATA_LICENSE.md`). The new v2 rows are LLM-generated benign text written for this project.
