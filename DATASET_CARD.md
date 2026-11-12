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
| allenai/wildjailbreak | odc-by |
| databricks/databricks-dolly-15k | cc-by-sa-3.0 |
| deepset/prompt-injections | apache-2.0 |
| jackhhao/jailbreak-classification | apache-2.0 |

Research demo. Live game data is never included in the public dataset.
